#include "lessons.hpp"
#include "kalman_cv.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace training
{
  // 第四讲：位姿、坐标与基础跟踪
  //
  // 输入格式：
  //   detections : const std::vector<Detection>&，第三讲输出的本帧观测
  //                mode=kArmor 时是装甲板，mode=kRune 时是符文叶片
  //                符文叶片的 Detection::center 里带着 R 标圆心（全帧共享）
  //   mode       : DetMode，决定解算分支：
  //                · kArmor：用装甲板四角 + 大小板物理尺寸做 solvePnP
  //                · kRune ：叶片四角 + 圆心 -> 极坐标 (r, θ)，再做角速度估计
  //   timestamp  : TimePoint，本帧图像时刻只能与其他 TimePoint 求时间差
  //   imu_q      : const Eigen::Quaterniond&，图像时刻的归一化姿态四元数
  //   calibration: const CameraCalibration&：
  //                matrix=3x3 CV_64F 相机内参distortion=1x5 CV_64F 畸变
  //                camera_to_gimbal=相机到云台的 3x3 旋转矩阵
  //                camera_to_gimbal_translation=相机到云台平移，单位 m
  //   state      : TrackingState&，仅本讲使用的跨帧状态函数需要原地更新
  //
  // 输出格式：
  //   std::optional<Target>有目标时：
  //   - position : IMU 稳定参考系中、相对云台原点的目标向量，单位 m
  //   - velocity : 同一参考坐标系中的目标速度，单位 m/s
  //   本课程统一使用 x 向前、y 向左、z 向上的右手坐标系
  //   无有效观测、PnP 失败或跟踪未建立时返回 std::nullopt
  //
  // 打符思路：
  //   1) 叶片中心 b = 四角均值R 标圆心 C 来自 Detection::center
  //   2) r = ‖b − C‖，θ = atan2(b.y − C.y, b.x − C.x)
  //   3) ω = Δθ/Δt（注意 ±π 展开，用 std::remainder）
  //   4) 叶片三维位置用四角 solvePnP击打点是"叶片绕 C 旋转 ω·t_flight 后"的位置
  //   5) 只用 1 片叶子也能解：r 与 θ 都来自这一片 + 圆心

  namespace
  {
    // 装甲板物理尺寸（m）
    constexpr double kLargeW{0.225}; // 大板宽
    constexpr double kSmallW{0.135}; // 小板宽
    constexpr double kArmorH{0.125}; // 板高
    constexpr int kDetectFrames{5};  // 连续命中5帧后进入 TRACKING
    constexpr int kTempLostMax{10};  // 允许连续丢失10帧
    constexpr double kMaxReprojPx{3.0};
    constexpr double kPredLeadS{0.250}; // 预测250ms后的位置
    // 用二维框宽估测距离（针孔模型，不做PnP）：
    // 像素宽 w = fx * W_real / Z  ->  Z ~ fx * W_real / w
    // 按近似距离排序
    double estimate_distance(const Detection &d, double fx)
    {
      const double W = d.large_armor ? kLargeW : kSmallW;
      const double w = std::max(static_cast<double>(d.box.width), 1.0);
      return fx * W / w;
    }
    // 按距离排序
    cv::Point2f center_of(const cv::Point2f c[4])
    {
      return (c[0] + c[1] + c[2] + c[3]) * 0.25F;
    }
    // 装甲板 4 角（板中心为原点，z=0），顺序与 Detection::corners 一致：
    // 左上、右上、右下、左下
    std::vector<cv::Point3f> object_points(bool large)
    {
      const float W = static_cast<float>(large ? kLargeW : kSmallW) * 0.5F;
      const float H = static_cast<float>(kArmorH) * 0.5F;
      return {{-W, -H, 0.F}, {W, -H, 0.F}, {W, H, 0.F}, {-W, H, 0.F}};
    }
    double reproj_error(const std::vector<cv::Point3f> &obj,
                        const std::vector<cv::Point2f> &img,
                        const cv::Vec3d &rvec, const cv::Vec3d &tvec,
                        const cv::Mat &K, const cv::Mat &D)
    {
      std::vector<cv::Point2f> proj;
      cv::projectPoints(obj, rvec, tvec, K, D, proj);
      double s = 0.0;
      for (size_t i = 0; i < proj.size(); ++i)
        s += cv::norm(proj[i] - img[i]);
      return s / static_cast<double>(proj.size());
    }
    // 对单块板做PnP：返回是否成功
    bool solve_armor(const Detection &d, const CameraCalibration &cal,
                     cv::Vec3d &rvec, cv::Vec3d &tvec, double &err_px)
    {
      const auto obj = object_points(d.large_armor);
      std::vector<cv::Point2f> img(d.corners.begin(), d.corners.end());
      const cv::Mat K = cal.matrix;
      const cv::Mat D = cal.distortion.empty() ? cv::Mat() : cal.distortion;
      std::vector<cv::Vec3d> rvecs, tvecs;
      try
      {
        cv::solvePnPGeneric(obj, img, K, D, rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
      }
      catch (const cv::Exception &)
      {
        return false;
      }
      if (rvecs.empty())
        return false;
      double best = 1e9;
      for (size_t i = 0; i < rvecs.size(); ++i)
      {
        if (tvecs[i][2] <= 0.0) // 必须在相机前方（排除镜像解）
          continue;
        const double e = reproj_error(obj, img, rvecs[i], tvecs[i], K, D);
        if (e < best)
        {
          best = e;
          rvec = rvecs[i];
          tvec = tvecs[i];
        }
      }
      if (best > kMaxReprojPx)
        return false;
      err_px = best;
      return true;
    }
    // 相机系 -> 云台系 -> 世界系
    Eigen::Vector3d to_world(const Eigen::Vector3d &p_cam,
                             const CameraCalibration &cal,
                             const Eigen::Quaterniond &imu_q)
    {
      Eigen::Matrix3d R_c2g = Eigen::Matrix3d::Identity();
      Eigen::Vector3d t_c2g = Eigen::Vector3d::Zero();
      if (cal.camera_to_gimbal.rows() == 3 && cal.camera_to_gimbal.cols() == 3)
        R_c2g = cal.camera_to_gimbal;
      t_c2g = cal.camera_to_gimbal_translation;
      const Eigen::Vector3d p_gimbal = R_c2g * p_cam + t_c2g;
      const Eigen::Matrix3d R_g2w = imu_q.toRotationMatrix();
      return R_g2w * p_gimbal;
    }
    // 世界系 -> 相机系（用于标注预测框）
    Eigen::Vector3d to_camera(const Eigen::Vector3d &p_world,
                              const CameraCalibration &cal,
                              const Eigen::Quaterniond &imu_q)
    {
      const Eigen::Matrix3d R_c2g = cal.camera_to_gimbal;
      const Eigen::Vector3d t_c2g = cal.camera_to_gimbal_translation;
      const Eigen::Matrix3d R_g2w = imu_q.toRotationMatrix();
      const Eigen::Vector3d p_gimbal = R_g2w.transpose() * p_world;
      return R_c2g.transpose() * (p_gimbal - t_c2g);
    }
  } // namespace

  std::optional<Target> lesson04_tracking(const std::vector<Detection> &detections,
                                          DetMode mode,
                                          TimePoint timestamp,
                                          const Eigen::Quaterniond &imu_q,
                                          const CameraCalibration &calibration,
                                          TrackingState &state)
  {
    // ===== 实现区 BEGIN =====

    // 暂不实现打符
    if (mode != DetMode::kArmor)
      return std::nullopt;

    if (calibration.matrix.empty() || detections.empty())
    {
      // 无观测 -> 状态机降级
      if (state.phase == TrackingState::Phase::kTracking)
      {
        if (++state.lost_frames > kTempLostMax)
          state.phase = TrackingState::Phase::kLost;
        else
          state.phase = TrackingState::Phase::kTempLost;
      }
      else if (state.phase != TrackingState::Phase::kLost)
      {
        state.phase = TrackingState::Phase::kLost;
        state.same_target_frames = 0;
      }
      return std::nullopt;
    }

    // 1. 筛选：通过二维框宽估测的距离筛选
    const double fx{calibration.matrix.at<double>(0, 0)};

    int best_idx{-1};
    double best_dist{1e9};
    for (size_t i{0}; i < detections.size(); ++i)
    {
      const double dist{estimate_distance(detections[i], fx)};
      if (dist < best_dist)
      {
        best_dist = dist;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0)
    {
      return std::nullopt;
    }

    const Detection &d{detections[static_cast<size_t>(best_idx)]};
    const cv::Point2f px{center_of(d.corners.data())};

    // 2. PnP解算
    cv::Vec3d rvec, tvec;
    double err_px{0.0};
    if (!solve_armor(d, calibration, rvec, tvec, err_px))
    {
      return std::nullopt;
    }

    // 3. 坐标变换
    const Eigen::Vector3d p_cam(tvec[0], tvec[1], tvec[2]);
    const Eigen::Vector3d p_world(to_world(p_cam, calibration, imu_q));

    // 4. 选板
    const bool same{(state.target_class_id == d.class_id) &&
                    (state.target_px.x >= 0.0F) &&
                    (cv::norm(px - state.target_px) < (0.6F * std::max(d.box.width, 8.0F)))};

    if (same)
      ++state.same_target_frames;
    else
      state.same_target_frames = 1;

    switch (state.phase)
    {
    case TrackingState::Phase::kLost:
    case TrackingState::Phase::kDetecting:
      state.phase = (state.same_target_frames >= kDetectFrames)
                        ? TrackingState::Phase::kTracking
                        : TrackingState::Phase::kDetecting;
      break;

    case TrackingState::Phase::kTempLost:
    case TrackingState::Phase::kTracking:
      state.phase = TrackingState::Phase::kTracking;
      break;
    }

    state.lost_frames = 0;
    state.target_class_id = d.class_id;
    state.target_px = px;
    state.target_box_px = d.box.width;
    state.distance_m = p_cam.norm();
    state.reproj_err_px = err_px;

    // 5. 匀速卡尔曼滤波
    double dt{0.0};
    if (state.last_stamp.time_since_epoch().count() != 0)
      dt = std::chrono::duration<double>(timestamp - state.last_stamp).count();
    state.last_stamp = timestamp;

    if (!state.kf.initialized())
    {
      state.kf.init(p_world);
    }
    else
    {
      state.kf.predict(dt);
      state.kf.update(p_world);
    }

    state.position = state.kf.position();
    state.position = state.kf.velocity();

    // 非 tracking 模式下不发送目标数据
    if (state.phase != TrackingState::Phase::kTracking)
      return std::nullopt;

    // 6. 返回预测框 & 250ms历史数据
    const Eigen::Vector3d p_pred {state.kf.predict_future(kPredLeadS)};
    const Eigen::Vector3d p_pred_cam {to_camera(p_pred, calibration, imu_q)};
    // 此处沿用当前姿态绘制预测框
    const auto obj {object_points(d.large_armor)};
    std::vector<cv::Point3f> obj_pred;
    for (const auto &o : obj) {
      obj_pred.emplace_back(o.x + static_cast<float> (p_pred_cam.x()),
                            o.y + static_cast<float> (p_pred_cam.y()),
                            o.z + static_cast<float> (p_pred_cam.z()));
    }
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj_pred, cv::Vec3d(0,0,0),
                      cv::Vec3d(p_pred_cam.x(),p_pred_cam.y(),p_pred_cam.z()),
                      calibration.matrix,calibration.distortion,proj);
    
    if (proj.size() == 4) {
      for (int i{0}; i < 4; ++i) {
        state.pred_corners[i] = proj[static_cast<size_t>(i)];
      }
      state.has_pred = true;
    }

    state.pred_hist.push_back({timestamp, p_pred});
    while (state.pred_hist.size() > 64)
    {
      state.pred_hist.pop_front();
    }
    Target t;
    t.position = state.position;
    t.position = state.velocity;
    return t;
    
    // ===== 实现区 END =====
  }

  // 由 lesson3 的观测更新跨帧状态
  // main 不直接写入 TrackingState 字段
  void lesson04_observe(TrackingState &state, const std::vector<Detection> &detections)
  {
    if (detections.empty())
    {
      state.lost_frames += 1;
      return;
    }
    state.lost_frames = 0;
    const Detection &d = detections.front();
    state.last_center = (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
    state.last_box_px = std::max(d.box.width, d.box.height);
  }

  RoiPlan lesson04_roi_plan(const TrackingState &state,
                            const CameraCalibration &calibration,
                            int img_width, int img_height,
                            float target_width_m)
  {
    RoiPlan plan;
    if (img_width <= 0 || img_height <= 0)
      return plan;
    const int short_side = std::min(img_width, img_height);
    // ① 预期像素尺寸：优先用 PnP 距离，退化为上一帧框宽
    float pred_px = 0.F;
    if (state.last_distance_m > 0.1 && target_width_m > 1e-4F && calibration.valid())
    {
      const double fx = calibration.matrix.at<double>(0, 0);
      pred_px = static_cast<float>(fx * target_width_m / state.last_distance_m);
    }
    else if (state.last_box_px > 4.F)
    {
      pred_px = state.last_box_px;
    }
    else
    {
      return plan; // 没有任何先验，本帧走全图
    }
    // ② 目标太大（离得太近）就不裁，直接全图
    //    经验阈值：目标超过短边的 45%，裁剪很容易切掉一部分
    if (pred_px > 0.45F * static_cast<float>(short_side))
      return plan;
    // ③ 裁剪边长 = 2.5 倍目标尺寸，且不小于 128，不大于短边
    float side = std::clamp(2.5F * pred_px, 128.F, static_cast<float>(short_side));
    if (side >= 0.95F * static_cast<float>(short_side))
      return plan;
    // ④ 中心：优先用上一帧目标中心，否则画面中心
    cv::Point2f c = state.last_center;
    if (c.x < 0.F || c.y < 0.F)
      c = cv::Point2f(img_width * 0.5F, img_height * 0.5F);
    // ⑤ 裁剪框贴边时把中心拉回画面内
    const float half = side * 0.5F;
    c.x = std::clamp(c.x, half, static_cast<float>(img_width) - half);
    c.y = std::clamp(c.y, half, static_cast<float>(img_height) - half);
    plan.use = true;
    plan.center = c;
    plan.side = side;
    return plan;
  }

} // namespace training
