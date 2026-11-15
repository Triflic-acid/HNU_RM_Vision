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

    // ---------- 卡尔曼噪声参数 ----------
    // 关键：configure() 之前从未被调用，滤波器一直用默认 q_accel=8 m/s²，
    // 意味着每帧速度允许抖动 √(q·dt) ≈ 0.68 m/s，
    // 预测 250 ms 时被放大成 ±170 mm —— 这就是"静止靶也抖"的根因。
    constexpr double kQAccel{0.40};   // 加速度标准差 m/s²（机器人实际 <1）
    constexpr double kRPosInit{0.08}; // 初始量测标准差 m（之后被 set_R 覆盖）

    // 预测位移上限：速度估计有噪声时，0.25 s 外推会把误差放大成米级跳变
    constexpr double kPredMaxShift{0.80};

    // ---------- 观测有效性门限 ----------
    constexpr float  kMinBoxPx{40.0F};      // 框宽下限：小于它解算不可信
    constexpr double kDistRatioMax{1.8};    // PnP 距离 / 单目距离 上限
    constexpr double kDistRatioMin{0.55};   // 下限

    // 噪声量测模型（距离自适应）
    // 由针孔几何推出：
    //   深度 Z = fx·W/w   ->  ∂Z/∂w = −Z²/(fx·W)
    //   横向 X = Z·u/fx   ->  ∂X/∂u = Z/fx
    // 因此角点像素噪声 σ_px 传播成
    //   横向 σ_lat = Z·σ_px/fx        （∝ Z）
    //   深度 σ_dep = Z²·σ_px/(fx·W)   （∝ Z²，且与板宽 W 成反比）
    // 2 m、小板(135 mm)、σ_px=3 px：σ_lat≈4.6 mm，σ_dep≈82 mm（实测 80 mm）
    constexpr double kCornerNoisePx{3.0};
    constexpr double kSigLatMin{0.002};
    constexpr double kSigLatMax{0.05};
    constexpr double kSigDepMin{0.01};
    constexpr double kSigDepMax{0.60};

    // 用二维框宽估测距离（针孔模型，不算PnP）：
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
    bool solve_armor(const Detection &d,
                     const CameraCalibration &cal,
                     const cv::Vec3d &rvec_prev,
                     bool has_prev,
                     cv::Vec3d &rvec,
                     cv::Vec3d &tvec,
                     double &err_px)
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
      // 暂时关闭双解消歧：只按重投影误差选最小（保留 IPPE 的两个解，但不用法向判据）
      // 参数 rvec_prev / has_prev 保留是为了不改调用点，当前不参与判定
      (void)rvec_prev;
      (void)has_prev;

      double best{1e9};
      bool found{false};

      for (size_t i = 0; i < rvecs.size(); ++i)
      {
        if (tvecs[i][2] <= 0.0)
          continue;

        const double e = reproj_error(obj, img, rvecs[i], tvecs[i], K, D);
        if (e > kMaxReprojPx)
          continue;

        if (e < best)
        {
          best = e;
          rvec = rvecs[i];
          tvec = tvecs[i];
          found = true;
        }
      }

      if (!found)
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

    // 本帧没有有效观测时的统一降级（检测为空/PnP 失败）
    const auto degrade = [&state]() -> std::optional<Target>
    {
      if (state.phase == TrackingState::Phase::kTracking ||
          state.phase == TrackingState::Phase::kTempLost)
      {
        if (++state.lost_frames > kTempLostMax)
          state.phase = TrackingState::Phase::kLost;
        else
          state.phase = TrackingState::Phase::kTempLost;
      }
      else
      {
        state.phase = TrackingState::Phase::kLost;
        state.same_target_frames = 0;
      }
      state.has_pred = false;
      return std::nullopt;
    };

    // 空结果降级
    if (calibration.matrix.empty() || detections.empty())
      return degrade();

    // 1. 筛选：通过二维框宽估测的距离筛选
    const double fx{calibration.matrix.at<double>(0, 0)};

    int best_idx{-1};
    double best_dist{1e9};
    for (size_t i{0}; i < detections.size(); ++i)
    {
      // 门 1：框太小时单目距离和 PnP 深度都不可信。
      // 实测跳变帧的框宽中位 30 px、正常帧 81 px，40 px 正好把两类分开。
      if (detections[i].box.width < kMinBoxPx)
        continue;

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

    state.selected_idx = best_idx;
    const Detection &d{detections[static_cast<size_t>(best_idx)]};
    const cv::Point2f px{center_of(d.corners.data())};

    // 2. PnP解算
    cv::Vec3d rvec, tvec;
    double err_px{0.0};
    if (!solve_armor(d, calibration, state.last_rvec, state.has_rvec,
                     rvec, tvec, err_px))
      return degrade(); // PnP 失败返回本帧无有效观测

    state.last_rvec = rvec;
    state.has_rvec = true;

    // 3. 坐标变换
    const Eigen::Vector3d p_cam(tvec[0], tvec[1], tvec[2]);

    // 门 2：PnP 距离必须与单目估距一致。
    // 远处小目标（框 30 px）会被解到 16 m，重投影反而更小（0.31 px），
    // 所以重投影门限拦不住，只能靠两路距离互相校验。
    {
      const double d_pnp = p_cam.norm();
      const double d_mono = estimate_distance(d, fx);
      if (d_mono > 0.1 && (d_pnp > kDistRatioMax * d_mono || d_pnp < kDistRatioMin * d_mono))
        return degrade();
    }
    const Eigen::Vector3d p_world(to_world(p_cam, calibration, imu_q));

    //   相机系：x/y 横向、z 深度 -> 对角阵
    //   再旋转到世界系：R_world = M·R_cam·Mᵀ，M = R_g2w·R_c2g
    {
      const double fx = calibration.matrix.at<double>(0, 0);
      const double W = d.large_armor ? kLargeW : kSmallW;
      const double Z = std::max(p_cam.z(), 0.05); // 用深度，不用模长

      double sig_lat = Z * kCornerNoisePx / fx;
      double sig_dep = Z * Z * kCornerNoisePx / (fx * W);
      sig_lat = std::clamp(sig_lat, kSigLatMin, kSigLatMax);
      sig_dep = std::clamp(sig_dep, kSigDepMin, kSigDepMax);

      Eigen::Matrix3d R_cam = Eigen::Matrix3d::Zero();
      R_cam(0, 0) = sig_lat * sig_lat;
      R_cam(1, 1) = sig_lat * sig_lat;
      R_cam(2, 2) = sig_dep * sig_dep;

      const Eigen::Matrix3d R_c2g =
          (calibration.camera_to_gimbal.rows() == 3 &&
           calibration.camera_to_gimbal.cols() == 3)
              ? calibration.camera_to_gimbal
              : Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d M = imu_q.toRotationMatrix() * R_c2g;

      state.kf.set_R(M * R_cam * M.transpose());
    }

    // 4. 选板
    // 编号会抖（实测 310 次换号里 142 次位置几乎没动），所以不能只看 class_id：
    // 位置极近时视为同一块，即使编号变了。
    const float px_shift = (state.target_px.x >= 0.0F)
                               ? static_cast<float>(cv::norm(px - state.target_px))
                               : 1e9F;
    const float box_w = std::max(d.box.width, 8.0F);
    const bool same{(state.target_px.x >= 0.0F) &&
                    (px_shift < 1.0F * box_w) &&
                    (state.target_class_id == d.class_id || px_shift < 0.35F * box_w)};

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
      state.kf.configure(kQAccel, kRPosInit);
      state.kf.init(p_world);
    }
    else
    {
      state.kf.predict(dt);
      state.kf.update(p_world);
    }

    state.position = state.kf.position();
    state.velocity = state.kf.velocity();

    // 非 tracking 模式下不发送目标数据
    if (state.phase != TrackingState::Phase::kTracking)
    {
      state.has_pred = false;
      return std::nullopt;
    }

    // 6. 返回预测框 & 250ms历史数据
    Eigen::Vector3d p_pred{state.kf.predict_future(kPredLeadS)};
    {
      // 速度估计不准时，250 ms 外推会把位置放大成米级跳变；限幅后最坏只是"提前量不足"
      const Eigen::Vector3d shift{p_pred - state.kf.position()};
      const double n{shift.norm()};
      if (n > kPredMaxShift)
        p_pred = state.kf.position() + shift * (kPredMaxShift / n);
    }
    const Eigen::Vector3d p_pred_cam{to_camera(p_pred, calibration, imu_q)};
    // 此处沿用当前姿态绘制预测框
    const auto obj{object_points(d.large_armor)};
    std::vector<cv::Point3f> obj_pred;
    for (const auto &o : obj)
    {
      obj_pred.emplace_back(o.x + static_cast<float>(p_pred_cam.x()),
                            o.y + static_cast<float>(p_pred_cam.y()),
                            o.z + static_cast<float>(p_pred_cam.z()));
    }
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj_pred, cv::Vec3d(0, 0, 0),
                      cv::Vec3d(p_pred_cam.x(), p_pred_cam.y(), p_pred_cam.z()),
                      calibration.matrix, calibration.distortion, proj);

    if (proj.size() == 4)
    {
      for (int i{0}; i < 4; ++i)
      {
        state.pred_corners[i] = proj[static_cast<size_t>(i)];
      }
      state.has_pred = true;
    }

    state.pred_hist.push_back({timestamp, p_pred,
                               d.class_id, static_cast<float>(d.confidence), d.box.width,
                               static_cast<float>(state.distance_m),
                               static_cast<float>(err_px),
                               state.phase,
                               state.same_target_frames,
                               state.lost_frames});
    while (state.pred_hist.size() > 64)
    {
      state.pred_hist.pop_front();
    }
    Target t;
    t.position = state.position;
    t.velocity = state.velocity;
    return t;

    // ===== 实现区 END =====
  }

  // 由 lesson3 的观测更新跨帧状态
  // main 不直接写入 TrackingState 字段
  RoiPlan lesson04_roi_plan(const TrackingState &state,
                            const CameraCalibration &calibration,
                            int img_width, int img_height,
                            float target_width_m)
  {
    RoiPlan plan;
    if (img_width <= 0 || img_height <= 0)
      return plan;
    const int short_side = std::min(img_width, img_height);
    // 1. 预期像素尺寸：优先用 PnP 距离，退化为上一帧框宽
    float pred_px{0.0F};
    if (state.distance_m > 0.1 && target_width_m > 1e-4F && calibration.valid())
    {
      const double fx = calibration.matrix.at<double>(0, 0);
      pred_px = static_cast<float>(fx * target_width_m / state.distance_m);
    }
    else if (state.target_box_px > 4.0F)
    {
      pred_px = state.target_box_px;
    }
    else
    {
      return plan; // 没有先验时，不作ROI
    }

    // 2. 预测本帧的目标位置：ROI 在检测之前算，只能用上一帧状态外推
    //    延迟 = 一帧间隔（约 28 ms）+ 传输/推理，取 40 ms
    constexpr float kLatencyS{0.040F}; // 单位 s
    const cv::Point2f pred_c{state.target_px.x + state.px_velocity.x * kLatencyS,
                             state.target_px.y + state.px_velocity.y * kLatencyS};
    const float drift = cv::norm(state.px_velocity) * kLatencyS;

    // 3. 目标太大（太近）/目标太快不裁
    //    经验阈值：目标超过短边的 45%，容易切掉一部分
    if (pred_px > 0.45F * static_cast<float>(short_side) || drift > 0.5F * pred_px)
      return plan;
    // 4. 裁剪边长 = 2.5 倍目标尺寸 + 车子速度造成的外扩
    const float side{std::clamp(2.5F * pred_px + 2.0F * drift, 128.0F, static_cast<float>(short_side))};
    if (side >= 0.95F * static_cast<float>(short_side))
      return plan;
    // 5. 中心：优先预测中心，否则画面中心
    cv::Point2f c{pred_c};
    if (c.x < 0.0F || c.y < 0.0F)
      c = cv::Point2f(img_width * 0.5F, img_height * 0.5F);
    // 6. 裁剪框贴边时把中心拉回画面内，确保裁减完整
    const float half = side * 0.5F;
    c.x = std::clamp(c.x, half, static_cast<float>(img_width) - half);
    c.y = std::clamp(c.y, half, static_cast<float>(img_height) - half);

    plan.use = true;
    plan.center = c;
    plan.side = side;

    return plan;
  }

} // namespace training
