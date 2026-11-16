/*
lesson4 实现
参考同济大学开源项目 sp_vision_35
*/

#include "lessons.hpp"
#include "debug_flags.hpp"
#include "ballistics.hpp"
#include "kalman_cv.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cstdlib>
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
    constexpr double kQAccel{0.30};   // 水平加速度标准差 m/s²
    constexpr double kRPosInit{0.08}; // 初始量测标准差 m（之后被 set_R 覆盖）
    constexpr double kQAccelZ{0.02};  // 垂直加速度噪声（几乎为 0：地面机器人不做垂直机动）
    constexpr double kQTheta{0.05};   // 航向抖动 rad
    constexpr double kQOmega{0.40};   // 角加速度噪声 rad/s²（2.0 时 ω 被噪声带跑）

    // 车体上的装甲板块数。参考同济 sp_vision_25 的 set_target()：
    //   · 平衡步兵（大板 + 3/4/5 号）      r=0.2000, N=2
    //   · 前哨站                          r=0.2765, N=3
    //   · 基地                            r=0.3205, N=3
    //   · 普通步兵/英雄/哨兵               r=0.2000, N=4
    // 注意 N 真的不总是 4 —— 平衡步兵只有 2 块。
    int armor_count(int class_id, bool large_armor)
    {
      if (large_armor && (class_id == 3 || class_id == 4 || class_id == 5))
        return 2; // 平衡步兵
      if (class_id == 0)
        return 3; // 前哨站
      return 4;
    }

    // 装甲环半径先验（m），数值取自同济 sp_vision_25 的实车配置：
    //   普通车 0.20 / 前哨站 0.2765 / 基地 0.3205
    // 用途：圆周 EKF 里 r 的初值。
    double armor_ring_prior(int class_id)
    {
      // 数值来自同济 sp_vision_25 的实际配置（普通 0.2 / 前哨站 0.2765）
      if (class_id == 0)
        return 0.2765; // 前哨站
      return 0.2000;   // 其余
    }

    // 预测位移上限：速度估计有噪声时，0.25 s 外推会把误差放大成米级跳变
    constexpr double kPredMaxShift{0.80};

    // ---------- 观测有效性门限 ----------
    constexpr float kMinBoxPx{40.0F}; // 框宽下限：小于它解算不可信
    // 切板死区：新候选正面度必须比锁定板高这么多才切（0.17 ≈ 10°）
    constexpr double kSwitchDeadZone{0.17};
    // 旋转 EKF：omega 超过它才认为是旋转靶；装甲板 yaw 的量测噪声
    // 整车观测（多板圆周 EKF）。打开后旋转段用它的预测；关闭则退回纯 CTRV。
    constexpr bool kOnlineArmorNum{false}; // 板数在线判别（默认关：单板时不可辨识，易抖动）
    constexpr bool kEnableSpinEkf{true};
    constexpr double kSpinOmegaThresh{1.0}; // rad/s
    constexpr double kAngleSigma{0.20};     // rad
    constexpr double kDistRatioMax{1.8};    // PnP 距离 / 单目距离 上限
    constexpr double kDistRatioMin{0.55};   // 下限
    // 时间维度的距离比：超过它认为换了个物体（不是同一块板）
    constexpr double kDistJumpRatio{1.35};
    // 连续被门 2b 拒绝多少帧后强制重新同步（防止 EMA 陈旧导致永久锁死）
    constexpr int kDistRejectMax{12};

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

    // 用二维框宽估测距离（针孔模型，不需要 PnP）：
    //   像素宽 w = fx * W / Z  ->  Z ≈ fx * W / w
    double estimate_distance(const Detection &d, double fx)
    {
      const double W = d.large_armor ? kLargeW : kSmallW;
      const double w = std::max(static_cast<double>(d.box.width), 1.0);
      return fx * W / w;
    }

    // 装甲板"正面度"：≈ cos(法向与视线夹角)
    //   投影宽度 w = fx·W·cosα / z  →  cosα = w·z /(fx·W)
    // 吉大的做法是"打最正对的那块板"；这里用同一判据，且不需要 PnP。
    double frontality(const Detection &d, double fx)
    {
      const double W = d.large_armor ? kLargeW : kSmallW;
      const double z = fx * W / std::max(static_cast<double>(d.box.width), 1.0);
      return std::clamp(d.box.width * z / (fx * W), 0.0, 1.0);
    }

    cv::Point2f center_of(const cv::Point2f c[4])
    {
      return (c[0] + c[1] + c[2] + c[3]) * 0.25F;
    }

    // 装甲板法向在世界系下的 yaw（rad），并消去 PI 的符号模糊。
    //
    // 装甲板是左右对称的，PnP 给出的法向可以整体翻 180 度而不改变重投影，
    // 直接把 atan2 结果喂进滤波器会让 theta 来回跳 pi，omega 被带飞。
    // 真实装甲板的法向一定**朝外**（背离车体中心），用它来定符号：
    //   令 r_hat = (p - C)/|p - C|，若 n·r_hat < 0 则取 -n。
    double armor_yaw_world(const cv::Vec3d &rvec, const CameraCalibration &cal,
                           const Eigen::Quaterniond &imu_q,
                           const Eigen::Vector3d &center,
                           const Eigen::Vector3d &p_world)
    {
      cv::Mat R;
      cv::Rodrigues(rvec, R);
      const Eigen::Matrix3d Rc2g =
          (cal.camera_to_gimbal.rows() == 3 && cal.camera_to_gimbal.cols() == 3)
              ? cal.camera_to_gimbal
              : Eigen::Matrix3d::Identity();
      Eigen::Vector3d n(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
      Eigen::Vector3d nw = imu_q.toRotationMatrix() * Rc2g * n;
      nw.z() = 0.0; // 只用水平分量算 yaw

      const Eigen::Vector3d radial(p_world.x() - center.x(),
                                   p_world.y() - center.y(), 0.0);
      if (nw.norm() > 1e-6 && radial.norm() > 1e-6 && nw.dot(radial) < 0.0)
        nw = -nw; // 翻到朝外的那一侧

      return std::atan2(nw.y(), nw.x());
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
      // 双解消歧当前关闭：只按重投影误差选最小（见文档"已知问题"）
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
    // 云台/相机相对底盘的固定偏置（rad）。
    // 实测（2026-11-16）：直接按 vision_recv_data 的 yaw 建 R_g2w 时，
    // 目标世界坐标会整体转 -90°（p_world ≈ Rz(-90°)·p_gimbal），
    // 导致 lesson05 算出 yaw≈-92° 而不是 0°，云台永远对不上。
    // 这里补 +90°。可用环境变量 GIMBAL_YAW_OFFSET_DEG 现场标定。
    double gimbal_yaw_offset()
    {
      static const double v = []
      {
        double deg = 90.0;
        if (const char *e = std::getenv("GIMBAL_YAW_OFFSET_DEG"))
          deg = std::atof(e);
        return deg * M_PI / 180.0;
      }();
      return v;
    }

    Eigen::Matrix3d r_gimbal_to_world(const Eigen::Quaterniond &imu_q)
    {
      const double off = gimbal_yaw_offset();
      if (std::abs(off) < 1e-9)
        return imu_q.toRotationMatrix();
      Eigen::Matrix3d Rz = Eigen::Matrix3d::Identity();
      Rz(0, 0) = std::cos(off);
      Rz(0, 1) = -std::sin(off);
      Rz(1, 0) = std::sin(off);
      Rz(1, 1) = std::cos(off);
      return Rz * imu_q.toRotationMatrix();
    }

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
      return r_gimbal_to_world(imu_q) * p_gimbal;
    }
    // 世界系 -> 相机系（用于标注预测框）
    Eigen::Vector3d to_camera(const Eigen::Vector3d &p_world,
                              const CameraCalibration &cal,
                              const Eigen::Quaterniond &imu_q)
    {
      const Eigen::Matrix3d R_c2g = cal.camera_to_gimbal;
      const Eigen::Vector3d t_c2g = cal.camera_to_gimbal_translation;
      const Eigen::Matrix3d R_g2w = r_gimbal_to_world(imu_q);
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

    // 选板：正面度最大者（吉大方案），带死区迟滞避免帧间反复切板。
    // 门 1：框太小时单目距离和 PnP 深度都不可信，直接跳过。
    int best_idx{-1};
    double best_face{-1.0};
    for (size_t i{0}; i < detections.size(); ++i)
    {
      if (detections[i].box.width < kMinBoxPx)
        continue;
      const double face{frontality(detections[i], fx)};
      if (face > best_face)
      {
        best_face = face;
        best_idx = static_cast<int>(i);
      }
    }
    if (best_idx < 0)
      return std::nullopt;

    // 死区：上一帧锁定的板仍可见时，只有新候选"明显更正对"才切板
    if (state.target_px.x >= 0.0F && state.locked_face > 0.0F)
    {
      for (size_t i{0}; i < detections.size(); ++i)
      {
        const cv::Point2f c{center_of(detections[i].corners.data())};
        const double dpx{cv::norm(c - state.target_px)};
        const bool same_plate =
            dpx < 1.0 * std::max(detections[i].box.width, 8.0F) ||
            (detections[i].class_id == state.target_class_id &&
             dpx < 2.0 * std::max(detections[i].box.width, 8.0F));
        if (same_plate)
        {
          const double face{frontality(detections[i], fx)};
          if (face + kSwitchDeadZone > best_face) // 没明显更正对 → 保持锁定
          {
            best_idx = static_cast<int>(i);
            best_face = face;
          }
          break;
        }
      }
    }
    state.locked_face = static_cast<float>(best_face);

    state.selected_idx = best_idx;
    const Detection &d{detections[static_cast<size_t>(best_idx)]};
    const cv::Point2f px{center_of(d.corners.data())};

    // 注：半径先验现在由 spin_ekf 自己设置（见旋转段 set_prior_radius），
    // 旧的 SpinModel 已不再参与主链路，这里不再调用它的接口。

    // 2. PnP解算
    cv::Vec3d rvec, tvec;
    double err_px{0.0};
    if (!solve_armor(d, calibration, rvec, tvec, err_px))
      return degrade(); // PnP 失败返回本帧无有效观测

    // 3. 坐标变换
    const Eigen::Vector3d p_cam(tvec[0], tvec[1], tvec[2]);

    // 门 2b：时间维度——距离不能突然跳变。
    // 实测（2026-11-16）：选板偶尔挑到 4~5.4 m 的远处目标（正常 2.7 m），
    // 它的 z 更高（2.9 m vs 1.9 m），一帧就把垂直估计拽上去 → 上下抖动。
    {
      const double d_now = p_cam.norm();
      const bool jump =
          state.dist_ema > 0.5 &&
          (d_now > kDistJumpRatio * state.dist_ema || d_now < state.dist_ema / kDistJumpRatio);

      if (jump)
      {
        // 逃生：连续拒绝超过 kDistRejectMax 帧，说明是我们自己的 EMA 陈旧了
        // （目标真的走远了/靠近了），此时重新同步，不能永久锁死。
        if (++state.dist_reject > kDistRejectMax)
        {
          state.dist_ema = d_now;
          state.dist_reject = 0;
        }
        else
        {
          return degrade();
        }
      }
      else
      {
        state.dist_reject = 0;
        state.dist_ema = (state.dist_ema > 0.0) ? 0.9 * state.dist_ema + 0.1 * d_now : d_now;
      }
    }

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

      // 云台快速转动时，imu 四元数与图像时刻不同步，世界系位置会额外摆动。
      // 用"相对上一帧的转角"放大量测噪声，让滤波器自动降低对这类帧的信任。
      double inflate = 1.0;
      if (state.has_prev_q)
      {
        const double dq = state.prev_q.angularDistance(imu_q); // rad，两帧转角
        const double dtq =
            std::max(std::chrono::duration<double>(timestamp - state.prev_q_stamp).count(), 1e-3);
        const double rate = dq / dtq;                     // rad/s
        inflate = 1.0 + std::clamp(rate / 2.0, 0.0, 8.0); // 2 rad/s 起明显放大
      }
      state.prev_q = imu_q;
      state.prev_q_stamp = timestamp;
      state.has_prev_q = true;

      if (training::dbg::cv_filter)
        state.kf.set_R(M * R_cam * M.transpose() * inflate * inflate);
      state.ekf.set_R(M * R_cam * M.transpose() * inflate * inflate);
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

    // 流水线延迟：图像时刻 -> 现在（含传输、推理、决策）。
    // 同济把这类延迟单独列出来（图像传输 + 处理 + 通信 + 控制 ≈15 ms），
    // 并指出预测时间必须把它们加在飞行时间之外，否则瞄准系统性滞后。
    {
      const auto now_tp = std::chrono::steady_clock::now();
      const double lat = std::chrono::duration<double>(now_tp - timestamp).count();
      state.pipeline_latency_s = std::clamp(lat, 0.0, 0.5);
    }

    // 用 CTRV-EKF 同时覆盖两种目标：
    //   · 直线平移：ω≈0，退化成匀速模型
    //   · 小陀螺/前哨站：ω 描述绕车体中心的圆周运动（匀速卡尔曼在这里必然跟不上）
    if (!state.ekf.initialized())
    {
      state.ekf.configure(kQAccel, kQAccelZ, kQTheta, kQOmega, kRPosInit);
      state.ekf.init(p_world);
    }
    else
    {
      state.ekf.predict(dt);
      state.ekf.update(p_world);
    }

    state.position = state.ekf.position();
    state.velocity = state.ekf.velocity();
    state.spin_rate = state.ekf.turn_rate(); // rad/s，供日志/诊断

    // ---- 对比开关：同时跑匀速卡尔曼，供离线对比（不影响主链路）----
    if (training::dbg::cv_filter)
    {
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
      state.cv_position = state.kf.position();
      state.cv_velocity = state.kf.velocity();
      state.cv_valid = true;
    }

    // 非 tracking 模式下不发送目标数据
    if (state.phase != TrackingState::Phase::kTracking)
    {
      state.has_pred = false;
      return std::nullopt;
    }

    // 6. 返回预测框 & 250ms历史数据
    // 旋转模型**默认关闭**。实测（outputs/history）旋转段位移 90 分位：
    //   仅 CTRV        ：87~177 mm   ← 最好
    //   + 旋转 EKF     ：672~845 mm  ← 更差
    // 所以统一用 CTRV 出预测；旋转 EKF 代码保留，改开关即可对比。
    if (kEnableSpinEkf)
    {
      // 整车观测：把当帧**所有**可见板都做一次 PnP，一起喂给圆周 EKF。
      // 一块板转走/换板时，另一块仍在约束车的中心/转速，跳变被吸收。
      if (!state.spin_ekf.initialized())
      {
        state.spin_ekf.set_prior_radius(armor_ring_prior(d.class_id));
        state.spin_ekf.init(p_world);
      }
      state.spin_ekf.predict(dt);

      const int N = armor_count(d.class_id, d.large_armor);
      std::vector<EkfSpin::PlateObs> plates;
      plates.reserve(detections.size());
      for (const auto &d2 : detections)
      {
        if (d2.box.width < kMinBoxPx)
          continue;
        cv::Vec3d rv2, tv2;
        double e2 = 0.0;
        if (!solve_armor(d2, calibration, rv2, tv2, e2))
          continue;
        const Eigen::Vector3d p2(tv2[0], tv2[1], tv2[2]);
        const Eigen::Vector3d w2 = to_world(p2, calibration, imu_q);
        EkfSpin::PlateObs o;
        o.p = w2;
        o.yaw = armor_yaw_world(rv2, calibration, imu_q, state.spin_ekf.center(), w2);
        o.sigma_a = kAngleSigma;
        plates.push_back(o);
      }
      // 板数：先按车型给初值；在线判别默认关闭（见文档"板数辨识"一节）
      state.spin_ekf.set_armor_num(N);
      if (kOnlineArmorNum)
        state.spin_ekf.set_armor_num_candidates({3, 4});
      state.spin_ekf.update_multi(plates, N);

      const double w_spin = state.spin_ekf.omega();
      // 半径撞到边界说明圆周模型发散 → 退回 CTRV，别用错弧长外推
      state.spin_active = std::abs(w_spin) > kSpinOmegaThresh &&
                          state.spin_ekf.radius_trusted();
      if (state.spin_active)
        state.spin_rate = w_spin;
    }
    else
    {
      state.spin_active = false; // 只用 CTRV
    }

    Eigen::Vector3d p_pred = state.spin_active
                                 ? state.spin_ekf.predict_future(kPredLeadS)
                                 : state.ekf.predict_future(kPredLeadS);
    if (!state.spin_active)
    {
      // 速度估计不准时，250 ms 外推会把位置放大成米级跳变；限幅后最坏只是"提前量不足"
      // 旋转模型走的是圆周，位移本来就可能接近直径，不做限幅
      const Eigen::Vector3d shift{p_pred - state.ekf.position()};
      const double n{shift.norm()};
      if (n > kPredMaxShift)
        p_pred = state.ekf.position() + shift * (kPredMaxShift / n);
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

    // ---- 对比开关：CV 的预测框（品红）----
    if (training::dbg::cv_filter && state.cv_valid)
    {
      const Eigen::Vector3d p_cv = state.kf.predict_future(kPredLeadS);
      const Eigen::Vector3d p_cv_cam = to_camera(p_cv, calibration, imu_q);
      std::vector<cv::Point3f> obj_cv;
      for (const auto &o : obj)
        obj_cv.emplace_back(o.x + static_cast<float>(p_cv_cam.x()),
                            o.y + static_cast<float>(p_cv_cam.y()),
                            o.z + static_cast<float>(p_cv_cam.z()));
      std::vector<cv::Point2f> proj_cv;
      cv::projectPoints(obj_cv, cv::Vec3d(0, 0, 0),
                        cv::Vec3d(p_cv_cam.x(), p_cv_cam.y(), p_cv_cam.z()),
                        calibration.matrix, calibration.distortion, proj_cv);
      if (proj_cv.size() == 4)
      {
        for (int i{0}; i < 4; ++i)
          state.cv_pred_corners[i] = proj_cv[static_cast<size_t>(i)];
        state.cv_has_pred = true;
      }
    }
    else
    {
      state.cv_has_pred = false;
    }

    // ---- 弹道解算（北工大《弹道解算及云台角度求解算法》）----
    // 目标世界状态转到云台系 → Newton 解 (yaw, pitch, τ) → 输出击打点
    {
      ProjectileParams pp;
      pp.v0 = 25.0; // 先用仿真默认弹速
      pp.rho = 1.225;
      pp.drag = g_ballistics_drag;           // 空气阻力开关
      pp.latency = state.pipeline_latency_s; // 处理+开火延迟
      const Eigen::Matrix3d R_g2w = r_gimbal_to_world(imu_q);
      const Eigen::Vector3d p_g = R_g2w.transpose() * state.position;
      const Eigen::Vector3d v_g = R_g2w.transpose() * state.velocity;

      const AimSolution aim = solve_aim(p_g, v_g, pp);
      state.aim_valid = aim.ok;
      if (aim.ok)
      {
        state.aim_yaw = aim.yaw;
        state.aim_pitch = aim.pitch;
        state.aim_tau = aim.tau;
        state.aim_point = R_g2w * aim.hit; // 击打点转回世界系
      }
    }

    state.pred_hist.push_back({timestamp, p_pred,
                               d.class_id, static_cast<float>(d.confidence), d.box.width,
                               static_cast<float>(state.distance_m),
                               static_cast<float>(err_px),
                               state.phase,
                               state.same_target_frames,
                               state.lost_frames,
                               static_cast<float>(state.spin_rate),
                               static_cast<float>(state.velocity.z()),
                               static_cast<float>(state.spin_ekf.radius()),
                               state.spin_active ? 1 : 0,
                               static_cast<float>(state.pipeline_latency_s * 1000.0),
                               static_cast<float>(state.aim_point.x()),
                               static_cast<float>(state.aim_point.y()),
                               static_cast<float>(state.aim_point.z()),
                               static_cast<float>(state.aim_tau)});
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
