#include "lessons.hpp"

namespace training
{
  // 第四讲：位姿、坐标与基础跟踪
  //
  // 输入格式：
  //   detections : const std::vector<Detection>&，第三讲输出的本帧观测。
  //                mode=kArmor 时是装甲板，mode=kRune 时是符文叶片；
  //                符文叶片的 Detection::center 里带着 R 标圆心（全帧共享）。
  //   mode       : DetMode，决定解算分支：
  //                · kArmor：用装甲板四角 + 大小板物理尺寸做 solvePnP
  //                · kRune ：叶片四角 + 圆心 → 极坐标 (r, θ)，再做角速度估计
  //   timestamp  : TimePoint，本帧图像时刻；只能与其他 TimePoint 求时间差。
  //   imu_q      : const Eigen::Quaterniond&，图像时刻的归一化姿态四元数。
  //   calibration: const CameraCalibration&：
  //                matrix=3x3 CV_64F 相机内参；distortion=1x5 CV_64F 畸变；
  //                camera_to_gimbal=相机到云台的 3x3 旋转矩阵；
  //                camera_to_gimbal_translation=相机到云台平移，单位 m。
  //   state      : TrackingState&，仅本讲使用的跨帧状态；函数需要原地更新。
  //
  // 输出格式：
  //   std::optional<Target>。有目标时：
  //   - position : IMU 稳定参考系中、相对云台原点的目标向量，单位 m；
  //   - velocity : 同一参考坐标系中的目标速度，单位 m/s。
  //   本课程统一使用 x 向前、y 向左、z 向上的右手坐标系。
  //   无有效观测、PnP 失败或跟踪未建立时返回 std::nullopt。
  //
  // 打符思路：
  //   1) 叶片中心 b = 四角均值；R 标圆心 C 来自 Detection::center
  //   2) r = ‖b − C‖，θ = atan2(b.y − C.y, b.x − C.x)
  //   3) ω = Δθ/Δt（注意 ±π 展开，用 std::remainder）
  //   4) 叶片三维位置用四角 solvePnP；击打点是"叶片绕 C 旋转 ω·t_flight 后"的位置
  //   5) 只用 1 片叶子也能解：r 与 θ 都来自这一片 + 圆心
  std::optional<Target> lesson04_tracking(const std::vector<Detection> &detections,
                                          DetMode mode, TimePoint timestamp, const Eigen::Quaterniond &imu_q,
                                          const CameraCalibration &calibration, TrackingState &state)
  {
    // ===== 实现区 BEGIN =====
    (void)detections;
    (void)mode;
    (void)timestamp;
    (void)imu_q;
    (void)calibration;
    (void)state;
    return std::nullopt;
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
