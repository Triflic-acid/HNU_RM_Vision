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
} // namespace training