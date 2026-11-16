#include "lessons.hpp"

#include "ballistics.hpp"

#include <algorithm>
#include <cmath>

namespace training {

namespace {
// 开火角误差阈值。仿真器 gimbal_command_deadband_deg = 0.05°，
// 但那是"指令死区"，开火判据要宽一些：0.5° ≈ 5 m 处 44 mm，仍在装甲板内。
constexpr double kFireAngleTol = 0.5 * M_PI / 180.0;
}  // namespace

// 第五讲：弹道解算
//
// 输入格式：
//   target       : const Target&；position 单位 m，velocity 单位 m/s，二者均采用
//                  第四讲规定的 x 向前、y 向左、z 向上坐标系。
//                  装甲板与能量机关共用该结构，区别只在于第四讲怎么算出它。
//   timestamp    : TimePoint，target 对应的图像时刻，用于计算预测时间。
//   bullet_speed : double，弹丸初速度，单位 m/s，必须为有限正数。
//
// 输出格式：
//   Command：
//   - control : true 表示请求云台跟随；无法产生安全解时必须为 false；
//   - shoot   : true 表示请求发射，最终仍受 IO 层 allow_fire 和有效性门控；
//   - yaw     : 世界/IMU 参考系下的绝对偏航角，单位 rad；
//   - pitch   : 世界/IMU 参考系下的绝对俯仰角，单位 rad。
//
// 验收：仿真器完成静止靶、平移靶任务。
//
// 打符额外注意：
//   能量机关是圆周运动，目标点在 t_flight 内会转过 ω·t_flight，
//   所以 position 应当由第四讲**已经预测到击打时刻**再传进来；
//   也可以在这里用 velocity 做线性外推，但圆周运动下线性外推误差较大。
Command lesson05_aiming(const Target& target, TimePoint timestamp, double bullet_speed) {
  (void)timestamp;

  Command cmd;

  // 参数校验：弹速必须是有限正数
  if (!(bullet_speed > 1.0) || !std::isfinite(bullet_speed))
    return cmd;

  // ---- 弹道参数：直接从仿真器模型读 ----
  //   simulation/simulator/config.toml
  //     [projectile] speed = 25.0, diameter = 0.017, mass = 0.0032
  //     [projectile.aerodynamics] enabled = false, air_density = 1.225, Cd = 0.22516
  ProjectileParams pp;
  pp.v0 = bullet_speed;          // 用 IO 层读到的真实初速
  pp.diameter = 0.017;           // 17 mm
  pp.mass = 0.0032;              // 3.2 g
  pp.Cd = 0.22516;               // 仿真配置里的球体阻力系数
  pp.rho = 1.225;
  pp.drag = g_ballistics_drag;   // 开关（默认关，与仿真一致）
  pp.g = 9.8;
  pp.latency = 0.0;              // 第五讲拿不到状态里的延迟，先置 0

  // ---- 解算 ----
  // Target 已经在"IMU 稳定参考系、相对云台原点"，与云台系同向（x 前 y 左 z 上），
  // 所以解出的 yaw/pitch 就是**绝对角**，可直接下发给电控。
  Eigen::Vector3d p = target.position;
  Eigen::Vector3d v = target.velocity;

  // 目标太近：不给解。注意**不能**因为 p.x() <= 0 就拒绝——
  // 云台初始朝向偏 90° 时目标在侧面（x≈0、y=±2），一票否决会导致
  // "永远不发指令 → 云台永远不转 → 目标永远在侧面" 的死锁（实测 2026-11-16）。
  if (p.head<2>().norm() < 0.5)
    return cmd;

  const AimSolution sol = solve_aim(p, v, pp);
  if (!sol.ok)
    return cmd;

  // ---- 输出 ----
  constexpr double kPitchLimit = 0.785;   // 与仿真器 gimbal_pitch_limit 一致（±45°）
  cmd.yaw = std::remainder(sol.yaw, 2.0 * M_PI);          // 归一化到 [-π, π]
  cmd.pitch = std::clamp(sol.pitch, -kPitchLimit, kPitchLimit);
  cmd.control = true;

  // 开火门控：云台当前角与解算角的误差都小于阈值才允许开火。
  // 当前角来自 lesson02 订阅的 vision_recv_data（main 每帧写进 g_gimbal_*）。
  const double dyaw = std::remainder(cmd.yaw - g_gimbal_yaw, 2.0 * M_PI);
  const double dpitch = cmd.pitch - g_gimbal_pitch;
  cmd.shoot = std::abs(dyaw) < kFireAngleTol && std::abs(dpitch) < kFireAngleTol;
  return cmd;
}
}  // namespace training
