#pragma once
// 弹道解算 + 云台角求解（参考北京工业大学《弹道解算及云台角度求解算法》）
//
// 物理模型：重力 + 与速度平方成正比的空气阻力
//   dp/dt = v
//   dv/dt = -c·|v|·v + (0, 0, -g)
//   c = ρ·C_D·π·d² / (8m)          （球形弹丸）
//
// 命中条件：弹丸在 τ 秒后的位置 = 目标在 τ 秒后的位置
//   F(q) = p_bullet(ψ, ϑ, τ) − p̂_target(τ) = 0 ,  q = [ψ, ϑ, τ]
// 用 Newton 迭代求解，雅可比用数值差分（每次迭代 3 次额外积分，代价可忽略）。
//
// 坐标系：云台系（x 前、y 左、z 上），枪口在原点。
//   yaw   = atan2(y, x)
//   pitch = atan2(z, sqrt(x²+y²))

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace training
{
  // 空气阻力总开关。默认关（与 simulation/simulator/config.toml 的
  // [projectile.aerodynamics] enabled = false 一致）。
  // 运行时可用按键或环境变量打开，不改代码即可对比两种弹道。
  inline bool g_ballistics_drag{false};

  // 当前云台绝对角（rad）。由 main 每帧从 SensorData::imu_q 解出后写入，
  // lesson05 用它做开火门控（"指到哪"与"该指到哪"的差）。
  // 来源：lesson02 订阅的 vision_recv_data 里的 yaw / pitch / roll。
  inline double g_gimbal_yaw{0.0};
  inline double g_gimbal_pitch{0.0};

  // 云台系 → 世界系 的固定偏置角（rad）。
  // vision_recv_data 里的 yaw 与真实世界朝向差这个角度；
  // 可在运行时用 [ / ] 键微调，找到让云台真正对准目标的值。
  // 默认 0：真正的换算是 R_g2w = imu_q^T（见讲义），偏置只用于现场微调
  inline double g_gimbal_yaw_offset{0.0};

  // 预测提前量 n（秒）。讲义要求 n 是变量、暂定 250 ms；
  // 运行时可用 n 键在 {0, 100, 250, 500} ms 之间切换，logged 到 gui_events.log。
  inline double g_pred_lead_s{0.25};

  // 目标在**云台系**下的位置/速度。由 lesson04 直接写入（PnP 结果，不经过世界系变换），
  // 因此与"云台系→世界系"的约定无关。lesson05 用它算角度增量，天然消除坐标系偏置。
  inline Eigen::Vector3d g_target_gimbal{Eigen::Vector3d::Zero()};
  inline Eigen::Vector3d g_target_gimbal_vel{Eigen::Vector3d::Zero()};
  inline bool g_target_gimbal_valid{false};

  struct ProjectileParams
  {
    double v0{25.0};        // 初速 m/s
    double mass{0.0032};    // 弹丸质量 kg（17 mm 约 3.2 g）
    double diameter{0.017}; // 直径 m
    double Cd{0.47};        // 阻力系数（球体）
    double rho{1.225};      // 空气密度 kg/m³
    double g{9.8};          // 重力加速度
    double latency{0.0};    // 开火延迟 + 处理延迟（s）
    bool drag{true};        // 空气阻力开关（仿真 [projectile.aerodynamics] enabled=false）

    // c = ρ·C_D·π·d²/(8m)，关掉阻力时返回 0（积分自动退化成抛体）
    double drag_coeff() const
    {
      if (!drag)
        return 0.0;
      return rho * Cd * M_PI * diameter * diameter / (8.0 * mass);
    }
  };

  struct AimSolution
  {
    bool ok{false};
    double yaw{0.0};                              // rad
    double pitch{0.0};                            // rad
    double tau{0.0};                              // 飞行时间 s
    Eigen::Vector3d hit{Eigen::Vector3d::Zero()}; // 击打点（云台系）
    int iters{0};
  };

  // 弹道积分：RK4 推进 τ 秒，返回位置与速度
  inline void ballistic_rk4(const ProjectileParams &pp, double yaw, double pitch,
                            double tau, Eigen::Vector3d &p, Eigen::Vector3d &v)
  {
    const double c = pp.drag_coeff();
    const double v0 = pp.v0;
    p.setZero();
    v = Eigen::Vector3d(v0 * std::cos(pitch) * std::cos(yaw),
                        v0 * std::cos(pitch) * std::sin(yaw),
                        v0 * std::sin(pitch));

    auto acc = [&](const Eigen::Vector3d &vel) -> Eigen::Vector3d
    {
      const double s = c * vel.norm();
      return Eigen::Vector3d(-s * vel.x(), -s * vel.y(), -pp.g - s * vel.z());
    };

    const int N = std::max(8, static_cast<int>(tau / 0.002)); // 2 ms 步长
    const double h = (N > 0) ? tau / N : 0.0;
    for (int i = 0; i < N; ++i)
    {
      const Eigen::Vector3d k1p = v, k1v = acc(v);
      const Eigen::Vector3d k2p = v + 0.5 * h * k1v, k2v = acc(v + 0.5 * h * k1v);
      const Eigen::Vector3d k3p = v + 0.5 * h * k2v, k3v = acc(v + 0.5 * h * k2v);
      const Eigen::Vector3d k4p = v + h * k3v, k4v = acc(v + h * k3v);
      p += h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p);
      v += h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }
  }

  // 求云台角与飞行时间。
  //   p_target : 目标当前在云台系的位置
  //   v_target : 目标速度（云台系）
  inline AimSolution solve_aim(const Eigen::Vector3d &p_target,
                               const Eigen::Vector3d &v_target,
                               const ProjectileParams &pp,
                               int max_iters = 8)
  {
    AimSolution sol;
    const double horiz = std::hypot(p_target.x(), p_target.y());
    if (horiz < 0.5 || pp.v0 < 1.0)
      return sol;

    // 初值：直线弹道 + 先用"目标静止"的时间
    double tau = p_target.norm() / pp.v0;
    double yaw = std::atan2(p_target.y(), p_target.x());
    double pitch = std::atan2(p_target.z(), horiz);

    const double kClipYaw = 0.05, kClipPitch = 0.05, kClipTau = 0.05;

    for (int it = 0; it < max_iters; ++it)
    {
      // 目标在未来 τ（再加开火延迟）的位置
      const double t_hit = tau + pp.latency;
      const Eigen::Vector3d p_t = p_target + v_target * t_hit;

      Eigen::Vector3d p_b, v_b;
      ballistic_rk4(pp, yaw, pitch, tau, p_b, v_b);
      const Eigen::Vector3d F = p_b - p_t;
      if (F.norm() < 1e-3) // 1 mm
      {
        sol.ok = true;
        sol.iters = it;
        break;
      }

      // 数值雅可比 J = [∂F/∂ψ, ∂F/∂ϑ, ∂F/∂τ]
      Eigen::Matrix3d J;
      const double eps = 1e-4;
      Eigen::Vector3d pa, va;
      for (int k = 0; k < 3; ++k)
      {
        double dyaw = yaw, dpitch = pitch, dtau = tau + eps;
        if (k == 0)
        {
          dyaw = yaw + eps;
          dtau = tau;
        }
        if (k == 1)
        {
          dpitch = pitch + eps;
          dtau = tau;
        }
        ballistic_rk4(pp, dyaw, dpitch, dtau, pa, va);
        const Eigen::Vector3d p_t_k =
            (k == 2) ? (p_target + v_target * (dtau + pp.latency)) : (p_target + v_target * t_hit);
        J.col(k) = (pa - p_t_k - F) / eps;
      }

      Eigen::Vector3d dq = J.fullPivLu().solve(-F);
      if (!dq.allFinite())
        break;
      dq.x() = std::clamp(dq.x(), -kClipYaw, kClipYaw);
      dq.y() = std::clamp(dq.y(), -kClipPitch, kClipPitch);
      dq.z() = std::clamp(dq.z(), -kClipTau, kClipTau);
      yaw += dq.x();
      pitch += dq.y();
      tau += dq.z();
      if (tau < 0.01)
        tau = 0.01;
      sol.iters = it + 1;
    }

    // 最终校验
    Eigen::Vector3d p_b, v_b;
    ballistic_rk4(pp, yaw, pitch, tau, p_b, v_b);
    const Eigen::Vector3d p_t = p_target + v_target * (tau + pp.latency);
    sol.hit = p_t;
    sol.yaw = yaw;
    sol.pitch = pitch;
    sol.tau = tau;
    if (!sol.ok)
      sol.ok = (p_b - p_t).norm() < 0.02; // 20 mm
    return sol;
  }
} // namespace training
