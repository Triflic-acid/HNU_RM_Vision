#pragma once
// 扩展卡尔曼滤波（EKF）—— CTRV 运动模型 + 笛卡尔位置量测
//
// 这是**新增文件**，不改动 kalman_cv.hpp / lesson04_tracking.cpp / tracking_state.hpp。
// 想用就把它加进 TrackingState 替换 KalmanCV 即可。
//
// ------------------------------------------------------------------
// 为什么用 EKF 而不是匀速卡尔曼
// ------------------------------------------------------------------
// 匀速卡尔曼（CV）在笛卡尔系是**线性**模型，用不上 EKF 的线性化。
// 要真正发挥 EKF，需要非线性过程模型：CTRV（恒定速度 + 恒定角速度转弯）。
//
//   x = [ px, py, v, θ, ω, pz, vz ]ᵀ
//
//   px' = px + (v/ω)·[ sin(θ+ωΔt) − sinθ ]
//   py' = py + (v/ω)·[ −cos(θ+ωΔt) + cosθ ]
//   v'  = v
//   θ'  = θ + ωΔt
//   ω'  = ω
//   pz' = pz + vz·Δt
//   vz' = vz
//
// 这个模型对两类目标都合适：
//   · 装甲板：车体一边平移一边自转（θ 是运动方向，ω 是转向角速度）
//   · 能量机关：叶片绕圆心做圆周运动（v = ω·r，θ 即相位）
//
// 量测仍是我们已有的 PnP 位置 z = [px,py,pz]，是线性的，所以更新步是普通 KF，
// 只有预测步需要在线性化点求雅可比 F = ∂f/∂x（本实现用**数值差分**，7 维足够快）。
//
// ------------------------------------------------------------------
// 参数怎么选
// ------------------------------------------------------------------
//   q_accel_xy : 水平加速度噪声（m/s²），目标机动越猛越大
//   q_accel_z  : 垂直加速度噪声（m/s²）
//   q_theta    : 航向抖动（rad），一般很小
//   q_omega    : 角加速度噪声（rad/s²），转弯越剧烈越大
//   r_pos      : PnP 位置噪声（m），3 m 处约 0.02
//
// 建议初值：q_accel_xy = 6.0，q_accel_z = 2.0，q_theta = 0.05，q_omega = 4.0，r_pos = 0.02

#include <Eigen/Dense>

#include <cmath>

namespace training
{
  class EkfCtrv
  {
  public:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Mat7 = Eigen::Matrix<double, 7, 7>;

    // 直接给定量测协方差（世界系 3x3），与 KalmanCV::set_R 语义一致
    // 自适应过程噪声开关（默认开）
    void set_adaptive(bool on) { adaptive_ = on; }
    double q_scale() const { return q_adapt_; }

    void set_R(const Eigen::Matrix3d &R)
    {
      R_ = 0.5 * (R + R.transpose());
    }

    // 与 KalmanCV 保持一致：目标速度物理上限（m/s）
    static constexpr double kMaxSpeed = 8.0;
    // 角速度物理上限（rad/s）：小陀螺实测 4~7，留余量到 12
    static constexpr double kMaxOmega = 12.0;
    // 垂直速度上限（m/s）：地面机器人几乎不上下移动，钳紧可以抑制深度噪声带来的 z 抖动
    static constexpr double kMaxVz = 1.5;

    void configure(double q_accel_xy = 6.0, double q_accel_z = 2.0,
                   double q_theta = 0.05, double q_omega = 4.0,
                   double r_pos = 0.02)
    {
      q_accel_xy_ = q_accel_xy;
      q_accel_z_ = q_accel_z;
      q_theta_ = q_theta;
      q_omega_ = q_omega;
      r_pos_ = r_pos;
    }

    bool initialized() const { return initialized_; }

    void reset()
    {
      initialized_ = false;
      x_.setZero();
      P_.setIdentity();
    }

    // 用第一帧位置初始化：速度/角速度未知，给大方差
    void init(const Eigen::Vector3d &p)
    {
      x_.setZero();
      x_(0) = p.x();
      x_(1) = p.y();
      x_(5) = p.z();

      P_.setZero();
      P_.diagonal() << R_(0, 0), R_(1, 1), 25.0, 1.0, 4.0, R_(2, 2), 4.0;
      initialized_ = true;
    }

    void predict(double dt)
    {
      if (!initialized_ || dt <= 0.0)
        return;

      const Vec7 x_pred = f(x_, dt);

      // 数值雅可比
      Mat7 F = Mat7::Identity();
      constexpr double h = 1e-6;
      for (int i = 0; i < 7; ++i)
      {
        Vec7 dx = Vec7::Zero();
        dx(i) = h;
        F.col(i) = (f(x_ + dx, dt) - f(x_ - dx, dt)) / (2.0 * h);
      }

      const Mat7 Q = process_noise(dt);

      x_ = x_pred;
      P_ = F * P_ * F.transpose() + Q;
      P_ = 0.5 * (P_ + P_.transpose());

      // ω 太小就钳到 0，避免 v/ω 数值爆炸
      if (std::abs(x_(4)) < 1e-4)
        x_(4) = 0.0;
    }

    void update(const Eigen::Vector3d &p)
    {
      if (!initialized_)
      {
        init(p);
        return;
      }

      Eigen::Matrix<double, 3, 7> H = Eigen::Matrix<double, 3, 7>::Zero();
      H(0, 0) = 1.0;
      H(1, 1) = 1.0;
      H(2, 5) = 1.0;

      const Eigen::Matrix3d &R = R_;
      Eigen::Vector3d y = p - H * x_;

      // ---- 自适应过程噪声 ----
      // 平移靶是正弦运动（幅度 0.5 m、周期 4 s → 最大加速度 ≈ 1.2 m/s²），
      // 固定 q_accel=0.3 会明显迟滞；但单纯调大 q 又会让静止靶抖。
      // 做法：看新息（残差）的大小动态调 q —— 大残差说明机动，放大 q 追上去；
      //       小残差说明平稳，缩小 q 平滑输出。
      if (adaptive_)
      {
        // 按**估计速度**自适应（比按新息稳定）：
        //   几乎静止 → q 收紧到 0.55 倍（抗抖动）
        //   明显运动 → q 放大到 2.2 倍（减 250 ms 外推的迟滞）
        const double sp = std::hypot(x_(2) * std::cos(x_(3)), x_(2) * std::sin(x_(3)));
        const double t = std::clamp((sp - 0.15) / (0.60 - 0.15), 0.0, 1.0);
        const double gain = 0.55 + (2.2 - 0.55) * t;
        q_adapt_ = 0.85 * q_adapt_ + 0.15 * gain;
        q_accel_eff_ = q_accel_xy_ * q_adapt_;
      }
      else
      {
        q_accel_eff_ = q_accel_xy_;
      }

      const Eigen::Matrix3d S = H * P_ * H.transpose() + R;
      const Eigen::Matrix<double, 7, 3> K = P_ * H.transpose() * S.inverse();

      x_ = x_ + K * y;

      const Mat7 I = Mat7::Identity();
      P_ = (I - K * H) * P_;
      P_ = 0.5 * (P_ + P_.transpose());

      // 物理钳位：速度、角速度都不允许超出实际机器人能力
      if (std::abs(x_(2)) > kMaxSpeed)
        x_(2) = std::copysign(kMaxSpeed, x_(2));
      if (std::abs(x_(4)) > kMaxOmega)
        x_(4) = std::copysign(kMaxOmega, x_(4));
      if (std::abs(x_(6)) > kMaxVz)
        x_(6) = std::copysign(kMaxVz, x_(6));
    }

    // 当前位置 / 速度（速度由极坐标还原成笛卡尔）
    Eigen::Vector3d position() const { return Eigen::Vector3d(x_(0), x_(1), x_(5)); }

    Eigen::Vector3d velocity() const
    {
      return Eigen::Vector3d(x_(2) * std::cos(x_(3)),
                             x_(2) * std::sin(x_(3)),
                             x_(6));
    }

    // 航向 / 角速度（符文场景可用于算 ω）
    double heading() const { return x_(3); }
    double turn_rate() const { return x_(4); }
    double speed() const { return x_(2); }

    // 按 CTRV 推导 dt 秒后位置（用于画预测框）
    Eigen::Vector3d predict_future(double dt) const
    {
      const Vec7 xf = f(x_, dt);
      return Eigen::Vector3d(xf(0), xf(1), xf(5));
    }

  private:
    static Vec7 f(const Vec7 &x, double dt)
    {
      const double v = x(2), th = x(3), w = x(4);
      Vec7 y = x;

      if (std::abs(w) > 1e-4)
      {
        y(0) = x(0) + (v / w) * (std::sin(th + w * dt) - std::sin(th));
        y(1) = x(1) + (v / w) * (-std::cos(th + w * dt) + std::cos(th));
      }
      else
      {
        // ω→0 的极限：退化成匀速直线
        y(0) = x(0) + v * std::cos(th) * dt;
        y(1) = x(1) + v * std::sin(th) * dt;
      }

      y(3) = th + w * dt;
      y(4) = w;
      y(5) = x(5) + x(6) * dt;
      y(6) = x(6);
      return y;
    }

    Mat7 process_noise(double dt) const
    {
      Mat7 Q = Mat7::Zero();
      const double qxy = q_accel_eff_ * q_accel_eff_;
      const double qz = q_accel_z_ * q_accel_z_;

      Q(0, 0) = qxy * dt * dt * dt / 3.0;
      Q(1, 1) = Q(0, 0);
      Q(2, 2) = qxy * dt; // 速度
      Q(3, 3) = q_theta_ * q_theta_ * dt;
      Q(4, 4) = q_omega_ * q_omega_ * dt;
      Q(5, 5) = qz * dt * dt * dt / 3.0;
      Q(6, 6) = qz * dt;

      // 位置/速度耦合项（与匀速卡尔曼一致）
      Q(0, 2) = Q(2, 0) = qxy * dt * dt / 2.0;
      Q(1, 2) = Q(2, 1) = Q(0, 2);
      Q(5, 6) = Q(6, 5) = qz * dt * dt / 2.0;
      return Q;
    }

    bool initialized_{false};
    double q_accel_xy_{6.0};
    double q_accel_eff_{6.0}; // 当前生效的（自适应后）
    bool adaptive_{true};
    double q_adapt_{1.0}; // 增益，1.0 = 不调整
    double q_accel_z_{2.0};
    double q_theta_{0.05};
    double q_omega_{4.0};
    double r_pos_{0.02};

    Eigen::Matrix3d R_{Eigen::Matrix3d::Identity() * 0.02 * 0.02};
    Vec7 x_{Vec7::Zero()};
    Mat7 P_{Mat7::Identity()};
  };
} // namespace training
