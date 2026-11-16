#pragma once
// 旋转目标 EKF：状态 = 转轴中心 + 相位 + 角速度 + 半径
//
// 设计来源：深圳大学 RobotPilots《RP-26AutoAim-Frame》公开接口
//   src/core/app/interface/type/TrackResult.hpp 中的 KalmanState：
//       X_Center, V_X_Center, Y_Armor, V_Y_Armor, Z_Center, V_Z_Center,
//       Theta(与竖直轴夹角, 俯视顺时针为正), V_W(角速度), Radius(转轴半径)
//   他们把**半径作为状态量**在线估计，这正是"每辆车半径不同"的解法；
//   同时半径/中心是慢变量，换装甲板（相位跳变）不会破坏它们。
//
// 本实现把该 9 维状态映射到我们的世界系（x 前、y 左、z 上）：
//   x = [ xc, vxc, yc, vyc, θ, ω, r, z, vz ]
//   量测 h(x) = [ xc + r·cosθ,  yc + r·sinθ,  z ]
//
// 与包内 spin_model.hpp 的区别：
//   SpinModel 用滑窗最小二乘离线估 C、r、ω，再外推；
//   本文件是真正的 EKF：噪声按协方差传播，r 与 ω 都有方差，预测自动加权。

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace training
{
  class EkfSpin
  {
  public:
    using Vec9 = Eigen::Matrix<double, 9, 1>;
    using Mat9 = Eigen::Matrix<double, 9, 9>;

    static constexpr double kMaxOmega = 12.0;   // rad/s
    static constexpr double kMaxSpeed = 8.0;    // m/s
    static constexpr double kMaxVz = 1.5;       // m/s
    static constexpr double kMinRadius = 0.08;  // m
    static constexpr double kMaxRadius = 0.35;  // m，硬上限
    // 可信区间：装甲环半径落在 [0.12, 0.28] 才认为 r 估得对
    static constexpr double kTrustRadiusLo = 0.12;
    static constexpr double kTrustRadiusHi = 0.28;

    void configure(double q_center = 1.0,   // 中心平动加速度噪声 m/s²
                   double q_theta = 0.30,   // 相位噪声 rad/√s
                   double q_omega = 0.60,   // 角加速度噪声 rad/s²
                   double q_radius = 0.02,  // 半径漂移 m/√s（很小：半径是常数）
                   double q_z = 0.02,       // 垂直加速度噪声
                   double r_meas = 0.05)    // 位置量测噪声 m
    {
      q_center_ = q_center;
      q_theta_ = q_theta;
      q_omega_ = q_omega;
      q_radius_ = q_radius;
      q_z_ = q_z;
      r_meas_ = r_meas;
      R_ = Eigen::Matrix3d::Identity() * r_meas_ * r_meas_;
    }

    void set_R(const Eigen::Matrix3d &R) { R_ = 0.5 * (R + R.transpose()); }

    // 板数不写死：给定候选集合（常见 {3,4}），由观测量在线判别。
    void set_armor_num_candidates(const std::vector<int> &cands)
    {
      candidates_.clear();
      for (int c : cands)
        if (c >= 2 && c <= 8)
          candidates_.push_back(c);
      if (candidates_.empty())
        candidates_.push_back(4);
    }
    int armor_num() const { return armor_num_; }

    // 固定半径：不再把 r 当状态估计，每帧强制回先验。
    // 理由（实测 2026-11-16）：r 会一路漂到上限 0.35（真实 0.20），
    // 导致圆周外推的弧长放大 75%，预测点飞到目标外。同济也是按车型给常数半径。
    void set_fixed_radius(bool fix) { fixed_radius_ = fix; }

    // 显式指定板数（按车型配置时用；调用后不再做在线判别，除非再设候选集）
    void set_armor_num(int n)
    {
      armor_num_ = std::clamp(n, 2, 8);
      candidates_.clear();
      candidates_.push_back(armor_num_);
    }

    void set_prior_radius(double r)
    {
      prior_r_ = std::clamp(r, kMinRadius, kMaxRadius);
      if (!initialized_)
        x_(6) = prior_r_;
    }

    bool initialized() const { return initialized_; }
    void reset() { initialized_ = false; x_.setZero(); P_.setIdentity(); }

    // 用第一个量测初始化：半径取先验，相位假设 0，中心反推
    void init(const Eigen::Vector3d &p)
    {
      x_.setZero();
      x_(6) = prior_r_;
      x_(0) = p.x() - prior_r_;   // 相位 0 => 板在中心右侧
      x_(2) = p.y();
      x_(7) = p.z();

      P_.setZero();
      P_(0, 0) = 4.0;   // 中心位置很不确定
      P_(2, 2) = 4.0;
      P_(1, 1) = 25.0;  // 速度
      P_(3, 3) = 25.0;
      P_(4, 4) = M_PI * M_PI;   // 相位完全未知
      P_(5, 5) = 16.0;          // ω 未知（±4 rad/s）
      P_(6, 6) = fixed_radius_ ? 1e-10 : 0.01 * 0.01;   // 半径：固定或按先验
      P_(7, 7) = R_(2, 2);
      P_(8, 8) = 4.0;
      initialized_ = true;
    }

    void predict(double dt)
    {
      if (!initialized_ || dt <= 0.0)
        return;

      // 线性状态转移（θ、r 不动，只有中心/相位/高度积分）
      Mat9 F = Mat9::Identity();
      F(0, 1) = dt;    // xc += vxc·dt
      F(2, 3) = dt;    // yc += vyc·dt
      F(4, 5) = dt;    // θ  += ω·dt
      F(7, 8) = dt;    // z  += vz·dt

      Mat9 Q = Mat9::Zero();
      const double qc = q_center_ * q_center_;
      Q(0, 0) = Q(2, 2) = qc * dt * dt * dt / 3.0;
      Q(0, 1) = Q(1, 0) = qc * dt * dt / 2.0;
      Q(2, 3) = Q(3, 2) = qc * dt * dt / 2.0;
      Q(1, 1) = Q(3, 3) = qc * dt;
      Q(4, 4) = q_theta_ * q_theta_ * dt;
      Q(5, 5) = q_omega_ * q_omega_ * dt;
      Q(6, 6) = fixed_radius_ ? 0.0 : q_radius_ * q_radius_ * dt;
      const double qz = q_z_ * q_z_;
      Q(7, 7) = qz * dt * dt * dt / 3.0;
      Q(7, 8) = Q(8, 7) = qz * dt * dt / 2.0;
      Q(8, 8) = qz * dt;

      x_ = F * x_;
      P_ = F * P_ * F.transpose() + Q;
      P_ = 0.5 * (P_ + P_.transpose());
      wrap_theta();
    }

    // 量测更新：p 为装甲板中心（世界系）
    void update(const Eigen::Vector3d &p)
    {
      if (!initialized_)
      {
        init(p);
        return;
      }

      const double th = x_(4), r = x_(6);

      Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
      H(0, 0) = 1.0;
      H(0, 4) = -r * std::sin(th);
      H(0, 6) = std::cos(th);
      H(1, 2) = 1.0;
      H(1, 4) = r * std::cos(th);
      H(1, 6) = std::sin(th);
      H(2, 7) = 1.0;

      const Eigen::Vector3d h(x_(0) + r * std::cos(th),
                              x_(2) + r * std::sin(th),
                              x_(7));
      const Eigen::Vector3d y = p - h;

      const Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
      const Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * S.inverse();

      x_ = x_ + K * y;
      const Mat9 I = Mat9::Identity();
      P_ = (I - K * H) * P_;
      P_ = 0.5 * (P_ + P_.transpose());

      clamp_state();
      wrap_theta();
    }

    // 带"装甲板朝向"的量测更新。
    //
    // 关键：只用位置(x,y,z)时，θ 与 ω 无法区分（滤波器总可以用观测去改 θ），
    // 同济的做法是把**装甲板自身的 yaw 角**也作为量测（他们的 ypda 里的 a）。
    // 这样 θ 被直接观测，ω 才能通过状态转移被学出来。
    //
    //   p      : 装甲板中心（世界系）
    //   a_world: 装甲板法向的 yaw（世界系，rad），可由 PnP 的 rvec 求出
    //   sigma_a: 该角度的噪声（rad），切换板/斜视时给大一些
    // 单块板更新（id = 该板在车体上的编号，0..N-1）
    //   θ_id = θ + id·2π/N
    void update_with_angle(const Eigen::Vector3d &p, double a_world, double sigma_a,
                           int id = 0, int armor_num = 4)
    {
      if (!initialized_)
      {
        init(p);
        return;
      }

      const double th = x_(4) + id * 2.0 * M_PI / std::max(armor_num, 1);
      const double r = x_(6);
      const double c = std::cos(th), s0 = std::sin(th);

      Eigen::Matrix<double, 4, 9> H = Eigen::Matrix<double, 4, 9>::Zero();
      H(0, 0) = 1.0;
      H(0, 4) = -r * s0;
      H(0, 6) = c;
      H(1, 2) = 1.0;
      H(1, 4) = r * c;
      H(1, 6) = s0;
      H(2, 7) = 1.0;
      H(3, 4) = 1.0;                       // 相位直接可观测

      Eigen::Matrix<double, 4, 1> y;
      y(0) = p.x() - (x_(0) + r * c);
      y(1) = p.y() - (x_(2) + r * s0);
      y(2) = p.z() - x_(7);
      double da = a_world - th;
      while (da > M_PI) da -= 2 * M_PI;
      while (da < -M_PI) da += 2 * M_PI;
      y(3) = da;

      Eigen::Matrix4d Rm = Eigen::Matrix4d::Zero();
      Rm.topLeftCorner<3, 3>() = R_;
      Rm(3, 3) = sigma_a * sigma_a;

      const Eigen::Matrix4d S = H * P_ * H.transpose() + Rm;
      const Eigen::Matrix<double, 9, 4> K = P_ * H.transpose() * S.inverse();

      x_ = x_ + K * y;
      const Mat9 I = Mat9::Identity();
      P_ = (I - K * H) * P_;
      P_ = 0.5 * (P_ + P_.transpose());

      clamp_state();
      wrap_theta();

      if (fixed_radius_)
      {
        x_(6) = prior_r_;                 // 强制回先验
        P_.row(6).setZero();
        P_.col(6).setZero();
        P_(6, 6) = 1e-10;
      }
    }

    // ---- 多板观测 ----
    struct PlateObs
    {
      Eigen::Vector3d p;      // 世界系板中心
      double yaw{0.0};        // 板法向世界 yaw（朝外）
      double sigma_a{0.2};    // 该板 yaw 的噪声
    };

    // 用当帧所有可见板更新：先按角度残差关联，再逐块更新。
    // 这是"整车观测"的关键——一块板消失/换板时，另一块仍在约束车的状态。
    // 返回成功关联的板数。
    int update_multi(const std::vector<PlateObs> &plates, int /*armor_num_hint*/ = 0)
    {
      if (plates.empty())
        return 0;
      if (!initialized_)
      {
        init(plates.front().p);
        return 0;
      }

      // ---- ① 板数判别：对每个候选 N 算"关联残差"，取残差最小者 ----
      // 残差定义：每块板相对预测相位的角度差（取最近的 id），残差越小说明 N 越对。
      if (candidates_.size() > 1 && plates.size() >= 2)
      {
        int best_n = armor_num_;
        double best_score = 1e9;
        for (int n : candidates_)
        {
          const double sc = gap_score(plates, n);
          if (sc < 0.0)                       // 该 N 下两块板重合，无法判定
            continue;
          if (sc < best_score)
          {
            best_score = sc;
            best_n = n;
          }
        }
        if (best_score < 0.15 && best_n != armor_num_)   // 残差 < 15% 且明显更好
        {
          if (best_n == pending_n_)
            ++pending_count_;
          else
          {
            pending_n_ = best_n;
            pending_count_ = 1;
          }
          if (pending_count_ >= 3)
          {
            armor_num_ = best_n;
            pending_count_ = 0;
          }
        }
        else
        {
          pending_count_ = 0;
        }
      }

      const int N = armor_num_;
      const double kAssocMax = 45.0 * M_PI / 180.0;   // 关联门限

      int used = 0;
      for (const auto &obs : plates)
      {
        double phi_meas = std::atan2(obs.p.y() - x_(2), obs.p.x() - x_(0));

        int best_id = -1;
        double best_err = 1e9;
        for (int id = 0; id < N; ++id)
        {
          const double th_id = x_(4) + id * 2.0 * M_PI / N;
          double d = phi_meas - th_id;
          while (d > M_PI) d -= 2 * M_PI;
          while (d < -M_PI) d += 2 * M_PI;
          if (std::abs(d) < best_err)
          {
            best_err = std::abs(d);
            best_id = id;
          }
        }
        if (best_id < 0 || best_err > kAssocMax)
          continue;

        update_with_angle(obs.p, obs.yaw, obs.sigma_a, best_id, N);
        ++used;
      }
      return used;
    }

    // 用"同帧两块板之间的夹角"判板数：
    //   真实夹角必然是 2π/N 的整数倍。只有一帧看到 ≥2 块板时才可判，
    //   只看到 1 块时 N 在数学上不可辨识，保持当前值。
    //   返回归一化残差（0 = 完全符合），<-1 无法判定。
    double gap_score(const std::vector<PlateObs> &plates, int N) const
    {
      const double step = 2.0 * M_PI / static_cast<double>(N);
      double err = 0.0;
      int cnt = 0;
      for (size_t i = 0; i < plates.size(); ++i)
      {
        for (size_t j = i + 1; j < plates.size(); ++j)
        {
          const double phi_i = std::atan2(plates[i].p.y() - x_(2), plates[i].p.x() - x_(0));
          const double phi_j = std::atan2(plates[j].p.y() - x_(2), plates[j].p.x() - x_(0));
          double g = phi_j - phi_i;
          while (g > M_PI) g -= 2 * M_PI;
          while (g < -M_PI) g += 2 * M_PI;
          const int k = static_cast<int>(std::lround(std::abs(g) / step));
          if (k == 0)
            continue;
          err += std::abs(std::abs(g) - k * step) / step;
          ++cnt;
        }
      }
      if (cnt == 0)
        return -1.0;                      // 无法判定
      return err / static_cast<double>(cnt);
    }

    // 预测 lead 秒后的位置
    Eigen::Vector3d predict_future(double lead) const
    {
      if (!initialized_)
        return Eigen::Vector3d::Zero();
      const double th = x_(4) + x_(5) * lead;
      const double r = x_(6);
      return Eigen::Vector3d(x_(0) + x_(1) * lead + r * std::cos(th),
                             x_(2) + x_(3) * lead + r * std::sin(th),
                             x_(7) + x_(8) * lead);
    }

    Eigen::Vector3d center() const { return Eigen::Vector3d(x_(0), x_(2), x_(7)); }
    double radius() const { return x_(6); }
    // r 是否落在可信区间（撞到 clamp 说明估计发散，此时不该用它外推）
    bool radius_trusted() const
    {
      return x_(6) > kTrustRadiusLo && x_(6) < kTrustRadiusHi;
    }
    double omega() const { return x_(5); }
    double phase() const { return x_(4); }

  private:
    void wrap_theta()
    {
      while (x_(4) > M_PI)
        x_(4) -= 2 * M_PI;
      while (x_(4) < -M_PI)
        x_(4) += 2 * M_PI;
    }

    void clamp_state()
    {
      x_(1) = std::clamp(x_(1), -kMaxSpeed, kMaxSpeed);
      x_(3) = std::clamp(x_(3), -kMaxSpeed, kMaxSpeed);
      x_(5) = std::clamp(x_(5), -kMaxOmega, kMaxOmega);
      x_(6) = std::clamp(x_(6), kMinRadius, kMaxRadius);
      x_(8) = std::clamp(x_(8), -kMaxVz, kMaxVz);
    }

    bool fixed_radius_{false};
    bool initialized_{false};
    double q_center_{1.0}, q_theta_{0.30}, q_omega_{0.60}, q_radius_{0.02};
    double q_z_{0.02}, r_meas_{0.05};
    double prior_r_{0.20};
    std::vector<int> candidates_{3, 4};
    int armor_num_{4};
    int pending_n_{-1};
    int pending_count_{0};
    double score_ema_{0.0};
    Eigen::Matrix3d R_{Eigen::Matrix3d::Identity() * 0.05 * 0.05};
    Vec9 x_{Vec9::Zero()};
    Mat9 P_{Mat9::Identity()};
  };
} // namespace training
