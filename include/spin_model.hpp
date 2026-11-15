#pragma once
// 旋转目标（小陀螺 / 前哨站）预测模型
//
// 为什么需要它：
//   CTRV 假设"速度方向以 ω 匀速旋转"，隐含**目标是同一个点**。
//   但小陀螺时装甲板绕车体中心公转，一块板转出视野后必须换下一块，
//   位置会跳 2r（约 0.4 m）。CTRV 把这种跳变当成"极高的角速度"，
//   ω 会被推到 8~12 rad/s，预测沿错误方向甩出去。
//
// 本模型把状态拆成两部分：
//   · 车体中心 C、半径 r  —— 慢变量，用滑动窗口的中位数估计，换板不影响
//   · 相位 φ              —— 每帧直接由当前观测重锚定，换板自动跟上
//   · 角速度 ω            —— 只用"相邻帧仍属同一块板"的样本做线性拟合
//
// 预测：P(t+Δ) = C + r·[cos(φ + ωΔ), sin(φ + ωΔ), 0] + (0,0,z + vz·Δ)

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace training
{
  class SpinModel
  {
  public:
    struct Sample
    {
      double t;          // 秒
      Eigen::Vector3d p; // 世界系
    };

    static constexpr size_t kWindow = 45;        // 约 2.5 s @18 FPS
    static constexpr double kMinRadius = 0.10;   // m，小于它说明不是绕圈运动
    static constexpr double kMinOmega = 1.0;     // rad/s，超过它才认为是旋转靶
    static constexpr double kMaxPhaseJump = 0.9; // rad，相邻帧相位差超过它视为换板

    void reset()
    {
      buf_.clear();
      center_.setZero();
      radius_ = 0.0;
      omega_ = 0.0;
      phase_ = 0.0;
      spin_ = false;
    }

    // 每帧喂一次当前观测（世界系位置）
    void push(double t_sec, const Eigen::Vector3d &p)
    {
      buf_.push_back({t_sec, p});
      while (buf_.size() > kWindow)
        buf_.pop_front();
      update();
    }

    bool spinning() const { return spin_; }
    double omega() const { return omega_; }
    double radius() const { return radius_; }
    Eigen::Vector3d center() const { return center_; }
    Eigen::Vector3d last() const { return buf_.empty() ? Eigen::Vector3d::Zero() : buf_.back().p; }

    // 预测 lead 秒后的位置（仅当 spinning() 为真时可信）
    Eigen::Vector3d predict(double lead_s) const
    {
      if (!spin_ || buf_.empty())
        return last();

      const double phi = phase_ + omega_ * lead_s;
      Eigen::Vector3d out = center_;
      out.x() += radius_ * std::cos(phi);
      out.y() += radius_ * std::sin(phi);
      out.z() = buf_.back().p.z(); // 高度不做圆周外推
      return out;
    }

  private:
    void update()
    {
      if (buf_.size() < 8)
      {
        spin_ = false;
        return;
      }

      // 1. 中心 / 半径：取中位数，抗换板造成的离群
      std::vector<double> xs, ys;
      xs.reserve(buf_.size());
      ys.reserve(buf_.size());
      for (const auto &s : buf_)
      {
        xs.push_back(s.p.x());
        ys.push_back(s.p.y());
      }
      std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
      std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
      center_.x() = xs[xs.size() / 2];
      center_.y() = ys[ys.size() / 2];
      center_.z() = buf_.back().p.z();

      std::vector<double> rs;
      rs.reserve(buf_.size());
      for (const auto &s : buf_)
        rs.push_back(std::hypot(s.p.x() - center_.x(), s.p.y() - center_.y()));
      std::nth_element(rs.begin(), rs.begin() + rs.size() / 2, rs.end());
      radius_ = rs[rs.size() / 2];

      // 2. 相位：由当前观测直接算，换板后自动跟上
      const Eigen::Vector3d &p = buf_.back().p;
      phase_ = std::atan2(p.y() - center_.y(), p.x() - center_.x());

      // 3. 角速度：只用"相邻帧同板"的样本做最小二乘（t 对 unwrap 后的 φ）
      double sum_t = 0, sum_phi = 0, sum_tt = 0, sum_tphi = 0;
      int n = 0;
      double prev_phi = 0;
      bool has_prev = false;
      for (const auto &s : buf_)
      {
        const double phi = std::atan2(s.p.y() - center_.y(), s.p.x() - center_.x());
        if (has_prev)
        {
          double d = phi - prev_phi;
          while (d > M_PI)
            d -= 2 * M_PI;
          while (d < -M_PI)
            d += 2 * M_PI;
          if (std::abs(d) < kMaxPhaseJump) // 否则视为换板，跳过
          {
            sum_t += s.t;
            sum_phi += phi;
            sum_tt += s.t * s.t;
            sum_tphi += s.t * phi;
            ++n;
          }
        }
        prev_phi = phi;
        has_prev = true;
      }

      if (n < 6)
      {
        spin_ = false;
        omega_ = 0.0;
        return;
      }

      const double den = n * sum_tt - sum_t * sum_t;
      omega_ = (std::abs(den) < 1e-9) ? 0.0 : (n * sum_tphi - sum_t * sum_phi) / den;

      spin_ = (std::abs(omega_) > kMinOmega) && (radius_ > kMinRadius);
    }

    std::deque<Sample> buf_;
    Eigen::Vector3d center_{Eigen::Vector3d::Zero()};
    double radius_{0.0};
    double omega_{0.0};
    double phase_{0.0};
    bool spin_{false};
  };
} // namespace training
