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
//   · 车体中心 C、半径 r  —— 慢变量，用滑动窗口内的点做「已知半径的圆拟合」，换板不影响
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
    static constexpr double kMinRadius = 0.08;   // m，小于它说明不是绕圈运动
    static constexpr double kMaxRadius = 0.35;   // m，装甲环半径的物理上限
    static constexpr double kPriorRadius = 0.20; // m，仿真/实车的先验半径
    static constexpr double kMinOmega = 1.0;     // rad/s，超过它才认为是旋转靶
    static constexpr double kMaxPhaseJump = 0.9; // rad，相邻帧相位差超过它视为换板
    static constexpr double kMaxFitRms = 0.05;   // m，圆拟合残差上限（超过则认为本窗口不可信）
    // 半径在线估计：只有弧覆盖 ≥180° 时自由拟合才可信（实测 90° 时误差 39 mm，180° 时 4 mm）
    static constexpr double kMinArcForRadius = M_PI;
    static constexpr double kRadiusEmaAlpha = 0.15; // 半径 EMA 系数（慢更新）

    void reset()
    {
      buf_.clear();
      center_.setZero();
      radius_ = 0.0;
      omega_ = 0.0;
      phase_ = 0.0;
      spin_ = false;
      fit_rms_ = 1e9;
    }

    // 每帧喂一次当前观测（世界系位置）
    void push(double t_sec, const Eigen::Vector3d &p)
    {
      buf_.push_back({t_sec, p});
      while (buf_.size() > kWindow)
        buf_.pop_front();
      update();
    }

    // 不同车型的装甲环半径不同：切换目标/车型时设置先验
    void set_prior_radius(double r)
    {
      prior_ = std::clamp(r, kMinRadius, kMaxRadius);
      if (!has_estimate_)
        radius_ = prior_;
    }

    bool spinning() const { return spin_; }
    double omega() const { return omega_; }
    double radius() const { return radius_; }
    double fit_rms() const { return fit_rms_; }
    bool plausible() const { return radius_ >= kMinRadius && radius_ <= kMaxRadius; }
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

      // 1. 半径：到"种子中心"距离的中位数（对离群鲁棒）
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
      const Eigen::Vector2d seed(xs[xs.size() / 2], ys[ys.size() / 2]);

      // 半径用**先验值**而不是从数据估：
      //   弧可见时"到种子中心的距离中位数"会系统性偏小（种子中心本身偏），
      //   而半径一旦估错，相位 φ = atan2(p−C) 的噪声会被放大 r/Δr 倍，
      //   进而把 ω 的最小二乘斜率拉飞（实测 ω 被拉到 8 rad/s，真实 2 rad/s）。
      // 半径：默认用先验；只有弧覆盖 ≥180° 才用自由拟合结果做 EMA 修正。
      // （实测：90° 弧自由拟合的 r 误差 39 mm，180° 时 4 mm，360° 时 1 mm）
      if (!has_estimate_)
        radius_ = prior_;

      if (arc_span_() >= kMinArcForRadius)
      {
        const double r_fit = kasa_radius_();
        if (r_fit > kMinRadius && r_fit < kMaxRadius)
        {
          radius_ = has_estimate_
                        ? (1.0 - kRadiusEmaAlpha) * radius_ + kRadiusEmaAlpha * r_fit
                        : r_fit;
          has_estimate_ = true;
        }
      }

      // 2. 圆心：已知半径 r，Gauss-Newton 拟合 min Σ(|p_i − C| − r)²
      //
      //    为什么不能直接用"坐标中位数/均值"当圆心：
      //    装甲板只在朝向相机的一小段圆弧上可见，样本集中在弧上，
      //    均值/中位数都会被拉向弧的方向（实测偏差 150~190 mm，而半径才 200 mm）。
      //    而"到所有点距离都为 r"这个约束在弧上同样有唯一解，从弧附近起步即可收敛（实测 5~7 mm）。
      Eigen::Vector2d C = seed;
      for (int it = 0; it < 12; ++it)
      {
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d g = Eigen::Vector2d::Zero();
        for (const auto &s : buf_)
        {
          const Eigen::Vector2d d = C - Eigen::Vector2d(s.p.x(), s.p.y());
          const double dist = d.norm();
          if (dist < 1e-6)
            continue;
          const Eigen::Vector2d J = d / dist;   // ∂dist/∂C
          const double res = dist - radius_;
          H += J * J.transpose();
          g += J * res;
        }
        if (std::abs(H.determinant()) < 1e-9)
          break;
        C -= H.ldlt().solve(g);
      }
      center_.x() = C.x();
      center_.y() = C.y();
      center_.z() = buf_.back().p.z();

      // 拟合残差：所有点到圆的距离的 RMS。弧太短/圆心解歪时残差会明显变大，
      // 用它作为"这次能否相信圆周模型"的判据。
      double sq = 0.0;
      for (const auto &s2 : buf_)
      {
        const double d = std::hypot(s2.p.x() - C.x(), s2.p.y() - C.y()) - radius_;
        sq += d * d;
      }
      fit_rms_ = std::sqrt(sq / static_cast<double>(buf_.size()));

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

      spin_ = (std::abs(omega_) > kMinOmega) &&
              (radius_ > kMinRadius) && (radius_ < kMaxRadius) &&
              (fit_rms_ < kMaxFitRms);
    }

    // 窗口内点在圆上的角度覆盖范围（rad）
    double arc_span_() const
    {
      if (buf_.size() < 2)
        return 0.0;
      double lo = 1e9, hi = -1e9;
      for (const auto &s : buf_)
      {
        const double a = std::atan2(s.p.y() - center_.y(), s.p.x() - center_.x());
        lo = std::min(lo, a);
        hi = std::max(hi, a);
      }
      return hi - lo;
    }

    // Kåsa 代数圆拟合：x²+y² + Dx + Ey + F = 0，返回半径
    double kasa_radius_() const
    {
      const size_t n = buf_.size();
      Eigen::MatrixXd A(n, 3);
      Eigen::VectorXd b(n);
      for (size_t i = 0; i < n; ++i)
      {
        A(i, 0) = buf_[i].p.x();
        A(i, 1) = buf_[i].p.y();
        A(i, 2) = 1.0;
        b(i) = -(buf_[i].p.x() * buf_[i].p.x() + buf_[i].p.y() * buf_[i].p.y());
      }
      const Eigen::Vector3d s = A.colPivHouseholderQr().solve(b);
      const Eigen::Vector2d C(-s(0) / 2.0, -s(1) / 2.0);
      return std::sqrt(std::max(0.0, C.squaredNorm() - s(2)));
    }

    std::deque<Sample> buf_;
    Eigen::Vector3d center_{Eigen::Vector3d::Zero()};
    double radius_{0.20};
    double prior_{0.20};
    bool has_estimate_{false};
    double omega_{0.0};
    double phase_{0.0};
    bool spin_{false};
    double fit_rms_{1e9};
  };
} // namespace training
