/*
匀速卡尔曼滤波模型
通过距离自适应量测噪声进行了优化
*/

#pragma once

//
// 状态 x = [px py pz vx vy vz]ᵀ，量测 z = [px py pz]ᵀ
// 过程模型：位置按速度积分，速度保持不变（随机游走）
#include <Eigen/Dense>
namespace training
{
    class KalmanCV
    {
    public:
        using Vec6 = Eigen::Matrix<double, 6, 1>;
        using Mat6 = Eigen::Matrix<double, 6, 6>;
        // q_accel: 过程噪声（加速度标准差，m/s²），目标机动越剧烈取值越大
        // r_pos  : 量测噪声（PnP 位置标准差，m）
        void configure(double q_accel = 8.0, double r_pos = 0.02)
        {
            q_accel_ = q_accel;
            r_pos_ = r_pos;
            R_ = Eigen::Matrix3d::Identity() * r_pos * r_pos;
        }

        // 直接给定量测协方差矩阵（世界系，3x3）。
        //
        // 单目 PnP 的噪声是**各向异性**的：
        //   · 横向（垂直于视线）：σ_lat ≈ Z·σ_px/fx          → 2 m 处约 2 mm
        //   · 深度（沿视线）    ：σ_dep ≈ Z²·σ_px/(fx·W)      → 2 m 处约 80 mm
        // 深度方向噪声比横向大 40 倍，用标量 r_pos 无法表达。
        void set_R(const Eigen::Matrix3d &R)
        {
            R_ = 0.5 * (R + R.transpose());
        }
        bool initialized() const { return initialized_; }
        void reset()
        {
            initialized_ = false;
            x_.setZero();
            P_.setIdentity();
        }
        // 用第一次量测初始化（速度置 0）
        void init(const Eigen::Vector3d &p)
        {
            x_.setZero();
            x_.head<3>() = p;
            P_.setZero();
            P_.block<3, 3>(0, 0) = R_;
            P_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * 25.0; // 速度未知
            initialized_ = true;
        }
        // 预测 dt 秒
        void predict(double dt)
        {
            if (!initialized_ || dt <= 0.0)
                return;
            Mat6 F = Mat6::Identity();
            F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
            
            // 离散白噪声加速度模型
            const double q = q_accel_ * q_accel_;
            Mat6 Q = Mat6::Zero();
            const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
            Q.block<3, 3>(0, 0) = I3 * (q * dt * dt * dt / 3.0);
            Q.block<3, 3>(0, 3) = I3 * (q * dt * dt / 2.0);
            Q.block<3, 3>(3, 0) = I3 * (q * dt * dt / 2.0);
            Q.block<3, 3>(3, 3) = I3 * (q * dt);
            x_ = F * x_;
            P_ = F * P_ * F.transpose() + Q;
            P_ = 0.5 * (P_ + P_.transpose()); // 保对称
        }
        // 用位置量测更新
        void update(const Eigen::Vector3d &p)
        {
            if (!initialized_)
            {
                init(p);
                return;
            }
            Eigen::Matrix<double, 3, 6> H = Eigen::Matrix<double, 3, 6>::Zero();
            H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            const Eigen::Matrix3d &R = R_;
            const Eigen::Matrix3d S = H * P_ * H.transpose() + R;
            const Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * S.inverse();
            x_ = x_ + K * (p - H * x_);
            const Mat6 I = Mat6::Identity();
            P_ = (I - K * H) * P_;
            P_ = 0.5 * (P_ + P_.transpose());
        }
        Eigen::Vector3d position() const { return x_.head<3>(); }
        Eigen::Vector3d velocity() const { return x_.tail<3>(); }
        Eigen::Vector3d predict_future(double dt) const { return x_.head<3>() + x_.tail<3>() * dt; }

    private:
        bool initialized_{false};
        double q_accel_{8.0};
        double r_pos_{0.02};
        Eigen::Matrix3d R_{Eigen::Matrix3d::Identity() * 0.02 * 0.02};
        Vec6 x_{Vec6::Zero()};
        Mat6 P_{Mat6::Identity()};
    };
} // namespace training