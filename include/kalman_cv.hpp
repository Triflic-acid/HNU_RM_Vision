/*
匀速卡尔曼滤波模型
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
            P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * r_pos_ * r_pos_;
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
            Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * r_pos_ * r_pos_;
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
        Vec6 x_{Vec6::Zero()};
        Mat6 P_{Mat6::Identity()};
    };
} // namespace training