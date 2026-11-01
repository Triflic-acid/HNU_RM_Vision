#include "io_context.hpp"

#include <cstdio>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace training::io
{

  namespace
  {

    constexpr double kDeg2Rad = kPi / 180.0; // 角度制->弧度制
    constexpr double kRad2Deg = 180.0 / kPi; // 弧度制->角度制
    // 与 simulation/simulator/config.toml 的 gimbal_pitch_limit 一致
    constexpr double kMaxPitch = 45.0 * kDeg2Rad;
    // 反馈与图像时间差超过设定值默认认反馈过期
    constexpr int64_t kFeedbackStaleNs = 100'000'000; // 100 ms
    // read() 一帧等待的最长时间，超时返回 nullopt（使主循环发出停止指令）
    constexpr auto kFrameTimeout = std::chrono::milliseconds(50);

    // 姿态角（deg，pitch 抬头为正）-> 四元数（x前 y左 z上）
    Eigen::Quaterniond to_quaternion(double yaw_deg, double pitch_deg, double roll_deg)
    {
      const Eigen::AngleAxisd yaw(yaw_deg * kDeg2Rad, Eigen::Vector3d::UnitZ());
      const Eigen::AngleAxisd pitch(-pitch_deg * kDeg2Rad, Eigen::Vector3d::UnitY());
      const Eigen::AngleAxisd roll(roll_deg * kDeg2Rad, Eigen::Vector3d::UnitX());
      Eigen::Quaterniond q = yaw * pitch * roll;
      q.normalize();
      return q;
    }

    // sensor_msgs/Image -> 私有的 BGR CV_8UC3
    bool to_bgr(const sensor_msgs::msg::Image &msg, cv::Mat &out)
    {
      if (msg.width == 0 || msg.height == 0 || msg.data.empty())
        return false;
      const int width = static_cast<int>(msg.width);
      const int height = static_cast<int>(msg.height);
      auto *data = const_cast<unsigned char *>(msg.data.data()); // 只读视图，随后立刻clone
      if (msg.encoding == "bgr8")
      {
        out = cv::Mat(height, width, CV_8UC3, data, msg.step).clone();
      }
      else if (msg.encoding == "rgb8")
      {
        const cv::Mat view(height, width, CV_8UC3, data, msg.step);
        cv::cvtColor(view, out, cv::COLOR_RGB2BGR);
      }
      else if (msg.encoding == "mono8")
      {
        const cv::Mat view(height, width, CV_8UC1, data, msg.step);
        cv::cvtColor(view, out, cv::COLOR_GRAY2BGR);
      }
      else
      {
        return false;
      }
      return !out.empty();
    }
    // read() 返回 nullopt 时的限流日志
    void log_nullopt(const char *why)
    {
      static auto next = Clock::now();
      const auto now = Clock::now();
      if (now < next)
        return;
      next = now + std::chrono::seconds(1);
      std::printf("[read] -> nullopt (%s)\n", why);
      std::fflush(stdout);
    }

  } // namespace

  void IoContext::open()
  {
    if (!rclcpp::ok())
    {
      rclcpp::init(0, nullptr, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
    }
    node_ = std::make_shared<rclcpp::Node>("training_io");

    auto qos = rclcpp::SensorDataQoS();
    qos.keep_last(5);

    image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>("/image_raw", qos, [this](sensor_msgs::msg::Image::SharedPtr msg)
                                                                     { latest_image_ = std::move(msg); });

    recv_sub_ = node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>("/vision_receive_data", qos, [this](hnurm_interfaces::msg::VisionRecvData::SharedPtr msg)
                                                                                  { latest_recv_ = std::move(msg); });

    send_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>("/vision_send_data", qos);

    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);

    base_ros_ = node_->now();
    base_steady_ = Clock::now();
  }

  std::optional<SensorData> IoContext::read()
  {
    if (!executor_)
      return std::nullopt;

    const auto ddl = Clock::now() + kFrameTimeout;

    while (Clock::now() < ddl)
    {
      executor_->spin_once(std::chrono::milliseconds(2));
      if (!latest_image_)
        continue;
      const rclcpp::Time stamp(latest_image_->header.stamp);
      if (!have_frame_ || stamp != last_frame_stamp_)
        break;
    }

    if (!latest_image_){
      log_nullopt("No image!");
      return std::nullopt;}
    const rclcpp::Time stamp(latest_image_->header.stamp);
    if (have_frame_ && stamp == last_frame_stamp_) {
      log_nullopt("Duplicate!");
      return std::nullopt; // 同一帧不重复
      }
    if (!latest_recv_){
      log_nullopt("No feedback!");
      return std::nullopt; // 没有姿态
    }
    // 检查图像与反馈的同步性
    const rclcpp::Time feedback_stamp(latest_recv_->header.stamp);
    if (std::llabs((stamp - feedback_stamp).nanoseconds()) > kFeedbackStaleNs)
    {
      log_nullopt("TLE");
      return std::nullopt;
    }

    SensorData data;
    if (!to_bgr(*latest_image_, data.image)){
      log_nullopt("Bad bgr!");
      return std::nullopt;
    }

    data.timestamp = base_steady_ +
                     std::chrono::nanoseconds((stamp - base_ros_).nanoseconds());
    data.imu_q = to_quaternion(latest_recv_->yaw, latest_recv_->pitch, latest_recv_->roll);

    if (latest_recv_->bullet_speed.data > 0)
    {
      bullet_speed_ = static_cast<double>(latest_recv_->bullet_speed.data);
    }
    data.bullet_speed = bullet_speed_;

    have_frame_ = true;
    last_frame_stamp_ = stamp;

    // 演示用

    ++frame_count_;
    const auto now = Clock::now();
    if (now >= next_log_)
    {
      next_log_ = now + std::chrono::milliseconds(500); // 2 行/秒，视频里更"活"
      std::printf("[IO] yaw=%7.2f  pitch=%6.2f  bullet=%5.1f m/s  frames=%d\n",
                  latest_recv_->yaw, latest_recv_->pitch, data.bullet_speed, frame_count_);
      std::fflush(stdout);
    }

    return data;
  }

  void IoContext::write(const Command &command)
  {

    if (!send_pub_ || !node_ || !rclcpp::ok())
      return;
    hnurm_interfaces::msg::VisionSendData msg;

    msg.header.stamp = node_->now();     // 打时间戳(计算结束的时间)
    msg.header.frame_id = "gimbal_link"; // 云台系

    const bool feedback_ok{latest_recv_ && latest_recv_->work_mode.data == hnurm_interfaces::msg::WorkMode::AUTO_AIM};
    const bool angle_ok{std::isfinite(command.yaw) && std::isfinite(command.pitch) && std::abs(command.pitch) <= kMaxPitch};

    const bool active{command.control && feedback_ok && angle_ok};

    if (!active)
    {
      msg.target_state.data = hnurm_interfaces::msg::TargetState::LOST_TARGET;
      msg.yaw = 0.0F;
      msg.pitch = 0.0F;
      msg.target_distance = 0.0F;
    }

    else
    {
      msg.target_state.data = (command.shoot && config.allow_fire)
                                  ? hnurm_interfaces::msg::TargetState::FIRE
                                  : hnurm_interfaces::msg::TargetState::CONVERGING;

      // 转为角度制
      msg.yaw = static_cast<float>(command.yaw * kRad2Deg);
      msg.pitch = static_cast<float>(command.pitch * kRad2Deg);

      msg.target_distance = 0.0F;
    }

    send_pub_->publish(msg);

    // 演示用
    static uint8_t last_state = 255;
    if (msg.target_state.data != last_state)
    {
      last_state = msg.target_state.data;
      std::printf("[Write] target_state -> %u (yaw=%.2f pitch=%.2f)\n",
                  msg.target_state.data, msg.yaw, msg.pitch);
      std::fflush(stdout);
    }
    return;
  }

  void IoContext::close()
  {
    if (node_ && rclcpp::ok())
    {
      Command stop{};
      write(stop);

      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    if (executor_)
    {
      executor_->cancel();
      if (node_)
        executor_->remove_node(node_);
      executor_.reset();
    }

    image_sub_.reset();
    recv_sub_.reset();
    send_pub_.reset();

    node_.reset();

    if (rclcpp::ok())
      rclcpp::shutdown();

    return;
  }
} // namespace training::io
