#include "config.hpp"
#include "types.hpp"
#include <memory>
#include <optional>

#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace training::io
{
  struct IoContext
  {
    IoContext() = default;
    explicit IoContext(const Config &c) : config(c) {}

    Config config;
    void open();
    std::optional<SensorData> read();
    void write(const Command &command);
    void close();

  private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr recv_sub_;
    rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr send_pub_;

    sensor_msgs::msg::Image::SharedPtr latest_image_;
    hnurm_interfaces::msg::VisionRecvData::SharedPtr latest_recv_;

    rclcpp::Time base_ros_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_frame_stamp_{0, 0, RCL_ROS_TIME};

    TimePoint base_steady_;

    bool have_frame_{false};

    double bullet_speed_{25.0};
  };
} // namespace training::io
