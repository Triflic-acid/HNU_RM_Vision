#include "lessons.hpp"
#include "io_context.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <csignal>
#include <vector>

namespace
{
  volatile std::sig_atomic_t interrupted = 0;
  void on_signal(int) { interrupted = 1; }
  // 画检测框：四边 + 四角 + 类别&置信度
  void draw_armors(cv::Mat &canvas, const std::vector<training::Armor> &armors)
  {
    for (const auto &a : armors)
    {
      for (int i = 0; i < 4; ++i)
        cv::line(canvas, a.corners[i], a.corners[(i + 1) % 4],
                 cv::Scalar(0, 255, 0), 2);
      for (const auto &p : a.corners)
        cv::circle(canvas, cv::Point(cvRound(p.x), cvRound(p.y)), 3,
                   cv::Scalar(0, 0, 255), -1);
      char text[64];
      std::snprintf(text, sizeof(text), "id=%d %.2f", a.class_id, a.confidence);
      cv::putText(canvas, text,
                  cv::Point(cvRound(a.corners[0].x), cvRound(a.corners[0].y) - 8),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
  }
}

int main(int argc, char **argv)
{
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const auto config = training::load_config(argc > 1 ? argv[1] : "configs/simulation.yaml");
  const auto calibration = training::load_calibration(config);
  training::io::IoContext devices{config};
  training::TrackingState tracking;
  devices.open();

  while (!interrupted)
  {
    const auto input = devices.read();

    training::Command command{};
    std::vector<training::Armor> armors;

    if (input)
    {
      armors = training::lesson03_detection(input->image);
      const auto target = training::lesson04_tracking(
          armors, input->timestamp, input->imu_q, calibration, tracking);
      if (target)
        command = training::lesson05_aiming(*target, input->timestamp, input->bullet_speed);
    }
    if (interrupted)
      break;
    devices.write(command);

    if (input && devices.config.display)
    {
      cv::Mat canvas{input->image.clone()};
      draw_armors(canvas, armors);

      cv::imshow("RoboMaster training", canvas);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q')
        break;
    }
  }

  devices.close();
}
