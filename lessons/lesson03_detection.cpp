// 第三讲：装甲板检测（YOLO26-Pose + ONNX Runtime）
//
// 由于 ROS2 Humble 的 OpenCV 4.5.4 太旧，无法解析 ONNX，故不使用 cv::dnn:
// （解析 DFL 解码部分的 Add 节点时报 "blob_0.size == blob_1.size"），
// 换 opset 10/11/12/13、开不开 simplify 都不行；OpenCV 5.0 才支持。
// ONNX Runtime 的 C++ API 没有这个限制，而且推理更快。
//
// 支持两种模型，由 DetMode 选择：
//
//   kArmor  装甲板   14 类  kpt_shape [4,2]  输出 [1, 26, 8400]
//   kRune   符文叶片  1 类  kpt_shape [4,2]  输出 [1, 13, 8400]
//
// 两者解码完全一致：nc = C - 4 - 8；差别只有模型文件、类别表、角点顺序，
// 以及符文要多一步 R 标圆心（模板匹配）。

#include "lessons.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

namespace training
{
  namespace
  {
    using Clock = std::chrono::steady_clock;

    constexpr int kInput{640};
    constexpr float kConfThres{0.25F};
    constexpr float kNmsThres{0.45F};
    constexpr bool kDetVerbose{false};

    // ---- 能量机关圆心 R logo（模板匹配）----
    constexpr float kMatchThres{0.55F}; // 响应 0.96~0.99
    constexpr float kRatioLo{2.0F};     // 圆心半径下限 = 2.0 * 叶片框宽
    constexpr float kRatioHi{3.4F};     // 圆心半径上限 = 3.4 * 叶片框宽

    struct ClassInfo
    {
      int id;
      bool large;
    };

    constexpr const char *kArmorModel{"models/best.onnx"};
    constexpr const char *kRuneModel{"models/rune.onnx"};
    constexpr const char *kRuneTemplate{"assets/rune_logo.png"};

    // 装甲板类别表（14 类）
    //   id   : 装甲板编号（1~5），0=哨兵，6=前哨站
    //   large: 大板 225x125 / 小板 135x125
    constexpr ClassInfo kArmorTable[14] = {
        {1, true},  // B1 英雄（大板）
        {2, false}, // B2 工程
        {3, false}, // B3 步兵
        {4, false}, // B4 步兵
        {5, false}, // B5 步兵
        {6, false}, // BO 前哨站
        {0, false}, // BS 哨兵
        {1, true},  // R1 英雄（大板）
        {2, false}, // R2 工程
        {3, false}, // R3 步兵
        {4, false}, // R4 步兵
        {5, false}, // R5 步兵
        {6, false}, // RO 前哨站
        {0, false}, // RS 哨兵
    };

    const char *model_path(DetMode mode)
    {
      const char *env = std::getenv(mode == DetMode::kArmor ? "ARMOR_MODEL" : "RUNE_MODEL");
      if (env)
        return env;
      return mode == DetMode::kArmor ? kArmorModel : kRuneModel;
    }

    cv::Mat letterbox(const cv::Mat &src, int size, float &scale, int &pad_x, int &pad_y)
    {
      const int width{src.cols}, height{src.rows};
      scale = std::min(size / static_cast<float>(width), size / static_cast<float>(height));

      const int new_width{static_cast<int>(std::round(width * scale))};
      const int new_height{static_cast<int>(std::round(height * scale))};
      pad_x = (size - new_width) / 2;
      pad_y = (size - new_height) / 2;

      cv::Mat out(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
      cv::Mat resized;
      cv::resize(src, resized, cv::Size(new_width, new_height));
      resized.copyTo(out(cv::Rect(pad_x, pad_y, new_width, new_height)));
      return out;
    }

    // 两种模式
    // Sessions
    Ort::Session *session_ptr(DetMode mode)
    {
      static std::unique_ptr<Ort::Env> env;
      static std::unique_ptr<Ort::Session> sessions[2];
      static bool failed[2]{false, false};

      const int k = (mode == DetMode::kArmor) ? 0 : 1;
      if (!sessions[k] && !failed[k])
      {
        try
        {
          if (!env)
            env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rm_det");
          Ort::SessionOptions opts;
          opts.SetIntraOpNumThreads(4);
          opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
          sessions[k] = std::make_unique<Ort::Session>(*env, model_path(mode), opts);
        }
        catch (const std::exception &e)
        {
          std::printf("[det] 模型加载失败（%s）：%s\n[det] 请检查 %s（应在仓库根目录运行）\n",
                      mode == DetMode::kArmor ? "armor" : "rune", e.what(), model_path(mode));
          std::fflush(stdout);
          failed[k] = true;
        }
      }
      return sessions[k].get();
    }

    // 颜色无关的强度图：V = max(R,G,B)
    //
    // 红方/蓝方的 R 标色相不同，但都是"彩色高饱和发光体"：
    //   灰度 = 0.299R + 0.587G + 0.114B  → 红 logo 亮、蓝 logo 暗，模板对不上
    //   V    = max(R,G,B)                → 两种颜色都接近饱和，形状一致
    // 实测（红/蓝各 60 张）：灰度 蓝方命中 7/60；V 通道 红蓝各 60/60。
    cv::Mat value_channel(const cv::Mat &bgr)
    {
      cv::Mat hsv, v;
      cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
      cv::extractChannel(hsv, v, 2); // H=0, S=1, V=2
      return v;
    }

    // ---- 能量机关平均模板 ----
    const std::vector<cv::Mat> &rune_templates()
    {
      static std::vector<cv::Mat> tpls;
      static bool tried = false;
      if (tpls.empty() && !tried)
      {
        tried = true;
        std::vector<cv::String> files;
        const char *env = std::getenv("RUNE_TEMPLATE");
        if (env)
          files.emplace_back(env);
        else
          cv::glob("assets/rune_logo*.png", files, false);

        for (const auto &f : files)
        {
          const cv::Mat bgr = cv::imread(f, cv::IMREAD_COLOR);
          if (bgr.empty())
            continue;
          tpls.push_back(value_channel(bgr)); // V 通道：红/蓝通吃
        }
        if (tpls.empty())
        {
          std::printf("[det] 符文模板缺失：assets/rune_logo*.png\n");
          std::fflush(stdout);
        }
        else
        {
          std::printf("[det] 符文模板 %zu 张\n", tpls.size());
          std::fflush(stdout);
        }
      }
      return tpls;
    }

    // 响应图的抛物线亚像素峰值
    cv::Point2f subpixel_peak(const cv::Mat &resp, cv::Point peak)
    {
      auto at = [&](int x, int y)
      { return resp.at<float>(y, x); };
      auto para = [](float a, float b, float c)
      {
        const float d = a - 2.F * b + c;
        return std::abs(d) < 1e-6F ? 0.F : 0.5F * (a - c) / d;
      };
      float dx{0.F}, dy{0.F};
      if (peak.x > 0 && peak.x < resp.cols - 1)
        dx = para(at(peak.x - 1, peak.y), at(peak.x, peak.y), at(peak.x + 1, peak.y));
      if (peak.y > 0 && peak.y < resp.rows - 1)
        dy = para(at(peak.x, peak.y - 1), at(peak.x, peak.y), at(peak.x, peak.y + 1));
      return cv::Point2f(peak.x + std::clamp(dx, -0.5F, 0.5F),
                         peak.y + std::clamp(dy, -0.5F, 0.5F));
    }

    // 在叶片周围的环形带里找 R 标圆心
    std::optional<cv::Point2f> find_rune_center(const cv::Mat &value,
                                                const cv::Rect2f &blade_box,
                                                const cv::Point2f &blade_center,
                                                const std::vector<cv::Mat> &tpls,
                                                float &score)
    {
      score = 0.F;
      const float w = std::max(blade_box.width, 4.F);
      const float r_lo = kRatioLo * w;
      const float r_hi = kRatioHi * w;

      const int half = static_cast<int>(std::ceil(r_hi)) + 4;
      cv::Rect roi(static_cast<int>(blade_center.x) - half,
                   static_cast<int>(blade_center.y) - half,
                   half * 2, half * 2);
      roi &= cv::Rect(0, 0, value.cols, value.rows);
      if (roi.width < 16 || roi.height < 16)
      {
        return std::nullopt;
      }

      const cv::Mat patch = value(roi);
      std::optional<cv::Point2f> best;
      float best_score{-1.F};

      for (const auto &tpl : tpls)
      {
        const int native = std::max(tpl.cols, tpl.rows);
        static constexpr float kScales[] = {0.30F, 0.34F, 0.38F, 0.42F, 0.46F, 0.50F};
        for (const float k : kScales)
        {
          const float sc = k * w / static_cast<float>(native);
          if (sc < 0.04F || sc > 4.F)
            continue;
          if (static_cast<int>(native * sc) < 12)
            continue;
          cv::Mat t;
          cv::resize(tpl, t, cv::Size(), sc, sc, cv::INTER_LINEAR);
          if (t.cols >= patch.cols || t.rows >= patch.rows)
            continue;

          cv::Mat resp;
          cv::matchTemplate(patch, t, resp, cv::TM_CCOEFF_NORMED);

          double mx{0.0};
          cv::Point ml;
          cv::minMaxLoc(resp, nullptr, &mx, nullptr, &ml);
          if (static_cast<float>(mx) <= best_score)
            continue;

          const cv::Point2f peak = subpixel_peak(resp, ml) +
                                   cv::Point2f(t.cols / 2.F, t.rows / 2.F) +
                                   cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));

          // 半径门限：画面里其他 R logo 落在环外
          const float r = cv::norm(peak - blade_center);
          if (r < r_lo || r > r_hi)
            continue;

          best_score = static_cast<float>(mx);
          best = peak;
        }
      } // for each template

      if (best && best_score >= kMatchThres)
      {
        score = best_score;
        return best;
      }
      return std::nullopt;
    }

  } // namespace

  std::vector<Detection> lesson03_detection(const cv::Mat &image, DetMode mode)
  {
    std::vector<Detection> detections;
    if (image.empty() || image.type() != CV_8UC3)
      return detections;

    const auto t0 = Clock::now();

    // ---- 1. 预处理 ----
    float scale{1.F};
    int pad_x{0}, pad_y{0};
    const cv::Mat padded = letterbox(image, kInput, scale, pad_x, pad_y);

    // static：复用同一块 4.9 MB 缓冲，避免每帧 malloc/free（单线程安全）
    static std::vector<float> input(3 * kInput * kInput);
    const int plane = kInput * kInput;
    for (int y = 0; y < kInput; ++y)
    {
      const auto *row = padded.ptr<cv::Vec3b>(y);
      for (int x = 0; x < kInput; ++x)
      {
        const auto &bgr = row[x];
        const int idx = y * kInput + x;
        input[0 * plane + idx] = bgr[2] / 255.F; // R
        input[1 * plane + idx] = bgr[1] / 255.F; // G
        input[2 * plane + idx] = bgr[0] / 255.F; // B
      }
    }

    // ---- 2. 推理 ----
    auto *session = session_ptr(mode);
    if (!session)
      return detections;

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> in_shape{1, 3, kInput, kInput};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), in_shape.data(), in_shape.size());

    const char *in_names[] = {"images"};
    const char *out_names[] = {"output0"};

    auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    const auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, C, N]
    if (shape.size() != 3 || shape[0] != 1)
      return detections;

    const int C{static_cast<int>(shape[1])};
    const int N{static_cast<int>(shape[2])};
    const int nc{C - 4 - 8}; // 4 框 + 4 关键点×2
    const int nc_max{(mode == DetMode::kArmor) ? 14 : 1};
    if (nc <= 0 || nc > nc_max)
      return detections;

    const float *p = outputs[0].GetTensorData<float>();

    // ---- 3. 解码 ----
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::array<cv::Point2f, 4>> kpts;

    for (int i = 0; i < N; ++i)
    {
      int best_class{0};
      float best_score{0.F};
      for (int c = 0; c < nc; ++c)
      {
        const float s = p[(4 + c) * N + i];
        if (s > best_score)
        {
          best_score = s;
          best_class = c;
        }
      }
      if (best_score < kConfThres)
        continue;

      const float cx = p[0 * N + i];
      const float cy = p[1 * N + i];
      const float w = p[2 * N + i];
      const float h = p[3 * N + i];

      std::array<cv::Point2f, 4> kp{};
      for (int k = 0; k < 4; ++k)
      {
        const float x = p[(4 + nc + k * 2 + 0) * N + i];
        const float y = p[(4 + nc + k * 2 + 1) * N + i];
        kp[k] = cv::Point2f((x - pad_x) / scale, (y - pad_y) / scale);
      }

      const float x0 = (cx - w / 2.F - pad_x) / scale;
      const float y0 = (cy - h / 2.F - pad_y) / scale;
      const float bw = w / scale;
      const float bh = h / scale;
      if (bw < 2.F || bh < 2.F)
        continue;

      boxes.emplace_back(static_cast<int>(x0), static_cast<int>(y0),
                         static_cast<int>(bw), static_cast<int>(bh));
      scores.push_back(best_score);
      class_ids.push_back(best_class);
      kpts.push_back(kp);
    }

    if (boxes.empty())
    {
      if (kDetVerbose)
      {
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("[det] %.1f ms, 0 targets\n", ms);
      }
      return detections;
    }

    // ---- 4. NMS ----
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConfThres, kNmsThres, keep);

    // ---- 5. 组装 ----
    detections.reserve(keep.size());
    for (const int idx : keep)
    {
      Detection d;
      d.box = cv::Rect2f(boxes[idx]);
      d.confidence = scores[idx];

      if (mode == DetMode::kArmor)
      {
        // 模型点序 左上、左下、右下、右上 → Armor 要求 左上、右上、右下、左下
        d.corners = {kpts[idx][0], kpts[idx][3], kpts[idx][2], kpts[idx][1]};
        d.class_id = kArmorTable[class_ids[idx]].id;
        d.large_armor = kArmorTable[class_ids[idx]].large;
      }
      else
      {
        // 符文：点序与数据集一致，直接用 左上、左下、右下、右上
        d.corners = kpts[idx];
        d.class_id = 0;
        d.large_armor = false;
      }
      detections.push_back(d);
    }

    // ---- 6. 能量机关圆心（仅在找到叶片的情况下搜索）----
    if (mode == DetMode::kRune && !detections.empty())
    {
      const auto &tpls = rune_templates();
      if (!tpls.empty())
      {
        cv::Mat value = value_channel(image);

        std::optional<cv::Point2f> center;
        float best{-1.F};
        for (const auto &d : detections)
        {
          const cv::Point2f c = (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
          float score{0.F};
          const auto found = find_rune_center(value, d.box, c, tpls, score);
          if (found && score > best) // 多叶片时取响应最高的环带结果
          {
            center = found;
            best = score;
          }
        }
        // 圆心全帧共享，写到每个叶片上
        for (auto &d : detections)
          d.center = center;
      }
    }

    if (kDetVerbose)
    {
      const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      std::printf("[det] %s %.1f ms, %zu targets\n",
                  mode == DetMode::kArmor ? "armor" : "rune", ms, detections.size());
      std::fflush(stdout);
    }

    return detections;
  }

} // namespace training
