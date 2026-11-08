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
//   kRune   能量机关叶片  1 类  kpt_shape [4,2]  输出 [1, 13, 8400]
//
// 两者解码完全一致：nc = C - 4 - 8；差别只有模型文件、类别表、角点顺序，
// 能量机关多一步 R 标圆心（模板匹配）。

#include "lessons.hpp"

#include "debug_flags.hpp"

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
#include <string>
#include <filesystem>

namespace training
{
  namespace
  {
    using Clock = std::chrono::steady_clock;

    constexpr int kInput{640};
    constexpr float kConfThres{0.25F};
    constexpr float kNmsThres{0.45F};

    // ==== 能量机关圆心 R logo（模板匹配）====
    constexpr float kMatchThres{0.70F}; // 响应 0.96~0.99
    constexpr float kRatioLo{0.8F};     // 圆心半径下限 = 0.8 * 叶片框宽
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

    cv::Mat value_channel(const cv::Mat &bgr)
    {
      cv::Mat hsv, v;
      cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
      cv::extractChannel(hsv, v, 2); // H=0, S=1, V=2
      return v;
    }

    // ==== 能量机关平均模板 ====
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
          std::printf("[det] 能量机关模板缺失：assets/rune_logo*.png\n");
          std::fflush(stdout);
        }
        else if (training::dbg::det)
        {
          std::printf("[det] 能量机关模板 %zu 张\n", tpls.size());
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
        if (training::dbg::rune)
        {
          const float rr = cv::norm(*best - blade_center) / w;
          std::printf("[rune] w=%.0f r/w=%.2f score=%.3f center=(%.0f,%.0f)\n",
                      w, rr, best_score, best->x, best->y);
          std::fflush(stdout);
        }
        
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

    // ==== 1. 预处理 ====
    float scale{1.F};
    int pad_x{0}, pad_y{0};
    const cv::Mat padded = letterbox(image, kInput, scale, pad_x, pad_y);

    static std::vector<float> input(3 * kInput * kInput); // 复用缓冲区
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

    // ==== 2. 推理 ====
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

    // ==== 3. 解码 ====
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
      if (training::dbg::det)
      {
        const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        std::printf("[det] %.1f ms, 0 targets\n", ms);
      }
      return detections;
    }

    // ==== 4. NMS ====
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, kConfThres, kNmsThres, keep);

    // ==== 5. 组装 ====
    detections.reserve(keep.size());
    for (const int idx : keep)
    {
      Detection d;
      d.box = cv::Rect2f(boxes[idx]);
      d.confidence = scores[idx];

      if (mode == DetMode::kArmor)
      {
        // 模型点序 左上、左下、右下、右上 -> Armor 要求 左上、右上、右下、左下
        d.corners = {kpts[idx][0], kpts[idx][3], kpts[idx][2], kpts[idx][1]};
        d.class_id = kArmorTable[class_ids[idx]].id;
        d.large_armor = kArmorTable[class_ids[idx]].large;
      }
      else
      {
        // 能量机关：点序与数据集一致，直接用 左上、左下、右下、右上
        d.corners = kpts[idx];
        d.class_id = 0;
        d.large_armor = false;
      }
      detections.push_back(d);
    }

    // ==== 6. 能量机关圆心（仅在找到叶片的情况下搜索）====
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

    if (training::dbg::det)
    {
      const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      std::printf("[det] %s %.1f ms, %zu targets\n",
                  mode == DetMode::kArmor ? "armor" : "rune", ms, detections.size());
      std::fflush(stdout);
    }

    return detections;
  }

  // 模块 A：ROI 放大推理、

  // 以 center 为中心裁出一个正方形（边长 = 图像宽 / zoom），resize 到 640 后复用 lesson03_detection（内部 letterbox 对 640 是恒等），再把结果按 sx/sy 换算回原图
  // ROI 为方形，x/y scale 相同，不会引入 letterbox 的不等比
  std::vector<Detection> lesson03_detection_roi(const cv::Mat &image,
                                                cv::Point2f center,
                                                float zoom,
                                                DetMode mode)
  {
    std::vector<Detection> out;
    if (image.empty() || image.type() != CV_8UC3 || zoom <= 0.1F)
      return out;
    int side = static_cast<int>(std::round(image.cols / zoom));
    side = std::min(side, std::min(image.cols, image.rows));
    if (side < 32)
      return out;
    int x = static_cast<int>(std::round(center.x - side / 2.0));
    int y = static_cast<int>(std::round(center.y - side / 2.0));
    x = std::clamp(x, 0, image.cols - side);
    y = std::clamp(y, 0, image.rows - side);
    const cv::Mat crop = image(cv::Rect(x, y, side, side));
    cv::Mat up;
    cv::resize(crop, up, cv::Size(kInput, kInput), 0, 0, cv::INTER_LINEAR);
    out = lesson03_detection(up, mode);
    const float sx = static_cast<float>(side) / kInput;
    const float sy = sx;
    for (auto &d : out)
    {
      for (auto &c : d.corners)
      {
        c.x = c.x * sx + x;
        c.y = c.y * sy + y;
      }
      d.box.x = d.box.x * sx + x;
      d.box.y = d.box.y * sy + y;
      d.box.width *= sx;
      d.box.height *= sy;
      if (d.center)
      {
        d.center->x = d.center->x * sx + x;
        d.center->y = d.center->y * sy + y;
      }
    }
    return out;
  }
  // 模块 B：亚像素角点精修

  // 直对原图做 cornerSubPix
  // 点锁定在图像内（3 px 边距），否则 OpenCV 会抛异常
  // 位移超过 max_shift 视为精修失败 -> 保留网络原值
  // 精修后必须仍是凸四边形，否则丢弃
  namespace
  {
    bool quad_is_convex(const std::array<cv::Point2f, 4> &q)
    {
      float s[4];
      for (int i = 0; i < 4; ++i)
      {
        const auto &a = q[i], &b = q[(i + 1) % 4], &c = q[(i + 2) % 4];
        s[i] = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      }
      const bool pos = std::all_of(s, s + 4, [](float v)
                                   { return v > 0; });
      const bool neg = std::all_of(s, s + 4, [](float v)
                                   { return v < 0; });
      return pos || neg;
    }
  } // namespace
  void lesson03_refine_subpix(const cv::Mat &image,
                              std::vector<Detection> &detections,
                              int win,
                              float max_shift)
  {
    if (image.empty() || detections.empty())
      return;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    const int w = std::max(3, win | 1); // 强制奇数
    for (auto &d : detections)
    {
      std::vector<cv::Point2f> pts(d.corners.begin(), d.corners.end());
      for (auto &p : pts)
      {
        p.x = std::clamp(p.x, 3.F, static_cast<float>(image.cols - 4));
        p.y = std::clamp(p.y, 3.F, static_cast<float>(image.rows - 4));
      }
      std::vector<cv::Point2f> refined = pts;
      try
      {
        cv::cornerSubPix(gray, refined, cv::Size(w, w), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER,
                                          20, 0.03));
      }
      catch (const cv::Exception &)
      {
        continue; // 精修失败，保留原值
      }
      bool ok = true;
      for (int i = 0; i < 4; ++i)
        if (cv::norm(refined[i] - d.corners[i]) > max_shift)
          ok = false;
      const std::array<cv::Point2f, 4> cand{refined[0], refined[1], refined[2], refined[3]};
      if (!ok || !quad_is_convex(cand))
        continue;
      d.corners = cand;
    }
  }

  // 模块 C：四边拟合直线 -> 求交（用于装甲板/能量机关的精修）

  // 沿法向搜索梯度极值 -> 亚像素点
  // 每条边取十几个点做最小二乘 -> 单点噪声被摊薄（1/√n）
  // 相邻两条边求交 -> 交点天然亚像素
  namespace
  {
    // 双线性采样
    inline float sample_bilinear(const cv::Mat &g, float x, float y)
    {
      if (x < 0.F || y < 0.F || x >= g.cols - 1.F || y >= g.rows - 1.F)
        return 0.F;
      const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
      const float fx = x - x0, fy = y - y0;
      const float v00 = g.at<uchar>(y0, x0), v01 = g.at<uchar>(y0, x0 + 1);
      const float v10 = g.at<uchar>(y0 + 1, x0), v11 = g.at<uchar>(y0 + 1, x0 + 1);
      return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
    }
    // 沿法向一维搜索梯度极值，抛物线插值到亚像素
    bool edge_point_1d(const cv::Mat &g, cv::Point2f seed, cv::Point2f nrm,
                       int radius, cv::Point2f &out)
    {
      const int N = 2 * radius + 1;
      std::vector<float> prof(N);
      for (int i = 0; i < N; ++i)
      {
        const float t = static_cast<float>(i - radius);
        prof[i] = sample_bilinear(g, seed.x + nrm.x * t, seed.y + nrm.y * t);
      }
      int best = -1;
      float best_v = 0.F;
      for (int i = 1; i < N - 1; ++i)
      {
        const float grad = std::abs(prof[i + 1] - prof[i - 1]) * 0.5F;
        if (grad > best_v)
        {
          best_v = grad;
          best = i;
        }
      }
      if (best <= 0 || best >= N - 1 || best_v < 3.F)
        return false;
      const float y0 = prof[best - 1], y1 = prof[best], y2 = prof[best + 1];
      const float den = y0 - 2.F * y1 + y2;
      const float delta = (std::abs(den) < 1e-6F) ? 0.F : 0.5F * (y0 - y2) / den;
      const float t = static_cast<float>(best - radius) + std::clamp(delta, -0.5F, 0.5F);
      out = seed + nrm * t;
      return true;
    }
    // (vx,vy,x0,y0) -> ax+by+c=0
    cv::Vec3f line_abc(const cv::Vec4f &l)
    {
      const float vx = l[0], vy = l[1], x0 = l[2], y0 = l[3];
      return cv::Vec3f(-vy, vx, -(vx * y0 - vy * x0));
    }
    bool intersect_abc(const cv::Vec3f &l1, const cv::Vec3f &l2, cv::Point2f &p)
    {
      const float d = l1[0] * l2[1] - l1[1] * l2[0];
      if (std::abs(d) < 1e-6F)
        return false;
      p.x = (l1[1] * l2[2] - l1[2] * l2[1]) / d;
      p.y = (l1[2] * l2[0] - l1[0] * l2[2]) / d;
      return true;
    }
  }

  void lesson03_refine_quad_lines(const cv::Mat &image,
                                  std::vector<Detection> &detections,
                                  int pts_per_edge,
                                  float max_shift)
  {
    if (image.empty() || detections.empty())
      return;

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    const int per_edge = std::max(6, pts_per_edge);

    for (auto &d : detections)
    {
      float min_side = 1e9F;
      for (int i = 0; i < 4; ++i)
        min_side = std::min(min_side, static_cast<float>(cv::norm(d.corners[(i + 1) % 4] - d.corners[i])));
      const int radius = std::max(2, static_cast<int>(std::round(min_side * 0.04F)));
      if (min_side < 8.F)
        continue;

      cv::Vec3f lines[4];
      bool ok = true;
      for (int e = 0; e < 4; ++e)
      {
        const cv::Point2f a = d.corners[e];
        const cv::Point2f b = d.corners[(e + 1) % 4];
        const cv::Point2f dir = b - a;
        const float len = static_cast<float>(cv::norm(dir));
        if (len < 6.F)
        {
          ok = false;
          break;
        }
        const cv::Point2f u = dir / len;
        const cv::Point2f nrm(-u.y, u.x);

        std::vector<cv::Point2f> pts;
        for (int i = 1; i <= per_edge; ++i)
        {
          const float t = 0.15F + 0.70F * static_cast<float>(i) / (per_edge + 1);
          cv::Point2f sp;
          if (edge_point_1d(gray, a + u * (len * t), nrm, radius, sp))
            pts.push_back(sp);
        }
        if (pts.size() < 5)
        {
          ok = false;
          break;
        }
        cv::Vec4f ln;
        cv::fitLine(pts, ln, cv::DIST_L2, 0, 0.01, 0.01);
        lines[e] = line_abc(ln);
      }
      if (!ok)
        continue;

      std::array<cv::Point2f, 4> cand{};
      bool ok2 = true;
      for (int i = 0; i < 4; ++i)
        if (!intersect_abc(lines[(i + 3) % 4], lines[i], cand[i]))
          ok2 = false;
      if (!ok2)
        continue;

      float shift = 0.F;
      for (int i = 0; i < 4; ++i)
        shift = std::max(shift, static_cast<float>(cv::norm(cand[i] - d.corners[i])));
      if (shift > max_shift || !quad_is_convex(cand))
        continue;

      d.corners = cand;
    }
  }

  std::optional<cv::Point2f> lesson03_phase_shift(const cv::Mat &prev,
                                                  const cv::Mat &cur,
                                                  cv::Rect roi,
                                                  double min_response)
  {
    if (prev.empty() || cur.empty())
      return std::nullopt;

    roi &= cv::Rect(0, 0, std::min(prev.cols, cur.cols), std::min(prev.rows, cur.rows));
    if (roi.width < 16 || roi.height < 16)
      return std::nullopt;

    cv::Mat g1, g2;
    if (prev.channels() == 3)
      cv::cvtColor(prev, g1, cv::COLOR_BGR2GRAY);
    else
      g1 = prev;
    if (cur.channels() == 3)
      cv::cvtColor(cur, g2, cv::COLOR_BGR2GRAY);
    else
      g2 = cur;

    cv::Mat a = g1(roi).clone(), b = g2(roi).clone();
    a.convertTo(a, CV_32F);
    b.convertTo(b, CV_32F);

    cv::Mat hann;
    cv::createHanningWindow(hann, a.size(), CV_32F);

    double response = 0.0;
    const cv::Point2d shift = cv::phaseCorrelate(a, b, hann, &response);
    if (response < min_response)
      return std::nullopt;

    return cv::Point2f(static_cast<float>(shift.x), static_cast<float>(shift.y));
  }

} // namespace training
