#include "ppocrv6_native/detection/db_postprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#if __has_include(<polyclipping/clipper.hpp>)
#include <polyclipping/clipper.hpp>
#else
#include <clipper.hpp>
#endif
#endif

namespace ppocrv6_native::detection {
namespace {

void append_float(std::ostringstream &out, double value) {
  if (std::isfinite(value)) {
    out << std::setprecision(9) << value;
  } else {
    out << "0.0";
  }
}

float mean_y(const Box &box) {
  float sum = 0.0f;
  for (const auto &point : box.points) {
    sum += point.y;
  }
  return sum / 4.0f;
}

float mean_x(const Box &box) {
  float sum = 0.0f;
  for (const auto &point : box.points) {
    sum += point.x;
  }
  return sum / 4.0f;
}

std::pair<int, float> reading_order_key(const DetectionBox &box) {
  const float height = std::max(distance(box.box[0], box.box[3]), 1.0f);
  const int row = static_cast<int>(
      std::round(mean_y(box.box) / std::max(height * 0.7f, 1.0f)));
  return {row, mean_x(box.box)};
}

float clamp_round(float value, float low, float high) {
  return std::clamp(std::round(value), low, high);
}

#if !defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
struct Component {
  int xmin = 0;
  int ymin = 0;
  int xmax = 0;
  int ymax = 0;
  int pixels = 0;
  double sum = 0.0;
};

DetectionBox component_to_box(const Component &component, int pred_height,
                              int pred_width, int dest_height, int dest_width,
                              const DetectionPostprocessConfig &config) {
  const float width =
      static_cast<float>(std::max(component.xmax - component.xmin + 1, 1));
  const float height =
      static_cast<float>(std::max(component.ymax - component.ymin + 1, 1));
  const float area = width * height;
  const float perimeter = 2.0f * (width + height);
  const float grow = perimeter > 0.0f ? area * config.unclip_ratio / perimeter
                                      : 0.0f;

  const float x0 = std::max(0.0f, static_cast<float>(component.xmin) - grow);
  const float y0 = std::max(0.0f, static_cast<float>(component.ymin) - grow);
  const float x1 =
      std::min(static_cast<float>(pred_width - 1),
               static_cast<float>(component.xmax) + grow);
  const float y1 =
      std::min(static_cast<float>(pred_height - 1),
               static_cast<float>(component.ymax) + grow);

  const float width_scale = static_cast<float>(dest_width) /
                            static_cast<float>(std::max(pred_width, 1));
  const float height_scale = static_cast<float>(dest_height) /
                             static_cast<float>(std::max(pred_height, 1));

  DetectionBox out;
  out.score =
      component.pixels > 0
          ? static_cast<float>(component.sum / static_cast<double>(component.pixels))
          : 0.0f;
  out.component_pixels = component.pixels;
  out.box = Box{
      Point2f{clamp_round(x0 * width_scale, 0.0f,
                          static_cast<float>(dest_width)),
              clamp_round(y0 * height_scale, 0.0f,
                          static_cast<float>(dest_height))},
      Point2f{clamp_round(x1 * width_scale, 0.0f,
                          static_cast<float>(dest_width)),
              clamp_round(y0 * height_scale, 0.0f,
                          static_cast<float>(dest_height))},
      Point2f{clamp_round(x1 * width_scale, 0.0f,
                          static_cast<float>(dest_width)),
              clamp_round(y1 * height_scale, 0.0f,
                          static_cast<float>(dest_height))},
      Point2f{clamp_round(x0 * width_scale, 0.0f,
                          static_cast<float>(dest_width)),
              clamp_round(y1 * height_scale, 0.0f,
                          static_cast<float>(dest_height))},
  };
  return out;
}
#endif

#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
std::pair<std::vector<cv::Point2f>, float>
get_mini_boxes(const std::vector<cv::Point> &contour) {
  const cv::RotatedRect rect = cv::minAreaRect(contour);
  cv::Point2f raw_points[4];
  rect.points(raw_points);
  std::vector<cv::Point2f> points(raw_points, raw_points + 4);
  std::sort(points.begin(), points.end(),
            [](const cv::Point2f &a, const cv::Point2f &b) {
              return a.x < b.x;
            });

  int index_1 = 0;
  int index_2 = 2;
  int index_3 = 3;
  int index_4 = 1;
  if (points[1].y > points[0].y) {
    index_1 = 0;
    index_4 = 1;
  } else {
    index_1 = 1;
    index_4 = 0;
  }
  if (points[3].y > points[2].y) {
    index_2 = 2;
    index_3 = 3;
  } else {
    index_2 = 3;
    index_3 = 2;
  }

  std::vector<cv::Point2f> box{
      points[static_cast<std::size_t>(index_1)],
      points[static_cast<std::size_t>(index_2)],
      points[static_cast<std::size_t>(index_3)],
      points[static_cast<std::size_t>(index_4)],
  };
  return {box, std::min(rect.size.width, rect.size.height)};
}

std::pair<std::vector<cv::Point2f>, float>
get_mini_boxes(const std::vector<cv::Point2f> &contour) {
  std::vector<cv::Point> int_contour;
  int_contour.reserve(contour.size());
  for (const auto &point : contour) {
    int_contour.emplace_back(static_cast<int>(std::round(point.x)),
                             static_cast<int>(std::round(point.y)));
  }
  return get_mini_boxes(int_contour);
}

float box_score_fast(const cv::Mat &pred, const std::vector<cv::Point2f> &box) {
  const int h = pred.rows;
  const int w = pred.cols;
  float min_x = box[0].x;
  float max_x = box[0].x;
  float min_y = box[0].y;
  float max_y = box[0].y;
  for (const auto &point : box) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }

  const int xmin =
      std::max(0, std::min(static_cast<int>(std::floor(min_x)), w - 1));
  const int xmax =
      std::max(0, std::min(static_cast<int>(std::ceil(max_x)), w - 1));
  const int ymin =
      std::max(0, std::min(static_cast<int>(std::floor(min_y)), h - 1));
  const int ymax =
      std::max(0, std::min(static_cast<int>(std::ceil(max_y)), h - 1));
  if (xmax < xmin || ymax < ymin) {
    return 0.0f;
  }

  cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
  std::vector<cv::Point> local_box;
  local_box.reserve(box.size());
  for (const auto &point : box) {
    local_box.emplace_back(static_cast<int>(point.x - static_cast<float>(xmin)),
                           static_cast<int>(point.y - static_cast<float>(ymin)));
  }
  std::vector<std::vector<cv::Point>> polygons{local_box};
  cv::fillPoly(mask, polygons, cv::Scalar(1));
  const cv::Mat roi = pred(cv::Rect(xmin, ymin, xmax - xmin + 1,
                                    ymax - ymin + 1));
  return static_cast<float>(cv::mean(roi, mask)[0]);
}

std::vector<cv::Point2f> unclip(const std::vector<cv::Point2f> &box,
                                float unclip_ratio) {
  const double area = cv::contourArea(box);
  const double length = cv::arcLength(box, true);
  if (length <= 0.0) {
    return {};
  }
  const double distance = area * static_cast<double>(unclip_ratio) / length;

  ClipperLib::Path subject;
  subject.reserve(box.size());
  for (const auto &point : box) {
    subject.emplace_back(static_cast<ClipperLib::cInt>(point.x),
                         static_cast<ClipperLib::cInt>(point.y));
  }

  ClipperLib::ClipperOffset offset;
  offset.AddPath(subject, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  ClipperLib::Paths solution;
  offset.Execute(solution, distance);
  if (solution.size() != 1) {
    return {};
  }

  std::vector<cv::Point2f> expanded;
  expanded.reserve(solution[0].size());
  for (const auto &point : solution[0]) {
    expanded.emplace_back(static_cast<float>(point.X),
                          static_cast<float>(point.Y));
  }
  return expanded;
}

std::vector<DetectionBox>
postprocess_db_map_opencv(const float *pred_data, int pred_height,
                          int pred_width, int dest_height, int dest_width,
                          const DetectionPostprocessConfig &config) {
  cv::Mat pred(pred_height, pred_width, CV_32FC1,
               const_cast<float *>(pred_data));
  cv::Mat bitmap;
  cv::compare(pred, config.thresh, bitmap, cv::CMP_GT);

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  const float width_scale =
      static_cast<float>(dest_width) / static_cast<float>(pred_width);
  const float height_scale =
      static_cast<float>(dest_height) / static_cast<float>(pred_height);
  const int candidate_count =
      std::min(static_cast<int>(contours.size()), config.max_candidates);

  std::vector<DetectionBox> boxes;
  boxes.reserve(static_cast<std::size_t>(candidate_count));
  for (int i = 0; i < candidate_count; ++i) {
    const auto &[points, short_side] = get_mini_boxes(contours[i]);
    if (short_side < static_cast<float>(config.min_size)) {
      continue;
    }
    const float score = box_score_fast(pred, points);
    if (config.box_thresh > score) {
      continue;
    }

    const auto expanded = unclip(points, config.unclip_ratio);
    if (expanded.size() < 4) {
      continue;
    }
    const auto &[box_points, unclipped_short_side] =
        get_mini_boxes(expanded);
    if (unclipped_short_side < static_cast<float>(config.min_size + 2)) {
      continue;
    }

    DetectionBox out;
    out.score = score;
    out.component_pixels = static_cast<int>(std::max(0.0, cv::contourArea(contours[i])));
    for (std::size_t point_index = 0; point_index < box_points.size();
         ++point_index) {
      const auto &point = box_points[point_index];
      out.box[point_index] = Point2f{
          clamp_round(point.x * width_scale, 0.0f,
                      static_cast<float>(dest_width)),
          clamp_round(point.y * height_scale, 0.0f,
                      static_cast<float>(dest_height)),
      };
    }
    boxes.push_back(out);
  }

  std::stable_sort(boxes.begin(), boxes.end(),
                   [](const DetectionBox &a, const DetectionBox &b) {
                     const auto ak = reading_order_key(a);
                     const auto bk = reading_order_key(b);
                     if (ak.first != bk.first) {
                       return ak.first < bk.first;
                     }
                     return ak.second < bk.second;
                   });
  return boxes;
}
#endif

} // namespace

std::vector<DetectionBox>
postprocess_db_map(const float *pred, int pred_height, int pred_width,
                   int dest_height, int dest_width,
                   const DetectionPostprocessConfig &config) {
  if (pred == nullptr) {
    throw std::runtime_error("DB postprocess input pointer is null");
  }
  if (pred_height <= 0 || pred_width <= 0 || dest_height <= 0 ||
      dest_width <= 0) {
    throw std::runtime_error("invalid DB postprocess shape");
  }
  if (config.max_candidates <= 0 || config.min_size <= 0) {
    throw std::runtime_error("invalid DB postprocess config");
  }

#if defined(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
  return postprocess_db_map_opencv(pred, pred_height, pred_width, dest_height,
                                   dest_width, config);
#else
  const std::size_t pixel_count =
      static_cast<std::size_t>(pred_height) * static_cast<std::size_t>(pred_width);
  std::vector<std::uint8_t> visited(pixel_count, 0);
  std::vector<DetectionBox> boxes;
  boxes.reserve(256);
  std::vector<int> stack;

  const auto index_of = [pred_width](int y, int x) {
    return y * pred_width + x;
  };

  int candidates = 0;
  for (int y = 0; y < pred_height; ++y) {
    for (int x = 0; x < pred_width; ++x) {
      const int start_index = index_of(y, x);
      if (visited[static_cast<std::size_t>(start_index)] != 0 ||
          pred[start_index] <= config.thresh) {
        continue;
      }
      if (candidates >= config.max_candidates) {
        break;
      }
      ++candidates;

      Component component{x, y, x, y, 0, 0.0};
      visited[static_cast<std::size_t>(start_index)] = 1;
      stack.clear();
      stack.push_back(start_index);
      while (!stack.empty()) {
        const int current = stack.back();
        stack.pop_back();
        const int cy = current / pred_width;
        const int cx = current - cy * pred_width;
        component.xmin = std::min(component.xmin, cx);
        component.ymin = std::min(component.ymin, cy);
        component.xmax = std::max(component.xmax, cx);
        component.ymax = std::max(component.ymax, cy);
        ++component.pixels;
        component.sum += static_cast<double>(pred[current]);

        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
              continue;
            }
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || nx >= pred_width || ny < 0 || ny >= pred_height) {
              continue;
            }
            const int next = index_of(ny, nx);
            const auto next_index = static_cast<std::size_t>(next);
            if (visited[next_index] != 0 || pred[next] <= config.thresh) {
              continue;
            }
            visited[next_index] = 1;
            stack.push_back(next);
          }
        }
      }

      const int width = component.xmax - component.xmin + 1;
      const int height = component.ymax - component.ymin + 1;
      if (std::min(width, height) < config.min_size) {
        continue;
      }
      DetectionBox box = component_to_box(component, pred_height, pred_width,
                                          dest_height, dest_width, config);
      if (box.score < config.box_thresh) {
        continue;
      }
      const float out_width = std::max(distance(box.box[0], box.box[1]), 1.0f);
      const float out_height = std::max(distance(box.box[0], box.box[3]), 1.0f);
      if (std::min(out_width, out_height) <
          static_cast<float>(config.min_size + 2)) {
        continue;
      }
      boxes.push_back(box);
    }
    if (candidates >= config.max_candidates) {
      break;
    }
  }

  std::stable_sort(boxes.begin(), boxes.end(),
                   [](const DetectionBox &a, const DetectionBox &b) {
                     const auto ak = reading_order_key(a);
                     const auto bk = reading_order_key(b);
                     if (ak.first != bk.first) {
                       return ak.first < bk.first;
                     }
                     return ak.second < bk.second;
                   });
  return boxes;
#endif
}

std::string detection_boxes_to_json(const std::vector<DetectionBox> &boxes) {
  std::ostringstream out;
  out << '{';
  out << "\"boxes\":[";
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (i != 0) {
      out << ',';
    }
    const auto &box = boxes[i];
    out << '{';
    out << "\"score\":";
    append_float(out, box.score);
    out << ",\"component_pixels\":" << box.component_pixels;
    out << ",\"box\":[";
    for (std::size_t point_index = 0; point_index < box.box.points.size();
         ++point_index) {
      if (point_index != 0) {
        out << ',';
      }
      const auto &point = box.box.points[point_index];
      out << '[';
      append_float(out, point.x);
      out << ',';
      append_float(out, point.y);
      out << ']';
    }
    out << "]}";
  }
  out << "]}";
  return out.str();
}

} // namespace ppocrv6_native::detection
