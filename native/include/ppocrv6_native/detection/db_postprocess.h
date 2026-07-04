#pragma once

#include "ppocrv6_native/common/box.h"

#include <string>
#include <vector>

namespace ppocrv6_native::detection {

struct DetectionPostprocessConfig {
  float thresh = 0.2f;
  float box_thresh = 0.45f;
  float unclip_ratio = 1.4f;
  int max_candidates = 3000;
  int min_size = 3;
};

struct DetectionBox {
  Box box;
  float score = 0.0f;
  int component_pixels = 0;
};

std::vector<DetectionBox>
postprocess_db_map(const float *pred, int pred_height, int pred_width,
                   int dest_height, int dest_width,
                   const DetectionPostprocessConfig &config = {});

std::string detection_boxes_to_json(const std::vector<DetectionBox> &boxes);

} // namespace ppocrv6_native::detection
