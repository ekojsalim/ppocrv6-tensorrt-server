#pragma once

#include <span>
#include <string>
#include <vector>

namespace ppocrv6_native::recognition {

struct DecodedText {
  std::string text;
  float score = 0.0f;
  std::vector<int> class_ids;
  std::vector<float> per_char_scores;
};

DecodedText ctc_greedy_decode(std::span<const int> indices,
                              std::span<const float> scores,
                              const std::vector<std::string> &characters,
                              int blank_id = 0);

std::vector<DecodedText> ctc_greedy_decode_batch(
    std::span<const int> indices, std::span<const float> scores,
    int batch_size, int seq_len, const std::vector<std::string> &characters,
    int blank_id = 0);

} // namespace ppocrv6_native::recognition
