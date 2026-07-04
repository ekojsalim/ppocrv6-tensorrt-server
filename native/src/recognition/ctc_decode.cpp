#include "ppocrv6_native/recognition/ctc_decode.h"

#include <stdexcept>

namespace ppocrv6_native::recognition {

DecodedText ctc_greedy_decode(std::span<const int> indices,
                              std::span<const float> scores,
                              const std::vector<std::string> &characters,
                              int blank_id) {
  if (!scores.empty() && scores.size() != indices.size()) {
    throw std::invalid_argument("scores must be empty or match indices size");
  }

  DecodedText result;
  result.text.reserve(indices.size());
  result.class_ids.reserve(indices.size());
  result.per_char_scores.reserve(indices.size());

  int previous = -1;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const int class_id = indices[i];
    if (class_id == previous) {
      previous = class_id;
      continue;
    }
    previous = class_id;

    if (class_id == blank_id) {
      continue;
    }

    if (class_id >= 0 &&
        class_id < static_cast<int>(characters.size())) {
      result.text += characters[static_cast<std::size_t>(class_id)];
    } else {
      result.text += "<" + std::to_string(class_id) + ">";
    }
    result.class_ids.push_back(class_id);
    result.per_char_scores.push_back(scores.empty() ? 1.0f : scores[i]);
  }

  if (!result.per_char_scores.empty()) {
    float sum = 0.0f;
    for (float score : result.per_char_scores) {
      sum += score;
    }
    result.score = sum / static_cast<float>(result.per_char_scores.size());
  }
  return result;
}

std::vector<DecodedText> ctc_greedy_decode_batch(
    std::span<const int> indices, std::span<const float> scores,
    int batch_size, int seq_len, const std::vector<std::string> &characters,
    int blank_id) {
  if (batch_size < 0 || seq_len < 0) {
    throw std::invalid_argument("batch_size and seq_len must be non-negative");
  }
  const std::size_t row_size = static_cast<std::size_t>(seq_len);
  const std::size_t expected = static_cast<std::size_t>(batch_size) * row_size;
  if (indices.size() != expected) {
    throw std::invalid_argument("indices size does not match batch shape");
  }
  if (!scores.empty() && scores.size() != expected) {
    throw std::invalid_argument("scores size does not match batch shape");
  }

  std::vector<DecodedText> results;
  results.reserve(static_cast<std::size_t>(batch_size));
  for (int row = 0; row < batch_size; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_size;
    const auto index_row = indices.subspan(offset, row_size);
    const auto score_row =
        scores.empty() ? std::span<const float>{}
                       : scores.subspan(offset, row_size);
    results.push_back(
        ctc_greedy_decode(index_row, score_row, characters, blank_id));
  }
  return results;
}

} // namespace ppocrv6_native::recognition
