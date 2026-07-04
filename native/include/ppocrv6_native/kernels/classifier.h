#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ppocrv6_native::kernels {

struct TiledClassifierWorkspaceShape {
  int num_row_blocks = 0;
  int num_vocab_blocks = 0;
  std::size_t partial_count = 0;
};

TiledClassifierWorkspaceShape tiled_classifier_workspace_shape(
    int rows, int vocab_size, int row_block_size, int vocab_tile_size);

void cuda_linear_argmax_prob(const half *hidden, const half *weight,
                             const half *bias, int *out_indices,
                             float *out_prob, float *out_max_logits, int rows,
                             int hidden_size, int vocab_size, int threads,
                             cudaStream_t stream = nullptr);

void cuda_linear_argmax_prob_tiled(
    const half *hidden, const half *weight, const half *bias, int *out_indices,
    float *out_prob, float *out_max_logits, int rows, int hidden_size,
    int vocab_size, int row_block_size, int vocab_tile_size, int threads,
    int *partial_ids, float *partial_max, float *partial_sum,
    cudaStream_t stream = nullptr);

void cuda_linear_argmax_prob_wmma(
    const half *hidden, const half *weight, const half *bias, int *out_indices,
    float *out_prob, float *out_max_logits, int rows, int hidden_size,
    int vocab_size, int vocab_tile_size, int *partial_ids, float *partial_max,
    float *partial_sum, cudaStream_t stream = nullptr);

} // namespace ppocrv6_native::kernels
