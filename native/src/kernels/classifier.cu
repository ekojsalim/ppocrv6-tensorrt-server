#include "ppocrv6_native/kernels/classifier.h"

#include "ppocrv6_native/common/cuda_check.h"

#include <cfloat>
#include <climits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdexcept>

namespace ppocrv6_native::kernels {
namespace {

constexpr int kMaxRowsPerBlock = 8;
constexpr int kWmmaRowsPerBlock = 16;
constexpr int kWmmaTile = 16;
constexpr int kWarpSize = 32;

int ceil_div(int value, int divisor) {
  return (value + divisor - 1) / divisor;
}

int next_power_of_two(int value) {
  int out = 1;
  while (out < value) {
    out <<= 1;
  }
  return out;
}

__global__ void linear_argmax_prob_kernel(
    const half *__restrict__ hidden, const half *__restrict__ weight,
    const half *__restrict__ bias, int *__restrict__ out_indices,
    float *__restrict__ out_prob, float *__restrict__ out_max_logits, int rows,
    int hidden_size, int vocab_size) {
  extern __shared__ unsigned char shared_raw[];
  float *shared_max = reinterpret_cast<float *>(shared_raw);
  int *shared_id = reinterpret_cast<int *>(shared_max + blockDim.x);
  float *shared_sum = reinterpret_cast<float *>(shared_id + blockDim.x);

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) {
    return;
  }

  float best = -FLT_MAX;
  int best_id = 0;

  for (int v = tid; v < vocab_size; v += blockDim.x) {
    float acc = __half2float(bias[v]);
    for (int h = 0; h < hidden_size; ++h) {
      const float x = __half2float(hidden[row * hidden_size + h]);
      const float w = __half2float(weight[h * vocab_size + v]);
      acc += x * w;
    }
    if (acc > best || (acc == best && v < best_id)) {
      best = acc;
      best_id = v;
    }
  }

  shared_max[tid] = best;
  shared_id[tid] = best_id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float other = shared_max[tid + stride];
      const int other_id = shared_id[tid + stride];
      if (other > shared_max[tid] ||
          (other == shared_max[tid] && other_id < shared_id[tid])) {
        shared_max[tid] = other;
        shared_id[tid] = other_id;
      }
    }
    __syncthreads();
  }

  const float row_max = shared_max[0];
  float local_sum = 0.0f;
  for (int v = tid; v < vocab_size; v += blockDim.x) {
    float acc = __half2float(bias[v]);
    for (int h = 0; h < hidden_size; ++h) {
      const float x = __half2float(hidden[row * hidden_size + h]);
      const float w = __half2float(weight[h * vocab_size + v]);
      acc += x * w;
    }
    local_sum += __expf(acc - row_max);
  }

  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_indices[row] = shared_id[0];
    out_max_logits[row] = row_max;
    out_prob[row] = 1.0f / shared_sum[0];
  }
}

__global__ void linear_argmax_prob_tiled_partial_kernel(
    const half *__restrict__ hidden, const half *__restrict__ weight,
    const half *__restrict__ bias, int *__restrict__ partial_ids,
    float *__restrict__ partial_max, float *__restrict__ partial_sum, int rows,
    int hidden_size, int vocab_size, int row_block_size, int vocab_tile_size,
    int num_vocab_blocks) {
  extern __shared__ unsigned char shared_raw[];
  float *shared_hidden = reinterpret_cast<float *>(shared_raw);
  float *shared_max = shared_hidden + row_block_size * hidden_size;
  int *shared_id = reinterpret_cast<int *>(shared_max +
                                           row_block_size * blockDim.x);
  float *shared_sum =
      reinterpret_cast<float *>(shared_id + row_block_size * blockDim.x);

  const int row_block = blockIdx.x;
  const int vocab_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int row_start = row_block * row_block_size;
  const int vocab_start = vocab_block * vocab_tile_size;
  const int vocab_end = min(vocab_start + vocab_tile_size, vocab_size);

  for (int i = tid; i < row_block_size * hidden_size; i += blockDim.x) {
    const int lane = i / hidden_size;
    const int h = i - lane * hidden_size;
    const int row = row_start + lane;
    shared_hidden[i] =
        row < rows ? __half2float(hidden[row * hidden_size + h]) : 0.0f;
  }
  __syncthreads();

  float local_max[kWmmaRowsPerBlock];
  int local_id[kWmmaRowsPerBlock];
  float local_sum[kWmmaRowsPerBlock];
  for (int lane = 0; lane < kWmmaRowsPerBlock; ++lane) {
    local_max[lane] = -FLT_MAX;
    local_id[lane] = vocab_size + 1;
    local_sum[lane] = 0.0f;
  }

  for (int v = vocab_start + tid; v < vocab_end; v += blockDim.x) {
    const float bias_value = __half2float(bias[v]);
    for (int lane = 0; lane < row_block_size; ++lane) {
      const int row = row_start + lane;
      if (row >= rows) {
        continue;
      }
      float acc = bias_value;
      const float *row_hidden = shared_hidden + lane * hidden_size;
      for (int h = 0; h < hidden_size; ++h) {
        acc += row_hidden[h] * __half2float(weight[h * vocab_size + v]);
      }

      const float previous_max = local_max[lane];
      if (acc > previous_max) {
        local_sum[lane] = local_sum[lane] * __expf(previous_max - acc) + 1.0f;
        local_max[lane] = acc;
        local_id[lane] = v;
      } else {
        local_sum[lane] += __expf(acc - previous_max);
        if (acc == previous_max && v < local_id[lane]) {
          local_id[lane] = v;
        }
      }
    }
  }

  for (int lane = 0; lane < row_block_size; ++lane) {
    const int offset = lane * blockDim.x + tid;
    shared_max[offset] = local_max[lane];
    shared_id[offset] = local_id[lane];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int lane = 0; lane < row_block_size; ++lane) {
        const int dst = lane * blockDim.x + tid;
        const int src = dst + stride;
        const float other = shared_max[src];
        const int other_id = shared_id[src];
        if (other > shared_max[dst] ||
            (other == shared_max[dst] && other_id < shared_id[dst])) {
          shared_max[dst] = other;
          shared_id[dst] = other_id;
        }
      }
    }
    __syncthreads();
  }

  for (int lane = 0; lane < row_block_size; ++lane) {
    const float tile_max = shared_max[lane * blockDim.x];
    const float contribution =
        local_sum[lane] == 0.0f ? 0.0f
                                : local_sum[lane] *
                                      __expf(local_max[lane] - tile_max);
    shared_sum[lane * blockDim.x + tid] = contribution;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int lane = 0; lane < row_block_size; ++lane) {
        const int dst = lane * blockDim.x + tid;
        shared_sum[dst] += shared_sum[dst + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int lane = 0; lane < row_block_size; ++lane) {
      const int row = row_start + lane;
      if (row >= rows) {
        continue;
      }
      const int partial_offset =
          (row_block * num_vocab_blocks + vocab_block) * row_block_size + lane;
      partial_ids[partial_offset] = shared_id[lane * blockDim.x];
      partial_max[partial_offset] = shared_max[lane * blockDim.x];
      partial_sum[partial_offset] = shared_sum[lane * blockDim.x];
    }
  }
}

__global__ void linear_argmax_prob_tiled_final_kernel(
    const int *__restrict__ partial_ids, const float *__restrict__ partial_max,
    const float *__restrict__ partial_sum, int *__restrict__ out_indices,
    float *__restrict__ out_prob, float *__restrict__ out_max_logits, int rows,
    int row_block_size, int num_vocab_blocks) {
  extern __shared__ unsigned char shared_raw[];
  float *shared_max = reinterpret_cast<float *>(shared_raw);
  int *shared_id = reinterpret_cast<int *>(shared_max +
                                           row_block_size * blockDim.x);
  float *shared_sum =
      reinterpret_cast<float *>(shared_id + row_block_size * blockDim.x);

  const int row_block = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_start = row_block * row_block_size;

  float local_max[kWmmaRowsPerBlock];
  int local_id[kWmmaRowsPerBlock];
  float local_sum[kWmmaRowsPerBlock];
  for (int lane = 0; lane < kWmmaRowsPerBlock; ++lane) {
    local_max[lane] = -FLT_MAX;
    local_id[lane] = INT_MAX;
    local_sum[lane] = 0.0f;
  }

  if (tid < num_vocab_blocks) {
    for (int lane = 0; lane < row_block_size; ++lane) {
      const int row = row_start + lane;
      if (row >= rows) {
        continue;
      }
      const int partial_offset =
          (row_block * num_vocab_blocks + tid) * row_block_size + lane;
      local_max[lane] = partial_max[partial_offset];
      local_id[lane] = partial_ids[partial_offset];
      local_sum[lane] = partial_sum[partial_offset];
    }
  }

  for (int lane = 0; lane < row_block_size; ++lane) {
    const int offset = lane * blockDim.x + tid;
    shared_max[offset] = local_max[lane];
    shared_id[offset] = local_id[lane];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int lane = 0; lane < row_block_size; ++lane) {
        const int dst = lane * blockDim.x + tid;
        const int src = dst + stride;
        const float other = shared_max[src];
        const int other_id = shared_id[src];
        if (other > shared_max[dst] ||
            (other == shared_max[dst] && other_id < shared_id[dst])) {
          shared_max[dst] = other;
          shared_id[dst] = other_id;
        }
      }
    }
    __syncthreads();
  }

  for (int lane = 0; lane < row_block_size; ++lane) {
    const float row_max = shared_max[lane * blockDim.x];
    const float contribution =
        local_sum[lane] == 0.0f ? 0.0f
                                : local_sum[lane] *
                                      __expf(local_max[lane] - row_max);
    shared_sum[lane * blockDim.x + tid] = contribution;
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int lane = 0; lane < row_block_size; ++lane) {
        const int dst = lane * blockDim.x + tid;
        shared_sum[dst] += shared_sum[dst + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int lane = 0; lane < row_block_size; ++lane) {
      const int row = row_start + lane;
      if (row >= rows) {
        continue;
      }
      const int offset = lane * blockDim.x;
      out_indices[row] = shared_id[offset];
      out_max_logits[row] = shared_max[offset];
      out_prob[row] = 1.0f / shared_sum[offset];
    }
  }
}

__device__ void warp_reduce_best(float &best, int &best_id) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffff, best, offset);
    const int other_id = __shfl_down_sync(0xffffffff, best_id, offset);
    if (other > best || (other == best && other_id < best_id)) {
      best = other;
      best_id = other_id;
    }
  }
}

__device__ float warp_reduce_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void linear_argmax_prob_wmma_partial_kernel(
    const half *__restrict__ hidden, const half *__restrict__ weight,
    const half *__restrict__ bias, int *__restrict__ partial_ids,
    float *__restrict__ partial_max, float *__restrict__ partial_sum, int rows,
    int hidden_size, int vocab_size, int vocab_tile_size,
    int num_vocab_blocks) {
  extern __shared__ float tile_logits[];
  const int row_block = blockIdx.x;
  const int vocab_block = blockIdx.y;
  const int row_start = row_block * kWmmaRowsPerBlock;
  const int vocab_start = vocab_block * vocab_tile_size;
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int num_col_warps = (vocab_tile_size + kWmmaTile - 1) / kWmmaTile;
  const int col_start = vocab_start + warp_id * kWmmaTile;
  const int col_offset = warp_id * kWmmaTile;
  const bool full_rows = row_start + kWmmaRowsPerBlock <= rows;
  const bool full_cols = col_start + kWmmaTile <= vocab_size;

  for (int i = threadIdx.x; i < kWmmaRowsPerBlock * vocab_tile_size;
       i += blockDim.x) {
    tile_logits[i] = -FLT_MAX;
  }
  __syncthreads();

  if (warp_id < num_col_warps) {
    if (full_rows && full_cols && hidden_size % kWmmaTile == 0) {
      using namespace nvcuda;
      wmma::fragment<wmma::matrix_a, kWmmaTile, kWmmaTile, kWmmaTile, half,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b, kWmmaTile, kWmmaTile, kWmmaTile, half,
                     wmma::row_major>
          b_frag;
      wmma::fragment<wmma::accumulator, kWmmaTile, kWmmaTile, kWmmaTile, float>
          acc_frag;
      wmma::fill_fragment(acc_frag, 0.0f);
      for (int k = 0; k < hidden_size; k += kWmmaTile) {
        wmma::load_matrix_sync(
            a_frag, hidden + row_start * hidden_size + k, hidden_size);
        wmma::load_matrix_sync(
            b_frag, weight + k * vocab_size + col_start, vocab_size);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
      wmma::store_matrix_sync(tile_logits + col_offset, acc_frag,
                              vocab_tile_size, wmma::mem_row_major);
    } else {
      for (int idx = lane; idx < kWmmaRowsPerBlock * kWmmaTile;
           idx += warpSize) {
        const int row_lane = idx / kWmmaTile;
        const int col_lane = idx - row_lane * kWmmaTile;
        const int row = row_start + row_lane;
        const int col = col_start + col_lane;
        if (row >= rows || col >= vocab_size ||
            col >= vocab_start + vocab_tile_size) {
          continue;
        }
        float acc = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
          acc += __half2float(hidden[row * hidden_size + h]) *
                 __half2float(weight[h * vocab_size + col]);
        }
        tile_logits[row_lane * vocab_tile_size + col_offset + col_lane] = acc;
      }
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kWmmaRowsPerBlock * vocab_tile_size;
       i += blockDim.x) {
    const int row_lane = i / vocab_tile_size;
    const int col_lane = i - row_lane * vocab_tile_size;
    const int row = row_start + row_lane;
    const int col = vocab_start + col_lane;
    if (row < rows && col < vocab_size) {
      tile_logits[i] += __half2float(bias[col]);
    } else {
      tile_logits[i] = -FLT_MAX;
    }
  }
  __syncthreads();

  if (warp_id < kWmmaRowsPerBlock) {
    const int row = row_start + warp_id;
    float best = -FLT_MAX;
    int best_id = INT_MAX;
    if (row < rows) {
      for (int col_lane = lane; col_lane < vocab_tile_size;
           col_lane += warpSize) {
        const int col = vocab_start + col_lane;
        if (col >= vocab_size) {
          continue;
        }
        const float value = tile_logits[warp_id * vocab_tile_size + col_lane];
        if (value > best || (value == best && col < best_id)) {
          best = value;
          best_id = col;
        }
      }
    }
    warp_reduce_best(best, best_id);
    best = __shfl_sync(0xffffffff, best, 0);

    float sum = 0.0f;
    if (row < rows) {
      for (int col_lane = lane; col_lane < vocab_tile_size;
           col_lane += warpSize) {
        const int col = vocab_start + col_lane;
        if (col >= vocab_size) {
          continue;
        }
        sum += __expf(tile_logits[warp_id * vocab_tile_size + col_lane] - best);
      }
    }
    sum = warp_reduce_sum(sum);

    if (lane == 0 && row < rows) {
      const int partial_offset =
          (row_block * num_vocab_blocks + vocab_block) * kWmmaRowsPerBlock +
          warp_id;
      partial_ids[partial_offset] = best_id;
      partial_max[partial_offset] = best;
      partial_sum[partial_offset] = sum;
    }
  }
}

} // namespace

TiledClassifierWorkspaceShape tiled_classifier_workspace_shape(
    int rows, int vocab_size, int row_block_size, int vocab_tile_size) {
  if (rows <= 0 || vocab_size <= 0) {
    return {};
  }
  if (row_block_size <= 0 || row_block_size > kWmmaRowsPerBlock ||
      vocab_tile_size <= 0) {
    throw std::invalid_argument("invalid tiled classifier workspace shape");
  }
  TiledClassifierWorkspaceShape shape;
  shape.num_row_blocks = ceil_div(rows, row_block_size);
  shape.num_vocab_blocks = ceil_div(vocab_size, vocab_tile_size);
  shape.partial_count = static_cast<std::size_t>(shape.num_row_blocks) *
                        static_cast<std::size_t>(shape.num_vocab_blocks) *
                        static_cast<std::size_t>(row_block_size);
  return shape;
}

void cuda_linear_argmax_prob(const half *hidden, const half *weight,
                             const half *bias, int *out_indices,
                             float *out_prob, float *out_max_logits, int rows,
                             int hidden_size, int vocab_size, int threads,
                             cudaStream_t stream) {
  if (rows <= 0) {
    return;
  }
  const int shared_bytes =
      threads * (static_cast<int>(sizeof(float)) * 2 + static_cast<int>(sizeof(int)));
  linear_argmax_prob_kernel<<<rows, threads, shared_bytes, stream>>>(
      hidden, weight, bias, out_indices, out_prob, out_max_logits, rows,
      hidden_size, vocab_size);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());
}

void cuda_linear_argmax_prob_tiled(
    const half *hidden, const half *weight, const half *bias, int *out_indices,
    float *out_prob, float *out_max_logits, int rows, int hidden_size,
    int vocab_size, int row_block_size, int vocab_tile_size, int threads,
    int *partial_ids, float *partial_max, float *partial_sum,
    cudaStream_t stream) {
  if (rows <= 0) {
    return;
  }
  if (hidden_size <= 0 || vocab_size <= 0 || row_block_size <= 0 ||
      row_block_size > kMaxRowsPerBlock || vocab_tile_size <= 0 ||
      threads <= 0 || (threads & (threads - 1)) != 0 ||
      partial_ids == nullptr || partial_max == nullptr ||
      partial_sum == nullptr) {
    throw std::invalid_argument("invalid tiled classifier launch arguments");
  }

  const auto workspace = tiled_classifier_workspace_shape(
      rows, vocab_size, row_block_size, vocab_tile_size);
  const dim3 partial_grid(workspace.num_row_blocks, workspace.num_vocab_blocks);
  const int partial_shared_bytes =
      row_block_size * hidden_size * static_cast<int>(sizeof(float)) +
      row_block_size * threads *
          (static_cast<int>(sizeof(float)) * 2 + static_cast<int>(sizeof(int)));
  linear_argmax_prob_tiled_partial_kernel<<<partial_grid, threads,
                                            partial_shared_bytes, stream>>>(
      hidden, weight, bias, partial_ids, partial_max, partial_sum, rows,
      hidden_size, vocab_size, row_block_size, vocab_tile_size,
      workspace.num_vocab_blocks);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());

  const int final_threads = next_power_of_two(workspace.num_vocab_blocks);
  if (final_threads > 1024) {
    throw std::invalid_argument("too many vocab blocks for tiled classifier final reduction");
  }
  const int final_shared_bytes =
      row_block_size * final_threads *
      (static_cast<int>(sizeof(float)) * 2 + static_cast<int>(sizeof(int)));
  linear_argmax_prob_tiled_final_kernel<<<workspace.num_row_blocks,
                                          final_threads, final_shared_bytes,
                                          stream>>>(
      partial_ids, partial_max, partial_sum, out_indices, out_prob,
      out_max_logits, rows, row_block_size, workspace.num_vocab_blocks);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());
}

void cuda_linear_argmax_prob_wmma(
    const half *hidden, const half *weight, const half *bias, int *out_indices,
    float *out_prob, float *out_max_logits, int rows, int hidden_size,
    int vocab_size, int vocab_tile_size, int *partial_ids, float *partial_max,
    float *partial_sum, cudaStream_t stream) {
  if (rows <= 0) {
    return;
  }
  if (hidden_size <= 0 || vocab_size <= 0 || hidden_size % kWmmaTile != 0 ||
      vocab_tile_size <= 0 || vocab_tile_size % kWmmaTile != 0 ||
      partial_ids == nullptr || partial_max == nullptr ||
      partial_sum == nullptr) {
    throw std::invalid_argument("invalid WMMA classifier launch arguments");
  }
  const auto workspace = tiled_classifier_workspace_shape(
      rows, vocab_size, kWmmaRowsPerBlock, vocab_tile_size);
  const int col_warps = ceil_div(vocab_tile_size, kWmmaTile);
  const int reduce_warps = kWmmaRowsPerBlock;
  const int block_warps = std::max(col_warps, reduce_warps);
  const int threads = block_warps * kWarpSize;
  if (threads > 1024) {
    throw std::invalid_argument("too many WMMA classifier threads per block");
  }

  const dim3 partial_grid(workspace.num_row_blocks, workspace.num_vocab_blocks);
  const int partial_shared_bytes =
      kWmmaRowsPerBlock * vocab_tile_size * static_cast<int>(sizeof(float));
  linear_argmax_prob_wmma_partial_kernel<<<partial_grid, threads,
                                           partial_shared_bytes, stream>>>(
      hidden, weight, bias, partial_ids, partial_max, partial_sum, rows,
      hidden_size, vocab_size, vocab_tile_size, workspace.num_vocab_blocks);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());

  const int final_threads = next_power_of_two(workspace.num_vocab_blocks);
  if (final_threads > 1024) {
    throw std::invalid_argument("too many vocab blocks for WMMA classifier final reduction");
  }
  const int final_shared_bytes =
      kWmmaRowsPerBlock * final_threads *
      (static_cast<int>(sizeof(float)) * 2 + static_cast<int>(sizeof(int)));
  linear_argmax_prob_tiled_final_kernel<<<workspace.num_row_blocks,
                                          final_threads, final_shared_bytes,
                                          stream>>>(
      partial_ids, partial_max, partial_sum, out_indices, out_prob,
      out_max_logits, rows, kWmmaRowsPerBlock, workspace.num_vocab_blocks);
  PPOCRV6_CUDA_CHECK(cudaGetLastError());
}

} // namespace ppocrv6_native::kernels
