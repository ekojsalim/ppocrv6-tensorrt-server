#include "ppocrv6_native/plugin/classifier_plugin.h"

#include "ppocrv6_native/kernels/classifier.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>

namespace ppocrv6_native::plugin {
namespace {

constexpr const char *kPluginType = "PPOCRv6WmmaClassifier";
constexpr const char *kPluginVersion = "1";
constexpr int kWmmaRowsPerBlock = 16;

int volume2(const nvinfer1::Dims &dims) {
  if (dims.nbDims < 2 || dims.d[0] <= 0 || dims.d[1] <= 0) {
    return 0;
  }
  return dims.d[0] * dims.d[1];
}

int vocab_size_from_desc(const nvinfer1::PluginTensorDesc *inputs) {
  if (inputs[2].dims.nbDims == 1) {
    return inputs[2].dims.d[0];
  }
  return inputs[1].dims.nbDims >= 2 ? inputs[1].dims.d[1] : 0;
}

std::size_t workspace_bytes(int rows, int vocab_size, int vocab_tile_size) {
  if (rows <= 0 || vocab_size <= 0 || vocab_tile_size <= 0) {
    return 0;
  }
  const auto shape = kernels::tiled_classifier_workspace_shape(
      rows, vocab_size, kWmmaRowsPerBlock, vocab_tile_size);
  return shape.partial_count * (sizeof(int) + sizeof(float) + sizeof(float));
}

} // namespace

WmmaClassifierPlugin::WmmaClassifierPlugin(int vocab_tile_size)
    : vocab_tile_size_(vocab_tile_size) {}

WmmaClassifierPlugin::WmmaClassifierPlugin(const void *data, std::size_t length) {
  if (data != nullptr && length >= sizeof(int32_t)) {
    std::memcpy(&vocab_tile_size_, data, sizeof(int32_t));
  }
}

nvinfer1::IPluginV2DynamicExt *WmmaClassifierPlugin::clone() const noexcept {
  auto *plugin = new WmmaClassifierPlugin(vocab_tile_size_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::DimsExprs WmmaClassifierPlugin::getOutputDimensions(
    int32_t output_index, const nvinfer1::DimsExprs *inputs,
    int32_t nb_inputs, nvinfer1::IExprBuilder &expr_builder) noexcept {
  (void)expr_builder;
  nvinfer1::DimsExprs out{};
  if (output_index < 0 || output_index >= getNbOutputs() || nb_inputs != 3 ||
      inputs[0].nbDims != 3) {
    out.nbDims = -1;
    return out;
  }
  out.nbDims = 2;
  out.d[0] = inputs[0].d[0];
  out.d[1] = inputs[0].d[1];
  return out;
}

bool WmmaClassifierPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc *in_out, int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  if (nb_inputs != 3 || nb_outputs != 3 || pos < 0 || pos >= 6) {
    return false;
  }
  if (in_out[pos].format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }
  if (pos < 3) {
    return in_out[pos].type == nvinfer1::DataType::kHALF;
  }
  if (pos == 3) {
    return in_out[pos].type == nvinfer1::DataType::kINT32;
  }
  return in_out[pos].type == nvinfer1::DataType::kFLOAT;
}

void WmmaClassifierPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *in, int32_t nb_inputs,
    const nvinfer1::DynamicPluginTensorDesc *out,
    int32_t nb_outputs) noexcept {
  (void)in;
  (void)nb_inputs;
  (void)out;
  (void)nb_outputs;
}

std::size_t WmmaClassifierPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int32_t nb_inputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int32_t nb_outputs) const noexcept {
  (void)outputs;
  (void)nb_outputs;
  if (nb_inputs != 3 || inputs[0].dims.nbDims != 3) {
    return 0;
  }
  const int rows = volume2(inputs[0].dims);
  const int vocab_size = vocab_size_from_desc(inputs);
  return workspace_bytes(rows, vocab_size, vocab_tile_size_);
}

int32_t WmmaClassifierPlugin::enqueue(
    const nvinfer1::PluginTensorDesc *input_desc,
    const nvinfer1::PluginTensorDesc *output_desc, const void *const *inputs,
    void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  (void)output_desc;
  try {
    const auto &hidden_dims = input_desc[0].dims;
    if (hidden_dims.nbDims != 3 || input_desc[1].dims.nbDims != 2 ||
        input_desc[2].dims.nbDims != 1) {
      return 1;
    }
    const int rows = hidden_dims.d[0] * hidden_dims.d[1];
    const int hidden_size = hidden_dims.d[2];
    const int vocab_size = input_desc[2].dims.d[0];
    const auto shape = kernels::tiled_classifier_workspace_shape(
        rows, vocab_size, kWmmaRowsPerBlock, vocab_tile_size_);

    auto *base = static_cast<unsigned char *>(workspace);
    auto *partial_ids = reinterpret_cast<int *>(base);
    auto *partial_max =
        reinterpret_cast<float *>(base + shape.partial_count * sizeof(int));
    auto *partial_sum = reinterpret_cast<float *>(
        base + shape.partial_count * (sizeof(int) + sizeof(float)));

    kernels::cuda_linear_argmax_prob_wmma(
        static_cast<const half *>(inputs[0]), static_cast<const half *>(inputs[1]),
        static_cast<const half *>(inputs[2]), static_cast<int *>(outputs[0]),
        static_cast<float *>(outputs[1]), static_cast<float *>(outputs[2]),
        rows, hidden_size, vocab_size, vocab_tile_size_, partial_ids,
        partial_max, partial_sum, stream);
    return 0;
  } catch (...) {
    return 2;
  }
}

const char *WmmaClassifierPlugin::getPluginType() const noexcept {
  return kPluginType;
}

const char *WmmaClassifierPlugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}

int32_t WmmaClassifierPlugin::getNbOutputs() const noexcept { return 3; }

int32_t WmmaClassifierPlugin::initialize() noexcept { return 0; }

void WmmaClassifierPlugin::terminate() noexcept {}

std::size_t WmmaClassifierPlugin::getSerializationSize() const noexcept {
  return sizeof(int32_t);
}

void WmmaClassifierPlugin::serialize(void *buffer) const noexcept {
  const auto value = static_cast<int32_t>(vocab_tile_size_);
  std::memcpy(buffer, &value, sizeof(value));
}

void WmmaClassifierPlugin::destroy() noexcept { delete this; }

void WmmaClassifierPlugin::setPluginNamespace(
    const char *plugin_namespace) noexcept {
  namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
}

const char *WmmaClassifierPlugin::getPluginNamespace() const noexcept {
  return namespace_.c_str();
}

nvinfer1::DataType WmmaClassifierPlugin::getOutputDataType(
    int32_t index, const nvinfer1::DataType *input_types,
    int32_t nb_inputs) const noexcept {
  (void)input_types;
  (void)nb_inputs;
  return index == 0 ? nvinfer1::DataType::kINT32
                    : nvinfer1::DataType::kFLOAT;
}

void WmmaClassifierPlugin::attachToContext(
    cudnnContext *cudnn, cublasContext *cublas,
    nvinfer1::IGpuAllocator *allocator) noexcept {
  (void)cudnn;
  (void)cublas;
  (void)allocator;
}

void WmmaClassifierPlugin::detachFromContext() noexcept {}

} // namespace ppocrv6_native::plugin
