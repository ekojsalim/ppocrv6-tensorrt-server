#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace ppocrv6_native::plugin {

class WmmaClassifierPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
  explicit WmmaClassifierPlugin(int vocab_tile_size = 512);
  WmmaClassifierPlugin(const void *data, std::size_t length);

  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int32_t output_index, const nvinfer1::DimsExprs *inputs,
      int32_t nb_inputs, nvinfer1::IExprBuilder &expr_builder) noexcept override;
  bool supportsFormatCombination(
      int32_t pos, const nvinfer1::PluginTensorDesc *in_out,
      int32_t nb_inputs, int32_t nb_outputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int32_t nb_inputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int32_t nb_outputs) noexcept override;
  std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                               int32_t nb_inputs,
                               const nvinfer1::PluginTensorDesc *outputs,
                               int32_t nb_outputs) const noexcept override;
  int32_t enqueue(const nvinfer1::PluginTensorDesc *input_desc,
                  const nvinfer1::PluginTensorDesc *output_desc,
                  const void *const *inputs, void *const *outputs,
                  void *workspace, cudaStream_t stream) noexcept override;

  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int32_t getNbOutputs() const noexcept override;
  int32_t initialize() noexcept override;
  void terminate() noexcept override;
  std::size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *plugin_namespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;
  nvinfer1::DataType getOutputDataType(
      int32_t index, const nvinfer1::DataType *input_types,
      int32_t nb_inputs) const noexcept override;
  void attachToContext(cudnnContext *cudnn, cublasContext *cublas,
                       nvinfer1::IGpuAllocator *allocator) noexcept override;
  void detachFromContext() noexcept override;

private:
  int vocab_tile_size_ = 512;
  std::string namespace_;
};

} // namespace ppocrv6_native::plugin
