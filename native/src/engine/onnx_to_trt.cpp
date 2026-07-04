#include "ppocrv6_native/engine/onnx_to_trt.h"

#include "ppocrv6_native/engine/trt_engine.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace ppocrv6_native::engine {
namespace {

struct BuildLogger final : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT build] " << msg << '\n';
    }
  }
};

bool valid_profile(const RecognitionProfile &profile) {
  return profile.min_batch > 0 && profile.opt_batch >= profile.min_batch &&
         profile.max_batch >= profile.opt_batch && profile.channels == 3 &&
         profile.height == 48 && profile.min_width > 0 &&
         profile.opt_width >= profile.min_width &&
         profile.max_width >= profile.opt_width &&
         profile.workspace_bytes > 0 &&
         profile.builder_optimization_level >= 0 &&
         profile.builder_optimization_level <= 5;
}

bool valid_profile(const DetectionProfile &profile) {
  return profile.min_batch > 0 && profile.opt_batch >= profile.min_batch &&
         profile.max_batch >= profile.opt_batch && profile.channels == 3 &&
         profile.min_height > 0 && profile.opt_height >= profile.min_height &&
         profile.max_height >= profile.opt_height && profile.min_width > 0 &&
         profile.opt_width >= profile.min_width &&
         profile.max_width >= profile.opt_width &&
         profile.workspace_bytes > 0 &&
         profile.builder_optimization_level >= 0 &&
         profile.builder_optimization_level <= 5;
}

} // namespace

bool build_recognition_hidden_engine(const std::string &onnx_path,
                                     const std::string &engine_path,
                                     const RecognitionProfile &profile) {
  return build_recognition_hidden_engine(
      onnx_path, engine_path, std::vector<RecognitionProfile>{profile});
}

bool build_recognition_hidden_engine(
    const std::string &onnx_path, const std::string &engine_path,
    const std::vector<RecognitionProfile> &profiles) {
  if (profiles.empty()) {
    std::cerr << "[TRT] no recognition profiles provided\n";
    return false;
  }
  for (std::size_t index = 0; index < profiles.size(); ++index) {
    if (!valid_profile(profiles[index])) {
      std::cerr << "[TRT] invalid recognition profile at index " << index
                << '\n';
      return false;
    }
    if (profiles[index].channels != profiles[0].channels ||
        profiles[index].height != profiles[0].height ||
        profiles[index].fp16 != profiles[0].fp16 ||
        profiles[index].builder_optimization_level !=
            profiles[0].builder_optimization_level) {
      std::cerr << "[TRT] recognition profiles must share channels, height, "
                   "fp16, and builder optimization level\n";
      return false;
    }
  }
  if (!std::filesystem::exists(onnx_path)) {
    std::cerr << "[TRT] ONNX not found: " << onnx_path << '\n';
    return false;
  }

  BuildLogger logger;
  auto builder =
      std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "[TRT] failed to create builder\n";
    return false;
  }

  auto network =
      std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
  if (!network) {
    std::cerr << "[TRT] failed to create network\n";
    return false;
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser ||
      !parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "[TRT] failed to parse ONNX: " << onnx_path << '\n';
    return false;
  }

  if (network->getNbInputs() != 1) {
    std::cerr << "[TRT] expected one hidden-recognizer input, got "
              << network->getNbInputs() << '\n';
    return false;
  }

  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "[TRT] failed to create builder config\n";
    return false;
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             std::max_element(
                                 profiles.begin(), profiles.end(),
                                 [](const RecognitionProfile &a,
                                    const RecognitionProfile &b) {
                                   return a.workspace_bytes < b.workspace_bytes;
                                 })
                                 ->workspace_bytes);
  config->setBuilderOptimizationLevel(profiles[0].builder_optimization_level);
  if (profiles[0].fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto *input = network->getInput(0);
  const char *input_name = input->getName();
  std::cout << "[TRT] building hidden recognizer: " << onnx_path << '\n'
            << "[TRT] input=" << input_name << '\n';
  for (std::size_t index = 0; index < profiles.size(); ++index) {
    const auto &profile = profiles[index];
    auto *trt_profile = builder->createOptimizationProfile();
    if (!trt_profile) {
      std::cerr << "[TRT] failed to create optimization profile " << index
                << '\n';
      return false;
    }
    trt_profile->setDimensions(
        input_name, nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{profile.min_batch, profile.channels, profile.height,
                        profile.min_width});
    trt_profile->setDimensions(
        input_name, nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{profile.opt_batch, profile.channels, profile.height,
                        profile.opt_width});
    trt_profile->setDimensions(
        input_name, nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{profile.max_batch, profile.channels, profile.height,
                        profile.max_width});
    if (!trt_profile->isValid()) {
      std::cerr << "[TRT] optimization profile " << index << " is invalid\n";
      return false;
    }
    const int added = config->addOptimizationProfile(trt_profile);
    if (added < 0) {
      std::cerr << "[TRT] failed to add optimization profile " << index
                << '\n';
      return false;
    }
    std::cout << "[TRT] profile[" << added << "] min=(" << profile.min_batch
              << ",3,48," << profile.min_width << ") opt=("
              << profile.opt_batch << ",3,48," << profile.opt_width
              << ") max=(" << profile.max_batch << ",3,48,"
              << profile.max_width << ")\n";
  }

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) {
    std::cerr << "[TRT] buildSerializedNetwork failed\n";
    return false;
  }

  const auto out_path = std::filesystem::path(engine_path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  const auto tmp_path = engine_path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out) {
    std::cerr << "[TRT] failed to open engine output: " << tmp_path << '\n';
    return false;
  }
  out.write(static_cast<const char *>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
  out.close();
  if (!out) {
    std::cerr << "[TRT] failed to write engine output: " << tmp_path << '\n';
    return false;
  }
  std::filesystem::rename(tmp_path, engine_path);
  std::cout << "[TRT] wrote engine: " << engine_path << " ("
            << static_cast<double>(plan->size()) / (1024.0 * 1024.0)
            << " MiB)\n";
  return true;
}

bool build_detection_engine(const std::string &onnx_path,
                            const std::string &engine_path,
                            const DetectionProfile &profile) {
  if (!valid_profile(profile)) {
    std::cerr << "[TRT] invalid detection profile\n";
    return false;
  }
  if (!std::filesystem::exists(onnx_path)) {
    std::cerr << "[TRT] ONNX not found: " << onnx_path << '\n';
    return false;
  }

  BuildLogger logger;
  auto builder =
      std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "[TRT] failed to create builder\n";
    return false;
  }

  auto network =
      std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
  if (!network) {
    std::cerr << "[TRT] failed to create network\n";
    return false;
  }

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser ||
      !parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "[TRT] failed to parse ONNX: " << onnx_path << '\n';
    return false;
  }

  if (network->getNbInputs() != 1) {
    std::cerr << "[TRT] expected one detector input, got "
              << network->getNbInputs() << '\n';
    return false;
  }

  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cerr << "[TRT] failed to create builder config\n";
    return false;
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                             profile.workspace_bytes);
  config->setBuilderOptimizationLevel(profile.builder_optimization_level);
  if (profile.fp16 && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto *input = network->getInput(0);
  const char *input_name = input->getName();
  auto *trt_profile = builder->createOptimizationProfile();
  if (!trt_profile) {
    std::cerr << "[TRT] failed to create detector optimization profile\n";
    return false;
  }
  trt_profile->setDimensions(
      input_name, nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims4{profile.min_batch, profile.channels, profile.min_height,
                      profile.min_width});
  trt_profile->setDimensions(
      input_name, nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims4{profile.opt_batch, profile.channels, profile.opt_height,
                      profile.opt_width});
  trt_profile->setDimensions(
      input_name, nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims4{profile.max_batch, profile.channels, profile.max_height,
                      profile.max_width});
  if (!trt_profile->isValid()) {
    std::cerr << "[TRT] detector optimization profile is invalid\n";
    return false;
  }
  if (config->addOptimizationProfile(trt_profile) < 0) {
    std::cerr << "[TRT] failed to add detector optimization profile\n";
    return false;
  }

  std::cout << "[TRT] building detector: " << onnx_path << '\n'
            << "[TRT] input=" << input_name << '\n'
            << "[TRT] profile[0] min=(" << profile.min_batch << ",3,"
            << profile.min_height << ',' << profile.min_width << ") opt=("
            << profile.opt_batch << ",3," << profile.opt_height << ','
            << profile.opt_width << ") max=(" << profile.max_batch << ",3,"
            << profile.max_height << ',' << profile.max_width << ")\n";

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) {
    std::cerr << "[TRT] buildSerializedNetwork failed\n";
    return false;
  }

  const auto out_path = std::filesystem::path(engine_path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  const auto tmp_path = engine_path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out) {
    std::cerr << "[TRT] failed to open engine output: " << tmp_path << '\n';
    return false;
  }
  out.write(static_cast<const char *>(plan->data()),
            static_cast<std::streamsize>(plan->size()));
  out.close();
  if (!out) {
    std::cerr << "[TRT] failed to write engine output: " << tmp_path << '\n';
    return false;
  }
  std::filesystem::rename(tmp_path, engine_path);
  std::cout << "[TRT] wrote engine: " << engine_path << " ("
            << static_cast<double>(plan->size()) / (1024.0 * 1024.0)
            << " MiB)\n";
  return true;
}

} // namespace ppocrv6_native::engine
