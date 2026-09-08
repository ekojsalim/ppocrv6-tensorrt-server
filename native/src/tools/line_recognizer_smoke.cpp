#include "ppocrv6_native/full_page/full_page_worker.h"
#include "ppocrv6_native/common/cuda_ptr.h"

#include <iostream>
#include <stdexcept>

using namespace ppocrv6_native;

int main(int argc, char **argv) {
  if (argc != 2) throw std::runtime_error("usage: line_recognizer_smoke MODEL_DIR");
  const std::string root = argv[1];
  full_page::FullPageWorkerConfig config;
  config.detector.engine_path = root + "/engines/det-fp16-b1-h256-1280-w256-1280.trt";
  config.detector.warmup_runs = 0;
  auto &rec = config.recognizer;
  rec.engine_path = root + "/engines/rec-hidden-multiprofile-glyph-line.trt";
  rec.weight_path = root + "/classifier/weight.fp16.bin";
  rec.bias_path = root + "/classifier/bias.fp16.bin";
  rec.characters_path = root + "/classifier/characters.txt";
  rec.warmup_runs = 0;
  auto worker = std::make_shared<recognition::RecognitionWorker>(rec);
  auto full = std::make_unique<full_page::FullPageWorker>(config, worker);
  cudaStream_t stream;
  PPOCRV6_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int cases = 0;
  for (int width : {128, 640, 960, 1280, 1600, 2400, 3200}) {
    for (int batch : {1, 12}) {
      std::vector<float> tensor(static_cast<std::size_t>(batch) * 3 * 48 * width);
      for (std::size_t i = 0; i < tensor.size(); ++i) {
        tensor[i] = static_cast<float>((i * 17 + i / width) % 255) / 127.5f - 1.0f;
      }
      CudaPtr<float> device(tensor.size());
      PPOCRV6_CUDA_CHECK(cudaMemcpy(device.get(), tensor.data(),
          tensor.size() * sizeof(float), cudaMemcpyHostToDevice));
      auto host_result = worker->recognize_f32(tensor.data(), batch, width, batch, true);
      // This is the same device-input entry point used by FullPageWorker.
      auto roi_result = worker->recognize_device_f32_on_stream(
          device.get(), batch, width, batch, true, stream);
      for (int i = 0; i < batch; ++i) {
        const auto &a = host_result.predictions[i];
        const auto &b = roi_result.predictions[i];
        if (a.decoded.text != b.decoded.text || a.decoded.score != b.decoded.score ||
            a.timestep_class_ids != b.timestep_class_ids ||
            a.timestep_scores != b.timestep_scores) {
          throw std::runtime_error("host/ROI recognition parity failed");
        }
      }
      ++cases;
    }
  }
  PPOCRV6_CUDA_CHECK(cudaStreamDestroy(stream));
  std::weak_ptr<recognition::RecognitionWorker> lifetime = worker;
  worker.reset();
  if (lifetime.expired()) throw std::runtime_error("full page did not retain worker");
  (void)full->info_json();
  full.reset();
  if (!lifetime.expired()) throw std::runtime_error("shared worker leaked");
  std::cout << "PASS normalized tensor parity cases=" << cases << ", shared lifetime\n";
}
