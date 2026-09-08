#pragma once
#include "ppocrv6_native/c_api/recognizer.h"
#include "ppocrv6_native/recognition/recognition_worker.h"
#include <memory>

// Shared ownership lets a full-page worker borrow the line recognizer safely,
// even when the original C handle is destroyed first.
struct ppocrv6_recognizer {
  std::shared_ptr<ppocrv6_native::recognition::RecognitionWorker> worker;
};
