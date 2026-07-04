cmake_minimum_required(VERSION 3.24)

set(PPOCRV6_CUDA_ARCHITECTURES_WAS_SET_BY_USER OFF)
if(DEFINED CMAKE_CUDA_ARCHITECTURES OR DEFINED ENV{CUDAARCHS})
  set(PPOCRV6_CUDA_ARCHITECTURES_WAS_SET_BY_USER ON)
endif()

if(NOT PPOCRV6_CUDA_ARCHITECTURES_WAS_SET_BY_USER)
  find_program(PPOCRV6_NVIDIA_SMI nvidia-smi)
  set(PPOCRV6_DETECTED_CUDA_ARCHITECTURES "")
  if(PPOCRV6_NVIDIA_SMI)
    execute_process(
      COMMAND "${PPOCRV6_NVIDIA_SMI}"
              --query-gpu=compute_cap
              --format=csv,noheader
      RESULT_VARIABLE PPOCRV6_NVIDIA_SMI_RESULT
      OUTPUT_VARIABLE PPOCRV6_NVIDIA_SMI_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PPOCRV6_NVIDIA_SMI_RESULT EQUAL 0
       AND NOT PPOCRV6_NVIDIA_SMI_OUTPUT STREQUAL "")
      string(REPLACE "\n" ";" PPOCRV6_COMPUTE_CAPS "${PPOCRV6_NVIDIA_SMI_OUTPUT}")
      foreach(PPOCRV6_COMPUTE_CAP IN LISTS PPOCRV6_COMPUTE_CAPS)
        string(STRIP "${PPOCRV6_COMPUTE_CAP}" PPOCRV6_COMPUTE_CAP)
        string(REGEX REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1\\2"
               PPOCRV6_ARCH "${PPOCRV6_COMPUTE_CAP}")
        if(PPOCRV6_ARCH MATCHES "^[0-9]+$")
          list(APPEND PPOCRV6_DETECTED_CUDA_ARCHITECTURES "${PPOCRV6_ARCH}")
        endif()
      endforeach()
      list(REMOVE_DUPLICATES PPOCRV6_DETECTED_CUDA_ARCHITECTURES)
    endif()
  endif()

  if(PPOCRV6_DETECTED_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${PPOCRV6_DETECTED_CUDA_ARCHITECTURES}"
        CACHE STRING "CUDA architectures" FORCE)
    message(STATUS
            "CUDA architectures detected from nvidia-smi: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES all-major CACHE STRING "CUDA architectures" FORCE)
    message(STATUS
            "CUDA architectures defaulting to all-major; pass -DCMAKE_CUDA_ARCHITECTURES=<archs> to override")
  endif()
endif()

project(ppocrv6_native VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PPOCRV6_NATIVE_ENABLE_TENSORRT
       "Build TensorRT-dependent targets when TensorRT is available" ON)
option(PPOCRV6_NATIVE_ENABLE_OPENCV_POSTPROCESS
       "Use OpenCV and polyclipping for DB detector postprocess parity when available" ON)
option(PPOCRV6_NATIVE_BUILD_BENCH_TOOLS
       "Build optional native benchmark and diagnostic tools" OFF)
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation prefix, for example /usr/local/tensorrt")

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(PPOCRV6_NATIVE_ENABLE_OPENCV_POSTPROCESS)
  find_package(OpenCV QUIET COMPONENTS core imgproc)
  find_path(POLYCLIPPING_INCLUDE_DIR
    NAMES polyclipping/clipper.hpp clipper.hpp
    HINTS /usr/include /usr/local/include
  )
  find_library(POLYCLIPPING_LIBRARY
    NAMES polyclipping
    HINTS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
  )
  if(OpenCV_FOUND AND POLYCLIPPING_INCLUDE_DIR AND POLYCLIPPING_LIBRARY)
    set(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS ON)
    message(STATUS "OpenCV DB postprocess enabled: ${OpenCV_VERSION}")
  else()
    message(STATUS "OpenCV DB postprocess disabled; using dependency-light fallback")
  endif()
endif()

add_library(ppocrv6_native_core STATIC
  src/detection/db_postprocess.cpp
  src/recognition/ctc_decode.cpp
)
target_include_directories(ppocrv6_native_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(ppocrv6_native_core PUBLIC cxx_std_20)
target_compile_options(ppocrv6_native_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
if(PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS)
  target_compile_definitions(ppocrv6_native_core PUBLIC
    PPOCRV6_NATIVE_HAVE_OPENCV_POSTPROCESS=1
  )
  target_include_directories(ppocrv6_native_core PUBLIC
    "${POLYCLIPPING_INCLUDE_DIR}"
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(ppocrv6_native_core PUBLIC
    ${OpenCV_LIBS}
    "${POLYCLIPPING_LIBRARY}"
  )
endif()

add_library(ppocrv6_native_cuda STATIC
  src/kernels/roi_warp.cu
  src/kernels/classifier.cu
)
target_include_directories(ppocrv6_native_cuda PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(ppocrv6_native_cuda PUBLIC
  ppocrv6_native_core
  CUDA::cudart
)
target_compile_options(ppocrv6_native_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_executable(ppocrv6_native_smoke
  src/tools/native_smoke.cpp
)
target_link_libraries(ppocrv6_native_smoke PRIVATE
  ppocrv6_native_core
)

add_executable(ppocrv6_ctc_smoke
  src/tools/ctc_smoke.cpp
)
target_link_libraries(ppocrv6_ctc_smoke PRIVATE
  ppocrv6_native_core
)

add_executable(ppocrv6_roi_warp_smoke
  src/tools/roi_warp_smoke.cu
)
target_link_libraries(ppocrv6_roi_warp_smoke PRIVATE
  ppocrv6_native_cuda
)

add_executable(ppocrv6_roi_warp_file_smoke
  src/tools/roi_warp_file_smoke.cu
)
target_link_libraries(ppocrv6_roi_warp_file_smoke PRIVATE
  ppocrv6_native_cuda
)

add_executable(ppocrv6_classifier_smoke
  src/tools/classifier_smoke.cu
)
target_link_libraries(ppocrv6_classifier_smoke PRIVATE
  ppocrv6_native_cuda
)

if(PPOCRV6_NATIVE_ENABLE_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
      "${TENSORRT_ROOT}/include"
      "$ENV{TENSORRT_ROOT}/include"
      /usr/local/tensorrt/include
      /usr/include/x86_64-linux-gnu
      /usr/include
  )
  find_path(TENSORRT_ONNX_PARSER_INCLUDE_DIR NvOnnxParser.h
    HINTS
      "${TENSORRT_ROOT}/include"
      "$ENV{TENSORRT_ROOT}/include"
      /usr/local/tensorrt/include
      /usr/include/x86_64-linux-gnu
      /usr/include
  )
  find_library(TENSORRT_NVINFER_LIBRARY nvinfer
    HINTS
      "${TENSORRT_ROOT}/lib"
      "${TENSORRT_ROOT}/lib64"
      "$ENV{TENSORRT_ROOT}/lib"
      "$ENV{TENSORRT_ROOT}/lib64"
      /usr/local/tensorrt/lib
      /usr/local/tensorrt/lib64
      /usr/lib/x86_64-linux-gnu
  )
  find_library(TENSORRT_NVONNXPARSER_LIBRARY nvonnxparser
    HINTS
      "${TENSORRT_ROOT}/lib"
      "${TENSORRT_ROOT}/lib64"
      "$ENV{TENSORRT_ROOT}/lib"
      "$ENV{TENSORRT_ROOT}/lib64"
      /usr/local/tensorrt/lib
      /usr/local/tensorrt/lib64
      /usr/lib/x86_64-linux-gnu
  )

  if(TENSORRT_INCLUDE_DIR AND TENSORRT_ONNX_PARSER_INCLUDE_DIR AND TENSORRT_NVINFER_LIBRARY AND TENSORRT_NVONNXPARSER_LIBRARY)
    message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
    add_library(ppocrv6_native_trt STATIC
      src/engine/onnx_to_trt.cpp
      src/engine/trt_engine.cpp
      src/plugin/classifier_plugin.cpp
      src/detection/detection_worker.cpp
      src/full_page/full_page_worker.cpp
      src/recognition/recognition_worker.cpp
    )
    target_include_directories(ppocrv6_native_trt PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${TENSORRT_INCLUDE_DIR}"
      "${TENSORRT_ONNX_PARSER_INCLUDE_DIR}"
    )
    target_link_libraries(ppocrv6_native_trt PUBLIC
      ppocrv6_native_core
      ppocrv6_native_cuda
      CUDA::cudart
      "${TENSORRT_NVINFER_LIBRARY}"
      "${TENSORRT_NVONNXPARSER_LIBRARY}"
    )
    target_compile_features(ppocrv6_native_trt PUBLIC cxx_std_20)
    target_compile_options(ppocrv6_native_trt PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )

    add_executable(ppocrv6_trt_multi_profile_smoke
      src/tools/trt_multi_profile_smoke.cpp
    )
    target_link_libraries(ppocrv6_trt_multi_profile_smoke PRIVATE
      ppocrv6_native_trt
    )

    add_executable(ppocrv6_trt_detection_bench
      src/tools/trt_detection_bench.cpp
    )
    target_link_libraries(ppocrv6_trt_detection_bench PRIVATE
      ppocrv6_native_trt
    )

    add_executable(ppocrv6_detection_worker_smoke
      src/tools/detection_worker_smoke.cpp
    )
    target_link_libraries(ppocrv6_detection_worker_smoke PRIVATE
      ppocrv6_native_trt
    )

    add_executable(ppocrv6_full_page_worker_smoke
      src/tools/full_page_worker_smoke.cpp
    )
    target_link_libraries(ppocrv6_full_page_worker_smoke PRIVATE
      ppocrv6_native_trt
    )

    if(PPOCRV6_NATIVE_BUILD_BENCH_TOOLS)
      add_executable(ppocrv6_trt_hidden_inspect
        src/tools/trt_hidden_inspect.cpp
      )
      target_link_libraries(ppocrv6_trt_hidden_inspect PRIVATE
        ppocrv6_native_trt
      )

      add_executable(ppocrv6_trt_classifier_decode
        src/tools/trt_classifier_decode.cpp
      )
      target_link_libraries(ppocrv6_trt_classifier_decode PRIVATE
        ppocrv6_native_trt
        ppocrv6_native_cuda
      )

      add_executable(ppocrv6_trt_classifier_plugin_smoke
        src/tools/trt_classifier_plugin_smoke.cpp
      )
      target_link_libraries(ppocrv6_trt_classifier_plugin_smoke PRIVATE
        ppocrv6_native_trt
      )

      add_executable(ppocrv6_trt_multistream_recognition_bench
        src/tools/trt_multistream_recognition_bench.cpp
      )
      target_link_libraries(ppocrv6_trt_multistream_recognition_bench PRIVATE
        ppocrv6_native_trt
      )

      add_executable(ppocrv6_trt_full_recognition_bench
        src/tools/trt_full_recognition_bench.cpp
      )
      target_link_libraries(ppocrv6_trt_full_recognition_bench PRIVATE
        ppocrv6_native_trt
      )
    endif()

    add_library(ppocrv6_native_c_abi SHARED
      src/c_api/common.cpp
      src/c_api/detector_c_api.cpp
      src/c_api/full_page_c_api.cpp
      src/c_api/recognizer_c_api.cpp
    )
    target_link_libraries(ppocrv6_native_c_abi PRIVATE
      ppocrv6_native_trt
    )
    target_compile_features(ppocrv6_native_c_abi PRIVATE cxx_std_20)
    target_compile_options(ppocrv6_native_c_abi PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
  else()
    message(STATUS "TensorRT not found; skipping TensorRT-dependent targets")
  endif()
endif()
