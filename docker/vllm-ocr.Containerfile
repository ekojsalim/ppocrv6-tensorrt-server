ARG VLLM_IMAGE=docker.io/vllm/vllm-openai:v0.23.0
ARG TRT_VERSION=10.14.1.48-1+cuda13.0

FROM ${VLLM_IMAGE} AS ocr-runtime-base

ARG TRT_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get \
      -o Acquire::Retries=5 \
      -o Acquire::http::Timeout=60 \
      -o Acquire::https::Timeout=60 \
      install -y --no-install-recommends \
      "libnvinfer10=${TRT_VERSION}" \
      "libnvonnxparsers10=${TRT_VERSION}" \
      libopencv-core4.5d \
      libopencv-imgproc4.5d \
      libpolyclipping22; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ppocrv6/lib /models/ppocrv6-medium

FROM ocr-runtime-base AS ocr-builder

ARG TRT_VERSION
ARG RUST_VERSION=1.96.0
ARG CMAKE_VERSION=3.31.6
ARG CMAKE_CUDA_ARCHITECTURES=all-major

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:${PATH}

RUN set -eux; \
    apt-get update; \
    apt-get \
      -o Acquire::Retries=5 \
      -o Acquire::http::Timeout=60 \
      -o Acquire::https::Timeout=60 \
      install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      "libnvinfer-headers-dev=${TRT_VERSION}" \
      libopencv-core-dev \
      libopencv-imgproc-dev \
      libpolyclipping-dev \
      pkg-config \
      python3-pip; \
    mkdir -p /tmp/trt-dev; \
    cd /tmp/trt-dev; \
    apt-get download "libnvonnxparsers-dev=${TRT_VERSION}"; \
    dpkg-deb -x libnvonnxparsers-dev_*.deb /; \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvinfer.so.10 \
      /usr/lib/x86_64-linux-gnu/libnvinfer.so; \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10 \
      /usr/lib/x86_64-linux-gnu/libnvonnxparser.so; \
    cd /; \
    rm -rf /tmp/trt-dev; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "cmake==${CMAKE_VERSION}" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /workspace
COPY native ./native
COPY server ./server

RUN cmake -S native -B native/build-vllm-ocr \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DPPOCRV6_NATIVE_BUILD_BENCH_TOOLS=OFF && \
    cmake --build native/build-vllm-ocr -j2 --target ppocrv6_native_c_abi && \
    cargo build --release --locked --manifest-path server/Cargo.toml

FROM ocr-runtime-base AS ocr-runtime

# This keeps the vLLM image entrypoint intact. Start the OCR server explicitly,
# or use a deployment-specific process supervisor when serving both in one
# container.

COPY --from=ocr-builder \
  /workspace/native/build-vllm-ocr/libppocrv6_native_c_abi.so \
  /opt/ppocrv6/lib/libppocrv6_native_c_abi.so
COPY --from=ocr-builder \
  /workspace/server/target/release/ppocrv6-tensorrt-server \
  /usr/local/bin/ppocrv6-tensorrt-server

ENV PPOCRV6_NATIVE_LIB=/opt/ppocrv6/lib/libppocrv6_native_c_abi.so
ENV PPOCRV6_GLYPH_HOST=0.0.0.0
ENV PPOCRV6_GLYPH_ENGINE=/models/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt
ENV PPOCRV6_GLYPH_WEIGHT=/models/ppocrv6-medium/classifier/weight.fp16.bin
ENV PPOCRV6_GLYPH_BIAS=/models/ppocrv6-medium/classifier/bias.fp16.bin
ENV PPOCRV6_GLYPH_CHARACTERS=/models/ppocrv6-medium/classifier/characters.txt
ENV PPOCRV6_OCR_DET_ENGINE=/models/ppocrv6-medium/engines/det-fp16-b1-h256-1280-w256-1280.trt
ENV PPOCRV6_OCR_REC_ENGINE=/models/ppocrv6-medium/engines/rec-hidden-multiprofile-glyph-line.trt

EXPOSE 8184
