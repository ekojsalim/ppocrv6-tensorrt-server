ARG BASE_IMAGE=localhost/ppocrv6-vllm-ocr:trt10.14-native-base

FROM ${BASE_IMAGE} AS builder

ARG TRT_VERSION=10.14.1.48-1+cuda13.0
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

RUN cmake -S native -B native/build-vllm-overlay \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DPPOCRV6_NATIVE_BUILD_BENCH_TOOLS=OFF && \
    cmake --build native/build-vllm-overlay -j2 \
      --target ppocrv6_native_c_abi && \
    cargo build --release --locked --manifest-path server/Cargo.toml

FROM ${BASE_IMAGE}

COPY --from=builder \
  /workspace/native/build-vllm-overlay/libppocrv6_native_c_abi.so \
  /opt/ppocrv6/lib/libppocrv6_native_c_abi.so
COPY --from=builder \
  /workspace/server/target/release/ppocrv6-tensorrt-server \
  /usr/local/bin/ppocrv6-tensorrt-server
