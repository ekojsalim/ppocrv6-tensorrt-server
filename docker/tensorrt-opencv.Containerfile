ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:26.03-py3
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV RUSTUP_HOME=/opt/rustup
ENV CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:${PATH}

ARG RUST_VERSION=1.96.0
ARG CMAKE_VERSION=3.31.6
ARG CMAKE_CUDA_ARCHITECTURES=all-major

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      libopencv-dev \
      libpolyclipping-dev \
      python3-pip \
      pkg-config; \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "cmake==${CMAKE_VERSION}" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"

WORKDIR /workspace
COPY native ./native
COPY server ./server

RUN cmake -S native -B native/build-standalone \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DPPOCRV6_NATIVE_BUILD_BENCH_TOOLS=OFF && \
    cmake --build native/build-standalone -j2 --target ppocrv6_native_c_abi && \
    cargo build --release --locked --manifest-path server/Cargo.toml

FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      libopencv-core406 \
      libopencv-imgproc406 \
      libpolyclipping22; \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/ppocrv6/lib /models/ppocrv6-medium

COPY --from=builder \
  /workspace/native/build-standalone/libppocrv6_native_c_abi.so \
  /opt/ppocrv6/lib/libppocrv6_native_c_abi.so
COPY --from=builder \
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
ENTRYPOINT ["/usr/local/bin/ppocrv6-tensorrt-server"]
