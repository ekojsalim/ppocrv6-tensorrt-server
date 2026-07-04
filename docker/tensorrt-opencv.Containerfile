ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:26.03-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libopencv-dev \
      libpolyclipping-dev \
      pkg-config && \
    rm -rf /var/lib/apt/lists/*
