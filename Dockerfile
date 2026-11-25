# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Multi-stage build.
#   docker build --target cpu-ci .   -> CPU-only build + full ctest + privacy audit (no ROCm, no devices)
#   docker build --target rocm .     -> build inside a ROCm development image for gfx906 hosts
# UNQUALIFIED GPU build candidate, not a compiler match or a supported image.
# Require full gfx906 HSACO/ISA and device qualification before publishing a digest.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4.1

FROM ubuntu:24.04 AS cpu-ci
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git python3 ca-certificates && \
    rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash builder
WORKDIR /src
COPY --chown=builder:builder . .
USER builder
RUN cmake -S . -B build-cpu -G Ninja -DVEGAJET_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build-cpu -j"$(nproc)" && \
    ctest --test-dir build-cpu --output-on-failure && \
    python3 -m unittest discover -s tests -p 'test_tools*.py' && \
    if [ -d .git ]; then tools/privacy_audit.sh; fi

FROM ${ROCM_IMAGE} AS rocm
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build python3 linux-libc-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# Host tools and planners; GPU code objects are built by the hsaco_* targets (see README "GPU handoff").
# linux-libc-dev is explicit: core/kfd_runtime.cpp includes <linux/kfd_ioctl.h>, and build-essential only
# *recommends* it, which --no-install-recommends drops.
RUN cmake -S . -B build-rocm -G Ninja -DVEGAJET_CPU_ONLY=OFF && cmake --build build-rocm -j
ENTRYPOINT ["/src/build-rocm/vegajet"]
