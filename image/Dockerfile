# Builds the tensorfold-native runtime image for the GB10s (aarch64, sm_121).
#
# Stage 1 compiles TensorFold's native CUDA engine at a pinned commit with the Zig version
# TensorFold pins (.zig-version) and the CUDA toolkit of the base image. The default base
# (nvcr.io/nvidia/pytorch:26.07-py3) is an arm64 CUDA-devel image that already ships nvcc and
# NCCL and is present on the build host; any arm64 CUDA 13.x devel base works.
#
# Stage 2 is a small runtime image: the binary, the libraries it needs beyond the base
# (NCCL — the hosts have no system libnccl), and nothing else. GPU access comes from the host
# driver through --gpus all at run time; the builder's libcuda stub is never copied.
#
# Build (scripts/prepare.sh does this):
#   docker build -f image/Dockerfile -t glm53-native-tf:<tag> \
#     --build-arg BASE_IMAGE=... --build-arg ZIG_VERSION=... \
#     --build-arg TF_COMMIT=... --build-arg SM_ARCH=121 image/

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3

FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION=0.17.0
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_COMMIT
ARG SM_ARCH=121
COPY build-native.sh /build/build-native.sh
COPY patches/ /build/patches/
RUN bash /build/build-native.sh

FROM ubuntu:24.04 AS runtime
# RDMA user space (libibverbs + the mlx5 provider): NCCL's IB transport and the engine's
# RoCE gather both dlopen libibverbs.so.1 at run time. Without it NCCL falls back to
# socket transport and the RoCE gather logs "VerbsUnavailable" and never comes up
# (ashhart/TensorFold#852's gather; the #971 media path needs it too).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers rdma-core \
    && rm -rf /var/lib/apt/lists/*
# CUDA 13 cuBLAS runtime (libcublasLt.so.13 + libcublas.so.13): the vision tower's
# projections run cuBLASLt, and this runtime base is driver-only — the engine refuses to
# serve --vision without it ("install the NVIDIA CUDA 13 cuBLAS runtime; a driver-only
# host does not include it"; receipt: campaign-20261028-151636/25-visionen.md). Copied
# from the builder stage's CUDA 13 toolkit (the same NVCR base production runs);
# /opt/tensorfold/lib is already on the loader path below. Production's parity line:
# a 1.05 GiB tower with 1.41 GiB of workspace, 2048 tokens a picture, 16384 a clip.
COPY --from=build /usr/local/cuda/targets/sbsa-linux/lib/libcublasLt.so.13 \
     /usr/local/cuda/targets/sbsa-linux/lib/libcublas.so.13 \
     /opt/tensorfold/lib/
COPY --from=build /out/ /opt/tensorfold/
RUN echo "/opt/tensorfold/lib" > /etc/ld.so.conf.d/tensorfold.conf \
    && ldconfig \
    && /opt/tensorfold/bin/tensorfold-native --version
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
