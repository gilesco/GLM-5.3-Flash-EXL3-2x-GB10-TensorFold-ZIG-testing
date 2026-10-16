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
RUN bash /build/build-native.sh

FROM ubuntu:24.04 AS runtime
COPY --from=build /out/ /opt/tensorfold/
RUN echo "/opt/tensorfold/lib" > /etc/ld.so.conf.d/tensorfold.conf \
    && ldconfig \
    && /opt/tensorfold/bin/tensorfold-native --version
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
