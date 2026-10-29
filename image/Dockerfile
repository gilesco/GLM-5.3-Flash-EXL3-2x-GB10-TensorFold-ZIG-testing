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

# The vision decode libraries at the engine's PINNED versions: the engine dlsyms the tj3
# API and version-checks what it loads, so distro libjpeg-turbo (2.x) and libwebp (1.3.x)
# do not carry it (the round-52/53 receipts: "libturbojpeg.so.0 is required for JPEG
# inputs (--vision); install libjpeg-turbo 3.1.4.1..."). Built here and named to the
# sonames the engine searches for in ../lib (its loader also accepts
# TENSORFOLD_LIBTURBOJPEG / TENSORFOLD_LIBWEBP pointing at a file):
#   libturbojpeg.so.0   <- libjpeg-turbo 3.1.4.1 (JPEG inputs)
#   libwebpdecoder.so.3 <- libwebp 1.6.0 (WebP inputs)
# /out/lib ships into the runtime stage's /opt/tensorfold/lib, which is both the image's
# loader path and the engine's ../lib search location.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nasm ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/jt.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.4.1.tar.gz \
    && mkdir /tmp/jt && tar -xzf /tmp/jt.tar.gz -C /tmp/jt --strip-components=1 \
    && cmake -S /tmp/jt -B /tmp/jt/b -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_SHARED=1 -DENABLE_STATIC=0 -DWITH_JAVA=0 -DWITH_TURBOJPEG=1 \
    && cmake --build /tmp/jt/b -j"$(nproc)" --target turbojpeg \
    && cp "$(find /tmp/jt/b -name 'libturbojpeg.so*' -type f | head -1)" /out/lib/libturbojpeg.so.0 \
    && curl -fsSL -o /tmp/wp.tar.gz https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.tar.gz \
    && mkdir /tmp/wp && tar -xzf /tmp/wp.tar.gz -C /tmp/wp --strip-components=1 \
    && cmake -S /tmp/wp -B /tmp/wp/b -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=1 \
    && cmake --build /tmp/wp/b -j"$(nproc)" --target webpdecoder \
    && cp "$(find /tmp/wp/b -name 'libwebpdecoder.so*' -type f | head -1)" /out/lib/libwebpdecoder.so.3 \
    && rm -rf /tmp/jt /tmp/jt.tar.gz /tmp/wp /tmp/wp.tar.gz

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
# The vision chat template: the checkpoint's own template is text-only — the EXL3
# conversion line's media branch emits a "you are unable to process this image" reminder
# instead of an image path (both the AZ-Focus requant AND the stock Mia EXL3 ship it).
# Production's images work anyway because the PYTHON engine routes media parts through
# the HF processor (which writes the markers itself); the native engine delegates the
# media branch to the template, so without an image path every image request 500s
# (receipt: campaign-20261028-154906). This template is the checkpoint's own with ONLY
# the media branch replaced — generated from the tokenizer's exact media-token bytes
# (they carry zero-width characters) by gen-vision-template.py; start.sh passes it via
# --chat-template under VISION=1.
COPY share/glm5-next-vision.jinja /opt/tensorfold/share/
COPY share/glm5-next-effort-tail.jinja /opt/tensorfold/share/
COPY --from=build /out/ /opt/tensorfold/
RUN echo "/opt/tensorfold/lib" > /etc/ld.so.conf.d/tensorfold.conf \
    && ldconfig \
    && /opt/tensorfold/bin/tensorfold-native --version
ENTRYPOINT ["/opt/tensorfold/bin/tensorfold-native"]
