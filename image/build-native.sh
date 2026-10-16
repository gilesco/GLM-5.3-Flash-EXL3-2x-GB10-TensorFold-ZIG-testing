#!/usr/bin/env bash
# Runs inside the build stage of image/Dockerfile: installs Zig, checks out TensorFold at the
# pinned commit, builds the CUDA fatbins (sm_121 for the GB10) and the native server, and
# collects binary + libraries into /out. Every step echoes what it runs, so a failed first
# build on new hardware is readable in the docker build log.
set -euo pipefail

ZIG_VERSION="${ZIG_VERSION:?}"
TF_REPO="${TF_REPO:?}"
TF_COMMIT="${TF_COMMIT:?}"
SM_ARCH="${SM_ARCH:-121}"

echo "== tools present =="
command -v curl || { apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils git; }
command -v git || apt-get install -y --no-install-recommends git
NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
[[ -x "$NVCC" ]] || { echo "no nvcc in the builder image (BASE_IMAGE must be a CUDA devel image)" >&2; exit 1; }
"$NVCC" --version | tail -1
HOST_ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)"   # aarch64 on the GB10s
case "$HOST_ARCH" in aarch64|arm64) ZIG_TARGET="aarch64-linux-gnu"; ZIG_KEY="aarch64-linux" ;;
  x86_64|amd64) ZIG_TARGET="x86_64-linux-gnu"; ZIG_KEY="x86_64-linux" ;;
  *) echo "unsupported builder arch: $HOST_ARCH" >&2; exit 1 ;; esac

echo "== zig $ZIG_VERSION =="
if ! command -v zig >/dev/null; then
  URL="$(curl -s https://ziglang.org/download/index.json \
    | python3 -c "import json,sys;print(json.load(sys.stdin)['$ZIG_VERSION']['$ZIG_KEY']['tarball'])")"
  echo "zig from $URL"
  curl -sL "$URL" -o /tmp/zig.tar.xz
  mkdir -p /usr/local/zig && tar -xJf /tmp/zig.tar.xz -C /usr/local/zig --strip-components=1
  ln -sf /usr/local/zig/zig /usr/local/bin/zig
fi
zig version

echo "== checkout $TF_COMMIT =="
git clone "$TF_REPO" /src/TensorFold
git -C /src/TensorFold checkout --detach "$TF_COMMIT"
echo "checked out: $(git -C /src/TensorFold rev-parse HEAD)"

cd /src/TensorFold
echo "== fatbins (sm_$SM_ARCH) =="
zig build fatbins -Dtarget="$ZIG_TARGET" -Dnvcc="$NVCC" -Dsm="$SM_ARCH"
ls -la zig-out/fatbin/ || true

echo "== native server =="
# packaging/README.md: Linux targets use baseline CPU features; -Dsm/-Dnvcc select the GPU
# inputs. If a future build.zig renames those options, fall back to the plain target build.
zig build native -Dtarget="$ZIG_TARGET" -Dcpu=baseline -Dsm="$SM_ARCH" -Dnvcc="$NVCC" \
  || zig build native -Dtarget="$ZIG_TARGET" -Dcpu=baseline
BIN=/src/TensorFold/zig-out/native/bin/tensorfold-native
[[ -x "$BIN" ]] || { echo "build did not produce $BIN" >&2; exit 1; }
"$BIN" --version

echo "== collect /out =="
mkdir -p /out/bin /out/lib
cp "$BIN" /out/bin/
# NCCL: the hosts have no system libnccl and the engine opens it at run time; take the
# builder's. libcuda is deliberately NOT copied — at run time the host's driver provides it.
ldd "$BIN" > /out/ldd.txt || true
NCCL_SO="$(ldconfig -p | awk '/libnccl\.so\.2/{print $NF; exit}')"
if [[ -n "$NCCL_SO" && -e "$NCCL_SO" ]]; then
  cp -L "$NCCL_SO" /out/lib/libnccl.so.2
  echo "bundled $(readlink -f "$NCCL_SO") as /out/lib/libnccl.so.2"
else
  echo "WARNING: no libnccl.so.2 found in the builder image; rank startup over hosts will fail until one is bundled" >&2
  [[ -n "${ALLOW_NO_NCCL:-}" ]] || exit 1
fi
echo "== /out ready =="
ls -laR /out
