#!/usr/bin/env bash
# One-time (per version/config) preparation, run on the head Spark:
#   1. resolve the TensorFold version (latest v1.* release by default; see config.sh),
#      discovering scripts/local.sh first if the site has none (REDISCOVER=1 re-detects),
#   2. preflight both Sparks (docker, GPU, disk, link reachability, the checkpoint),
#   3. build the small runtime image (compiles tensorfold-native inside the build stage),
#   4. copy the image to the worker,
#   5. write the prepared marker start.sh compares against.
# Re-running is cheap: docker's cache makes an unchanged build a no-op. start.sh runs this
# automatically whenever the marker no longer matches config.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# --- 0. the site -----------------------------------------------------------------
# scripts/local.sh carries this site's identity. When it does not exist, discover
# it (the cabled CX7 rails, their addresses, the worker's ssh target) instead of
# making the user fill a template; REDISCOVER=1 re-runs detection on an existing
# file. config.sh must be sourced AFTER this: its guard dies on an empty WORKER.
if [[ ! -f scripts/local.sh || "${REDISCOVER:-0}" == 1 ]]; then
  REDISCOVER="${REDISCOVER:-0}" ./scripts/discover-site.sh
fi

source ./config.sh

# --- 1. resolve the version --------------------------------------------------------------
resolve_version() {
  if [[ -n "$TF_COMMIT" ]]; then
    TF_RESOLVED_REF="commit"; TF_RESOLVED_COMMIT="$TF_COMMIT"; return
  fi
  local tag
  if [[ "$TF_REF" == "latest-v1" ]]; then
    tag=$(curl -fsSL "https://api.github.com/repos/ashhart/TensorFold/releases?per_page=30" \
      | python3 -c "import json,sys;rs=json.load(sys.stdin);print(next(r['tag_name'] for r in rs if r['tag_name'].startswith('v1.')))" 2>/dev/null) \
      || tag=$(git ls-remote --tags "$TF_REPO" 'refs/tags/v1.*' | awk -F/ '{print $3}' | grep -v '\^{}' | sort -V | tail -1) \
      || die "could not resolve the latest v1.* release (set TF_REF=v1.0.12 or check the network)"
    log "latest v1.* release: $tag"
  else
    tag="$TF_REF"
  fi
  local sha
  sha=$(git ls-remote "$TF_REPO" "refs/tags/$tag" | awk '{print $1}')
  local peeled; peeled=$(git ls-remote "$TF_REPO" "refs/tags/$tag^{}" | awk '{print $1}')
  [[ -n "$peeled" ]] && sha=$peeled
  [[ -n "$sha" ]] || die "tag $tag not found in $TF_REPO"
  TF_RESOLVED_REF="$tag"; TF_RESOLVED_COMMIT="$sha"
}
resolve_version
IMAGE_REF="${IMAGE_NAME}:${TF_RESOLVED_REF}"
log "TensorFold $TF_RESOLVED_REF ($TF_RESOLVED_COMMIT); image $IMAGE_REF"

# --- 2. preflight ------------------------------------------------------------------------
command -v docker >/dev/null || die "docker is required on the head"
command -v git >/dev/null || die "git is required on the head (tag resolution)"
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || die "no GPU visible on the head"
free_g=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
(( free_g > 200 )) || warn "only ${free_g}G free on / — the EXL3 checkpoint is ~176 GB"
worker "command -v docker >/dev/null" || die "docker missing on $WORKER"
worker "nvidia-smi -L >/dev/null" || die "no GPU visible on $WORKER"

snapshot_present() { local h=$1; if [[ "$h" == local ]]; then
    [[ -e "$MODEL_SNAPSHOT/model.safetensors.index.json" ]]
  else worker "[[ -e '$MODEL_SNAPSHOT/model.safetensors.index.json' ]]"; fi; }
for host in local "$WORKER"; do
  if snapshot_present "$host"; then log "checkpoint $MODEL_ID@$MODEL_REVISION present on ${host/local/the head}"
  elif [[ "${DOWNLOAD:-0}" == 1 && "$host" == local ]]; then
    log "downloading $MODEL_ID@$MODEL_REVISION into $HF_CACHE (about 183 GB) ..."
    docker run --rm -u "$(id -u):$(id -g)" -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
      ${HF_TOKEN:+-e HF_TOKEN} python:3.12-slim sh -c \
      "pip install -q 'huggingface_hub[cli]' && hf download '$MODEL_ID' --revision '$MODEL_REVISION'"
  else
    die "checkpoint snapshot missing on ${host/local/the head}: $MODEL_SNAPSHOT
     get it with:  DOWNLOAD=1 scripts/prepare.sh     (downloads on the head, then rsyncs to the worker)"
  fi
done
if [[ "${DOWNLOAD:-0}" == 1 ]]; then
  log "copying the checkpoint to $WORKER (skipped when its cache already has it)"
  worker "[[ -e '$MODEL_SNAPSHOT/model.safetensors.index.json' ]]" || \
    rsync -a --info=progress2 "$HF_CACHE/hub/models--${MODEL_ID//\//--}/" \
      "$WORKER:$HF_CACHE/hub/models--${MODEL_ID//\//--}/"
fi

# --- 3. build the runtime image ----------------------------------------------------------
# Stage the optional patch overlay (config.sh's PATCHES; empty = pristine upstream).
mkdir -p image/patches
find image/patches -name '*.patch' -delete 2>/dev/null || true
if [[ -n "$PATCHES" ]]; then
  for _p in ${PATCHES//,/ }; do
    [[ -f "patches/$_p" ]] || die "PATCHES names '$_p' but patches/$_p does not exist"
    cp "patches/$_p" image/patches/
  done
  log "patch overlay: $(ls image/patches/*.patch 2>/dev/null | wc -l) patch(es) staged ($PATCHES)"
fi
log "building $IMAGE_REF (the first build compiles TensorFold; after that docker's cache answers)"
docker build -f image/Dockerfile -t "$IMAGE_REF" image/ \
  --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg ZIG_VERSION="$ZIG_VERSION" \
  --build-arg TF_REPO="$TF_REPO" --build-arg TF_COMMIT="$TF_RESOLVED_COMMIT" \
  --build-arg SM_ARCH="$SM_ARCH"
log "smoke: $IMAGE_REF --version"
docker run --rm "$IMAGE_REF" --version

# --- 4. distribute to the worker ---------------------------------------------------------
log "copying $IMAGE_REF to $WORKER (a small image; the builder stages stay here)"
docker save "$IMAGE_REF" | worker docker load
worker "docker run --rm $IMAGE_REF --version" || die "the image does not run on $WORKER"

# --- 5. markers --------------------------------------------------------------------------
mkdir -p "$STATE_DIR" "$LOG_DIR"
TF_RESOLVED_REF="$TF_RESOLVED_REF" TF_RESOLVED_COMMIT="$TF_RESOLVED_COMMIT" IMAGE_REF="$IMAGE_REF" \
  bash -c 'cd "'"$PWD"'"; source ./config.sh; prepared_line' > "$STATE_DIR/prepared" \
  || die "could not write the prepared marker"
bash -c 'cd "'"$PWD"'"; source ./config.sh; config_line' > "$STATE_DIR/prepared.cfg" \
  || die "could not write the config fingerprint"
worker "mkdir -p '$STATE_DIR' '$LOG_DIR'"
log "prepared: $(cat "$STATE_DIR/prepared")"
log "next: ./scripts/start.sh   (stops nothing; ./scripts/stop.sh first if a server runs)"
