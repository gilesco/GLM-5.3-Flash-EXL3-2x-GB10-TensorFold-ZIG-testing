# GLM-5.3-Flash NVFP4 on 2× GB10 — TensorFold v1 (testing)

> ## ⚠️ Testing line — significantly different from the production recipe
>
> This is **not** the EXL3/Python deployment. It serves GLM-5.3-Flash on two GB10s with
> TensorFold's **native CUDA engine** (first shipped in the **v1.0.x** line by
> [PR #804](https://github.com/ashhart/TensorFold/pull/804)) from **NVIDIA's NVFP4
> checkpoint** — a different engine, a different checkpoint and no patch stack. It is slower
> than production today, has no CUDA vision, and the family is still a preview upstream.
> Read [docs/DELTAS.md](docs/DELTAS.md) before pointing real traffic at it. Production stays
> [MiaAI-Lab's EXL3 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
> until [docs/VALIDATION.md](docs/VALIDATION.md) passes here.

Two DGX-Spark-class GB10 machines (here: `brain02` = head/rank 0, `brain03` = worker/rank 1)
serve one GLM-5.3-Flash instance, tensor-parallel over the ConnectX-7 link: rank 0 runs the
engine and the API, rank 1 holds its share of every axis and follows rank 0's steps over
NCCL. Everything is Docker; the hosts need nothing but docker, ssh and the NVIDIA driver.

```text
   brain02 (head)                                brain03 (worker)
   ┌─────────────────────────┐                   ┌─────────────────────────┐
   │ glm53-native-tf:<v1.x>  │                   │ glm53-native-tf:<v1.x>  │
   │ rank 0: serves :8000    │ ← CX7 2× rails →  │ rank 1: follows         │
   └────────────┬────────────┘                   └─────────────────────────┘
                │ OpenAI-style /v1 (chat/completions, models, tokenize, ...)
             clients (OpenAI-compatible)
```

> The default port is **8000** — deliberately the same as the production EXL3 stack, so
> clients keep one endpoint when switching engines. The price is that the two stacks are
> mutually exclusive: `start.sh` refuses a busy port, so stop one before starting the other.

## Why this exists

- **Upstream instead of a patch fork**: the EXL3 recipe carries ~106 patches against the
  Python engine and rebasing them is the slow part of every upgrade. This line runs upstream
  code at a pinned commit; a new v1.* release is a version bump and a rebuild.
- **The NVFP4 route is the upstream-supported GLM-on-CUDA path**: the family reads
  `nvidia/GLM-5.3-Flash-NVFP4` as stored (W4A16 with fp32 accumulate), drafts with the
  checkpoint's MTP layer, and admits memory across ranks.

## Status (receipts so far)

| What | Where it is recorded |
| --- | --- |
| Exactness (drafted=plain, resumed=fresh, concurrent=solo), speed, memory on 2× GB10, TP=2 | [PR #804](https://github.com/ashhart/TensorFold/pull/804)'s description — the PR author's pair |
| Lane checks at TP=2 **and TP=4** on the EXL3 glue branch | [#593](https://github.com/ashhart/TensorFold/issues/593) (BHCC2025) |
| Release-gate serve of the whole checkpoint from a release build | **not yet done upstream** (first four layers only) — which is why [docs/VALIDATION.md](docs/VALIDATION.md) exists |

## Requirements (this site)

- brain02 + brain03: GB10, driver 580.178.04, kernel 7.0.0-1019-nvidia, docker + nvidia
  runtime, key-based ssh head → worker (`ssh brain03` works), ~200 GB free disk each
  (the checkpoint is ~183 GB).
- The checkpoint already in `~/.cache/huggingface` **on both** Sparks (this site has
  revision `09b04e5e…`; see below for the alternatives).
- One CX7 cable per rail between the pair (this site: two rails, `192.168.2.0/30` and
  `192.168.2.4/30`); NCCL bootstraps on the head's rail address.
- The build runs on the head inside `nvcr.io/nvidia/pytorch:26.07-py3` (arm64, has nvcc and
  NCCL — already pulled); the resulting runtime image is ~a few hundred MB and is copied to
  the worker with `docker save | ssh docker load`.

## Quick start (on the head)

> **First launch on this site?** Follow [docs/LAUNCH.md](docs/LAUNCH.md) — it has the
> production coexistence rules (both stacks serve on 8000, so only one runs at a time; stop
> production first), the verified environment facts and the failure playbook.

```sh
cp scripts/local.sh.example scripts/local.sh   # adjust if the defaults below do not match
./scripts/prepare.sh      # resolve version, preflight, build + distribute the image
./scripts/start.sh        # rank 1 on the worker first, rank 0 here, waits for /health
./scripts/status.sh       # containers, /health, last log lines
./scripts/stop.sh         # stops both, saves gzipped logs
```

- `DRY_RUN=1 ./scripts/start.sh` prints both ranks' docker commands and changes nothing.
- `DOWNLOAD=1 ./scripts/prepare.sh` fetches the checkpoint into the head's HF cache (and
  rsyncs it to the worker) when a host lacks it.
- The first build compiles TensorFold (Zig + nvcc, sm_121) and takes a while; every later
  build with the same commit is a docker-cache hit.

Then point any OpenAI-compatible client at `http://<head>:8000/v1`, model name
`GLM-5.3-Flash-NVFP4` (`SERVED_NAME`): tool-eval-bench on the head (the test plan in
[docs/TOOL_EVAL_TESTING.md](docs/TOOL_EVAL_TESTING.md)) or a plain curl.
`context_length`-style settings take what validation shows fits (40,960 today).

## Versions: how the "latest v1.*" default works

`TF_REF` defaults to `latest-v1`: at prepare time the newest `v1.*` release is resolved to a
tag + commit and **recorded in the prepared marker**, so a running pair never drifts and a
restart does not silently upgrade. A new upstream release is picked up by re-running
`./scripts/prepare.sh` (or by the next `start.sh` after the marker is removed). To pin:

```sh
echo "TF_REF=v1.0.12" >> scripts/local.sh    # a tag; or TF_COMMIT=<sha> for an exact tree
```

The build args and the image tag carry the resolved commit; `docker build`'s cache keys on
it, so pinning and releasing coexist cleanly.

## Configuration

All settings live in `config.sh` (defaults) and `scripts/local.sh` / `.env` (this site's).
The ones that matter:

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKER` | `giles@brain03` | rank 1's ssh target (key-based) |
| `HEAD_FABRIC_ADDR` / `WORKER_FABRIC_ADDR` | `192.168.2.1` / `.2` | the CX7 rails; NCCL bootstraps on the head's |
| `NCCL_SOCKET_IFNAME` | `enp1s0f1np1,enP2p1s0f1np1` | netdevs NCCL may use (both rails) |
| `TF_REF` | `latest-v1` | the TensorFold line to track; or a `v1.0.x` tag / `TF_COMMIT` |
| `MODEL_ID` / `MODEL_REVISION` | `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e…` | the checkpoint; the ranks verify each other's bytes at start |
| `CONTEXT` / `PARALLEL` | `40960` / `4` | window and streams; the PR-qualified window is 40960 — larger is unmeasured (bf16 KV), admission refuses what does not fit |
| `DRAFTS` | `1` | `0` → `--no-drafts` (the serial reference; MTP is the only drafter) |
| `SERVED_NAME` / `PORT` / `HOST` | `GLM-5.3-Flash-NVFP4` / `8000` / `0.0.0.0` | API surface; the port is production's, so the stacks are mutually exclusive |
| `API_KEY` | empty | recommended; sent as `Authorization: Bearer` |
| `MEMORY_LIMIT_GB` | engine default | → `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` |
| `BASE_IMAGE` / `ZIG_VERSION` / `SM_ARCH` | NGC pytorch 26.07 / `0.17.0` / `121` | the builder |

## Checkpoint revisions

| Revision | What it is |
| --- | --- |
| `09b04e5e…` *(default)* | already in both Sparks' HF caches |
| `423acf37…` | the revision PR #804's receipts were measured on |
| `da920bb0…` | upstream main today (adds layer 45 to the ignore list) |

The engine reads tensors as stored (not the ignore list), so later revisions are expected to
work; the ranks cross-check every byte before serving, so a mixed pair stops loudly.

## Coexistence and rollback

Nothing here touches the production stack: different image (`glm53-native-tf:*` vs
`tensorfold-glm53:*`), different container name (`glm53-native-tf` vs `glm53-flash-tf`),
different kernel/state/log dirs. To go back: `./scripts/stop.sh`, then start production as
usual. To free the disk later: `docker rmi glm53-native-tf:<tag>` and delete the
`models--nvidia--GLM-5.3-Flash-NVFP4` blobs.

## Provenance

- Deployment shape and script conventions follow
  [MiaAI-Lab's EXL3 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (this repo started as its testing fork; the two are intentionally no longer the same
  deployment).
- The served engine is [ashhart/TensorFold](https://github.com/ashhart/TensorFold) at a
  pinned commit; the GLM CUDA family is PR
  [#804](https://github.com/ashhart/TensorFold/pull/804) (tracker:
  [#593](https://github.com/ashhart/TensorFold/issues/593)). Apache-2.0; see LICENSE, NOTICE
  and upstream's THIRD_PARTY_NOTICES.
- The checkpoint is `nvidia/GLM-5.3-Flash-NVFP4` (NVIDIA's ModelOpt export of GLM-5.3-Flash;
  its license and card apply).
