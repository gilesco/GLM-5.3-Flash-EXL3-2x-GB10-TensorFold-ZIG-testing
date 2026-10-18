# GLM-5.3-Flash EXL3 on 2× GB10 — TensorFold native v1.1 (testing)

> ## ⚠️ Testing line — significantly different from the production recipe
>
> **2026-10-18 (v1.1.0): this line now serves Mia-AiLab's EXL3 checkpoint with the DFlash2
> drafter** (`--drafter`), the two-rank path upstream measured for the 1.1.0 release. The
> NVFP4 route below is kept working — the engine reads `config.json` and admits either
> checkpoint — but it is 1.1.0's known issue ("the weights leave no room for its streams
> beside the memory reserve" on two GB10s) and was not run upstream for the release; switch
> back with `MODEL_ID`/`MODEL_REVISION` in `scripts/local.sh`. The NVFP4 receipts in
> docs/DELTAS.md and docs/VALIDATION.md still describe that route.
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
- **Same checkpoint as production, upstream's engine**: since v1.1.0 (2026-10-18) this line
  serves Mia-AiLab's EXL3 checkpoint with the DFlash2 drafter — the two-rank GLM path
  upstream measured for that release (#855/#976, drafter in #979) — on the native CUDA
  engine. The NVFP4 route that first shipped here (PR #804, the v1.0.x line) stays one
  `local.sh` override away, but it is 1.1.0's known issue upstream (not run for the release:
  the weights leave no room for its streams beside the memory reserve on two GB10s).

## Status (receipts so far)

| What | Where it is recorded |
| --- | --- |
| Exactness (drafted=plain, resumed=fresh, concurrent=solo) of the **EXL3 two-rank + DFlash2** path on a GB10 pair, release build | upstream 1.1.0's release notes (#855/#976, #979) — upstream's pair, not ours |
| Exactness, speed, memory of the NVFP4 route on 2× GB10, TP=2 | [PR #804](https://github.com/ashhart/TensorFold/pull/804)'s description — the PR author's pair (the v1.0.x line; 1.1.0 did not re-run it) |
| Lane checks at TP=2 **and TP=4** on the EXL3 glue branch | [#593](https://github.com/ashhart/TensorFold/issues/593) (BHCC2025) |
| Our own serve of the whole checkpoint from the release build on **this** pair | **not yet done** — which is why [docs/VALIDATION.md](docs/VALIDATION.md) exists |

## Requirements (this site)

- brain02 + brain03: GB10, driver 580.178.04, kernel 7.0.0-1019-nvidia, docker + nvidia
  runtime, key-based ssh head → worker (`ssh brain03` works), ~200 GB free disk each
  (the EXL3 checkpoint is ~176 GB).
- The checkpoints already in `~/.cache/huggingface` **on both** Sparks (this site has the
  EXL3 default `078455ff…`, the DFlash2 drafter `bf582e4e…`, and the NVFP4 fallback
  `09b04e5e…`; see the revisions table below).
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
`GLM-5.3-Flash-EXL3` (`SERVED_NAME`): tool-eval-bench on the head (the test plan in
[docs/TOOL_EVAL_TESTING.md](docs/TOOL_EVAL_TESTING.md)) or a plain curl.
`context_length`-style settings take what validation shows fits (40,960 is the config
default; the EXL3 route is unmeasured on this pair at every window).

## Versions: how the "latest v1.*" default works

`TF_REF` defaults to `latest-v1`: at prepare time the newest `v1.*` release is resolved to a
tag + commit and **recorded in the prepared marker**, so a running pair never drifts and a
restart does not silently upgrade. A new upstream release is picked up by re-running
`./scripts/prepare.sh` (or by the next `start.sh` after the marker is removed). To pin:

```sh
echo "TF_REF=v1.1.0" >> scripts/local.sh    # a tag; or TF_COMMIT=<sha> for an exact tree
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
| `TF_REF` | `latest-v1` | the TensorFold line to track; or a `v1.x` tag / `TF_COMMIT` |
| `MODEL_ID` / `MODEL_REVISION` | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `078455ff…` | the checkpoint; the ranks verify each other's bytes at start |
| `DFLASH2_ID` / `DFLASH2_REVISION` | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e…` | the drafter passed as `--drafter` when `DRAFTS=1` and the snapshot is present (CC BY-NC-ND 4.0, non-commercial) |
| `CONTEXT` / `PARALLEL` | `40960` / `4` | window and streams; nothing past 40960 has served on this pair — climb per docs/VALIDATION.md |
| `DRAFTS` | `1` | `0` → `--no-drafts` (the serial reference); `1` drafts with DFlash2 (else the MTP head) |
| `SERVED_NAME` / `PORT` / `HOST` | `GLM-5.3-Flash-EXL3` / `8000` / `0.0.0.0` | API surface; the port is production's, so the stacks are mutually exclusive |
| `API_KEY` | empty | recommended; sent as `Authorization: Bearer` |
| `MEMORY_LIMIT_GB` | engine default | → `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` |
| `BASE_IMAGE` / `ZIG_VERSION` / `SM_ARCH` | NGC pytorch 26.07 / `0.17.0` / `121` | the builder |

## Checkpoint revisions

| Revision | What it is |
| --- | --- |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `078455ff…` *(default)* | production's checkpoint; already in both Sparks' HF caches |
| `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e…` | the drafter; already in both caches (non-commercial terms) |
| `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e…` | the NVFP4 fallback route (also cached; `423acf37…` is the revision PR #804's receipts were measured on) |

The engine reads `config.json` (`quant_method: "exl3"` vs ModelOpt NVFP4) and picks the
reader, so switching routes is a `local.sh` override, not a code change. The ranks
cross-check every byte before serving, so a mixed pair stops loudly.

## Coexistence and rollback

Nothing here touches the production stack: different image (`glm53-native-tf:*` vs
`tensorfold-glm53:*`), different container name (`glm53-native-tf` vs `glm53-flash-tf`),
different kernel/state/log dirs. To go back: `./scripts/stop.sh`, then start production as
usual. To free disk later: `docker rmi glm53-native-tf:<tag>`, and delete the
`models--nvidia--GLM-5.3-Flash-NVFP4` blobs **only** — never the `models--Mia-AiLab--GLM-5.3-Flash-EXL3-*`
ones: production serves from them.

## Provenance

- Deployment shape and script conventions follow
  [MiaAI-Lab's EXL3 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (this repo started as its testing fork; the two are intentionally no longer the same
  deployment).
- The served engine is [ashhart/TensorFold](https://github.com/ashhart/TensorFold) at a
  pinned commit; the GLM CUDA family first shipped with PR
  [#804](https://github.com/ashhart/TensorFold/pull/804) (tracker:
  [#593](https://github.com/ashhart/TensorFold/issues/593)); the EXL3 two-rank and DFlash2
  work is #855/#976 and #979, first released in v1.1.0. Apache-2.0; see LICENSE, NOTICE
  and upstream's THIRD_PARTY_NOTICES.
- The checkpoint is `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` (a byte-identical re-host
  of `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, ShapleyMCG License 1.0 — attribution asked); the
  drafter is `incoai/GLM-5.3-Flash-DFlash2` (CC BY-NC-ND 4.0, non-commercial). Their licenses
  and cards apply.
