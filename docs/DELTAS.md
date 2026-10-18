# What differs from the EXL3 recipe, and why

This repo is a **testing line** for the native engine, next to
[MiaAI-Lab's GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(our production). Since 2026-10-18 both serve the **same checkpoint** — the difference is the
engine and the patch policy. Read this before pointing real traffic at it.

## The engine

| | EXL3 recipe (production) | this repo |
| --- | --- | --- |
| Engine | TensorFold **Python 0.6.x** in a PyTorch-based image, plus a ~106-patch recipe | TensorFold **native** (`tensorfold-native`, Zig, no Python) at a pinned **v1.\*** release (v1.1.0, `76aafcf2…`, as of 2026-10-18) |
| Checkpoint | Mia-AiLab EXL3 4 bpw (~176 GB) + DFlash2 drafter | **the same** Mia-AiLab EXL3 checkpoint + DFlash2 (`--drafter`); until 2026-10-18 this line served NVIDIA's NVFP4 (~183 GB) with MTP drafts — the v1.0.x route, kept as a `local.sh` fallback (upstream's known issue at 1.1.0: not run for the release, the weights leave no room for its streams on two GB10s) |
| Code carried locally | a patch series against TensorFold's site-packages | **none** — upstream code, built at a pinned commit |
| Upgrade path | rebase the patch series by hand (see the old repo's REBASE-NOTES) | bump `TF_REF` (or take the default `latest-v1`); prepare rebuilds the image |

The native GLM CUDA family landed upstream in PR
[#804](https://github.com/ashhart/TensorFold/pull/804) (tracked in
[#593](https://github.com/ashhart/TensorFold/issues/593)); the EXL3 two-rank reader and
DFlash2 drafter are #855/#976 and #979, first released in **v1.1.0**.

## Honest caveats (as of the v1.1.0 line)

- **Preview status**: the EXL3 two-rank path's exactness receipts (drafted=plain,
  concurrent=solo, resumed=fresh, ranks watching each other) are upstream's own GB10 pair,
  made for the 1.1.0 gate — not ours. Nothing has served from this repo's build yet;
  docs/VALIDATION.md is the gate to run before trusting it.
- **Speed: unmeasured here.** Upstream's release notes give no tok/s for EXL3-native
  two-rank. Anchors: production (Python engine) measures ~62–63 tok/s single-stream decode
  and ~2,050 tok/s prefill; this line's NVFP4-era baseline measured 27.9–33.5 tok/s drafted
  at `--context 40960 --parallel 2`. Expect to fill the VALIDATION table from scratch.
- **Vision: upstream has it, we have not run it.** 1.1.0 added `--vision` for GLM EXL3
  two-rank (images and video; video needs libtfvideo). Untested on this pair; the NVFP4
  route has no CUDA vision. Anything that sends images stays on production until validated.
- **DFlash2 via `--drafter`**, the path upstream measured, passed automatically when the
  snapshot is present; without it the server falls back to the MTP head (with a warning),
  and `DRAFTS=0` is the serial reference.
- **Memory fit is the open question.** The EXL3 weights are heavier than NVFP4's
  (BF16 linears vs W4A16 dense), and the NVFP4-era math (93.72 GiB/rank, 0.89 GiB/stream at
  40960) does not transfer. The admission system refuses what a rank cannot hold, naming it;
  the context ladder rules from 2026-10-16 (start at 131072, measure the overshoot) still
  apply.
- **TP=2 only**: the native ranks meet over `NCCL_COMM_ID`, and a third rank on a two-host
  cabled pair has no bootstrap interface. The old stack's `start-tp3.sh` stays Python-only.
- **The native-vs-Python gap is structural and upstream-known** (see docs/LAUNCH.md's session
  log, 2026-10-18): native loses ~20-45% decode and 1.7x cold prefill to the Python engine
  cross-backend ([ashhart/TensorFold#918](https://github.com/ashhart/TensorFold/issues/918),
  open since 1.0.3); prefill still blocks replies in native (the unported `--decode-share`
  row of [#840](https://github.com/ashhart/TensorFold/issues/840); the Metal analog is
  [#1032](https://github.com/ashhart/TensorFold/issues/1032)) — our d8192/c4 collapse
  (8.6 aggregate) is that defect; and on GB10 two engine threads spin per rank and heat the
  CPU zones ([#1033](https://github.com/ashhart/TensorFold/issues/1033)) — pin the containers
  to the efficiency cores (`CPUSET=0-4,10-14`) until upstream picks a remedy. The Python
  0.6.6 engine has its own TP=2 GLM decay bug
  ([#993](https://github.com/ashhart/TensorFold/issues/993)) — production is not immune
  either.

## What carries over (and is why this is still worth testing)

- Same topology and wiring: rank 0 on the head, rank 1 over ssh, NCCL over the two CX7 rails,
  the same docker run contract (`--network host`, `/dev/infiniband`, `IPC_LOCK`, memlock).
- The rank checks upstream added are strong: before serving, both ranks verify they run the
  same build, cut and **checkpoint bytes** (sha256 of every shard), and a dead rank stops the
  other within about a second.
- Admission across ranks: every rank's free memory is gathered after loading and the server
  admits by the smallest, naming a rank that refuses.
- Zero local patches: upstream fixes and speedups land by bumping the version, not by rebasing
  a 106-patch series by hand.
