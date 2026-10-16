# What differs from the EXL3 recipe, and why

This repo is a **testing line**, deliberately different from
[MiaAI-Lab's GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
(our production). Read this before pointing real traffic at it.

## The engine

| | EXL3 recipe (production) | this repo |
| --- | --- | --- |
| Engine | TensorFold **Python 0.6.x** in a PyTorch-based image, plus a ~106-patch recipe | TensorFold **native** (`tensorfold-native`, Zig, no Python) at a pinned **v1.\*** release |
| Checkpoint | Mia-AiLab EXL3 4 bpw (~176 GB) + DFlash2 drafter | **nvidia/GLM-5.3-Flash-NVFP4** (~183 GB), read as stored; drafts from the checkpoint's own MTP layer |
| Code carried locally | a patch series against TensorFold's site-packages | **none** — upstream code, built at a pinned commit |
| Upgrade path | rebase the patch series by hand (see the old repo's REBASE-NOTES) | bump `TF_REF` (or take the default `latest-v1`); prepare rebuilds the image |

The native GLM CUDA family landed upstream in PR
[#804](https://github.com/ashhart/TensorFold/pull/804) (tracked in
[#593](https://github.com/ashhart/TensorFold/issues/593)).

## Honest caveats (as of the v1.0.x line that first shipped it)

- **Preview status**: upstream's release notes list the GLM CUDA family among previews built
  from source, and its maintainer gate had served only the first four layers of the real
  checkpoint. The full-checkpoint receipts are the PR author's own (two GB10s, TP=2). We are
  early adopters; docs/VALIDATION.md is the gate to run here before trusting it.
- **Speed**: the PR measured, on two GB10s (`--context 40960 --parallel 2`, RoCE/NCCL),
  drafted **27.9–33.5 tok/s** and `"draft": false` **17.6 tok/s**; 4 concurrent streams
  52.8/68.0 tok/s aggregate; cold prompts ~1,440 tok/s at 8k (22.7 s at 32k). Our production
  stack measures ~62–63 tok/s single-stream decode and ~2,050 tok/s prefill. Expect roughly
  **half the decode and ~⅔ the prefill** until the native family matures.
- **No vision on CUDA**: GLM image/video input exists on Metal only in this line. The NVFP4
  checkpoint contains the tower, but the CUDA family does not load it. Anything that sends
  images stays on the old stack for now.
- **No DFlash2, no copy-drafts**: MTP only (the PR: ~1.9× over plain).
- **bf16 KV**: the family's gate keeps KV bf16; the old stack's fp8 latent cache is why it
  fits a 1M-token window. Large contexts on 128 GB are untested here — the PR qualified
  `--context 40960` (`--parallel 2`); the admission system refuses what a rank cannot hold,
  naming it.
- **TP=2 only**: the native ranks meet over `NCCL_COMM_ID`, and a third rank on this site's
  ring of /30s has no bootstrap interface; upstream's multi-host bootstrap fabric is a
  separate change. The old stack's `start-tp3.sh` stays Python-only.

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
