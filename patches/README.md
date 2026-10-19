# Patch overlay policy

This directory carries **optional local patches** applied to the TensorFold tree at image
build. `config.sh`'s `PATCHES` knob selects them (empty = pristine upstream, the default and
the tested configuration); `scripts/prepare.sh` stages the named files into `image/patches/`
and the Docker build applies them right after the TensorFold checkout.

## The rules

1. **Upstream first.** A patch exists here only while upstream lacks the feature (check
   #840's carry-over tracker and the latest release notes). The moment an upstream release
   covers it, the patch is dropped and the version bump takes over.
2. **A receipt or it doesn't go in.** Each patch carries, in this file or the patch header:
   what it changes, the measured effect (config, hardware, cell-by-cell numbers from
   `run-bench-matrix.sh` or the engine's own counters), the exact TreeFold commit it was
   ported against, and who owns rebasing it at the next bump.
3. **One concern per patch.** A patch that bundles several changes cannot be measured or
   dropped independently.
4. **Rebase cost is the bill.** The ecosystem's 77-patch stack (jayleaton's repo) targets
   TensorFold 0.3.4 and buys ~3% over its own control — their headline numbers come from
   configuration, not patches. Port a patch only when its measured win is worth the
   rebase tax on every future version bump.

## Candidates under watch (from the ecosystem review, 2026-10-19)

- **fp8 latent KV for GLM CUDA** (jayleaton 0220) — halves KV bytes; upstream has fp8 KV for
  the Qwen family only (`--kv-cache fp8`).
- **16-row verify windows** (jayleaton 0380) — more tokens per round; upstream's GLM window
  policy is per-release.
- **L2 weight prefetch in decode** (jayleaton 0460).
- **RoCE all-gather wiring** (jayleaton 0230/0350) — upstream has the kernel (#852) but no
  family calls it; their measured prefill gain was small (+1.5% was the NCCL channel set,
  which is already adoptable via env).

None of these are carried yet. Measure stock v1.1.1 + `TEMPERATURE=0` + the NCCL set first —
if the greedy cells land near the ecosystem's numbers, none of these is worth its tax.
