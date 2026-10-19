# Test plan: raise the context to 512K (math first, then a ladder)

**Goal:** move this stack from the default `--context 40960` to a **524,288-token window
(512K)** on the two-GB10 pair (head = rank 0, worker = rank 1), with retrieval quality and
decode speed measured at every step. **The ladder is iso-budget by design**: 131072×4,
262144×2 and 524288×1 are all exactly 524,288 stream tokens, so memory is constant across
every rung — only the distribution changes. Nothing above 524,288 is in scope (the
checkpoint's native 1M window needs fp8 latent KV; see the conclusion).

> **2026-10-19: the math below is EXL3-native**, from the v1.1.0 launch's startup line
> (2026-10-18 window) and the knob matrix's server lines. The NVFP4-era math it replaces is
> kept as an appendix at the bottom. The ladder method is unchanged.

**Rule:** math before restarts; a rung the admission refuses is a recorded outcome, not an
OOM. Results and session notes land as commits in this repo.

## 1. The memory math (EXL3-native, from the engine's own startup line)

At `--context 40960 --parallel 4` the engine accounts itself as (v1.1.0, DFlash2 loaded):

| Item | GiB |
| --- | --- |
| Model (per rank) | 86.11 (82.31 with `DENSE_BITS=8`) |
| Streams: 4 × 0.71 | 2.84 |
| Prompt cache | 11.2 (engine-sized; `PROMPT_CACHE_GIB` overrides) |
| Reserve | 12.2 |
| Free after the reserve | 13.3 — of which the cache took 11.2, **~2.1 truly unallocated** |

- **Per-token KV ≈ 17.75 KiB** (0.71 GiB ÷ 40,960) — the latent cache plus indexer pools,
  **replicated on both ranks** (TP halves weights, not KV). Cheaper per token than the
  NVFP4 route's 22.8 KiB, and `DENSE_BITS=8` frees a further 3.8 GiB of model.
- **Empirical anchor, not naive subtraction**: total − model − reserve leaves ~23 GiB on
  paper, but the startup line only showed 13.3 free — CUDA context, NCCL buffers, the
  drafter (0.57 GiB) and scratch eat the rest. **Extrapolate from the measured free line**,
  and treat the engine's estimate as optimistic: the NVFP4 ladder's 262144 rung ran ~6 GiB
  past its estimate (2026-10-16). **The overshoot margin below is 6 GiB for that reason.**
- **2026-10-19: v1.1.1's anchor is lower than 1.1.0's** — measured free-after-reserve
  **11.0 GiB** (cache 8.9) against 1.1.0's 13.3/11.2, i.e. **1.1.1 holds ~2.3 GiB more**
  (suspect the K-slice regrouping or the stepped-admission structures; unconfirmed). The
  table below uses the **v1.1.1 anchor**: every rung's usable drops ~2.3 GiB, and the
  verdicts survive — but re-measure the anchor on whatever version serves the ladder.

### What fits where (reserve trimmed 12.2 → 8: +4.2 GiB; prompt cache → 1.5)

Free-after-reserve at the rung = 11.0 (the v1.1.1 anchor) + 4.2 − (rung's total stream
tokens − 163,840) × 17.75 KiB. The **usable** column subtracts the 6 GiB overshoot margin.
Every ladder rung has the same 524,288 stream tokens — 9.2 GiB predicted, 3.2 usable.

| Configuration | Stream tokens | Predicted free | Usable (− 6 GiB margin) | Verdict |
| --- | --- | --- | --- | --- |
| 131072 × `--parallel 4` | 524,288 | 9.2 GiB | 3.2 | **fits — rung 1** |
| 262144 × `--parallel 2` | 524,288 | 9.2 GiB | 3.2 | **fits — rung 2** |
| 262144 × `--parallel 4` | 1,048,576 | 0.5 GiB | negative | **does not fit** |
| **524288 × `--parallel 1`** | 524,288 | 9.2 GiB | 3.2 | **fits — the goal** |
| 1048576 × `--parallel 1` | 1,048,576 | 0.5 GiB | negative | **does not fit on bf16 latent** |
| 1048576 × 1, `DENSE_BITS=8` | 1,048,576 | 4.3 GiB | negative | **does not fit** (the 1.1.0 anchor's 6.6 was borderline; 1.1.1's lower anchor ends it) |

**Conclusion: the 512K goal fits the iso-budget ladder at every distribution — 131072×4,
262144×2, 524288×1 — with ~3.2 GiB of margin past the measured overshoot.** The
checkpoint's native 1M window does **not** fit on the bf16 latent cache (on either
anchor): the structural fix is fp8 latent KV for GLM CUDA (production reaches 1M×4 with
fp8 KV; the patch candidate is first on `patches/README.md`'s watchlist).

## 2. The knobs (all applied via config, no code paths)

| Knob | Value for the ladder | Where |
| --- | --- | --- |
| `CONTEXT` | the rung | env / local.sh |
| `PARALLEL` | 1 (262144 also at 2) | env / local.sh |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 (from 12.2) | env — forwarded by `env_args()` |
| `PROMPT_CACHE_GIB` | 1.5 (v1.1.1's default is 8.9; 1.1.0's was 11.2) | env → start.sh's `--prompt-cache-gib` |
| `DENSE_BITS=8 HEAD_BITS=8` | optional +3.8 GiB | env — the knob-matrix winner |

If the reserve env is ignored, `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` is the fallback bound
(also forwarded).

## 3. The ladder (scripts/context-ladder.sh)

Rungs: **131072 → 262144 → 524288** — all iso-budget (524,288 stream tokens each).
131072 and 262144 may also probe ×2/×4 once ×1 is proven (the ×4 probe at 131072 doubles as
1.1.1's batched-fresh-prompts memory check). Per rung:

1. `stop.sh`, then `start.sh` with the rung's env — the admission either accepts (log line
   with the new memory model) or refuses (recorded, ladder continues).
2. Record `/health` (active/peak memory) **and `MemAvailable` on both hosts under load** —
   the engine's estimate has run ~6 GiB past reality once already; MemAvailable is the
   ground truth.
3. Probe retrieval + decode with the bench's needle test — one passphrase buried at half
   depth in a rung-length prompt: `tool-eval-bench run --needle-only --needle-lengths <rung>
   --needle-depths 2 --seed 42`. This both fills the window (the memory peak) and scores
   long-context quality.

Budget: ~10–15 min per rung (a ~4 min restart with the warm digest cache; the 524288
needle prefills ~524K tokens at ~1.4–2K tok/s ≈ 4–6 min).

## 4. What each rung answers

- **131072 / 262144** — the EXL3 route's first long-context qualification (nothing past
  40,960 has served on this pair). Compare with the Python engine's same-model curve:
  50.9 tok/s @32k → 31.9 @262k (−40%); the native numbers land in `results/ctx-*/SUMMARY.md`.
- **The overshoot check runs at every rung**: `peak` vs `active` in `/health` after a
  rung-length prefill, cross-checked against MemAvailable (the old engine overshot its
  estimate by up to ~10 GiB at 1M-token prompts; if `peak` approaches the total, stop the
  ladder there).
- **524288** — the goal: needle found, decode rate recorded. With `TEMPERATURE=0` and
  DFlash2 drafting, expect the greedy rate measured in the knob matrix (24.0 solo at 40960
  on 1.1.0) less the depth penalty — record, don't predict.

## 5. Decision rules

- A rung's needle misses, `peak` crowds the total, or MemAvailable dips under ~6 GiB under
  load → the ladder stops; the last passing rung is the tested maximum (recorded here and
  in the session log).
- **512K achieved (524,288)** → decide explicitly which distribution is the resting config
  (524288×1 for solo agents; 262144×2 or 131072×4 for concurrent traffic), and take the
  fp8-latent-KV route only if 1M-class windows ever become a requirement (production's
  1M×4; for us a patch — see `patches/README.md`) or an upstream ask.
- The bench matrix (docs/TOOL_EVAL_TESTING.md) stays at 40,960 — quality and context are
  separate campaigns; rerun the matrix at 524288 only after this ladder passes.

## 6. Risks

- Nothing past 40,960 has served on this pair on any checkpoint — every rung is a first; a
  refusal or a crash is information, and `stop.sh`/`start.sh` recover either way.
- The NVFP4 ladder's overshoot (engine ~6 GiB optimistic at 262144) is the reason for the
  margin column; KDA's chunked recurrence and the indexer's pool scans are the parts most
  likely to behave differently at depth — the needle at half depth is their probe.
- Never trim the reserve to zero: prefill scratch and the ranks' CUDA context live there.
- 1.1.1's batched fresh prompts change how concurrent prefills share the window — the
  131072×4 rung doubles as their first memory check.

## Appendix: the NVFP4-era math (2026-10-18, superseded — kept for the comparison)

NVFP4-native (v1.0.12): model 93.72 GiB/rank, 0.89 GiB/stream at 40960 (**22.8 KiB/token**),
reserve 12.2, prompt cache 3.9 → 500K ≈ 10.9 GiB/stream: feasible at p1 with both knobs
trimmed (~7 GiB slack), not at p2. The EXL3 route is *cheaper per token* (17.75 vs 22.8)
with a *smaller model* (86.11 vs 93.72) — the pivot made every rung easier, not harder.
#1040's NVFP4 run at 245760×1 (3.16 GiB/stream ≈ 13.7 KiB/token) is consistent with that.

## Status

- [x] math (this document, EXL3-native 2026-10-19; goal 524,288 = 512K, iso-budget ladder)
- [ ] 131072 (×1 proven, then ×2/×4 probe)
- [ ] 262144 (×1/×2)
- [ ] 524288 (the goal)
- [ ] resting-config decision + upstream ask (fp8 latent KV) drafted
