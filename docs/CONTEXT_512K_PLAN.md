# Test plan: raise the context to 512K (math first, then a ladder)

> **2026-10-23: GOAL MET — and over-delivered.** The first qualification campaign
> (`results/campaign-20261023-011329/`) served and qualified the top rung directly:
> **524288 × PARALLEL=4 on dense4fp8** (83.5 GiB model, 3.61 GiB/stream fp8, **15.8 GiB
> free** after reserve), with the tool-eval at **92/100 — identical to the
> production-parity dense8 cell** — and depth retrieval 12/12 up to 384k tokens. The
> iso-budget rungs below never needed climbing: fp8 KV made the top rung fit at 4×
> concurrency. The climb found the stream ceiling at **5** (6/8 refused by the engine's
> own 12.2 GiB-reserve arithmetic — `campaign-20261023-130428/`), and the depth-row
> "concurrency cliff" was withdrawn as a bench artifact (steady-state aggregate decode
> RISES with streams — TOOL_EVAL_TESTING.md's round-6 receipt). The remaining open
> thread is the KV-pages branch test (paged bf16 KV; round-8 items in
> TOOL_EVAL_TESTING.md).

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
- **2026-10-22: the v1.1.2 anchor recovers to 1.1.0's level** — matrix server lines
  measure **13.3 GiB free / 11.2 cache** on baseline (and **19.4 / 16.0 on dense4**);
  1.1.1's dip was transient (plausibly #993's bounded snapshot pages). The rungs below
  keep the conservative 11.0 anchor until the ladder itself re-measures.
- **2026-10-23: per-config anchors and the fp8 stream facts** (search-matrix server
  lines, `results/run-20261022-172325/`):
  - free-after-reserve / prompt cache: baseline **13.3/11.2**, dense4 **19.4/16.0**,
    dense4chan4 **21.3/16.0** (the NCCL channel set frees ~2 GiB of workspace but costs
    decode — see TOOL_EVAL_TESTING.md), fp8 **13.7/12.3**.
  - **Per-stream cost at the 40960 window: 0.71 GiB bf16 → 0.48 GiB fp8.** Splitting the
    delta: ~0.25 GiB fixed per stream + KV ≈ **11.8 KiB/token bf16, 5.9 fp8**. The fp8
    caches work as advertised (−32% stream bytes at 40960); their payoff scales with the
    window — at 524,288 a single bf16 stream's KV alone is ~5.9 GiB, fp8 ~2.9.
  - **Recomputed rungs** (iso-budget 524,288 stream tokens; plan rate 17.75 KiB/token
    beyond the base 163,840; reserve trim +4.2; 6 GiB overshoot margin — the plan's
    conservative rate kept; my 40960-split says the marginal cost is lower, so these
    are the pessimistic numbers):
    - rung cost bf16 ≈ **6.2 GiB** → usable after margin: baseline **5.3**, dense4 **11.3**
    - rung cost fp8 ≈ **3.1–4.5 GiB** (if the whole marginal rate is KV-proportional it
      halves to 8.9 KiB/token; the fixed share is unknown until the ladder measures it)
      → usable after margin: baseline **≥ 8.4**, dense4 **≥ 14.4**
  - **Verdict: every rung fits on every v1.1.2 config — the 512K constraint is quality
    (the gates), not memory.** dense4 makes each rung comfortable; dense4+fp8 doubles
    the slack again. The 1M×1 rung stays out of scope (goal cap) but would be borderline
    on dense4 bf16 (~1.8 usable) and easy on any fp8 config — noted only as headroom.
    The ladder script re-measures the true marginal rate at rung time; update this table
    from its admission lines.

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
- **524288** — the goal: needle found, decode rate recorded. **Measured, not predicted**:
  12/12 facts retrieved to 384k depth (fp8 KV); greedy solo ~48 t/s bench / ~54 t/s
  steady-state at 512K on dense4fp8 — the 1.1.0-era 24.0 solo was that version's number.

## 5. Decision rules

- A rung's needle misses, `peak` crowds the total, or MemAvailable dips under ~6 GiB under
  load → the ladder stops; the last passing rung is the tested maximum (recorded here and
  in the session log).
- **512K achieved (524,288)** → **decided (2026-10-23, round-2/3 receipts): the resting
  config is 524288 × PARALLEL=4 on dense4fp8** — all gates PASS, quality = the parity
  cell's, throughput leader in every cell, ~2 GiB cheaper than dense8. Five streams admit
  when a workload wants them (the engine's own arithmetic); the fp8-latent-KV route to 1M
  came UPSTREAM in v1.1.2 (`--kv-cache fp8`), so the old `patches/README.md` watch item
  is moot.
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
- [x] ~~131072 / 262144 rungs~~ — never needed: fp8 KV admitted the top rung directly
      (the plan's own iso-budget math; `campaign-20261023-011329/`).
- [x] 524288 (the goal) — qualified end-to-end at ×4: gates PASS, depth 12/12 to 384k,
      tool-eval 92, throughput tables in TOOL_EVAL_TESTING.md.
- [x] resting-config decision — **dense4fp8 @ 524288 × PARALLEL=4**; the fp8-latent-KV
      upstream ask was answered by upstream itself in v1.1.2.
