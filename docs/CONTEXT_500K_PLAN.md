# Test plan: raise the context to 500K (math first, then a ladder)

**Goal:** move this stack from the default `--context 40960` to a **500,000-token
window** on the two-GB10 pair (brain02 rank 0, brain03 rank 1), with retrieval quality and
decode speed measured at every step.

> **2026-10-18:** the math below was derived on the **NVFP4 route** (93.72 GiB/rank model,
> 0.89 GiB/stream at 40960). The line has since pivoted to the EXL3 checkpoint (heavier
> weights, different memory model) — re-derive section 1 from the EXL3 server's startup line
> before trusting any rung. The ladder method stands.

**Rule:** math before restarts; a rung the admission refuses is a recorded outcome, not an
OOM. Results and session notes land as commits in this repo.

## 1. The memory math (from the engine's own startup line)

At `--context 40960 --parallel 4` the engine accounts itself as:

| Item | GiB |
| --- | --- |
| Model (per rank) | 93.72 |
| Streams: 4 × 0.89 | 3.56 |
| Prompt cache | 3.9 |
| Reserve | 12.2 |
| Unallocated | 6.5 |

- **Per-token KV ≈ 22.8 KiB** (0.89 GiB ÷ 40,960) — bf16 latent (`kv_lora_rank` 512) +
  indexer pools, **replicated on both ranks** (TP halves weights, not KV).
- **At 500K per stream: ≈ 10.9 GiB** (upper bound; ~10.3–10.7 once fixed per-stream buffers
  are discounted).

### What fits where

| Configuration | Stream cost | Verdict |
| --- | --- | --- |
| 500K, `--parallel 1`, defaults untouched | 10.9 | ~0.8 GiB short of the stream budget (10.06) |
| 500K, `--parallel 1`, reserve 12.2 → 8, prompt cache 3.9 → 1.5 | 10.9 | **fits, ~7 GiB slack** |
| 500K, `--parallel 2` | 21.7 | **does not fit** (~3+ GiB over even with both knobs trimmed) |
| Max context, `--parallel 1`, defaults | — | ~463K tokens |
| ~430–450K, `--parallel 2`, trimmed | ~19.5 | borderline — only worth a probe after 500K/p1 is proven |

**Conclusion: 500K is feasible at `--parallel 1` with two knobs. Multi-stream at 500K is
not, on bf16 KV** — the structural fix is an fp8 latent/indexer option upstream (the old
stack's 1M window ran on fp8 KV); that belongs on the tracker next to #971.

## 2. The knobs (all applied via config, no code paths)

| Knob | Value for the ladder | Where |
| --- | --- | --- |
| `CONTEXT` | the rung | env / local.sh |
| `PARALLEL` | 1 | env / local.sh |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 (from 12.2) | env — forwarded by `env_args()` |
| `--prompt-cache-gib` | 1.5 (from 3.9) | `PROMPT_CACHE_GIB` env → start.sh |

No reserve flag exists on the native binary; the env is the documented path. If the engine
ignores it, `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` is the fallback bound (also forwarded).

## 3. The ladder (scripts/context-ladder.sh)

Rungs: **131072 → 262144 → 400000 → 500000**, `--parallel 1`, knobs above. Per rung:

1. `stop.sh`, then `start.sh` with the rung's env — the admission either accepts (log line
   with the new memory model) or refuses (recorded, ladder continues).
2. Record `/health` (active/peak memory) and the server's startup line.
3. Probe retrieval + decode with the bench's needle test — one passphrase buried at half
   depth in a rung-length prompt: `tool-eval-bench run --needle-only --needle-lengths <rung>
   --needle-depths 2 --seed 42`. This both fills the window (the memory peak) and scores
   long-context quality.

Budget: ~10–15 min per rung (a ~4 min restart; the 500K needle prefills ~500K tokens at
~1.2–1.4K tok/s ≈ 6–8 min).

## 4. What each rung answers

- **131072 / 262144** — the native family's first long-context qualification at all
  (upstream has served nothing past 40,960). Compare with the Python engine's same-model
  curve: 50.9 tok/s @32k → 31.9 @262k (−40%); the native numbers land in
  `results/ctx-*/SUMMARY.md`.
- **400000** — the overshoot check: `peak` vs `active` in `/health` after a rung-length
  prefill (the old engine overshot its estimate by up to ~10 GiB at 1M-token prompts; if
  `peak` approaches the total, stop the ladder there).
- **500000** — the goal: needle found, decode rate recorded (expect roughly half the
  40960-token rate, ~13–17 tok/s drafted), memory within the trimmed budget.

## 5. Decision rules

- A rung's needle misses or `peak` crowds the total → the ladder stops; the last passing
  rung is the tested maximum (recorded here and in the session log).
- `500K` achieved at `--parallel 1` → decide explicitly: accept single-stream at long
  context (agent/bench workloads), or request fp8 KV upstream for multi-stream.
- The bench matrix (docs/TOOL_EVAL_TESTING.md) stays at 40,960 — quality and context are
  separate campaigns; rerun the matrix at 500K only after this ladder passes.

## 6. Risks

- The native family has never served past 40,960 — every rung is a first; a refusal or a
  crash is information, and `stop.sh`/`start.sh` recover either way.
- KDA's chunked recurrence and the indexer's pool scans are the parts most likely to behave
  differently at depth; the needle at half depth is their probe.
- Never trim the reserve to zero: prefill scratch and the ranks' CUDA context live there.

## Status

- [x] math (this document)
- [ ] 131072
- [ ] 262144
- [ ] 400000
- [ ] 500000
- [ ] decision + upstream ask (fp8 KV) drafted
