# Test plan: performance first, then quality (phased)

**Strategy:** quick performance measurements select the best server configuration; the
scenario suites (tool-calling, Hard Mode) run only on the winner. **No `--hardmode` in
early testing.**

**Rule of the campaign:** within a phase, the bench options are **fixed and identical for
every test**; the server configuration is the only variable. Results and session notes land
as commits in this repo.

## The matrix (server configurations under test)

| Label | `PARALLEL` | `RECALL` | Extra | What it isolates |
| --- | --- | --- | --- | --- |
| `baseline` | 4 | off | — | today's default — the reference point |
| `pdl` | 4 | off | `PDL=1` | programmatic dependent launch on the mm kernels |
| `dense8` | 4 | off | `DENSE_BITS=8 HEAD_BITS=8` | dense/head packed to 8 bits at load (memory + packed mm tiles; exactness gates hold *within* the config) |
| `parallel6` | 6 | off | — | two more lanes (EXL3 model: 0.71 GiB/stream, 13.3 GiB free after reserve at p4) |
| `parallel8` | 8 | off | — | four more lanes |
| `recall` | 4 | on | — | Recall alone (drafts copy from previously-read text; verified-exact by design) |
| `recall6` | 6 | on | — | both — the candidate "best" configuration |

## The phases

### Phase 1 — quick performance (default; minutes per configuration, not hours)

`PHASE=perf scripts/run-bench-matrix.sh` (the default): the bench runs **`--perf-only`** —
llama-benchy pp/tg throughput and latency against the endpoint, no scenarios. All four
configurations in well under an hour including restarts.

Fixed options (identical every run): `--seed 42 --perf --parallel 4 --timeout 240
--no-live --format openai --model "$SERVED_NAME"` (the config default is
`GLM-5.3-Flash-EXL3`; the tokenizer follows `MODEL_ID`) against `http://127.0.0.1:8000/v1`.

**Outcome:** the fastest configuration whose latency/throughput profile fits the traffic
(single-stream tg, concurrent aggregate, pp).

### Phase 2 — quality on the winner only

`PHASE=tools scripts/run-bench-matrix.sh <winner>`: the 69 standard tool-calling scenarios
plus `--perf` in the same run — quality and speed from one configuration, no Hard Mode.
Score within noise of the configuration's Phase-1 expectations and of production's anchor
(91/100 was achieved **with** Hard Mode; the standard-only comparison is the 69-scenario
score).

### Phase 3 — Hard Mode, last

`PHASE=hardmode scripts/run-bench-matrix.sh <winner>`: adds the 28 Hard-Mode scenarios
(the production anchor's 91/100 / 177-194 points shape). Only run once Phase 2 passes.

## Procedure

```sh
cd ~/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing
PHASE=perf scripts/run-bench-matrix.sh                    # all four configurations
PHASE=perf scripts/run-bench-matrix.sh baseline recall    # or a subset
NO_BENCH=1 scripts/run-bench-matrix.sh recall6            # restart + serve only
```

- Each configuration **restarts the server** (interrupts clients — run in a quiet window),
  waits for `/health` (~4 min), then benches.
- Results land in `results/<run-id>/<label>/` (bench log, `result.json`, the server's own
  memory-model line); `SUMMARY.md` collects the score/perf lines.
- After each run the script clears the recall record, so no configuration inherits
  another's state.

## Reading the results

| Metric | Source | Anchor |
| --- | --- | --- |
| pp / tg tok/s, latency | `--perf` (llama-benchy) | production: 1,829 pp / 63.0 tg single; 78.2 / 67.1 tg at 2 and 4 concurrent (EXL3 stack — external reference) |
| Scenario score | Phase 2/3 output, `result.json` | production: 91/100, 177/194 points **with** Hard Mode |
| Server memory model | `<label>.server.log` | NVFP4-era anchor: model 93.72 GiB; streams 0.89 GiB at 4 / ~1.4 GiB at 6 — **re-derive from the EXL3 server's startup line** |

**Decision rule:** the Phase-1 winner = best tg/pp at acceptable latency; it earns Phase 2.
Phase 2's score within noise (±1 scenario / ~2 points) of expectation → it stands; a drop
→ fall back to `baseline` and re-test. Recall is exact by construction; `parallel6`'s only
risk is memory headroom (~1.2 GiB left — watch `<label>.server.log` and `/health`).

The context campaign (docs/CONTEXT_512K_PLAN.md) is separate and already performance-first
(needle probes); its ladder runs at `--parallel 1` and never touches the scenario suites.

## Status

- [x] RECALL / PROMPT_CACHE_GIB / knob support in the scripts
- [x] 7-config matrix run on **v1.1.0** (2026-10-18, `results/run-20261018-174541/`):
      **dense8 won** (solo 21.2 → 24.0 tok/s, −3.8 GiB model); pdl noise; parallel6/8 no
      gain; recall unrewarded by the fresh-prompt bench shape. Deep-prompt concurrency
      collapse (8.6 agg) structural — see docs/LAUNCH.md's session log.
- [x] ~~v1.1.1 re-baseline~~ superseded: v1.1.2 landed the same week (quick window measured
      1.1.1 greedy 24.7 / sampled 25.9 solo; 1.1.2 rebased over it).
- [x] **5-config matrix on v1.1.2 + RoCE (2026-10-22, `results/run-20261022-154406/`)** —
      every number moved; every restart came up on the RoCE gather (`2 HCA(s) … RDMA
      writes`, no fallback):

  | config | solo tg d0 / d4096 / d8192 | c4 agg d0 / d4096 | pp2048 | free / cache |
  |---|---:|---:|---:|---|
  | baseline | 36.4 / 38.9 / 35.1 | 37.8 / 19.6 | 1,026 | 13.3 / 11.2 GiB |
  | **dense4** | **55.4 / 54.7 / 51.4** | **42.6 / 21.1** | **1,053** | **19.4 / 16.0** |
  | dense6 | 47.1 / 47.0 / 47.2 | 41.3 / 20.6 | 1,042 | 18.4 / 16.0 |
  | dense8 | 43.8 / 41.0 / 43.0 | 40.3 / 20.1 | 1,032 | 17.4 / 15.3 |
  | pdl | 40.0 / 35.6 / 34.6 | 37.2 / 19.4 | 1,028 | 13.4 / 11.2 |

  vs v1.1.0 baseline (657 pp / 21.2 solo / 25.3 c4 / 8.6 c4@d8192): **prefill +56%,
  solo decode +72% (dense4 +161%), TTFT d0 3,117 → 1,995 ms**. Three changes bundled
  (engine 1.1.0→1.1.2, RoCE RDMA up, HCA set) — attribution needs a `TF_ROCE=0`
  control run. dense4's win is clean and monotonic (4 < 6 < 8 < 16 bits), unlike
  1.1.0's dense8-noise. The memory anchor recovered to 13.3/11.2 (1.1.1's 11.0/8.9
  was transient — likely #993's bounded snapshot pages).
- [ ] **Qualify dense4** (exactness gate + recall probes) — at 55.4 solo it approaches
      production Python's 62–63 and frees 6 GiB for the 512K ladder; 4-bit dense linears
      are a quality risk until the gate passes.
- [ ] Attribution control: `TF_ROCE=0` baseline (one restart, stock knob).
- [ ] Concurrency cliff remains: aggregate decode is ~flat solo→c4 (36.4 → 37.8) and
      dense4 *drops* (55.4 → 42.6); at d8192 c4 it is 13.5 agg. Diagnose with
      `TF_GLM_PROF=1` at c4@d4096.
- [ ] Corrected probes 3 (prefill ~14k) and 4 (exactness) on v1.1.2 baseline, then dense4.
- [ ] NCCL channel set (PASSTHROUGH=1, MIN/MAX=4) untested on the RoCE path.
- [ ] Phase 2/3 on the qualified winner.
