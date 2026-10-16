# Test plan: performance first, then quality (phased)

**Strategy:** quick performance measurements select the best server configuration; the
scenario suites (tool-calling, Hard Mode) run only on the winner. **No `--hardmode` in
early testing.**

**Rule of the campaign:** within a phase, the bench options are **fixed and identical for
every test**; the server configuration is the only variable. Local commits only — **do not
push to origin** while this is a test.

## The matrix (server configurations under test)

| Label | `PARALLEL` | `RECALL` | What it isolates |
| --- | --- | --- | --- |
| `baseline` | 4 | off | today's default — the reference point |
| `recall` | 4 | on | Recall alone (drafts copy from previously-read text; verified-exact by design) |
| `parallel6` | 6 | off | the sixth lane (5.34 GiB of streams against 6.5 GiB free) |
| `recall6` | 6 | on | both — the candidate "best" configuration |

## The phases

### Phase 1 — quick performance (default; minutes per configuration, not hours)

`PHASE=perf scripts/run-bench-matrix.sh` (the default): the bench runs **`--perf-only`** —
llama-benchy pp/tg throughput and latency against the endpoint, no scenarios. All four
configurations in well under an hour including restarts.

Fixed options (identical every run): `--seed 42 --perf --parallel 4 --timeout 240
--no-live --format openai --model GLM-5.3-Flash-NVFP4` against `http://127.0.0.1:8000/v1`.

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
cd ~/GLM-5.3-Flash-NVFP4-2x-GB10-TensorFold-v1-testing
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
| Server memory model | `<label>.server.log` | model 93.72 GiB; streams 0.89 GiB at 4 / ~1.4 GiB at 6 |

**Decision rule:** the Phase-1 winner = best tg/pp at acceptable latency; it earns Phase 2.
Phase 2's score within noise (±1 scenario / ~2 points) of expectation → it stands; a drop
→ fall back to `baseline` and re-test. Recall is exact by construction; `parallel6`'s only
risk is memory headroom (~1.2 GiB left — watch `<label>.server.log` and `/health`).

The context campaign (docs/CONTEXT_500K_PLAN.md) is separate and already performance-first
(needle probes); its ladder runs at `--parallel 1` and never touches the scenario suites.

## Status

- [x] code applied (RECALL + PROMPT_CACHE_GIB support; local commits, not pushed)
- [ ] Phase 1: `baseline` / `recall` / `parallel6` / `recall6` perf-only runs
- [ ] Phase 1 winner chosen
- [ ] Phase 2: standard scenarios on the winner
- [ ] Phase 3: Hard Mode on the winner (the production-anchor comparison)
