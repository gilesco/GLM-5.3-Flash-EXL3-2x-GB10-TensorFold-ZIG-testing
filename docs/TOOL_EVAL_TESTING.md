# Test plan: performance with quality (tool-eval-bench matrix)

**Question:** do the no-code optimizations (`--recall`, a sixth lane) help or hurt this
stack's core promise — tool-calling **quality** first, with **performance** measured by the
bench's `--perf` option on the same run?

**Rule of the campaign:** the bench options are **fixed and identical for every test**; the
server configuration is the only variable. Local commits only — **do not push to origin**
while this is a test.

## The matrix

| Label | `PARALLEL` | `RECALL` | What it isolates |
| --- | --- | --- | --- |
| `baseline` | 4 | off | today's default — the reference point |
| `recall` | 4 | on | Recall alone (drafts copy from previously-read text; verified-exact by design) |
| `parallel6` | 6 | off | the sixth lane alone (5.34 GiB of streams against 6.5 GiB free) |
| `recall6` | 6 | on | both — the candidate "best" configuration |

## Fixed bench command (identical in every run)

Run on brain02, against the local server:

```sh
~/tool-eval-bench/.venv/bin/tool-eval-bench run \
  --base-url http://127.0.0.1:8000/v1 \
  --model GLM-5.3-Flash-NVFP4 \
  --format openai \
  --seed 42 \
  --hardmode \
  --perf \
  --parallel 4 \
  --timeout 240 \
  --no-live \
  --label <label> \
  --output-dir results/<run-id>/<label> \
  --json-file results/<run-id>/<label>/result.json
```

Why each option is pinned:

- `--seed 42 --hardmode` — the full 97-scenario suite (69 standard + 28 Hard Mode),
  deterministic; the same shape production's anchor run used.
- `--perf` — the key ask: runs the bundled `llama-benchy` against the same endpoint for
  pp/tg throughput and latency, alongside the scenario scoring. Quality and speed come out
  of one run under one configuration.
- `--parallel 4` — the bench's own in-flight scenario count, held fixed so server-side lane
  differences are attributable to `PARALLEL`, not bench concurrency.
- `--timeout 240` — generous: this server decodes at ~20–28 tok/s with thinking on; short
  timeouts would fail scenarios on speed, not quality.
- `--no-live` — headless runs (the campaign is unattended); everything still lands in the
  JSON and logs.
- `--format openai --model GLM-5.3-Flash-NVFP4` — explicit; the served name is stable.

Thinking stays **on** (the server default) in every run — the quality path; `--no-think`
would confound quality comparisons.

## Procedure

```sh
cd ~/GLM-5.3-Flash-NVFP4-2x-GB10-TensorFold-v1-testing
scripts/run-bench-matrix.sh                 # all four, in order
scripts/run-bench-matrix.sh baseline recall # or any subset, in this order
NO_BENCH=1 scripts/run-bench-matrix.sh recall6   # restart + serve only (bring your own bench)
```

- Each configuration **restarts the server** (interrupts clients — run in a quiet window),
  waits for `/health` (~4 min), then benches.
- Budget: the bench itself is hours per configuration at this server's decode rate (97
  scenarios, thinking on, ~20–28 tok/s); plan an overnight run for all four.
- Every run leaves `results/<run-id>/<label>/` — the bench log, `result.json` and the
  server's startup line (its own memory model). `SUMMARY.md` collects the score/perf lines.
- After each run the script clears the server's recall record, so no configuration inherits
  another's state.

## Reading the results

| Metric | Source | Baseline to beat |
| --- | --- | --- |
| Scenario score | bench output, `result.json` | production anchor: **91/100, 177/194 points** (EXL3 stack, `--perf --hardmode --seed 42`) — external reference only; the matrix's internal baseline is `baseline` |
| pp / tg tok/s | `--perf` (llama-benchy) | production anchor: 1,829 pp / 63.0 tg single; 78.2 / 67.1 tg at 2 and 4 concurrent |
| Server memory model | `<label>.server.log` | model 93.72 GiB; streams at 0.89 GiB (4) / ~1.4 GiB (6) |

**Decision rule:** adopt a configuration when its scenario score is within noise of
`baseline` (±1 scenario / ~2 points) and its tg/pp or per-scenario wall time improves.
Recall is exact by construction, so its risks are only speed under a cold record and memory;
`parallel6` risks nothing but memory headroom (~1.2 GiB left at six streams — watch the
server's startup line and `/health`).

## Status

- [ ] code applied (RECALL support; local commit, not pushed)
- [ ] `baseline` run
- [ ] `recall` run
- [ ] `parallel6` run
- [ ] `recall6` run
- [ ] decision recorded here
