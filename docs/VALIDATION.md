# Validation gate (run when brain02/brain03 are free)

Nothing here is trusted until these pass on our own pair. The commands assume the server from
`scripts/start.sh` is up and a TensorFold checkout of the same version is available for its
python bench tools (they are standard-library clients; `tools/bench_openai.py`,
`tools/bench_concurrent.py`, `tools/prefill_cold.py` in the TensorFold tree).

## 0. Before the GPU work

- `scripts/prepare.sh` clean from an empty state; `docker run --rm <image> --version` OK.
- `DRY_RUN=1 scripts/start.sh` prints the two docker commands as expected.

## 1. Bring-up

- [ ] `scripts/start.sh` reaches `/health`; both containers stay up 10 minutes idle.
- [ ] Rank logs show the checkpoint digest exchange (both ranks agreeing on bytes) and both
      ranks' post-load memory (`/memory` if the route exists; `docker stats` otherwise).
- [ ] `scripts/stop.sh` stops both and leaves two gzipped logs in `$LOG_DIR`.

## 2. Exactness (the same gates the PR used, through the API)

- [ ] **Drafted = plain**: one prompt at temperature 0 with drafts on and with `DRAFTS=0`
      (or `--no-drafts`) — same tokens.
- [ ] **Resumed = fresh**: a two-turn conversation vs the same second turn sent to a server
      started without the history — same tokens (the prompt cache's snap hooks).
- [ ] **Concurrent = solo**: 4 concurrent requests vs each run alone — same tokens
      (`bench_concurrent.py --alone --serial` against the server).
- [ ] Sampled request at a seed twice — deterministic across the two ranks.

## 3. Capacity

- [ ] `/memory` (or `/health`) peak device memory at `--context 40960 --parallel 4` ≤ ~100 GiB
      on **both** ranks (the PR saw 97.3 GiB on rank 0 at `--parallel 2`).
- [ ] Long-context fit probe: raise `CONTEXT` stepwise (131072, 262144, 524288) on a restart;
      record where admission refuses, and the free memory left. The old stack served 1M tokens
      on fp8 KV; bf16 KV will refuse earlier — find the knee before promising anything.
- [ ] One rank under `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=21` less than the other refuses streams
      naming that rank (the PR's admission check).

## 4. Speed (same method as the old repo's README table, for a fair row)

- [ ] sparkDash or `bench_concurrent.py`: aggregate + per-request tok/s and TTFT at 1/2/4
      concurrent, prose and code prompts, GPU clocks capped at 2,200 MHz as before.
- [ ] `prefill_cold.py`-style cold prompt reading at 2k/8k/32k/65k/131k.
- [ ] Compare against the old table in the EXL3 recipe's README; record in this file.

| Metric | EXL3 production (measured 2026-10-11) | NVFP4 native (fill in) |
| --- | --- | --- |
| decode 1 / 2 / 4 concurrent (prose) | 61.8 / 75.2 / 105.5 tok/s | |
| prefill 8k / 32k / 131k | 2055 / 2018 / 1900 tok/s | |
| TTFT 8k | 4.20 s | |

## 5. Client compatibility

- [ ] hermes agent: custom endpoint `http://<head>:8888/v1`, model `GLM-5.3-Flash-NVFP4`;
      a tool-calling loop works end to end (the native server has its own tool-call parsing —
      check streamed calls arrive whole, as the old stack's GlmCallStreamer did).
- [ ] `toolcheck.py` / `tool-eval-bench --perf` from the old repo's tools against this server.
- [ ] Structured outputs (`response_format`), `/tokenize`, `/v1/models`.

## 6. Engine-level checks (optional, in the builder image)

The runtime image ships no test programs (upstream's packaging omits them). To run the PR's
`tf-cuda-test` checks (`glm-ref`, `glm-shared-check`, `glm-lanes-concurrent`, `glm-lanes-cover`),
open a shell in the build stage of the same commit on one Spark:

```sh
docker run --rm -it --gpus all <builder-stage-image> bash   # see image/build-native.sh for the layout
zig build test-cuda-host && zig build test                   # host tests need no GPU
tf-cuda-test glm-ref && tf-cuda-test glm-shared-check        # on the GPU, TP=1 on one host
```

The TP=2 lane checks ran at build time on the PR author's pair; re-running them here is the
last word on our hardware.
