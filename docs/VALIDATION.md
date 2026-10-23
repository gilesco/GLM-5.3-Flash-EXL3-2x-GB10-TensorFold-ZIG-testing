# Validation gate (run when both hosts are free)

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

## 2. Exactness (the same gates upstream's EXL3 receipts used, through the API)

- [ ] **Drafted = plain**: one prompt at temperature 0 with drafts on (DFlash2 via
      `--drafter`) and with `DRAFTS=0` (or `--no-drafts`) — same tokens. Also check a drafted
      run with the MTP fallback (remove the DFlash2 snapshot from the path or `DRAFTS=1`
      without it) equals the DFlash2-drafted tokens.
- [ ] **Resumed = fresh**: a two-turn conversation vs the same second turn sent to a server
      started without the history — same tokens (the prompt cache's snap hooks).
- [ ] **Concurrent = solo**: 4 concurrent requests vs each run alone — same tokens
      (`bench_concurrent.py --alone --serial` against the server).
- [ ] Sampled request at a seed twice — deterministic across the two ranks.

> **Native 1.1.x receipt (2026-10-23, first campaign): the gates bind on content,
> not bytes.** The Zig engine is byte-level nondeterministic at greedy — the same
> prompt twice on the same server produces identical content with different raw
> bytes (reasoning/metadata level), where the 0.6.x Python engine was byte-stable.
> Whole-message sha gates therefore can never pass here, and the first run's g2/g4
> "FAILs" were all content-identical outputs. From this receipt on: g1–g4 compare
> **normalized content** (`content_sha` — whitespace-folded, reasoning excluded)
> with raw-sha equality recorded as a bonus line; determinism beyond content is a
> recorded property, not a gate. g3 was also redesigned: the same 3-message
> conversation sent twice (cold compute vs full prefix-cache hit) with
> `cached_tokens` receipts per arm, and `max_tokens=1024` so thinking cannot eat
> the answer (the first run's empty g3 reply was 512 tokens of thinking, plus the
> probe had compared two different prompts — unrunnable by design).

## 3. Capacity

- [ ] Peak device memory at `--context 40960 --parallel 4` recorded on **both** ranks. No
      anchor exists for EXL3-native two-rank (the NVFP4 PR saw 97.3 GiB on rank 0 at
      `--parallel 2`; EXL3's BF16 linears weigh more — expect a new number, not the old one).
- [ ] Long-context fit probe: raise `CONTEXT` stepwise (131072, 262144, 524288) on a restart;
      record where admission refuses, and the free memory left. The old stack served 1M tokens
      on fp8 KV; the native latent cache will refuse earlier — find the knee before promising
      anything. The 2026-10-16 ladder rules apply: start at 131072 with trimmed knobs
      (reserve 8 / prompt cache 1.5) and measure the overshoot before climbing.
- [ ] One rank under `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=21` less than the other refuses streams
      naming that rank (the PR's admission check).

## 4. Speed (same method as the old repo's README table, for a fair row)

- [ ] sparkDash or `bench_concurrent.py`: aggregate + per-request tok/s and TTFT at 1/2/4
      concurrent, prose and code prompts, GPU clocks capped at 2,200 MHz as before.
- [ ] `prefill_cold.py`-style cold prompt reading at 2k/8k/32k/65k/131k.
- [ ] Compare against the old table in the EXL3 recipe's README; record in this file.

| Metric | EXL3 production (measured 2026-10-11) | native EXL3 (fill in) |
| --- | --- | --- |
| decode 1 / 2 / 4 concurrent (prose) | 61.8 / 75.2 / 105.5 tok/s | |
| prefill 8k / 32k / 131k | 2055 / 2018 / 1900 tok/s | |
| TTFT 8k | 4.20 s | |

## 5. Client compatibility

No agent runtime is required: **tool-eval-bench is the client** (see
[TOOL_EVAL_TESTING.md](TOOL_EVAL_TESTING.md) — the same fixed command on every
configuration).

- [ ] The full bench run passes end to end: 97 scenarios (69 standard + 28 Hard Mode) with
      streamed tool calls, multi-turn tool results and `response_format`/structured cases —
      the scenarios themselves exercise the client-side API surface an agent would.
- [ ] `tool_choice` handling matches what clients send: `"auto"` works (the engine refuses
      `"required"`/named-function today — record any client that needs it).
- [ ] Streaming reaches `[DONE]` with `finish_reason` set (the bench's traces show it; a
      failed stream counts as a failed scenario).
- [ ] `/metrics` stays sane through a bench run (counters advance, no error rates).

## 6. Engine-level checks (optional, in the builder image)

The runtime image ships no test programs (upstream's packaging omits them). To run the
engine's own checks (`tf-cuda-test exl3`, `tf-cuda-test exl3-tp2`, `tf-cuda-test glm-ref`
and friends), open a shell in the build stage of the same commit on one Spark:

```sh
docker run --rm -it --gpus all <builder-stage-image> bash   # see image/build-native.sh for the layout
zig build test-cuda-host && zig build test                   # host tests need no GPU
tf-cuda-test exl3 && tf-cuda-test exl3-tp2                   # on the GPU; exl3-tp2 is the GLM split rules over two ranks
```

The EXL3 and TP=2 checks ran at build time on upstream's pair for the 1.1.0 gate; re-running
them here is the last word on our hardware.
