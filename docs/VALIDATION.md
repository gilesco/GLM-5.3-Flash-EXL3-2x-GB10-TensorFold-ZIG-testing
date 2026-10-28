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

- [x] **The modified checkpoint's battery** (`campaign-20261024-142659/13-top4qual.md`,
      round 26): ALL of the above PASS on real top-4 routing (`num_experts_per_tok` 4,
      `glm-topk4b`) — the exactness guarantees hold on the site's served model.
      Tool-eval 93 (production's anchor: 95); hardmode 89 (production's anchor: 91).

> **Native 1.1.x receipt (2026-10-23, corrected same day): the gates bind on content,
> and the first nondeterminism reading was wrong.** Whole-message sha is an invalid
> determinism instrument on BOTH engines: every response envelope carries a fresh
> `id`, `created` timestamp and engine-metadata object, so raw shas can never match
> across two requests. The prodqual control proved it: on production 0.6.6 the same
> greedy probe twice is **content-identical with raw bytes differing in exactly
> `id`/`created`/metadata — and `choices` byte-identical**. Re-checking the native
> run's g1 pair shows the same: **`choices` byte-identical there too** (same
> completion counts). So the native engine is token-deterministic at greedy, exactly
> like production — the first campaign's "byte-level engine nondeterminism" note was
> the instrument, not the engine. The gates still bind on normalized content
> (`content_sha`), which is envelope-proof on any engine; with the envelope excluded,
> drafted=plain, resumed=fresh and concurrent=solo hold to the byte on both engines.
> g3 was also redesigned: the same 3-message conversation sent twice (cold compute vs
> full prefix-cache hit) with `cached_tokens` receipts per arm, and `max_tokens=1024`
> so thinking cannot eat the answer (the first run's empty g3 reply was 512 tokens of
> thinking, plus the probe had compared two different prompts — unrunnable by design).
> Open rigor item: g3's `cached_tokens=0` on both arms — the engine does not populate
> that usage field, so the cache-hit path is content-verified but not counter-verified
> (a /metrics `tensorfold_cached_tokens_total` check closes it).

## 3. Capacity

> **2026-10-23: closed by the qualification campaign** (`results/campaign-20261023-*`,
> tables in docs/TOOL_EVAL_TESTING.md). What the receipts show:

- [x] Peak device memory recorded via the serve line on both ranks at every phase: dense8+fp8
      at 524288×4 = model 85.46 GiB, 3.61 GiB/stream, 13.8 GiB left after a 12.2 GiB
      reserve; dense4fp8 = 83.50 GiB / 15.9 GiB left (the serve line is in every receipt).
- [x] Long-context fit: the ladder closed at the top — **524288 serves at PARALLEL=5**
      (the engine's own arithmetic: 8 streams need 25.29 GiB, budget leaves ~15.8); 6+
      is blocked by the engine's 12.2 GiB reserve, which no flag exposes. The iso-budget
      intermediate rungs were never needed (fp8 KV made the top rung fit directly).
- [ ] One rank under `TENSORFOLD_CUDA_MEMORY_LIMIT_GB=21` less than the other refuses streams
      naming that rank (the PR's admission check) — still untested (needs a deliberate
      asymmetry; low priority).

## 4. Speed (same method as the old repo's README table, for a fair row)

> **2026-10-27: re-measured on engine v1.1.6** (`eea9dc9c`) — the campaign receipts
> `campaign-20261027-124115`/`-173843` (duel + benchy + knobs) against production's fresh
> idle duel receipt (round 33, its new patch series). The 2026-10-23 rows below are the
> v1.1.0-era history:
>
> **2026-10-28: the engine is v1.1.8 with `TF_GLM_INDEX_CUDA=1`** (round 49's adoption;
> `campaign-20261028-141916/22-ixgates.md`) — the duel under the knob reads within noise
> of the anchors (solo 57.0/56.9, c2 93.7, c4 152.3); the 2026-10-27 rows stand as the
> current comparison, and the depth-prefill axis now has its own ladder
> (LAUNCH.md round 47).

| Metric | EXL3 production (fresh receipts) | native EXL3 dense4fp8 @524288 (v1.1.6) |
| --- | --- | --- |
| warm restart → serving | **93.7 s** (the engine's own line) | **256 s wall** (the knobs cells' timed restarts 246–260 s; decomposition still pending) |
| decode solo (bench tg128) | 54.4 tok/s | 59.1 tok/s |
| decode solo (duel, steady) | 58.6 tok/s | **64.8 tok/s — CROSSED** |
| decode c2 aggregate (duel) | 87.0 tok/s | **108.0 tok/s — CROSSED** |
| decode c4 aggregate (duel) | 154.6 tok/s | **167.4 tok/s — CROSSED** |
| prefill pp2048 (benchy) | 1,514 tok/s | 1,177 tok/s (the 26.9k-token wall pass: 1,175) |
| TTFT d0 (1-token) | 186 ms | 148 ms |
| TTFT d8192 (1-token) | 142 ms (flat-to-improving — its prefix cache) | 4,954 ms (linear — the round-34 frontier) |

> **The bar (user, 2026-10-23): this line succeeds when it surpasses production 0.6.6 on
> loading time, generation tok/s, and prefill rate.** As of v1.1.6 generation tok/s is
> **CROSSED on every front** (solo 111%, c2 124%, c4 108%); loading (2.7×) and prefill
> (−22%) are the open fronts, and TTFT at depth is the third (the round-34 root cause).
> Read with docs/TOOL_EVAL_TESTING.md's "cliff withdrawn" receipt: the bench's tg-at-depth
> rows include prefill in the denominator, so only the steady-state rows compare decode
> fairly. The prefill gap (and its TTFT
> consequences) lives in the MoE expert path of prompt passes (90% router+routed), not
> attention — section profiling receipt in TOOL_EVAL_TESTING.md.

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
