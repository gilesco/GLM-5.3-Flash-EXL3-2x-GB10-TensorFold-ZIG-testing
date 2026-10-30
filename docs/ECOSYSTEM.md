# The ecosystem around us: what other GLM-5.3-Flash stacks do, and what it means for this repo

Researched 2026-10-23 from forum thread 381350 (read end-to-end — receipts in
docs/LAUNCH.md) plus each repo's own README. Question this page answers: **is
there anything in the community stacks that should change how we run, tune, or
build on the 1.1.x native line — without importing their complexity?**

Our standing bias, unchanged by the research: this repo's differentiator is
*easy to use, easy to get started with*. Ideas are welcome as **config knobs,
preflight checks, or doc pointers**; frameworks, patch ladders, and multi-file
config regimes are not.

## The field at a glance

| repo | engine | what it runs | headline numbers (their own or the thread's) |
|---|---|---|---|
| kindlingai/glm-5.3-flash-gx10 | vLLM + "mentat" scheduler | NVFP4, DFlash2 adaptive-k, dual-root RoCE | won kindlingai's six-config shootout; all-reduce 110→190 Gb/s, +11% prefill |
| technigmaai/glm-5.3-flash-gb10-tp2-native | vLLM native TP2 | NVFP4 FP4-Local, DFlash2 | thread's quality/speed leader: 96 max TEB, 2678 pp |
| kindlingai/glm-5.3-flash-exlr8-gx10 | vLLM + exlr8 MoE kernel | **EXL3-trellis experts** (k2/k3/k4) | decode +3–8% vs NVFP4, KV +15%, prefill −7.7% (open gap); k3 TP2: 84.2 code t/s, 4.2M KV pool |
| eugr/spark-vllm-docker | vLLM (regular + B12X images) | recipe system for many models; GLM-5.3 = B12X + MTP + 1M ctx | the hub the other vLLM stacks are derived from |
| Entrpi/glm-5.3-flash-exl3-2x-spark | vLLM | **EXL3 4-bit** checkpoint, 1.29M ctx at parity | the other "easy button"; one-shot installer with rollback |
| goshi/tensorfold-spark-dashboard-metrics-exporter | bridge, engine-agnostic | **TensorFold metrics → Spark Dashboard** | zero-dependency; primary profile targets our engine line |
| 0rand / DragonScale / rodman80 / gitcommit90 / kindlingai gr8 / aidenle | misc | NVFP4, W4A16, single-Spark, from-scratch CUDA | different hardware targets or quant lines — noted, not studied |

Community benchmarks of *our* repos from the thread (receipts in LAUNCH.md):
**G647 (production 0.6.6, fp8 dense+KV) = 93.0 max TEB / 52.7 t/s sweep / 93.9%
depth score — the best EXL3 stack measured**; bernisse runs our 1.1.2 line
unqualified for weeks at 94/100 tool-eval; giles8 confirms real concurrency
(>100 tok/s aggregate).

## What transfers to us — three tiers

### Tier 1 — adopt now (zero complexity, config/doc-level)

1. **goshi's metrics exporter is built for our engine.** Its `zig-v1` profile
   targets exactly the TensorFold native line (`tensorfold_*` underscore
   families + `/health` JSON), and `python-v1` targets production's 0.6.6
   (`tensorfold:` colon families). Zero engine changes, stdlib-only, systemd
   unit included, honest omissions (no faked KV% or queue numbers), persistent
   counters across restarts. This is a free operability win for the whole site:
   run it beside whichever stack is up, point Spark Dashboard at :8004.
   *Caveat to verify on our first trial:* the README's examples assume the
   engine on `:8002`; ours serves on `:8000` — set `TF_UPSTREAM` accordingly.
2. **Never pin `NCCL_IB_GID_INDEX`** (kindlingai's hard-won rule: RoCE GID
   tables reorder across reboots; derive at start). We already don't pin it —
   recorded here as an **invariant**, so a future tuning session doesn't
   "fix" that.
3. **CPU clock verification at preflight** (MaTeZZ's finding, now in
   `scripts/run-qualification.sh`): kernel 6.17 ran Grace at floor clock with
   `scaling_cur_freq` lying; the `acpi_cppc/feedback_ctrs` probe is the honest
   number. Our hosts run the good 7.0.0-1019 kernel — the campaign logs
   delivered MHz anyway, so a regression can't hide.
4. **Headless hosts for the heavy runs** (eugr + kindlingai both require it:
   desktop sessions cost GB10 real memory and, per kindlingai, CPU/GPU share
   one power budget). Both our hosts are SSH-driven already; noted so a future
   "why is decode low" session checks this early.

### Tier 2 — measure on our stack, hand the data to upstream (not code we write)

1. **Adaptive-k DFlash2.** kindlingai picks draft length per step from a live
   cost/acceptance model; MaTeZZ measured acceptance ~2.3 (prose) / 4.5 (code)
   / 5.8 (JSON) at temp 0 — draft depth should match the task. TensorFold's
   DFlash2 uses fixed rounds. We can't set this via config, but our
   `TF_GLM_PROF` sessions + the campaign's cliff phase produce exactly the
   acceptance data an upstream adaptive-k (or deeper rows per #1059) needs.
   Collect, don't implement.
2. **Prefill fairness.** kindlingai caps one prefill's share of each scheduler
   step (`LONG_PREFILL_TOKEN_THRESHOLD=2304`) after a 120k-token prefill
   blocked a 12-token request for 78–90s. Our engine interleaves prefill and
   decode (prompt passes) — the thread says vLLM stacks can't. Whether a deep
   prefill stalls concurrent decode *on our line* is exactly what the
   campaign's depth + cliff phases will show. If yes: that's a measured
   upstream ask with numbers attached.
3. **Spin-wait / heat.** kindlingai's `busy_loop_s=0.002` patch trades spin
   latency for ~20°C and decode loss on GB10's shared power budget. Upstream
   1.1.x already moved followers to sleeping ranks (#1053). The head's idle
   behavior between steps is unmeasured for us — a one-window `TF_GLM_PROF`
   + `clock_mhz` check closes it.
4. **Structured-output draft truncation** (MaTeZZ's mechanism for our
   concurrency cliff): one `json_schema` request disables draft truncation for
   the whole batch (31.5 → 22 tok/s per stream). The campaign's cliff phase
   tests this directly; if it reproduces on 1.1.2, it's the single most
   valuable bug report we could hand upstream (as data, not a patch).

### Tier 3 — watch, don't adopt (engine-specific or out of scope)

1. **EXLR8 (mmastrac's EXL3-trellis experts).** The one genuinely EXL3-family
   development in the field: per-layer k2/k3/k4 expert rates decoded by a
   custom MoE kernel, beating NVFP4 on decode (+3–8%) and KV (+15%) with an
   open prefill gap (−7.7%). Serving an EXLR8 checkpoint would need a loader
   in upstream TensorFold — out of our hands, but if it lands, our line gets
   smaller experts and bigger KV windows from the same trellis math we already
   serve. **Watch item #1.**
2. **Dual-root RDMA collectives** (kindlingai's arx, both CX7 PCIe roots,
   all-reduce 110→190 Gb/s). Our `NCCL_IB_HCA` (the site's cabled HCAs, set in
   `local.sh`) already names both roots and rank logs confirm 2 HCAs —
   effectively the same idea via NCCL. Nothing to change; the rank-log line
   stays in every receipt.
3. **B12X / megamoe / marlin kernels, mentat scheduler, InstantTensor
   loading** — vLLM-side engineering. Our warm digest cache loads in ~223s,
   comparable to kindlingai's ~3-minute processed-weight snapshot restarts.
4. **PMU128 prefix-cache patch** (eugr stack; 92–97% hit rates on full-prefix
   hits) — maps to upstream's existing prompt cache; revisit only if our
   cache-hit path ever shows recompute waste.
5. **RecoverSSM, dispram, 3–4-node rings, NFS weight sharing, W4A16,
   single-Spark quants** — different topologies or kernels; no path to us.

## Ease-of-use: our stance vs the field

Entrpi is the community's other "easy button" (one-shot installer with
rollback, 1.29M context at parity) — and it's vLLM-based, which means the
config regime is a `.env` with drift risk between releases; the thread
contains at least one write-up of exactly that failure (configs breaking
under them across upgrades). technigmaai fights the same problem with
discipline: a menu launcher plus `VERSIONS.md` pinning every image.

Our answer is structural, not cosmetic:

- **all knobs are guarded** in `config.sh` (one file, validated, with the
  prepared-marker check) — site identity lives in git-ignored
  `scripts/local.sh` / `docs/SITE.md`, so upgrades can't clobber it;
- **one-command qualification** (`scripts/run-qualification.sh`) with a
  production guard, phase markers, and append-only receipts — reproducibility
  is a default, not a discipline the operator must maintain;
- **every change lands with its receipt** in docs/LAUNCH.md.

That is the line we hold while borrowing: exporter → doc pointer; fairness /
adaptive-k / cliff → measured data for upstream; EXLR8 → watch. Nothing in
this research justifies adding a second config system, a patch ladder, or a
menu launcher to this repo.

## Upstream-watch list (track only — we post nothing to upstream repos)

| item | why | trigger to revisit |
|---|---|---|
| EXLR8 checkpoint loader in TensorFold | EXL3-family; smaller experts, bigger KV | upstream issue/PR appearing |
| adaptive-k or deeper DFlash2 rows (#1059) | acceptance data says depth should flex | engine release notes |
| prefill/decode fairness knobs | if our cliff phase shows deep-prefill stalls | campaign verdict |
| head idle spin behavior | GB10 shared power budget | TF_GLM_PROF window |

## Upstream PRs that bear on our fronts (rechecked 2026-10-24 evening; perf-gated policy: nothing to bump to)

The releases: v1.1.3 (2026-10-23) remains the newest and carries **no glm/cuda perf
content** (round 13's verdict stands); `main` is 136 commits ahead of our pin —
Flash Next/HIP/Windows/app heavy, **nothing glm-perf**. Staged locally instead: the
**GLM slot-leak fix** (upstream `7d330e24d`, #1218 — a released stream left its queued
draw's ring slot taken; every completed/cancelled request leaked a slot until restart;
1-line `patches/glm-slot-leak.patch`, bit-invisible) and the **router-ks2 patch**
(PERF_PLAN Front 2 Lever A2).

The open PR field, by front:

- **Front 2 (prefill)**: **#1176 — qwen CUDA: `--prompt-precision int8`, prompt matmuls
  on int8 activations (cold prefill 20–27% shorter)** — a measured prefill mechanism on
  the other MoE family; a glm port candidate if our router patch lands and prefill still
  lags. **#1228 — preserve completed CUDA prefill across cancellation and retry** —
  robustness of prefill work, worth watching.
- **Front 1 (concurrency/round economics)**: **#1168 — qwen CUDA: decode attention on
  tensor cores (fp8 and bf16 caches), decode after long prompts** — our kda+mla are 40%
  of a verify round; a glm port candidate. **#1171 — qmmf half tiles for decode calls
  whose 64-column grid leaves SMs idle** — same family of fix. **#1155 (Flash Next's
  shared-round batched drafting) closed UNMERGED**, but its maintainer fix-ups
  (`c4d36404c`, "the shared pass's helpers split out") landed pieces — the mechanism
  precedent our glm ask can cite.
- **Front 3 (memory/paging)**: **#1163 + #1211 — glm/cuda: KV pages from the core pool**
  (`TF_KV_PAGES`). Today every
  stream holds a whole window (3.61 GiB of fp8 latent cache at 524288) and the
  admission refusal's math follows; with paging, streams take 512-row pages from the
  core pool and kept states hold only rows up to their position. **At TP=2 the default
  (`auto`) keeps a window per stream — `TF_KV_PAGES=on` opts us in.** If it admits
  6+ streams at 524288 and the concurrency curve changes, both the ladder ceiling and
  the cliff get re-measured. Heads live on Bizuayeu's fork:
  `Bizuayeu/TensorFold` @ `87e4417ebec6` (`pr/glm-kv-pages-r9`, stacked on #1163).
  Our prepare already takes `TF_REPO`/`TF_COMMIT` overrides — buildable and
  qualifiable as-is, and `config.sh` now forwards `TF_KV_*` to both ranks.
  **Measured on the branch head (2026-10-23, round 9): paging works on the bf16 KV
  path only — the engine exits at startup on `--kv-cache fp8` × `TF_KV_PAGES=on`**
  ("fp8 keeps a window per stream"), so every fp8 KV cell refuses. Under paging the
  engine admits 5, 6 and 8 streams at 524288 with the pool fixed at ~one full window
  (1028–1032 pages, ~6.4 GiB, demand-allocated, ~220 MiB fixed per stream) — depth ×
  streams is the pool's budget, not per-stream windows. All §2 gates PASS on paged
  bf16; the bench favors paged-bf16 over the fp8-window config on every row (+5–9%)
  and the solo row reaches production parity — but without fp8 KV it cannot displace
  dense4fp8 as the serving config (a full-depth stream costs 2× and the pool caps
  deep concurrency at one window). The upstream ask this feeds: **paging × fp8 KV
  for glm/cuda** — with fp8's halved bytes under the pool, a paged config could
  serve deep streams AND high concurrency at once.
- **#1080 — long-prompt prefill 5.4×** (qwen CUDA family): 128→4096-row chunks,
  tensor-core prompt attention on e4m3 rows, and v0.6.6's delta-rule chain —
  75,616 tokens to first token in 144.5 s → 26.9 s on one RTX 5090. Our engine's
  "prompts in passes of up to 2048 rows" and production's +49% prefill lead sit
  exactly in this PR's blast radius. Qwen-targeted today (`BobClawblaw/TensorFold`
  @ `12f8a9c40d53`); a glm/cuda port is the watch item.
- **#1215 — round-class GPU timers** (`TF_Q27_PROFILE=1` logs each round's class GPU
  times, replies carry `gpu_seconds`): the instrumentation pattern our concurrency
  diagnosis wants — route / experts / dense-attention / **drafter** per round. If
  TF_GLM_PROF lacks class timers, this is the shape of the ask.
- **#1079 — DFlash2/lanes decode +9–15% (qwen27) — TESTED HERE, A WASH, DROPPED**
  (2026-10-23, `campaign-20261023-185520/`): the staged patch passed all §2 gates
  ("replies unchanged" holds on our model) and then moved nothing — solo d0 +3.3%,
  solo d4096 −8.6%, rest ±2%. Lesson recorded: **our gates are the "replies unchanged"
  instrument, and the A/B — not upstream's numbers — decides keep-or-drop.** Receipt in
  patches/README.md.
- **#1143 — Flash Next: NVFP4 experts on prompt tiles, decode-form sums, bit for bit** —
  the expert-batching-in-prompt-passes mechanism our prefill gap wants (prompt chunks
  here are 90% router+routed; the router alone is ~910 ms of a 1,892 ms chunk). Family-
  mismatched (Flash Next files) — the glm/cuda port of this mechanism, plus #1080's
  chunking, is the standing prefill ask upstream.
- (#827 — the lane backend, one stream per slot, on DeepSeek/Metal: the architecture
  direction for per-stream concurrency; not our engine today.)

**Plan this buys us:** (1) baseline `TF_GLM_PROF` concurrency diagnosis on the current
build, then (2) build the `pr/glm-kv-pages-r9` head and re-run the campaign against it —
admission (does 6 fit now?), the §2 gates on paged KV (exactness first, as always), and
the throughput curve. The engine code itself stays upstream's; our job is the
qualification receipts.

- 2026-10-27 — **the cross-request prefix-cache gap (TTFT-at-depth), as upstream data** (found
  reading v1.1.6 at `eea9dc9c`; nothing posted upstream): the native engine's prompt cache keys
  states at conversation-structure marks only, and `server/chat.zig` populates `shared_prefixes`
  from the system prefix alone (three cuts at system_len−2048/−512/system_len). Cross-request
  shared prefixes inside user content therefore always re-prefill (linear TTFT), where
  block-level prefix caching (vLLM-APC-style, and our production's python engine) reuses them.
  The span machinery is not a blocker (spans are token-derived); #1321 covers only simultaneous
  arrivals; the `--learn` imprint path exists but is fed only system cuts from chat. A
  grid-boundary marks policy for long prompts is the natural upstream-shaped fix.

- 2026-10-27 (evening) — **v1.1.7 released (2026-10-27, daily cadence: 1.1.5→10-25, 1.1.6→10-26,
  1.1.7→10-27); 120 commits on main since our pin (`eea9dc9c`), 57 in the release**. GLM-relevant:
  in v1.1.7 — a prefill diagnostic (`--prompt-precision int8` fallback counting), the grammar
  compiled once per request, a streamed-tool-call XML newline framing fix (marginal tool-eval
  relevance); no GLM/CUDA perf headline. **Post-1.1.7 on main (unreleased) — the item that aims
  at our line: #1430/#1449 `TF_GLM_INDEX_CUDA=1`** — a prompt chunk's DSA indexer in two dedicated
  CUDA kernels, byte-equal (the kernel contract: byte-equal, stable across repeats, same reply
  tokens), opt-in, `families/glm/cuda_forward.zig` + `cuda_index.zig`. **Upstream's GB10
  measurements: cold-prompt first token 2% sooner at 131k, 9% at 524k, 16% at 1M.** Also on main:
  the prompt cache's find() walks a longest-first index and uses extension links instead of a
  whole-store scan (lookup speed; the MARKS POLICY is unchanged — our round-34 cross-request
  prefix diagnosis remains unfixed upstream and stays our patch candidate), the spill/rows-store
  write-behind + read-ahead work (#1464), Flash Next/Mac items that don't apply.
- 2026-10-28 — **#1449 verified on our hardware, and the depth curve transfers**
  (`campaign-20261028-113509/20-indexdepth.md`): at upstream's own first claimed point
  (131k) we measured **−2.6%** cold-TTFT (claimed −2%); the ladder up to 393k reads
  −2.7%/−3.6%/−4.5%/−5.5% with the flat-ish on-arm marginal slopes (0.816→0.882) against
  the RISING off-arm (0.860→0.976) — cold prefill is superlinear at depth and the indexer
  kernels counteract it. Their deeper points (524k −9%, 1M −16%) sit beyond our ladder;
  our extrapolation at 524k is ≈ −6.5% — same direction, smaller at our depth. Loading,
  decode, and short-prompt TTFT are unchanged (the knob is prefill-side only). Banked as
  an ADOPT-PENDING-GATES (one gated serve owed before our config default flips).
- 2026-10-28 (later) — **the gated serve ran: ADOPTED.** `campaign-20261028-141916/
  22-ixgates.md`: §2 gates ALL PASS under the knob (the byte-equal contract shown on our
  serve), the duel within noise of the v1.1.8 references. Our config default is now
  `TF_GLM_INDEX_CUDA=1`.
- 2026-10-28 (later) — **two upstream-bound data points from the same window**
  (`campaign-20261028-141916/23-gibdisc.md`, `24-spillgib.md`): (1) **`--spill-gib`
  refuses GLM-5.3-Flash outright** — the flag's own words: it "keeps evicted states on
  disk for Nemotron, the Qwen hybrids and Flash Next only on CUDA". The #1464 spill work
  does not cover GLM yet; the churn-heavy agentic pattern on GLM has no disk fallback.
  (2) **`--prompt-cache-gib` did not size the kept-states store on our GB10 serve** —
  GIB=16 → the store read 2.6 GiB with default-identical counters (behavioral, round 47);
  GIB=10 → 2.5 GiB (round 48) — the engine's own default figure either way. One open
  read: whether a smaller CONTEXT frees the budget (one cheap serve, queued).
- 2026-10-28 — **the resume-cost gap is architectural, and the engine already carries the
  remedy shape** (source read at v1.1.8, `zig/src/families/glm/kept.zig` +
  `core/prompt_cache.zig`): the GLM family's prompt cache resumes by COPYING the whole
  kept state back into a stream's pools (`copySnap`), while the engine also carries a
  paged prompt store (`prompt_radix.zig`, `rules.paged` — remap-style resumes) that the
  GLM family does not wire (`engine_paged=false`; the snap_* decls never set paged). The
  python 0.6.x engine's block cache remaps pages (production's 65 ms resume at 7.2k
  tokens vs the native line's ~1,434 ms — measured rounds 43–45). Data point for
  upstream: wiring the paged store for the GLM CUDA family (the backend's page-ownership
  callbacks) is the path to production-class resume times on the native line. Nothing
  posted upstream.

## Community sweep — the NVIDIA GB10 GLM-5.3-Flash thread + DeepSeek field (2026-10-29, round 73)

Source: forums.developer.nvidia.com/t/381350 (936 posts; the user is giles8 in-thread),
plus the DeepSeek-on-Sparks field. The findings, ranked by leverage:

1. **The effort-line template bug — a measured quality lever** (posts 915/933): the
   stock, MiaAI and technigma templates put the "Reasoning Effort" hint at the TOP of
   the prompt; on long sessions the model ignores it (max ≈ high; measured 11–528
   reasoning tokens at 84K history vs 1,900–7,000 with the line moved to just before
   the final assistant turn — "effort_tail"). Second effect: with the line at the top,
   every effort switch invalidates the whole prefix cache. A second, distinct template
   gap: reasoning collapse right after TOOL RESULTS at long context (technigma's
   TOOL_REASONING_VALIDATION.md documents + fixes it with a reminder). OUR LINE FAILS
   TC-51 AND TC-92 DETERMINISTICALLY — on both the Focus and the AZ runs (verified in
   our receipts) — while technigma's stack is clean on both (full 97). Tenant-isolation
   failures are plausibly the same tool-result/template family. This is a stack-level,
   +2-3-point quality lever that no checkpoint change touches.
2. **Do we wait for TensorFold updates? Yes — with a named watch list** (post 926,
   bernisse's peer review of this line): upstream 1.1.4+ carries #1163+#1211 (GLM CUDA
   KV pages from the core pool; #1163 mergeable-clean), #1218 (a 1-line ring-slot leak
   fix — concurrent-cold serving fails at the 64th request without it), #1168
   (tensor-core fp8 decode attention, 1.5–2.3x on qwen) and #1176 (int8 prompt
   matmuls, −20–27% prefill) — the last two attack our exact decode/prefill gaps if
   they generalize to GLM's DSA rows. Caveat bernisse confirms independently: fp8 KV ×
   TF_KV_PAGES are mutually exclusive today (bf16-only) — our round-59 finding. Check
   which of these our v1.1.8 base already carries before any window.
3. **The router-pathology fix is the field's highest-leverage kernel work** (bernisse:
   "would put native ahead of both stacks" — the user's own diagnosis implies slope
   0.662 → ~0.25 ms/tok). bernisse's cross-quant bake-off on a second 2x GB10 pair:
   the fork (0.6.6+106) beats native 1.1.x today — code decode 91.6 vs 62.3 t/s, 4x
   concurrent 639 vs 217, prefill 64k 2,483 vs 1,448 (technigma NVFP4: 2,933) — the
   router fix is what flips native ahead of both.
4. **DeepSeek 4.1 / V4-Flash on this hardware** (the user's "deepeek results"):
   DeepSeek-V4.1-Flash-1017 runs ON TensorFold's python engine at EXL3 2.9bpw —
   ajvazan's field table: tool-eval 93/93 (high/max), depth 211/212, 2.1–2.3x the
   vLLM kit. The field's quality table: technigma native 92/**96** (max — the field
   best), kindlingai 92/95.67, the user's fork (G644a) 91/92 with the FIELD-BEST
   285 pts/hr, DSV4-0731-vLLM 89.67/89. Structural caveat ajvazan verified: this TF
   build does not implement V4.1's numeric effort budget (high == max structurally) —
   GLM vs DSv4.1 rows must not be compared on effort semantics. V4-Flash-0731 on the
   vLLM kit: 10B+ production tokens, stable at a 2200 MHz clock ceiling, 2.53M-token
   KV, DSpark K=5. ds4/dspark: V4 Flash Q2/MXFP4 on 2 Sparks at 41.7–83.3 t/s with
   DSpark (code ≈ 2x prose). Field Signal's methodology caution: the 114–141 t/s
   headline numbers are count-to-200 drafts; real 1,024-token work runs ~47–54 (GLM)
   vs ~54–98 (DeepSeek) — DeepSeek wins single-stream rows, GLM wins 4-concurrent.
5. **Portable serving ideas** (DGPP engine's changelog, several cover GLM):
   vocabulary-output memory retains only generation rows (3.79 GiB → 7.58 MiB);
   attention workspace follows the per-request context limit (−4.49 GiB); concurrent
   prompts share a forward pass; short requests no longer evict long-conversation
   snapshots below a 1,024-token threshold. OllieO's long-running production settings
   on our fork (post 932, since Oct 9): TENSORFOLD_RANDOM_SEEDS=1 (fresh seed per
   unseeded request), TF_GLM_KEEP_EFFORT=1 (fork patch 1002: thinking-off prompts
   keep the effort line for same-prefix cache hits), TF_GLM_CACHE_ENTRIES=48,
   PARALLEL=8, MAX_TOKENS=65536, SPILL_GIB=64.
6. **Methodology cautions from the field**: a one-line template change (a thinking
   reminder after tool results) moved median turn time 3.4 → 4.5 s with engine speed
   unchanged — never compare medians across weeks, fixed probes only (bernisse). The
   high→max score inversion (94 high / 91 max) reproduces across stacks — it is model
   behavior, not stack artifact (rle, post 934, both TF EXL3 and eugr W4A16).

Actionable for this line, in order: (a) the effort-tail + tool-result template fixes
against TC-51/TC-92 on our line — the only known +2-3-point lever that is pure stack;
(b) check the v1.1.8 base for #1163/#1211/#1218/#1168/#1176 before the next perf
window; (c) the router-pathology fix remains ours to land; (d) the fork's settings
(RANDOM_SEEDS/KEEP_EFFORT/CACHE_ENTRIES) are worth an A/B on the standing config.

## Upstream v1.1.9 review (round 83, 2026-10-29) — our pin is v1.1.8 (812fb040)

Tag-to-tag diff audited (v1.1.8...v1.1.9: 50 commits, 256 files, zig/src + zig/kernels
concentrated). **What 1.1.9 gives this line** — three GLM-5.3-Flash CUDA items, all
bit-exact ("same picks in the same order" / "replies are the same tokens"), all
authored by @Bizuayeu (the fork owner from our watch list — the community is
maintaining our exact topology upstream):

1. **#1404 — DSA indexer split across ranks (TP≥2)**: from position 2048 each rank
   scores/picks pools for its share of a prompt chunk's rows, all-gather shares the
   picks; `TF_GLM_INDEX_SPLIT` tunes/disables. Measured **1–2% prefill at 128k on
   2×GB10 TP=2** — our exact topology, but noise-level against the Front 2 A2 prize.
2. **#1424 — DSA select decode widening**: 1,024-thread blocks at long positions over
   4,096-pool buckets, ≤48 rows. +1.8% decode after a 136k prompt at TP=3; **flat at
   TP=2/65k**. Our 512k×4 window sits inside the covered range; expect small-positive.
3. **#1352 — `TENSORFOLD_PREFILL_SHARE`**: a GLM prompt step ends on the GPU before
   returning when decoding streams are billed for it (timing, not tokens). A fairness
   lever for PARALLEL=4 decode-under-long-prefill; opt-in below 1.

**What 1.1.9 does NOT give us**: no router-GEMM work (the Front 2 A2 pathology
remains ours — `cuda_mm.zig`/`weights.zig` untouched; the Kolibri #1460 routed-expert
weight-copy fix is that family's kernels, the same CLASS of L2 pathology but not our
code path); **no fp8×KV-pages marriage** (the ledger watch stays open); no GLM
vision/template/effort changes (the effort-tail adoption is untouched); the expert-kernel
item (#1351) explicitly compiles GLM's kernels to the same code as before.

**Verdict**: a hold-and-fold-in, not a priority. ~1–2% prefill + 0–2% decode does not
justify a dedicated re-qualification window (the fingerprint moves, so §2 gates +
depth + hardmode re-verify). **The efficient path: base the router A2 patch on
v1.1.9** — A2 already forces an image rebuild and a full re-qualification, so one
window adopts both; if A2 slips, take 1.1.9 as a cheap standalone fold-in later.

## Community performance sweep (round 88, 2026-10-29) — the MiaAI-Lab fork's patch table

Upstream main has moved past v1.1.9 (36470cec vs the 465e48b6 tag) — untagged GLM work to
scan before the next pin decision. The big find: **MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold**
— our exact model on 2 Sparks (TP=2), 1,940–1,980 tok/s prefill at 8k–32k (1,641 at 262k;
our slope 0.659 ms/tok ≈ 1,517 tok/s), carrying a measured patch stack. Ranked for this line:

1. **The HC-split/overlap set** (`0010-glm-hc-split`, `0033-glm-prefill-overlap2`,
   `0017/0022-overlap-priority`, `0064-glm-split-connect-early`): hyper-connection glue
   split by rows between the ranks, exchanges overlapped with the next rows' work —
   **50k prefill ~1,270 → ~1,730 tok/s (+36%); decode pays ~1.5%**. TP=2-native, our
   topology, and the same fix CLASS as our 1100 ms mystery (bubbles/serialization →
   overlap is what works). The biggest prefill lever found anywhere.
2. **`0086-glm-prompt-scores-loop`** (BadAd84): the DSA indexer's prompt scoring takes one
   row + 32 pool blocks per program instead of reloading each row's queries per 64-pool
   block — **scoring 36% faster; cold prefill at 226k 138.3 → 126.6 s; the same bits**.
   Bit-exact long-context prefill — directly our slope, adjacent to the section our prof
   shows but does not itemize.
3. **The KDA chunked set** (`0012/0014/0039`): the linear-attention layers' prompt chunks
   in chunked WY form, one kernel — **50k prefill 29.3 → 26.4 s; 149k 91.1 → 84.5 s**.
4. Carried already on our side: dense fp8/q4 (we serve dense4fp8), lean prompt scratch.

Also: serapis (upstream) invites PRs on the forum thread ("I'll happily consider it"); the
fork's `FILL_BUDGET_MS`/`FILL_DRAFTS`/`STREAM_SMOOTH` decode-gap tuning table (gap p50
630 → 81 ms with budget 200 + drafts) is the lane-host fairness knob our PARALLEL=4 ITL
p95 could reuse.

**Actionable, in order**: (a) scan upstream main 465e48b6..36470cec for these graduating
(the credit style suggests they do); (b) if absent, port 0086 first (bit-exact, smallest
diff, our exact pathology neighborhood) and A/B it same-session; (c) the HC-split set is
the +36% prize but multi-patch — port behind a window with the full battery; (d) the ncu
capture rerun stands — the overlap findings strengthen the bubbles hypothesis.

## Round 94 research addendum (2026-10-30)

- Upstream 465e48b6..cefb72fd: Nemotron CUDA (#548), Qwen3.8-27B pair-seam kernels (#1432),
  and a new CORE primitive — the row-split pair seam (e06a6eef, "split simd_qmm over K"):
  row-splitting between ranks as core infrastructure. When the HC-split/overlap set
  graduates, expect it to land through this primitive — re-check the seam's GLM coverage
  before porting hand-rolled fork patches.
- Fork newest (10-25): v0.6.0 published image pinned at 123 patches; +0128 (spill tier in
  blocks shared by later turns), +0129 (KV sharding — per-conversation latent cache on the
  owner Spark). Serving-memory, not prefill; relevant to long multi-agent sessions.
- Forum thread (384789, through Oct 13): the engine's deterministic high-context looping
  pathology (long-form generation repeats a paragraph; identical payload fails identically;
  7 restarts/hours vs the EXL3-vLLM engine's one loop in 22+ h). Watch upstream for a fix
  before trusting long-form receipts at depth. tool-eval-bench TF support is upstream
  (PR #190); the community hardmode protocol matches ours. jordanm82's recipe transplants
  o_proj ("dealign", layers 15-44) with latent FP8 — untested pattern.
- **Engine-lineage correction (round 99)**: the MiaAI-Lab fork's ENTIRE patch stack
  (0010/0033/0086/0064/0128/0129 — all `.py`: comm/decode/forward/hcsplit/engine/sparse)
  targets TensorFold's PYTHON/Triton engine. Our testing line is the NATIVE CUDA engine
  (tf_glm_* kernels) — the fork's patches do not port. Their +36%/0086 numbers describe
  the python engine's ceiling for this model on 2×Sparks, not ours. Upstream's row-split
  pair seam (e06a6eef, native-core: pair_stage/pair_gather/pair_argmax/pair_pick) is the
  native-shape overlap mechanism to watch for GLM adoption.
