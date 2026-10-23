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

## Upstream PRs that bear on our two gaps (checked 2026-10-23, all open)

The concurrency and performance gaps may not stay ours to diagnose — three open
glm/cuda-family PRs target exactly the mechanisms behind them:

- **#1163 + #1211 — glm/cuda: KV pages from the core pool** (`TF_KV_PAGES`). Today every
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
