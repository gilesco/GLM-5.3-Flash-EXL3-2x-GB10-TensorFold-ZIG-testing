# Launch runbook — read this before touching the machines

Handover document for whoever attempts the first launch of this testing stack. The site's
own verified facts (hosts, addresses, driver, disk, caches) are **not in this repository** —
they live in `docs/SITE.md` (git-ignored; see below). Everything else below was verified on
2026-10-16 by the session that wrote this repo; re-check anything that looks different
before acting.

## 0. The situation

- **The site runs the PRODUCTION stack on the same two GB10 machines (128 GB unified memory
  each, aarch64)**: the EXL3/Python TensorFold recipe in containers named `glm53-flash-tf`
  on both hosts. It serves GLM-5.3-Flash EXL3 @ `078455ff` with DFlash2 on **port 8000** —
  rank 0 on the head, rank 1 on the worker, rendezvous on the head's CX7 rail address.
- **This repo is the testing line** (native engine, EXL3 only — the NVFP4 route has its
  own repo; no patches — see `../README.md`
  and `docs/DELTAS.md`). Its launch **requires both GPUs free**, so **production must be
  stopped first**. Stopping the agent session does NOT stop the containers; someone must stop
  them explicitly (step 4).
- Do not modify anything under `~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold*` on the
  machines, and do not delete their HF cache — that is the rollback path (and production
  serves from it).

## 1. Facts that are part of the recipe

| Fact | Value |
| --- | --- |
| Arch | aarch64 throughout (arm64 images and builds) |
| Toolchain on hosts | none (no zig, no nvcc, no system libnccl) — hence the containerized build; NCCL is bundled into the runtime image |
| Checkpoint | default `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `078455ff…` + drafter `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e…`, served from the hosts' `~/.cache/huggingface` (the EXL3 one is production's checkpoint: `quant_method: exl3`, 4-bit mcg routed experts). Fallback route: `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e…` |
| zig 0.17.0 aarch64 tarball | reachable at ziglang.org (verified) |
| Site identity | `WORKER`, the rail addresses and netdevs — set in `scripts/local.sh` (git-ignored); `docs/SITE.md` records one site's real values |

## 2. Launch (on the head)

```sh
# 1. get the repo onto the head
git clone https://github.com/gilesco/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing
cd GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing
cp scripts/local.sh.example scripts/local.sh      # fill in this site's values
                                                  # (docs/SITE.md lists one site's real ones)

# 2. resolve version, build the image (LONG the first time: Zig + nvcc for sm_121),
#    smoke-test it, copy it to the worker, write the prepared marker
./scripts/prepare.sh

# 3. inspect what will run (needs nothing, changes nothing)
DRY_RUN=1 ./scripts/start.sh

# 4. STOP PRODUCTION on both hosts (from the production checkout on the head)
~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold-tip/stop.sh
# verify: no glm53-flash-tf container on either host, and
#   nvidia-smi --query-gpu=memory.used --format=csv   shows both GPUs nearly idle

# 5. launch: rank 1 on the worker first, rank 0 here, then wait for /health
./scripts/start.sh

# 6. verify
curl -s http://127.0.0.1:8000/health ; curl -s http://127.0.0.1:8000/v1/models
./scripts/status.sh
```

`start.sh` refuses to run while unified memory is mostly held (MemAvailable < 30 GB) or the
port is taken, so a launch over a live production stack fails in seconds with a named reason.

Clients point at **port 8000** (`http://<head>:8000/v1`, model `GLM-5.3-Flash-EXL3` —
`SERVED_NAME`) — the
same port production serves, deliberately, so clients keep one endpoint when switching
engines. The consequence: **the two stacks are mutually exclusive** — stop one before
starting the other (`start.sh`'s port guard enforces it).

## 3. When something fails

| Symptom | First response |
| --- | --- |
| `prepare.sh` build fails | Read the docker build log — `image/build-native.sh` echoes every step. Check free disk, and that `BASE_IMAGE` (default NGC 26.07) is present. A changed upstream build.zig may have renamed `-Dsm`/`-Dnvcc` — the script falls back automatically; if both forms fail, read `packaging/README.md` at the pinned tag. |
| `could not start rank 1` | `ssh "$WORKER" true` from the head (key-based ssh), and `docker info` there. |
| A rank exits during start | `docker logs glm53-native-tf` (rank 0) and `ssh "$WORKER" docker logs glm53-native-tf` (rank 1). The ranks verify each other's build/cut/checkpoint bytes before loading and name any mismatch. |
| NCCL timeout / rank waits forever | `ping "$WORKER_FABRIC_ADDR"` from the head; add `NCCL_DEBUG=INFO` to `scripts/local.sh` and restart; check the rails are up (`ip -br addr` shows every netdev from `NCCL_SOCKET_IFNAME` UP on both hosts). |
| `/health` never comes up | Both logs (see above); the engine's admission names a rank and its reason when memory is short. `HEALTH_TIMEOUT_S` (default 900) can be raised for the first cold load. |
| libnccl/libstdc++ errors from the binary | The runtime image bundles libnccl only; if `--version` worked in prepare but serve complains, note the missing soname from `ldd` output and add it to `image/build-native.sh`'s collect step. |

Log everything you see into a new section at the bottom of this file (append, never rewrite
history) — the next session starts from your notes.

## 4. Rollback (restore production)

```sh
cd /path/to/this/repo && ./scripts/stop.sh          # stops both testing ranks, saves logs
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold-tip && ./start.sh
# verify: docker ps shows glm53-flash-tf on both hosts, port 8000 answers on the head
```

The testing stack shares nothing with production except the port (both 8000, by design —
only one runs at a time): different image (`glm53-native-tf:*` vs `tensorfold-glm53:*`),
container name (`glm53-native-tf` vs `glm53-flash-tf`), state/log dirs. Optional cleanup to
reclaim ~191 GB per host:
`docker rmi glm53-native-tf:<tag>` and remove
`~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4` (**do not delete any
checkpoint blobs** — the NVFP4 line and production serve from the same caches).

## 5. After a successful launch

Work through [VALIDATION.md](VALIDATION.md) in order (exactness → capacity → speed →
clients) and record results in its table. Until then, this stack is a curiosity, not a
server.

## 6. Session log (append-only)

- **2026-10-16** (session that built this repo): repo created and pushed at `46f05e5`;
  shellcheck clean; DRY_RUN/marker/version-resolution flows tested from the workstation;
  both checkpoints verified admit-compatible and complete; builder base verified
  (nvcc 13.3 + NCCL, aarch64); no GPU work attempted (production was serving).
- **2026-10-16 first launch: SUCCESS.** Production stopped cleanly; prepare built
  `glm53-native-tf:v1.0.12` (first build only ~3 min on the Grace CPU) and distributed it to
  the worker; both ranks verified each other's checkpoint bytes and **loaded in 229.3 s**;
  `/health` ok (`backend cuda`, `NVIDIA GB10`, model `GLM-5.3-Flash-NVFP4`), peak memory
  100.6 GB (rank 0, `--context 40960 --parallel 4`) — ~21 GiB spare. Inference works
  end to end; measured ~22 tok/s drafted on 512-token completions (thinking on). Notes:
  - fixed live: `worker()` passed no remote command — a bare ssh ate stdin (the docker-save
    tar reached the worker's bash; commit `c8066c0`) — re-run prepare after the fix; the
    docker cache made it instant.
  - GB10's nvidia-smi reports memory as `[N/A]`; busy-checks use MemAvailable (`6e0a22b`).
  - container `docker stats` MEM is meaningless for device allocations on unified memory —
    use the engine's `/health` `memory` block.
  - `reasoning_effort: low` is advertised in `/v1/models` but the model still thought in the
    probe — pin down the no-think path in the validation phase.
  - over-ssh scripts: background with `< /dev/null` or ssh hangs on inherited stdin.
- **2026-10-16** port default changed 8888 → **8000** (same endpoint as production, so
  clients never reconfigure when switching engines); the stacks are now explicitly mutually
  exclusive and `start.sh`'s guard enforces it. The running instance was restarted onto 8000
  and verified healthy (`/health` ok, 100.6 GB active, the old port closed).
- **2026-10-16 Phase 1 perf campaign started** (`PHASE=perf`, `--perf-only`): baseline
  (p4) / recall (p4) / parallel6 (p6) / recall6 (p6). Findings so far:
  - llama-benchy needs `--tokenizer nvidia/GLM-5.3-Flash-NVFP4` on an offline host — the
    served name is not a Hub id; fixed in the fixed bench options (`34fcb46`).
  - a `pkill -f` whose pattern appears in the same remote command kills the invoking ssh
    shell (self-match) — kill and launch must be separate ssh calls, bracket-quoted.
  - baseline memory under load: 100.64 GB active at rest → 107.29 GB while llama-benchy
    fills KV; MemAvailable 18.7 → 12.2 GiB. Healthy margin, peak tracking works.
- **2026-10-16 first perf numbers (recall config, p4 @ 40960):** the standing 512-token
  probe, three runs: recall cold 24.4 s (~21.0 tok/s, drafts 342/445) — baseline-equal;
  recall warm 17.5/17.6 s (**~29.4 tok/s, drafts 383/383 accepted**) — **+29% over the
  baseline probes (23.45/23.59 s)** with identical output sha (exact). Recall validated as
  the top no-code optimization for repeated/agent traffic; fresh content decodes at
  baseline speed. Remaining Phase-1 matrix (llama-benchy pp/tg per config) still to run.
- **2026-10-16 long-context attempt (halted): 262144/p1 overshoots.** Restarted at
  `--context 262144 --parallel 1` with default knobs (reserve 12.2, prompt cache 3.9): the
  engine's model predicted ~6 GiB slack, but at load+prefill MemAvailable fell to
  **0.3 GiB** — the real usage ran ~6 GiB past the engine's accounting (the long-context
  overshoot; indexer pool-scan scratch at depth is the suspect). Halted per resource
  caution; memory recovered to 117 GiB after stop. Next attempt must start at the ladder's
  first rung (131072, with the trimmed knobs reserve 8 / prompt cache 1.5) and measure the
  overshoot before climbing. **Nothing past 40,960 has served yet.**
- **2026-10-16 memory model extracted** (read-only, from the startup log + config.json):
  model 93.72 GiB/rank; **0.89 GiB per stream at a 40,960-token window (≈22.8 KiB/token,
  bf16 latent + indexer pools, replicated on both ranks)**; 12.2 GiB reserve; 3.9 GiB prompt
  cache; 6.5 GiB free at --parallel 4. 45 layers = 33 KDA + 12 DSA; indexer selects top-2,048
  tokens/row (attention cost flat with context; the index scan grows). Used for the 500K
  feasibility math (docs/DELTAS.md): **500K fits only at --parallel 1** (and only with the
  reserve trimmed ~2 GiB); --parallel 2+ does not fit on bf16 KV.
- **2026-10-16 vision review** (docs/DELTAS.md said CUDA vision was missing — confirmed): no
  upstream PR or issue exists for GLM-5.3-Flash CUDA vision; the nearest precedent
  (#652/#653, "dense path ports to a MoE family in ~70 lines") is a draft against the
  **python-0.6** engine (not native Zig), unmerged and closed. Native CUDA vision exists
  in-tree only for Nemotron Super VL. Decision: do not fork the engine for this — vision
  workloads stay on the production stack until an upstream PR lands; a proposal issue on
  ashhart/TensorFold is the cheap accelerator (our pair is the TP=2 test hardware the gate
  needs). Filed as
  [ashhart/TensorFold#971](https://github.com/ashhart/TensorFold/issues/971) (proposal +
  test-hardware offer, LLM-agent-posted on gilesco's behalf and approved by them, per their
  request; no implementation commitment).
- **2026-10-18 rebase to TensorFold v1.1.0 + the EXL3 pivot (prepared, NOT started).**
  Upstream's latest release, v1.1.0 (tag `76aafcf2…`), resolved by `TF_REF=latest-v1`; image
  `glm53-native-tf:v1.1.0` built on the head (Zig 0.17.0, sm_121, ~4 min compile) and
  distributed to the worker; prepared marker written. **Nothing was started** — production
  serves port 8000 and was never touched; `start.sh` is the manual next step. Changes:
  - **Checkpoint pivot, NVFP4 → EXL3.** v1.1.0's known issues name our old route: the NVFP4
    two-rank path was *not run upstream for the release* ("the weights leave no room for its
    streams beside the memory reserve" on two GB10s). The EXL3 two-rank path is the one they
    measured (#855/#976; DFlash2 in #979). config.sh now defaults to
    `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold@078455ff` (production's pin, in both
    caches) and passes `--drafter` with DFlash2 `incoai/…@bf582e4e` when its snapshot is
    present (DRAFTS=0 still passes `--no-drafts`). The NVFP4 route stays one
    `MODEL_ID`/`MODEL_REVISION` override in local.sh away.
  - **The production recipe's `zig-engine` branch was evaluated and rejected**: it pins
    TensorFold v1.0.7 + ~326k patch lines + a captured Triton AOT set, is unqualified by its
    own docs, and 1.1.0 absorbed its goal upstream. What was borrowed: its CONTEXT=131072 /
    PARALLEL=8 defaults for this checkpoint family on GB10s (noted in local.sh.example).
  - **Cosmetic quirk:** the binary built from the v1.1.0 tag still prints
    `tensorfold-native 1.0.12` — upstream did not bump `.version` in build.zig.zon at the
    tag. The pin is the commit sha; ignore the version string.
  - **Repo moved** to `gilesco/GLM-5.3-Flash-EXL3-2x-GB10-TensorFold-v1.1-testing` (full
    history); the old NVFP4 testing repo stays untouched as the v1.0.x record.
  - **Site details moved out of the repo**: hosts, addresses, driver, ssh targets and disk
    facts now live in `docs/SITE.md` (git-ignored); config.sh requires site identity from
    `scripts/local.sh` instead of shipping this site's values as defaults.
  - Not yet done here: first start on 1.1.0 (memory fit at 131072 unmeasured on this pair;
    ladder rules from 2026-10-16 still apply), the EXL3-context of docs/DELTAS.md and
    docs/VALIDATION.md.
- **2026-10-18 first EXL3 launch on v1.1.0: SUCCESS, then swapped back to NVFP4.** Loaded in
  **843.6 s**: the new-in-1.1.0 digest verify read 163.6 GiB in **624.8 s (~262 MB/s, cold,
  effectively single-threaded — upstream's 199 s receipt was on faster storage)**, then the
  weights load. **EXL3-native memory model (both ranks, from the startup line):** model
  **86.11 GiB/rank**; **0.71 GiB/stream at 40960** (NVFP4 was 93.72 / 0.89); 12.2 GiB
  reserve; **11.2 GiB prompt cache**; **13.3 GiB free at p4**. DFlash2 loaded: 0.57 GiB on
  device, linears at 4 bits; one 128-token probe drafted **84/99 accepted**. Perf verdict
  from the operator: "uninspiring" — **no tok/s was measured before the swap back** (the
  probe attempt hit the port after the NVFP4 stack returned), so the EXL3-native baseline
  number is still missing and is the first item of the next test window. Operational notes:
  - `docker logs` is **empty during the digest phase** (buffered stdout) — do not mistake
    for a hang; the liveness signal is the read counter:
    `docker exec glm53-native-tf sh -c 'cat /proc/1/io'` climbing on both ranks, in lockstep.
  - The engine's memory accounting looks conservative vs the NVFP4-era overshoot finding
    (13.3 GiB free claimed at p4) — trust but verify at the first ladder rung.
  - Recipe updates from this session: the digest cache is now a mounted volume
    (`$STATE_DIR/tensorfold-cache:/root/.cache/tensorfold` — saves the 624.8 s re-read on
    every restart); `DENSE_BITS`/`HEAD_BITS`/`PDL` knobs added (v1.1.0's pack-at-load and
    programmatic dependent launch); the bench matrix grew to seven configurations
    (baseline, pdl, dense8, parallel6, parallel8, recall, recall6) — one variable each.
- **2026-10-18 the first EXL3-native numbers (matrix run-20261018-174541, baseline row).**
  Config: v1.1.0, EXL3 + DFlash2, `--context 40960 --parallel 4`, server `GLM-5.3-Flash-EXL3`
  on port 8000 (NVFP4 stack stopped first — the matrix's own stop.sh did it); llama-benchy
  pp2048/tg128, depths 0/4096/8192, concurrency 1/2/4, 3 runs, --no-cache:

  | Cell | pp tok/s | tg tok/s | TTFT |
  |---|---:|---:|---:|
  | d0 solo | ~657* | **21.2** | 3.1 s |
  | d0 c4 | ~598 | 25.3 agg | 8.4 s |
  | d8192 solo | ~652 | 22.1 | 15.7 s |
  | d8192 c4 | ~625 | **8.6 agg** | 43.8 s |

  (\* prefill estimated from e2e_ttft; llama-benchy#33's early-chunk caveat.) Reading: decode
  ~21 tok/s solo is **below the NVFP4 route's drafted ~28-33 and a third of production's
  ~62** (Python engine); prefill ~625 tok/s is ~⅓ of production's ~2,050 and under half the
  NVFP4-native's ~1,440. Concurrency barely scales at d0 (c4 aggregate 25.3, +19% over solo)
  and **collapses at depth** (d8192 c4 aggregate 8.6 — streams starve each other). The rails
  are NOT the bottleneck (measured ~2.6 GB/s per rail, ~10% of line rate; pinned
  NCCL_SOCKET_IFNAME verified inside the live container). The knobs matrix (pdl, dense8,
  parallel6/8, recall) tests whether scheduling/packing helps. Operational: the bench's
  progress events sit in Python's 8 KB stdout buffer through the pipe until exit — the
  server's `done chatcmpl` count in docker logs is the true progress signal; and the tty-ful
  fix (PYTHONUNBUFFERED=1 in the matrix) waits for the matrix process to end first (bash
  streams the script it runs — never edit a script mid-flight).
- **2026-10-18 knob matrix complete (run-20261018-174541, all 7 configs, perf-only).**
  tg tok/s cells (d0 solo / d0 c4-aggregate / d8192 solo / d8192 c4-aggregate); pp ~590-660
  unchanged in every config:

  | config | d0 solo | d0 c4 | d8192 solo | d8192 c4 | model GiB |
  |---|---:|---:|---:|---:|---:|
  | baseline | 21.2 | 25.3 | 22.1 | 8.6 | 86.11 |
  | pdl | 21.6 | 24.9 | 22.3 | 8.2 | 86.11 |
  | **dense8** | **24.0** | **26.6** | 22.3 | 8.7 | **82.31** |
  | parallel6 | 21.6 | 25.6 | 19.7 | 8.5 | 86.11 |
  | parallel8 | 20.8 | 25.6 | 18.1 | 8.5 | 86.11 |
  | recall | 20.1 | 24.1 | 20.0 | 8.5 | 86.11 |
  | recall6 | 21.0 | 24.4 | 19.9 | 8.6 | 86.11 |

  Verdicts:
  - **dense8 is the only winner**: solo decode +13% (21.2 → 24.0), c4 +5% (25.3 → 26.6), and
    it frees memory (model −3.8 GiB; free-after-reserve 13.2 → 17.5 GiB; prompt cache 11.1 →
    15.4 GiB). Denser packing is the follow-up — `dense4`/`dense6` rows added to the matrix,
    with a quality check owed for `dense4`.
  - **pdl: noise-to-inconclusive** (d0 +0.4, d4096 solo +2.7 but d4096 c2 worse, d8192 flat)
    — the knob stays, don't count on it.
  - **parallel6/8: no gain, slightly negative at depth** — the concurrency ceiling is not
    admission-side; extra lanes trade prompt cache for nothing (11.1 → 8.5 GiB at p8).
  - **recall: unrewarded by this bench's shape** — it runs `--no-cache` on fresh prompts
    (cache hits stayed 0), and the +29% NVFP4-era figure was warm repeats. Not refuted;
    just not this measurement.
  - **The deep-prompt concurrency collapse is structural**: d8192 c4 sits at 8.5-8.7
    aggregate in EVERY config, d4096 c2 at ~14-15 — no shipped knob touches it. That is
    per-layer exchange/verify-window contention, i.e. upstream work; 1.1.1's planned GLM
    CUDA fresh-prompt batching and prompt scoring (#1014-#1018) is the nearest candidate.
    Line's next move: re-run the matrix there.
- **2026-10-18 investigation: why v1.1-native trails the Python engine, with receipts.**
  The operator switched production back to Python 0.6.6 EXL3 (its decode/preill lead held
  over every v1.1 config we measured) and asked what upstream says. Findings, by strength:
  - **#1033 (GB10-specific, config-level fix available)**: the native engine spins two CPU
    cores per rank (CUDA sync — `CU_CTX_SCHED_BLOCKING_SYNC` would stop it at -12% decode;
    and NCCL's progress thread — spins regardless), and on a GB10 five of seven ACPI thermal
    zones are the CPU's. As-shipped, a 40-reply run hits the 94.7 C heat watch and stops at
    the 7th reply; **pinned to the efficiency cores (0-4, 10-14) it runs 40/40 at -1.1%
    decode / -2.7% prefill**. Recipe response: `CPUSET` knob added to config.sh
    (→ `--cpuset-cpus` on both ranks; empty = as-shipped). Set `CPUSET=0-4,10-14` in
    local.sh for long runs.
  - **#918 (structural, open since 1.0.3, cross-backend)**: an independent team shadow-benched
    native vs Python 0.6.6 on Metal 27B: sampled decode -21% to -34%, 8-stream -17% to -45%,
    **cold prefill 1.7x slower at 1.0.7 and again at 1.0.8** — and 1.0.10's prefill fix
    (simd_qmm) does not reach uniform checkpoints. Our GB10 GLM gap (decode ~-64%, prefill
    ~-70% vs production) is that gap plus greener EXL3 CUDA kernels plus the GB10 spin.
  - **#840 (the official carry-over tracker)**: the native engine is behind the Python engine
    by design and rows are being ported. The perf-relevant one: **`--decode-share` (#248) is
    an unclaimed row — prefill does not block replies in the Python engine, and native has
    no equivalent**. That is our d8192/c4 collapse (8.6 aggregate in every config): a
    concurrent deep prefill starves decode. Metal's native analog is #1032.
  - **#993 (the Python engine's own GLM TP=2 bug)**: with uptime, prefill decays to 18.8 s
    for 13 tokens and decode 40 → 14 tok/s, an idle rank 1 reads 96% GPU (the #305
    idle-follower spin), and a restart clears it. Production is not immune — worth
    mentioning to whoever runs it long.
  - **Mia's zig-engine branch adds no missing magic**: its own qual notes say decode at 8
    streams is *under* the Python engine even with Python-derived captured Triton kernels —
    the native lane-core round architecture is part of the ceiling, not just kernels.
  Net: the line's thesis stands (upstream fixes land by bumping), but the gap is real,
  multi-issue, and partly deployable-around (CPUSET now; --decode-share and prefill parity
  wait for upstream). Worth offering our receipts to #1033 (they asked for a direction
  preference; we have an independent GB10 pair and EXL3-v1.1.0 data).
- **2026-10-19 fold in v1.1.1 (released today; prepared, NOT started).** v1.1.1 is the
  release this line was waiting for — what it carries, and what it changes:
  - **GLM CUDA batched fresh prompts (#1014-#1018 landed)**: stepped serving interleaves
    fresh prompt chunks with running decode, and whole-prompt serving can batch eligible
    fresh text prompts across ranks (≤16 prompts, 2,048 rows per pass). This is the direct
    candidate for the d8192/c4 collapse (8.6 aggregate in every 1.1.0 config) — the matrix's
    baseline row on 1.1.1 is the first comparison to make.
  - **Bits move in 1.1.1**: prompt projections use one K slice at every row count by default
    and "can change replies versus 1.1.0". All 1.1.0 matrix numbers are therefore a stale
    baseline — re-run `baseline` first. `PROMPT_K=shape` knob added (→ `TF_GLM_PROMPT_K`)
    for the A/B; it needs `TF_GLM_MM=1`, the engine default.
  - **#1033 resolved our way**: "GB10 efficiency-core pinning is documented in the runbook";
    the engine-level spin fix stays open — so `CPUSET=0-4,10-14` remains the remedy and now
    has upstream's endorsement.
  - **#1040: the NVFP4 two-rank route loads and serves on 1.1.0 on two GB10s** (community
    run) — the fallback route's known issue is cleared, though "scored prompts and more than
    two ranks are untested".
  - Also: the version string is fixed ("source builds report the correct release version");
    new Host-header security (every request needs one valid `Host`; loopback binds accept
    loopback IPs — our curl/health checks pass, but a client sending an unusual `Host` needs
    `--allowed-host NAME`); `#1030`'s missing-library errors now name the library.
  - Fold-in: config.sh's PROMPT_K knob; `scripts/prepare.sh` re-run on the head resolves
    latest-v1 → v1.1.1 (zig still 0.17.0), rebuilds + distributes the image, updates the
    marker. Production untouched; nothing started.
  - **Prepare receipts (17:28)**: v1.1.1 @ `b10231b9` built in 225 s and distributed to the
    worker; smoke `tensorfold-native 1.1.1` (version string correct at last); prepared
    marker updated; the local.sh WORKER changed to the worker's rail IP in the
    same window (SITE.md updated to match).
- **2026-10-19 ecosystem review: what the other GLM-on-GB10 repos know that we didn't.**
  Repos found (GitHub): jayleaton/glm53-tensorfold-spark (115★, 77 patches, an upstream
  contributor), TechGuardCoders/glm53-tensorfold-spark-recipe (one-command install on top of
  jayleaton's), Bizuayeu/GLM-5.3-Flash-NVFP4-DGX-Sparks-BIZ (the #1040/#1033 authors),
  MiaAI-Lab's 3x-Spark and TP4 lines, Aevonix's RTX PRO 6000 recipes, bertholomus's TP4 and
  DeepSeek recipes. What folds into OUR stock-engine line today (config/env only):
  - **TEMPERATURE knob added** — the ecosystem's "one setting most people miss": stock
    TensorFold drafts with DFlash2 on greedy (temperature 0) requests ONLY, so agent
    frameworks that omit temperature get 1.0 and no DFlash2 drafts. jayleaton measured
    4.36 → 5.73 tokens/step (~68 → ~83 tok/s) with `--temperature=0` on this hardware
    class; TechGuard ships it as their default. This also re-frames our matrix numbers:
    llama-benchy's sampled cells (t=0.7) ran without DFlash2 — the greedy cells are the
    drafted reference. Knob added (empty: engine default 1.0; clients sending their own
    temperature are unaffected).
  - **NCCL tunings adopted as env pass-through**: jayleaton's W19 measured set —
    `NCCL_IB_HCA` with both functions of the linked CX7 port, `NCCL_PASSTHROUGH=1`,
    `NCCL_MIN/MAX_NCHANNELS=4` — took a 4 MiB all-gather from 322 to 142 µs, prefill +1.5%,
    and freed ~1.4 GiB of NCCL memory. env_args now forwards every NCCL_* (and the
    TF_HEAT_* bands) from the environment to both ranks; the measured set is documented in
    local.sh.example. Our 2-cable site needs its own `ibdev2netdev` check (4 functions).
  - **TF_HEAT_* bands now pass through** — Bizuayeu's #1040 run used 92/88 and finished
    40/40 replies unthrottled; our CPUSET knob remains the #1033-documented alternative.
  - **What we cannot adopt without forking (and mostly shouldn't)**: jayleaton's 77-patch
    stack (fp8 latent KV, the shared 1M KV pool, NVMe session tier, L2 weight prefetch,
    deeper verify windows...) is where their 49-83 tok/s lives. Mapping it against stock:
    1.1.0/1.1.1 have since absorbed several pieces (structured output, required/named tool
    calls, batched fresh prompts, vision, DENSE_BITS ≈ their 4-bit non-experts), the rest
    belongs in #840's carry-over tracker. Our thesis (unpatched upstream) stands; their
    repo is the reference for what's worth asking upstream for.
  - **The NVFP4 route measured 2× our EXL3 on the same hardware class**: #1040's regression
    on v1.1.0, two GB10s, NVFP4, TP=2, 245760 context: decode median 45 tok/s (vs our 21-24),
    prefill 1,402 (vs our ~625), whole 199k prompt found. A checkpoint-choice data point,
    not a line change: EXL3 stays for production parity; the numbers are recorded in
    docs/VALIDATION.md's comparison table when the next window runs.
- **2026-10-19 quick window on v1.1.1: first launch + probes (production restored after).**
  - Launch clean: **loaded in 222.7 s** — the warm digest cache works (no verify pass).
    v1.1.1 memory model: model 86.12 GiB; 0.71 GiB/stream at 40960; **11.0 GiB free after
    the 12.2 reserve; 8.9 GiB prompt cache** — 1.1.1 holds **~2.3 GiB more** than 1.1.0
    (13.3/11.2); suspect the K-slice regrouping or the stepped-admission structures
    (unconfirmed). The ladder anchor in docs/CONTEXT_512K_PLAN.md is updated; 524288×1
    still fits (usable ~3.2 GiB after the margin).
  - **Greedy 512-token probe: 24.7 tok/s** — v1.1.1 ≥ our best 1.1.0 config (dense8's
    24.0), +3.5% over 1.1.0's baseline, still ~half of #1040's NVFP4 45.
  - **Sampled 0.7 probe: 25.9 tok/s — no temperature trap on stock 1.1.1**: sampled
    requests draft as fast as greedy ones. jayleaton's "temperature trap" is their patch
    0010 (auto-deep-dflash-drafts), not stock behavior. The TEMPERATURE knob was removed
    from local.sh the same window (receipt note left in place).
  - **Probes 3 (prefill) and 4 (exactness) were botched by the probe design, not the
    engine**: probe 3's "7.5k-token" prompt was actually ~41k tokens (over the 40960
    window — the server rightly refused, hence no usage); probe 4's max_tokens 32 was
    fully consumed by thinking, so content was empty and the sha was the empty-string
    hash. Corrected probes issued (14k-token prompt; 512 tokens; hash the whole message).
    **The exactness gate has NOT yet passed on v1.1.1** — first item next window.
  - Operational: production stop/start uses the **-tip** checkout
    (`~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold-tip`); the non-suffixed directory
    exists on the head and is not the live one (paths fixed in the docs and SITE.md).
- **2026-10-21/22 review window (no GPUs touched; production ran throughout): v1.1.2
  reviewed + the RoCE root cause found and fixed in the image.**
  - **v1.1.2 review** (release notes + shallow source read, tag commit `4ff78441...`;
    Zig stays 0.17.0). What it gives our EXL3 TP=2 pair:
    - **#1053 — waiting GLM ranks sleep instead of polling at full load**, waking for
      rank 0's next step. This is the #1033 follower-spin fix; may make our CPUSET knob
      unnecessary — test next window.
    - **#993 fixed for GLM CUDA**: committed prompt prefixes kept in bounded,
      preallocated snapshot pages inside the prompt-cache budget — the uptime-decay
      fix, native side.
    - **`TF_GLM_PROF=1` diagnostics**: per-rank prompt-chunk stages, transport phases,
      event waits — the tool for the collapse/exchange investigation (passes through
      our env_args).
    - **FP8 latent cache prepared behind `--kv-cache fp8`** (BF16 default; native
      quality and device qualification pending) — the 1M-window unlock exists in stock
      now; WE are the natural qualification pair for our 512K ladder.
    - Shared-lane scheduling: spare rows go to other streams when a forced window does
      not fit; shared prompt scheduling prefers admitted prompts with less work left;
      #1059 adds one deeper draft row every eighth round. TP=3 KDA projections (#1067)
      land (not our topology). Security/HTTP hardening (header/body time bounds, 128-
      connection cap, key-file/perms). Tool params decode as JSON (#1121);
      `--thinking-reserve`, `--loop-guard` for repeated reasoning.
  - **RoCE root cause PROVEN — we never used the RoCE path, and neither did the NVFP4
    media run.** Upstream's #971 closer (2026-10-21): our media receipts "used NCCL
    fallback; it does not qualify the separate RoCE media path." Investigation:
    - Stock engine wiring: `cuda_native.zig:188` calls `openRoce` unconditionally at
      GLM startup; `TF_ROCE` defaults to wanted (unset = on, both ranks must ask);
      HCA list from `TF_ROCE_HCA` else `NCCL_IB_HCA`. Two-rank small all-gathers
      (greedy picks, split-reduce, dflash) route through `group.allGather` → direct
      verbs writes when the gather is up; media rows cross ranks the same way.
    - **Saved rank logs contain `warning: RoCE all-gather: VerbsUnavailable (); NCCL
      carries every gather` on every launch** — `roce.Gather.init` dlopens
      `libibverbs.so.1` and our **runtime image ships neither libibverbs nor libmlx5**
      (we bundle only the binary + libnccl). Without verbs, NCCL's own IB transport is
      also unavailable → NCCL ran TCP sockets over the CX7, so our 2.6 GB/s "rails"
      were likely not RDMA at all. Plausible share of the prefill gap (625 vs
      jayleaton's 1,630 tok/s — their repo bundles libibverbs + the mlx5 provider).
    - `ibdev2netdev` on the head maps the two cabled CX7 HCAs to their rail netdevs
      (both Up; the f0 functions are the uncabled ones) — jayleaton's exact measured
      HCA set; this site's device names live in the site files, not the repo.
  - **Fix, three files**: runtime stage installs `libibverbs1 ibverbs-providers
    rdma-core` (distro 50.0 matches the host kernel's rdma-core 50.0 exactly; the
    builder's newer pair deliberately not bundled to avoid kernel-ABI skew. Gotcha
    caught on rebuild: Ubuntu's `rdma-core` package does NOT pull the provider
    plugins — without `ibverbs-providers` the image had libibverbs but no mlx5
    driver, and verbs could not open the CX-7s at all); `env_args` now forwards
    `TF_ROCE_*`; `local.sh.example` documents the `NCCL_IB_HCA` form (the DGX Spark's
    typical cabled CX7 HCA names) as REQUIRED for the RoCE gather + NCCL RDMA (with
    the ibdev2netdev discovery walkthrough). Site: `NCCL_IB_HCA` added to the head's
    `local.sh` and probed.
  - **v1.1.2 image built and distributed** (rebuild after the providers fix;
    compile stage cached): smoke `tensorfold-native 1.1.2` (the 1.1.1+ version-
    string fix holds); image on both hosts; marker
    `tf=v1.1.2@4ff78441... patches=` (pristine); `libmlx5-rdmav34.so` +
    `libmlx5.so.1.24` verified inside the image at
    `/usr/lib/aarch64-linux-gnu/libibverbs/`.
  - **Next window verification list**: rank logs must NOT contain "VerbsUnavailable";
    baseline probes re-run (24.7 greedy reference); corrected probes 3+4 still owed;
    `TF_GLM_PROF=1` transport phases optional.
- **2026-10-22 launch + 5-config matrix on v1.1.2 (user-driven window): everything
  moved; RoCE confirmed up on every restart.**
  - **RoCE gather up**: rank logs carry `info: RoCE all-gather: 2 HCA(s), shards up to
    4096 KiB by RDMA writes` (10×, one per rank per restart) and zero
    `VerbsUnavailable`/fallback warnings — the image fix + `NCCL_IB_HCA` worked. The
    #971 media path now has a qualifying EXL3 configuration on site.
  - **Matrix** (`results/run-20261022-154406/`; baseline/dense4/dense6/dense8/pdl;
    table in docs/TOOL_EVAL_TESTING.md): vs the v1.1.0 matrix, **prefill pp2048
    657 → 1,026 (+56%), solo decode 21.2 → 36.4 (+72%), TTFT d0 −36%, c4@d8192
    8.6 → 13.5 agg**. **dense4 is the new leader: 55.4 solo / 42.6 c4 / 1,053 pp,
    19.4 GiB free + 16.0 cache** — approaches production Python's 62–63 solo decode.
    Clean monotonic ordering in dense bits (unlike 1.1.0's dense8-within-noise). PDL
    marginal (+3.6 solo, worse at depth). Memory anchor back to 13.3/11.2 (1.1.1's
    11.0/8.9 transient — plausibly #993).
  - **Caveat — three changes bundled** (engine 1.1.0→1.1.2, RoCE RDMA, HCA set): the
    split needs a `TF_ROCE=0` control run (stock knob, one restart).
  - **Unchanged structural issue**: aggregate decode does not scale with streams
    (baseline 36.4 solo → 37.8 c4; dense4 drops 55.4 → 42.6; d8192 c4 13.5). Next
    diagnostic: `TF_GLM_PROF=1` at c4@d4096.
  - **Next: qualify dense4** (exactness gate + recall — 4-bit dense linears are a
    quality risk), corrected probes 3+4 on baseline then dense4, `TF_ROCE=0` control,
    NCCL channel set, then the ladder on the winner.
- **2026-10-22/23 search matrix digested (`results/run-20261022-172325/`; tables and
  verdicts in docs/TOOL_EVAL_TESTING.md).**
  - **dense4 alone stays the leader; nothing stacks on it cleanly.** dense4pdl −13%
    solo (PDL's sign flips with the config), dense4chan4 −10% solo, dense4h8 slower
    still. chan4 on baseline: no prefill gain, decode ≤ baseline at depth — its value
    is ~2 GiB of freed NCCL workspace. fp8: works (stream 0.71 → 0.48 GiB), ≈ baseline
    speed −3% — a memory knob with a small speed tax.
  - **Attribution answered by roce0**: the engine's RoCE gather adds ~5–10% solo
    decode; prefill is unchanged without it → the 657→1,026 prefill jump came from
    NCCL RDMA replacing TCP, not the gather. (Receipt for the next upstream note.)
  - **fp8 stream facts for the ladder**: per-stream at 40960 = ~0.25 GiB fixed +
    KV ≈ 11.8 KiB/token bf16 / 5.9 fp8. Recomputed rungs: every 512K iso-budget rung
    fits on every v1.1.2 config even at the plan's conservative 17.75 KiB/token rate
    (baseline 5.3 usable after margin; dense4 11.3); fp8 roughly halves rung costs.
    **The 512K constraint is the quality gates, not memory** — docs/CONTEXT_512K_PLAN.md
    updated with the per-config anchors.
  - **Next: `dense4fp8` cell added to the matrix** (leader + fp8 caches, the ladder
    candidate), then gates on dense4 and fp8, then the ladder on the qualified winner.
- **2026-10-23 forum thread read end-to-end (381350, 45 pages, Aug 26 → Oct 22) — the
  community has benchmarked OUR repos, and the end-of-thread findings reframe the plan.**
  User's reframe recorded: production is the `-tip` fork (Python 0.6.6, DENSE=fp8,
  KV=fp8, COMM=roce, CONTEXT=524288, DFlash2, admission 87.45/101.21 GiB, cache slots
  524304); we are NOT migrating to 0.6.6 — **the task is replicating the 0.6.x
  improvements on the 1.1.x native line**.
  - **ajvazan's 14-config consolidated benchmark (Oct 12, #809) includes our repos:**
    - **G647 = "TensorFold v0.6.6-tip + 106 patches (gilesco+MiaAI merge), DENSE=fp8 —
      the current production stack"**: tool-eval **91.00±0.00 high / 93.00±0.00 max**
      (best TensorFold-line max in the field), cold prefill 1,601 t/s, gen sweep
      **52.7 t/s** (2.1× Canada W4A16), **KV pool 2,287,616 tokens at 1M context**
      (deepest in the field with the q4 variants), Cyrillic depth test **93.9%**
      (199/212), **zero loops, zero contamination**. Called "the production pick".
    - **G644a/G = gilesco TF v0.6.4 q4-dense era**: 92/92, fastest TF decode
      (56.5–61.8 t/s sweep), but the Cyrillic depth test showed the **q4 degradation
      band at 192k–384k** (fact tasks 12→10/16, 9–14 cuts, 2 contamination rows).
      This validates production's 2026-10-07 DENSE q4→fp8 switch from outside.
    - Both TF stacks' weak cells: deep concurrency (d4096×c4 27.3, d8192×c4 18.3) —
      the concurrency cliff, now with community numbers. Strongest cells: 0-depth
      c2/c4 (67.6/59.6).
  - **Depth degradation is real on every fast stack**: fact-task accuracy collapses
    384k–480k (length-cuts with filler-echo) on fp8/q4/vLLM-NVFP4; only Canada W4A16
    marlin holds 100% at every depth (at 2.1× slower decode). **A depth-quality gate
    must be part of our 512K qualification** — ajvazan's Cyrillic depth test is the
    template (fact/needle tasks at 8k/64k/192k/384k, temp 0, exec-checked).
  - **The 1.1.x native line already has community traction**: giles8 (Oct 2): "Mia's
    GLM on TensorFold now feels like c4 with >100 tokens/s on 2 Spark — and the
    concurrency is REAL". bernisse (Oct 22): TensorFold **1.1.2 bare metal with fp8
    KV unqualified, fine for weeks — tool-eval 94/100**, the best EXL3 score in the
    thread. Direct community support for our `--kv-cache fp8` qualification.
  - **Bizuayeu (Oct 14)**: "TensorFold GLM 5.3 family native CUDA inference engine
    completed; detailed tuning and verification" + container re-measurement coming —
    upstream momentum on our line.
  - **MaTeZZ's kernel finding (Sep 30, #610)**: DGX OS 7.5/kernel 6.17 can run the
    Grace CPU at floor frequency (~1.38 GHz vs 3.9) with scaling_cur_freq lying; fix
    was the 7.0 kernel + cppc auto_sel_mode → **+8–10% decode**. Our hosts run
    7.0.0-1019 — likely fine, verify with the acpi_cppc feedback_ctrs probe. Also
    their mechanism for our concurrency cliff: **one structured-output request forces
    draft-length truncation off for the whole batch** (31.5 → 22 tok/s per stream) —
    test with/without a json_schema request in the batch under TF_GLM_PROF.
  - **Other notables**: DFlash2 is CC BY-NC-ND (non-commercial — production carries
    it; the MTP fallback is the licensed path); vLLM stacks cannot prefill and decode
    simultaneously (TensorFold can — our scheduling advantage); EXLR8 (new EXL3-
    trellis experts quant, kindlingai stack) posts 2,300 t/s prefill / 4.2M KV —
    vLLM-line competition, the NVFP4 repo's domain.
  - **The 0.6.x → 1.1.x replication map** (the plan's spine):
    | production (0.6.x) | 1.1.x native | action |
    |---|---|---|
    | CONTEXT=524288 | env-supported, untested at scale | ladder + parity cell |
    | KV=fp8 (qualified, drafted=serial) | `--kv-cache fp8` prepared, unqualified | gates on fp8 + dense4fp8 |
    | DENSE=fp8 | DENSE_BITS=8 | parity cell (dense8); community says fp8-dense > q4 at depth — mirrors our dense4-vs-dense8 gate |
    | COMM=roce | fixed this window ✓ | receipts logged |
    | DFlash2 | same drafter via DRAFTS=1 ✓ | nodraft cell quantifies |
    | fork knobs (SPLIT/SHARED_PREFIX/KDA_CHUNKED/COPY_CODE/MULTI_PREFILL/STREAM_SMOOTH, decode floor, hybrid prefix, loop guard) | upstream equivalents: #1053 sleeping ranks, #993 snapshot pages, shared lanes, prompt passes, --loop-guard | knob-by-knob mapping; gaps → upstream asks |

## 2026-10-23 — campaign script finished; ecosystem research pass

- **`scripts/run-qualification.sh` debug pass** (found on review, fixed before any run):
  a leftover `$prompt_combine` in the g3 probe would have killed the script under
  `set -u`; the g5 seed-body line never applied (req_file ignored `MSG_SEED_BODY`);
  curl timeout 240s would kill the 384k depth probe (prefill alone ≈ 384s at our
  ~1,000 pp) → chat() now defaults `-m 900` with per-call override; `sha_msg`/`content_of`
  made defensive (parse failure → raw-file sha / empty string, not a crash);
  `req_file` now gets `SERVED_MODEL="$SERVED_NAME"` explicitly; `restart` gained the
  production guard (refuses if glm53-flash-tf is up here or on the worker) and
  `restart_fb` gives every cell the PARALLEL=4→2 fallback (kvqual cells included);
  depth-fact placement fixed to ~60% of the filler (old arithmetic put it at ~22%
  while the comment claimed "near the end"); cliff background curls guarded with
  `|| true` so one failed request can't `wait`-kill the phase; `clock_mhz` runs
  locally on the head instead of ssh-to-self. `bash -n` clean.
- **Ecosystem research pass → docs/ECOSYSTEM.md.** Read the repos ajvazan and others
  referenced (kindlingai gx10 + exlr8, technigmaai tp2-native, eugr/spark-vllm-docker,
  Entrpi's EXL3 stack, goshi's exporter) against our line. Headlines:
  - **goshi's `tensorfold-spark-dashboard-metrics-exporter` targets our engine line
    directly** (`zig-v1` profile: `tensorfold_*` families + `/health` JSON) and
    production's 0.6.6 too (`python-v1`); zero-dependency, systemd unit, honest
    omissions. Adopt as a doc pointer; note its examples assume upstream on :8002,
    ours serves :8000.
  - kindlingai gx10: adaptive-k DFlash2 (draft length picked per step from a live
    acceptance/cost model — matches MaTeZZ's acceptance data), prefill-fairness cap
    (a 120k prefill blocked a 12-token request 78–90s until capped), dual-root
    all-reduce 110→190 Gb/s (we already name both HCAs; rank logs confirm 2), spin-wait
    patch (~20°C cost on GB10's shared power budget), and the rule we adopt as an
    invariant: **never pin NCCL_IB_GID_INDEX** (RoCE GID tables reorder across reboots).
  - EXLR8 (mmastrac): EXL3-trellis experts at k2/k3/k4 decoded by a custom MoE kernel —
    decode +3–8% vs NVFP4, KV +15%, prefill −7.7% open. The one EXL3-family development
    in the field; needs an upstream TensorFold loader → watch item #1, not adoptable here.
  - Entrpi (EXL3 on vLLM) is the other easy button; its failure mode per the thread is
    .env drift across releases. Our guarded config.sh + git-ignored site identity +
    receipt discipline is the structural answer — hold the line, import no frameworks.
  - Verdict recorded in ECOSYSTEM.md: Tier 1 adopt-now (exporter, GID invariant, clock
    probe — already in the campaign), Tier 2 measure-and-hand-upstream (adaptive-k,
    prefill fairness, spin, structured-output truncation = the cliff phase), Tier 3
    watch (EXLR8, dual-root — done, b12x/megamoe/mentat/instanttensor — engine-specific).
- Campaign + ecosystem doc ready to run in the user's next GPU window: stop production,
  `bash scripts/run-qualification.sh` (it captures the production reference probe first,
  then asks to proceed).

## 2026-10-23 (later) — first campaign digested: 512K qualified, gates reinterpret, cliff absent

- **The run** (`results/campaign-20261023-011329/`, 01:13–02:45, unattended): all 8 phases
  completed. Both bench cells — parity (dense8+fp8 KV+524288×4) and candidate dense4fp8 —
  scored **92/100 tool-eval (125/136, ★★★★★, 0 errors, identical points)**; dense4fp8 edges
  deployability 68 vs 67 and responsiveness 11 vs 10. Depth 12/12 at 8k/64k/192k/384k. Rung
  524288×1 fits (3.61 GiB/stream fp8, 13.9 GiB left). **The 512K ladder goal is met at 4×
  concurrency with 15.8 GiB spare** — CONTEXT_512K_PLAN.md carries the banner now.
- **The gate "FAILs" were the instrument, not the cache.** Every gate output was semantically
  identical (primes+sum across drafted/plain, solo/c1–c4). g1 exposed the root cause: the
  native engine is **byte-level nondeterministic at greedy** (same prompt, same server, raw
  shas differ, content matches) — unlike byte-stable production Python 0.6.6. So §2's
  sha-of-message gates can never bind on this engine; VALIDATION.md §2 now carries the
  receipt and the gates compare normalized content (`content_sha`), raw-sha as bonus line.
  g3 was doubly broken: it compared two DIFFERENT prompts (turn 2 alone vs turn 1+2 —
  unrunnable by design) and 512 max_tokens was eaten by 2,242 chars of thinking on the
  resumed arm (empty content). g3 now sends the same 3-message conversation twice — cold
  compute vs full cache hit — with per-arm `cached_tokens` receipts and max_tokens 1024.
- **Cliff verdict: absent.** Schema batch 17.9s vs plain 21.2s for 4×256 — MaTeZZ's
  structured-output draft-truncation mechanism does NOT reproduce on native 1.1.x. (The
  general solo→c4 aggregate flatness remains open — TF_GLM_PROF next.)
- **Campaign script fixes** (committed same day, ready for `PHASES="kvqual prefill"` rerun):
  content-form gates, g3 redesign, prefill probe timed via `curl -w %{time_total}` with
  `max_tokens=1` (wall ≈ prefill; the first run's probe emitted 26,869 tokens — filler
  denser than the 4 chars/token estimate — and no timing at all).
- **Two housekeeping findings:** (1) start.sh re-prepares when the config fingerprint changes
  — the new KV_CACHE knob guaranteed a rebuild — and prepare.sh builds from upstream HEAD,
  so the campaign ran 1.1.2 at **1351ed16** while the Oct-22 matrix receipts are from
  **4ff78441** (same engine version; cross-receipt comparisons carry this footnote). (2) Both
  fp8 cells served 524288 only because the launching shell exported CONTEXT — dense4fp8's
  matrix cell now pins CONTEXT=524288 explicitly.
- **Open after this:** throughput tables for parity/dense4fp8 at 512K (the campaign's bench
  call ran full tool-eval instead of `--perf-only` — one matrix window fills it), the
  content-form gate rerun (~20 min), dense4fp8's own kvqual, PARALLEL=6/8 climb at 512K,
  TF_GLM_PROF at c4@d4096, then phase 2/3 quality on the winner.

## 2026-10-23 (round 2) — gates all-PASS, dense4fp8 qualified, climb refused at defaults

- **`results/campaign-20261023-093827/`**: the content-form gates ran on BOTH configs and
  **everything passed** — g1 content-deterministic (raw bytes differ, as documented), g2
  drafted=plain, g3 cache-hit=computed (both arms: "alpha-7391, bravo-2250, charlie-8118"),
  g4 concurrent=solo, g5 seeded-sampling content-stable. **The 0.6.x exactness replication
  is now on paper for the native line** (VALIDATION §2 receipt). Caveat: g3's
  cached_tokens=0 on both arms — the engine doesn't populate the usage field, so the
  cache-hit path itself is unverified; a /metrics counter check closes it.
- **Throughput at 524288×PARALLEL=4 fp8 KV** (pp2048 tg128): dense4fp8 **48.3/49.0/48.0
  solo** vs parity 42.9/43.9/44.0 (+13%); c2/c4 lead +2–3% everywhere; pp2048 1,025 vs
  1,018. Solo 48.3 vs the Oct-22 dense4 55.4 (40960/bf16) prices the 512K window+fp8 at
  ~13% of solo. **Depth cliff reconfirmed: 48.3 solo → 13.3 agg at c4@d8192
  (3.3/stream), TTFT 2.0→28.4s** — the open performance question for TF_GLM_PROF.
- **Prefill: 1,063 t/s on a 26,869-token probe** (wall 25.3s, max_tokens=1) — no
  deep-prefill collapse; the upstream fairness note has its native-line data point.
- **Climb: PARALLEL=6 and 8 both refused at 512K with default knobs** — recorded
  outcomes; the kept-states store's worst-case sizing is the suspected cause. The climb
  phase now retries each N with `PROMPT_CACHE_GIB=4` before recording the refusal.
- **Script hardening while digesting:** `restart` now propagates start.sh's exit code
  (set -e is suppressed in if-contexts — refusals were invisible before, which the
  climb phase depends on); run_gates guards every restart; bench_perf prints the saved
  report's results table (the live tail is JSON events, which is why round 2's first
  throughput receipt had no numbers — the tables above came from the report files).
- **Verdict: dense4fp8 is the qualified 512K serving config** — gates PASS, tool-eval
  92 = parity, throughput leads every cell, 2 GiB cheaper than parity. Next: climb with
  the trimmed cache, TF_GLM_PROF cliff diagnosis, then phase 2/3 quality on the winner.

## 2026-10-23 (round 3) — prodqual: production head-to-head + the determinism correction

- **prodqual ran read-only against live production** (`campaign-20261023-110446/`):
  pg1/pg3/pg4/pg5 all PASS; **production tool-eval 95 (131 pts, ★★★★★, deployability 73,
  responsiveness 21)** vs our dense4fp8's 92 (125 pts, 68/11) on the same harness,
  version and seed. Perf: production pp2048 1,530 (ours 1,025), solo 51.6/56.0/53.1
  (ours 48.3/49.0/48.0), and the decisive one — **production's aggregate decode RISES
  with concurrency (51.6 → 67.3 at c2 → 60.5 at c4) while ours falls (48.3 → 41.3 →
  38.7)**; c4@d8192: 19.0 vs 13.3. TTFT 1,339 vs 1,999 ms. The native line's real gap
  is concurrency scheduling + prefill, now measured like-for-like.
- **The determinism narrative corrected.** pg1 on production: content-identical, raw
  bytes differ in exactly `id`/`created`/metadata — and `choices` byte-identical.
  Re-checked our g1 pair: same result. **Both engines are token-deterministic at
  greedy**; the first campaign's "byte-level engine nondeterminism" was the instrument
  (whole-envelope sha can never match across requests), not the engine. VALIDATION §2
  and the TOOL_EVAL tables corrected. The content-form gates remain the right
  instrument (envelope-proof); with the envelope excluded, the §2 equalities hold to
  the byte on both engines.
- **Campaign script survived a real failure and got hardened**: the prodqual perf bench
  exited non-zero (after saving a complete report) and `set -e` killed the campaign
  before the handover prompt — production was never touched. bench_cell/bench_perf now
  record a non-zero exit as a receipt NOTE and always extract the saved report's table.
- **Open**: climb with the trimmed prompt cache (production now stopped first), g3's
  /metrics cache-counter check, the 138-vs-136 max-points bench quirk, phase 2/3
  quality on dense4fp8, TF_GLM_PROF concurrency diagnosis (now with production's
  scaling curve as the reference).

## 2026-10-23 (round 4) — the climb ceiling is the engine's own arithmetic

- **PARALLEL=6/8 refused even with PROMPT_CACHE_GIB=4** (`campaign-20261023-121811/`).
  The refusal line carries the full math: **8 streams need 25.29 GiB at 524288, budget
  leaves 15.83 GiB — "serve --parallel 5, or lower --context"**. So the ceiling at the
  goal window is **PARALLEL=5** (15.81 of 15.83 GiB), and the blocker above it is the
  engine's **12.2 GiB reserve** — no flag exposes it (`serve --help` is usage-only).
  PROMPT_CACHE_GIB=4 freed +0.06 GiB (it ≈ the default) — not the lever. The climb
  phase now runs 5/6/8 so a rerun completes the ladder with the engine's suggestion.
- **Quality-gap location found in the prodqual receipt**: production passed
  TC-65/66/67/69 (nested schema / enum / multi-tool structured outputs) where both our
  cells took partials; both engines failed TC-68. The 3-point gap lives entirely in
  the structured-output category, identically across dense8 and dense4fp8 → an
  engine-level decoding-path difference (native vs Python), not a quant effect.
- Situation after four rounds: **dense4fp8 = the qualified 512K config** (gates PASS,
  TEB 92, throughput leader, 5-stream ceiling known); exactness replicated with the
  determinism narrative corrected; head-to-head vs production measured (95 vs 92,
  production's aggregate decode rises with streams while ours falls — the one big
  open engineering item, with TF_GLM_PROF the next tool).

## 2026-10-23 (round 5) — ladder closed at 5; upstream PRs found for both gaps

- **Ladder complete**: PARALLEL=5 ADMITTED at 524288 (climb receipt
  `campaign-20261023-130428/`), and the engine flexed the prompt cache itself (1.4 GiB)
  to make room. climb5's c4 row ≈ the PARALLEL=4 cell (38.7 agg @d0, 13.3 @d8192) — the
  bench drives 4 clients, so admission is the achievement. 6/8 refused, per the
  engine's own arithmetic. **The 512K goal is closed at its true maximum.**
- **Upstream scan (ashhart/TensorFold)**: 16 commits since our build pin (1351ed16) —
  Flash Next vision work, tool envelopes, a Windows build fix; nothing concurrency.
  But three OPEN PRs target our gaps directly (receipts in ECOSYSTEM.md):
  - **#1163+#1211 glm/cuda KV paging** (`TF_KV_PAGES`; heads on Bizuayeu's fork,
    r9 @ 87e4417ebec6): streams become 512-row pages from the core pool, kept states
    sized by position — the mechanism that could unlock 6+ streams at 524288 and
    change the concurrency curve. `auto` pages only from 3 ranks up; we opt in with
    `TF_KV_PAGES=on` at TP=2.
  - **#1080 prefill 5.4×** (qwen CUDA; 4096-row chunks + tensor-core prompt attention
    + the delta-rule chain) — our prefill gap to production sits in this blast radius;
    glm/cuda port is the watch item.
  - **#1215 round-class GPU timers** (`TF_Q27_PROFILE`) — the instrumentation pattern
    for the concurrency diagnosis (route/experts/attention/**drafter** per round).
- **Plumbing added**: `config.sh` env passthrough now forwards `TF_KV_*` to both
  ranks (TF_KV_PAGES ready); `prepare.sh` already takes TF_REPO/TF_COMMIT overrides,
  so a PR-branch build is `TF_REPO=…Bizuayeu/TensorFold.git TF_COMMIT=87e4417ebec6
  ./scripts/prepare.sh` — no script changes needed.
- **Next**: TF_GLM_PROF baseline diagnosis on the current build, then build + qualify
  the KV-pages head (admission, §2 gates on paged KV, throughput curve).

## 2026-10-23 (round 5b) — KV-pages image staged, prof phase ready

- **Built and distributed `glm53-native-tf:kvpages-r9`** (the #1163+#1211 head,
  Bizuayeu/TensorFold @ 87e4417ebec6): side-tagged so the current build stays
  servable for the baseline. Smoke: tensorfold-native 1.1.2 on both ranks. The build
  (~7 min compile) also warms prepare's layer cache — the branch's `prepare.sh` run
  with the same TF_REPO/TF_COMMIT will be a cache hit.
- **Source reads**: TF_GLM_PROF exists on our pin (glm/cuda_prof.zig — per-forward
  CUDA-event sections, reports every 64 calls per draft-window profiler; sections
  gate/embed/kda/mla/gather_a/dense/router/routed/shared/gather_b/head/draw/mtp/
  commit). TF_KV_PAGES exists on our pin only for Nemotron — GLM paging is exactly
  what #1163 adds; `auto` pages only from 3 ranks up, so our TP=2 qualification sets
  `TF_KV_PAGES=on`.
- **prof phase added** (`a28d055`): solo/c4 × d0/d4096 probes at max_tokens 2048
  (≈78 calls per draft-window profiler → every window reports), walls + completion
  tokens + aggregate rates, /health drafted/accepted/rounds deltas, and both ranks'
  `glm prof` section lines into the receipt.
- Waiting on: production stopped → (1) `PHASES="prof"` on the current build, (2)
  prepare the branch, (3) `PHASES="kvqual4 throughput climb" TF_KV_PAGES=on` against it.

## 2026-10-23 (round 6) — the prof receipt: the cliff was a bench artifact

- **`campaign-20261023-145449/` (user-run, TF_GLM_PROF=1)**: with long generations the
  aggregate decode RISES solo→c4 — 53.7/57.6 t/s solo (d0/d4096-class), **99.2/102.8
  at c4** — matching production's steady-state (103.7, derived from its bench rows:
  total time minus TTFT). The bench's falling tg numbers were **prefill polluting
  128-token generations** (tg divides by total time; at c4@d8192 prefill is ~80% of
  the wall). The matrix's "structural collapse" interpretation is withdrawn.
- **Section economics (the actual diagnosis)**: wall≈gpu everywhere (zero bubbles);
  `routed` experts dominate (46% of a 2-row verify, marginal ~6.0 ms per draft row);
  the RoCE gather is 4–8% (innocent); attention 13–16 ms/call (gentle with depth);
  prompt passes are **90% router+routed** — the prefill gap to production is the MoE
  expert path, NOT attention (so #1080's kernels are the wrong tool for our gap; the
  GLM upstream conversation is expert batching in prompt passes).
- **Reframed consequences**: more streams = more aggregate for agent workloads (the
  KV-pages unlock buys real throughput); long-generation probes are the honest
  steady-state bench (the prof phase doubles as one); our real remaining gaps are
  prefill (MoE-bound), short-generation latency (TTFT-driven), and the 3-point
  structured-output quality gap.
- KV-pages image (`kvpages-r9`) already built and distributed to both ranks — the
  branch qualification (kvqual4 gates on paged KV, throughput, climb 5/6/8 with
  TF_KV_PAGES=on) is ready to run in the next production-down window.

## 2026-10-23 (round 7) — branch qualification voided: start.sh's auto-prepare dropped the pin

- **The KV-pages campaign (`campaign-20261023-151516/`) ran on the WRONG image.** The
  user's `TF_REPO`/`TF_COMMIT` prepare built the branch and wrote the marker at
  87e4417e; the campaign's first restart then differed from the marker's config
  fingerprint (phase knobs), so start.sh auto-ran prepare — **without the session
  env** — resolved the default ref (1351ed16), cache-hit the rebuild and overwrote
  the branch tag + marker. TF_KV_PAGES=on reached a container whose engine ignores it
  for GLM (Nemotron-only on our pin). The kvqual4 PASS receipts are round-2 repeats;
  the climb refusals are pre-paging arithmetic. Results voided as a branch test.
- **Footgun recorded**: a version pin via session env does not survive start.sh's
  auto-prepare. Pins must live in scripts/local.sh (sourced by config.sh before the
  TF_* defaults, git-ignored) — local.sh.example gains the commented line. The
  side-tagged kvpages-r9 image remains on both ranks for a future window.
- **Decision recorded (user)**: PARALLEL=6/8 not worth pursuing — with steady-state
  aggregate already at ~99–110 t/s (c4) and matching production, the marginal gain
  from two more streams does not buy anything we need. The KV-pages branch test is
  deferred until upstream merges #1163/#1211 and a rebase picks it up; the paged-KV
  §2 gate runs then (one command with the pin in local.sh).

## 2026-10-23 (round 8 prep) — the pin verified; branch marker locked

- **TF_REPO/TF_COMMIT pinned in scripts/local.sh** (site file, git-ignored) at
  Bizuayeu/TensorFold @ 87e4417e (the #1163+#1211 KV-pages head). `./scripts/prepare.sh`
  run with the pin: resolved `tf=commit@87e4417e`, image `glm53-native-tf:commit`
  (a commit pin tags as `:commit` — no more tag collision with the main-line build),
  smoke 1.1.2, distributed to the worker. The round-7 failure mode (auto-prepare
  re-resolving the default ref) is closed: any future auto-prepare inherits the pin
  from local.sh and cache-hits the same build.
- **The window's command** (production stopped, ~40 min):
  `PHASES="kvqual4 throughput climb" TF_KV_PAGES=on ./scripts/run-qualification.sh`
  — gates on paged KV, the throughput tables, and the 5/6/8 climb (with paging the
  admission arithmetic may finally differ; 6 admitting would be the branch's headline).
  After: restart production, and the pin comes OUT of local.sh when the branch test
  concludes (or when #1163/#1211 merge and a rebase carries them).

## 2026-10-23 (round 8A) — the pin worked, then the branch refused: fp8 KV × paging are exclusive

- The window (`campaign-20261023-165319`) ran the RIGHT image (` glm53-native-tf:commit`,
  marker `tf=commit@87e4417e`, `TF_KV_PAGES=on` verified in the container env) — the
  local.sh pin closed round 7's failure mode exactly as designed (every auto-prepare
  re-resolved the branch from the pin).
- But every phase's server exited at startup: on the KV-pages branch, `TF_KV_PAGES`
  and `--kv-cache fp8` are **mutually exclusive** — the engine prints
  `tensorfold: --kv-cache fp8 keeps a window per stream: unset TF_KV_PAGES, or set it
  to auto or off` and exits. Since every existing cell gates/benches fp8 KV, kvqual4
  returned "SKIP: server refused to start", throughput both cells refused, and climb
  5/6/8 refused on the fatal config (not memory arithmetic). The "quick" run the user
  observed was four phases of refused launches — server never lived to print the
  sm_121 admission line ("no server line" in all receipts).
- Real data point for upstream: GLM paging at 87e4417e pages the bf16 KV path only.
- **The agent now runs GPU windows itself** — the user switched this agent to a cloud
  model, decoupling it from production uptime. run-qualification.sh gains three
  paged-bf16 phases (`kvqual4P` with a no-paging admission control, `throughputP`,
  `climbP`) that set `TF_KV_PAGES=on` explicitly and leave KV_CACHE unset. The
  agent launches: `PHASES="kvqual4P throughputP climbP" ./scripts/run-qualification.sh`.

## 2026-10-23 (round 8C) — site identifiers scrubbed from the repo (user)

- **(user) The repo is generic**: no hostnames, ssh targets, IPs, netdev or HCA
  names in any committed file. Committed prose says "the head" / "the worker";
  real values live only in git-ignored `scripts/local.sh` / `docs/SITE.md`.
- History entries in this log were redacted mechanically (the HCA/netdev
  mentions and one ssh path); receipt filenames, public GitHub handles and the
  repo's own URLs are untouched. The architecture SVG's host labels are now
  generic ("HEAD HOST" / "WORKER HOST").
- `results/` added to .gitignore — receipts live only on the head's checkout and
  must never be committable by a careless `git add -A` there.

## 2026-10-23 (round 8D) — prepare.sh discovers the site (ease-of-use)

- **`scripts/discover-site.sh`**: when `scripts/local.sh` is missing, prepare.sh now
  detects the site instead of failing on the empty-WORKER guard — the cabled CX7
  HCAs/netdevs (carrier-up links only, sysfs-based, lexically sorted so both hosts
  pair the same rail), the rails' IPv4 addresses, and the worker (ssh as the same
  user to the head's rail subnet's .2, the point-to-point convention;
  WORKER_TARGET=... overrides the guess). It writes exactly five keys into
  local.sh; unassigned rails are reported with the exact `ip addr add` commands
  for both hosts (discovery does not assign addresses — that is a root decision).
- **`REDISCOVER=1 ./scripts/prepare.sh`** re-runs detection on an existing local.sh:
  timestamped backup, the five keys updated in place, every other entry kept.
- README's requirements and quick-start now lead with the discovery path;
  `local.sh.example` stays as the by-hand walkthrough.
- NOTE: prepare.sh/discover-site.sh reach the head's checkout only AFTER the
  running campaign completes (mid-run script swaps are the AGENTS.md rule).

## 2026-10-23 (round 9) — the KV-pages branch qualified: gates PASS, 8 streams, bf16 only

- `campaign-20261023-170125/` (17:01–18:20, agent-run window): kvqual4P + throughputP +
  climbP on the #1163+#1211 head (`glm53-native-tf:commit` @ 87e4417e, pinned via
  local.sh).
- **Gates: all §2 PASS on paged bf16 KV at 524288 × PARALLEL=4** (content form). The
  no-paging control refused with the engine's own arithmetic ("4 streams need 19.77 GiB,
  budget leaves 12.06: serve --parallel 2") — the comparison is valid.
- **Admission: 5, 6 AND 8 streams admit** at 512K under paging (window-mode refused 6+ on
  the 12.2 GiB reserve). The pool is fixed at ~one full window (1028–1032 pages, 6.4 GiB)
  shared on demand; each stream costs ~220 MiB fixed (180 without the drafter; DFlash2
  adds ~2.8 GiB of model and a bigger prompt cache). Depth × streams is the budget:
  "enough for one at a 524288-token window".
- **Bench (same instrument on both configs)**: paged-bf16 leads dense4fp8-window on every
  row — solo d0 51.4 vs 48.3, c2 43.2 vs 41.3, c4 42.0 vs 38.7, d8192 c4 14.1 vs 13.3,
  pp ~1,037 vs 1,025, TTFT d0 1,976 vs 1,999. The solo row reaches PRODUCTION parity
  (51.0–51.4 vs 51.6). Caveats: rows are the TTFT-polluted instrument; the bench loads at
  most c4 clients regardless of server streams (the 5/6/8 rungs measured ADMISSION, not
  >4-client load); the steady-state probe on paged KV is owed before calling the curve.
- **fp8 KV × paging stay exclusive on the branch** — paged-bf16 is a shallow-many
  concurrency option, not a serving-config replacement: dense4fp8 window-mode keeps 5
  FULL-window streams where the paged pool caps deep concurrency at ~one window.
- **Housekeeping after the window**: bench_perf/run_cell's report-capture globs fixed
  (an off-by-one path level had been silently dropping tables from receipts); the TF pin
  removed from the head's local.sh and prepare re-run — marker back to main (1351ed16),
  so a bare start serves the qualified build; testing stack stopped.
- **(user) Launch defaults now track the qualified optimum**: config.sh defaults are
  dense4fp8 @ 524288 × PARALLEL=4 (receipts cited in its comments); README table matches.

## 2026-10-23 (round 10) — upstream PR scan for immediate performance gains; #1079 staged

- **Production is back on the head** (the agent is local again — GPU windows are the
  user's again).
- **Upstream scan** (releases + 50 open PRs + main since our pin): newest engine release
  is still **v1.1.2**; main since 1351ed16 is app/infra/Flash-Next/grammar — no glm/cuda
  perf movement, our pin stays current.
- **#1079 staged as a test patch** (`patches/1079-qwen-dflash2-shared-decode.patch`):
  qwen-measured DFlash2 decode +9–15%, "replies unchanged" — and it touches the SHARED
  core (core/lanes windows/proposer/drafter/lone_costs, cuda/sample) that GLM's DFlash2
  windows ride on. `git apply --check` clean at our pin (verified in a scratch clone at
  1351ed16). The §2 gates in the test window decide "replies unchanged" on OUR model;
  the tables decide keep-or-drop against the dense4fp8 receipts.
- **#1171 (qmmf half tiles)**: bit-for-bit but MXFP8/BF16-only — FP4 excluded, and our
  dense4/head4 packing is FP4 qmmf; diff does not apply at our pin (one registry hunk);
  small expected gain. Parked with reasoning in patches/README.md.
- **#1078 (TF_DRAFT_NODES)**: qwen-tree-only, no path to GLM's DFlash2. **#1143** (Flash
  Next: NVFP4 experts on prompt tiles) is the mechanism our prefill gap wants — the
  glm/cuda port joins the upstream ask with #1080.
- **Mechanism note**: the PATCHES knob is part of the config fingerprint, so a window
  run with `PATCHES=...` exported re-prepares with the overlay automatically; the patch
  is committed (patches/ is tracked) so the head gets it by the normal sync.

## 2026-10-23 (round 11) — #1079 A/B is a wash (dropped); the perf bar and the knob sweep

- **(user) The testing line's bar, restated**: to succeed it must **surpass production
  0.6.6 on model loading time, generation tok/s, and prefill rate**. Quality round
  approved but performance leads.
- **#1079 A/B** (`campaign-20261023-185520/`, the PATCHES window): gates all PASS on the
  patched build — "replies unchanged" holds on our model — but the throughput is a
  wash (solo d0 +3.3%, solo d4096 −8.6%, everything else ±2%, pp/TTFT unchanged). The
  qwen gains live in qwen's tree-draft kernels; our chain drafting doesn't hit the
  shared-lanes changes. **Dropped**: knob removed from local.sh, patch file removed,
  receipt kept in patches/README.md and the ledger.
- **Loading-time finding**: production 0.6.6 logs "loaded in **93.7 s**"; our restart
  walls run **~265 s**. Behind 2.8× on the user's first metric, cause unmeasured —
  candidates: the load-time packing (dense4/head4 pack at load), the DFlash2 drafter
  load, or the two-rank orchestration. The new knobs phase times every cell and records
  the engine's own "loaded in" line vs the wall.
- **Source scan of the pin found five untested TF_GLM_ knobs** with direct decode/prefill
  semantics (DRAFT_CONFIDENCE, PROMPT_STEPS, OVERLAP_PIECES, KDA_DECODE_WIDE, SHARED) —
  the new `knobs` phase sweeps them with timed restarts + the bench matrix.
- The `knobs` phase: 5 knob cells + a load-nodraft cell, each = timed restart +
  engine-load-line + bench. ~65 min; production must be stopped (user's window).

## 2026-10-23 (round 12) — the knob sweep ingested: one sharp lead, the loading gap located

- `campaign-20261023-200302/` (the `knobs` phase, six cells): every cell = timed restart +
  engine load line + the bench matrix.
- **Loading decomposition (the user's first metric front): the gap is INSIDE the engine.**
  Engine "loaded in" 241.5–253.5 s across all cells vs walls 251–269 s — orchestration
  (docker + ssh + rank handshake) is only ~8–17 s, and the DFlash2 drafter costs just
  ~11 s of load (241.5 nodraft vs 252.7 full). Production 0.6.6 loads the same checkpoint
  in 93.7 s → the ~2.7× gap is the native loader's convert-and-pack path
  (dense4/head4 packing + fp8 KV prep at load). Patch candidate: a packed-weights cache
  beside the existing byte-verify digest cache.
- **`TF_GLM_PROMPT_STEPS=0` (prompt passes run whole): c2 @ d8192 aggregate 17.3 → 33.1 t/s
  (1.9×)** — the sharpest movement any knob has produced, and the d8192-c2 row is tight
  across rounds (16.9–17.7), so the jump is real-looking; c4 is unchanged (13.6 vs 13.3)
  and solo d0 dipped (44.7 vs 49.9). NEEDS A CONFIRM CELL before the ledger calls it.
- conf085: solo d0 +3.8% (51.8) — weak, and the duel's acceptance/ITL instruments should
  decide it, not the bench rows. conf055/overlap6/kdawide0: no effect (kdawide0 −7% solo;
  drop). load-nodraft confirms the drafter's decode value: 29.6 vs 49.9 t/s solo (+68%).
- Next window: a steps0 confirm cell (+ the duel phase with production up first), then
  the loading-patch investigation (load_plan/qpack prepack cache).

## 2026-10-23 (round 13) — the duel corrects the scoreboard; version policy goes perf-gated

- **(user) Version bumps are perf-gated**: an upstream release is followed ONLY if it
  carries measured glm/cuda performance. v1.1.3 (published 21:25 tonight) has none for
  our stack (the one perf item — greedy sampling without a full-vocab sort — is ~1%
  unverified; a draft-scheduling change measured −17%/+13% on Nemotron only), so **we
  stay at our qualified tree**: local.sh pins TF_COMMIT=1351ed16 (the default latest-v1
  would now resolve v1.1.3 implicitly). The builder's grammar fix (libtfgrammar, xgrammar
  v0.2.8) rides along at this pin for the quality round — a builder fix, not a version bump.
- **The duel's production arm (salvaged, campaign-20261023-215834) + ours arm
  (campaign-20261023-221204)
  rewrite the scoreboard** — the bench-derived anchors are RETIRED as polluted instruments:
  - solo decode measured: **55.2 vs 58.7 t/s (94%)** — the old 48.3-vs-51.6 rows were
    instrument understatement on both engines.
  - **c4 aggregate measured: 99.8 vs 154.5 (65%)** — production scales 2.6× vs our 1.8×;
    the "steady-state parity ~104" claim came from derived numbers and is withdrawn. The
    concurrency front reopens as the biggest real gap.
  - **Request overhead: PARITY** (200 vs 183 ms solo) — the bench's 1,999-vs-1,339 TTFT
    rows were concurrency queueing, not fixed overhead. The TTFT front as imagined
    closes; the queueing behavior folds into the concurrency front.
  - prefill slope sharply 2.5× (0.662 vs 0.265 ms/token) — the MoE prompt path, cleanly measured.
  - streaming: same per-token pace (17.0 ms both); ours arrives in 3.4-token DFlash2
    bursts vs production's per-token trickle — a rhythm/feel difference to record, not a gap.
- perf-anchors.tsv rewritten from the duel (7 rows); the chart regenerated + checked.

## 2026-10-23 (round 14) — steps0 retired; the bench's variance found; the continuing-look opens

- **steps0 RETIRED**: the confirm runs landed d8192-c2 at 17.4/17.6 (baseline 17.3) — the
  1.9× lead was a one-off (receipt: campaign-20261023-221204/08-confirm.md). No knob
  adopted. ALSO: the confirm's d0-c4 rows swung
  45.8 → 57.9 and d4096-c4 19.5 → 40.8 across runs (baseline 38.7/19.2) — **the bench's
  shallow-depth concurrent rows are unstable instruments** (swings up to ~2×); single-run
  A/Bs on them are void as evidence. The duel's probes and the d8192 rows are the stable
  instruments. (The #1079 wash verdict stands via the duel's instrument, not the bench's.)
- **(user) The continuing look opens**: docs/PERF_PLAN.md — the scoreboard, the three
  fronts in attack order (concurrency scaling 65%, prefill slope 40%, loading 35%), every
  lever with mechanism + verification plan, and the research log. First levers: (1) do our
  verify rounds batch across streams at c4? (prof at c4, then a #1155-mechanism glm port
  if per-stream); (2) the AZ checkpoint (top-4 routing, 12.5B active, community-tested) —
  ~+25% prefill/decode expected, our gates + tool-eval decide; (3) the packing cache for
  loading. Production is back up (the user moved the agent back to local serving) — GPU
  work is window-gated again; the research and documentation run continuously.

## 2026-10-23 (round 15) — the AZ lever approved and moving; the blog's Act VII spliced

- **(user) The AZ checkpoint attempt is approved** — "a new checkpoint will mean seriously
  more quality checks. Let's try this though." The quality battery is expanded
  accordingly (PERF_PLAN Front 2 Lever A): §2 gates + full tool-eval + structured-output
  scenarios + hardmode TEB + duel probes, for BOTH experiments below.
- The AZ download (~169 GiB) is running on the head under nohup (console
  /tmp/az-download.console) — no GPU needed; 1.2 TB free.
- **Loader verification at our pin** (source cloned to /tmp/tf-src on the head): the
  native loader is EXL3-only — no GPTQ/autoround path in zig/src — so AZ as distributed
  cannot serve natively. But the glm config parser reads num_experts_per_tok dynamically
  (families/glm/config.zig:163) with the routing kernels taking topk as a runtime config
  field, so: **Experiment 1** = top4 routing on our own checkpoint (symlinked variant
  dir + a config edit — zero download, window-cheap, measures the compute saving and the
  un-retrained quality floor); **Experiment 2** = the full AZ after an EXL3 requant
  (int4→bf16→EXL3, window-gated, hours, disk choreography noted in the plan).
- The blog's Act VII spliced (vague, per user): the retired steady-parity claim corrected
  to the head-to-head's honest shape, the stale bench-derived scoreboard figure dropped
  from the act, two figures live. Pre-edit backup exists on the blog host.

- Window prep (same sitting): the campaign gained two session-specific phases for this
  window — `prof4` (Front 1: the duel arm under TF_GLM_PROF=1 at c4 answers whether
  verify rounds batch across streams; then TF_GLM_SHARED=0 answers the sharing-vs-thrash
  question) and `top4` (Front 2 Lever A Experiment 1: a symlinked `top4cfg` variant
  snapshot with num_experts_per_tok 8→4, served via a MODEL_REVISION override — prepare's
  rsync distributes it to the worker and the shared blobs keep the byte cross-check
  green; an admission refusal is a receipt). `duel_arm` was hoisted to the shared helpers
  so all phases run the identical five instruments; a `c4_reprobe` helper refreshes the
  64-call prof window before extraction. Perf only in `top4` — the quality battery stays
  with the quality round (user's "seriously more quality checks").

## 2026-10-24 (round 16) — prof4 answered Front 1's two levers; top4 crashed on a missing worker rsync

Receipts: campaign-20261024-000708/09-prof4.md (cells a+b) and /10-top4.md (the failure).

- The 23:35/23:36 attempts died in the 23:42 reboot (the user's "restart" was a pair
  reboot: containers shutdown-killed, /tmp wiped, the -tip production stack never
  restarted). Nothing was lost; the AZ download's ~184 GB of chunks survived in the Xet
  CAS (the model dir's 640K is just the small files — a re-run resumes from the CAS).
- **Front 1 Lever A (batching): ANSWERED** — at c4 our shared verify windows run **8 rows
  in one 76.5 ms round** (4 streams × ~2 draft rows; solo is 2 rows/41 ms). Verify batches
  across streams; the #1155-mechanism glm port is the wrong patch for the c4 gap.
- **Front 1 Lever B (TF_GLM_SHARED=0): ANSWERED, retired** — windows alone at c4 =
  51.6 t/s (4 × 52 ms unshared rounds) vs sharing's 70–105 in the same sitting. Sharing
  IS the scaling mechanism; the knob stays default.
- Front 1 now narrows to round economics: acceptance at concurrency, the window cost
  tables (alone 52.0+6.8/row vs shared 76.0+4.1/row), draft/verify overlap.
- **Methodology caveat**: the window's absolute numbers carry the agent's own inference —
  the session rides the engine under test and was generating during cell a's probes
  (solo 14.2/34.1, Δ58% = agent traffic). The duel receipt (221204) stays the clean
  baseline; probe windows run with the agent silent from now on.
- **top4 crashed as a receipt**: the variant served the head fine but prepare refuses a
  locally-built snapshot missing on the worker (it only rsyncs what it downloads itself).
  Fixed: the phase pushes the variant to the worker explicitly. Re-running top4+summary;
  the agent stays silent during the probes.

## 2026-10-24 (round 17, evening) — the innards cross-examination opens; the prompt-router pathology found

- top4's second attempt (receipts: campaign-20261024-003708/10-top4.md + campaign.log)
  failed on the port guard — prepare SUCCEEDED end-to-end (the
  round-16 rsync fix works: both hosts verified the variant, the image cache answered,
  the marker written with model=@top4cfg) and start.sh refused :8000 because production
  was back up (00:43). The variant is fully prepared on both hosts; the next window
  serves it with zero prep. CAUTION: the head's prepared state is currently top4cfg — a
  bare testing-start before the next top4 run would serve the variant silently.
- (user) production is up evenings; the agent rides it (confirmed by production's
  /health: this session IS its requests_total=2 / prompt_tokens_total=180224). The
  cross-examination of the engine innards runs as source analysis while the GPUs serve
  production.
- **The prompt-router pathology (PERF_PLAN Front 2 Lever A2)**: the router section eats
  910–1125 ms of every 2048-row prompt chunk (55–69% of chunk GPU time) for a ~4.8 GFLOP
  GEMM production does in ~1 ms — and the same kernels cost 47 µs/layer at m=2. The
  superlinear blowup is the prefill front's real cause; patch candidates written up with
  the evidence chain. Expected win: prefill slope 0.662 → ~0.25 (production parity).
- Gotcha corrected: production's /health DOES carry drafter counters now (rounds 1221 /
  drafted 2569 / accepted 1610 = 62.7% acceptance, tokens = rounds + accepted exactly);
  OURS is the health that lacks them (backend/device/live/memory/model/status only) —
  our acceptance instruments stay the request-log lines and /metrics.

- Watcher postscript (same sitting): the 003708 campaign's summary died on the receipts
  glob — `0*.md` misses `10-top4.md` (starts with 1); with no 0-prefixed receipt in that
  run's dir the glob matched nothing, cat failed on the literal pattern, and set -e ended
  the campaign inside its own summary (no done line; CAMPAIGN.md partially written). The
  summary's globs now accept `[0-9]*.md` — all numbered receipts. The receipts themselves
  (10-top4.md's port-guard failure) were already complete on disk.

## 2026-10-24 (round 18) — the upstream sweep: nothing to bump to; two patches staged, three PRs on the watch

- Upstream check against our pin (1351ed16): v1.1.3 is still the newest release and
  carries no glm/cuda perf content — **the perf gate holds, no bump**. main is 136
  commits ahead: Flash Next/HIP/Windows/app heavy, nothing glm-perf.
- **Staged `patches/glm-slot-leak.patch`** (upstream 7d330e24d, #1218): a released
  stream left its queued draw's ring slot taken — every completed/cancelled request
  leaked a stream slot until restart. 1 line, bit-invisible, carried until v1.1.4.
- **Watch list refreshed (ECOSYSTEM.md)**: #1176 (qwen int8 prompt matmuls, cold prefill
  20–27% shorter — a glm port candidate behind the router patch), #1168 (qwen decode
  attention on tensor cores — our kda+mla are 40% of a verify round), #1228 (prefill
  preserved across cancellation), #1155 closed unmerged with its fix-ups landing pieces
  (the batched-drafting precedent), #1163+#1211 (glm KV pages, still open).
- The next window's experiment list stands: router-ks2 A/B, top4 variant, acceptance
  counters at c4 — plus the slot-leak patch riding the same build.

## 2026-10-24 (round 19) — the window prep: top4 + routerks2 A/B, staged for an unattended run

- New session phase `routerks2` (PERF_PLAN Front 2 Lever A2): cell a = stock serve
  (the duel arm + chunk-prof extraction + the drafter acceptance lines from the request
  log and /metrics — the Front 1 counters); cell b = the patched build
  (PATCHES=glm-router-ks2.patch,glm-slot-leak.patch — the slot-leak fix rides along)
  with the same probes plus the content-identity gate (the patched greedy reply must
  equal the stock cell's; the ks change is bit-visible by design — the gate decides the
  patch like #1079). Receipt: 11-routerks2.md.
- Window order: top4 FIRST (the prepared marker is still top4cfg — zero prep), then
  routerks2's stock cell (re-prepare to the default revision, fast), then the patched
  cell (a real rebuild), then summary. Estimated ~60 min. The agent stays OFF the
  engine during the probes (the round-16 pollution lesson): the session rides cloud
  while production is down.

## 2026-10-24 (round 20) — the window digested: the router patch dies by its gate; acceptance is not our gap; the top4 config bug found and fixed

Receipts: campaign-20261024-115242/10-top4.md + /11-routerks2.md. The window completed
(12:26 done.) with the agent silent — and the stock cell reproduced the duel exactly
(solo 55.3/55.1 Δ0.3%, c2 73.5, c4 101.8, slope 0.659, overhead 199 ms): the duel
baseline stands, the instruments are stable.

- **glm-router-ks2: RETIRED by the content gate.** The patched cell's greedy reply
  DIFFERS from stock (a token flipped — bit-visible as the source warned) AND the router
  section did not move (1096 → 1090 ms of the chunk): the pipeline-depth hypothesis is
  rejected as the pathology's cause. The patched cell's decode gains (solo 64.3, c2 89.8,
  c4 119.6) are an acceptance lottery riding the bit change — unadoptable. The gate did
  its job; the pathology stays open (next candidates in PERF_PLAN Front 2 A2).
- **Acceptance is NOT our concurrency gap**: our /metrics exports the drafter counters
  (tensorfold:mtp_drafted/accepted_total): **ours 77.2%** (15660/20292) vs production's
  **62.7%** (1610/2569, its /health). Front 1 narrows to the shared-round cost scaling
  at c4: our 8-row shared round = 76.5 ms (routed marginal ~5.5 ms/row at topk=8) vs
  production's ~34 ms implied round rhythm.
- **top4 served topk=8 three times** — the config edit never persisted: the snapshot's
  config.json is a SYMLINK into the shared blobs (cp -P preserved it; the edit aliased
  the base checkpoint or silently vanished; the blob's mtime proves it was never
  written). The base checkpoint is intact (mtime Oct 3; the stock cell's authentic
  topk=8 behavior confirms it). Fixed in the phase: the variant's config is now a REAL
  file, verified topk=4 before serving, verified on both hosts (12:48). The next
  window's top4 runs with zero prep — and this time the engine will actually route 4.
- glm-slot-leak.patch stays staged (bit-invisible, unaffected).
- The scoreboard and chart are UNCHANGED (no qualified fact moved: the patch died by
  its own gate, the top4 experiment has not actually run yet).

## 2026-10-24 (round 21) — the next window staged: top4 (real) + the router kernel A/B

- `glm-router-part.patch`: forces the unstaged router kernel for every launch —
  bit-identical math in theory (same ks slicing/summation), the content gate vs the
  115242 stock reply verifies. The cost read splits the pathology: shared by both
  kernels → replace with a real GEMM; tile-specific → fix the tile's K-depth/w-layout.
  New session phase `routerpart` (receipt 12-routerpart.md); glm-slot-leak rides the
  build. Window order: top4 FIRST (the variant config is now a real topk=4 file on both
  hosts — zero prep, and this time the engine will actually route 4), then routerpart,
  then summary. ~45-50 min.

## 2026-10-24 (round 22) — the window digested: the router pathology is in BOTH kernels; the top4 refusal was a config-parse failure, fixed byte-surgically

Receipts: campaign-20261024-125609/10-top4.md + /12-routerpart.md.

- **routerpart (the kernel A/B): DECISIVE.** The content gate says **IDENTICAL** (the
  forced fallback is bit-safe — same ks slicing/summation, as reasoned), and the cost
  read says the unstaged kernel costs the **same** as the tile (router 1140 vs 1096 ms
  of the chunk; solo 54.1, c4 100.6, slope 0.673 — stock-neutral). **Both kernels share
  the pathology: the hand-rolled mma approach itself is the bottleneck. The fix is a
  real GEMM replacement (the cuda_mm class) — with the bit-exactness constraint now
  explicit: any accumulation reorder is bit-visible, so the replacement must keep the
  K-slicing/summation structure or the gate kills it.**
- **top4 refused on a budget illusion, not topk**: "163.6 GiB of weights do not fit
  103.5" — the WHOLE checkpoint billed to one rank. Root cause chain: `weightShare`
  reads config.json, parses it with the engine's parser, and EVERY failure path
  silently returns the whole; my python json.dump round-trip produced bytes that parser
  rejects (the earlier symlinked config parsed fine — the three topk=8 serves). Fixed:
  the variant config is now a **byte-surgical copy** (diff vs base = exactly one line,
  verified both hosts), and the phase's build step does the same byte replacement.
- glm-slot-leak rode the routerpart build and served fine (informal validation).
- The scoreboard stays unchanged. The next window: top4 (now actually topk=4) + the
  GEMM-replacement scoping.

- Window prep (round 22 addendum): the top4 phase now also captures the drafter
  acceptance counters (/metrics + request-log lines) — halving the routed experts may
  shift acceptance, and the counter read comes free with the probes. The window is
  PHASES="top4 summary" (~20 min): the variant config is byte-surgical on both hosts,
  the phase verifies topk=4 before serving, and the budget arithmetic resolves to the
  rank share (~83.5 GiB) now that the config parses.

## 2026-10-24 (round 23) — the topk pin found; the loop goes autonomous (user handed the restarts over)

- Receipt: campaign-20261024-132628/10-top4.md — the byte-surgical variant STILL refused
  the same 163.6 arithmetic, in 16 seconds (no load at all: the admission check fires at
  startup). Root cause one level deeper than round 22: the parse VALIDATES against
  `want` — a snapshot of the struct's compiled-in defaults taken before the json
  (`const want = c;`) — and requires `c.topk == want.topk`; topk=4 is
  `error.NotGlm53Flash`, whose silent catch in `weightShare` bills the WHOLE checkpoint
  to the budget. **The top4 experiment was never a config tweak: it needs the reference
  shape rebuilt.** Staged `patches/glm-topk4.patch` (the struct default `topk: u32 = 8`
  → `4`; the kernels specialize for slots=5 at load).
- **(user) The loop is autonomous from here**: on cloud, the agent may restart the
  testing stack at will until the qualification gets through; production stays down and
  untouched.
- Launching: PHASES="top4 summary" PATCHES="glm-topk4.patch" — the build applies ONLY
  the topk4 patch; the expected sequence is build (~10 min) → load (~4.2 min) → probes
  (~9 min) → the receipt tells whether the routed section actually halves.

## 2026-10-24 (round 24) — top4 GOT THROUGH: real top-4 routing measured; the AZ thesis holds

Receipts: campaign-20261024-135458/10-top4.md. Attempt four (the ladder: admission
illusions ×2 → the parse's `want` pin → the EXL3 global; each stage moved one level
deeper; the fingerprint-hashes-names gotcha cost attempt three — a patch content change
needs a name change, the round-7 lesson's sibling).

- **The serve: real topk=4** (slots=5 scratch — the serve line's 83.35 GiB; the
  fingerprint's patch-rename forced the rebuild). Load 714 s in the receipt = build +
  load; the container's own line: loaded in 343.8 s (the first variant load packs new
  5-slot layouts — ~+87 s over the base's 257; a repeat-load number wanted).
- **The compute claim CONFIRMED (unretrained selector, our own checkpoint)**: routed
  **18.45 → 11.0 ms** at 2 rows (−40% — the weight-traffic union isn't slot-proportional,
  hence not −44%); verify rounds 40 → 32.3 ms; 8-row shared rounds 76.5 → 53.3 ms.
  **Solo 63.5/63.6 (Δ0.2%) — 108% of production's 58.7: the generation-tok/s front
  CROSSES THE BAR at solo.** c2 90.0 = 103% of production's 87.1. **c4 120.4 = 78%**
  (was 65% — the concurrency front jumped a fifth of the gap in one lever). Prefill
  slope 0.593 (−10%) — still 2.2× production's 0.265: the front now needs the GEMM
  router replacement (the ROUTER section, 1096 ms of the chunk, untouched by topk).
- **The acceptance shift is real**: 4.8 tokens/chunk (427 chunks for 2048) vs 3.37
  (608) at topk=8 — the fewer-expert target is easier to draft against. The /metrics
  counter read was cut by `head -2` (a phase bug — the value lines got truncated);
  the request-log lines didn't match the grep. Fix both next window; the exact
  acceptance number at topk=4 is still open.
- **The quality question stands, unresolved**: topk=4-unretrained is a different model
  behavior — the (user) bar says a new checkpoint/model behavior means seriously more
  quality checks (§2 gates + full tool-eval + structured-output + TEB). Until that
  battery runs, this is a measured lever, NOT a servable default; config.sh stays
  topk=8. The AZ full checkpoint (the retrained selector + requant path) is the
  quality-safe route to the same win.
- Scoreboard/chart: UNCHANGED (the served default is still stock topk=8).

## 2026-10-24 (round 25) — the quality battery on top-4, staged

- New session phase `top4qual`: the battery on the real topk=4 serve — §2 gates
  (run_gates with the top4cfg config) + the full tool-eval (run-bench-matrix
  PHASE=tools; the structured-output scenarios ride it) + hardmode (the 28 scenarios).
  Receipt: 13-top4qual.md. The reference bar: the stock dense4fp8 cell's 92/100
  (125/136 pts, 0 errors). A battery failure sends the lever to the AZ full checkpoint
  (the retrained selector) instead of a default change.
- The bench-matrix gained a `top4` config entry (PATCHES=glm-topk4b.patch +
  MODEL_REVISION=top4cfg). The prepared state is already the top-4 serve (from attempt
  four) — the battery's serves need no rebuild.
- Window: PHASES="top4qual summary" — gates ~12 min + tools ~20 min + hardmode ~20 min.

## 2026-10-24 (round 26) — the top-4 quality battery: gates ALL PASS, tool-eval 93 (above stock's 92), hardmode 89 vs production's 91 — the default fork opens (user)

Receipts: campaign-20261024-142659/13-top4qual.md.

- **§2 gates on real topk=4: ALL PASS** — g1 greedy determinism (content), g2
  drafted=plain (the drafter is exact on the new model), g3 cache-hit=computed, g4
  concurrent=solo (4/4), g5 seeded (NOTE form — the standard).
- **The full tool-eval: 93/100 — ABOVE the stock bar** (92; production's anchor is 95 —
  the top-4 serve closes 1 of the 3 points to production). The structured-output
  scenarios rode the suite.
- **Hardmode: 89/100** vs production's anchor **91** — the FIRST hardmode datum on this
  line (stock dense4fp8 never ran hardmode; the ledger's Phase 2/3 item stays open for
  that). Two points under production on the adversarial suite.
- **The fork (user decision)**: (a) adopt top-4-unretrained as the served default — the
  perf receipts (solo 108%, c4 78%) + the battery (gates pass, tool-eval above stock)
  back it, and the scoreboard/chart finally move; but the served model then differs
  from production's stock behavior, and hardmode runs 2 under production's anchor.
  (b) hold stock as the default and route the lever through the AZ full checkpoint
  (the retrained selector — quality-safe by construction, the download's ~184 GB of
  chunks survived the reboot, the requant is window+GPU work). The agent's
  recommendation: (b) — the comparison story stays honest until the retrained selector
  is measured. Scoreboard and config.sh UNCHANGED this round either way.

## 2026-10-24 (round 27) — (user) the default moves to top-4; the AZ full checkpoint route opens

- **(user) The fork resolved: adopt.** config.sh's qualified defaults now serve the
  top4cfg checkpoint with PATCHES=glm-topk4b.patch (receipts cited in the comments:
  perf 135458 + battery 142659). Stock remains one env away
  (MODEL_REVISION=078455ffe6472f9a52fbc1139f58b9db2881b25c, PATCHES=).
- **The scoreboard moved for real** (the anchors + the chart regenerated from the
  receipts): solo **108%** and c2 **103%** of production — the generation front CROSSED
  the bar; overhead 119%, streaming ~189% — two more rows over the line; c4 **78%**
  (from 65%); prefill **45%** (the router GEMM is the remaining fix); loading **27%**
  (344 s — the 5-slot pack is heavier than the base's 257; the packing cache is the fix).
- Next: the AZ full checkpoint route (the retrained selector — the quality-safe upgrade
  to the same lever). The download's ~184 GB of Xet chunks survived the reboot; resuming
  it now (no GPU). Before staging a requant, check whether the community already
  published an EXL3 variant of the AZ checkpoint.

## 2026-10-24 (round 28) — the AZ route: the download verified complete; the requant plan staged; AZ-Focus discovered

- The AZ download is COMPLETE (the Xet CAS survived the reboot — the resume finished in
  seconds): 170 G materialized, 37 shards, topk=4, 288 experts, auto-round int4. The
  Experiment-2 checkpoint (the retrained selector) is on disk.
- The route staged in PERF_PLAN: requant auto-round int4 → EXL3 4bpw (streaming convert
  first — no ~640 GB intermediate if exllamav3 dequants per-layer; else the disk
  choreography), serve with glm-topk4b (the reference is already topk=4), the battery +
  the perf probes, and — if it passes — the default upgrades to the retrained selector.
  GPU: hours, window-gated.
- **Discovered en route**: AxonZeta/GLM-5.3-Flash-AZ-Focus-EXL3 — a DIFFERENT mod
  (AxonZeta's own, from Z.AI's BF16): 40–50% fewer thinking tokens at max effort with
  accuracy level or up (AIME 81.7 vs 75.0, GPQA up, SWE-bench 92 vs 95), topk=8 intact,
  EXL3 4bpw native — serve-ready as-is. A candidate lever for the agentic workload
  (fewer thinking tokens = effective throughput), recorded in PERF_PLAN; not the AZ route.

## 2026-10-24 (round 29) — the AZ requant window staged: the dequant, the convert, the battery

- scripts/az_dequant.py: streams the AZ auto-round int4 (the GPTQ layout decoded from
  the shard headers: qweight [in/8,out] I32, qzeros [groups,out/8], scales [groups,out]
  F16, group 128; everything else BF16 pass-through) to a plain fp16 HF dir. The GPTQ
  zero convention auto-detects on the first expert tensor (the wrong one leaves every
  weight offset by a scale step — mean|w| decides). RAM stays per-shard; the output
  carries a real safetensors index + the config minus quantization_config.
- scripts/az-window.sh: the orchestrator (dequant → exllamav3 convert -b 4 -hb 16 -eb 16
  → place as local/GLM-5.3-Flash-AZ-EXL3 → the worker rsync → the azqual campaign).
  exllamav3 1.6.4 installed at ~/.venv-exl3 (the host nvcc 13.0 + ninja; the JIT builds
  on first import).
- The azqual phase: §2 gates + the duel arm + the tool-eval + hardmode on the AZ-EXL3
  serve (the bench-matrix gained the azexl3 config entry). References: top4cfg's
  135458/142659 receipts. A battery pass upgrades the default to the retrained selector.

- README currency pass (same sitting): the config table + the revisions table now carry
  the round-27 default (top4cfg + PATCHES=glm-topk4b; stock one env away; the AZ
  requant listed as staged). The vision question settled: the engine code + the
  checkpoint's tower (347 visual tensors) are present, but the line has never passed
  --vision — the README now says "vision here is unserved" instead of the ambiguous
  "does not carry it". A VISION knob + a vision gate round is a small future window item.

## 2026-10-24 (round 30, evening) — (user) the convert runs overnight; Focus verified test-ready

- **(user) The trade resolved: the convert runs overnight** (production down until the
  window completes; ~10 h remaining at the measured 58 tensors/min, layer 4 of 45; the
  window's done lands ~06:00 with the placement + the battery). The alternatives
  (a pausable resume now, or pivoting to the test-ready Focus) are recorded in the plan.
- **The AZ-Focus-EXL3 download: COMPLETE + verified** on the worker (164 GiB resolving,
  48 shards) — the reasoning-efficiency lever (topk=8, EXL3-native, no requant) is
  test-ready for its own battery whenever the user calls it.
- Overnight watcher armed: fires on the campaign's done. OR pipeline death (the
  checkpointed stages resume on any re-run of block 2; the staging survives).

- 21:18 crash (the overnight run): the converter's calibration forward hit
  "weight must be [dim, conv_kernel_size]" — the AZ checkpoint's conv1d weights are
  HF-style [dim, 1, K]; exllamav3's decode-path triton kernel wants [dim, K]. Fixed:
  az_squeeze_conv.py rewrites the conv1d tensors in the fp16 dir (they live mostly in
  model_extra_conv.safetensors — a minutes-fast surgical fix), az_dequant squeezes on
  pass-through (future runs), and the orchestrator runs the squeeze as stage 1b before
  resuming the convert (the 8+ layers of checkpointed work survive).

- Overnight post-mortem (01:48 death, found 08:57): the healer SELF-MATCHED (its own
  command line contains "scripts/az-window.sh" — the liveness pgrep saw itself as the
  pipeline alive all night) AND the conv1d fix was wrong twice: the squeeze flattened
  the CANONICAL bf16 splits ([c,1,K] — what TensorFold's convs() reads, weights.zig:372)
  and left the NON-canonical F32 mixed convs ([3c,1,K]) in place, so the converter's
  split-cat produced 40960 channels against 24576 expected. Fixed: az_conv_fix.py
  (drop the F32 mixed, unsqueeze the bf16 splits — idempotent, atomic), the dequant's
  pass-through drops the F32 convs (future runs), stage 1b points at the new script.
  GPU verified clean (no memory held; 6122 = the frozen progress counter, not GPU tensors).

## (round 31) The requant saga: three refusals, one format gap (2026-10-26)

- **azqual refusal 1 — the codebook** (`campaign-20261026-010238/14-azqual.md`): the
  1.6.4 convert defaulted to `-cb mul1` + float `bits: 4.0`; the engine's `admit()`
  (families/glm/cuda_exl3.zig) reads integer bits 4 + the "mcg" codebook only. Both
  ranks refused (UnsupportedQuant) in 0.1 s. The receipt is the refusal's own
  arithmetic — recorded, not retried blind.
- **The mcg rework** (`406334f`): `-cb mcg` baked into `scripts/az-window.sh`, fresh
  WORK/EXLOUT (the mul1 qtensors invalidated), a byte-surgical bits normalization +
  hard assert (`scripts/az_bits_fix.py`) before placement, and the retry-wrapped
  worker sync (6 attempts, -W, keepalives). The 12-h mcg convert ran clean; the
  assert PASSED; the sync placed 153.4 GiB on the worker (the delta-mode trap cost
  it 5× — see the sync lesson below).
- **azqual refusal 2 — the checkpoint format** (`campaign-20261026-210257`): the
  engine admitted the mcg checkpoint and then refused at tensor load
  (MissingTensor: `layers.0.self_attn.q_proj.weight`). Root cause: exllamav3 1.6.4
  writes the NEW checkpoint layout (fused `qkv_proj.mcg/.trellis/.suh/.svh` quads,
  merged `conv1d.weight`); the engine (pin 1351ed16 — and upstream main, verified
  byte-same) reads the OLD layout (`q_proj.weight` + conv splits + per-module bits
  in quantization_config). **Nobody's engine reads 1.6.4's layout yet.**
- **The AZ-Focus checkpoint (AxonZeta) is OLD-format** — `focusqual` (the battery
  phase, `37b07ec`) is unblocked; the thinking-efficiency track's first measured
  cell. (user) Focus battery first; the AZ-EXL3 format path (1.5.3 side-venv
  re-convert / repack shim / upstream watch) is decided after the Focus verdict.
- **Disk choreography done**: the mul1 dead weight freed (~499 G — az-az-work 191 G,
  az-az-exl3 154 G, the local mul1 placement 154 G); 958 G free on the head during
  the convert. Pending after the AZ verdicts: az-az-fp16 (599 G) + az-az-work-mcg
  (165 G). (user) the AZ quant publishes to huggingface.co IF its gates pass
  (ledger open item).
- **The sync lessons** (skill-smith log, unpromoted): (1) the 4 silent rsync deaths
  → the retry-wrapped sync pattern; (2) rsync's delta mode against a stale
  same-named destination (sender pegged 97%, receiver asleep 4.7%, 60 vs 320 MB/s;
  the rail does >1 GB/s on 4 streams) → wipe the destination + -W, deployed in
  stage 3 (`92723f4`).

## (round 32) The Focus battery: the first complete AZ-family verdict (2026-10-27)

- **The six-refusal root cause, solved by trace** (`b1f42bd`, the :diag/:diag2 receipts): the served
  default's `glm-topk4b` pin makes the engine's strict `config.parse` reject the Focus's field-less
  config (`NotGlm53Flash`); `weightShare`'s silent `catch return total` then bills the WHOLE checkpoint
  to one rank (163.6 vs ~102) — the identical refusal six times. The top4cfg survives the pin only
  because the ladder's surgery declared `num_experts_per_tok` in its config. The Focus (topk=8 by
  design) now serves PRISTINE: `PATCHES=` empty, config.sh `${PATCHES-…}` so an explicit empty survives.
  The diagnostic methodology: the trace built to its OWN tag (`:diag2` — the campaign's auto-prepare
  rebuilds `:commit` per its staging and had silently dropped an earlier diagnostic patch).
- **The battery** (`campaign-20261026-235703/15-focusqual.md`, `run-20261027-002357` + `005153`):
  §2 gates ALL PASS (g1–g5 content-form); **solo 57.9 t/s** (production ≈ 58.9 — at par; the top4cfg's
  108% stands); **c2 74.7** (the top4cfg's 90.0 — a step behind); **c4 119.8** (≈ the top4cfg's 120.4,
  2× production's 60.5); TTFT slope 0.663; **tool-eval 94** and **hardmode 90** — the highest-quality
  checkpoint this line has ever served, one point under production on BOTH instruments (the top4cfg:
  93/89; the stock: 92). The thinking-efficiency track's first measured cell: quality-first, at-par
  solo, concurrency a step behind the top4cfg.
- Production returned 01:29 after the battery (the agent ran the window end-to-end under the user's
  grant; the serve line confirmed `PATCHES=` riding in every restart).
- Open: the AZ-EXL3 requant still parks on the 1.6.4 format gap; the Focus's c2 gap (74.7 vs 90.0)
  joins the router/slope work as a perf question; the thinking-efficiency next steps (`TF_GLM_CLEAR_THINKING`
  #840 + the token-count instrumentation) now have their baseline cell.

## (round 33) The v1.1.6 rebase: adopted on a +26% concurrency landslide (2026-10-27)

- **The pin**: 1351ed16 → **eea9dc9c (Release 1.1.6)** — `local.sh` updated, the topk4b patch
  applies cleanly at the new tree, the campaign receipt carries `tf=commit@eea9dc9c…`.
- **The A/B** (`campaign-20261027-085035` vs `20261026-235703`, the identical Focus serve):
  gates PASS ×4 both; solo 57.9 = 57.9; **c2 74.7 → 94.3 (+26%)**; **c4 119.8 → 150.6 (+26%)**;
  slope 0.663 = 0.663; tool-eval 94 = 94; **hardmode 90 → 91 (ties production's anchor)**.
  The upstream #1322 measurement (+14% at two streams on NVFP4) transferred at +26% on our
  EXL3 checkpoint. ADOPTED — receipts-equal-or-better everywhere, dramatically better at
  concurrency.
- **The first A/B silently ran the old engine**: the auto-prepare keys on the config
  fingerprint (the patch names) — a TF_COMMIT pin change in local.sh does NOT move it, so
  `campaign-20261027-053451` re-ran the v1.1.2 engine and the numbers matched the baseline
  exactly. Caught by the image ID (9 h old). **A pin change needs an explicit prepare.sh.**
  (lessons log carries it.)
- **The scoreboard staleness**: the served default (the top4cfg) still carries v1.1.2-era
  anchors — its re-qual on v1.1.6 is the next window's battery before the scoreboard rows move.
- **The Focus's c2 gap is CLOSED by the rebase** (94.3 ≥ the top4cfg's 90.0) — the router/slope
  perf question narrows to the solo/c4 slope only.

## (round 33 addendum) Production's fresh duel receipt — the Focus crosses c2, the TTFT frontier named (2026-10-27)

- **Production re-measured** (idle-gated, the campaign's own `duel_arm` lifted verbatim against
  the -tip production with Mia's new patch series): **solo 58.6/58.5 (Δ 0.0%), c2 87.0, c4 154.6,
  ITL p50 13.7 ms, TTFT d0 186 ms** — and the TTFT **slope −0.005 ms/token** (d0 186 → d4096 161
  → d8192 142: production's TTFT IMPROVES with depth — its prefix cache hits the duel fills'
  shared prefixes across requests).
- **The head-to-head vs the Focus on v1.1.6** (the same probes): solo **98.8%**, **c2 108.4% —
  CROSSED** (94.3 vs 87.0), c4 **97.4%** (150.6 vs 154.6). The scoreboard's formal rows still wait
  for the top4cfg's v1.1.6 re-qual (both sides of that comparison have now moved).
- **The TTFT-at-depth frontier is the new #1 perf item**: ours grows +0.663 ms/token (d8192:
  5,645 ms) while production is flat (142 ms) — a 40× gap. Production's cross-request
  shared-prefix caching (the 0.6.x kept-state work + Mia's new patches) vs our engine's
  same-conversation-only cache (the g3 gate covers the same-conversation case; the
  cross-request shared-prefix case does not hit). Upstream's #1321 (the shared-prefix
  queue) and #1262 (the kept-state resumption) are in v1.1.6 — the frontier is the
  cross-request prefix mark, not the queue.
- The production benchy baseline also landed (idle, clean): pp2048 1,514-1,627 t/s, solo tg
  54.4 — consistent with the old anchors (Mia's patches did not move production's speed).

## (round 34) The TTFT-at-depth frontier: root cause found at source level (2026-10-27)

- **Method**: no GPU needed — the v1.1.6 source cloned at the pin (`eea9dc9c`) and read end to
  end: `zig/src/core/prompt_cache.zig` (the store), `zig/src/core/lane_stream.zig` (the
  per-request entry), `zig/src/server/chat.zig` (what a chat request passes),
  `zig/src/native/cuda_snaps.zig` + `families/glm/cuda_native.zig` (GLM's rules).
- **The cache is a token-prefix cache** — `find` matches a kept state's full token array as a
  strict prefix of the new prompt, with decode spans equal through the kept position. NOT
  same-conversation-only: any request whose tokens extend a kept state resumes it.
- **But states are kept only at conversation-structure marks** (`marksRewind`): the history
  boundary, the rewind point before the latest user message, the stable diff vs the
  conversation's previous prompt, and caller-provided `shared_prefixes` cuts — each snapped to
  the planned grid (`snap_grid = prompt_rows`), min_gap 256 apart.
- **And a chat request's shared cuts cover only the system prefix** (`server/chat.zig`):
  three positions at system_len−2048/−512/system_len, only when the system prefix exceeds 512
  tokens. Nothing else ever populates `shared_prefixes` from the HTTP payload.
- **Consequence (the frontier)**: a shared prefix inside **user content** — the duel's fill, any
  long document re-sent across conversations — never gets a kept state, so every request
  prefills it from zero: TTFT grows linearly (+0.663 ms/tok measured). Same-conversation
  follow-ups hit via the history + stable marks — exactly the g3 gate's same-conversation case.
  Production's python engine matches prefixes at arbitrary KV-page boundaries across requests,
  hence its flat/improving TTFT (186→161→142 ms).
- **The span machinery does not block the fix**: decode spans are token-derived per request
  (`replySpans` "from the tokens alone"), so identical token prefixes carry identical spans
  through the shared position — grid-boundary states inside long prompts would resume exactly.
- **Upstream's own paths** (v1.1.6, both present): #1321 dedups only requests arriving
  *together* on a new shared prefix; the `imprint`/`--learn` path writes shared cuts' states to
  disk "read back by later sessions and servers" — but the chat server's cuts are system-only.
  `Request.shared_prefixes` exists on the API struct but is not client-settable.
- **The proposed experiment (next window, needs a serve)**: a marks-policy patch keeping
  grid-boundary states inside long prompts for CUDA — measured by the duel probes (the TTFT
  slope is the direct instrument) with the full §2 gates battery around it (state keep/restore
  is exactness-sensitive; the resume marks are where determinism could break).
- Receipt: this entry + the clone at `/tmp/opencode/tf` (local workstation); no new numbers —
  the round 33 addendum's duel receipts are the measurements this explains.

## (round 35) Production's v3.0.1 score anchors — tool-eval 96, hardmode subset 90% (2026-10-27)

- **The morning's score attempt was killed** (the user's report: production was in use during
  the run — 4 concurrent requests at scenario ~53/69; timings AND scores compromised). Partials
  discarded. The idle-gated discipline the user set for the perf run was applied here too.
- **The clean re-run** (idle-gated: fired at 11:46 when /health showed 0 running ×3 checks,
  both suites back-to-back, ~40 min): receipt
  `~/tool-eval-bench/results/prod-score-rebaseline-20261027-114634/` on the head, bench
  `3.0.1.dev4+gb672b53ff`, model `GLM-5.3-Flash-EXL3`, seed 42, parallel 4, timeout 240.
- **tool-eval: 96/100** (133/138 points — 66 pass, 1 partial TC-40, 2 fail TC-43/TC-51; 69
  scenarios). **hardmode run: 94** (196/208 over all 104 scenarios = standard 69 + hardmode
  35). **hardmode subset: 63/70 = 90%** (30 pass, 3 partial, 2 fail TC-92/TC-104 — production
  scores 0 on TC-104, the newest scenario).
- **The cross-version boundary, receipted**: the Focus battery's hardmode run counted
  `total: 97` (69 + the old 28-scenario hardmode suite) in
  `results/campaign-20261026-235703/bench-hardmode.log` — today's run counts 104 (69 + 35).
  The old production anchors (95/91) are formally cross-version; kept as history only.
- **README correction (withdrawn + re-stated)**: the round 34 Quality row labeled the Focus's
  94/90 as "bench v3.0.1" — wrong, the battery ran the night BEFORE the bench update. The row
  now states the Focus's 94/90 as old-bench, pending our v3.0.1 re-bench.
- **State**: production's score side is re-baselined on v3.0.1 (96 / 94 / subset 90%). The
  cross-engine score comparison waits for OUR serves' re-bench (the Focus + the top4cfg on
  v3.0.1, next window). The perf/duel side is already comparable (round 33 addendum: c2
  108.4% CROSSED, c4 97.4%).

## (round 36) The window block ran the WRONG phases — and the accident qualified the default's v1.1.6 perf (2026-10-27)

- **The bug**: `run-window-block.sh` set `PHASES` without exporting it — the campaign never
  saw it and ran its full default phase list (12:41→17:22, ~4.7 h) instead of the two
  batteries. My session died with production as designed, so nothing could correct it
  mid-flight; the block itself wrapped cleanly (rc=0) and stopped our stack.
- **What the accident produced** (`results/campaign-20261027-124115/` — the FULL v1.1.6
  receipt set for the served top4cfg default, which the scoreboard needed anyway):
  §2 gates ALL PASS on every cell (g1–g4 content form); **the duel receipt: solo 64.8/64.9
  (Δ 0.2%), c2 108.0, c4 167.4 t/s, TTFT d0 148 ms, ITL mean 42.8 ms/chunk (427 chunks ≈
  4.8 tokens/chunk → 8.9 ms/token); prefill 1,175 tok/s** (26,869-token wall, was ~1,026);
  **loading 256 s** (the knobs cells' timed restarts 246–260 s, engine loaded ~248 s —
  the v1.1.2-era 344 s is gone); depth **12/12** at 8k/64k/192k/384k; fresh
  throughput/climb/knobs/confirm receipts. The prod arm of the duel skipped (production
  already down — its round-33 fresh receipt is the other side).
- **The scoreboard moved** (`docs/perf-anchors.tsv` + the regenerated chart): vs production's
  fresh duel receipt (58.6/87.0/154.6) — **solo 111%, c2 124%, c4 108% — every concurrency
  front CROSSED** (c4 was 78% at v1.1.2); TTFT overhead 126% (148 vs 186 ms); ITL 192%
  (8.9 vs 17.1 ms/token); prefill 78% (1,177 vs 1,514 benchy, TTFT-estimated both sides);
  loading 37% (256 vs 93.7 s). **TTFT at depth: 3%** (4,954 vs 142 ms at d8192) — the one
  open front, round 34's root cause.
- **Row evolution, explained**: the `prefill slope (ms/token)` row became `TTFT at depth
  (1-token, d8192)` — production's fresh slope is **−0.005 ms/tok** (flat-to-improving: its
  cross-request prefix cache), and a negative production value degenerates the chart's
  inverted-ratio math. The absolute d8192 pair tells the same story honestly. Not a silent
  edit: this entry is the explanation.
- **The wrapper fix, verified three ways** (verify-shell-config-edits): `bash -n`; a
  child-visibility probe (`bash -c 'echo $PHASES'` through the exact assignment+export);
  and the live guard path — with production UP, `STEPS=1` runs the wait once and aborts
  cleanly ("nothing was started", exit 1). The head's copy carries the fix.
- **Still owed (one more window, ~2.5–3 h)**: `top4qual` + `focusqual` — the tool-eval +
  hardmode scores on bench v3.0.1 for both serves (the score side of the comparison; the
  perf side is now fully receipted). The steps0 knob lead stayed inconclusive on v1.1.6
  (benchy-at-depth rows 19.8/52.9 across two identical restarts — the polluted instrument;
  duel-style probes at depth would decide it if pursued).

## (round 37) The corrected block: both batteries on bench v3.0.1 — the score comparison complete (2026-10-27)

- **The corrected wrapper ran as designed** (17:38→20:24, `PHASES="preflight top4qual
  focusqual summary"` honored — the export fix verified live): `campaign-20261027-173843`,
  rc=0, the stack stopped after, production back on the port.
- **§2 gates ALL PASS on both serves** (g1–g4 content form + the g5 note; the byte-level
  nondeterminism note is the engine's documented raw-bytes behavior, not a content fault).
- **The duel arms re-confirm round 36 within noise** — top4cfg: solo 64.6/64.7 (Δ 0.2%),
  c2 108.1, c4 168.1, TTFT d0 148 ms, slope 0.588 (round 36: 64.8/108.0/167.4/0.587 — the
  chart's rows stand, no churn); Focus: solo 57.4/57.5, c2 94.4, c4 153.3, slope 0.665
  (round 33 A/B: 57.9/94.3/150.6/0.663).
- **The v3.0.1 scores — the comparison is now apples-to-apples on all three serves**:
  | serve | tool-eval (/138) | hardmode run (/208) | hardmode subset (/70) | subset fails |
  | --- | --- | --- | --- | --- |
  | production (round 35) | **96** (133) | **94** (196) | **63 = 90%** | TC-92, TC-104 |
  | top4cfg (round 37) | **96** (133 — TIED, same points) | 89 (186) | **53 = 76%** | TC-85, TC-88, TC-89, TC-92, TC-98, TC-101, TC-104 |
  | Focus (round 37) | 95 (131) | 92 (191) | **60 = 86%** | TC-80, TC-92, TC-101, TC-104 |
- **The reading**: the top4cfg matches production point-for-point on the standard suite and
  is the perf leader by far (c2 124%, c4 109%, solo 110%) — but hard-mode costs it 10 subset
  points (76% vs 90%), concentrated in TC-85/88/89/98/101 (the two engines' shared fails are
  TC-92/TC-104). The Focus sits between on scores (95/86%) while at-par-to-crossed on perf
  (solo 98%, c2 108.5%, c4 99%). The default trade is a (user) decision: perf-first (keep
  the top4cfg) vs hard-mode-quality-first (the Focus). The unretrained top-4 selector is the
  suspected mechanism for the top4cfg's subset fails — the AZ retrained-selector route
  (parked) is the principled fix if the trade is deemed unacceptable.
- **The bar (user) scoreboard**: generation tok/s — CROSSED on every front. Prefill — 78%
  (open). Loading — 37% (open, 256 s vs 93.7 s, decomposition still pending). The bar is
  one of three fronts; the remaining two plus the TTFT-at-depth frontier (3%, round 34's
  root cause) are the perf way forward.

- **Round 37 addendum — the upstream look (the continuing-look mandate)**: v1.1.7 is out
  (daily release cadence) and main carries #1430 `TF_GLM_INDEX_CUDA=1` — GLM-5.3-Flash's
  prompt-chunk DSA indexer in two byte-equal CUDA kernels, upstream-measured on a GB10 at
  **cold-TTFT −2% @131k, −9% @524k, −16% @1M** — the first upstream item aimed at our
  TTFT-at-depth/prefill fronts since the v1.1.6 rebase. It is opt-in and unreleased (main
  only). Recommendation: **wait for v1.1.8** (the cadence says imminent) rather than pin
  unreleased main, then A/B the knob as a knobs-phase-shaped cell (gates first — the
  byte-equal claim makes them near-certain — then the duel probes at depth and a prefill
  pass; the pin change gets the explicit `prepare.sh` + image-ID check). The marks policy
  is untouched upstream: our patch experiment remains the cross-request fix.

## (round 38) The (user) hardmode bar — the Focus becomes the served default (2026-10-27)

- **The decision (user)**: "focus on hardmode — no one online is just running the standard
  set; hardmode is the quality measure and we need to either beat production or almost
  equal it." The target is the bench's own headline: production's **run-level 94**.
- **The receipts that frame it**: Focus **92** (2 under — almost equal; its whole gap is
  TC-38 + TC-80, both `wrong_args`, since TC-92/TC-104 fail on production too), top4cfg
  **89** (5 under — fails the bar). The failure-kind analysis (round 37) showed every fail
  on every serve is judgment-shaped (`wrong_args`/`missing_step`) — the top4cfg's deficit
  is the unretrained top-4 selector's, not the engine's.
- **Executed**: `config.sh` defaults swapped to the Focus serve shape
  (`MODEL_ID=local/GLM-5.3-Flash-AZ-Focus-EXL3 MODEL_REVISION=main PATCHES=` empty);
  README/DELTAS/ledger/PERF_PLAN updated; **the chart re-anchored to the Focus cell**:
  loading 41% (230 s — the focusqual restart walls 226–239), solo 98%, **c2 109% — CROSSED**,
  c4 99%, TTFT overhead 92%, **TTFT at depth 3%** (5,647 vs 142 ms), prefill 67%
  (1,020 t/s benchy), ITL 102%. The top4cfg's crossed rows (111/124/108) stay in the
  ledger as the documented perf leader.
- **The beat-94 track (the new quality line)**: (a) a multi-trial hardmode run (`--trials`)
  to measure run variance around TC-38/TC-80; (b) the AZ retrained-selector route — the
  principled fix that could hold the top-4 perf AND trained-routing quality; its blocker
  (the fused-quads EXL3 format) gets re-checked at v1.1.8 — upstream main now carries
  `exl3_layout.zig`/`eaq_container.zig` (active EXL3-format work); (c) the v1.1.8 A/B runs
  regardless (the indexer kernels + any reasoning-path changes).

## (round 39) The (user) bench-parallel decision: tool-eval-bench runs at --parallel 2 from now on (2026-10-27)

- **The decision (user)**: future tool-eval-bench runs are limited to **`--parallel 2`** —
  motivated by performance under load looking awful with RoCE chatter suspected (the
  counters so far show zero error-class events on both CX7 rails — the "chatter" is more
  likely the four concurrent scenario chains themselves contending on the RoCE gather than
  a transport fault; the markspatch cell's duel numbers will quantify it).
- **Scope**: the BENCH's `--parallel` (its `run` invocation) only — the serve's own
  `PARALLEL=4` stream count is untouched (its admission arithmetic is receipted).
- **The changes, at the block's wrap** (scripts are frozen mid-campaign): the
  `run-bench-matrix.sh` default, the `hmtrials`/`settingsab` phase invocations — all
  `--parallel 4` bench lines → `--parallel 2`.
- **Comparability note, stated up front**: the score anchors (production 94, the Focus 92,
  the top4cfg 89) were measured at `--parallel 4` — the SCORES are comparable across the
  change (content-deterministic per variant seed; parallel affects timing, not judgments),
  but the WALL-TIME/duration receipts of new runs are not comparable to the old ones.
  The benchmarks' own throughput rows will read slower per-request by design of the change.

## (round 40) The overnight block's three verdicts (2026-10-28 morning)

- **hmtrials — the variance answer: the Focus's 92 is rock-stable.** Variant seeds 7/8/9
  all scored exactly **92** (plus round-37's default-seed 92 = four draws, one score).
  The 2-point gap to production's run-level 94 is real capability distance, not run
  noise — beating 94 needs a capability change, not re-rolls. Receipts:
  `campaign-20261027-212746/16-hmtrials.md`.
- **settingsab — no lift from the recommended settings.** Temperature 1.0 / top_p 0.95
  scored **91** (one sampled draw, 1 under the greedy anchor). The recommended-settings
  hypothesis is dead for this suite; greedy stays the harness's sampling and the
  deterministic anchor. Receipt: `17-settingsab.md`.
- **markspatch — safe but unproven.** The interior-marks patch (grid-aligned prompt
  states) passed the FULL §2 gates battery (g1–g4 + the g5 note — exactness intact with
  the new resume path) — but the duel moved nothing: solo 57.9/57.9, c2 94.4, c4 151.9,
  TTFT d0 202 ms, **slope 0.664 vs the 0.665 baseline**. Reading: the duel instrument
  does not exercise the cross-request resume path the patch adds (each depth probe is a
  single cold request; the d8192 probe's shared fill prefix with the d4096 probe should
  have engaged the marks but didn't). The patch is NOT falsified — the instrument is
  insufficient. Next: a same-prefix-repeat probe (the same long prompt sent twice; TTFT#2
  vs TTFT#1 is the direct resume test) plus the engine's own prompt-cache counters during
  it (hits/misses/evicted — are interior states kept at all). Receipt: `18-markspatch.md`.
- **The wrap queue landed**: bench invocations at `--parallel 2` (the (user) round-39
  decision — the matrix default, the probe bench line, the hmtrials/settingsab cells;
  the perf-only benchy cells keep their parameterized client counts as the concurrency
  instrument); the redundant tools passes stripped from top4qual/azqual/focusqual (the
  hardmode run covers the 69 — the (user) hardmode-only directive); config.sh's two
  comment lines now describe the Focus default.
- The block wrapped at 00:29:59 (rc 0), the stack stopped clean, production back on the
  port this morning.

## (round 41) The two-shot instrument: the patch works, and the resume-cost gap is named (2026-10-28)

- **The three-way two-shot signature** (the same ~7.7k-token prompt twice per pair,
  `scripts/twoshot-probe.sh`; receipts `campaign-20261028-060716/18-markspatch.md` +
  this entry's production control):
  | serve | shot 1 (cold) | shot 2 (resume) | saving |
  | --- | --- | --- | --- |
  | production | 4,525 ms | **65 ms** | **99%** |
  | ours unpatched | 7,359 ms | 1,440 ms | 80% |
  | ours PATCHED (pair-1 shot-1, after the duel's same-fill prefill) | **1,445 ms** | 1,435 ms | already-resumed |
- **The marks-patch WORKS**: the patched serve's pair-1 shot-1 was already fast (1,445 ms
  vs the unpatched cold 7,359 ms) — it resumed the interior states the duel's d8192 probe
  had just kept (the duel's fill and the probe's fill are the same generator/words/count),
  and the cache counters corroborate (hits climbing 2→4→6, evicted 0). Cross-prompt
  prefix resume — the exact thing the conversation-structure marks never covered — is
  delivered by the patch, with the full §2 gates PASSing around it (second independent
  run). The patch is exactness-safe and functionally proven.
- **Identical-prompt resume already exists unpatched** (7,359 → 1,440 ms, 80%): upstream's
  engine resumes the same prompt without the patch (some end-of-pass state). The patch's
  addition is the cross-prompt case.
- **The duel's d4096→d8192 non-resume explained**: the two prompts share text but the
  junction tokenizes differently (fill_4096's last word + "\n\n" vs fill_8192's
  continuation), so the d4096 state is not an exact token-prefix of the d8192 prompt and
  `find` correctly rejects it. The duel instrument cannot measure cross-length resume;
  the two-shot is the right instrument.
- **The new named gap — restore cost**: our resumed shot costs **1,440 ms** where
  production's costs **65 ms** (22×). Production's block cache remaps already-resident
  blocks; ours restores a snapshot's KV bytes per resume. Even with perfect prefix
  matching, our resume floor is ~1.4 s at 7.7k tokens. That — plus the cold-prefill
  slope (unchanged 0.665, the indexer path → v1.1.8) — is the TTFT-at-depth frontier's
  decomposition.
- **The bench-slowness mechanism (last night's 2× scenario durations) is cache churn**:
  the engine's line during the scenario runs showed hits 0 / evicted 113 with the ~3.4 GiB
  state budget full — four concurrent multi-turn chains evict each other's states before
  their next turn, so every turn re-prefills. The patch adds states (more pressure); the
  lever for the churn is the state budget/eviction policy (`PROMPT_CACHE_GIB`), a knobs
  cell for the next window.
- The stack stopped clean after the block; production's control probe ran read-only
  against the idle serve (six 1-token requests).

## (round 42) Round 41's adoption call is REVISED — the A/B was confounded; the A/A/B probe is the decisive instrument (2026-10-28)

- **The revision (withdrawn, not silently)**: round 41's "the marks-patch works" rested on
  the patched serve's pair-1 shot-1 being already fast (1,445 ms) — but that serve had just
  run the duel, whose d8192 probe prefilled the SAME fill (the duel's generator and the
  probe's are identical — words, formula, count). The unpatched arm's pair-1 shot-1 was
  genuinely cold (7,359 ms). Warm-vs-cold across arms is not an A/B.
- **The deeper finding that forced it**: the UNPATCHED arm's own pair-1 shot-2 resumed 80%
  (7,359 → 1,440 ms) — the stock engine already keeps reusable states cross-request. The
  round-34 mark-policy diagnosis is therefore incomplete: whatever blocks resume in real
  workloads (the bench scenarios' hits-0 churn, the duel's junction divergence), a missing
  interior mark is not the whole story, and the patch's marks may be partially redundant.
- **The decisive instrument, now built**: the probe is A/A/B — body A (fill+tail1) twice
  (A1 cold, A2 = same-prompt resume), then body B (the SAME fill, a different tail): B1
  fast = cross-prompt resume (the fill's prefill reused across a changed tail). The
  markspatch phase now runs the probe FIRST on both arms (probe-before-gates/duel: cold
  readings), plus a post-duel probe on the patched arm (the cross-fill case). The counters
  echo between shots.
- **The adoption decision waits for the clean readings**: if the unpatched arm's B1 is
  already fast, the patch is redundant (do not adopt); if only the patched arm's B1 is
  fast, the patch earns its standing line. The production reference (65 ms) stands from
  round 41's control.

## (round 43) The A/A/B verdict: stock resume is real — the patch question needs one more discriminator (2026-10-28)

- **The A/A/B receipts** (`campaign-20261028-065603/18-markspatch.md`): the stock engine
  (UNPATCHED) already resumes **same-prompt (A2 = 1,434 ms, 80%) and cross-prompt (B1 =
  1,439 ms — the fill reused across a changed tail)** — identical to the patched arm on
  every A/A/B reading (patched: A2 1,434, B1 1,433). The mechanism is v1.1.6's mainline
  paged KV (page-level prefix sharing), not the Store's marks. **On this instrument the
  patch adds nothing.**
- **But the patched duel's TTFTs collapsed** (d4096 2,007 ms vs the 3,784 baseline;
  d8192 1,468 vs 5,650 — slope 0.155 vs 0.665) — and the patched duel ran AFTER the A/A/B
  probe had prefilled the same fill. The unpatched arm never ran a duel-after-probe
  sequence, so the cause is unresolved: the pages would explain it without the patch.
- **The discriminator is staged**: the markspatch phase now runs the duel on BOTH arms
  after each arm's probe. If the unpatched duel's d4096/d8192 drop like the patched ones,
  the pages do the work and the patch is redundant (do not adopt); if only the patched
  duel drops, the snapshot marks add value beyond the pages.
- **Standing regardless of the patch**: production's resume reference is 65 ms where ours
  is ~1,434 ms at 7.2k tokens — the page-resume/restore cost is the gap to close; and the
  cache churn under concurrent load (last night's hits-0/evicted-113) is the bench-slowness
  mechanism with `PROMPT_CACHE_GIB` as the lever.

## (round 44) The discriminator's verdict: the patch is redundant — and v1.1.8 is out with our levers (2026-10-28)

- **The discriminator's answer** (`campaign-20261028-073209/18-markspatch.md`): the
  UNPATCHED duel, run after the probe's fill prefill, collapsed exactly like the patched
  one — d4096 3,785 / d8192 1,485, slope **0.157** (the patched arm: 2,024/1,471/0.155 —
  identical within noise). **v1.1.6's paged KV does the cross-request resume; the marks
  patch is redundant and retires unadopted.** The A/A/B readings were identical on both
  arms too (A2 ~1,435, B1 ~1,435). The round-34 mark-policy diagnosis is superseded: the
  pages handle cross-request prefixes at page granularity; the round-37-era cold baselines
  were simply measured without a prior same-fill prefill.
- **v1.1.8 is out** (176 commits past our pin) with three items aimed at this line:
  **`TF_GLM_INDEX_CUDA=1`** (#1449 — the DSA indexer kernels: upstream's GB10 cold-TTFT
  −2%/−9%/−16% at 131k/524k/1M — the cold-prefill lever); **the fork rule** (#1291,
  `snap_fork` — opt-in per CUDA family: keep a state where a new prompt parts from an
  already-kept one — the repeated-document/changed-question pattern); and a GLM CUDA
  DFlash2 drafter-loader fix (rank-count sizing). Plus the longest-first prompt-cache
  find index and the KV-pool admission fix.
- **The adoption battery is staged** (the `v118` phase): the pin bumps to v1.1.8
  (`812fb040`) with `patches/glm-fork-rule.patch` (one line: `pub const snap_fork = true`,
  upstream's own opt-in mechanism); the battery = the §2 gates → the A/A/B probe on
  v1.1.8+fork (vs the stock A2 ~1,434/B1 ~1,435) → the INDEX_CUDA A/B (the duel unset
  vs =1 — the TTFT slope is the instrument) → hardmode at --parallel 2 (the score
  receipt; receipts-equal-or-better vs the Focus's 92 decides the adoption). The
  `--parallel 2` bench shape (round 39) and the hardmode-only directive (round 38) are
  baked into the phase.

## (round 45) v1.1.8 ADOPTED — receipts-equal-or-better on every instrument (2026-10-28)

- **The battery** (`campaign-20261028-095233/19-v118.md`, engine v1.1.8 `03d08f1e` + the
  fork rule staged): §2 gates **ALL PASS** (g1–g4 content form); **hardmode 92** — equal
  to the Focus's v1.1.6 receipts (rounds 37/40); solo 57.5/57.9, c2 93.9, c4 152.4 —
  within noise of v1.1.6; the A/A/B probe identical to stock (A2 1,432 / B1 1,438 vs
  1,434/1,435). **Adopted: the pin stands at v1.1.8 (`812fb040`), the standing config
  serves pristine (PATCHES empty — the fork rule showed no delta on the instrument and
  stays unstaged; the marks patch retired in round 44).**
- **The INDEX_CUDA A/B was confounded by the phase order — the lesson repeats**: the
  unset arm's duel ran after the probe's fill prefill (pages-resumed: d8192 1,472 ms),
  the INDEX_CUDA arm's duel ran cold on a fresh serve (d8192 5,627 ms — matching the
  v1.1.6 cold baselines exactly). The 8k-depth instrument cannot resolve the indexer:
  upstream's own curve claims ~0.3% at 8k and the real numbers at 131k/524k/1M. **The
  INDEX_CUDA receipt is owed at depth** — the depth ladder's 64k–384k TTFT rows with the
  env unset vs =1 is the next window's instrument.
- **What v1.1.8 brings the line**: the indexer kernels (the deep-depth lever, receipt
  owed), the fork rule (unstaged — no measured delta; the pages already cover the
  shared-prefix case), the longest-first cache index, the KV-pool admission fix, and the
  DFlash2 drafter-loader fix (rank-count sizing — a real GLM CUDA bug fix).
- **The frontier decomposition stands**: cold prefill (the indexer at depth — receipt
  owed), the resume/tail cost (~1,434 ms vs production's 65 ms — the granularity gap),
  and the cache churn under load (the `PROMPT_CACHE_GIB` lever).

## (round 46) Two window cells staged: the INDEX_CUDA depth ladder + the PROMPT_CACHE_GIB churn cell (2026-10-28)

- **`indexdepth`** (the owed round-45 receipt): the `TF_GLM_INDEX_CUDA` A/B at depth —
  the TTFT ladder 64k/128k/192k/256k/320k/384k (`max_tokens=1`, actual prompt_tokens read
  from each response's usage) plus a 512-token decode row at 96k, both arms on the standing
  config (Focus pristine, dense4fp8, 524288×4), each arm a fresh serve. **The ladder fills
  are per-row distinct from token 1** (decoration keyed on `(i + depth)`) — a close read of
  the duel's generator showed its rows are token-prefixes of each other, so it is NOT
  reused here; no ladder row can resume another's pages, and the prompt-cache counters
  double as the no-sharing proof (one miss per row). Walls are client-side; the fixed share
  is identical across arms, so the deltas stand.
- **`cachegib`** (the round-45 queue item): the `PROMPT_CACHE_GIB` churn cell — a fixed
  load of 24 distinct ~64k fills through 8 rounds × 4 concurrent streams (rounds 1–6 cover
  all 24, rounds 7–8 re-visit fills 1–8), engine-default store vs `PROMPT_CACHE_GIB=16`
  (auto-drop to 8 on an admission refusal — the ceiling itself is a receipt). Pressure
  math: the 3.9 GiB default store thrashes once KV state exceeds ~2.6 KB/token; the
  expected reading is default-arm revisit rounds ≈ cold (99%) vs gib16 revisit ≈ resume
  (the fabricated-data dry run of the receipt parser reads exactly this shape).
- Both phases are session-specific (retire with their receipts), both arms have identical
  pre-histories (fresh serve each), and both receipt parsers were dry-run against
  fabricated data before staging. Runbook: `PHASES="preflight indexdepth cachegib summary"`.

## (round 47) The window's receipts: INDEX_CUDA is a real, growing win at depth; the cache knob is budget-bound (2026-10-28)

- **`indexdepth` — the owed receipt is IN** (`campaign-20261028-113509/20-indexdepth.md`,
  both arms fresh serves on the standing config, Focus pristine, 524288×4):
  `TF_GLM_INDEX_CUDA=1` **flattens the depth-prefill growth**. TTFT deltas: 64k +0.0%,
  **131k −2.6%**, 192k −2.7%, 256k −3.6%, 320k −4.5%, **384k −5.5%** (350.5 → 331.4 s).
  The off-arm marginal slopes RISE with depth (0.860 → 0.976 ms/tty) — cold prefill is
  superlinear; the on-arm stays flat-ish (0.816 → 0.882). Overall cold slope 0.891 →
  0.842. The no-sharing proof held: exactly 8 misses per arm (the per-row-distinct
  fills did not resume each other). Loading unchanged (224.5–226.8 s), decode unchanged
  (38.3 vs 39.6 t/s at the 96k row — the indexer is prefill-side only), d0 unchanged
  (+0.7%, noise).
- **Upstream's claim transfers**: upstream (#1449) claimed −2% cold-TTFT at 131k — we
  **measured −2.6%** at exactly that depth; their curve's shape (gains growing with
  depth) matches our ladder's trend (−5.5% at 384k, extrapolating ≈ −6.5% at 524k vs
  their −9% — their deeper points sit beyond our ladder). Written into
  `docs/ECOSYSTEM.md` as a verification datum.
- **Adoption status: pending one gates serve** — the knob rebuilds the same indexer
  scoring in two dedicated kernels (upstream's contract is byte-equal), but our exactness
  gates never ran WITH it, so the discipline holds: **next window = one restart + §2
  gates with `TF_GLM_INDEX_CUDA=1` (~12 min), then the default flips** in config.sh.
- **`cachegib` — the knob is NOT free at the standing budget, and the churn mechanism is
  now a controlled fact** (`campaign-20261028-113509/21-cachegib.md`):
  - The **default arm (complete)**: 8 rounds × 4 concurrent ~116k-token fills — hits stay
    **0 across all 32 requests**, evictions climb **+5/round** (37 by round 8), and every
    revisit round re-prefills in full (walls 438.1–440.5 s, revisit = cold 99%+). The
    overnight bench's hits-0/evicted-113 pattern is now receipted as the store thrashing
    at 2.6 GiB under concurrent distinct-prompt load.
  - **The gib16 arm is void, and that is its own receipt**: `--prompt-cache-gib 16` is in
    the engine's argv (verified via `docker inspect`), yet the serve line reads **2.6 GiB
    prompt cache — identical to default** — and the arm's counters march default's
    trajectory exactly (evicted 2/7/12/17). Whether the engine budget-caps the knob (the
    serve line: 13.4 GiB left after a 12.2 GiB reserve) or ignores it is unresolved from
    one serve. **Discriminator cell queued**: one serve at `PROMPT_CACHE_GIB=10` with
    `CONTEXT=262144` — if the store lifts, the budget binds and the cache lever is a
    window/stream trade; if it stays 2.6, the flag doesn't lift the store upstream-bound
    (ECOSYSTEM data).
  - **The lever reframed**: at 524288×4 the kept-states store does not grow on request —
    the cache lever is shrinking CONTEXT/PARALLEL, or the **`--spill-gib` route (v1.1.8):
    evict to disk instead of dropping**, the natural next cell.
- **The window's slow half is owned**: the cg fills measured **115,771 actual
  prompt_tokens each** (designed ~64k; my word→token assumption was ~2, actual ~3.5) —
  each round is ~463k cold-prefill tokens at the machine's ~1,060–1,220 tok/s aggregate ≈
  438 s, so 8 rounds ≈ 58 min/arm. The gib16 arm's last 4 rounds (a proven no-op — the
  counters already matched default exactly) were killed at 13:3x to hand production back
  ~25 min early. Ops lesson rerun (the ledger's "watchers must exclude their own command
  text"): the kill's `pkill -f "cg-f"` matched the issuing ssh's own command line and cut
  the session mid-kill — the pgid TERM had already landed; the follow-up verified the
  window gone (bracket-trick pgrep), stopped the stack, and freed the port.
- **Wrap-queue**: (1) the gates-with-INDEX_CUDA serve then the config flip; (2) the GIB
  discriminator serve; (3) a `--spill-gib` churn cell (the load recipe now exists — the
  phase's bodies/curls are reusable); (4) the retired session phases (markspatch, v118,
  indexdepth, cachegib) get deleted from run-qualification.sh now that their receipts
  are in; (5) optional: a read-only prodqual extension to take PRODUCTION's depth-prefill
  ladder for the same table (a new comparison axis: our 0.84–0.98 ms/tty cold at depth vs
  theirs).

## (round 48) The ~50-minute window staged: the INDEX_CUDA adoption gates, the GIB discriminator, the spill cell (2026-10-28)

- **`ixgates`**: §2 gates + the standard duel WITH `TF_GLM_INDEX_CUDA=1` — the exactness
  receipt the adoption needs (the perf side is already in: round 47's depth ladder).
  Gates PASS → config.sh flips the knob as the standing default in the digest sitting.
- **`gibdisc`**: the round-47 discriminator — arm 1 serves `PROMPT_CACHE_GIB=10` at the
  standing 524288×4 (10 fits inside the serve line's 13.4-GiB slack: a budget cap would
  lift the store off 2.6 GiB); if it still reads 2.6, arm 2 drops the window to 262144 —
  a lift there means the knob works and the standing budget binds; still 2.6 means the
  flag doesn't lift the store at all (upstream-bound data).
- **`spillgib`**: the round-47 reframe's cell — `--spill-gib 32` (new: plumbed through
  start.sh/config.sh this round) with the round-47 churn load scaled to churn cheaply:
  16 distinct ~56k-token fills, 6 rounds × 4 streams (rounds 1–4 cover, 5–6 revisit fills
  1–8); the working set (~900k tokens) exceeds the store's ~810k-token hold, so rounds
  5–6 MUST come from somewhere — fast revisits = spill restores from disk; revisit = cold
  = spill does not (round 47's default arm is the control). A serve refusal on the new
  flag is itself the receipt.
- All three receipt parsers/serve-line greps dry-run against fabricated data and the
  observed serve line before staging. Runbook: `PHASES="preflight ixgates gibdisc spillgib summary"`,
  ~50 min (2–4 restarts + one ~16-min churn load + one ~14-min cover ladder).

## (round 49) INDEX_CUDA ADOPTED; the cache levers close; seven retired phases deleted (2026-10-28)

- **`TF_GLM_INDEX_CUDA=1` is the standing default** (`campaign-20261028-141916/
  22-ixgates.md`): §2 gates **ALL PASS** under the knob — upstream's byte-equal contract
  shown on our serve (g1 determinism, g2 drafted=plain, g3 resumed=fresh, g4
  concurrent=solo; g5 note as always) — and the duel is within noise of the v1.1.8
  references (solo 57.0/56.9, c2 93.7, c4 152.3 vs 57.5–57.9/93.9/152.4). Combined with
  round 47's depth ladder (−2.6% at 131k → −5.5% at 393k, growing with depth), the
  receipts-equal-or-better bar is cleared. config.sh exports the knob (outside the config
  fingerprint — no rebuild on the next start); the README table carries the row.
- **The gibdisc receipt's first conclusion is WITHDRAWN** (the correction protocol):
  `23-gibdisc.md` printed "arm1 LIFTED the store — the knob works" — that was my
  branch-anchoring bug (I matched the literal "2.6"): the data line reads **2.5 GiB with
  GIB=10**, i.e. the store did NOT lift to ~10. The full evidence now reads: GIB=16 →
  2.6 with default-identical counters (round 47, behavioral); GIB=10 → 2.5 (round 48).
  **`--prompt-cache-gib` does not size the kept-states store at 524288×4.** The last
  open read is one cheap serve (`GIB=10 @ CONTEXT=262144`) — queued.
- **The spill lever is CLOSED for GLM** (`24-spillgib.md`): the engine refused the serve
  — spill covers "Nemotron, the Qwen hybrids and Flash Next only on CUDA". Upstream-bound
  (ECOSYSTEM); the SPILL_GIB plumbing stays staged for the day upstream covers GLM.
- **Seven retired session phases deleted** from run-qualification.sh (markspatch, v118,
  indexdepth, cachegib, ixgates, gibdisc, spillgib — 522 lines; all receipts live in
  results/ and the docs). The script is back to its standing phases only.
- **The frontier after today**: cold prefill at depth is now receipted AND mitigated
  (the indexer knob — the standing config serves it). The remaining frontier items:
  (1) the resume/restore cost (~1,434 ms vs production's 65 ms at 7.2k — the source-level
  dig, no window needed); (2) the churn-under-load reality (the store is fixed at ~2.6 GiB
  and spill doesn't cover GLM — the honest statement is the store thrashes under
  distinct-prompt concurrency and the fix is upstream's); (3) the beat-94 quality track
  (the AZ retrained-selector route, blocked on the EXL3 requant format — watch upstream's
  exl3_layout/eaq_container work).

## (round 50) The audit: vision is a real parity gap (production serves it, we don't); the docs swept current (2026-10-28)

- **Vision: NOT included — and now it's a receipted gap, not a silent one.** The facts,
  all checked today: the AZ-Focus checkpoint carries the vision tower (347
  `model.visual.*` tensors in its safetensors index; `architectures:
  Glm5NextForConditionalGeneration`); TensorFold v1.1.8 supports GLM vision on CUDA
  behind `--vision` (`families/glm/cuda_vision.zig`, `cuda_images.zig`; the optional
  `--vision-image-tokens` limit; "every rank must hold the rows rank 0 sends" — a real
  memory term at 524288×4); **production's argv carries `--vision`** (docker inspect,
  read-only). Our serve has never passed it, no receipt exists for it, and the bench is
  text-only — so neither the quality bars nor the perf instruments have ever seen it.
  **Enabling is a (user) call**: a `VISION` config knob + a qualification cell (the
  admission arithmetic at 524288×4, the §2 gates, a smoke image request, and the
  carry-cost duel). Recorded as an open item in TOOL_EVAL_TESTING.md and the README's
  status table.
- **The docs sweep** (the second audit question): README's status moved to forty-nine
  rounds + the indexer adoption; DELTAS's header carries the knob; VALIDATION §4 and
  CONTEXT_512K_PLAN gained dated 2026-10-28 superseding notes (the engine is v1.1.8+knob;
  the gates re-qualified on it; the depth fact-ladder's last run is v1.1.6-era — re-run
  queued); PERF_PLAN gained the rounds-45–49 status line. The historical v1.1.6 mentions
  are dated entries and stand as history. The anchors TSV is unchanged — the knob's duel
  (round 48) is within noise of the standing row, and the receipts carry it.

## (round 51) VISION ENABLED as the standing default; the photo suite staged (2026-10-28)

- **The (user) calls, verbatim in substance**: vision is a want — "we should have enabled
  it by now"; 512k context is a want; **no more performance testing**; nothing below
  512k max context; **get vision admitted as a default before anything else**; and the
  test must check detail perception — "more than just a general question", long answers
  allowed. All recorded here as standing directives.
- **The source facts that made the flip safe** (v1.1.8): the checkpoint's 347
  `model.visual.*` tensors are **BF16** (verified in the safetensors headers — the
  requant's `-vb 16` recipe), which the engine's vision loader (`Lin` union: bf16 /
  MXFP8 / NVFP4) reads directly; `--vision` defaults to 4 images × 4096 tokens;
  requests are OpenAI-style `image_url` data-URL parts; the flag family also carries
  `--vision-urls`, `--vision-offload`, `--vision-max-images` (1–64),
  `--vision-image-tokens`. Production already serves `--vision` (round 50) — the parity
  gap closes with this default.
- **The flip**: config.sh defaults `VISION=1` → start.sh passes `--vision` (knob exports
  sit outside the config fingerprint — no rebuild). 524288×4 stands.
- **The test suite** (`assets/vision/`, committed): a synthetic four-quadrant PNG
  (byte-verified: TL=red TR=green BL=blue BR=white) plus **three Wikimedia Commons
  photos** — the Golden Gate Bridge from a beach, a plains zebra at a waterhole (rainbow,
  impala), a desert STOP sign with a row of numbered mailboxes — CC BY-SA with
  attribution (`NOTICE.md`, `_fetch-meta.json`). Each image's ground truth was **viewed
  by the agent** on production's vision-enabled serve. Per image the phase asks a
  targeted question (max_tokens 64) and a long-form detail question (max_tokens 1024),
  scoring **detail coverage n/m** against 4–7 agent-verified details (the receipt
  parser dry-run shows the scoring discriminates: a thin answer reads 5/7, a full one
  7/7).
- **The `visionen` phase** (staged, retires with its receipt): the standing serve's
  admission at 524288×4 with `--vision` (a PARALLEL=2 retry preserves the window if the
  arithmetic refuses), the four-image suite, and the §2 gates on the vision serve.
- **Dropped per the (user) call**: the queued `GIB=10 @ CONTEXT=262144` serve (a sub-512k
  config). Sub-512k cells are out of scope.

## (round 52) The vision refusal digested: the image lacked CUDA 13 cuBLAS — fixed in the Dockerfile (2026-10-28)

- **The refusal receipt** (`campaign-20261028-151636/25-visionen.md`): the vision serve
  refused on both PARALLEL attempts — the engine's own words: *"`libcublasLt.so.13` is
  required for CUDA vision towers (`--vision`) and CUDA cuBLASLt projections; install the
  NVIDIA CUDA 13 cuBLAS runtime; a driver-only host does not include it."* The checkpoint
  digest had already read (47/47 files cached); the failure is purely the runtime image.
- **The root cause**: our image is multi-stage — stage 1 builds on
  `nvcr.io/nvidia/pytorch:26.07-py3` (a CUDA-devel base WITH the toolkit, where the
  fatbins compile); stage 2 is a minimal `ubuntu:24.04` runtime that ships the binary,
  NCCL, and the RDMA user space — but NOT cuBLAS. The vision tower's projections run
  cuBLASLt; the main model path never needed it, so the gap was invisible until --vision.
  Production runs the same NVCR base and carries `/usr/local/cuda-13.3` with
  `libcublasLt.so.13` (verified read-only via docker exec) — that is how it serves vision.
- **The fix** (image/Dockerfile): stage 2 now `COPY --from=build`s
  `libcublasLt.so.13` + `libcublas.so.13` (the soname symlinks materialize as full
  copies) into `/opt/tensorfold/lib/` — already on the loader path via the image's
  ld.so.conf, so no LD_LIBRARY_PATH is needed.
- **The support target, from production's own serve line (the user's reference)**:
  *vision: image and video input, a 1.05 GiB tower with 1.41 GiB of workspace; at most
  2048 tokens a picture, 16384 a clip (128 frames at 2 a second)*. Our flag mapping:
  `--vision-image-tokens 2048` (production parity; our default was 4096),
  `--vision-video-tokens` already defaults to 16384, the 128-frames/2fps clip cap is the
  engine's own, and the tower/workspace figures are the engine's arithmetic. start.sh now
  passes `--vision --vision-image-tokens 2048` when VISION=1.
- **The rebuild**: the Dockerfile is outside the config fingerprint — an explicit
  `prepare.sh` runs on the head (CPU-side, production untouched); the visionen re-run
  follows once the image lands.

## (round 53) The rank mismatch digested: vision is rank-agreed + the decode libs are pinned (2026-10-28)

- **The refusal receipt** (`campaign-20261028-153511/25-visionen.md`): *"REFUSE rank
  settings: rank 1's vision differs from rank 0's (--vision (or its image tokens) given
  to some ranks and not others)"*. **The cause was my flag placement**: `--vision` went
  into `rank0_flags` (head-only server flags), but vision is a **rank-agreed setting**
  the engine cross-checks at start. The prompt-cache/spill flags above it are genuinely
  rank-0-side; vision is not. **Fix**: a new `rank_both_flags` array appended to BOTH
  rank commands in start.sh (rank 1's docker run on the worker included).
- **Two decode-lib warnings rode along** (head log): `libturbojpeg.so.0 is required for
  JPEG inputs (--vision)` and `libwebpdecoder.so.3 ... for WebP inputs`. Non-fatal at
  serve time but fatal at first JPEG input — and our test suite is three JPEGs. Source
  read: the engine dlsyms the **tj3 API and version-checks** what it loads
  (`core/picture/jpeg.zig` pins 3.1.4.1; `webp.zig` pins 1.6.0, code 0x010600) — distro
  libjpeg-turbo 2.x / libwebp 1.3.x do not carry it. Production's container carries
  neither (torchvision bundles its own codecs) — no base-image shortcut.
- **The fix** (image/Dockerfile, stage 1): build libjpeg-turbo **3.1.4.1** (cmake, SIMD
  via nasm) and libwebp **1.6.0**, install the real shared objects renamed to the
  sonames the engine searches — `libturbojpeg.so.0` and `libwebpdecoder.so.3` — into
  `/out/lib/`, which ships to the runtime stage's `/opt/tensorfold/lib/`: both the
  image's loader path AND the engine's `../lib` search location. (Also confirmed:
  `libtfgrammar.so` is already in the image — structured output is covered.)
- The rebuild runs on the head; the visionen re-run follows.

## (round 54) The 500s digested: the chat template is text-only — a vision template generated from the tokenizer (2026-10-28)

- **The serve came UP with the tower resident** (`campaign-20261028-154906/
  25-visionen.md`): model 84.75 GiB (the ~1.25 GiB vision tower, was 83.50), 12.6 GiB
  left — the cuBLAS fix and the rank agreement both held, and the §2 text gates ALL
  PASS. Every image request 500'd.
- **The engine's own warning named it** (head log): *"--vision: the chat template writes
  no image placeholder for an image part, so image requests will fail; serve with
  --chat-template naming a template that has an image path."* The source confirms the
  contract (server.zig `templateImages` probes the render and looks for
  `image_token_id`; images.zig `expand` counts ONE placeholder per part and expands each
  to the image's rows).
- **The root cause is the EXL3 conversion line, not our serve**: BOTH the AZ-Focus
  requant AND the stock Mia EXL3 ship the same **text-only chat template** — its media
  branch emits `"<reminder>You are unable to process this image because you don't have
  multi-modal input ability…</reminder>"`. **Production's images work anyway because the
  PYTHON engine routes media parts through the model's HF processor, which writes the
  markers itself** (`<|​begin_of_image|><|​image|><|​end_of_image|>` — the repo's own
  vision tools show the convention); the native engine delegates the media branch to
  the template. Upstream-bound data (ECOSYSTEM.md).
- **The fix**: `image/share/glm5-next-vision.jinja` — the checkpoint's own template with
  ONLY the media branch replaced (image → begin_of_image+image+end_of_image, video →
  begin_of_video+video+end_of_video, audio keeps the reminder), generated by
  `image/gen-vision-template.py` **from the tokenizer's exact bytes** (the media tokens
  carry zero-width characters — never hand-typed). Baked into the image at
  `/opt/tensorfold/share/`; start.sh passes `--chat-template` with it on BOTH ranks
  under VISION=1. The template is a build artifact of the tokenizer + base template —
  regenerate for a new checkpoint.

## (round 55) VISION VERIFIED — the suite passes on the native line at production-equivalent perception (2026-10-28)

- **The enablement receipt is complete** (`campaign-20261028-165021/25-visionen.md`):
  the serve came up with the tower resident (model 84.75 GiB, 12.6 GiB left), the
  startup template-probe warning is GONE, the §2 gates ALL PASS on the vision serve, and
  the four-image suite read **targeted 4/4, detail coverage 23/25** (the bridge answer
  names International Orange; the zebra answer catches the stripe transition and the
  open mouth; the sign answer reads STOP, the octagon, the mailboxes and their numbers).
- **The production comparison** (the same suite read-only against production,
  prodqual-style; both raw answer sets + the scored comparison are in the campaign
  dir): **ours 4/4 targeted, 23/25 detail — production 4/4 targeted, 22/25 detail**.
  The subjective read (the agent viewed every image and both full answer sets):
  equivalent depth and accuracy. Two honest notes: (1) production's targeted answers
  needed a 1024 budget (the python engine reasons before answering — at 64 tokens they
  truncated; ours answered complete at 64); (2) the one scoring divergence — production
  read specific mailbox numbers ("77", "783") that don't match the legible
  5835/5841/5909 (apparent misreads the keyword scorer under-credited); ours read the
  numbers generically and did not misattribute.
- **The ground-truth correction** (the correction protocol): both engines wrote the
  zebra "walking from left to right" — the manifest's original ground truth said "to
  the left" and was WRONG (the head faces right; re-viewed today). The manifest is
  corrected with a note; both engines out-read the provided ground truth on that detail.
- **The structural note**: both lines run the same GLM-5.3-Flash tower (BF16 in both
  checkpoints) — this comparison validates the native stack end to end (decode → tower
  → cuBLASLt projections → rows → the language model).
- **The chain that took five windows, all receipted**: the missing CUDA 13 cuBLAS
  runtime (round 52), the rank-agreed vision setting (round 53), the pinned decode
  libraries (round 53), libwebp's static-by-default CMake (round 53 fix), and the
  text-only EXL3 chat template (round 54 — the python engine's processor path masks it
  on production; upstream-bound data in ECOSYSTEM.md).

## (round 56) The depth re-run re-pointed to the standing config; the resume gap root-read at source level (2026-10-28)

- **The depth phase modernized** (the (user) catch: PARALLEL=2 was dense8-era legacy): the
  phase now serves the **standing defaults** (bare `restart` — dense4fp8 +
  `TF_GLM_INDEX_CUDA=1` + vision at 524288×4) and cites the v1.1.6-era 12/12 receipt as
  its comparison anchor. Staged for the next window.
- **The resume-cost dig, source level** (no window): our ~1,434 ms resume of a 7.2k-token
  state vs production's 65 ms is **architectural, not incidental**. The native engine's
  prompt cache is a SNAP-COPY design: a kept state is a private buffer ("its MLA prefixes
  beside its KDA states", `kept.zig`), and a resume (`restore` → `copySnap`) **copies the
  whole saved state back into the resuming slot's pools** (one command-buffer submission +
  `finish`), with a peer `mirror.send(.restore)` exchange around it and `overwrite`/
  `copyOut` bookkeeping when the resuming slot differs from the state's home. The state at
  7.2k tokens is ~23 MB (the cachegib calibration: ~3.2 KB/token) — the copy's bandwidth is
  trivial; the cost is the synchronization structure (the device-wide `finish`, the peer
  exchange, the request pipeline around it) — the resume-specific share is ~1.2 s of the
  1,434 ms wall. **Production's 65 ms is the python engine's page-remap design**: its kept
  pages ARE pool pages, reused by reference — nothing is copied.
- **The lever** (recorded in ECOSYSTEM.md as upstream-shaped, nothing posted): the engine
  carries a **paged prompt store** (`prompt_radix.zig`; `rules.paged`, "0: not paged") —
  a remap-style resume path — but the GLM family does not wire it (`engine_paged=false`
  default; the snap_* decls set lookahead/fork/grid, never paged). If upstream's paged
  store is GLM-ready, a family-level enable is the production-class resume lever; it
  needs the backend's page-ownership callbacks. The second lever is batching the peer
  mirror out of the restore's critical path.
- **The upstream watch**: zero commits on main since v1.1.8 (no v1.1.9 yet) — the beat-94
  blocker (the EXL3-format work for the AZ retrained-selector route) is unchanged; the
  watch stays open.

## (round 57) The standing config fully qualified: depth 12/12 on dense4fp8 + the indexer + vision (2026-10-28)

- **The depth gate passed clean** (`campaign-20261028-174914/04-depth.md`): 3/3 facts at
  8k, 64k, 192k, and 384k — **12/12** — on the standing default (dense4fp8 +
  `TF_GLM_INDEX_CUDA=1` + vision resident, 524288×4, the serve line's 84.75 GiB model
  with the tower). Equal to the v1.1.6-era 12/12 anchor.
- **The line's qualification is now complete end to end on ONE config**: the §2 gates
  (rounds 48/55), the depth fact ladder (this round), hardmode 92 / tool-eval 95
  (rounds 37–45's Focus receipts), the depth-prefill receipt (round 47), the vision
  suite at production-equivalent perception (round 55), and the perf anchors within
  noise. Everything the (user) asked for — vision, 512k, the indexer — serves as the
  default with receipts.
- **What remains open** (no windows needed): the upstream watch (v1.1.9+ decides the
  next adoption; the beat-94 blocker — the EXL3-format work for the AZ retrained-
  selector route — unchanged as of round 56's check); the resume lever (the paged
  prompt store, upstream-shaped, round 56); and the two disk items awaiting (user)
  calls (vllm-node-b12x 28 GB ×2; the AZ/Focus mirror copies ~154 G + ~328 G).

## (round 58) The disk choreography done; the requant's format gap root-read — the bridge is a dequantize (2026-10-28)

- **The publish item closes** ((user)): what we run is already out there (the served
  checkpoint IS AxonZeta's public release — MIT, the zai-org BF16 lineage, its
  azfocus_provenance.json carries the tensor-level lineage). Nothing to publish.
- **The disk choreography** ((user) confirmed): deleted the upstream mirrors
  (AxonZeta 164 G, azampatti 170 G), the TR3 variant (164 G) and two 1-G stubs — ~200 G
  freed (the blob store hardlinks dedup the logical sizes). Kept: the served Focus, the
  drafter, production's Mia stock, the requant working dir (the repack source), and the
  NVFP4 repo + spark-vllm docker untouched ((user): the vLLM docker is not this
  project's — the ledger's vllm-node item DROPPED). 2.0 T free.
- **The requant format gap, root-read to the byte level**: the layouts differ in THREE
  places — (1) the routed experts' `.mcg/.trellis/.suh/.svh` quads are STRUCTURALLY
  IDENTICAL in both (same dtypes/shapes; `.mcg` is `I32 [1]` vs `I32 []` — a marker
  reshape); (2) the merged `conv1d.weight [24576,1,4]` splits into
  `q/k/v_conv1d.weight [8192,1,4]` (a dim-0 slice); (3) **the 290 non-expert linears**:
  the 1.6.4 converter stored them as trellis-coded 4-bit payloads
  (`trellis I16 [256,1536,64]` + `suh/svh` scales), while the engine's old layout wants
  **BF16 `.weight`** (`q_proj.weight BF16 [8192,4096]`) — the engine quantizes those
  linears ITSELF at load ("290 linears packed (4/4 bits)").
- **So the bridge is a DEQUANTIZE, not a repack**: reconstruct BF16 weights from the
  trellis payloads (the pipeline's own az_dequant.py machinery — exllamav3's dequant —
  already proved this path on the azampatti source), emit the old layout (BF16 .weight
  for the 290 linears, the experts' quads copied through with the mcg reshape, the conv
  split, the old-shape quantization_config: bits 4 / codebook mcg / head_bits 16 /
  scope glm53_routed_experts_only), then the serve attempt is the format verdict and
  the azqual battery — gates + hardmode — finally runs: **the beat-94 verdict on the
  retrained selector unblocks.**
- **The both-hosts plan** ((user)): the dequant/re-emit job shards naturally by layer —
  the head and the worker each process half the 23 shards against a shared output dir,
  the expert quads and unquantized tensors copy byte-identical. Validation: one window,
  the serve attempt first (the refusal would be the receipt), then the battery on the
  green serve.

## (round 59) The requant bridge: the loader verification (the user's get-it-right-first-time standard) (2026-10-28)

- **The engine's EXL3 contract, from its own source** (`families/glm/cuda_exl3.zig` line
  26): *"The EXL3 checkpoints the kernels read: 4-bit routed experts on the 'mcg'
  codebook (everything else bf16)."* `admit()` requires integer `bits=4` +
  `codebook="mcg"` — the requant's config carries both. So the target layout is exactly:
  experts as 4-bit mcg quads, all other linears as BF16 `.weight` (the engine packs
  those itself at load — the serve line's "290 linears packed (4/4 bits)").
- **Verified against the loader, item by item**:
  - The routed experts: the quads are byte-structure-identical between the requant and
    the served Focus, and the `.mcg` markers carry the SAME magic value (-877912083) —
    the requant's experts are genuine 4-bit mcg (the -eb 16 did not change the mcg
    coding). **Copy-through with the []→[1] reshape is CONFIRMED.**
  - The conv1d: the loader accepts the MERGED form itself (`convs()` reads
    `conv1d.weight [3*channels, conv, 1]` and cuts it three ways — "mlx-lm's one"); the
    requant's tensor is `[24576, 1, 4]` — a last-two-dims transpose feeds the loader's
    own cut. **No 3-way split needed.**
  - The linears: `weights.zig` stacks per-projection tensors by exact name
    (`q_proj/k_proj/v_proj/...` — `src()` fails with the receipted MissingTensor), BF16
    at the old checkpoint's shapes. **The trellis payloads must dequantize to BF16
    `.weight` — confirmed required.**
  - The shards: the loader resolves tensors through the index's name→file mapping —
    shard naming is free.
- **Still open before the script is trusted** (the next sitting's first work): (1) the
  exllamav3 1.6.4 dequant path — the exact trellis→BF16 math and a callable API in the
  installed venv (az_dequant.py proved the machinery on a 16-bit source; the 4-bit
  trellis decode is the same reader); (2) the converter's qkv fusion order (q,k,v vs
  another order) — read from exllamav3's writer source, not assumed; (3) the fail-fast
  design: a one-layer dry-run printing the full name-mapping diff before the full run,
  per-tensor NaN/Inf/scale sanity on every dequantized weight, and a name-set equality
  check against the loader's expected inventory.

## (round 60) The repack script built — the name mapping verified EXACT on the real data (2026-10-28)

- **The writer-side verification completed** (the round-59 open checks): the dequant API
  is exllamav3's own — `Exl3HfLinear.dequantized_weight()` ("W as (in, out) fp16,
  reconstructed from the trellis" — trellis decode + Hadamard + scales, the exact
  inverse of the converter); the qkv fusion order is **q|k|v** (confirmed from
  `gated_delta_net`'s own arithmetic: `fdim_qkv == 2*nk*kd + nv*vd` and the conv stash
  map's piece order); the conv1d merged form slices q|k|v the same way.
- **The script** (`scripts/az-repack.py`, runs on the head's exllamav3 venv): four
  modes — `--mode names` (no torch/GPU: the name mapping applied to all 150,891
  requant tensors and diffed against the served Focus's inventory), `--mode dry` (the
  GPU gate: one dense + one expert layer end to end, per-tensor finite/absmax checks
  with the name in every failure, the name-set + shape equality vs the reference,
  the mcg magic verified on every expert group), `--mode full` (both hosts: the source
  staged to the worker whole-file, both halves on each host's GPU, the outputs merged
  and verified, the old-shape config + an honest repack_provenance.json written),
  and the internal work/merge modes. 26 fail-fast `die()` points, every one naming its
  tensor or step.
- **The name-level pre-flight ran on the real data: EXACT** — 150,226 produced names ==
  the reference's 150,226, zero missing, zero extra. The first `--mode names` run
  actually caught the one design bug (the conv1d: the reference uses three separate
  q/k/v_conv1d tensors — the 3-way slice, not the transpose I'd designed) before
  anything touched a GPU: 150,158 + 68 = 150,226 exactly.
- **The runbook** (the user drives): production offline → `git pull` on the head →
  `--mode dry` (the log comes back to the agent for inspection) → on the OK, `--mode
  full` (~1 h; the worker needs the exllamav3 venv — the script checks and prints the
  install line if missing, with a head-only fallback documented) → the placement + the
  serve window (the format verdict), then the azqual battery on green.

## (round 61) The repack runbook staged: the worker's dequant venv provisioned, the az2 battery phase written (2026-10-28)

- **The worker's exllamav3 venv provisioned by the agent** (the round-60 script's one
  prerequisite): `~/.venv-exl3` on the worker with torch 2.14.1+cu130,
  exllamav3 1.6.4 (the ext compiled for the device at first import — ninja + the CUDA 13
  toolkit + python3.12-dev were the missing pieces, all resolved), safetensors,
  transformers. **Verified to the last symbol**: `ext.reconstruct` present and
  `Exl3HfLinear` importable — the dequant path the repack calls is complete on both
  hosts. The install lessons: the venv's bin must be on PATH for torch's ninja check,
  and the ext compile needs python3.12-dev (Python.h) — neither is a surprise now.
- **The `az2` phase written** (session-specific; retires with the verdict): places the
  repacked checkpoint (`~/az-repack-out/merged`) into the HF cache as
  `local/GLM-5.3-Flash-AZ-EXL3`, serves it at the standing config (VISION=0 — the cell
  tests the selector's text quality; the requant's template is the text-only family
  template), and on a green load runs the §2 gates + hardmode at --parallel 2 —
  **the beat-94 verdict on the retrained selector**. A serve refusal is the format
  receipt (the engine names the tensor).
- **The runbook** (the user drives; every step fail-fast): production offline → the
  head: `--mode names` (re-confirmed EXACT) → `--mode dry` (the log comes back for
  inspection) → on the go, `--mode full` (both hosts, ~1 h) → the `az2` window (the
  format verdict + the battery).

## (round 62) The dry-run's traceback fixed host-agnostically: the ext's JIT needs PATH, and the head needed transformers (2026-10-28)

- **The dry-run's failure** was the same class the worker hit over ssh: the exllamav3
  ext JIT-compiles at import and torch runs ninja via subprocess — running the script
  through the venv's python does NOT put the venv's bin (ninja) or the CUDA toolkit's
  bin (nvcc) on PATH. The head had both tools installed; they were simply unreachable.
- **The fix is in the script, by construction**: `fix_path()` runs at `main()` before
  any import — the venv's bin joins PATH ahead (ninja), the CUDA toolkit's bin joins if
  it holds nvcc, and a missing ninja dies with the exact install line. Both hosts then
  verified with the literal import chain: **HEAD IMPORT OK, WORKER IMPORT OK** (the
  head additionally needed `transformers` — installed; the worker already had it from
  the round-61 provisioning).
- The user re-runs `--mode dry`; every subsequent mode inherits the fixed PATH.

## (round 63) The dry-run's second traceback: my meta-buffer bug — the assign semantics fix (2026-10-28)

- **The failure moved forward** (the user's offline run): the conv1d slice worked, the
  PATH fix held, and the dequant itself failed at `mod.to(device)` —
  *"Cannot copy out of meta tensor"*. **My bug**: `Exl3HfLinear` registers its buffers
  on the META device (no storage), and my `copy_`-into-meta + `.to()` construction
  can't work — the class is built for the HF loader's **assign** semantics.
- **The fix** (`dequant_linear`): `load_state_dict({trellis, suh, svh, mcg}, strict=True,
  assign=True)` — the meta buffers are REPLACED by the real CUDA tensors (the mcg meta
  shape declared as (1,) to match the reshaped marker), `.to()` removed, `finalize()`
  then builds the inner LinearEXL3 on real data.
- **The CPU-side verification** (no GPU, production untouched): the construction chain
  (meta → assign → real buffers → finalize) runs against layer 0's real qkv_proj
  tensors — the only GPU remainder is `ext.reconstruct` itself (exllamav3's own path,
  its symbol verified on both hosts).

## (round 64) The CPU mechanism check PASSED; the "stall" was the sink, not the process (2026-10-28)

- **The assign-check result** (the round-63 verification, run against layer 0's real
  qkv_proj tensors on the head): the full construction chain verified — the meta
  buffers replaced by real tensors via `load_state_dict(assign=True)` (trellis I16
  [256,1536,64], suh F16 [4096], svh F16 [24576], mcg I32 reshaped [] → [1]), all
  `is_meta == False` after the assign, and `finalize()` built the inner LinearEXL3
  with the mcg magic **-877912083** intact. The mechanism that broke the round-63
  dry-run is gone on real data; the GPU dequant (`ext.reconstruct`) is the only
  untested remainder, and its symbol is verified on both hosts.
- **The stall that wasn't**: the check looked hung (empty output for minutes, 2.4%
  CPU) — the output pipe ended in `tail`, which emits nothing until EOF, so emptiness
  was guaranteed; and the process was I/O-bound (first-load page-ins of torch + the
  ext .so under production's just-restarted memory pressure), which reads as near-zero
  CPU. It completed on its own, correctly. Second sink artifact this session (round
  62's `tail` hid a compile error) — captured as the `diagnose-hung-commands` skill
  (installed 2026-10-28, ~/.config/opencode/skills/).
- **Practical note for the re-run**: the first dequant call pays the exllamav3 import
  (~1–2 min silent on the loaded head after the conv1d line) — a pause, not a hang;
  the script's own log lines are flushed live thereafter.

## (round 65) The dry-run's third failure was the last shape bug: the dequant transpose; the full shape contract verified (2026-10-28)

- **The failure** (the user's offline dry-run, production down): the PATH fix, the
  conv1d slice, all 3,456 expert-quad copies, the 35 plain copies and every |max|
  (0.10–0.41, indexer.wq_b 1.73 — sane) PASSED; the dequant itself worked and the
  shape gate caught the final bug: all 15 dequanted linears came out **(in, out)** —
  exllamav3's kernel convention (`get_weight_tensor` builds `(in_features,
  out_features)`) — where the loader reads the HF/nn.Linear **(out, in)** (down_proj
  [12288, 4096] vs the reference [4096, 12288], exactly swapped).
- **The fix**: the transpose lives in `dequant_linear` (`return w.t().contiguous()`),
  and the fused qkv split became a ROW split (`w[pi*third:(pi+1)*third]`) — the fused
  row order q|k|v is the verified round-60 order, preserved under the transpose.
- **The full shape contract verified CPU-side** (a header-only replay of the mapping
  against the reference — no torch, no GPU, production untouched): **all 150,226
  produced tensors match the reference exactly** — 148,608 expert quads (the [] → [1]
  mcg reshape confirmed against the reference's own shape), 34 conv1d → 102 slices,
  267 quad groups → 335 weights (the arithmetic pins exactly 34 fused qkv groups =
  the 34 delta-net layers, each +2), 1,181 plain. The check itself needed two fixes
  (a with-block scoping bug; a q vs q_proj naming bug in the checker — the repack's
  names-mode was already EXACT and unaffected).
- **What remains for the dry-run to prove**: only the CUDA path of the assign+
  finalize+reconstruct chain (the CPU check covered CPU; the CUDA assign is untested
  since the round-63 rewrite) and the GPU dequant values (finite + |max| per tensor).
  The full run's merge re-checks every shape again.

## (round 66) The full run's first attempt failed on a broken check, not a broken worker (2026-10-28)

- **The failure** (the user's offline `--mode full`, 435 s in): staging to the worker
  completed (~7 min, whole-file rsync), then the worker's venv probe died —
  "lacks torch+cuda/exllamav3/safetensors". **The worker was healthy**; the probe was
  broken. It ran as a bare ssh `python -c "import …"` WITHOUT the run's PATH: the ext
  import's rebuild check found no ninja (the round-62 trap in a new spot — any probe
  of the run environment must run through the run's own PATH setup), and the same
  command with `~/.venv-exl3/bin:/usr/local/cuda/bin` joined ahead passed ("bare chain
  OK" verified on the worker live).
- **The fix** (`80a0945`): a `--mode check` gate — `fix_path()` + the full import chain
  (torch+cuda, `ext.reconstruct`, `Exl3HfLinear`, safetensors, transformers) + the
  disk floor (~85 GiB for a host's output half) — and `full()` now **scps the script
  to the worker first and gates it through `--mode check`**, so the probe IS the run
  environment by construction. A stale dead line (`-c 'import torch'`) in the same
  block was removed.
- **The both-hosts recheck ran through the real path: CHECK OK on both** — the head
  (1960 GiB free) and the worker (1989 GiB free), identical stacks (torch
  2.14.1+cu130, GB10, the full dequant import chain). Note: the head's first ext load
  sat ~158 s silently (cold page cache under production's memory pressure) while the
  worker's took 2.8 s — the diagnose-hung-commands pattern again; the full run's
  check absorbs the wait.
- **The second failure of the attempt** (337 s in, after the staging and a green
  worker check): the worker's half died on
  `FileNotFoundError: /home/giles/az-az-exl3-mcg/model.safetensors.index.json` — the
  script's hardcoded head-side source path. The worker's staged copy lives at
  `/home/giles/az-repack-src` and work mode never pointed at it; right behind it was
  a second latent failure (work mode also reads the reference inventory from the
  head's served-Focus path). **Fixes**: work mode takes `--src` (full() passes the
  staged copy), and the reference check degrades to a warning in work mode — the
  merge's verification binds regardless.
- **The worker requirement checklist, verified**: the venv + the full import chain
  (round 66's CHECK OK through the script's own gate); CUDA visible (GB10); 1,989 GiB
  free; the staged source complete and byte-matched (154G, 23 shards + index); the
  script scp'd by full() before the check; the reference snapshot complete (58 files,
  same as the head — the conv1d belt-and-suspenders check will actually run there);
  the output dir auto-created; head→worker ssh proven by the staging itself. The one
  requirement that stays the user's: the GPUs free — production offline on both hosts.
- **The third failure** (524 s in, at the head's in-process half): `work() missing 1
  required positional argument 'src'` — the round-66b `--src` threading missed the
  SECOND call site inside full(). **The down-the-line audit it triggered found a
  guaranteed merge failure**: a leftover pre-round-60 in-place rename in merge()'s
  partial-map loop renamed shards to `of-00023` BEFORE the merged/ move globbed for
  `of-NNNNN` — the move would find nothing and the shape check would die on a missing
  output shard. The merge path had never executed and could never have succeeded.
  **Fixes** (`2aaa591`): the call site, the leftover rename removed, the conv1d
  blocks `.clone()` instead of `.contiguous()` (a dim-0 slice is contiguous, so
  contiguous() returned a storage-sharing view; the worker's 12 conv1d tensors
  empirically survived save_file, but the clone kills the class), and full() cleans
  stale work-*/merged dirs (idempotent re-runs).
- **The synthetic merge test** (fixtures on the head, CPU-only, no GPU): the REAL
  merge() run against a fake source/reference/partials — on the pre-fix script it
  reproduced the exact predicted "missing output shard" failure; on the fixed script
  it PASSES with independent post-verification (the merged dir re-read: 9 tensors,
  shapes, the old-shape config, the provenance). The never-run merge path is now
  proven before the next GPU window.
- **What the failed attempt still proved**: the worker's half completed end to end
  (~3 min for its 11 shards — the whole remote pipeline works: staging, the
  script-run check, work with --src, the partial maps). The re-run is fully
  idempotent: it re-stages, re-checks, and redoes both halves (~15–20 min total —
  the "~1 h" estimate was conservative).
- **The triple-check battery** (the user's ask; independent methods, not re-runs):
  (1) DATA INTEGRITY via header byte-range sweeps — the source (23 shards /
  150,891 tensors), the reference (47 / 150,226), the worker's staged copy
  (23 / 150,891) and the worker's completed output (11 / 70,021): zero truncation,
  index cross-checks exact where an index exists. (2) SCRIPT IDENTITY: the head's
  checkout sha256 == the audited local file; py_compile + module import OK; the
  --mode names regression still EXACT. (3) ORCHESTRATION: the six remote command
  constructions (2 rsync, 1 scp, 3 ssh) all previously exercised live except the
  fetch, which is the staging rsync's sibling.
- **The staging question answered**: the 154G stage already on the worker is
  complete and identical (mtimes preserved across the rail) — the full() pre-stage
  `rm -rf` was removed (e4c6a25): rsync's own size+mtime quick check now skips an
  up-to-date stage to a zero-transfer pass, while a partial one still completes
  (-W + --delete; the round-31 delta trap is handled by -W, not by wiping). A
  re-run's staging drops from ~6 min to seconds; the next run's expected total is
  ~15 min (the worker's half is redone — 3 min — keeping the run self-verifying).

## (round 67) The repack COMPLETED — 8 min 46 s end to end; the beat-94 blocker is a checkpoint on disk (2026-10-28)

- **The receipt** (/tmp/az-repack-full.log on the head): preflight → staging SKIPPED
  by the quick check in 0.7 s (the round-66d change paying off) → the worker's
  script-run check → its 11 shards in 74 s → our 12 shards in 275 s → the fetch
  (~166 s, ~80 G at ~480 MB/s) → the merge: **name-set + shape equality vs the served
  Focus: PASS (150,226 tensors)**. Final stats: 148,608 expert quads copied,
  102 conv1d slices, **335 dequanted** (34 fused qkv × 3 + 233 singles — exactly the
  predicted arithmetic), 1,181 plain. The independent integrity sweep on merged/:
  23 shards / 150,226 tensors, zero truncation, index exact. **164 G — byte-for-byte
  the same size as the served Focus reference, as the same-tensor-set-same-dtypes
  contract predicts.**
- **Why 9 minutes where the requant took ~20 hours** — the user's question, answered:
  the ~20-hour job was the ORIGINAL requant (exllamav3's converter encoding the base
  to 4-bit trellis) — quantization is an OPTIMIZATION SEARCH (choosing the best
  trellis codes per weight tile to minimize reconstruction error), compute-bound per
  tile. This repack performs **no quantization at all**: the experts (~99% of the
  bytes) copy byte-identical, the 335 linears DECODE (trellis → lookup → Hadamard →
  scale — a single memory-bound pass, ~0.2 s each), the rest is disk I/O. Decode is
  the inverse's cheap half; the 20 h was the encoder's search. The double-
  quantization caveat stands: the 335 linears get re-packed by the ENGINE at serve
  load (converter 4-bit → bf16 → the load-time pack) — that is the az2 serve's job.
- **State**: ~/az-repack-out/merged is a complete old-layout checkpoint with the
  old-shape quantization_config (bits 4, codebook mcg, head_bits 16, scope
  glm53_routed_experts_only) and repack_provenance.json. Production is back up (the
  user's serve). **Next: the az2 window** — placement into the cache, the serve =
  the format verdict, then §2 gates + hardmode at --parallel 2 = the beat-94 verdict.

## (round 68) The az2 window ran: the serve refusal diagnosed — the requant needs the reference-shape patch (2026-10-28)

- **Production is verified serving** (the user's "not serving" was the load window:
  the container is Up minutes before the API binds; the probe answers).
- **The az2 window ran and the serve REFUSED** — the exact round-26 signature:
  "the checkpoint's 163.6 GiB of weights do not fit the 102.0 GiB budget" = the
  whole-checkpoint billing = the strict config.parse rejected the config and the
  silent catch billed everything to one rank (the wrapper itself worked perfectly:
  rc=0, the stack stopped, the port freed for production).
- **The diagnosis chain**: the config diff (requant vs the served Focus, nested
  text_config, 58 keys each) is EXACTLY ONE field — `num_experts_per_tok` 4 vs 8;
  the quantization_configs are identical (our merge wrote the Focus's shape). The
  default PATCHES is EMPTY (config.sh: pristine = the Focus's serving shape, topk-8
  by design) — so the az2 serve ran PRISTINE against a top-4 config → NotGlm53Flash.
  **The requant is a top-4 model** (azampatti's retrained selector; the Focus is
  top-8 by design) and the matrix's `azexl3` cell already defined its serve shape:
  `PATCHES=glm-topk4b.patch` — the patch changes the family's topk default 8→4 AND
  the EXL3 kernels' top_k specialization ("the reference shape the kernels
  specialize for; the variant checkpoint routes top-4").
- **The fix**: PATCHES=glm-topk4b.patch wired into the az2 phase's restart AND
  run_gates (the receipt's 26-az2.md refusal stands as the format receipt of the
  pristine attempt; FORCE=1 overwrites it with the patched attempt).
- **The look-ahead sweep**: the merged dir carries the full tokenizer family
  (tokenizer.json, tokenizer_config.json, generation_config.json, chat_template.jinja
  — no serve-side gap); the placement already happened and skips cleanly; with a
  valid parse the weightShare arithmetic returns to the round-26 receipt's ~81.8 GiB
  share < the ~102 GiB budget; the wrapper's wrap-up is proven (rc=0, port freed);
  the re-run needs FORCE=1 (phase_done bypasses .done markers). Note: the Focus's 92
  was a pristine-topk8 serve — the AZ cell is a patched-topk4 serve (the matrix's
  design); the beat-94 bar is production's 94 either way.

## (round 69) The second az2 attempt: RankMismatch — the worker's stale placement; the both-host placement fixed (2026-10-28)

- **The refusal was NEW and different**: "REFUSE rank settings: rank 1's config
  differs from rank 0's (config.json) → RankMismatch". The engine serves as two
  ranks and each reads ITS OWN copy of the checkpoint: the az2 phase placed the
  repacked checkpoint into the HEAD's cache only, and the worker's cache held a
  STALE pre-repack placement whose quantization_config still carried the converter's
  1.6.4 shape (verified live: worker `version: 1.6.4, calibration…` vs head
  `bits 4, codebook mcg, scope…`). prepare's model copy to the worker is guarded by
  "skipped when its cache already has it" — the stale copy satisfied it, so the
  repacked checkpoint never reached the worker.
- **The look-ahead that closed the next trap too**: prepare.sh `docker save | docker
  load`s the freshly-built image to the worker on EVERY prepare and verifies it with
  `--version` — so the glm-topk4b patch reaches BOTH ranks automatically (the
  fingerprint changes with PATCHES → the image rebuilds → the worker gets it). The
  only un-provisioned piece was the checkpoint copy.
- **The fix** (the az2 phase): after the head placement, the phase compares the
  config.json digests across the two hosts and, on any difference, resets the
  worker's cache entry and rsyncs the repacked checkpoint over (whole-file, ~164 G,
  some minutes), then re-verifies the digest — idempotent (a matching pair skips the
  sync on re-runs). The re-run command is unchanged (FORCE=1 still required).

## (round 70) The third az2 attempt: UnexpectedTensor — the dtype contract; the repack fixes the 388 and gains dtype gates (2026-10-28)

- **The third refusal was the engine's actual tensor loader** — progress: the rank
  handshake PASSED (the round-69 sync worked; the patched parse accepted the config).
  The error was bare ("cannot load … with its own glue kernels (UnexpectedTensor)",
  no tensor named). **A dtype sweep against the served Focus found it**: 388 tensors
  differ in dtype with shapes identical — 274 unquantized weights (the requant left
  them unquantized; the byte-copy kept the source model's F16) + 114 state tensors
  (F32) where the engine's proven-good layout is uniformly BF16. The shape contract
  never checked dtypes — the one axis the loader cares about. The 335 dequanted
  weights, the expert quads and the conv slices are ABSENT from the diff: their
  dtypes were already correct.
- **The repack fix**: ref_inventory carries dtypes; the plain pass-through CONVERTS
  to the reference's dtype (F16→BF16, F32→BF16 — the reference is the contract, the
  engine serves it today); the dry-run AND the merge now gate dtype equality too
  (a future dtype drift dies at the gate, not at a serve). The stats carry
  dtype_fixed (expect 388).
- **The az2 phase's placement currency** moved from config.json's digest to
  SHARD 1's digest (a dtype fix does not touch config.json — a config-only guard
  would re-serve the stale placement after the re-repack). Three-way: merged vs the
  head's placed copy vs the worker's; each side refreshes only on a difference.
- **The re-run sequence adds a verification gate between the repack and the window**:
  the user runs --mode full, the agent verifies the dtype contract + integrity on the
  new merged/ BEFORE the window launches — no fourth blind window.

## (round 70b) The treble check: one real bug caught in the round-70 fix; the diagnosis corroborated externally (2026-10-28)

- **On the code** (re-read + research): the round-70 dry-run dtype gate used a PARTIAL
  torch→safetensors map — torch "int16"/"int32" stayed unmapped against the headers'
  "I16"/"I32", which would have FALSE-DIED the dry-run on the first expert quad (the
  dry layers contain 3,456 int quads). Fixed with the full vocabulary (exllamav3's own
  torch<->ST map as the canonical reference); the mapping sanity-checked for every
  dtype class the dry layers produce. The plain-path conversion now dies NAMED on an
  unknown reference dtype instead of KeyError. Everything else verified as-written.
- **On the opinions** (external corroboration): (1) the requant's F16 unquantized
  weights are the CONVERTER's documented behavior — exllamav3's LinearFP16 stores
  unquantized weights as half regardless of the source dtype (source: exllamav3
  modules/quant/fp16.py, loader/safetensors.py) — not our error, and the conversion to
  the reference's BF16 is exactly the right correction; (2) the stock GLM-5.3-Flash
  routes **8 of 288 experts** (vLLM recipes, the HF internals dataset, the
  architecture notes — all agreeing), so the Focus's topk=8 is stock and the requant's
  topk=4 is the retrained selector's genuine design deviation (the efficiency play:
  fewer active experts) — the glm-topk4b patch serves exactly that; (3) the public
  architecture split (34 KDA + 11 MLA layers, the first 3 dense, the MTP draft) matches
  our tensor inventory class-for-class (34 fused qkv + 34 conv1d; layer 3's MLA indexer
  tensors in the dry) — the requant is the same architecture, the reference is a valid
  structural contract; (4) the F32→BF16 state-tensor conversions match the engine's
  storage contract (the reference stores BF16 and serves) — HF's fp32 router-bias
  convention is a training-side detail, not the engine's storage format.
- **On the research**: the safetensors dtype vocabulary (F16/BF16/F32/I16/I32/I64/…)
  confirmed from the spec and multiple implementations; the exllamav3 converter's
  dtype behavior confirmed from its source; the GLM-5.3-Flash routing width confirmed
  from three independent public sources.

## (round 70c) The treble check's third catch: the dtype conversion counter was an overcount (2026-10-28)

- **The bug**: the plain-path conversion condition compared `t.dtype` (a torch dtype) to `rd`
  (the reference's safetensors dtype STRING — "BF16", "F16", "F32"), which is NEVER
  equal → the conversion fired for EVERY plain tensor, inflating dtype_fixed to 712
  on the worker's half (all of its plain tensors) when the real mismatch count is
  388 (per the dtype sweep on the pre-fix merged dir). The conversion OUTPUT was
  still correct (`.to(rd_torch)` produces the right dtype), so the merge would have
  passed — but the log lied about how much work the conversion did.
- **The fix**: compare torch.dtype to torch.dtype (`rd_torch = {BF16: torch.bfloat16,
  F16: torch.float16, F32: torch.float32}[rd]`); only count when an actual
  conversion happens. The unknown-dtype die is preserved.
- **The verification gate still holds**: the merge's dtype check (ST-string to
  ST-string) and the dry-run's dtype gate (torch-to-ST map) are both correct; the
  re-run's stats will now report the honest count (≈194 on the worker's half,
  ~388 total).

## (round 71) The re-repack COMPLETED with the dtype fix — the verification gate is green (2026-10-28)

- **The receipt** (/tmp/az-repack-full3.log): the detached run finished at 401.2 s
  (the launch-suspension saga ended by running detached). Stats: 148,608 expert quads,
  102 conv slices, 335 dequanted, 1,181 plain, **dtype_fixed: 388** — the honest
  count, split 217 on the head + 171 on the worker, matching the round-70 dtype
  sweep's 388 exactly (the counter bug is confirmed gone). The merge: name-set +
  shape equality vs the served Focus PASS on all 150,226; the merge's dtype gate
  passed (no "dtype mismatch" die).
- **The verification gate** (the agent's own independent checks, run before the
  window): the dtype+shape contract reports **0 diffs** ("merged is byte-class-
  identical to the reference"); the integrity sweep reports **zero truncation, index
  exact** (23 shards / 150,226 tensors); the merged dir is 164 G — byte-for-byte
  the reference's size. Every structural axis (names, shapes, dtypes, byte integrity)
  is verified before the serve.
- **State**: ~/az-repack-out/merged is the fixed checkpoint; production is back up
  (the user restarted it). **Next: the az2 window** — the head's placed copy and
  the worker's are both stale (the shard-1 digests differ from the new merged dir),
  so the phase refreshes both (~6 min sync), serves with the top-4 patch (the format
  verdict), then gates + hardmode (the beat-94 verdict).

## (round 72) The az2 window: the serve LOADED — the format rescue is complete; the beat-94 verdict is NO at 89 (2026-10-29)

- **The format verdict: GREEN.** The repacked checkpoint served under the top-4
  reference-shape patch: the serve line reads "model 83.32 GiB; 4 streams at once,
  3.61 GiB each at a 524288-token window (fp8 caches), of 14.7 GiB left" — both
  placements (the head refreshed, the worker synced via the shard-1 digest) matched
  and the rank handshake passed. The §2 gates all PASS on content (g1 determinism,
  g2 drafted=plain, g3 resumed=fresh, g4 concurrent=solo; g5 note: seeded samples
  content-identical). The three-refusal arc (pristine parse → RankMismatch →
  UnexpectedTensor) is closed with a served checkpoint.
- **The beat-94 verdict: NO — 89** on hardmode at --parallel 2 (the (user) bench
  shape), vs production's 94 and the Focus's 92 (receipt: results/campaign-20261028-
  234233/az2/az2-hm.json, final_score 89 over 104 scenarios).
- **The scenario-level read** (az2-hm.log vs the round-37 Focus bench log, 104
  common scenarios): az2 passed 90, the Focus 94 — az2 LOST 3 (TC-63, TC-85, TC-98)
  and GAINED 2 (TC-38, TC-101). A near-tie with the Focus: the selector performs at
  Focus-adjacent level; the losses are concentrated, not diffuse — consistent with a
  routing-quality trade (top-4 vs top-8) plus the double-quantized linears, not a
  broken checkpoint.
- **The confound, recorded honestly**: the 335 dequanted linears passed through two
  4-bit quantizations (the requant's converter, then the engine's load-time pack).
  The azampatti source was deleted in the round-58 cleanup, so routing-vs-quantization
  cannot be de-confounded without a re-download + re-convert. 89 is a LOWER BOUND on
  the selector's quality; the ceiling (the Focus's 92) still does not beat
  production's 94.
- **The bookkeeping**: the beat-94 ledger item CLOSED with the receipt
  (docs/TOOL_EVAL_TESTING.md); the az2 phase RETIRED from scripts/run-qualification.sh
  (session-specific, verdict delivered — the receipt lives in results/); the standing
  config is untouched (dense4fp8 + INDEX_CUDA + vision remains the qualified default).
  The hmtrials phase is the next session-specific retirement candidate (its spread
  question is now answered) — left for the user's next sitting.

## (round 74) The recommendations land: the effort-tail template built + wired; the base audit resolved; the router fix located (2026-10-29)

- **Lever 1 — the effort-tail template is BUILT AND WIRED**: the bug confirmed in our
  own templates (both the Focus checkpoint's chat_template.jinja and our vision
  template emit `<|system|>Reasoning Effort: …` at line 3 — the prompt top). The
  variant (image/share/glm5-next-effort-tail.jinja): the top emission replaced by a
  provenance comment, the effort line now emitted just before the final assistant
  turn, ONLY when the request ends with a user message (grebneke's recipe — avoids
  5x over-reasoning on routine tool-result turns). Wired end to end: the Dockerfile
  stages it next to the vision template; config.sh gains CHAT_TEMPLATE (empty =
  the checkpoint's default); start.sh adds the rank-agreed --chat-template flag with
  a VISION=1 conflict guard (the two would collide; merging the fix into the vision
  template is the follow-up). The guard's logic sandbox-verified on all four cases —
  after catching a bash-precedence slip that would have killed every serve.
  **Quality targets: TC-51 and TC-92 fail deterministically on our line on BOTH
  checkpoints (verified in our receipts) — technigma is the only stack clean on both;
  bernisse pre-warned exactly this. The A/B window: prepare rebuild →
  CHAT_TEMPLATE=effort-tail VISION=0 → the battery + the TC checks.**
- **Lever 2 — the base audit resolved**: our pin 812fb040 IS upstream Release 1.1.8
  (03d08f1e). The KV-pages code is ALREADY in our base (TF_KV_PAGES in
  glm/cuda_native.zig + cuda_pages_test.zig) — usable with bf16 KV only; the
  fp8xpages combination is the open upstream item (bernisse's caveat and our round-59
  finding agree independently). The #1218 slot-leak / #1168 fp8-attention / #1176
  int8-matmul fix-ups ship in tags as old as v1.0.0-beta.10 — a targeted code check
  against our tree remains to confirm each (time-boxed here). Net: the "wait for
  TensorFold" answer shifts — our base already carries most of the watch list; the
  open upstream item is pagesxfp8, and upstream main has diverged from our line.
- **Lever 3 — the router fix is DESIGNED AND SCOPED, not implemented**: PERF_PLAN
  Front 2 Lever A2 (the 910-1125 ms router section per 2048-row chunk, 55-69% of
  chunk GPU time, for a ~4.8 GFLOP GEMM production does in ~1 ms; the ks hypothesis
  rejected with a clean A/B; the 8-slice bit-stable replacement design scoped).
  bernisse's field assessment: the highest-leverage kernel work in the field. This is
  the next work block: implement the Zig patch, then a window to build + measure.

## (round 75) The effort-tail A/B is STAGED: the template proven by a render gate, the window phase written (2026-10-29)

- **The render gate PASSED pre-window** (jinja semantics, on the head, CPU-only):
  four placement cases verified — user-ending+max (the hint exactly once, immediately
  before the final assistant turn, NOT at the top, the rest of the render identical),
  tool-result-ending (NO hint — the user-only rule), undefined effort (Max, tail —
  the stock's line-2 coercion), None (coerced to Max, tail — stock-consistent). The
  harness needed jinja2.ext.loopcontrols (the stock template uses break) — the same
  extension serving stacks enable. Two test-expectation corrections along the way
  (the stock's None coercion; the single-line top render) — the gate now tests the
  STOCK's actual semantics, not assumed ones.
- **The efforttail phase written** (session-specific, retires with the verdict): an
  explicit prepare first (the Dockerfile stages the template; a Dockerfile change does
  not move the fingerprint — the round-27 lesson), then TWO LEGS in the same window
  (bernisse's rule): baseline (the checkpoint's own template) vs CHAT_TEMPLATE=
  effort-tail, each VISION=0 on the Focus at the standing quant env, each running the
  hardmode battery with final_score + TC-51/TC-92 extracted into 27-efforttail.md.
  The flip-list comparison happens at digest (the agent reads both logs). Wired into
  NEEDS_GPU; the summary's [0-9]*.md glob catches 27- (the round-17 gotcha is fixed
  upstream of this receipt name).
- **The window runbook** (the user drives, detached): production down →
  PHASES="preflight efforttail summary" FORCE=1 setsid nohup scripts/run-window-block.sh
  </dev/null >/tmp/et.console 2>&1 & → the wrapper: prepare (rebuild+staging, ~5-10 min)
  → leg baseline (~35 min) → leg tail (~35 min) → summary → the port frees →
  production back. ~80-90 min total. I read 27-efforttail.md + both logs for the
  verdict: the score delta, the TC-51/TC-92 statuses, and the flip list.

## (round 76) The effort-tail A/B verdict: +1 score, TC-92 FIXED — the template-family diagnosis confirmed (2026-10-29)

- **The receipt** (results/campaign-20261029-103523/27-efforttail.md): leg baseline
  (the checkpoint's own template) **92** with TC-51 FAIL + TC-92 FAIL; leg tail
  (effort-tail) **93** with TC-51 FAIL + **TC-92 PASS**. Same-session legs, both
  served cleanly (83.50 GiB, 4 streams at the 524288 window — the serve line identical
  across legs; the only variable was the template).
- **The flip list** (104 common scenarios): baseline passed 94, tail passed 94 — the
  tail LOST 1 (TC-100) and GAINED 3 (TC-38, TC-101, **TC-92 tenant isolation**). The
  score moved +1 because the gained scenarios carry more points. **TC-92 flipping to
  pass confirms the template-family diagnosis**: the tenant-isolation failure was the
  effort-hint placement, not the model, not a kernel — bernisse pre-warned exactly
  this scenario and technigma's stack (the template-fix family) is clean on it.
- **The baseline reproduced**: 92 today vs the round-37 Focus 92 — the standing
  receipt is stable across weeks (the default template unchanged).
- **The remaining gap**: TC-51 still fails on both legs — that is the SECOND template
  gap (reasoning collapse after tool results; technigma's TOOL_REASONING_VALIDATION.md
  fix, not yet applied on our line). The vision template still carries the top-hint
  bug (merging the effort-tail fix into it is the follow-up). One small loss (TC-100)
  — unexamined, single scenario.
- **The adoption question** (the user's call): CHAT_TEMPLATE=effort-tail as the
  standing default — the evidence is +1 score, TC-92 fixed, 1 minor loss, zero
  serve-side cost (a template file, no kernel change).

## (round 77) The effort-tail ADOPTED — with vision intact: the fix merged into the vision template (2026-10-29)

- **The adoption** ((user): "adopt, but we want vision still"): CHAT_TEMPLATE defaults
  to effort-tail in config.sh with the round-76 receipts cited; the revert is
  CHAT_TEMPLATE= (the checkpoint's own top-hint template).
- **The vision interplay resolved by merging, not by a flag fight**: the vision
  template received the same two edits (the top emission → a provenance comment; the
  tail emission, user-message-ending only — the vision template differs from the
  stock in exactly one region, the media branch, so the fix is mechanical). The render
  gate re-run EXTENDED: the vision template passes the text placement cases AND an
  image-bearing case (the media tokens render, the hint sits once at the tail). The
  serve matrix verified in a sandbox: VISION=1 → the vision template (+fix, no extra
  flag); VISION=0 + effort-tail → the text template; VISION=0 + empty → the
  checkpoint's own template (the revert path).
- **The validation window pending**: the adopted default changes the vision serve's
  prompt — the vision suite (assets/vision) + a bench re-verification belong in the
  next window before the adoption is fully receipted on the vision path.
- Workflow note: the vision-template render test initially ran against the head's
  STALE checkout (the patch was local-only) — the gate failed correctly and passed
  after the file synced. The gate catches staleness too.

## (round 78) The validation window staged: the adopted default receipts on the vision path (2026-10-29)

- **The etadopt phase written** (session-specific, retires with the receipt): an
  explicit prepare (the vision template's CONTENT changed — content does not move the
  fingerprint, the round-27 lesson) → the standing-default serve (VISION=1 +
  CHAT_TEMPLATE=effort-tail flow from config.sh; the vision template carries the fix —
  the serve line names the template) → the vision suite (assets/vision via the
  round-55 runner, now repoed at scripts/vision-suite-probe.py — promoted out of /tmp
  for reproducibility) → ONE hardmode run re-verifying the score on the vision serve.
  The receipt (28-etadopt.md) extracts the suite result + final_score + TC-51/TC-92.
- **The comparisons the window delivers**: the vision suite vs its round-55 baseline
  (4/4 targeted + 23/25 detail) and the hardmode score vs the round-76 receipts (93
  tail text-only / 92 baseline) — the adopted default's first vision-path datapoint.
  TC-51/TC-92 tracked (the round-76 tail state: TC-51 fail, TC-92 pass).
- Wiring note: the phase anchors before the summary block (the doubled separator left
  by the round-75 insertion made the old anchor ambiguous — the insertion now anchors
  on `if want summary; then`); NEEDS_GPU updated.

## (round 79) The adoption is FULLY RECEIPTED: the vision path holds — 93 on the vision serve, the suite 8/8 (2026-10-29)

- **The receipt** (results/campaign-20261029-120042/28-etadopt.md): the standing
  default served (VISION=1 + the vision template with the effort-tail fix — the serve
  line names the template; 84.75 GiB with the tower, 4 streams at the 524288 window,
  12.6 GiB left — the standing vision config's line). The vision suite: **8/8 ok**
  (the four ground truths hit: red / golden gate / zebra / stop; the targeted answers
  clean — quadrants "Red", sign "STOP"). The hardmode: **93** — matching the round-76
  tail receipt exactly; TC-92 pass ✓, TC-51 fail (expected — its fix is the next A/B).
- **The manifest note**: the current manifest is the trimmed 4-item version (4 images
  × targeted+detail = 8 checks), so the round-55 detail counts (23/25) do not
  cross-compare — the round-55 receipt predates the manifest trim. The window's
  comparison is the current suite against its ground truths: 8/8.
- **The adoption is fully receipted on both paths**: text 93 (round 76) + vision 93
  with the suite 8/8 (this window). The standing default is the effort-tail template
  family with vision intact. The next levers, in order: TC-51's tool-result reminder
  A/B (the second template gap, still failing on both legs), then the router patch
  (PERF_PLAN Front 2 A2's 8-slice bit-stable design — the implementation work block).

## (round 80) The TC-51 lever staged: the tool-continuation reminder variant + the etreminder A/B (2026-10-29)

- **The mechanism, from technigma's 23c7b03 diff** (fetched in full): with the last
  message a tool result, the template injects a short synthetic user turn —
  "Continue the original task using the latest tool result. Before your next answer
  or tool call, briefly assess the result in the thinking block. Keep that assessment
  separate from the user-facing answer." — immediately before the assistant prefix.
  Their before/after replay: ZERO reasoning tokens at 38k/500k/900k with the original
  template (raw completions began with </think>) → 240/212/194 with the reminder +
  correct plans. Honest limits recorded: their fresh-data reminder-off controls also
  passed, so the A/B measures it on OUR bench (our TC-51 fails deterministically on
  both checkpoints — the reproduced-family failure our line actually exhibits).
- **The variant** (image/share/glm5-next-effort-tail-tr.jinja): the adopted
  effort-tail template + the reminder block, mirrored to our template's structure (no
  thinking_enabled var — our generation prompt always opens <think>; the opt-out
  kwarg tool_reasoning_reminder=false honored). The two fixes are MUTUALLY EXCLUSIVE
  by condition (tool-ending vs user-ending) — they compose without interaction. The
  render gate extended to nine checks: the reminder fires exactly once on
  tool-ending, silent on user-ending, the opt-out honored, the effort hint intact on
  user-ending. ALL PASS.
- **The etreminder phase written** (session-specific): explicit prepare → two
  same-session legs (effort-tail vs effort-tail-tr, the adopted default as the
  control) on the Focus at VISION=0, hardmode + TC-51/TC-92 per leg, the flip
  analysis at digest. Wired into NEEDS_GPU.

## (round 81) The etreminder A/B verdict: REJECTED — the reminder is net-negative on our bench (2026-10-29)

- **The receipt** (results/campaign-20261029-125309/29-etreminder.md): leg tail (the
  adopted control) **93**, TC-51 fail, TC-92 pass — reproducing the round-76/79
  receipts exactly. Leg tail-tr (the tool-continuation reminder) **90** — a 3-point
  regression. **TC-51 unchanged (fail)** — the reminder did NOT fix it — and **TC-92
  REGRESSED to fail** (the effort-tail gain was lost).
- **The flip list** (104 scenarios): tail-tr passed 92 vs the control's 94 — LOST 4
  (TC-74, TC-85, TC-92, TC-98), GAINED 1 (TC-100). Note: TC-85 and TC-98 also moved
  in the az2-vs-focus comparison — marginal scenarios sensitive to prompt structure.
- **The interpretation**: technigma's reminder mitigates THEIR reproduced failure
  (zero reasoning on tool continuations at very long context); on OUR bench the
  synthetic user turn is net-negative — it perturbs prompt structure the scenarios
  depend on. The reminder is a mitigation for a specific reproduced failure mode, not
  a general quality lever. The variant is NOT adopted; the file stays as a receipted
  dead end (CHAT_TEMPLATE only serves it if explicitly selected).
- **TC-51 remains open** — the documented candidate fix is rejected, so its cause
  needs diagnosis: read the scenario definition from the bench repo and reproduce the
  actual-vs-expected locally. The quality queue after that: the router patch (PERF_PLAN
  Front 2 A2 — the big perf lever), the settings A/B.
- The A/B discipline worked as designed: a harmful change was caught before adoption.

## (round 82) TC-51 diagnosed and CLOSED as a shared failure; the real gap decomposed; the (user) no-overfitting principle (2026-10-29)

- **The TC-51 mechanism** (from the bench source `planning/tc51.py` + our run's
  transcript, results/campaign-20261029-125309/etreminder/et-tail.json): the model's
  trace is semantically complete — contacts → event with all 3 engineers (right date,
  time, title) → email — with coherent reasoning at every turn. It hard-fails on
  ORDERING: it batched `send_email` in the SAME assistant turn as
  `create_calendar_event`, and the bench's dependency rule
  (`create_calendar_event → send_email`) requires the notification in a strictly
  LATER turn; the premature-notification check zeroes the whole trace (fail, 0
  points — not even partial). Had it sent NO email, it would have PASSED via the
  event-invite path (the eval accepts the event's attendees as the notification).
- **The decisive cross-check — production fails TC-51 too, identically**: the bench
  repo's own production rebaseline receipt (prod-score-rebaseline-20261027,
  hardmode.json) shows TC-51 fail/0, parallel_tool_turns [2], and the SAME summary
  string ("Batched send_email with create_calendar_event in the same turn instead of
  waiting for the create_calendar_event result"). TC-51 is a benchmark-policy trap
  (strict serialization of declared dependency pairs vs the model's correct judgment
  that the two calls are independent) — BOTH stacks fall into it. It was never a gap
  between our line and production. The reminder A/B's null result is fully explained:
  reasoning was present every turn; the failure is turn-ordering policy, not the
  reasoning-collapse family the reminder targeted.
- **The real 93-vs-94 gap, per-scenario vs production's rebaseline** (104 common):
  we pass 94 / prod 96. We WIN TC-92 (+2 — production FAILS tenant isolation; our
  effort-tail fix is the clean side), TC-40, TC-90 (+1 each). We LOSE TC-80 (−2,
  mutated before reading/deciding), TC-100 (−2, deferred a `retryable:true`
  single-account retry to the user — good non-idempotency safety judgment; the bench
  scores autonomy), TC-62 (−1, skipped chain steps before the email), TC-63 (−1, an
  extra unrelated call), TC-68 (−1, schema-compliant JSON wrapped in Markdown fences).
- **The reading**: two-directional agency-calibration variance (over-eager on TC-80/
  62/63, under-eager on TC-100) + one formatting habit (TC-68) — not a systematic
  capability gap. The only lever that moves these is targeted prompt surgery, which
  is exactly what the reminder experiment showed to be net-negative (TC-92 regressed).
- **(user) principle, recorded**: improvements must make the model genuinely better —
  NOT targeted at specific TCs — unless a change is a real general improvement. Under
  that principle: no template surgery for the calibration scenarios (it would be
  benchmark-overfitting); the effort-tail adoption stands on its general merits
  (long-context reasoning engagement, receipted 92→93 with a real behavioral fix);
  the remaining levers are the GENERAL ones — the router patch (PERF_PLAN Front 2 A2,
  every request benefits) and the stack-level settings A/B.

## (round 83) Upstream v1.1.9 reviewed: three bit-exact GLM items, no router work, no fp8×pages — hold-and-fold-in (2026-10-29)

- **The audit**: v1.1.8...v1.1.9 = 50 commits / 256 files; the changelog matches the
  diff. Three GLM-5.3-Flash CUDA items, all bit-exact, all by @Bizuayeu: #1404 the
  TP≥2 DSA indexer split (1–2% prefill at 128k on 2×GB10 — our topology),
  #1424 the DSA select decode widening (+1.8% at TP=3/136k, flat at TP=2/65k),
  #1352 TENSORFOLD_PREFILL_SHARE (decode-under-prefill fairness, opt-in).
- **Not included**: no router-GEMM work (Front 2 A2 remains ours — the Kolibri
  #1460 item is the same CLASS of routed-expert L2 pathology but different kernels),
  no fp8×KV-pages (the ledger watch stays open), no GLM vision/template changes.
- **(agent recommendation, pending user)**: hold the pin; base the router A2 patch on
  v1.1.9 so one re-qualification window adopts both. Recorded in ECOSYSTEM.md.

## (round 84) Front 2 A2 IMPLEMENTED: the staged chain router patch, compiled clean on v1.1.9 (2026-10-29)

- **The design de-risked in source first** (the v1.1.9 tree read end to end): the
  hand-rolled router kernels accumulate each output as ONE continuous mma.m16n8k16 chain
  per 512-wide K slice (K ascending from zero); glm_mm's kernels accumulate per-64-group
  (fresh `d` per group, `acc += d`). fp32 non-associativity means calling stock glm_mm
  per slice would flip low bits — the content gate (#1079 protocol) would kill it. The
  PERF_PLAN's "preserving the summation structure via sliceSum" was necessary but not
  sufficient; the fix is a dedicated **chain kernel**: glm_mm's exact prompt shape
  (128-row blocks, ROWS_FIRST L2 sharing, 3-stage cp.async, PDL, 256 threads) with the
  mma chained directly into one accumulator — same fragments, same order, same bits by
  construction, staged loads instead of the per-step global reads that are the pathology.
- **The patch** (patches/glm-router-gemm.patch, diffed against v1.1.9, 8 files +209/−3):
  `tf_glm_router_mm` (the chain kernel, glue image, Set-auto-loaded);
  `glm_mm_tile16s` (strided tile16 — each MoE gate tiles into 8 contiguous K-slice
  layouts at engine init, a same-bytes re-layout, one buffer owned by the mm kernels);
  `glue.router` gains the `tiles` dispatch (m ≥ 128, aligned, tiled — decode rows and
  unaligned callers keep the hand-rolled kernels); the MoE layer carries `rtiles`; the
  engine init builds the tiles under TF_GLM_ROUTER_MM=1 (OFF by default — the patch is
  inert without the env, so one image serves both A/B legs); contract records added
  (copied from the shape sibling until the contract run prints actuals).
- **Compile receipt**: fatbins + native build RC=0 in the builder container
  (nvcc sm_121, zig 0.17.0, v1.1.9) — three fix iterations (bpairs is glm_mm-local —
  inlined the trivial bf16 copy; the `tiles` param; DeviceBuffer.ptr is a plain u64).
- **The routergemm phase written** (session-specific): ONE prepare (the patch overlay),
  two same-session legs — stock (patched binary, env off = the executed hand-rolled
  path: v1.1.9's true baseline) vs gemm (TF_GLM_ROUTER_MM=1) — duel solo + c4 + chunk
  prof per leg, hardmode on the gemm leg, the content gate stock-vs-gemm in the receipt.
  The pin moves 812fb040 → 465e48b6 for this window (the patch diffs the v1.1.9 tree;
  1.1.9's GLM items are bit-exact upstream). patches/README carries the full design +
  the rebase-ownership note.

## (round 85) The routergemm A/B verdict: bit-safe PROVEN, perf prize NOT there — the pathology is implementation-independent (2026-10-29)

- **The receipt** (results/campaign-20261029-152500/30-routergemm.md): **content gate
  IDENTICAL** — the staged chain router reproduces the hand-rolled kernels' replies
  byte-for-byte; the round-84 bit-stability design is PROVEN on hardware. Hardmode **93**
  on the gemm leg (TC-92 pass, TC-51 fail as everywhere) — quality untouched. The pin's
  v1.1.9 base is receipted (leg stock: solo 54.1, c4 140.7, slope 0.659 — the new base
  line; bit-exact from 1.1.8 as upstream claimed).
- **The perf read: ZERO movement.** Chunk router section 1109 (stock) vs 1114 ms (chain)
  at 2048 rows; slope 0.659 vs 0.660; solo/c4 within noise. The chain kernel carries
  glm_mm's ENTIRE staging machinery and costs the same as the direct-load hand-rolled
  kernels. The m=64 chunk's router is 54.9 ms — far from FLOP-proportional.
- **The conclusion**: three kernel probes (ks 8→2, part-forced, the staged chain GEMM)
  agree — the ~1100 ms router section is NOT the kernel's implementation. The staging
  hypothesis is dead. Remaining suspects: the prof-span semantics (what the span really
  brackets), host-enqueue bubbles landing inside the spans, or a systemic engine
  property (clocks / stream serialization). **Next instrument: an ncu capture of one
  chunk's router kernels** (true kernel duration, SM busy %, stall reasons) — a small
  diagnostic window, no A/B needed.
- **Decisions**: the patch is NOT adopted (no measured win — the rebase-tax rule;
  patches/README updated with the round-85 status). The v1.1.9 pin STAYS (bit-exact
  upstream, the round-83 fold-in verdict; leg stock is the new base line). The bit-safe
  chain kernel stays in the repo as the probe artifact — if the profiler ever shows a
  real kernel-shaped cost, the fix rides on it.

## (round 86) The next window staged: the ncu capture + the settings A/B (2026-10-29)

- **ncurouter written** (session-specific, fail-honest): the serve container (patched
  image, TF_GLM_ROUTER_MM=1) is committed; ncu + its nsight-compute dir are copied in
  from the base image (the head HAS host ncu at /usr/local/cuda/bin/ncu; the serve image
  does NOT — the base does); a throwaway container runs the engine under ncu filtered to
  tf_glm_router_mm + tf_glm_slice_sum, one ~2k-token prompt fires one 2048-row chunk, the
  CSV lands in the receipt. **The read**: mean per-launch duration × 8 slices × 80 layers
  vs the section's 1109 ms — fast kernels = the section is bubbles (host enqueue / span
  semantics); slow kernels = the SASS metrics say why. An instrument refusal is recorded,
  not retried.
- **settingsab already wired** (the round-24 phase): one hardmode run at temperature 1.0 /
  top_p 0.95 against the greedy receipts — production was also measured greedy, so a lift
  is ours without touching the comparison's fairness. Sampled runs are non-deterministic —
  one run is a draw, not a receipt.
- One prepare is NOT needed (the patched image from campaign-20261029-152500 is current);
  both phases restart the serve themselves.

## (round 87) The settings A/B is a WASH (item closed); the ncu capture died on phase plumbing (2026-10-29)

- **settingsab** (results/campaign-20261029-165850/17-settingsab.md): hardmode **92** at
  temperature 1.0 / top_p 0.95 vs the greedy 92 anchor — no lift. The recommended-sampling
  lever is a documented no-op on our bench (production was also measured greedy, so nothing
  is forfeited). Item CLOSED; the greedy 92/93 stays the deterministic anchor; config.sh
  unchanged.
- **ncurouter**: the phase died right after the serve came up (rc=1 at 17:50:27, before any
  capture step logged) — an instrument bug in my base-image ncu-extraction plumbing under
  the campaign's strict mode, NOT an engine result. The fix pass: drop the fragile
  docker-create/export dance and tar the nsight-compute dir from a `docker run --rm` pipe
  straight into the serve container (the one path whose mechanics are proven), guarded
  end to end. Cheap to rerun — no prepare needed.

## (round 88) The community perf sweep: the MiaAI-Lab fork's patch table — +36% prefill exists in the field (2026-10-29)

- **The find**: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold — our exact model
  on our exact topology, 1,940–1,980 tok/s prefill at short context with a measured patch
  stack (we run ~1,517 tok/s at slope 0.659). Full table + ranking in ECOSYSTEM.md.
- **The levers, ranked**: (1) the HC-split/overlap set — +36% prefill at 50k, decode −1.5%,
  TP=2-native, and the same fix class our bubbles hypothesis predicts; (2) 0086
  prompt-scores-loop (BadAd84) — bit-exact, scoring −36%, cold prefill at 226k −8.5%;
  (3) the KDA chunked set — prefill −7–10%. Upstream main has moved past v1.1.9
  (36470cec) — scan for these graduating before porting.
- **Next**: the ncu rerun (the plumbing fix) + the upstream scan; the 0086 port is the
  first port candidate if upstream lacks it.

## (round 90) The autonomous window system: the agent-owned stop/restart, the cron safety net (2026-10-29)

- **(user)**: the agent may stop/restart production for a staged window — delegated to
  the script, not to the live session ("you stop production you will stop yourself").
- **scripts/autonomous-window.sh**: stop production → the window phases → restart
  production on EVERY exit path (trapped) — success, failure, interrupt. A flag file
  (/tmp/autonomous-testing.flag) marks an in-flight window for the cron; a second
  launch refuses while one is active. Multi-hour/multi-day safe: nothing depends on a
  live agent session.
- **scripts/autonomous-cron.sh** (cron every 30 min on the head): production down with
  no flag → restart it (the crash net; production must never stay down); a newer
  campaign receipt than the marker → write ~/.opencode-poke — the marker file the
  agent reads FIRST on any wake. The opencode CLI poke wires when its headless
  invocation lands; until then the user wakes the agent and it reads the poke file.
- The round-89 stale-chain incident's lesson is baked in: the orchestration lives in
  the repo script, never in an inline command chain.

## (round 92) The autonomous loop rebuilt on the real topology: the durable-inbox poke, the tip-owned production lifecycle (2026-10-30)

- **(user) restated the topology**: production on the head is the -tip checkout, started and
  stopped with ITS OWN start.sh/stop.sh (both ranks, rank logs saved, API-wait + smoke on
  start) — the raw docker stop/start the round-90/91 scripts used only touched the head rank;
  the agent harness runs on the agent host as a web server (systemd, basic auth); the testing
  repo lands on both hosts. Site values (the head's ssh target, the tip dir name, the server
  URL, the unit path) now come from git-ignored scripts/local.sh on each host — nothing
  site-specific committed.
- **The wake vector is PROVEN on hardware**: POST /api/session/{id}/prompt — the harness
  server's durable prompt inbox ("durably admit one session input and schedule agent-loop
  execution"). A throwaway session was created, prompted, answered in <10 s, deleted (204).
  The poke finds the session by project directory (newest first) and marks the campaign dir
  in a marker file, so each window pokes exactly once; a failed POST retries next tick.
- **scripts/autonomous-poke.sh** (agent-host cron, every 5 min): production up = container
  running AND /health answering (the port read from the tip's own config); then the durable
  prompt. **scripts/autonomous-window.sh** rewritten: production lifecycle through the tip's
  stop.sh/start.sh (the user-delegated stop), restore on every exit path via the trap, the
  flag file guards double launches, 30 s grace before the stop so the agent's closing message
  flushes. **scripts/autonomous-cron.sh** (head cron, every 5 min): the crash net only —
  production down with no flag → the tip's start.sh (flock-guarded); the poke half moved to
  the agent host.
- **The ncu generator's real bug found** (the round-90/91 windows never profiled): the
  run-capture.sh template was an f-string carrying a literal JSON body — Python ate the
  braces (ValueError: Invalid format specifier). Two latent bugs beside it: the capture
  script was copied into the serve container AFTER the commit (the capture container never
  had it), and the blind sleep 300 raced the ~4-min model load. The template is
  placeholder-based now, self-waits for the serve, self-queries the served model id, signals
  done; the serve container is stopped before the capture (port + GPUs free; the capture
  container is rank 0, the worker's rank 1 may not accept the fresh peer — fail-honest); the
  phase polls for the done marker up to 25 min. Generation PROVEN locally on an inspect.json
  fixture (rc=0, binds filtered, bash parses, no unresolved placeholders).
- **Next window**: preflight ncurouter summary through the rebuilt loop — the first full
  autonomous cycle: stop → phases → restart → poke. The poke marker is seeded with the
  current newest campaign (this round's diagnosis) so only the new window wakes the agent.

## (round 92) The autonomous loop rebuilt on the real topology: the durable-inbox wake, the production checkout's own lifecycle (2026-10-30)

- **(user) restated the topology**: production on the head is the -tip checkout, started and
  stopped with ITS OWN start.sh/stop.sh (both ranks, rank logs saved, API-wait + smoke test)
  — the raw docker stop/start the round-90/91 scripts used only touched the head rank, which
  is why the early attempts "did not work". The agent harness runs on the agent host (a
  systemd web server, basic auth). Site values (the head's ssh target, the tip dir name, the
  server URL, the unit path) now come from git-ignored scripts/local.sh on each host —
  nothing site-specific is committed (SSH_TARGET, PROD_TIP, OPENCODE_API, OPENCODE_UNIT).
- **The wake vector is PROVEN on hardware**: POST /api/session/{id}/prompt — the harness
  server's durable prompt inbox ("durably admit one session input and schedule agent-loop
  execution"). A throwaway session was created, prompted, answered in <10 s, deleted (204).
  The poke finds the agent's session by project directory (newest first) and marks the
  campaign dir in a marker file, so each window pokes exactly once; a failed POST retries
  next tick.
- **scripts/autonomous-poke.sh** (agent-host cron, every 5 min): production up = container
  running AND /health answering (the port read from the tip's own config), then the durable
  prompt. **scripts/autonomous-window.sh** rewritten: production lifecycle through the tip's
  stop.sh/start.sh (the user-delegated stop), restore on every exit path via the trap, the
  flag file guards double launches, 30 s grace before the stop so the agent's closing message
  flushes. **scripts/autonomous-cron.sh** (head cron, every 5 min): crash net only —
  production down with no flag → the tip's start.sh (flock-guarded); the poke half moved to
  the agent host.
- **(user) on redaction**: methodology may be generalized in committed text, but site
  specifics stay in git-ignored files — enforced on the new scripts (no hostnames, no
  absolute local paths; values only from local.sh with fail-honest `:?` guards).
- **The ncu generator's real bug found** (the round-90/91 windows never profiled): the
  run-capture.sh template was an f-string carrying a literal JSON body — Python ate the
  braces (ValueError: Invalid format specifier). Two latent bugs beside it: the capture
  script was copied into the serve container AFTER the commit (the capture container never
  had it), and the blind sleep 300 raced the ~4-min model load. The template is
  placeholder-based now, self-waits for the serve, self-queries the served model id, signals
  done; the serve container is stopped before the capture (port + GPUs free; the capture
  container is rank 0 and the worker's rank 1 may not accept the fresh peer — fail-honest);
  the phase polls for the done marker up to 25 min. Generation PROVEN locally on an
  inspect.json fixture (rc=0, binds filtered, bash parses, no unresolved placeholders).
  A bash gotcha cost a pass: an apostrophe inside a `${VAR:?message}` expansion is parsed as
  a quote — the guard's syntax check caught all three scripts.
- **Next window**: preflight ncurouter summary through the rebuilt loop — the first full
  autonomous cycle: stop → phases → restart → poke. The poke marker is seeded with the
  current newest campaign (this round's diagnosis) so only the new window wakes the agent.

## (round 93) The first COMPLETE autonomous cycle — and the capture's structural post-mortem (2026-10-30)

- **(user)**: standing autonomy granted — "rinse and repeat": research online, check
  ashhart/TensorFold for updates, PRs and forks, the NVIDIA forums, keep the repo current,
  continue the development; site specifics stay git-ignored (enforced: SSH_TARGET, PROD_TIP,
  OPENCODE_API, OPENCODE_UNIT only in scripts/local.sh per clone).
- **The cycle worked end to end**: launch → production stopped (both ranks, via the tip's
  stop.sh) → preflight + ncurouter + summary ran → production restored (start.sh) → the
  cron poke woke the agent. The wake loop is real.
- **The capture failed on three structural facts, now receipted** (results/campaign-
  20261030-011413/31-ncurouter.md + the console): (a) the capture container inherited NO
  binds — the model cache is a BIND mount the swap cannot carry (the engine died instantly:
  model path "neither a directory nor a HF repo id"); (b) the WORKER rank exits cleanly the
  moment the head rank's container stops — a throwaway rank 0 has no TP peer, ever; (c) the
  agent's double-launch race (two ssh attempts) stripped the flag mid-window, the head's
  crash-net cron then started production while the window ran, and the poke fired
  mid-window about an unfinished campaign.
- **Fixes, all committed and fixture-proven (3b22d74)**: the capture is now a WRAPPED
  RESTART — RANK0_WRAP/RUN_IMAGE hooks in the testing start.sh (rank 0's engine under ncu
  INSIDE the stack's normal lifecycle; rank 1 comes up normally and the pair forms);
  duration-metric-only collection (minimal per-launch overhead); WAIT_TIMEOUT bounds the
  boot; the reader reports the duration DISTRIBUTION (boot/warmup/smoke/prompt mix
  shapes). The DRY_RUN fixture on the head shows rank 0 wrapped + the committed image,
  rank 1 clean. scripts/autonomous-launch.sh is now the ONLY launcher (sync + flag-refuse
  + detached start); the window's refusal path no longer touches production or writes a
  completion file (STARTED gate); the poke hands off while a window flag exists.

## (round 94) The research pass: upstream's row-split pair seam, the fork's KV/spill patches, the forum's looping pathology (2026-10-30)

- **Upstream (465e48b6..cefb72fd, no new tag)**: the recent work is Nemotron CUDA (#548)
  and Qwen3.8-27B pair-seam kernels (#1432) — including a NEW CORE primitive,
  "the row-split pair seam, split simd_qmm over K" (e06a6eef): row-splitting between the
  ranks as a core mechanism — the native shape of what the fork's HC-split set hand-rolls.
  No GLM performance items graduated; our v1.1.9 pin stands until something GLM-relevant
  lands.
- **The fork (MiaAI-Lab)**: newest 2026-10-25 — the published v0.6.0 image pinned with 123
  patches; new since the round-88 sweep: patch 0128 (spill tier in blocks shared by later
  turns) and 0129 (KV sharding — each conversation's latent cache once on its owner Spark),
  both serving-memory items, not prefill. The HC-split/overlap set (round 88's +36%) and
  0086 (bit-exact scoring) remain the port candidates; 0086 stays first (smallest diff,
  bit-exact).
- **The forums (thread through Oct 13)**: a DETERMINISTIC high-context looping pathology on
  the TensorFold engine — long-form document processing loops on the same paragraph, same
  payload fails identically every time (the reporter: 7 restarts in hours on TF vs one
  loop in 22+ h on the EXL3/vLLM engine); a standing watch item — our greedy determinism
  would reproduce it identically if a bench scenario hit it. tool-eval-bench gained
  official TensorFold support upstream (PR #190, in main) and the community's hardmode
  protocol (thinking+max effort, temp 0.1, trials 5) converges on ours. New recipe
  pattern spotted: jordanm82's "dealign o_proj transplant (layers 15–44), latent FP8" —
  a quality-preserving transplant, untested by us. Adoption is broad (k8s helm charts,
  metric TUIs, vLLM backports of TF's speedups).
- **Next**: the ncu window relaunch through the hardened loop — the wrap design's first
  live test; the read splits the router section into kernel time vs bubbles.

## (round 95) The wrapped restart's first live test: one bug, found by its own fail-honest branch, fixed (2026-10-30)

- The wrap-design window (campaign-20261030-014654) failed in 5 s with a clean receipt:
  rank 1's docker run on the worker could not find `rg-ncu-img` — the committed instrument
  image exists on the head alone and the worker tried to PULL it. The loop mechanics were
  perfect: the fail-honest branch recorded the error, the campaign wrapped, and the
  window's restore had production serving again in 97 s.
- **Fix**: rank 1 never needed the instrument image — its engine bits are identical. The
  RUN_IMAGE override now applies to rank 0's image only; rank 1 always runs the prepared
  ref (no 2 GB docker save/load over ssh). DRY fixture on the head: rank 0 = rg-ncu-img +
  the ncu wrap, rank 1 = the prepared image, unwrapped.
- Next window relaunches the same phases — the wrap design's second live test.

## (round 96) The wrap becomes the container entrypoint (2026-10-30)

- The second live test failed in seconds with another exact receipt: the engine printed
  its USAGE — the image's ENTRYPOINT is the engine binary itself, so prepending the
  wrapper to the command gave the engine argv `ncu --kernel-name ...` (garbage to it).
- **Fix**: RANK0_WRAP now overrides `--entrypoint` with the wrapper binary; the image's
  engine path (inspected from the prepared image) and the serve args follow as the
  wrapper's arguments. DRY fixture: `--entrypoint /usr/local/bin/ncu ... rg-ncu-img
  --kernel-name ... --log-file /tmp/rg-ncu.csv /opt/tensorfold/bin/tensorfold-native
  serve ...` — ncu execs the engine as its child, exactly the profiling shape.
- Next window relaunches the same phases.

## (round 97) ERR_NVGPUCTRPERM: the last instrument gate — SYS_ADMIN satisfies it (2026-10-30)

- The third live test's engine booted under ncu and answered a REAL prompt (1053-byte
  reply, try 1) — the wrap-as-entrypoint design works — but the CSV held only ncu's
  banner: `ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
  Performance Counters`. The driver loads with `RmProfilingAdminOnly: 1`.
- Hardware test (throwaway container, tiny matmul, production untouched): with
  `--cap-add SYS_ADMIN` ncu connected, profiled, and produced duration rows
  (`gpu__time_duration.sum`, ns). No driver module reload needed.
- **Fix**: the wrapped rank-0 command now carries `--cap-add SYS_ADMIN` (only when
  RANK0_WRAP is set). DRY fixture confirms cap + entrypoint together.
- Two side lessons from the test: the committed serve image has no python3 (its CUDA
  program test needed the base image), and NGC's python lives at /opt/conda but this
  box resolves /usr/bin/python3.
- Next window: the full capture — boot + smoke + one 2048-row prompt under the counter-
  enabled profiler; the receipt should finally carry the duration distribution.

## (round 98) FRONT 2 A2 RESOLVED: the router section is ~97% bubbles — device-event spans starve between µs-scale kernels (2026-10-30)

- **The verdict data** (campaign-20261030-023707/31-ncurouter.md, completed offline after the
  reader's banner-skip fix): under the counter-permitted wrapped restart, 512 kernel rows —
  336 x tf_glm_router_mm (p50 36.2 µs, p95 45.6 µs, grid (11,5,1) constant) + 176 x
  tf_glm_slice_sum (p50 10.5 µs, p95 65.5 µs; grid (1,1,1) x121 — launch-latency stubs).
  Kernel time per 2048-row chunk: **p50 29.9 ms, p95 71.1 ms** against the section's
  ~1109 ms — **kernels 2.7% at p50 (6.4% at p95); the other ~1080 ms is gaps**.
- **The span semantics settled from source**: zig/src/families/glm/cuda_prompt_trace.zig —
  "paired same-stream events" — the engine's stage ms= values are CUDA EVENTS (device
  time). The 1109 ms was device time: the stream idles between µs-scale kernels — host
  enqueue / sync starvation, NOT kernel cost. Hypothesis four's answer: the section needs
  OVERLAP (fewer, bigger launches; exchange overlapped with compute), the exact fix class
  of the fork's HC-split/overlap set (+36% prefill) and upstream's new core row-split pair
  seam (e06a6eef).
- The ncu instrument is complete and receipted: wrap-as-entrypoint + SYS_ADMIN (the
  RmProfilingAdminOnly=1 gate) + rank 1 on the prepared image + duration-metric-only +
  bounded wait. Reader fixed (ncu banners precede the CSV header — skip to the "ID" line).
- **Recon for the next window**: 0086-glm-prompt-scores-loop is a 131-line, ONE-file
  (tensorfold/families/glm5_next/cuda/sparse.py — python, no fatbin rebuild), bit-exact
  patch (the fork's receipt: scoring -36%, cold prefill at 226k -8.5%); the fork is cloned
  on the head (/tmp/tf-fork) with 0010/0033/0086 present. Next: verify 0086 applies to the
  v1.1.9 pin, port it, and A/B it (prefill instrument + content parity) in a window; the
  HC-split set stages behind it as the confirmed-class big prize.

## (round 99) The port plan re-scoped: the fork's stack is the PYTHON engine — our line is native (2026-10-30)

- **0086 is not portable**: the patch targets `tensorfold/families/glm5_next/cuda/sparse.py`
  — a Triton kernel in the PYTHON engine — and our pin has NO glm5_next python files at all
  (the native CUDA line's scoring is tf_glm_* kernels). The fork's WHOLE stack is python-side
  (0010: comm.py/decode.py; 0033: forward.py/hcsplit.py; 0064: engine.py) — none of it ports
  to the native line. The round-88 "port 0086 first" plan is withdrawn; the fork's NUMBERS
  (1,940–1,980 tok/s prefill on 2×Sparks) remain the field reference for the model's
  potential, now understood as a DIFFERENT engine.
- **Upstream's row-split pair seam (e06a6eef) is native-core** — four kernels
  (pair_stage/pair_gather/pair_argmax/pair_pick) exchanging whole output rows instead of K
  halves — but its steps 2–6 are Qwen-specific; GLM coverage is future. WATCH, not port.
- **The fix path for our bubbles verdict is native-side**, guided by a clean stage map: the
  new **profchunk** phase (no ncu — the round-98 capture proved the paired-event spans
  inflate ~7× under a profiler) restarts the probe config with TF_GLM_PROF=1, fires one
  ~2k-token prompt, and aggregates the LAST chunk's spans per (site,stage) in device time —
  naming which stage owns the chunk and how much of the wall is inter-stage gaps.
- Next window: preflight profchunk summary — the stage-resolved bubbles map.

## (round 100) Front 2 A2 finally resolved: the "1109 ms router section" was a span-under-concurrency artifact (2026-10-30)

- **The clean stage map** (campaign-20261030-030326/32-profchunk.md — the profchunk phase,
  no ncu): a lone 1322-row chunk = wall 1283 ms; ffn/ffn (the expert GEMMs) 636 ms — the
  real compute; ffn/router **8.6 ms total (~0.2 ms/layer)**; wait_filled spans ~1019 ms
  overlapping across streams (spans sum to 143% of the wall — they are concurrent-stream
  pairs, per cuda_prompt_trace.zig's design).
- **The round-85 number's source, found**: `glm prof prompt chunk to 2048 rows (4 calls):
  wall 1693.44 ms, gpu 1968.41 ms a call; ... router 1109.46 ...` — measured under 4-way
  concurrent load (gpu > wall proves overlap). Paired-event spans absorb device-busy time
  when other streams share the GPU: the router stage — the one with MANY µs-scale launches
  — was the canary absorbing the enqueue/queue contention. The three round-85 kernel probes
  "agreed" because they all measured the same artifact. The m=64-chunk 54.9 ms reading: the
  same mechanism.
- **What is solid**: the router kernels are µs-scale (ncu, round 98); the router STAGE is
  negligible in clean spans (round 99's map); the expert GEMMs own ~half the lone-chunk
  wall; and the qualified perf deficit is the c4 aggregate (99.8 vs production's 154.5 t/s,
  65%). The native line's bottleneck hypothesis: **launch/enqueue efficiency under
  concurrency** — which the fork's python engine dodges with bigger fused Triton kernels.
- **The artifact becomes the instrument**: the new profload phase runs the same serve under
  TF_GLM_PROF=1 through (1) a lone 14k prefill and (2) four concurrent 2k prefills, and
  reads the engine's own `glm prof prompt chunk` aggregates side by side — where the
  concurrency cost lands, stage-attributed.
- Front 2 A2 CLOSED (the mystery was the measurement, not the kernels); the perf front
  refocuses on the concurrency deficit. No config change; no chart change (no new
  qualified perf fact yet — profload's read may produce one).
