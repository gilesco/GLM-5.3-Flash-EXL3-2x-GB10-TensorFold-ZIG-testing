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
