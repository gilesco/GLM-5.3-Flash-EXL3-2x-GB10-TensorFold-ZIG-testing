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
- **This repo is the testing line** (native engine, the EXL3 checkpoint since 2026-10-18 —
  NVFP4 is the fallback route; no patches — see `../README.md`
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
~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/stop.sh
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
cd ~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold && ./start.sh
# verify: docker ps shows glm53-flash-tf on both hosts, port 8000 answers on the head
```

The testing stack shares nothing with production except the port (both 8000, by design —
only one runs at a time): different image (`glm53-native-tf:*` vs `tensorfold-glm53:*`),
container name (`glm53-native-tf` vs `glm53-flash-tf`), state/log dirs. Optional cleanup to
reclaim ~191 GB per host:
`docker rmi glm53-native-tf:<tag>` and remove
`~/.cache/huggingface/hub/models--nvidia--GLM-5.3-Flash-NVFP4` (only if you keep no NVFP4
capability — the checkpoint redownloads at ~183 GB; **never** remove the
`models--Mia-AiLab--GLM-5.3-Flash-EXL3-*` or `models--incoai--GLM-5.3-Flash-DFlash2` trees:
production serves from them).

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
