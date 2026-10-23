# AGENTS.md — how to work in this repo

> All edits in this repository are AI-generated (OpenCode agents running GLM, by Z.ai);
> the README banner carries the provenance disclaimer. This file is the working
> conventions digest — pointers to the real trackers, not a second copy of them.

## What this repo is

The EXL3-only **testing line** for GLM-5.3-Flash on a two-GB10 pair (head `brain02`
= rank 0 + API, worker `brain03` = rank 1), served by TensorFold's **native CUDA
engine**. It is not the production deployment — production is the EXL3/Python recipe
(`~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold-tip` on brain02, container
`glm53-flash-tf`, same port 8000) and it outranks everything here.

Hard boundaries, never crossed:

- **Production is never stopped, restarted, or modified by an agent** — the user owns
  its lifecycle. The campaign script guards on it: GPU phases refuse to run while
  production is up, and `prodqual` probes it read-only.
- This line is **EXL3 only** (the NVFP4 route has its own repo) and the context goal is
  **524,288 tokens — nothing above**.
- **Nothing is posted upstream** (no issues/PRs to ashhart/TensorFold or other repos).
  Findings for upstream are written into `docs/ECOSYSTEM.md` as data.
- Site identity (IPs, ssh targets, rail devices, HF cache paths) lives only in
  git-ignored `scripts/local.sh` / `docs/SITE.md`. Never commit it.

## Where evidence and decisions live — use these, don't spawn alternatives

| Need | The tracker | Not this |
| --- | --- | --- |
| Run receipts, raw outputs | `results/campaign-*/` (one dir per run, on brain02) | ad-hoc scratch dirs or files |
| Session log, decisions, incidents | `docs/LAUNCH.md` — **append-only**, one `## (round N)` header each | new log files |
| Matrix status, open items, result tables | `docs/TOOL_EVAL_TESTING.md` — the ledger | per-round summary files |
| Gate definitions + the native-line receipt | `docs/VALIDATION.md` | new gate docs |
| Production-vs-native comparison | `docs/DELTAS.md` | — |
| Upstream/community watch | `docs/ECOSYSTEM.md` | — |
| Context-ladder math and its outcome | `docs/CONTEXT_512K_PLAN.md` | — |
| This site's real values | `docs/SITE.md` (git-ignored) | — |

## Rules of the road

- **Every number in the docs quotes a receipt** — a file in `results/` or a LAUNCH.md
  entry. No number without a run behind it. Correcting a wrong number gets a round
  entry explaining the withdrawal; never edit it silently.
- **A new measurement is a phase in `scripts/run-qualification.sh`**, not a one-off
  script. The campaign is phased (`.done` markers, `PHASES=` selectable, `FORCE=1`
  re-run), production-guarded, and its receipts ARE the evidence chain. Session-specific
  phases say so in their comment and are deleted when they retire.
- **Version pins go in `scripts/local.sh`** (`TF_REPO`/`TF_COMMIT`), never the session
  env: `start.sh`'s auto-prepare re-resolves the default ref whenever the config
  fingerprint changes and silently drops a session pin (round 7's window was voided on
  exactly this; `local.sh.example` carries the example).
- **Never edit `scripts/*.sh` or `config.sh` while a campaign is running** — bash
  re-reads scripts lazily, and `config.sh` is re-sourced by every restart.
- **GPU windows are the user's unless they explicitly hand one over** (the agent used
  to BE production's served model; since the cloud-model switch the agent may run
  windows, but only when asked). Long runs go under `nohup` on brain02 with the
  console at `/tmp/<name>.console`, not in a foreground ssh.
- A round's digest updates **the same sitting**: LAUNCH.md (append), the ledger's
  status items, any affected plan doc, and the README status table when a qualified
  fact changes.
- `config.sh` defaults track the **qualified optimum** (with receipts cited in
  comments) — a bare `./scripts/start.sh` serves them. Site-only values (worker,
  rails, identity) stay in `local.sh`.
- User decisions are recorded as "(user)" in the LAUNCH.md round entry, with the
  reasoning — they override agent judgment.

## Commit and sync protocol

Edit locally → `bash -n scripts/*.sh` → commit (message names the finding, not the
file) → push → on brain02: `git fetch origin && git merge --ff-only origin/main`
(ff-only, never rebase shared history). Campaign receipts live on brain02's checkout
under `results/` and are not committed — quote them from there.

## Gotchas that have already cost us (don't relearn them)

- Bench **tg128 rows divide by TOTAL time** — prefill pollutes short generations at
  depth. Steady-state probes (long generations, e.g. max_tokens 2048) are the honest
  decode instrument.
- **Whole-envelope sha is not a determinism instrument** (envelope `id`/`created`/
  metadata vary per request on every engine). Gates bind on normalized `content_sha`.
- **`cached_tokens` in usage is unwired** (0 even on cache hits) — cache-hit
  verification is content + the `/metrics` counters (`tensorfold_cached_tokens_total`
  via goshi's exporter).
- On the KV-pages branch head (87e4417e), **fp8 KV × `TF_KV_PAGES` is a fatal
  combination** — bf16 is the only paged path.
- **Thinking eats small `max_tokens`** — gate probes use ≥1024.
- **`VAR=x func` persists in bash** — prodqual restores `SERVED_NAME` after its
  override; keep that pattern for any new override.
- The engine's admission refusal prints its own arithmetic — a refusal is a receipt,
  not a failure; record it instead of retrying blind.
