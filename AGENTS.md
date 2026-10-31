# AGENTS.md — how to work in this repo

> All edits in this repository are AI-generated (OpenCode agents running GLM, by Z.ai);
> the README banner carries the provenance disclaimer. This file is the working
> conventions digest — pointers to the real trackers, not a second copy of them.

## What this repo is

The EXL3-only **testing line** for GLM-5.3-Flash on a two-GB10 pair — one **head**
host (rank 0 + API) and one **worker** host (rank 1); this site's real hostnames,
addresses and devices live in git-ignored `scripts/local.sh` / `docs/SITE.md` —
served by TensorFold's **native CUDA
engine**. It is not the production deployment — production is the EXL3/Python recipe
(the `-tip` checkout on the head, container
`glm53-flash-tf`, serving opencode on 8000) and it outranks everything here.

Hard boundaries, never crossed:

- **Production's lifecycle is autonomously managed (user, round 129)**: the window
  flow and test dances stop production for GPU work and restart it on EVERY exit
  path; the crash-net restarts it whenever it is down with no window running. The
  guardrails: production always comes back; production's code/config is never
  modified beyond the intent-keyed port; `prodqual` probes it read-only. **The
  ports are intent-keyed (user)**: production started FOR OPENCODE serves 8000;
  production started FOR TESTING serves 8001 (`PORT=8001 $TIP/start.sh`); our
  testing stack always serves 8001 — opencode never connects to testing.
- This line is **EXL3 only** (the NVFP4 route has its own repo) and the context goal is
  **524,288 tokens — nothing above**.
- **Nothing is posted upstream** (no issues/PRs to ashhart/TensorFold or other repos).
  Findings for upstream are written into `docs/ECOSYSTEM.md` as data.
- Site identity (hostnames, IPs, ssh targets, rail devices, HCA names) lives only in
  git-ignored `scripts/local.sh` / `docs/SITE.md`. **Never commit it, in any file**:
  committed prose says "the head" / "the worker", never a hostname, address, netdev
  or HCA name.

## Where evidence and decisions live — use these, don't spawn alternatives

| Need | The tracker | Not this |
| --- | --- | --- |
| **The current status snapshot** | **`README.md` — current status and pointers ONLY; no history** | round references, narratives, duplicated scoreboards |
| Run receipts, raw outputs | `results/campaign-*/` (one dir per run, on the head's checkout) | ad-hoc scratch dirs or files |
| **The production-vs-native scoreboard** | **`docs/DELTAS.md`** (+ `docs/perf-anchors.tsv` for the chart's pairs) | second copies in the README or PERF_PLAN |
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
  windows, but only when asked). Long runs go under `nohup` on the head with the
  console at `/tmp/<name>.console`, not in a foreground ssh.
- A round's digest updates **the same sitting**: LAUNCH.md (append), the ledger's
  status items, any affected plan doc, and the README status table when a qualified
  fact changes. When a comparison metric changes, also update its anchors file
  (`docs/*-anchors.tsv` — the standing chart, and `docs/duel-anchors.tsv` once the
  duel phase has run) and re-run `scripts/render-perf-chart.sh <tsv>` — the chart
  SVGs are generated, never hand-edited: one bar = our performance as % of
  production, time metrics inverted, higher always better; the truth, bad or good.
  The guard and the campaign's summary phase flag a stale chart via `--check`.
- `config.sh` defaults track the **qualified optimum** (with receipts cited in
  comments) — a bare `./scripts/start.sh` serves them. Site-only values (worker,
  rails, identity) stay in `local.sh`.
- User decisions are recorded as "(user)" in the LAUNCH.md round entry, with the
  reasoning — they override agent judgment.

## Commit and sync protocol

Edit locally → commit (message names the finding, not the file) → push → on the head:
`git fetch origin && git merge --ff-only origin/main` (ff-only, never rebase shared
history). **The committed pre-commit hook runs the docs guard on every commit** —
activate it once per clone with `git config core.hooksPath scripts/hooks`; a failing
check blocks the commit until the doc it flags is fixed. Campaign receipts live on the
head's checkout under `results/` and are not committed — quote them from there.

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
