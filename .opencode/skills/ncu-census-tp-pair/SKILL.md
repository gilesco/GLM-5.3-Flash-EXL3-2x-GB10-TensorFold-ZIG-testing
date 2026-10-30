---
name: NCU Census on a TP Pair
description: Profile one rank of a two-rank docker/ssh tensor-parallel engine pair with Nsight Compute without tripping the engine's rank-settings fingerprint — wrap rank 0 only, keep both ranks on the same image, %q-quote wrapper regexes in the ssh command string. Use when an ncu census of a serving engine's kernels refuses with "rank settings differ", when a wrapped remote rank reports "command not found" on kernel-name tokens, or when a both-ranks-wrapped boot stalls with zero profiler rows.
---

# NCU Census on a TP Pair

Capturing per-kernel durations from ONE rank of a two-host tensor-parallel serving
pair (docker containers started over ssh, the engine checks that both ranks' settings
agree at startup) needs a specific topology. Four failure modes exist; each has a
distinct signature. This skill is the working topology plus the failure signature
table.

## When to use

- A wrapped restart refuses: `REFUSE rank settings: rank N's program differs from rank 0's (the executable)`.
- The remote rank's start fails with `bash: line 1: <kernel_name>: command not found`.
- Both ranks wrapped: the boot never becomes healthy, one rank dies with
  `another rank's process ended`, and the profile CSV holds zero kernel rows.
- Kernel rows are missing for kernels you know fired — and the filter came from a
  source tree you did not verify against the running engine's version string.

## The working topology

1. **Both ranks run the SAME image** — the engine's rank fingerprint hashes
   `/proc/self/exe` (the engine binary itself). Identical image ⇒ identical hash,
   whether or not one rank is wrapped. Different images (an old instrument image vs a
   freshly prepared one) ⇒ refusal, regardless of wrapping.
2. **Wrap rank 0 only.** The peer starts plain from the same image. Wrapping both
   ranks stalls the boot (zero profile rows; each rank's profiler holds its rank at
   the first profiled launch until the peer's line closes). The pair's collectives
   tolerate the wrapped rank being slow; they do not tolerate two profilers.
3. **The wrap is the container ENTRYPOINT** with the engine and its serve args
   appended (prepending the wrapper to the command instead hands the wrapper's flags
   to the engine, which prints its usage). The container needs the counter permission
   (`--cap-add SYS_ADMIN` on drivers with `RmProfilingAdminOnly=1`).
4. **Rank 1's command travels as a string through ssh — a remote shell parses it.**
   Every wrapper token that carries shell metacharacters (an ncu `--kernel-name
   regex:` with `|` alternation) must be shell-quoted — `printf '%q'` per token when
   building the string. Rank 0's command is an argv array; no quoting needed. This
   asymmetry is why a rank-0-only wrap "worked before" while the first rank-1 wrap
   exploded.
5. **Budget the launch count for the boot.** The engine's warmup runs the same
   kernels the probe will — with a 4-6 name filter the warmup consumes hundreds of
   the `--launch-count` budget before the probe fires. Size the budget
   (a few hundred) so per-launch replay overhead stays inside the health timeout;
   accept that the probe's later layers fall off the budget's end.

## Failure signature table

| signature | cause | fix |
| --- | --- | --- |
| `program differs (the executable)` | the two ranks run DIFFERENT engine binaries (stale instrument image vs fresh prepare) | rebuild/re-commit the instrument image from the current serve container; ship it to the peer host before the wrapped restart |
| `command not found` on kernel-name tokens on the remote rank | the ssh command string's regex pipes executed as shell pipes | `%q`-quote each wrapper token in the string |
| boot never healthy, zero CSV rows, peer dies first | BOTH ranks wrapped | wrap rank 0 only; peer plain on the same image |
| expected kernels absent from the CSV | filter names taken from a source tree that is not the served engine's source | read the kernel names from the tree at the SERVED version's ref (the engine prints its version/commit at boot) |

## Gotchas

- **Verify the source tree against the served engine's version string before
  tracing or designing from it.** An exported/patched working tree that predates the
  served ref will silently misattribute kernels (a stale export carried a contributed
  patch as if it were stock, so the trace named a kernel the served engine never
  launches). The engine's boot line names its version and commit — match them.
- The fingerprint compares the executable's hash, NOT the argv — ranks legitimately
  differ in rank flags and serve args. Don't try to make argvs identical.
- Decode-shaped warmup kernels that fire inside CUDA graph CAPTURE never execute at
  boot — they are invisible to the profiler (and to ncu row counts). Their absence
  from the CSV is expected, not a filter bug.
- The first profiled row often carries an inflated duration (profiler warm-up) —
  read percentiles, not min.
- The engine's own span/profiling log (if enabled) lands in the serve log and gives
  per-stage per-layer wall times alongside the per-kernel CSV — the two instruments
  cross-check each other (spans are inflated under the profiler; kernels are honest).

## Verification

- The census receipt shows: the prompt probe answered, the CSV row count equals the
  budget, every filter name's launch count is plausible against the model's layer
  count, and per-kernel p50/p95 print.
- The unwrapped peer produced no CSV (expected) and the pair formed without the
  refuse line in either rank's log.

## Gotchas

- **The decode kernels are invisible to ncu while the decode runs inside CUDA graph
  replays** (this engine's default). Set the shared graph-cache knob to off
  (`TF_GRAPH_CACHE=off` — "0 or off runs every step eagerly") so the decode runs
  eagerly and profiles; it is measurement-only (it costs the replay). Segment the
  capture by the last prefill-only kernel's row to split prefill from decode.
- A `--launch-skip` sized from arithmetic overshoots silently: a banners-only CSV
  means the skip passed every matched launch. Prefer a larger budget with no skip,
  then segment by row position.
- The python-engine knobs (this line's: `TF_GLM_MAX_ROWS`, `TF_GLM_PROMPT_GRID`,
  `TF_GLM_WIDE_GRAPHS`) do not exist in the native tree — check the engine's own
  knob set (the boot banner names its own) before hunting an env that isn't there.
- The engine's rank fingerprint includes the graph-cache settings' digest — the
  graph-cache env must be identical on both ranks (the restart env applies to both).
