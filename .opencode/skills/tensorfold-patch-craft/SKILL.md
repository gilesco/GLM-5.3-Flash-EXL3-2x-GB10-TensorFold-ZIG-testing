---
name: TensorFold Patch Craft
description: Craft, verify, and measure a local TensorFold engine patch on this testing line — writing the zig diff, the apply/symbol checks that git apply cannot do for you, the alphabetical staging order, shipping it through a resumecost window via RC_PATCHES, and reading the receipt (the boot wall, the content gate). Trigger words: engine patch, RC_PATCHES, patches/, git apply failed, duplicate struct member, the build refused the patch, boot marks, decompose the loader.
---

# TensorFold Patch Craft

One paragraph: this line measures engine changes as patches in `patches/`, applied
at image build against the pinned TensorFold tree, shipped through a resumecost
window (`RC_PATCHES="a.patch,b.patch" scripts/autonomous-launch.sh "preflight
resumecost summary"`), and judged by the campaign receipt: the restart wall, the
boot marks, and the content gate's `content_sha` (bit-safety). The state assumed:
the pin in `scripts/local.sh`, the source export at `/tmp/tf-src` (re-clone from
the pin's tag after any head reboot — /tmp is ephemeral), and a window slot.

## When to use

- A loader/serve change is hypothesized and needs a measured A/B against the pin.
- The build refused a patch (`PATCH FAILED`, `patch does not apply`) and the fix
  is not obvious.
- The build compiled but failed with a Zig symbol error (`duplicate struct member`).
- A patch window completed and the receipt needs reading (wall, marks, gate sha).

## Workflow

1. Generate the diff against the exact tree the build will apply it to: clone the
   pin (`git clone -q --depth 1 --branch <tag> https://github.com/ashhart/TensorFold /tmp/tf-src`),
   edit there, `git diff -- <file>` into `patches/<name>.patch` with a `#!`
   provenance header (what/why/port-against/owner/status).
2. Verify the FULL STACK in build order: the build applies the staged patches
   ALPHABETICALLY (`for p in /build/patches/*.patch`). A patch that assumes an
   earlier patch must sort after it — rename it (e.g. `glm-upload-marks3.patch`
   sorts after `glm-pinned-upload.patch`; `glm-boot-marks3.patch` does not).
   Verify: a scratch repo, apply each patch in alphabetical order, then check the
   symbols (below). Record the verify in the patch's README entry.
3. Symbol check (git apply is TEXTUAL — it cannot catch Zig semantics): grep the
   post-patch files for the names you introduced. Zig forbids a field and a
   declaration sharing a name — the Loader already had a `scratch()` fn when a
   patch added a `scratch` field (`duplicate struct member name 'scratch'`). Also
   grep for existing fields/fns with your candidate name BEFORE naming.
4. Compile-risk audit before spending a window: every Zig idiom copied from the
   same tree (the clock: `std.Io.Clock.awake.now(io)`; the integer-width rules —
   `i + 1 == layers.len` fails u32-vs-usize, use `@intCast`), the format-string
   placeholder counts, the imports reachable via the package root. A compile
   failure voids a ~25-minute window.
5. Ship: commit, push, sync the head (ff-only), launch
   `PHASES="preflight resumecost summary" RC_PATCHES="..." scripts/autonomous-launch.sh "preflight resumecost summary"`.
   The config fingerprint changes → the auto-prepare rebuilds (~15-25 min) → the
   boot runs the patched engine.
6. Read the receipt: `results/campaign-*/boot-full.log` (the marks, `loaded in`),
   `campaign.log` (the wall: `healthy after N s`), `34-resumecost.md` (the probes
   + the gate). Bit-safety = the probes' `sha=` matches the standing config's gate
   sha. A refusal printing its own arithmetic is a RECEIPT — record it, never
   retry blind.

## Gotchas

- **Alphabetical staging order is load-bearing** (see step 2). The voided
  campaign-20261031-000643 is the receipt.
- **git apply passes what the compiler rejects** — the symbol check is mandatory
  (campaign-20261031-233318's `duplicate struct member name 'scratch'`).
- **Zig locals shadow the enclosing struct's declarations too** — a `const limit`
  inside a struct's method collides with that struct's `limit()` fn
  (`local constant shadows declaration of 'limit'`); the pre-name grep must cover
  the LOCAL names you introduce against the whole enclosing struct, not just the
  top-level names you add. The field case (`scratch`) was caught pre-window; the
  local case cost a build (campaign-20261101-115038).
- **Do NOT try advice-based prefaulting on the GLM loader** (MADV_WILLNEED or
  MADV_POPULATE_READ per tensor): both REGRESSED the boot (+32%/+37%,
  campaigns -221626/-224013) — WILLNEED's async readahead races the fault loop
  into duplicate I/O; the population fattens the per-layer page-cache churn under
  the full unified pool. The reads' own cost is the mmap fault rate (~840 MiB/s
  measured); the fix that worked was upstream's parallel read-ahead (#1616), not advice.
- **The engine prints no boot progress on non-tty stdout** — decompose the boot
  with marks patches (`glm-boot-marks.patch`'s pattern: `std.log.info` phase
  lines, captured via `docker logs --timestamps` in the resumecost ours-branch —
  plain `docker logs` is untimestamped and the stopped container is rm'd at stop).
- **A patch edit on the head's /tmp/tf-src does not persist across reboots** —
  regenerate from the pin's tag; never treat /tmp scratch as state.
- **The nvidia-smi memory.used reads [N/A] on GB10** — sample utilization only,
  or read the room from the engine's own serve line.
- One concern per patch (patches/README rule 3): a bundled patch cannot be
  measured or dropped independently.

## Verification

- The scratch-repo stack apply passes in alphabetical order AND the symbol greps
  come back clean (each introduced name exactly once, no collisions).
- The window completes rc=0; `boot-full.log` carries the expected marks; the
  content gate's sha equals the standing config's; the wall delta matches the
  hypothesis (or the receipt explains why not — record it either way).
