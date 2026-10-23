#!/usr/bin/env bash
# The documentation guard — runs after every change (the committed pre-commit hook
# executes it; activate with: git config core.hooksPath scripts/hooks).
#
# Checks, each failing with its own fix line:
#   1. bash -n on every scripts/*.sh
#   2. site identifiers in tracked files: real hostnames / ssh targets / the site's
#      rail IPs anywhere; platform-generic device names outside local.sh.example
#   3. the perf chart matches docs/perf-anchors.tsv (render-perf-chart.sh --check)
#   4. the README's PHASES example only names phases the campaign script knows, and
#      its default phase list is complete (checked against the script's PHASES default)
#   5. config.sh's defaults (CONTEXT, PARALLEL, KV_CACHE, DENSE_BITS/HEAD_BITS) appear
#      in the README's configuration table
#   6. on the head (where results/ exists): every campaign dir cited in the docs exists,
#      and the newest campaign dir is mentioned in LAUNCH.md or the ledger
#   7. every local file link/img src in README, AGENTS.md and docs/*.md resolves
#      (relative to the containing file, as GitHub resolves them)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

FAILS=0
fail() { printf 'GUARD FAIL: %s\n' "$*" >&2; FAILS=$((FAILS + 1)); }
pass() { printf 'GUARD PASS: %s\n' "$*"; }

# --- 1. syntax -------------------------------------------------------------------
_bad=""
for f in scripts/*.sh; do bash -n "$f" 2>/dev/null || _bad+=" $f"; done
[[ -z $_bad ]] && pass "scripts/*.sh syntax" || fail "syntax errors in:$_bad — run bash -n scripts/<file>"

# --- 2. site identifiers ---------------------------------------------------------
# (the guard's own file is excluded: it must carry the patterns to detect them)
_site=$(git grep -nE 'brain0[23]|giles@|192\.168\.2\.' -- . ':!scripts/docs-guard.sh' 2>/dev/null | head -5 || true)
[[ -z $_site ]] || fail "site identifiers committed (hostnames/ssh/IPs) — see AGENTS.md:$_site"
_devs=$(git grep -lE 'enp1s0f1np1|enP2p1s0f1np1|rocep1s0f1|roceP2p1s0f1' -- . ':!scripts/local.sh.example' ':!scripts/docs-guard.sh' 2>/dev/null | head -5 || true)
[[ -z $_devs ]] || fail "device names outside local.sh.example: $_devs — genericize (AGENTS.md rule)"

# --- 3. the chart ----------------------------------------------------------------
if ./scripts/render-perf-chart.sh --check >/dev/null 2>&1; then
  pass "perf chart matches the anchors"
else
  fail "perf chart stale vs docs/perf-anchors.tsv — ./scripts/render-perf-chart.sh then commit both"
fi

# --- 4. the phase lists ----------------------------------------------------------
script_default=$(grep -m1 -oP 'PHASES="\$\{PHASES:-\K[^}]+' scripts/run-qualification.sh | tr ' ' '\n' | sort -u)
[[ -n $script_default ]] || fail "cannot read the script's default PHASES list"
readme_phases=$(grep -m1 -oP 'PHASES="\K[^"]+' README.md | tr ' ' '\n' | sort -u)
[[ -n $readme_phases ]] || fail "README's quick start lost its PHASES example"
_unknown=$(comm -23 <(printf '%s\n' "$readme_phases") <(printf '%s\n' "$script_default") | tr '\n' ' ')
[[ -z $_unknown ]] && pass "README's PHASES example names known phases" \
  || fail "README's PHASES example names unknown phases: $_unknown"

# --- 5. config defaults in the README table --------------------------------------
check_default() {  # $1 config var, $2 README grep pattern, $3 what must appear
  local d
  d=$(grep -m1 -oP "^$1=\"\\\$\\{$1:-\K[^}]+" config.sh || true)
  [[ -n $d ]] || { fail "cannot read $1's default from config.sh"; return; }
  if grep -q "^$2" README.md && grep "^$2" README.md | grep -q -- "$3"; then
    pass "README's $1 row carries the default ($3)"
  else
    fail "README's $1 row does not carry the config.sh default ($3) — update the config table"
  fi
}
check_default CONTEXT   '| `CONTEXT` / `PARALLEL`'  "$(grep -m1 -oP '^CONTEXT="\$\{CONTEXT:-\K[^}]+' config.sh)"
check_default KV_CACHE  '| `KV_CACHE`'              "$(grep -m1 -oP '^KV_CACHE="\$\{KV_CACHE:-\K[^}]+' config.sh)"
check_default DENSE_BITS '| `DENSE_BITS` / `HEAD_BITS`' "$(grep -m1 -oP '^DENSE_BITS="\$\{DENSE_BITS:-\K[^}]+' config.sh)"

# --- 6. receipt existence (the head only — results/ is not in git) ---------------
if [[ -d results ]]; then
  _phantom=""
  for c in $(grep -hoE 'campaign-[0-9]{8}-[0-9]{6}' docs/*.md | sort -u); do
    [[ -d "results/$c" ]] || _phantom+=" $c"
  done
  [[ -z $_phantom ]] && pass "every cited campaign dir exists" \
    || fail "docs cite campaign dirs that do not exist:$_phantom"
  newest=$(ls -t results | head -1)
  if grep -q "$newest" docs/LAUNCH.md docs/TOOL_EVAL_TESTING.md 2>/dev/null; then
    pass "the newest campaign ($newest) is in the log or the ledger"
  else
    fail "the newest campaign ($newest) is not in LAUNCH.md or the ledger — fold it in"
  fi
fi

# --- 7. links resolve ------------------------------------------------------------
_badlinks=""
for f in README.md AGENTS.md docs/*.md; do
  d=$(dirname "$f")
  while IFS= read -r l; do
    case $l in \#*|http*|mailto*) continue ;; esac
    l=${l%%#*}; [[ -n $l ]] || continue
    [[ -e "$d/$l" ]] || _badlinks+=" $f -> $l"
  done < <(grep -oE '\]\([^)]+\)|src="[^"]+"' "$f" 2>/dev/null \
    | sed -E 's/^\]\(|^src="//; s/\)$|"$//')
done
[[ -z $_badlinks ]] && pass "all local links resolve" || fail "broken local links:$_badlinks"

# --- summary ---------------------------------------------------------------------
if [[ $FAILS -gt 0 ]]; then
  printf 'GUARD: %d failure(s)\n' "$FAILS" >&2
  exit 1
fi
printf 'GUARD: all checks pass\n'
