#!/usr/bin/env bash
# Regenerate docs/perf-vs-production.svg from docs/perf-anchors.tsv.
#
#   ./scripts/render-perf-chart.sh           # write the SVG
#   ./scripts/render-perf-chart.sh --check   # render to a temp file and diff against
#                                            # the committed SVG: exit 1 + note on drift
#
# The anchors file carries one row per metric:
#   label|direction(lower|higher)|production|ours|raw label (receipt numbers as displayed)
# and its `#` comments carry the dated caption text. Update the TSV first, then re-run
# this script — the SVG is generated, never hand-edited. Bar widths are pct of
# production's performance (time metrics inverted), 6 px per point, green at >=100%.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

TSV=docs/perf-anchors.tsv
SVG=docs/perf-vs-production.svg

[[ -f $TSV ]] || { echo "[render-perf-chart] ERROR: $TSV missing" >&2; exit 2; }

dated=$(grep -m1 '^# dated=' "$TSV" | cut -d= -f2-)
receipts=$(grep -m1 '^# receipts=' "$TSV" | cut -d= -f2-)
[[ -n $dated && -n $receipts ]] || { echo "[render-perf-chart] ERROR: $TSV needs '# dated=' and '# receipts=' lines" >&2; exit 2; }

ROWS=()
while IFS='|' read -r label dir prod ours raw; do
  [[ $label =~ ^#.*$ || -z $label ]] && continue
  ROWS+=("$label|$dir|$prod|$ours|$raw")
done < <(grep -v '^#' "$TSV")
((${#ROWS[@]})) || { echo "[render-perf-chart] ERROR: no metric rows in $TSV" >&2; exit 2; }
N=${#ROWS[@]}

# per row: pct (int), bar width (pct*6), color; labels sit at width+156 / +194
BAR=""
i=0
for r in "${ROWS[@]}"; do
  IFS='|' read -r label dir prod ours raw <<< "$r"
  pct=$(awk -v d="$dir" -v p="$prod" -v o="$ours" 'BEGIN { v = (d=="lower") ? p/o*100 : o/p*100; printf "%.0f", v }')
  w=$((pct * 6))
  color=$([[ $pct -ge 100 ]] && echo "#3fb950" || echo "#d29922")
  by=$((30 + i * 36))
  ty=$(awk -v b="$by" 'BEGIN { printf "%.1f", b + 14.5 }')
  BAR+="    <text x=\"10\" y=\"$ty\">$label</text>"$'\n'
  BAR+="    <rect x=\"150\" y=\"$by\" width=\"$w\" height=\"20\" rx=\"3\" fill=\"$color\"/>"$'\n'
  BAR+="    <text x=\"$((156 + w))\" y=\"$ty\" font-weight=\"600\" fill=\"$color\">${pct}%</text>"$'\n'
  BAR+="    <text x=\"$((194 + w))\" y=\"$ty\" font-size=\"8.5\" fill=\"#8b949e\">$raw</text>"$'\n'
  i=$((i + 1))
done

LAST=$((30 + (N - 1) * 36 + 20))        # the last bar's bottom edge
CAPTION_Y=$((LAST + 26))
HEIGHT=$((LAST + 44))

render() {
  cat <<EOF
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 $HEIGHT" width="920" height="$HEIGHT" font-family="ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif">

  <text x="10" y="16" font-size="12" font-weight="600" fill="#e6edf3">The three fronts — this line vs production 0.6.6 (same harness, $dated receipts)</text>
  <text x="750" y="16" font-size="9.5" fill="#8b949e" text-anchor="middle">production = 100%</text>

  <!-- rows: label | bar | % + raw values; higher is always better (time metrics inverted) -->
  <g font-size="10.5" fill="#9fb1c9">
$BAR  </g>

  <!-- the 100% line -->
  <line x1="750" y1="24" x2="750" y2="$((LAST + 6))" stroke="#8b949e" stroke-opacity="0.55" stroke-dasharray="4 4"/>

  <text x="10" y="$CAPTION_Y" font-size="8.5" fill="#8b949e">Yellow = open fronts; green = won. Higher is always better (time metrics inverted). Receipts: $receipts</text>

</svg>
EOF
}

if [[ "${1:-}" == "--check" ]]; then
  TMP=$(mktemp /tmp/opencode/perf-chart.XXXXXX.svg)
  render > "$TMP"
  if cmp -s "$TMP" "$SVG"; then
    rm -f "$TMP"; echo "[render-perf-chart] the committed SVG matches the anchors ✓"; exit 0
  fi
  echo "[render-perf-chart] STALE: $SVG differs from the anchors — run ./scripts/render-perf-chart.sh and commit both" >&2
  diff "$SVG" "$TMP" | head -8 >&2 || true
  rm -f "$TMP"; exit 1
fi

mkdir -p /tmp/opencode
render > "$SVG"
echo "[render-perf-chart] wrote $SVG from $TSV ($N metrics)"
