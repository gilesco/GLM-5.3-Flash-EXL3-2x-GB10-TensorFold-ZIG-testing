#!/usr/bin/env bash
# Regenerate a comparison chart from its anchors file. The docs' rule: metrics that
# compare us against production are GRAPHS (generated, the truth bad or good — one
# convention: every bar = our performance as % of production, time metrics inverted,
# so higher is always better), while raw receipts stay tables.
#
#   ./scripts/render-perf-chart.sh                       # the standing pair:
#                                                        #   docs/perf-anchors.tsv -> docs/perf-vs-production.svg
#   ./scripts/render-perf-chart.sh docs/duel-anchors.tsv # a named pair: docs/duel-anchors.tsv -> docs/duel-vs-production.svg
#   ... --check                                          # render to a temp file and byte-compare against
#                                                        # the committed SVG; exit 1 + diff preview on drift
#
# The anchors file carries one row per metric:
#   label|direction(lower|higher)|production|ours|raw label (receipt numbers as displayed)
# and its '#' header carries the chart's text:
#   # title=...   # dated=...   # receipts=...
# Update the TSV, then re-run this script — never hand-edit the SVG.
#
# Layout: the labels sit left of a bar zone scaled so the LARGEST bar + its % label
# always end before the canvas edge; the raw values render INSIDE bars wide enough
# for them (right-aligned, dark text) and after the % otherwise; the raw label is
# truncated at its first paren or em-dash (the receipts stay in the TSV/ledger).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

CHECK=0; ARGS=()
for a in "$@"; do [[ $a == --check ]] && CHECK=1 || ARGS+=("$a"); done
TSV=${ARGS[0]:-docs/perf-anchors.tsv}
STEM=$(basename "$TSV" .tsv); STEM=${STEM%-anchors}   # perf -> docs/perf-vs-production.svg
SVG=${ARGS[1]:-docs/${STEM}-vs-production.svg}
[[ -f $TSV ]] || { echo "[render-perf-chart] ERROR: $TSV missing" >&2; exit 2; }

title=$(grep -m1 '^# title=' "$TSV" | cut -d= -f2-)
dated=$(grep -m1 '^# dated=' "$TSV" | cut -d= -f2-)
receipts=$(grep -m1 '^# receipts=' "$TSV" | cut -d= -f2-)
[[ -n $title && -n $dated && -n $receipts ]] \
  || { echo "[render-perf-chart] ERROR: $TSV needs '# title=', '# dated=' and '# receipts=' lines" >&2; exit 2; }

ROWS=()
while IFS='|' read -r label dir prod ours raw; do
  [[ $label =~ ^#.*$ || -z $label ]] && continue
  ROWS+=("$label|$dir|$prod|$ours|$raw")
done < <(grep -v '^#' "$TSV")
((${#ROWS[@]})) || { echo "[render-perf-chart] ERROR: no metric rows in $TSV" >&2; exit 2; }
N=${#ROWS[@]}

# the geometry: W=1000; labels 0..250; bars from 270; the 100% mark at 600;
# the zone past 100% scales to fit the largest bar's % label at the canvas edge.
W=1000; BX=270; X100=600; W100=$((X100 - BX))        # 330 px per 100%
MAXPCT=100
for r in "${ROWS[@]}"; do
  IFS='|' read -r label dir prod ours raw <<< "$r"
  pct=$(awk -v d="$dir" -v p="$prod" -v o="$ours" 'BEGIN { v = (d=="lower") ? p/o*100 : o/p*100; printf "%.0f", v }')
  (( pct > MAXPCT )) && MAXPCT=$pct
done
SCALE=$(awk -v m="$MAXPCT" -v z=$((W - BX - 90)) 'BEGIN { s = z / m; if (s > 3.5) s = 3.5; printf "%.4f", s }')  # px per %, capped so small maxima don't dwarf
BARX100=$(awk -v s="$SCALE" 'BEGIN { printf "%.0f", 100 * s }')
X100=$((BX + BARX100))

# the raw label: truncated at the first " (" or " —" (the receipts stay in the ledger)
short_raw() { echo "$1" | sed 's/ (.*$//; s/ —.*$//'; }

BAR=""
i=0
for r in "${ROWS[@]}"; do
  IFS='|' read -r label dir prod ours raw <<< "$r"
  pct=$(awk -v d="$dir" -v p="$prod" -v o="$ours" 'BEGIN { v = (d=="lower") ? p/o*100 : o/p*100; printf "%.0f", v }')
  w=$(awk -v s="$SCALE" -v p="$pct" 'BEGIN { printf "%.0f", p * s }')
  color=$([[ $pct -ge 100 ]] && echo "#3fb950" || echo "#d29922")
  by=$((34 + i * 40))
  ty=$(awk -v b="$by" 'BEGIN { printf "%.1f", b + 17.5 }')
  SHORT=$(short_raw "$raw")
  RAWW=$(( ${#SHORT} * 6 ))                # ~6 px a char at 10px text
  PCTW=$(( ${#pct} * 9 + 14 ))             # the % label's width at 13.5px bold
  BAR+="    <text x=\"12\" y=\"$ty\" font-size=\"12.5\" fill=\"#9fb1c9\">$label</text>"$'\n'
  BAR+="    <rect x=\"$BX\" y=\"$by\" width=\"$w\" height=\"26\" rx=\"4\" fill=\"$color\"/>"$'\n'
  if (( BX + w + PCTW + RAWW + 24 <= W )); then
    # room for both after the bar
    BAR+="    <text x=\"$((BX + w + 8))\" y=\"$ty\" font-size=\"13.5\" font-weight=\"700\" fill=\"$color\">${pct}%</text>"$'\n'
    BAR+="    <text x=\"$((BX + w + 8 + PCTW))\" y=\"$ty\" font-size=\"10\" fill=\"#8b949e\">$SHORT</text>"$'\n'
  elif (( BX + w + 8 + RAWW + 8 <= W )); then
    # the % inside the bar's right end (dark), the raw after the bar
    BAR+="    <text x=\"$((BX + w - 10))\" y=\"$ty\" font-size=\"13.5\" font-weight=\"700\" fill=\"#1a1f24\" text-anchor=\"end\">${pct}%</text>"$'\n'
    BAR+="    <text x=\"$((BX + w + 8))\" y=\"$ty\" font-size=\"10\" fill=\"#8b949e\">$SHORT</text>"$'\n'
  elif (( w >= RAWW + PCTW + 44 )); then
    # both inside the bar: the % right-aligned dark, the raw just left of it
    BAR+="    <text x=\"$((BX + w - 10))\" y=\"$ty\" font-size=\"13.5\" font-weight=\"700\" fill=\"#1a1f24\" text-anchor=\"end\">${pct}%</text>"$'\n'
    BAR+="    <text x=\"$((BX + w - 10 - PCTW - 10))\" y=\"$ty\" font-size=\"10\" fill=\"#1a1f24\" text-anchor=\"end\">$SHORT</text>"$'\n'
  else
    # a very short bar: the % outside, the raw inside if it fits, else inside right-aligned
    BAR+="    <text x=\"$((BX + w + 8))\" y=\"$ty\" font-size=\"13.5\" font-weight=\"700\" fill=\"$color\">${pct}%</text>"$'\n'
    if (( BX + w + PCTW + 16 + RAWW + 8 <= W )); then
      BAR+="    <text x=\"$((BX + w + 8 + PCTW))\" y=\"$ty\" font-size=\"10\" fill=\"#8b949e\">$SHORT</text>"$'\n'
    elif (( w >= RAWW + 20 )); then
      BAR+="    <text x=\"$((BX + w - 10))\" y=\"$ty\" font-size=\"10\" fill=\"#1a1f24\" text-anchor=\"end\">$SHORT</text>"$'\n'
    fi
  fi
  i=$((i + 1))
done

LAST=$((34 + (N - 1) * 40 + 26))        # the last bar's bottom edge
CAPTION_Y=$((LAST + 30))
HEIGHT=$((LAST + 50))

render() {
  cat <<EOF
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 $W $HEIGHT" width="$W" height="$HEIGHT" font-family="ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif">

  <text x="12" y="18" font-size="14.5" font-weight="700" fill="#e6edf3">$title</text>
  <text x="$X100" y="18" font-size="10.5" fill="#8b949e" text-anchor="middle">production = 100%</text>

  <!-- rows: label | bar | % + raw values; higher is always better (time metrics inverted) -->
  <g>
$BAR  </g>

  <!-- the 100% line -->
  <line x1="$X100" y1="26" x2="$X100" y2="$((LAST + 8))" stroke="#8b949e" stroke-opacity="0.55" stroke-dasharray="4 4"/>

  <text x="12" y="$CAPTION_Y" font-size="10" fill="#8b949e">Yellow = open fronts; green = won. Higher is always better (time metrics inverted). Receipts ($dated): $receipts</text>

</svg>
EOF
}

if [[ $CHECK == 1 ]]; then
  TMP=$(mktemp "${TMPDIR:-/tmp}/perf-chart.XXXXXX.svg")
  render > "$TMP"
  if cmp -s "$TMP" "$SVG"; then
    rm -f "$TMP"; echo "[render-perf-chart] $SVG matches its anchors ✓"; exit 0
  fi
  echo "[render-perf-chart] STALE: $SVG differs from $TSV — re-run this script and commit both" >&2
  diff "$SVG" "$TMP" | head -8 >&2 || true
  rm -f "$TMP"; exit 1
fi


render > "$SVG"
echo "[render-perf-chart] wrote $SVG from $TSV ($N metrics)"
