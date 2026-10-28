#!/usr/bin/env bash
# twoshot-probe.sh — the prefix-resume instrument (A/A/B shape).
#   A = fill + tail1, sent twice: A2 << A1 is the SAME-prompt resume signature.
#   B = fill + tail2 (a different tail, same fill): B1 fast means CROSS-prompt
#       resume — the fill's prefill reused even though the full prompts differ.
# Works against any OpenAI endpoint; on the head it also echoes the engine's
# prompt-cache counters between shots (the container glm53-native-tf).
#   usage: scripts/twoshot-probe.sh <base-url> <model> [pairs] [fill-words]
set -euo pipefail
BASE=${1:?base-url}
MODEL=${2:?model}
PAIRS=${3:-2}
WORDS=${4:-"harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry cistern gable linden sluice cobble rafter minnow heather bracken cellars windlass tarn larch furrow beck gantry lintel osier marl culvert"}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

cachesnap() { docker logs glm53-native-tf 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" 2>/dev/null || true; }

WORDS="$WORDS" MODEL="$MODEL" python3 - "$TMP" <<'PYEOF'
import json, os, sys
words = os.environ["WORDS"].split()
n = 2048
fill = " ".join(w if i % 11 == 0 else w + f"-{i % 89}" for i, w in ((i, words[i % len(words)]) for i in range(n)))
for name, tail in (("a", "Summarize the fill."), ("b", "List ten nouns from the fill.")):
    json.dump({"model": os.environ["MODEL"], "temperature": 0, "max_tokens": 1,
               "messages": [{"role": "user", "content": fill + "\n\n" + tail}]},
              open(f"{sys.argv[1]}/body-{name}.json", "w"))
PYEOF
curl -s -o /dev/null "$BASE/v1/models"   # warm the HTTP path

shot() { curl -s -o /dev/null -w '%{time_starttransfer}' -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data @"$TMP/body-$1.json"; }
ms() { python3 -c "print(f'{$1*1000:.0f}')"; }

echo "pairs=$PAIRS model=$MODEL (A = fill+tail1, B = fill+tail2)"
for p in $(seq 1 "$PAIRS"); do
  A1=$(shot a); C1=$(cachesnap || true)
  A2=$(shot a); C2=$(cachesnap || true)
  echo "pair $p: A1 $(ms $A1) ms | A2 $(ms $A2) ms | same-prompt saving $(python3 -c "print(f'{(1-$A2/$A1)*100:.0f}')")%  [counters before A2: ${C1:-n/a}]"
done
B1=$(shot b); CB=$(cachesnap || true)
echo "cross-prompt B1 (fill reused, new tail): $(ms $B1) ms  [counters: ${CB:-n/a}]"
