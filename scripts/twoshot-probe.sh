#!/usr/bin/env bash
# twoshot-probe.sh — the cross-request prefix-resume instrument.
# Sends the SAME long prompt twice per pair and compares TTFT: shot 2 << shot 1
# is the direct signature of a working prefix cache (the marks-patch's claim;
# production's block cache is the reference). Works against any OpenAI endpoint.
#   usage: scripts/twoshot-probe.sh <base-url> <model> [pairs] [fill-words]
# Counter capture: when the local container glm53-native-tf exists, the engine's
# own prompt-cache line (hits/misses/evicted) is echoed between shots.
set -euo pipefail
BASE=${1:?base-url}
MODEL=${2:?model}
PAIRS=${3:-3}
WORDS=${4:-"harbor lantern gravel meadow tideline quarry bramble inkwell thicket ferry cistern gable linden sluice cobble rafter minnow heather bracken cellars windlass tarn larch furrow beck gantry lintel osier marl culvert"}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

cachesnap() { docker logs glm53-native-tf 2>&1 | grep "prompt cache:" | tail -1 | grep -oE "hits [0-9]+, misses [0-9]+, evicted [0-9]+" 2>/dev/null || true; }

# one fill body, byte-identical across both shots of every pair (~2k words ~ 8k tokens)
WORDS="$WORDS" python3 - "$TMP/body.json" <<'PYEOF'
import json, os, sys
words = os.environ["WORDS"].split()
n = 2048
fill = " ".join(w if i % 11 == 0 else w + f"-{i % 89}" for i, w in ((i, words[i % len(words)]) for i in range(n)))
json.dump({"model": os.sys.argv[2] if len(os.sys.argv) > 2 else "", "temperature": 0,
           "max_tokens": 1, "messages": [{"role": "user", "content": fill + "\n\nSummarize the fill."}]},
          open(sys.argv[1], "w"))
PYEOF
python3 - "$TMP/body.json" "$MODEL" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); d["model"] = sys.argv[2]; json.dump(d, open(sys.argv[1], "w"))
PYEOF
curl -s -o /dev/null "$BASE/v1/models"   # warm the HTTP path

echo "pairs=$PAIRS model=$MODEL"
for p in $(seq 1 "$PAIRS"); do
  T1=$(curl -s -o /dev/null -w '%{time_starttransfer}' -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data @"$TMP/body.json")
  C1=$(cachesnap || true)
  T2=$(curl -s -o /dev/null -w '%{time_starttransfer}' -m 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' --data @"$TMP/body.json")
  C2=$(cachesnap || true)
  python3 -c "t1,t2=$T1,$T2; print(f'pair $p: shot1 {t1*1000:.0f} ms | shot2 {t2*1000:.0f} ms | resume saving {(1-t2/t1)*100:.0f}%')"
  [ -n "$C1" ] && echo "  counters: before-shot2 $C1"
done
