#!/usr/bin/env python3
"""Run the vision suite against an endpoint (read-only prodqual-style probe).

Usage: vision-suite-probe.py <base-url> <out-json>
Same manifest/questions as the visionen phase; answers saved raw for the side-by-side.
"""
import base64
import json
import sys
import urllib.request

base, outp = sys.argv[1].rstrip("/"), sys.argv[2]
manifest = json.load(open("assets/vision/manifest.json"))
mid = json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))["data"][0]["id"]
print("model:", mid)
R = []
for e in manifest:
    img = open("assets/vision/" + e["file"], "rb").read()
    mime = "image/png" if e["file"].endswith(".png") else "image/jpeg"
    durl = "data:%s;base64," % mime + base64.b64encode(img).decode()
    for kind, qkey, mt in (("targeted", "question", 64), ("detail", "detail_question", 1024)):
        body = {"model": mid, "temperature": 0, "max_tokens": mt,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": durl}},
                    {"type": "text", "text": e[qkey]}]}]}
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.loads(resp.read())
            R.append({"file": e["file"], "kind": kind,
                      "answer": d["choices"][0]["message"]["content"],
                      "pt": d["usage"]["prompt_tokens"], "ct": d["usage"]["completion_tokens"]})
            print("  %s %s: ok (%d out)" % (e["file"], kind, d["usage"]["completion_tokens"]))
        except Exception as ex:
            R.append({"file": e["file"], "kind": kind, "answer": "", "error": str(ex)})
            print("  %s %s: FAILED %s" % (e["file"], kind, ex))
json.dump(R, open(outp, "w"), indent=1)
print("saved:", outp)
