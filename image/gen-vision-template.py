#!/usr/bin/env python3
"""Generate the GLM-5.3-Flash vision chat template from a checkpoint.

Reads the checkpoint's tokenizer.json (for the exact media-token bytes — they carry
zero-width characters, so they must be extracted, never hand-typed) and its
chat_template.jinja (the text-only base), replaces the media branch with one that
emits the image/video placeholder markers the native engine's vision path expands,
and keeps the audio reminder. One argument: the snapshot directory.
"""
import json
import sys

snap = sys.argv[1].rstrip("/")
tk = json.load(open(snap + "/tokenizer.json"))
byid = {t["id"]: t["content"] for t in tk.get("added_tokens", [])}
B, I, E = byid[154830], byid[154854], byid[154831]
VB, V, VE = byid[154832], byid[154855], byid[154833]
assert "image" in I and "video" in V and "begin" in B, "unexpected token strings"

lines = open(snap + "/chat_template.jinja").read().split("\n")
out, i, replaced = [], 0, False
while i < len(lines):
    l = lines[i]
    if "item.type in [" in l and "image" in l and "input_audio" in l:
        indent = l[: len(l) - len(l.lstrip())]
        out.append(indent + '{%- elif item is mapping and item.type in ["image", "image_url"] -%}')
        out.append(indent + "    {{- " + json.dumps(B + I + E) + " }}")
        out.append(indent + '{%- elif item is mapping and item.type in ["video", "video_url"] -%}')
        out.append(indent + "    {{- " + json.dumps(VB + V + VE) + " }}")
        out.append(indent + '{%- elif item is mapping and item.type in ["audio", "audio_url", "input_audio"] -%}')
        out.append(indent + '    {%- set media_type = item.type | replace("_url", "") | replace("input_", "") -%}')
        out.append(indent + '    {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don\'t have multi-modal input ability. Try different methods.</reminder>" }}')
        i += 3  # the original elif + its two body lines
        replaced = True
        continue
    out.append(l)
    i += 1
assert replaced, "the media branch was not found"
open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/glm5-next-vision.jinja", "w").write("\n".join(out))
start = next(n for n, l in enumerate(out) if "image_url" in l and "elif" in l)
for l in out[start - 1 : start + 8]:
    print(repr(l[:120]))
print("written: /tmp/glm5-next-vision.jinja")
