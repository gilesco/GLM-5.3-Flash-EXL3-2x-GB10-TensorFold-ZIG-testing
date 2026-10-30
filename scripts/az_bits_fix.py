#!/usr/bin/env python3
"""Normalize the convert's quantization_config to the native engine's admit()
contract (families/glm/cuda_exl3.zig): `bits` must be a JSON INTEGER (std.json
parses 4.0 as .float -> UnsupportedQuant) and codebook must be "mcg".

Byte-surgical: regex on the file text (no json round-trip — this repo's standing
lesson is that round-trips fail strict parsers). Then VERIFIES by json-parsing
(read-only) that bits is int 4 and codebook is mcg in every quant file present.
Nonzero exit = do not place, do not serve.
"""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/<user>/az-az-exl3-mcg")
problems = []
for name in ("config.json", "quantization_config.json"):
    p = d / name
    if not p.exists():
        continue
    text = p.read_text()
    text2, n = re.subn(r'("bits":\s*)(\d+)\.0\b', r"\g<1>\g<2>", text)
    if n:
        p.write_text(text2)
        print(f"[bits-fix] {name}: normalized {n} integral-float bits value(s)")
    parsed = json.loads(p.read_text())
    q = parsed if name == "quantization_config.json" else parsed.get("quantization_config", {})
    if "quant_method" in q:
        bits = q.get("bits")
        book = q.get("codebook")
        if not (isinstance(bits, int) and not isinstance(bits, bool) and bits == 4):
            problems.append(f"{name}: bits={bits!r} (need int 4)")
        if book != "mcg":
            problems.append(f"{name}: codebook={book!r} (need 'mcg')")
if problems:
    for x in problems:
        print(f"[bits-fix] FAIL: {x}")
    sys.exit(1)
print("[bits-fix] OK: integer bits=4, codebook=mcg — admits under families/glm/cuda_exl3.zig")
