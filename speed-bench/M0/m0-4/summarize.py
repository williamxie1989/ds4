#!/usr/bin/env python3
"""Summarize the M0-4 window: steady decode ABBA, append baselines,
SPEC_ROWS A/B, GLM decode, MTP smoke settlement. Run after run_window.sh."""
import csv, glob, os, re, statistics, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
V41 = os.path.join(ROOT, "v41")
GLM = os.path.join(ROOT, "glm")
LOGS = os.path.join(ROOT, "logs")

def steady(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("gen_steady_tps"):
                out[int(r["ctx_tokens"])] = float(r["gen_steady_tps"])
    return out

def median_across(paths, ctx):
    vals = [v.get(ctx) for v in (steady(p) for p in paths) if v.get(ctx)]
    return statistics.median(vals) if vals else None

print("== V4.1 steady decode (gen_steady_tps medians) ==")
off = sorted(glob.glob(f"{V41}/off-*.csv"))
on = sorted(glob.glob(f"{V41}/on-*.csv"))
for ctx in (2048, 4096, 8192, 16384, 32768):
    o = median_across(off, ctx)
    n = median_across(on, ctx)
    if o and n:
        print(f"  ctx {ctx:>6}  OFF={o:6.2f} (n={len(off)})  SPEC_ON={n:6.2f} (n={len(on)})  delta={100*(n-o)/o:+5.1f}%")
    elif o:
        print(f"  ctx {ctx:>6}  OFF={o:6.2f}")

print("== V4.1 append baselines (prefill_tps of appended rows) ==")
for name in ("append831-cold", "append831-warm", "append1524-warm"):
    p = f"{V41}/{name}.csv"
    if not os.path.exists(p):
        print(f"  {name}: MISSING"); continue
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r.get("prefill_tps"):
                tps = float(r["prefill_tps"]); tok = int(r["prefill_tokens"])
                print(f"  {name}: ctx={r['ctx_tokens']:>7} append {tok:>5} rows @ {tps:7.1f} t/s = {tok/tps:6.2f} s")

print("== SPEC_ROWS small-append A/B ==")
for tag in ("small4-off", "small4-on", "small8-off", "small8-on"):
    p = f"{V41}/{tag}.csv"
    if not os.path.exists(p):
        print(f"  {tag}: MISSING"); continue
    tps = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            if r.get("prefill_tps"):
                tps.append(float(r["prefill_tps"]))
    if tps:
        print(f"  {tag}: n={len(tps)} median prefill {statistics.median(tps):7.1f} t/s")

for pair in (("small4-off", "small4-on"), ("small8-off", "small8-on")):
    med = {}
    for tag in pair:
        p = f"{V41}/{tag}.csv"
        if not os.path.exists(p): continue
        tps = [float(r["prefill_tps"]) for r in csv.DictReader(open(p)) if r.get("prefill_tps")]
        if tps: med[tag] = statistics.median(tps)
    if len(med) == 2:
        a, b = med[pair[0]], med[pair[1]]
        print(f"  {pair[1]} vs {pair[0]}: {100*(b-a)/a:+5.1f}%")

print("== parity dumps (logit JSON byte-diff OFF vs ON) ==")
for n in (4, 8):
    d_off = sorted(glob.glob(f"{ROOT}/dumps_off{n}/*.json"))
    d_on = sorted(glob.glob(f"{ROOT}/dumps_on{n}/*.json"))
    if not d_off or not d_on:
        print(f"  dumps {n}: missing (off={len(d_off)} on={len(d_on)})"); continue
    import hashlib
    def dig(d):
        return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in d}
    a, b = dig(d_off), dig(d_on)
    same_keys = set(a) == set(b)
    diffs = [k for k in a if k in b and a[k] != b[k]]
    print(f"  dumps {n}: files_off={len(a)} files_on={len(b)} same_names={same_keys} byte_diffs={len(diffs)}"
          + (f" -> {diffs[:3]}" if diffs else "  (BIT-EQUAL on dumped frontiers)"))

print("== GLM decode (median of G1/G2) ==")
gpaths = sorted(glob.glob(f"{GLM}/decode-G*.csv"))
for ctx in (2048, 4096, 8192, 12288, 16384):
    m = median_across(gpaths, ctx)
    if m: print(f"  ctx {ctx:>6}  {m:6.2f} t/s (n={len(gpaths)})")

print("== GLM MTP settlement line ==")
p = f"{LOGS}/m1_smoke.log"
if os.path.exists(p):
    txt = open(p, errors="replace").read()
    lines = [l for l in txt.splitlines() if "mtp" in l.lower() and ("t/s" in l or "accept" in l or "cyc" in l or "economy" in l or "verify" in l)]
    for l in lines[-12:]:
        print("  " + l)
    if not lines:
        print("  (no settlement lines found — check", p, ")")
else:
    print("  MISSING", p)
