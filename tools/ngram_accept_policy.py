#!/usr/bin/env python3
"""Acceptance-by-match-length and optimal draft length for n-gram drafting.

SUPERSEDED (2026-10-06) by ngram_cost_model.py -- do not cite its speedup
numbers.  Two defects, both of which inflate the result:

  * the "selective policy" table filters the step list down to the speculated
    steps and then divides conditional tokens by conditional cost, silently
    dropping the ~90% of steps that never speculate; read as an overall speedup
    that is wrong by construction (a policy speculating on 9.8% of steps cannot
    exceed 1.19x at any verify cost);
  * cost_per_step charges for accepted rows (rows = 1 + min(acc, d)) instead of
    the rows actually submitted to the verify -- rejected rows cost the same as
    accepted ones -- and it is off by one against the c(n) = 1 + a*(n-1) model
    its own docstring states.

Kept because the acceptance structure it measures (survival curve, mean accepted
by match length, per-content-type split) is correct and still cited.

Consumes the token stream cached by ngram_accept_sim.py, replays the same
greedy spec loop while recording (match length used, accepted count, span kind),
then answers the two questions the flat sweep cannot:

  * survival curve S(j) = P(at least j draft tokens accepted)
  * expected tokens/step for every draft cap d, and the best d under a verify
    cost model c(n) = 1 + a*(n-1)  (a = marginal cost of one extra verified row,
    in decode-step units; a=0.25 means a 4-row verify costs 1.75 decode steps)
  * a selective policy table: speculate only when the longest match is >= m
"""

import argparse
import collections
import json
import sys

CTX, GEN_TEXT, GEN_JSON = 0, 1, 2


def replay(ids, kind, k, min_k, max_draft):
    n = len(ids)
    MOD = (1 << 61) - 1
    B = 1_000_003
    h = [0] * (n + 1)
    for i in range(n):
        h[i + 1] = (h[i] * B + ids[i] + 1) % MOD
    BK = [1] * (k + 1)
    for j in range(1, k + 1):
        BK[j] = (BK[j - 1] * B) % MOD
    ks = list(range(min_k, k + 1))
    latest = {kk: {} for kk in ks}
    ins = {kk: 0 for kk in ks}

    def refresh(upto):
        for kk in ks:
            lim = upto - kk - 1
            s = ins[kk]
            while s <= lim and s + kk <= n:
                latest[kk][(h[s + kk] - h[s] * BK[kk]) % MOD] = s
                s += 1
            ins[kk] = s

    recs = []                      # (kk_used, acc, kind)
    i = 0
    while i < n:
        refresh(i)
        if kind[i] == CTX:
            i += 1
            continue
        draft, kk_used = [], 0
        for kk in reversed(ks):
            if i < kk:
                continue
            st = latest[kk].get((h[i] - h[i - kk] * BK[kk]) % MOD)
            if st is not None and st + kk < i and ids[st:st + kk] == ids[i - kk:i]:
                draft, kk_used = ids[st + kk: st + kk + max_draft], kk
                break
        acc = 0
        while acc < len(draft) and i + acc < n and kind[i + acc] != CTX \
                and ids[i + acc] == draft[acc]:
            acc += 1
        recs.append((kk_used, acc, kind[i]))
        i += 1 + acc
    return recs


def tokens_per_step(recs, d):
    """Mean tokens produced per step when the draft is capped at d tokens."""
    tot = 0
    for kk, acc, _ in recs:
        tot += 1 + min(acc, d)
    return tot / max(len(recs), 1)


def cost_per_step(recs, d, a):
    """Mean cost in decode-step units: no-draft steps cost 1, verify steps
    1 + a*rows where rows = 1 + draft actually verified (capped at d)."""
    tot = 0.0
    for kk, acc, _ in recs:
        if kk == 0:
            tot += 1.0
        else:
            rows = 1 + min(acc, d)          # tokens the verify actually produced
            tot += 1.0 + a * rows
    return tot / max(len(recs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/ds4_ngram/stream_big.json")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-draft", type=int, default=16)
    ap.add_argument("--out", default="/tmp/ds4_ngram/policy.md")
    args = ap.parse_args()

    d = json.load(open(args.cache))
    ids, kind = d["ids"], d["kind"]
    print("stream %d tokens" % len(ids))
    recs = replay(ids, kind, args.k, args.min_k, args.max_draft)
    steps = len(recs)
    print("steps %d" % steps)

    # survival curve
    surv = [sum(1 for r in recs if r[1] >= j) / steps for j in range(0, 12)]

    # per-match-length acceptance
    by_kk = collections.defaultdict(list)
    for kk, acc, kd in recs:
        by_kk[kk].append((acc, kd))

    lines = ["# n-gram drafting: acceptance structure and optimal draft length", "",
             "stream: %d tokens, %d verify steps" % (len(ids), steps), "",
             "## survival curve  S(j) = P(acc >= j)", "",
             "| j | P(acc>=j) |", "|---|---|"]
    for j in range(0, 12):
        lines.append("| %d | %.3f |" % (j, surv[j]))

    lines += ["", "## acceptance by match length", "",
              "| match len | steps | share | mean acc | acc|matched |", "|---|---|---|---|---|"]
    for kk in sorted(by_kk, reverse=True):
        v = by_kk[kk]
        if kk == 0:
            continue
        ma = sum(a for a, _ in v) / len(v)
        lines.append("| %d | %d | %.1f%% | %.2f | %.2f |"
                     % (kk, len(v), 100.0 * len(v) / steps, ma, ma))

    lines += ["", "## expected tokens/step and cost, by draft cap d", "",
              "| d | tokens/step | c(d) a=0.15 | c(d) a=0.25 | c(d) a=0.40 | speedup a=.15 | a=.25 | a=.40 |",
              "|---|---|---|---|---|---|---|---|"]
    best = {}
    for dd in range(1, 13):
        tps = tokens_per_step(recs, dd)
        c1, c2, c3 = (cost_per_step(recs, dd, a) for a in (0.15, 0.25, 0.40))
        s1, s2, s3 = tps / c1, tps / c2, tps / c3
        for tag, s in (("a15", s1), ("a25", s2), ("a40", s3)):
            if tag not in best or s > best[tag][1]:
                best[tag] = (dd, s)
        lines.append("| %d | %.3f | %.3f | %.3f | %.3f | **%.2fx** | **%.2fx** | **%.2fx** |"
                     % (dd, tps, c1, c2, c3, s1, s2, s3))

    lines += ["", "## selective policy: speculate only when longest match >= m", "",
              "| m | steps speculated | tokens/step | speedup a=.25 | speedup a=.40 |",
              "|---|---|---|---|---|"]
    for m in range(2, 7):
        sel = [r for r in recs if r[0] >= m]
        if not sel:
            continue
        tps = tokens_per_step(sel, 8)
        c25 = cost_per_step(sel, 8, 0.25)
        c40 = cost_per_step(sel, 8, 0.40)
        lines.append("| %d | %d (%.1f%%) | %.3f | %.2fx | %.2fx |"
                     % (m, len(sel), 100.0 * len(sel) / steps, tps, tps / c25, tps / c40))

    lines += ["", "## same, restricted to tool-call JSON spans", ""]
    js = [r for r in recs if r[2] == GEN_JSON]
    tx = [r for r in recs if r[2] == GEN_TEXT]
    for tag, sub in (("json", js), ("prose", tx)):
        if not sub:
            continue
        tps = tokens_per_step(sub, 8)
        lines.append("- %s: steps=%d tokens/step(d=8)=%.3f speedup a=.25=%.2fx a=.40=%.2fx"
                     % (tag, len(sub), tps, tps / cost_per_step(sub, 8, 0.25),
                        tps / cost_per_step(sub, 8, 0.40)))

    lines += ["", "best d per cost slope: " + ", ".join(
        "%s -> d=%d (%.2fx)" % (k, v[0], v[1]) for k, v in sorted(best.items())), ""]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-14:]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
