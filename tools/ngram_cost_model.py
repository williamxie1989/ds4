#!/usr/bin/env python3
"""Honest cost model for draftless n-gram speculative decoding.

Corrects two errors in `ngram_accept_policy.py` that inflated the headline
speedup of NGRAM_DRAFTING_STUDY.md:

  1. Its "selective policy" table filtered the step list down to the speculated
     steps and then divided *conditional* tokens by *conditional* cost, silently
     dropping the ~90% of steps that never speculate.  Reported as an overall
     speedup, that is wrong by construction: a policy that speculates on 9.8% of
     steps cannot exceed 1.19x no matter how cheap the verify is.
  2. Its cost model charged for the *accepted* rows (`rows = 1 + min(acc, d)`)
     rather than the rows actually submitted to the verify.  Rejected draft rows
     cost just as much as accepted ones.

This script replays the same greedy speculative loop but records, per step, the
match length used and the accepted count, then answers the question the decision
actually turns on:

  for a given cost slope a (marginal cost of one extra verified row, in units of
  one plain decode step), what is the best achievable end-to-end speedup, and
  which per-match-length draft caps achieve it?

Cost model (matches the docstring intent of the original script):
    one plain decode step ................ 1.0
    a verify submitting c draft rows ..... 1 + a*c
    tokens committed by such a step ...... 1 + min(acc, c)
    a step with no draft ................. 1.0 / 1 token
so `a = 0.25` means a 4-row verify costs 1.75 decode steps.

Two replays are reported because they bracket reality:
  * `per-session` -- the n-gram index is reset at every session boundary.  This
    is what a deployment can actually do: one conversation's own history.
  * `pooled`      -- one index over all sessions concatenated (what the original
    study measured).  Cross-session matches are not available in production, so
    this is the optimistic bound.

Usage:
  python3 tools/ngram_cost_model.py --cache /tmp/ds4_ngram/seg_big.json
  python3 tools/ngram_cost_model.py --cache /tmp/ds4_ngram/stream_big.json --pooled-only
"""

import argparse
import collections
import json
import sys

CTX, GEN_TEXT, GEN_JSON = 0, 1, 2
MOD = (1 << 61) - 1
BASE = 1_000_003


def replay(ids, kind, k, min_k, max_draft):
    """Greedy speculative replay.  Returns [(match_len, accepted), ...].

    The match must end strictly before the query starts (`st + kk < i`); without
    that the match is the query itself and the draft is the real future.
    """
    n = len(ids)
    h = [0] * (n + 1)
    for i in range(n):
        h[i + 1] = (h[i] * BASE + ids[i] + 1) % MOD
    BK = [1] * (k + 1)
    for j in range(1, k + 1):
        BK[j] = (BK[j - 1] * BASE) % MOD
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

    recs, leak = [], [0, 0]
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
                draft = ids[st + kk: st + kk + max_draft]
                kk_used = kk
                break
        if draft:
            leak[0] += 1
            if draft[0] == ids[i]:
                leak[1] += 1
        acc = 0
        while acc < len(draft) and i + acc < n and kind[i + acc] != CTX \
                and ids[i + acc] == draft[acc]:
            acc += 1
        recs.append((kk_used, acc))
        i += 1 + acc
    return recs, leak


def optimal_policy(by_len, steps, a, max_cap):
    """Dinkelbach: maximise sum(p*tok) / sum(p*cost) over independent caps.

    For a fixed lambda the objective separates per match length, so each cap is
    chosen independently; iterate lambda to the fixed point.
    """
    p = {kk: len(v) / steps for kk, v in by_len.items()}

    def tok(kk, c):
        if kk == 0 or c == 0:
            return 1.0
        v = by_len[kk]
        return 1.0 + sum(min(x, c) for x in v) / len(v)

    lam, caps = 1.0, {kk: 0 for kk in p}
    for _ in range(400):
        for kk in p:
            if kk == 0:
                caps[kk] = 0
                continue
            caps[kk] = max((tok(kk, c) - lam * (1 + a * c), c)
                           for c in range(0, max_cap + 1))[1]
        num = sum(p[kk] * tok(kk, caps[kk]) for kk in p)
        den = sum(p[kk] * (1 + a * caps[kk]) for kk in p)
        nxt = num / den
        if abs(nxt - lam) < 1e-13:
            lam = nxt
            break
        lam = nxt
    return lam, caps, num, den


def report(tag, recs, leak, max_cap, slopes, out):
    steps = len(recs)
    by_len = collections.defaultdict(list)
    for kk, acc in recs:
        by_len[kk].append(acc)
    ceiling = sum(1 + acc for _, acc in recs) / steps

    out.append("## %s" % tag)
    out.append("")
    out.append("%d verify steps, %d tokens, mean %.3f tokens/step; "
               "**a=0 ceiling %.3fx** (draft cap %d, so the ceiling is what one "
               "verify pass can confirm, not what the match run offered)"
               % (steps, sum(1 + a for _, a in recs), ceiling, ceiling, max_cap))
    out.append("")
    out.append("leak sentinel (draft[0] == real next token) = %.3f "
               "(must sit near the unconditional first-token hit rate, ~0.3; "
               "1.0 means the match is the query itself)" % (leak[1] / max(leak[0], 1)))
    out.append("")
    out.append("| longest match | steps | share | mean acc | P(acc>=1) | P(acc>=4) |")
    out.append("|---|---|---|---|---|---|")
    for kk in sorted(by_len, reverse=True):
        if kk == 0:
            continue
        v = by_len[kk]
        out.append("| %d | %d | %.2f%% | %.3f | %.3f | %.3f |"
                   % (kk, len(v), 100.0 * len(v) / steps, sum(v) / len(v),
                      sum(1 for x in v if x >= 1) / len(v),
                      sum(1 for x in v if x >= 4) / len(v)))
    out.append("")
    out.append("### best achievable end-to-end speedup")
    out.append("")
    out.append("| a | speedup | mean rows/step | speculate share | best caps (match_len:rows) |")
    out.append("|---|---|---|---|---|")
    for a in slopes:
        lam, caps, num, den = optimal_policy(by_len, steps, a, max_cap)
        used = {kk: c for kk, c in caps.items() if c > 0}
        share = sum(len(by_len[kk]) for kk in used) / steps
        capstr = " ".join("%d:%d" % (kk, used[kk]) for kk in sorted(used))
        out.append("| %.2f | **%.3fx** | %.3f | %.1f%% | %s |"
                   % (a, lam, den, 100.0 * share, capstr or "(never speculate)"))
    out.append("")
    return by_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/ds4_ngram/seg_big.json")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-draft", type=int, default=16)
    # A verify submits at most DS4_TP_BATCH_MAX_ROWS = 8 rows, so at most 7 draft
    # tokens can be checked in one pass.  The replay may record a longer
    # acceptance run; the policy cannot cash it in.
    ap.add_argument("--max-cap", type=int, default=7)
    ap.add_argument("--slopes", default="0,0.02,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.60,1.00")
    ap.add_argument("--pooled-only", action="store_true")
    ap.add_argument("--out", default="/tmp/ds4_ngram/cost_model.md")
    args = ap.parse_args()
    slopes = [float(x) for x in args.slopes.split(",")]

    d = json.load(open(args.cache))
    out = ["# n-gram drafting: honest speedup vs verify cost slope", "",
           "k=%d min_k=%d; replay records acceptance with a %d-token draft, but the "
           "policy is capped at %d draft rows (a verify submits at most "
           "DS4_TP_BATCH_MAX_ROWS = 8 rows)."
           % (args.k, args.min_k, args.max_draft, args.max_cap),
           "cost(verify of c draft rows) = 1 + a*c decode steps.", ""]

    if "segs" in d:
        if not args.pooled_only:
            recs, leaks = [], [0, 0]
            for s in d["segs"]:
                r, lk = replay(s["ids"], s["kind"], args.k, args.min_k, args.max_draft)
                recs += r
                leaks[0] += lk[0]
                leaks[1] += lk[1]
            report("per-session index (deployment-realistic)", recs, leaks,
                   args.max_cap, slopes, out)
        ids, kind = [], []
        for s in d["segs"]:
            ids += s["ids"]
            kind += s["kind"]
    else:
        ids, kind = d["ids"], d["kind"]
    recs, leaks = replay(ids, kind, args.k, args.min_k, args.max_draft)
    report("pooled index over all sessions (optimistic; what the study measured)",
           recs, leaks, args.max_cap, slopes, out)

    open(args.out, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
