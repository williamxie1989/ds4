#!/usr/bin/env python3
"""Draftless n-gram speculative decoding: acceptance simulation on real transcripts.

Answers one question with real data: if an n-gram lookup drafter were bolted onto
the existing verify path (ds4_session_eval_speculative*), how many tokens would
each verify step confirm on OUR workload?

Pipeline
--------
1. Recorded DSH session transcripts (zstd JSONL).  Model-generated spans come
   from `assistant/message` records, split into prose (reasoning/text) and
   tool-call JSON (arguments); user turns and tool results are context that the
   lookup may match against but never generates.
2. Tokenize with the model's own tokenizer, extracted offline from GGUF metadata
   (byte-level BPE, vocab 129280 + merges) under the engine's JoyAI pre-split
   rules (mirrored from ds4.c::bpe_tokenize_text).
3. Replay greedy decoding with an incremental "most recent occurrence" n-gram
   index (llama.cpp ngram-simple shape): at each generated position look up the
   last k tokens among positions that end strictly before it, draft what followed
   the most recent match, accept the longest matching prefix, advance by
   1 + accepted (the +1 is the model's own token at the first mismatch).

Speedup model: steps that find no draft cost 1; steps that find one cost c
(verify cost in decode-step units; the MTP line measured ~3 for its own verify
shape).  speedup = tokens / ((steps - matched) + matched * c).
"""

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import time

CACHE = "/tmp/ds4_ngram"
SESSIONS_GLOB = "~/.dsh/sessions/*/session-*/session.jsonl.zstd"
CTX, GEN_TEXT, GEN_JSON = 0, 1, 2


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
def build_tokenizer(gguf_path):
    from gguf import GGUFReader
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    r = GGUFReader(gguf_path)

    def _s(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8", "surrogateescape")
        return str(x)

    toks = [_s(t) for t in r.fields["tokenizer.ggml.tokens"].contents()]
    merges = [_s(m) for m in r.fields["tokenizer.ggml.merges"].contents()]
    vocab = {t: i for i, t in enumerate(toks)}
    pairs = []
    for m in merges:
        a, _, b = m.partition(" ")
        pairs.append((a, b))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs, unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tk.decoder = decoders.ByteLevel()
    return tk


_ASCII_PUNCT = set(b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _is_digit(c):
    return 0x30 <= c <= 0x39


def _is_alpha(c):
    return (0x41 <= c <= 0x5A) or (0x61 <= c <= 0x7A)


def _is_punct(c):
    return c in _ASCII_PUNCT


def _is_space(c):
    return c in (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C)


def _is_newline(c):
    return c in (0x0A, 0x0D)


def _cjk_cp(cp):
    return (0x2E80 <= cp <= 0x9FFF) or (0xA000 <= cp <= 0xA4CF) or \
           (0xAC00 <= cp <= 0xD7AF) or (0xF900 <= cp <= 0xFAFF) or \
           (0x3040 <= cp <= 0x30FF) or (0x20000 <= cp <= 0x3FFFF)


def _letter_like(cp):
    if cp < 128:
        return _is_alpha(cp)
    return chr(cp).isalpha() or (0x0300 <= cp <= 0x036F)


def _consume_letters(s, i):
    n = len(s)
    while i < n:
        cp = ord(s[i])
        if _letter_like(cp) or (cp == 0x200D and i + 1 < n):
            i += 1
        else:
            break
    return i


def joyai_split(text):
    """Pre-split pieces, mirroring ds4.c::bpe_tokenize_text (engine is the spec)."""
    n = len(text)
    pos = 0
    out = []
    while pos < n:
        start = pos
        cp = ord(text[pos])
        if cp < 128 and _is_digit(cp):
            nd = 0
            while pos < n and ord(text[pos]) < 128 and _is_digit(ord(text[pos])) and nd < 3:
                pos += 1
                nd += 1
        elif _cjk_cp(cp):
            while pos < n and _cjk_cp(ord(text[pos])):
                pos += 1
        elif cp < 128 and _is_punct(cp) and pos + 1 < n and ord(text[pos + 1]) < 128 \
                and _is_alpha(ord(text[pos + 1])):
            pos += 1
            while pos < n and ord(text[pos]) < 128 and _is_alpha(ord(text[pos])):
                pos += 1
        elif _letter_like(cp):
            pos = _consume_letters(text, pos)
        elif cp < 128 and not _is_newline(cp) and not _is_punct(cp) and pos + 1 < n \
                and _letter_like(ord(text[pos + 1])):
            pos += 1
            pos = _consume_letters(text, pos)
        elif cp == 0x20 and pos + 1 < n and ord(text[pos + 1]) < 128 \
                and _is_punct(ord(text[pos + 1])):
            pos += 1
            while pos < n and ord(text[pos]) < 128 and _is_punct(ord(text[pos])):
                pos += 1
            while pos < n and _is_newline(ord(text[pos])):
                pos += 1
        elif cp < 128 and _is_punct(cp):
            while pos < n and ord(text[pos]) < 128 and _is_punct(ord(text[pos])):
                pos += 1
            while pos < n and _is_newline(ord(text[pos])):
                pos += 1
        elif _is_space(cp):
            p = pos
            last_nl = 0
            while p < n and _is_space(ord(text[p])):
                sc = ord(text[p])
                p += 1
                if _is_newline(sc):
                    last_nl = p
            pos = last_nl if last_nl else p
        else:
            pos += 1
        if pos == start:
            pos += 1
        out.append(text[start:pos])
    return out


def tokenize_stream(tk, pieces):
    ids, kind = [], []
    for role, text in pieces:
        if not text:
            continue
        kk = {"ctx": CTX, "gen_text": GEN_TEXT, "gen_json": GEN_JSON}[role]
        for e in tk.encode_batch(joyai_split(text), add_special_tokens=False):
            ids.extend(e.ids)
            kind.extend([kk] * len(e.ids))
        ids.append(0)
        kind.append(CTX)
    return ids, kind


# --------------------------------------------------------------------------
# transcript extraction
# --------------------------------------------------------------------------
def _walk(node, out, depth=0):
    if depth > 8 or node is None or isinstance(node, str):
        return
    if isinstance(node, list):
        for x in node:
            _walk(x, out, depth + 1)
        return
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("reasoning", "text") and isinstance(node.get("text"), str):
            out.append(("gen_text", node["text"]))
        elif t in ("tool-call", "tool_call") and isinstance(node.get("arguments"), str):
            out.append(("gen_json", node["arguments"]))
        elif t == "tool-result":
            _walk(node.get("content"), out, depth + 1)
        else:
            for k, v in node.items():
                if k == "text" and isinstance(v, str):
                    out.append(("gen_text", v))
                elif k == "arguments" and isinstance(v, str):
                    out.append(("gen_json", v))
                else:
                    _walk(v, out, depth + 1)


def read_session(path, max_chars):
    try:
        raw = subprocess.run(["zstd", "-dc", path], capture_output=True,
                             check=True).stdout.decode("utf-8", "replace")
    except Exception as e:
        print("  skip %s: %s" % (path, e), file=sys.stderr)
        return []
    pieces, total = [], 0
    for line in raw.split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        t = rec.get("type")
        if t == "assistant/message":
            buf = []
            _walk(rec.get("data"), buf)
            for role, text in buf:
                if text:
                    pieces.append((role, text))
                    total += len(text)
        elif t in ("user/message", "tool/result"):
            buf = []
            _walk(rec.get("data"), buf)
            text = "\n".join(x[1] for x in buf)
            if text:
                pieces.append(("ctx", text))
                total += len(text)
        if total >= max_chars:
            break
    return pieces


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def simulate(ids, kind, k, min_k, max_draft):
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
        """index every n-gram that ends at or before `upto`"""
        for kk in ks:
            # the match must END strictly before the query starts: s+kk <= upto-1
            lim = upto - kk - 1
            s = ins[kk]
            while s <= lim and s + kk <= n:
                latest[kk][(h[s + kk] - h[s] * BK[kk]) % MOD] = s
                s += 1
            ins[kk] = s

    dist = collections.Counter()
    leak = [0, 0]                    # [steps with draft, steps where draft[0] == ids[i]]
    by_kind = collections.defaultdict(collections.Counter)
    steps = produced = matched = 0
    i = 0
    while i < n:
        refresh(i)                      # all matches end strictly before i
        if kind[i] == CTX:
            i += 1
            continue
        draft = []
        for kk in reversed(ks):
            if i < kk:
                continue
            st = latest[kk].get((h[i] - h[i - kk] * BK[kk]) % MOD)
            if st is not None and st + kk < i and ids[st:st + kk] == ids[i - kk:i]:
                draft = ids[st + kk: st + kk + max_draft]
                break
        if not draft:
            dist[0] += 1
            by_kind[kind[i]][0] += 1
            steps += 1
            produced += 1
            i += 1
            continue
        matched += 1
        leak[0] += 1
        if draft and draft[0] == ids[i]:
            leak[1] += 1
        acc = 0
        while acc < len(draft) and i + acc < n and kind[i + acc] != CTX \
                and ids[i + acc] == draft[acc]:
            acc += 1
        dist[acc] += 1
        by_kind[kind[i]][acc] += 1
        steps += 1
        produced += 1 + acc
        i += 1 + acc
    return dist, steps, produced, matched, by_kind, leak


def speedup(tokens, steps, matched, c):
    return tokens / max((steps - matched) + matched * c, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="gguf/DeepSeek-V4.1-Flash-Q2.gguf")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=400_000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-k", type=int, default=2)
    ap.add_argument("--max-draft", type=int, default=16)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--segments", action="store_true",
                    help="also write seg_<tag>.json with session boundaries kept, "
                         "which ngram_cost_model.py needs for the per-session index")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    cache_f = os.path.join(args.cache, "stream_%s.json" % args.tag)
    t0 = time.time()
    if os.path.exists(cache_f):
        d = json.load(open(cache_f))
        ids, kind = d["ids"], d["kind"]
        print("cache hit: %d tokens" % len(ids))
    else:
        files = sorted(glob.glob(os.path.expanduser(SESSIONS_GLOB)),
                       key=os.path.getsize, reverse=True)[:args.sessions]
        print("sessions:", len(files))
        tk = build_tokenizer(args.gguf)
        ids, kind = [], []
        segs = []
        for f in files:
            pieces = read_session(f, args.max_chars)
            if not pieces:
                continue
            a, b = tokenize_stream(tk, pieces)
            if args.segments and len(a) >= 500:
                segs.append({"name": f.split('/')[-2][:40], "ids": a, "kind": b})
            ids.extend(a)
            kind.extend(b)
            print("  %-52s pieces=%4d tokens=%8d (%.0fs)"
                  % (f.split('/')[-2][:50], len(pieces), len(a), time.time() - t0))
        json.dump({"ids": ids, "kind": kind}, open(cache_f, "w"))
        if args.segments:
            seg_f = os.path.join(args.cache, "seg_%s.json" % args.tag)
            json.dump({"segs": segs}, open(seg_f, "w"))
            print("wrote %s (%d sessions)" % (seg_f, len(segs)))
        print("tokenized %d tokens in %.0fs" % (len(ids), time.time() - t0))

    n = len(ids)
    ngen = sum(1 for x in kind if x != CTX)
    njson = sum(1 for x in kind if x == GEN_JSON)
    print("stream %d tokens | generated %d (%.1f%%) | tool-call JSON %d (%.1f%% of generated)"
          % (n, ngen, 100.0 * ngen / max(n, 1), njson, 100.0 * njson / max(ngen, 1)))

    cfgs = [(args.k, args.max_draft)]
    if args.sweep:
        cfgs = [(k, d) for k in (2, 3, 4, 6) for d in (4, 8, 16)]
    out = []
    for (k, md) in cfgs:
        dist, steps, produced, matched, by_kind, leak = simulate(ids, kind, k, args.min_k, md)
        row = {
            "k": k, "max_draft": md, "steps": steps, "tokens": produced,
            "matched": matched, "match_rate": matched / max(steps, 1),
            "mean_accept": sum(a * c for a, c in dist.items()) / max(steps, 1),
            "tokens_per_step": produced / max(steps, 1),
            "acc_when_matched": (sum(a * c for a, c in dist.items()) / matched) if matched else 0.0,
            "speedup_c2": speedup(produced, steps, matched, 2.0),
            "speedup_c25": speedup(produced, steps, matched, 2.5),
            "speedup_c3": speedup(produced, steps, matched, 3.0),
            "leak_rate": leak[1] / max(leak[0], 1),
            "hist": {str(a): dist.get(a, 0) for a in range(0, 17)},
            "by_kind": {("json" if kk == GEN_JSON else "text"): {
                "steps": sum(v.values()),
                "acc_when_matched": (sum(a * c for a, c in v.items() if a) /
                                     max(sum(c for a, c in v.items() if a), 1)),
                "mean_accept": (sum(a * c for a, c in v.items()) / max(sum(v.values()), 1)),
            } for kk, v in by_kind.items()},
        }
        out.append(row)
        print("k=%-2d draft=%-3d steps=%6d matched=%5.1f%% mean_acc=%.2f acc|matched=%.2f "
              "tok/step=%.2f | speedup c=2:%.2f c=2.5:%.2f c=3:%.2f"
              % (k, md, steps, 100 * row["match_rate"], row["mean_accept"],
                 row["acc_when_matched"], row["tokens_per_step"],
                 row["speedup_c2"], row["speedup_c25"], row["speedup_c3"]),
              "leak=%.3f" % row["leak_rate"])

    json.dump(out, open(os.path.join(args.cache, "sweep_%s.json" % args.tag), "w"), indent=2)
    print("wrote", os.path.join(args.cache, "sweep_%s.json" % args.tag))


if __name__ == "__main__":
    main()
