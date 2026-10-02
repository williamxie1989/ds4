#!/usr/bin/env python3
"""Prefix-reuse A/B verifier for ds4-server.

Modes:
  --baseline   Aggregate per-step cacheReadTokens/inputTokens recorded by DSH
               in a session transcript (the pre-fix ground truth; no server).
  --replay     Replay the transcript as sequentially-growing chat requests
               against a (patched) ds4-server, max_tokens=1, and report the
               cache-hit ratio per step.  --apply-prunes reproduces the
               mid-history prune episode; without it the prune is skipped so
               tool-turn reuse can be measured cleanly.

Pure stdlib; no model loads on this machine.
"""
import argparse
import json
import sys
import time
import urllib.request

def blocks_text(blocks):
    if isinstance(blocks, str):
        return blocks
    out = []
    for b in blocks or []:
        if isinstance(b, dict) and b.get('type') in ('text', 'reasoning'):
            out.append(b.get('text', ''))
    return '\n'.join(out)

def load_timeline(path):
    """Return (tools, steps). Each step is a list of message dicts to append
    in order; the request at step k sends the accumulated history."""
    tools = None
    steps = []          # list of list-of-msg
    cur = []
    seq_to_idx = {}     # transcript seq -> index into flat msg list
    flat = []
    prunes = []
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        t = d['type']
        if t == 'request/header' and tools is None:
            tools = d['data']['header'].get('tools') or []
        elif t == 'user/message':
            msg = {'role': 'user', 'content': blocks_text(d['data'].get('content'))}
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1; cur.append(msg)
        elif t == 'assistant/message':
            m = d['data']['message']
            reasoning, text, calls = [], [], []
            for b in m.get('content', []):
                bt = b.get('type')
                if bt == 'reasoning':
                    reasoning.append(b.get('text', ''))
                elif bt == 'text':
                    text.append(b.get('text', ''))
                elif bt == 'tool-call':
                    calls.append({'id': b.get('callId') or b.get('id') or ('call_%d' % d['seq']),
                                  'type': 'function',
                                  'function': {'name': b.get('name', ''),
                                               'arguments': json.dumps(b.get('arguments', {}), ensure_ascii=False)}})
            msg = {'role': 'assistant'}
            if text:
                msg['content'] = '\n'.join(text)
            if reasoning:
                msg['reasoning_content'] = '\n'.join(reasoning)
            if calls:
                msg['tool_calls'] = calls
            msg['content'] = msg.get('content') or ('' if calls else '')
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1; cur.append(msg)
        elif t == 'tool/result':
            m = d['data']['message']
            msg = {'role': 'tool', 'tool_call_id': m.get('toolCallId', ''),
                   'content': blocks_text(m.get('content'))}
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1; cur.append(msg)
        elif t == 'compaction/prune':
            prunes.append(d['seq'])
            cur.append(('__prune__', d['seq']))
        if cur:
            at_turn_end = (t == 'step/end')
            if at_turn_end:
                steps.append(cur); cur = []
    if cur:
        steps.append(cur)
    return tools, steps, flat, seq_to_idx, prunes

def apply_prunes(history, seq_to_idx, prune_seqs, shadowed):
    """history: list of msgs (appended order). Remove shadowed tool results."""
    drop = set()
    for pseq in prune_seqs:
        idx = seq_to_idx.get(pseq)
        if idx is not None:
            drop.add(idx)
    return [m for i, m in enumerate(history) if i not in drop]

def openai_tools(tools):
    return [{'type': 'function', 'function': t} for t in (tools or [])]

def extract_cached(usage):
    """Recursively find the server-reported cached-token count."""
    found = {}
    def walk(u):
        if not isinstance(u, dict):
            return
        for k, v in u.items():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, int) and 'cache' in k:
                found[k] = v
    walk(usage)
    for k in ('cached_tokens', 'prompt_cache_hit_tokens', 'cache_read_input_tokens'):
        if k in found:
            return found[k]
    return max(found.values()) if found else None

def replay(args, tools, steps, flat, seq_to_idx, prune_seqs, shadowed_map):
    base = args.base.rstrip('/')
    model = args.model
    history = []
    pending_prunes = []
    report = []
    for si, step in enumerate(steps):
        for item in step:
            if isinstance(item, tuple) and item[0] == '__prune__':
                pending_prunes.append(item[1])
            else:
                history.append(item)
        if args.apply_prunes and pending_prunes:
            history = apply_prunes(history, seq_to_idx, pending_prunes, None)
            pending_prunes = []
        if not history:
            continue
        body = {'model': model, 'messages': history, 'max_tokens': 1,
                'temperature': 0, 'stream': False}
        if tools:
            body['tools'] = openai_tools(tools)
        req = urllib.request.Request(
            base + '/v1/chat/completions',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json',
                     **({'Authorization': 'Bearer ' + args.token} if args.token else {})})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                resp = json.loads(r.read().decode())
        except Exception as e:
            print('step %d: REQUEST FAILED: %s' % (si, e)); break
        dt = time.time() - t0
        u = resp.get('usage', {})
        prompt = u.get('prompt_tokens') or u.get('input_tokens') or 0
        cached = extract_cached(u)
        ratio = (cached / prompt) if (cached is not None and prompt) else None
        report.append({'step': si, 'msgs': len(history), 'prompt': prompt,
                       'cached': cached, 'ratio': ratio, 'secs': round(dt, 1)})
        print('step %3d  msgs=%4d  prompt=%7s  cached=%7s  ratio=%s  %5.1fs' % (
            si, len(history), prompt, cached,
            ('%.4f' % ratio) if ratio is not None else 'n/a', dt), flush=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    good = [r for r in report if r['ratio'] is not None and r['step'] > 0]
    if good:
        bad = [r for r in good if r['ratio'] < 0.95]
        print('\n%d measured steps, %d below 0.95 reuse' % (len(good), len(bad)))
        for r in bad[:20]:
            print('  miss: step %d prompt=%s cached=%s ratio=%.4f' % (
                r['step'], r['prompt'], r['cached'], r['ratio']))

def baseline(args, path):
    print('step  input   cacheRead  ratio   (recorded pre-fix DSH usage)')
    rows = []
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        if d['type'] != 'assistant/message':
            continue
        u = d['data'].get('usage') or {}
        rd = u.get('cacheReadTokens') or 0
        wr = u.get('cacheWriteTokens') or 0
        prompt = (rd + wr) or u.get('inputTokens') or u.get('totalTokens') or 0
        if not prompt:
            continue
        ratio = rd / prompt
        rows.append((d['data'].get('step'), prompt, rd, wr, ratio))
    for step, tot, rd, wr, ratio in rows:
        print('%4s  %7s  %8s  %.4f' % (step, tot, rd, ratio))
    bad = [r for r in rows if r[4] < 0.95]
    print('\n%d assistant steps, %d recorded below 0.95 cache reuse' % (len(rows), len(bad)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', required=True)
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--replay', action='store_true')
    ap.add_argument('--base', default='http://127.0.0.1:8080', help='ds4-server base url')
    ap.add_argument('--model', default='deepseek-v4.1-flash')
    ap.add_argument('--token', default='')
    ap.add_argument('--apply-prunes', action='store_true')
    ap.add_argument('--timeout', type=int, default=1800)
    ap.add_argument('--out', default='/tmp/reuse_replay_report.json')
    args = ap.parse_args()
    if args.baseline:
        baseline(args, args.session)
        return
    tools, steps, flat, seq_to_idx, prune_seqs = load_timeline(args.session)
    print('session: %d request steps, %d tools, %d prune events' %
          (len(steps), len(tools or []), len(prune_seqs)))
    if args.replay:
        replay(args, tools, steps, flat, seq_to_idx, prune_seqs, None)
    else:
        print('(dry run: pass --replay to hit the server)')

if __name__ == '__main__':
    main()
