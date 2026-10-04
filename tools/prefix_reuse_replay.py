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
import os
import shutil
import sys
import time
import urllib.request


def snapshot_unix_ms(snapshot_file):
    """Read the '# snapshot_unix_ms' header (written microseconds after the
    counters were copied under lock) or None."""
    try:
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            for _ in range(8):
                line = f.readline()
                if not line:
                    break
                if line.startswith('# snapshot_unix_ms'):
                    return int(line.split()[2])
    except OSError:
        pass
    return None


def wait_snapshot_after(snapshot_file, done_wall, timeout):
    """Wait for an engine snapshot whose counter copy happened after
    done_wall (the moment a replayed step's response returned; every routed
    selection of that step was recorded before the response, so a snapshot
    copied afterwards includes the whole step).  Returns the snapshot path.
    The engine writes tmp+rename, so the file is never half-visible."""
    deadline = time.time() + timeout
    done_ms = done_wall * 1000.0
    while time.time() < deadline:
        ms = snapshot_unix_ms(snapshot_file)
        if ms is not None and ms >= done_ms:
            return snapshot_file
        time.sleep(0.1)
    raise TimeoutError('snapshot %s not republished after step within %ds'
                       % (snapshot_file, timeout))

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
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1
            cur.append(('m', len(flat) - 1))
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
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1
            cur.append(('m', len(flat) - 1))
        elif t == 'tool/result':
            m = d['data']['message']
            msg = {'role': 'tool', 'tool_call_id': m.get('toolCallId', ''),
                   'content': blocks_text(m.get('content'))}
            flat.append(msg); seq_to_idx[d['seq']] = len(flat) - 1
            cur.append(('m', len(flat) - 1))
        elif t == 'compaction/prune':
            shadowed = d['data'].get('shadowedSeqs') or []
            prunes.extend(shadowed)
            cur.append(('__prune__', shadowed))
        if cur:
            at_turn_end = (t == 'step/end')
            if at_turn_end:
                steps.append(cur); cur = []
    if cur:
        steps.append(cur)
    return tools, steps, flat, seq_to_idx, prunes

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
    dropped = set()
    report = []
    for si, step in enumerate(steps):
        for item in step:
            if isinstance(item, tuple) and item[0] == '__prune__':
                if args.apply_prunes:
                    for s in item[1]:
                        idx = seq_to_idx.get(s)
                        if idx is not None:
                            dropped.add(idx)
            else:
                history.append(item[1])
        msgs = [flat[i] for i in history if i not in dropped]
        if not msgs:
            continue
        body = {'model': model, 'messages': msgs, 'max_tokens': 1,
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
        snap_ms = None
        if args.snapshot_file and args.snapshot_dir:
            wait_snapshot_after(args.snapshot_file, time.time(),
                                args.snapshot_timeout)
            snap_ms = snapshot_unix_ms(args.snapshot_file)
            os.makedirs(args.snapshot_dir, exist_ok=True)
            dst = os.path.join(args.snapshot_dir, 'turn_%03d.txt' % (si + 1))
            shutil.copyfile(args.snapshot_file, dst)
        u = resp.get('usage', {})
        prompt = u.get('prompt_tokens') or u.get('input_tokens') or 0
        cached = extract_cached(u)
        ratio = (cached / prompt) if (cached is not None and prompt) else None
        report.append({'step': si, 'msgs': len(msgs), 'prompt': prompt,
                       'cached': cached, 'ratio': ratio, 'secs': round(dt, 1),
                       'snap_ms': snap_ms})
        print('step %3d  msgs=%4d  prompt=%7s  cached=%7s  ratio=%s  %5.1fs' % (
            si, len(msgs), prompt, cached,
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
    ap.add_argument('--snapshot-file', default='',
                    help='engine selected-hotlist snapshot path '
                         '(DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT); after '
                         'each step, wait for the first snapshot copied '
                         'after the step ended and archive it under '
                         '--snapshot-dir as turn_NNN.txt (cumulative, for '
                         'tools/hotlist_v2_from_turns.py per-turn deltas)')
    ap.add_argument('--snapshot-dir', default='')
    ap.add_argument('--snapshot-timeout', type=int, default=60)
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
