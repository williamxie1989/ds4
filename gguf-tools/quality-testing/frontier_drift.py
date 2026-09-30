#!/usr/bin/env python3
"""Measure decision-relevant drift between two ds4-bench frontier dump dirs.

compare_frontier_logits.py requires identical metadata on both legs, so it
cannot compare the default path against the --quality scalar control.  This
script compares two dump directories (same frontiers, same prompt) and
reports, per frontier:

  - top-1 argmax id match (0-tolerance rule 1),
  - max |delta logit| restricted to the baseline top-20 (rule 2),
  - whole-table max |delta| and changed-value count (context),
  - float32 bit hashes for both legs.

It reads the writer's canonical %.9g binary32 spellings and repacks them with
struct, so drift is measured on exact stored float32 bits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path


def load_dump(path: Path):
    doc = json.loads(path.read_text())
    values = []
    words = bytearray()
    for index, value in enumerate(doc['logits']):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f'{path}: logits[{index}] is not finite')
        raw = struct.pack('<f', parsed)
        words.extend(raw)
        values.append(struct.unpack('<f', raw)[0])
    meta = {key: doc[key] for key in
            ('model', 'backend', 'quality', 'quant_bits', 'ctx', 'vocab',
             'prompt_tokens', 'frontier_tokens', 'prefill_tokens', 'argmax_id')}
    return meta, values, hashlib.sha256(words).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline')
    ap.add_argument('candidate')
    ap.add_argument('--frontiers', type=int, nargs='+', required=True)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--output', type=Path, help='write a JSON report here')
    args = ap.parse_args()

    baseline = Path(args.baseline)
    candidate = Path(args.candidate)
    rows = []
    for frontier in args.frontiers:
        name = f'frontier_{frontier:06d}.logits.json'
        b_meta, b_vals, b_sha = load_dump(baseline / name)
        c_meta, c_vals, c_sha = load_dump(candidate / name)
        if len(b_vals) != len(c_vals):
            raise ValueError(f'{name}: vocab mismatch')
        top_k = args.top_k
        top_ids = sorted(range(len(b_vals)), key=b_vals.__getitem__,
                         reverse=True)[:top_k]
        top1_match = b_meta['argmax_id'] == c_meta['argmax_id']
        max_topk = max(abs(b_vals[i] - c_vals[i]) for i in top_ids)
        max_all = 0.0
        changed = 0
        for x, y in zip(b_vals, c_vals):
            if x != y:
                changed += 1
                d = abs(x - y)
                if d > max_all:
                    max_all = d
        rows.append({
            'frontier': frontier,
            'baseline_quality': b_meta['quality'],
            'candidate_quality': c_meta['quality'],
            'top1_match': top1_match,
            'baseline_argmax_id': b_meta['argmax_id'],
            'candidate_argmax_id': c_meta['argmax_id'],
            'top20_max_abs_delta': max_topk,
            'full_max_abs_delta': max_all,
            'changed_count': changed,
            'vocab': len(b_vals),
            'baseline_logits_sha256': b_sha,
            'candidate_logits_sha256': c_sha,
        })

    for row in rows:
        print(f"frontier={row['frontier']:6d} top1_match={int(row['top1_match'])} "
              f"top20_max|d|={row['top20_max_abs_delta']:.3f} "
              f"full_max|d|={row['full_max_abs_delta']:.3f} "
              f"changed={row['changed_count']}/{row['vocab']}")
    if args.output:
        report = {'format_version': 1, 'top_k': args.top_k,
                  'baseline': str(baseline), 'candidate': str(candidate),
                  'frontiers': rows}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    worst_top1 = not all(row['top1_match'] for row in rows)
    return 1 if worst_top1 else 0


if __name__ == '__main__':
    sys.exit(main())
