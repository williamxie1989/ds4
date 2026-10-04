#!/usr/bin/env python3
"""Generate ds4_streaming_hotlist_v2.inc from per-turn hotlist snapshots.

M3-1 (R2a-1) offline calibration.  Input: ordered cumulative snapshots of
the engine's selected-id hotlist counters (v1 text format, written by the
ds4-server env DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT around each replayed
turn).  Snapshots are cumulative per (layer, expert) hit counters, so the
routing profile of turn i is the difference between consecutive snapshots.

For every layer we rank experts by cross-turn co-occurrence: the number of
turns in which the expert was selected at least once ("turns present"),
tie-broken by total selections, then by expert id.  Per layer we keep the
top-K, K = fraction * per-layer cache slots, for the three calibration
fractions (60/75/85 %).

Outputs:
  --out-inc PATH      C array in ds4_streaming_hotlist.inc format,
                      interleaved layer-round-robin so any truncation at
                      load time (max_entries) stays per-layer fair.
  --out-json PATH     full per-layer per-fraction tables + provenance.
  --selftest          synthetic-data unit checks, no server, no model.

Pure stdlib.  Turn-delta semantics: a decrease of any counter between two
consecutive snapshots means the files are out of order or from different
process generations (a fresh process restarts counters at 0): this is a
hard error, the caller must re-collect.
"""
import argparse
import json
import os
import sys
import tempfile

DEFAULT_FRACTIONS = (0.60, 0.75, 0.85)


def parse_cumulative(path):
    """Parse one v1-format hotlist/snapshot file.

    Returns (records, selections, {(layer, expert): hits}).  Unknown '#'
    header lines are ignored, same tolerance as the engine loader."""
    records = selections = None
    counts = {}
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                parts = s.split()
                if len(parts) >= 3 and parts[1] == 'layer_records':
                    records = int(parts[2])
                elif len(parts) >= 3 and parts[1] == 'selections':
                    selections = int(parts[2])
                continue
            parts = s.split()
            if len(parts) < 3:
                raise ValueError('%s:%d: expected "layer expert hits [weight]"'
                                 % (path, lineno))
            layer, expert, hits = int(parts[0]), int(parts[1]), int(parts[2])
            key = (layer, expert)
            counts[key] = counts.get(key, 0) + hits
    return records, selections, counts


def turn_deltas(cum_files):
    """Diff consecutive cumulative snapshots into per-turn deltas.

    Returns a list with one entry per turn:
      ({layer: {expert: turn_hits}}, header_selections, selection_delta)
    An expert missing from a layer's dict did not appear in that turn."""
    turns = []
    prev = {}
    prev_sel = 0
    for path in cum_files:
        _, selections, counts = parse_cumulative(path)
        delta = {}
        for key, hits in counts.items():
            d = hits - prev.get(key, 0)
            if d < 0:
                raise ValueError(
                    '%s: counter (%d,%d) decreased by %d vs previous '
                    'snapshot: files out of order or process restarted '
                    'mid-sequence' % (path, key[0], key[1], -d))
            if d > 0:
                layer, expert = key
                delta.setdefault(layer, {})[expert] = d
        sel_delta = (selections - prev_sel
                     if selections is not None else None)
        turns.append((delta, selections, sel_delta))
        prev = counts
        if selections is not None:
            prev_sel = selections
    return turns


def rank_layers(turns):
    """Per layer: {expert: dict(turns_present, total_hits)} and per-layer
    turn counts."""
    per_layer = {}
    layer_turns = {}
    for norm, _, _ in turns:
        for layer, exps in norm.items():
            acc = per_layer.setdefault(layer, {})
            layer_turns[layer] = layer_turns.get(layer, 0) + 1
            for expert, hits in exps.items():
                st = acc.setdefault(expert, {'turns_present': 0,
                                             'total_hits': 0})
                st['turns_present'] += 1
                st['total_hits'] += hits
    return per_layer, layer_turns


def topk(per_layer, layer_turns, n_layers, cache_experts, fraction):
    """Ranked top-K per layer for one fraction.  K uses floor(fraction *
    slots_per_layer) slots (never more than the observed candidate count)."""
    slots_per_layer = max(1, int(cache_experts // n_layers))
    k = max(1, int(fraction * slots_per_layer))
    out = {}
    for layer, acc in per_layer.items():
        ranked = sorted(
            acc.items(),
            key=lambda kv: (-kv[1]['turns_present'], -kv[1]['total_hits'],
                            kv[0]))
        out[layer] = [(e, st['turns_present'], st['total_hits'],
                       layer_turns.get(layer, 0))
                      for e, st in ranked[:k]]
    return out, k, slots_per_layer


def emit_inc(path, tables_by_fraction, fraction_for_inc, model_tag):
    """Write the .inc with layer-round-robin interleaving of the chosen
    fraction's top-K tables."""
    table = tables_by_fraction[fraction_for_inc]
    layers = sorted(table.keys())
    pairs = []
    depth = 0
    while True:
        added = False
        for layer in layers:
            rows = table[layer]
            if depth < len(rows):
                pairs.append((layer, rows[depth][0]))
                added = True
        if not added:
            break
        depth += 1
    name = 'ds4_default_streaming_hotlist_v2'
    with open(path, 'w', encoding='utf-8') as f:
        f.write('/* Generated by tools/hotlist_v2_from_turns.py from '
                'replayed-session selected-id hotlist snapshots;\n'
                ' * cross-turn co-occurrence ranking, per-layer top-K, '
                'layer-round-robin interleaved so truncation at\n'
                ' * load time stays per-layer fair. */\n')
        f.write('/* %s streaming expert hotlist v2 (fraction %.2f). */\n'
                % (model_tag, fraction_for_inc))
        f.write('static const uint16_t %s[][2] = {\n' % name)
        for layer, expert in pairs:
            f.write('    {%d, %d},\n' % (layer, expert))
        f.write('};\n')
        f.write('static const uint32_t %s_count =\n'
                '    (uint32_t)(sizeof(%s) /\n'
                '               sizeof(%s[0]));\n' % (name, name, name))
    return pairs


def run(args):
    if args.files_from:
        # explicit manifest: one path per line, capture order (the harness
        # writes it after each turn, so order is guaranteed by construction)
        with open(args.files_from) as f:
            cum_files = [l.strip() for l in f if l.strip()]
    else:
        # positional args: rely on the harness naming (zero-padded turn
        # numbers); use --files-from when names are free-form
        cum_files = sorted(args.cumulative)
    if len(cum_files) < 2:
        sys.exit('need at least 2 cumulative snapshot files '
                 '(first turn + a later one)')
    turns = turn_deltas(cum_files)
    per_layer, layer_turns = rank_layers(turns)
    if not per_layer:
        sys.exit('no routing records found in snapshots')
    n_layers = args.layers if args.layers else max(per_layer) + 1
    fractions = tuple(args.fraction) if args.fraction else DEFAULT_FRACTIONS
    tables = {}
    meta = {}
    for fr in fractions:
        table, k, slots = topk(per_layer, layer_turns, n_layers,
                               args.cache_experts, fr)
        tables[fr] = table
        meta[str(fr)] = {'fraction': fr, 'k_per_layer': k,
                         'slots_per_layer': slots,
                         'layers': len(table)}
    if args.out_json:
        doc = {'inputs': cum_files, 'turns': len(turns),
               'cache_experts': args.cache_experts, 'layers': n_layers,
               'fractions': meta,
               'tables': {str(fr): {str(layer): table
                                    for layer, table in tables[fr].items()}
                          for fr in fractions}}
        with open(args.out_json, 'w') as f:
            json.dump(doc, f, indent=1)
        print('wrote %s' % args.out_json)
    if args.out_inc:
        pairs = emit_inc(args.out_inc, tables, args.fraction_for_inc,
                         args.model_tag)
        print('wrote %s (%d pairs, fraction %.2f)'
              % (args.out_inc, len(pairs), args.fraction_for_inc))
    return tables, meta


# --------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------

def _write_cum(path, counts, selections):
    with open(path, 'w') as f:
        f.write('# ds4 selected-id hotlist v1\n'
                '# layer_records %d\n'
                '# selections %d\n'
                '# snapshot_seq 1\n'
                '# snapshot_unix_ms 0\n'
                '# columns: layer expert hits weight\n' % (selections // 6,
                                                           selections))
        for (layer, expert), hits in sorted(counts.items()):
            if hits:
                f.write('%d %d %d 0\n' % (layer, expert, hits))


def selftest():
    ok = 0
    tmpdir = tempfile.mkdtemp(prefix='hotlist_v2_selftest_')
    # Scenario: 2 layers.  Layer 0: expert 10 in every turn (heavy),
    # expert 11 in half the turns, experts 20..25 sparse singles.
    # Layer 1: expert 7 always; expert 8 never after turn 1.
    T = 6
    cums = []
    cum = {}
    running_sel = 0
    for t in range(1, T + 1):
        wave = {}
        wave[(0, 10)] = 4
        if t % 2 == 0:
            wave[(0, 11)] = 2
        if t <= 2:
            wave[(1, 7)] = 3
            wave[(1, 8)] = 5
        if t == 3:
            wave[(0, 21)] = 1
        if t == 5:
            wave[(0, 22)] = 1
        for k, v in wave.items():
            cum[k] = cum.get(k, 0) + v
        running_sel += 6 * T  # records grow uniformly; only header bookkeeping
        p = os.path.join(tmpdir, 'cum_%03d.txt' % t)
        _write_cum(p, cum, running_sel)
        cums.append(p)

    turns = turn_deltas(cums)
    assert len(turns) == T, 'turn count'
    # turn2 layer0: experts 10,11; turn3: 10,21; turn1: 10,(1,7),(1,8)
    assert turns[1][0].get(0) == {10: 4, 11: 2}, turns[1][0]
    assert turns[2][0].get(0) == {10: 4, 21: 1}, turns[2][0]
    assert turns[0][0].get(1) == {7: 3, 8: 5}, turns[0][0]
    ok += 1

    per_layer, layer_turns = rank_layers(turns)
    # expert 10 present all 6 turns, > all others in layer 0
    l0 = sorted(per_layer[0].items(),
                key=lambda kv: (-kv[1]['turns_present'], kv[0]))
    assert l0[0][0] == 10 and l0[0][1]['turns_present'] == 6, l0[:3]
    assert per_layer[0][11]['turns_present'] == 3, 'expert 11 in 3 turns'
    assert per_layer[1][7]['turns_present'] == 2, 'expert 7 in 2 turns'
    assert layer_turns[0] == T, 'layer 0 has records in all 6 turns'
    ok += 1

    # K selection: cache 80 experts over 2 layers -> 40 slots/layer;
    # 60% -> 24 (larger than candidate set, so full set),
    # force tight: cache_experts=4 -> 2 slots/layer, 0.75 -> k=1
    table, k, slots = topk(per_layer, layer_turns, 2, 4, 0.75)
    assert slots == 2 and k == 1, (slots, k)
    assert table[0] == [(10, 6, per_layer[0][10]['total_hits'], 6)], table[0]
    ok += 1

    # inc emission: round-robin fairness + parse back
    inc = os.path.join(tmpdir, 'v2.inc')
    tables = {0.60: topk(per_layer, layer_turns, 2, 4, 0.60)[0],
              0.75: topk(per_layer, layer_turns, 2, 4, 0.75)[0],
              0.85: topk(per_layer, layer_turns, 2, 4, 0.85)[0]}
    pairs = emit_inc(inc, tables, 0.60, 'Test')
    text = open(inc).read()
    assert 'ds4_default_streaming_hotlist_v2[][2]' in text
    assert 'ds4_default_streaming_hotlist_v2_count' in text
    layers_order = [p[0] for p in pairs]
    # first pass must cover both layers before either layer repeats
    assert len(set(layers_order[:2])) == 2, layers_order
    ok += 1

    # out-of-order snapshot detection
    bad = os.path.join(tmpdir, 'bad.txt')
    _write_cum(bad, {(0, 10): 1}, 6)  # lower than previous cum
    try:
        turn_deltas(cums + [bad])
        raise AssertionError('expected ValueError on decreasing counter')
    except ValueError:
        pass
    ok += 1

    # header tolerance: unknown snapshot headers ignored, selections parsed
    recs, sels, counts = parse_cumulative(cums[-1])
    assert sels == running_sel and counts[(0, 10)] == 4 * T
    ok += 1

    print('hotlist_v2_from_turns selftest: %d/6 groups PASS' % ok)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cumulative', nargs='*',
                    help='ordered cumulative snapshot/hotlist files '
                         '(one per turn boundary; sorted order)')
    ap.add_argument('--files-from',
                    help='file listing cumulative snapshot paths, one per '
                         'line (overrides positional args)')
    ap.add_argument('--cache-experts', type=int, required=False,
                    help='streaming expert cache capacity in experts '
                         '(e.g. 6959 for auto V4.1)')
    ap.add_argument('--layers', type=int, default=0,
                    help='total routed layers (default: max seen + 1)')
    ap.add_argument('--fraction', type=float, action='append',
                    help='fraction(s) of per-layer slots to keep; repeat '
                         'for tiers (default 0.60 0.75 0.85)')
    ap.add_argument('--fraction-for-inc', type=float, default=0.75)
    ap.add_argument('--out-inc')
    ap.add_argument('--out-json')
    ap.add_argument('--model-tag', default='DeepSeek V4.1 Flash')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.cache_experts:
        sys.exit('--cache-experts is required')
    run(args)


if __name__ == '__main__':
    main()
