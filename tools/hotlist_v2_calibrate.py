#!/usr/bin/env python3
"""Paper hit-gain calibration for hotlist v2 (M3-1 / R2a-1).

Inputs are the same ordered cumulative selected-hotlist snapshots the
generator consumes (one per turn boundary).  For every layer we simulate,
offline, the expert-cache admission policy over the replayed turns and
measure the *next-turn gather* hit rate:

  baseline   pure LRU with the full per-layer slot budget C.  At prefill
             gather time the whole turn's unique expert set is fetched
             upfront (batch pread), so a turn's hit rate is measured
             against the cache state *before* the turn, not while it
             fills.  Within-turn insertion order is approximated by
             descending selection count (we have counts, not the exact
             record order; the approximation is one-sided only where a
             turn overflows the cache, and both arms share it).

  protected  K = fraction*C slots pinned to the hotlist-v2 candidate set
             (cross-turn co-occurrence top-K, M3-2's protected priority),
             the remaining C-K slots run the same LRU.  Candidate sets are
             rebuilt leave-one-out from the prefix turns 1..t-1 before
             each turn t, so the number is an honest generalization
             estimate, not an oracle overfit to the replayed turns.
             (The engine will load a single offline-generated hotlist; the
             leave-one-out curve is the unbiased proxy for that.)

Report: per layer and aggregate (micro-averaged over unique
(layer, turn) gather requests) hit rates and the delta in percentage
points, per fraction tier.  Gate for the task: aggregate gain >= 25 pp at
one of the three tiers.

--selftest runs synthetic-data checks of the simulator, no server, no model.
Pure stdlib.
"""
import argparse
import json
import os
import random
import sys
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hotlist_v2_from_turns as hv2  # noqa: E402

DEFAULT_FRACTIONS = (0.60, 0.75, 0.85)


class LRU:
    """Set-membership LRU with ordered apply (MRU last)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.state = OrderedDict()

    def touch(self, expert):
        if expert in self.state:
            self.state.move_to_end(expert)
            return True
        if self.capacity <= 0:
            return False
        self.state[expert] = True
        if len(self.state) > self.capacity:
            self.state.popitem(last=False)
        return False

    def __contains__(self, expert):
        return expert in self.state


def apply_turn_count(lru, turn_exps):
    """Count next-turn gather hits against the pre-turn state, then insert
    the turn's experts (MRU last) in descending selection-count order
    (ties by id).  Insertion order within a turn is an approximation: we
    have per-turn counts, not the exact record order; both arms share it."""
    hits = count_hits(turn_exps, lru, set())
    order = sorted(turn_exps.keys(), key=lambda e: (-turn_exps[e], e))
    for e in order:
        lru.touch(e)
    return hits


def _k(slots, fraction):
    # same K formula as the generator's topk()
    return min(slots, max(1, int(fraction * slots)))


def _prefix_protected(turn_layers_prefix, k):
    """Cross-turn co-occurrence top-K (set) for one layer from the prefix.
    Ranking: (turns_present desc, total_hits desc, expert asc) — identical
    to hotlist_v2_from_turns.rank_layers + topk."""
    stat = {}
    for exps in turn_layers_prefix:
        for e, h in exps.items():
            st = stat.setdefault(e, [0, 0])
            st[0] += 1
            st[1] += h
    ranked = sorted(stat.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
    return set(e for e, _ in ranked[:k])


def count_hits(turn_exps, lru, protected):
    return sum(1 for e in turn_exps
               if e in protected or e in lru)


def _apply_protected(lru, turn_exps, protected):
    """The non-protected slots see everything the turn selected that is
    not covered by the protected pins (the pins never age out)."""
    order = sorted((e for e in turn_exps if e not in protected),
                   key=lambda e: (-turn_exps[e], e))
    for e in order:
        lru.touch(e)


def simulate_layer(turn_layers, slots, fractions):
    """turn_layers: list of {expert: hits} per turn for one layer (empty
    dicts for turns without records).  Returns per-fraction aggregate hit
    counters over measured turns (t >= 2)."""
    n_turns = len(turn_layers)
    base_hits = base_total = 0
    lru_base = LRU(slots)
    arms = {fr: {'hits': 0, 'total': 0} for fr in fractions}
    lrus = {fr: LRU(max(0, slots - _k(slots, fr))) for fr in fractions}
    per_turn = []
    for t, turn_exps in enumerate(turn_layers):
        row = {'turn': t + 1, 'unique': len(turn_exps)}
        if turn_exps:
            bh = apply_turn_count(lru_base, turn_exps)
        else:
            bh = 0
        if t > 0 and turn_exps:
            base_hits += bh
            base_total += len(turn_exps)
            row['baseline'] = bh / len(turn_exps)
            for fr in fractions:
                prot = _prefix_protected(turn_layers[:t], _k(slots, fr))
                ph = count_hits(turn_exps, lrus[fr], prot)
                arms[fr]['hits'] += ph
                arms[fr]['total'] += len(turn_exps)
                row['protected_%d' % round(fr * 100)] = ph / len(turn_exps)
                _apply_protected(lrus[fr], turn_exps, prot)
        else:
            # first turn: cold start, not measured; still warms both arms
            for fr in fractions:
                _apply_protected(lrus[fr], turn_exps, set())
        per_turn.append(row)
    out = {'turns': n_turns,
           'measured_sets': base_total,
           'baseline_hits': base_hits,
           'baseline_rate': (base_hits / base_total) if base_total else None,
           'per_turn': per_turn}
    for fr in fractions:
        rate = (arms[fr]['hits'] / arms[fr]['total']
                if arms[fr]['total'] else None)
        out['protected_%d' % round(fr * 100)] = {
            'fraction': fr, 'k_per_layer': _k(slots, fr),
            'hits': arms[fr]['hits'], 'total': arms[fr]['total'],
            'rate': rate,
            'delta_pp': ((rate - out['baseline_rate']) * 100.0
                         if rate is not None and out['baseline_rate']
                         is not None else None)}
    return out


def _apply_protected(lru, turn_exps, protected):
    """The non-protected slots see everything the turn selected that is
    not covered by the protected pins (the pins never age out)."""
    order = sorted((e for e in turn_exps if e not in protected),
                   key=lambda e: (-turn_exps[e], e))
    for e in order:
        lru.touch(e)


def run(args):
    if args.files_from:
        with open(args.files_from) as f:
            cum_files = [l.strip() for l in f if l.strip()]
    else:
        cum_files = sorted(args.cumulative)
    if len(cum_files) < 2:
        sys.exit('need at least 2 cumulative snapshot files')
    turns = hv2.turn_deltas(cum_files)
    per_layer, _ = hv2.rank_layers(turns)
    n_layers = args.layers if args.layers else max(per_layer) + 1
    slots = max(1, args.cache_experts // n_layers)
    fractions = tuple(args.fraction) if args.fraction else DEFAULT_FRACTIONS

    layers_report = {}
    agg_base = {'hits': 0, 'total': 0}
    agg_prot = {fr: {'hits': 0, 'total': 0} for fr in fractions}
    for layer in sorted(per_layer):
        turn_layers = [d.get(layer, {}) for d, _, _ in turns]
        rep = simulate_layer(turn_layers, slots, fractions)
        layers_report[str(layer)] = rep
        agg_base['hits'] += rep['baseline_hits']
        agg_base['total'] += rep['measured_sets']
        for fr in fractions:
            agg_prot[fr]['hits'] += rep['protected_%d' % round(fr * 100)]['hits']
            agg_prot[fr]['total'] += rep['protected_%d' % round(fr * 100)]['total']

    base_rate = (agg_base['hits'] / agg_base['total']
                 if agg_base['total'] else 0.0)
    agg = {'baseline_rate': base_rate,
           'measured_sets': agg_base['total'],
           'slots_per_layer': slots, 'layers': len(layers_report),
           'turns': len(turns)}
    best = None
    for fr in fractions:
        rate = (agg_prot[fr]['hits'] / agg_prot[fr]['total']
                if agg_prot[fr]['total'] else None)
        delta = (rate - base_rate) * 100.0 if rate is not None else None
        agg['protected_%d' % round(fr * 100)] = {
            'fraction': fr, 'rate': rate, 'delta_pp': delta}
        if delta is not None and (best is None or delta > best[1]):
            best = (fr, delta)

    doc = {'inputs': cum_files, 'aggregate': agg,
           'layers': layers_report,
           'gate_pp': args.gate_pp,
           'gate_pass': best is not None and best[1] >= args.gate_pp,
           'best_fraction': best[0] if best else None,
           'best_delta_pp': best[1] if best else None}
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(doc, f, indent=1)
        print('wrote %s' % args.out_json)
    if args.out_md:
        write_md(args.out_md, doc)
        print('wrote %s' % args.out_md)
    # console summary
    print('aggregate baseline %.4f (%d measured expert-requests)'
          % (base_rate, agg_base['total']))
    for fr in fractions:
        a = agg['protected_%d' % round(fr * 100)]
        if a['rate'] is not None:
            print('  protected K=%d%%: %.4f  (delta %+.1f pp)'
                  % (round(fr * 100), a['rate'], a['delta_pp']))
    print('gate >= %g pp: %s' % (args.gate_pp,
                                 'PASS' if doc['gate_pass'] else 'FAIL'))
    return doc


def write_md(path, doc):
    agg = doc['aggregate']
    with open(path, 'w') as f:
        f.write('# M3-1 hotlist v2 paper hit-gain calibration\n\n')
        f.write('- inputs: %s\n' % ', '.join(
            os.path.basename(p) for p in doc['inputs']))
        f.write('- turns: %d, layers with records: %d, '
                'slots/layer: %d, measured expert-requests: %d\n\n'
                % (agg['turns'], agg['layers'], agg['slots_per_layer'],
                   agg['measured_sets']))
        f.write('## Aggregate (micro-averaged over (layer, turn) gathers)\n\n')
        f.write('| arm | hit rate | delta pp |\n|---|---:|---:|\n')
        f.write('| baseline (LRU) | %.4f | — |\n' % agg['baseline_rate'])
        for key in sorted(k for k in agg if k.startswith('protected_')):
            a = agg[key]
            f.write('| %s | %s | %s |\n'
                    % (key,
                       '%.4f' % a['rate'] if a['rate'] is not None else 'n/a',
                       '%+.1f' % a['delta_pp']
                       if a['delta_pp'] is not None else 'n/a'))
        f.write('\ngate >= %g pp: **%s**'
                % (doc['gate_pp'], 'PASS' if doc['gate_pass'] else 'FAIL'))
        if doc['best_fraction'] is not None:
            f.write(' (best K=%d%%, %+.1f pp)'
                    % (round(doc['best_fraction'] * 100),
                       doc['best_delta_pp']))
        f.write('\n\n## Per layer\n\n')
        f.write('| layer | turns | mean unique/turn | baseline |')
        frs = sorted(k.split('_')[1] for k in agg if k.startswith('protected_'))
        for fr in frs:
            f.write(' prot-%s%% | d-pp-%s |' % (fr, fr))
        f.write('\n|---|---:|---:|---:|')
        for _ in frs:
            f.write('---:|---:|')
        f.write('\n')
        for lkey in sorted(doc['layers'], key=int):
            rep = doc['layers'][lkey]
            if rep['measured_sets'] == 0:
                continue
            mean_u = (sum(r['unique'] for r in rep['per_turn'])
                      / max(1, len(rep['per_turn'])))
            f.write('| %s | %d | %.0f | %.4f |'
                    % (lkey, rep['turns'], mean_u,
                       rep['baseline_rate'] or 0.0))
            for fr in frs:
                p = rep.get('protected_%s' % fr)
                if p and p['rate'] is not None:
                    f.write(' %.4f | %+.1f |' % (p['rate'], p['delta_pp']))
                else:
                    f.write(' n/a | n/a |')
            f.write('\n')


# --------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------

def selftest():
    ok = 0
    # --- hand-computed LRU: capacity 2, turns {1},{2},{1},{3}, hits
    # measured against the pre-turn state:
    # before t1 {} -> 0; before t2 {1} -> 0; before t3 {1,2} -> 1;
    # before t4 {2,1} -> 0  =>  [0, 0, 1, 0]
    lru = LRU(2)
    seq = [{1: 1}, {2: 1}, {1: 1}, {3: 1}]
    counts = [apply_turn_count(lru, s) for s in seq]
    assert counts == [0, 0, 1, 0], counts
    ok += 1

    # --- heavy-core scenario: 150 slots; 60 core experts every turn + 180
    # rotating (284 space) => 240 unique/turn > slots, LRU thrashes the
    # core every turn.  Protected top-K must recover the core.
    rng = random.Random(1234)
    core = list(range(60))
    turns_seq = []
    for t in range(8):
        turn = {e: 3 for e in core}
        for e in rng.sample(range(100, 384), 180):
            turn[e] = 1
        turns_seq.append(turn)
    rep = simulate_layer(turns_seq, slots=150, fractions=(0.6,))
    assert rep['baseline_rate'] is not None
    prot = rep['protected_60']
    assert prot['delta_pp'] > 5.0, (rep['baseline_rate'], prot['rate'])
    ok += 1

    # --- uniform-random traffic: protection must not fabricate gains.
    # Each turn draws 200 of 384 uniformly; with 174 slots next-turn hit
    # rate ~ |S∩cache|/|S|; protected top-K over uniform noise should be
    # within a few pp of baseline (ranking is pure noise).
    rng = random.Random(99)
    turns_rand = []
    for t in range(6):
        picks = rng.sample(range(384), 200)
        turns_rand.append({e: 2 for e in picks})
    rep2 = simulate_layer(turns_rand, slots=174, fractions=(0.75,))
    d = rep2['protected_75']['delta_pp']
    assert abs(d) < 5.0, d
    ok += 1

    # --- leave-one-out integrity: first measured turn has empty candidate
    # set -> protected arm cannot out-hit baseline there.
    turns3 = [{i: 1 for i in range(300)}] * 3
    rep3 = simulate_layer(turns3, slots=100, fractions=(0.85,))
    r0 = rep3['per_turn'][1]
    assert r0['baseline'] == r0['protected_85'], r0
    ok += 1

    # --- first turn never measured (cold start excluded)
    assert rep3['per_turn'][0]['turn'] == 1
    assert 'baseline' not in rep3['per_turn'][0]
    ok += 1

    # --- end-to-end on synthetic cumulative files (parse path shared
    # with the generator): build 4 turns, run run(), sanity-check output
    tmpdir = tempfile.mkdtemp(prefix='hotlist_cal_selftest_')
    cum = {}
    files = []
    sel = 0
    for t in range(1, 5):
        for e in range(40):
            cum[(0, e)] = cum.get((0, e), 0) + 2  # core layer 0
        for e in rng.sample(range(100, 384), 60):
            cum[(0, e)] = cum.get((0, e), 0) + 1
        for e in range(30):
            cum[(1, e)] = cum.get((1, e), 0) + 1
        sel += 66 * 6
        p = os.path.join(tmpdir, 'cum_%03d.txt' % t)
        hv2._write_cum(p, cum, sel)
        files.append(p)
    import argparse as _ap
    args = _ap.Namespace(files_from=None, cumulative=files,
                         cache_experts=200, layers=2,
                         fraction=[0.75], out_json=os.path.join(
                             tmpdir, 'out.json'),
                         out_md=os.path.join(tmpdir, 'out.md'),
                         gate_pp=25.0)
    doc = run(args)
    assert doc['aggregate']['layers'] == 2
    assert doc['aggregate']['slots_per_layer'] == 100
    assert os.path.exists(args.out_json) and os.path.exists(args.out_md)
    # core covers 40 > K=75*100*0.75=75? K=75 slots... core=40<=75: gain
    ok += 1

    print('hotlist_v2_calibrate selftest: %d/6 groups PASS' % ok)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cumulative', nargs='*')
    ap.add_argument('--files-from')
    ap.add_argument('--cache-experts', type=int)
    ap.add_argument('--layers', type=int, default=0)
    ap.add_argument('--fraction', type=float, action='append')
    ap.add_argument('--gate-pp', type=float, default=25.0)
    ap.add_argument('--out-json')
    ap.add_argument('--out-md')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.cache_experts:
        sys.exit('--cache-experts is required')
    run(args)


if __name__ == '__main__':
    main()
