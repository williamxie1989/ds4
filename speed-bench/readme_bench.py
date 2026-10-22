#!/usr/bin/env python3
"""The README speed tables, every column measured in one session, every run from the same temperature.

    python3 speed-bench/readme_bench.py sweep  --a /tmp/ds4-base --b .
    python3 speed-bench/readme_bench.py shapes --a . --b . --kinds decode,cold

`sweep` is the "Against ds4" table: upstream's bench sweep (2048 to 32768 doubling, 128
tokens) for each build tree. `shapes` is the A/B harness's prompt, append and decode
shapes for the same trees.

The protocol (ported from sf-ds4-1flash, speed-bench/README.md, "README tables"):
- one block per build, each after a discarded warm-up run of that build, so every
  measured run follows a run of the same build: the first frontier of a run depends on
  the page cache the previous process left;
- the cool-down target is the chip's lowest reading in 60 s of idle after the warm-up,
  plus 2 C: a temperature the machine reaches after a hot run, not an idle floor it
  cannot (needs mactop; without it the harness waits a fixed --settle seconds);
- before every measured run, wait until the chip is at or below the target, then wait
  for thermal pressure to read Nominal again;
- inside a block the columns are rotated (A B, B A, ...), --runs runs each, and every
  value is a median;
- chip temperature and thermal state come from mactop every 2 s.

Not ported: the fork's fanboost (`--boost`) columns, its second-drive replica columns and
`fanboost pause/release` steps. They need hardware and root daemons this tree does not
assume; the columns here are the build trees alone.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_bench as ab  # noqa: E402

SWEEP = ['--ctx-start', '2048', '--ctx-max', '32768', '--step-mul', '2', '--gen-tokens', '128']
FRONTIERS = (2048, 4096, 8192, 16384, 32768)
SHAPES = ['decode', 'cold-5000', 'append']
SETTLE_SECONDS = 60
MARGIN_C = 2.0
COOL_TIMEOUT = 420


def chip_reading():
    """(chip C, thermal state) from one mactop sample, or (None, None) without mactop."""
    if not ab.mactop_available():
        return None, None
    out = subprocess.run(['mactop', '--headless', '--count', '1', '--interval', '1000'],
                         capture_output=True, text=True, timeout=30).stdout
    samples = ab.parse_mactop(out)
    if not samples:
        raise ab.Stop(2, 'mactop returned no reading')
    r = ab.reading(samples[0], ab.mactop_version())
    return r['temp'], r['thermal']


def settle_temperature():
    """The chip's lowest reading in SETTLE_SECONDS of idle: the floor a hot run returns to."""
    low, end = 1e9, time.time() + SETTLE_SECONDS
    while time.time() < end:
        temp, _ = chip_reading()
        if temp is not None:
            low = min(low, temp)
        time.sleep(2)
    return low if low < 1e9 else None


def cool_down(target, settle):
    start = time.time()
    if target is None:
        time.sleep(max(0.0, settle))
        return {'cool_s': round(time.time() - start), 'cool_reached': None}
    reached = False
    while time.time() - start < COOL_TIMEOUT:
        temp, thermal = chip_reading()
        if temp is not None and temp <= target and thermal == 'Nominal':
            reached = True
            break
        time.sleep(2)
    return {'cool_s': round(time.time() - start), 'cool_reached': reached}


def command(table, tree, kind, csv_path, model):
    if table == 'sweep':
        return [str(tree / ab.BENCH), '-m', str(model), '--prompt-file', str(ab.ROOT / ab.PROMPT),
                '--ssd-streaming', '--ctx-alloc', str(ab.CTX_ALLOC), *SWEEP,
                '--show-output', '--cache-stats', '--csv', str(csv_path)]
    return ab.bench_cmd(tree, kind, model, csv_path)


def measure(table, tree, name, kind, out, n, model, target, settle):
    cond = cool_down(target, settle)
    stem = out / 'logs' / f'{n:02d}-{name}-{kind}'
    start = time.time()
    with open(f'{stem}.err', 'wb') as err:
        rc = subprocess.run(command(table, tree, kind, f'{stem}.csv', model), cwd=tree,
                            env=ab.child_env([]), stdout=subprocess.DEVNULL, stderr=err).returncode
    end = time.time()
    if rc != 0:
        raise ab.Stop(1, f'{name} {kind} failed (exit {rc}); see {stem}.err')
    loader = ab.LOADER.search(Path(f'{stem}.err').read_text(errors='replace'))
    if not loader:
        raise ab.Stop(2, f'{name} {kind}: no SSD streaming cache line; see {stem}.err')
    rows = {int(r['ctx_tokens']): r for r in csv.DictReader(open(f'{stem}.csv'))}
    temp, thermal = chip_reading()
    print(f'  {n:2d} {name:<14} {kind:<11} {end - start:5.0f} s  cooled {cond["cool_s"]:3d} s'
          f'{"" if cond["cool_reached"] is not False else " (timeout)"}, '
          f'cache {loader[1]} GiB, end {temp if temp is not None else "?"} C {thermal or ""}', flush=True)
    return {'n': n, 'column': name, 'kind': kind, 'rows': rows, 'conditions': cond}


def values(table, run):
    rows = run['rows']
    if table == 'sweep':
        v = {}
        for f in FRONTIERS:
            v[f'prefill {f}'] = float(rows[f]['prefill_tps'])
            v[f'generation {f}'] = float(rows[f]['gen_tps'])
            v[f'steady decode {f}'] = float(rows[f]['gen_steady_tps'])
            v[f'first token {f} ms'] = float(rows[f]['gen_first_ms'])
        return v
    v = ab.metrics(run['kind'], rows)
    if run['kind'] == 'decode':
        v['256 tokens after 8192 s'] = int(rows[8192]['gen_tokens']) / float(rows[8192]['gen_tps'])
    return v


def higher_is_better(metric):
    return metric.startswith(('prefill ', 'generation', 'steady decode')) and 'ms' not in metric \
        or metric.startswith('decode ')


def summarize(table, columns, runs):
    per = {c: {} for c in columns}
    for run in runs:
        for k, v in values(table, run).items():
            if not (table == 'shapes' and k == 'prefill 5000'):
                per[run['column']].setdefault(k, []).append(v)
    base = columns[0]
    lines = [f'| Metric | {base} | ' + ' | '.join(f'{c} | vs {base}' for c in columns[1:]) + ' |',
             '|---|' + '---:|' * (2 * len(columns) - 1)]
    for metric in per[base]:
        b = statistics.median(per[base][metric])
        cells = [f'{b:.1f}']
        for c in columns[1:]:
            x = statistics.median(per[c][metric])
            gain = x / b - 1 if higher_is_better(metric) else b / x - 1
            cells += [f'{x:.1f}', f'{gain * 100:+.1f}%']
        lines.append(f'| {metric} | ' + ' | '.join(cells) + ' |')
    spread = ['', 'Spread (max-min)/median per column, worst metric:']
    for c in columns:
        worst = max(((max(v) - min(v)) / statistics.median(v), m) for m, v in per[c].items())
        spread.append(f'- {c}: {worst[0] * 100:.1f}% ({worst[1]})')
    return '\n'.join(lines + spread)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('table', choices=('sweep', 'shapes'))
    p.add_argument('--a', required=True, help='baseline build tree')
    p.add_argument('--b', help='candidate build tree (default: only --a)')
    p.add_argument('--kinds', default=','.join(SHAPES), help='shapes only: comma list of ab_bench kinds')
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--settle', type=float, default=30.0,
                   help='seconds to wait between runs when mactop is absent (default 30)')
    p.add_argument('-m', '--model', default=str(ab.ROOT / 'gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf'))
    p.add_argument('--out', default=None, help='output directory (default $TMPDIR/ds4-readme/<table>-<time>)')
    args = p.parse_args(argv)
    trees = {'A': Path(args.a).resolve()}
    columns = ['A']
    if args.b:
        trees['B'] = Path(args.b).resolve()
        columns.append('B')
    kinds = ab.select_kinds(args.kinds, '--kinds') if args.table == 'shapes' else ['sweep']
    for name, tree in trees.items():
        if not (tree / 'Makefile').is_file() or not (tree / 'metal').is_dir():
            raise ab.Stop(2, f'{tree}: not a build tree (needs a Makefile and metal/)')
        if subprocess.run(['make', '-C', str(tree), '-j8', ab.BENCH], capture_output=True).returncode:
            raise ab.Stop(2, f'make {ab.BENCH} failed in {tree}')
    model = Path(os.path.abspath(args.model))
    if not model.is_file():
        raise ab.Stop(2, f'{args.model}: model not found')
    ab.preflight(60.0)
    out = (Path(args.out).resolve() if args.out else
           Path(os.environ.get('TMPDIR', '/tmp')) / 'ds4-readme'
           / f'{args.table}-{datetime.now():%Y%m%d-%H%M}')
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    runs, n, target = [], 1, None
    try:
        for name in columns:
            # one block per build, so no run of one build follows a run of the other
            measure(args.table, trees[name], name, kinds[0], out, 0, model, target, args.settle)
            if target is None and ab.mactop_available():
                floor = settle_temperature()
                if floor is not None:
                    target = floor + MARGIN_C
                    print(f'after the warm-up, {SETTLE_SECONDS} s of idle reach {floor:.1f} C; every measured '
                          f'run starts at or below {target:.1f} C', flush=True)
            for r in range(args.runs):
                order = columns if r % 2 == 0 else columns[::-1]
                for kind in kinds:
                    for column in order:
                        runs.append(measure(args.table, trees[column], column, kind, out, n, model, target,
                                            args.settle))
                        n += 1
    except ab.Stop as stop:
        print(f'readme_bench: {stop}', file=sys.stderr)
        return stop.code
    (out / 'runs.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'rows'} for r in runs], indent=1))
    text = summarize(args.table, columns, runs)
    head = (f'floor {target - MARGIN_C:.1f} C, target {target:.1f} C, runs {args.runs}\n\n'
            if target is not None else f'no mactop; fixed settle {args.settle:g} s, runs {args.runs}\n\n')
    (out / 'summary.md').write_text(head + text + '\n')
    print('\n' + text)
    print(f'\noutput: {out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ab.Stop as e:
        print(f'readme_bench: {e}', file=sys.stderr)
        raise SystemExit(e.code)
