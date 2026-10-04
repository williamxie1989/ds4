#!/bin/sh
# V4.1 Flash × M5 Max 128 GB streaming decode A/B runner.
# See V41_M5MAX_BASELINE_SPEED_PLAN.md. Discipline: DSpark off, one leg = one
# ds4-bench load; REPS repeated by the caller, never inside a single process.
#
# Usage:
#   ./v41_m5max_streaming_ab.sh leg TAG REPS_INDEX [ENV=VAL ...]
#   ./v41_m5max_streaming_ab.sh summary TAG [CSV_DIR]
#
# A leg is always: Q2 streaming, warm weights, production-shaped context
# allocation (-c 393216), frontiers 2K/4K/8K/16K/32K, 512 greedy tokens each,
# default popularity preload (warm). Any extra ENV=VAL is exported verbatim so
# arms differ only by the named env/flags.
set -eu

MODEL=${MODEL:-gguf/DeepSeek-V4.1-Flash-Q2.gguf}
PROMPT=${PROMPT:-speed-bench/promessi_sposi.txt}
CSV_DIR=${CSV_DIR:-speed-bench/v41_m5max_ab}
MODE=${1:-}
TAG=${2:-}

case "$MODE" in
leg)
    REP=${3:?usage: $0 leg TAG REPS_INDEX [ENV=VAL ...]}
    shift 3
    mkdir -p "$CSV_DIR"
    while [ "$#" -gt 0 ]; do export "$1"; shift; done
    exec ./ds4-bench \
        --model "$MODEL" \
        --metal --ssd-streaming --warm-weights \
        --ctx-alloc 393216 \
        --ctx-start 2048 --ctx-max 32768 --step-mul 2 \
        --gen-tokens 512 \
        --prompt-file "$PROMPT" \
        ${EXTRA_ARGS:+$EXTRA_ARGS} \
        --csv "$CSV_DIR/$TAG-$REP.csv"
    ;;
summary)
    DIR=${3:-$CSV_DIR}
    # Per-frontier median of the generation t/s column across reps of TAG.
    for f in "$DIR/$TAG-"*.csv; do
        head -1 "$f" | awk -F, -v OFS=, '{for(i=1;i<=NF;i++) if ($i ~ /gen.*tps/ && $i !~ /first/) print i, $i}'
        break
    done > "$DIR/.$TAG-cols"
    python3 - "$TAG" "$DIR" <<'EOF'
import csv, glob, statistics, sys, os
tag, d = sys.argv[1], sys.argv[2]
cols = [l.split(',') for l in open(f"{d}/.{tag}-cols") if l.strip()]
rows = {}
for path in sorted(glob.glob(f"{d}/{tag}-*.csv")):
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for idx, name in cols:
                name = name.strip()
                v = r.get(name)
                if v:
                    rows.setdefault((r.get('ctx_tokens') or r.get('context') or name, name), []).append(float(v))
for (ctx, name), vals in sorted(rows.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
    print(f"{ctx:>8}  {name:<20} median={statistics.median(vals):9.2f}  n={len(vals)}  all={','.join(f'{v:.2f}' for v in vals)}")
EOF
    ;;
*)
    echo "usage: $0 leg TAG REPS_INDEX [ENV=VAL ...] | $0 summary TAG [CSV_DIR]" >&2
    exit 2
    ;;
esac
