#!/bin/sh
# M0-4 baseline-recheck window (ITERATION_PLAN_2026Q4 §3 M0-4 + M0-2 SPEC_ROWS A/B).
# Sequential legs, one load each, >=25s gaps, abort lines enforced between legs.
# Product ds4-server must be down for the whole window (user ruling 2026-10-05).
set -u
cd "$(dirname "$0")/../../.."   # repo root
OUT=speed-bench/M0/m0-4
LOG=$OUT/logs
mkdir -p "$LOG" "$OUT/v41" "$OUT/glm" "$OUT/dumps_off4" "$OUT/dumps_on4" "$OUT/dumps_off8" "$OUT/dumps_on8"
V41=gguf/DeepSeek-V4.1-Flash-Q2.gguf
GLM=gguf/GLM-5.3-Flash-Q2.gguf
PROMPT=speed-bench/promessi_sposi.txt
RIG=speed-bench/v41_m5max_streaming_ab.sh

abort() { echo "ABORT: $*" | tee -a "$OUT/window.log"; exit 3; }

gate() {
    # between-leg abort lines: swap > 6 GB, foreign ds4 process, <20 GiB free+spec
    SWAP=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v}')
    case "$SWAP" in ''|*[!0-9.]*) abort "swap parse failed";; esac
    [ "${SWAP%.*}" -gt 6144 ] && abort "swap ${SWAP}M > 6 GB"
    pgrep -x ds4-bench >/dev/null && [ "$1" != self ] && abort "stray ds4-bench found"
    pgrep -x ds4-server >/dev/null && abort "ds4-server (production) came up mid-window"
    pgrep -x ds4 >/dev/null && [ "$1" != self ] && abort "stray ds4 found"
    FS=$(vm_stat | awk '/Pages free/{f=$3}/inactive/{i=$3}/speculative/{s=$3}/purgeable/{p=$3} END{printf "%.0f", (f+i+s+p)*16384/2^30}')
    case "$FS" in ''|*[!0-9]*) abort "avail parse failed";; esac
    [ "$FS" -lt 80 ] && abort "composite avail ${FS} GiB < 80 GiB"
    echo "$(date +%H:%M:%S) gate ok swap=${SWAP}M avail=${FS}G" | tee -a "$OUT/window.log"
}

leg() {
    NAME=$1; shift
    echo "=== LEG $NAME start $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
    "$@" > "$LOG/$NAME.log" 2>&1
    RC=$?
    echo "=== LEG $NAME rc=$RC end $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
    sleep 25
    gate notself
}

# steady decode ABBA: A = OFF, B = SPEC_ROWS ON (rig legs, all frontiers 2K..32K x512)
export CSV_DIR=$OUT/v41
gate start
leg a1_off  "$RIG" leg off 101
leg b1_on   "$RIG" leg on 101 DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS=1
leg b2_on   "$RIG" leg on 102 DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS=1
leg a2_off  "$RIG" leg off 102

# append baselines (OFF): 831 and 1524 rows/step, cold-flag + default preload
leg w831_cold ./ds4-bench --model "$V41" --metal --ssd-streaming --ssd-streaming-cold \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 831 --ctx-max 11000 \
    --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/v41/append831-cold.csv"
leg w831_warm ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 831 --ctx-max 11000 \
    --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/v41/append831-warm.csv"
leg w1524 ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 1524 --ctx-max 12700 \
    --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/v41/append1524-warm.csv"

# SPEC_ROWS small-append A/B: 4 and 8 rows/step, logits dumped for a byte-diff parity probe
leg s4_off ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 4 --ctx-max 8240 \
    --gen-tokens 1 --prompt-file "$PROMPT" --csv "$OUT/v41/small4-off.csv" \
    --dump-frontier-logits-dir "$OUT/dumps_off4"
leg s4_on env DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS=1 ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 4 --ctx-max 8240 \
    --gen-tokens 1 --prompt-file "$PROMPT" --csv "$OUT/v41/small4-on.csv" \
    --dump-frontier-logits-dir "$OUT/dumps_on4"
leg s8_off ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 8 --ctx-max 8264 \
    --gen-tokens 1 --prompt-file "$PROMPT" --csv "$OUT/v41/small8-off.csv" \
    --dump-frontier-logits-dir "$OUT/dumps_off8"
leg s8_on env DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS=1 ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 8 --ctx-max 8264 \
    --gen-tokens 1 --prompt-file "$PROMPT" --csv "$OUT/v41/small8-on.csv" \
    --dump-frontier-logits-dir "$OUT/dumps_on8"

# GLM resident decode recheck x2: short..12k, 512 greedy tokens per frontier
leg g1 ./ds4-bench --model "$GLM" --metal --ctx-start 2048 --ctx-max 12288 --step-mul 2 \
    --gen-tokens 512 --prompt-file "$PROMPT" --csv "$OUT/glm/decode-G1.csv"
leg g2 ./ds4-bench --model "$GLM" --metal --ctx-start 2048 --ctx-max 12288 --step-mul 2 \
    --gen-tokens 512 --prompt-file "$PROMPT" --csv "$OUT/glm/decode-G2.csv"

# GLM MTP-timing smoke: bare --mtp, settlement line must print at session close
head -c 6000 "$PROMPT" > "$OUT/mtp_prompt.txt"
leg m1_smoke ./ds4 -m "$GLM" --metal -c 32768 --mtp --mtp-timing -n 256 -p "$(cat "$OUT/mtp_prompt.txt")"

echo "=== WINDOW DONE $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
