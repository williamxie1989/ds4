#!/bin/sh
# Second ABBA pair at production-like ctx (65k->75k) for the 3205 whole-layer band.
# Same discipline as append3205_abba.sh; swap growth fuse +3G over entry SWAP0.
set -u
cd "$(dirname "$0")/../../.."
V41=gguf/DeepSeek-V4.1-Flash-Q2.gguf
PROMPT=speed-bench/promessi_sposi.txt
OUT=speed-bench/M3/m3-4/run-$(date +%Y%m%d-%H%M%S)-a3205-ctx75k
mkdir -p "$OUT"
WLOG=$OUT/window.log
SWAP0=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print int(v)}')
echo "SWAP0=${SWAP0}M $(date +%H:%M:%S)" | tee -a "$WLOG"
swap_used() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print int(v)}'; }
leg() {
    TAG=$1; EXP=$2
    pgrep -fl 'ds4-server|ds4-bench|mlxCamlRunner|python.*omlx' >/dev/null && { echo "skip $TAG: LLM resident" | tee -a "$WLOG"; return 1; }
    SW=$(swap_used); [ "$SW" -gt $((SWAP0 + 3072)) ] && { echo "ABORT: swap ${SW}M grew >3G" | tee -a "$WLOG"; exit 3; }
    CE=""
    [ -n "$EXP" ] && CE="--ssd-streaming-cache-experts $EXP"
    echo "== leg $TAG start $(date +%H:%M:%S) swap=${SW}M ==" | tee -a "$WLOG"
    DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1 \
    ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights $CE \
        --ctx-alloc 393216 --ctx-start 65536 --step-mul 1 --step-incr 3205 --ctx-max 75200 \
        --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/a3205-$TAG.csv" >> "$OUT/a3205-$TAG.log" 2>&1
    echo "  leg $TAG rc=$? $(date +%H:%M:%S) swap=$(swap_used)M" | tee -a "$WLOG"
}
test -x ./ds4-bench || { echo "ds4-bench missing"; exit 3; }
leg auto ""
leg 9000 9000
echo "=== A3205 CTX75K DONE $(date +%H:%M:%S): $OUT ===" | tee -a "$WLOG"