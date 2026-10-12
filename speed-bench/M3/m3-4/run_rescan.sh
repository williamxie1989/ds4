#!/bin/sh
# M3-4 [W]: streaming cache capacity rescan auto/8000/9000/10000.
# Legs per arm: append831, append1524, decode ladder 2K..32K x512 (all warm,
# production-shaped ctx-alloc 393216). Cross-arm comparability: same prompt,
# same order; auto arm runs first tonight for paired comparison.
# Gates: per-arm trio preflight (no LLM residents, composite avail >=
# cache_GiB + 25, swap used <= 3072M) -> skip arm if unmet; swap > 8192M
# mid-run aborts the remaining arms. Nothing here touches port 8055.
set -u
cd "$(dirname "$0")/../../.."
V41=gguf/DeepSeek-V4.1-Flash-Q2.gguf
PROMPT=speed-bench/promessi_sposi.txt
OUT=speed-bench/M3/m3-4/run-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
WLOG=$OUT/window.log
abort() { echo "ABORT: $* $(date +%H:%M:%S)" | tee -a "$WLOG"; exit 3; }
avail_gib() { vm_stat | awk '/Pages free/{f=$3}/inactive/{i=$3}/speculative/{s=$3}/purgeable/{p=$3} END{printf "%.0f", (f+i+s+p)*16384/2^30}'; }
swap_used() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v+0}'; }
arm() { # $1=tag $2=cache experts or empty=auto $3=cache GiB estimate
    TAG=$1; EXP=$2; GIB=$3
    pgrep -fl 'ds4-server|ds4-bench|mlxCamlRunner|python.*omlx' >/dev/null && { echo "skip $TAG: LLM resident $(date +%H:%M:%S)" | tee -a "$WLOG"; return 1; }
    SW=$(swap_used); AV=$(avail_gib)
    NEED=$((GIB + 25))
    # Swap preflight exemption (user-approved 2026-10-12): this machine carries
    # ~13.1 GB of historical swap residue; the absolute 3072M floor would skip
    # every arm by construction. Gate on GROWTH instead: the mid-run fuse below
    # aborts if swap climbs more than 6 GB over SWAP0.
    SWINT=${SW%.*}
    [ "$SWINT" -gt $((SWAP0 + 2048)) ] && { echo "skip $TAG: swap ${SW}M grew >2G over SWAP0=${SWAP0}M $(date +%H:%M:%S)" | tee -a "$WLOG"; return 1; }
    [ "$AV" -lt "$NEED" ] && { echo "skip $TAG: avail ${AV}G < ${NEED}G $(date +%H:%M:%S)" | tee -a "$WLOG"; return 1; }
    CE=""
    [ -n "$EXP" ] && CE="--ssd-streaming-cache-experts $EXP"
    echo "== arm $TAG start $(date +%H:%M:%S) (experts=${EXP:-auto} avail=${AV}G swap=${SW}M) ==" | tee -a "$WLOG"
    ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights $CE \
        --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 831 --ctx-max 11000 \
        --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/append831-$TAG-p$PASS.csv" >> "$WLOG" 2>&1
    echo "  append831 rc=$? $(date +%H:%M:%S)" | tee -a "$WLOG"
    ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights $CE \
        --ctx-alloc 393216 --ctx-start 8192 --step-mul 1 --step-incr 1524 --ctx-max 12700 \
        --gen-tokens 0 --prompt-file "$PROMPT" --csv "$OUT/append1524-$TAG-p$PASS.csv" >> "$WLOG" 2>&1
    echo "  append1524 rc=$? $(date +%H:%M:%S)" | tee -a "$WLOG"
    ./ds4-bench --model "$V41" --metal --ssd-streaming --warm-weights $CE \
        --ctx-alloc 393216 --ctx-start 2048 --ctx-max 32768 --step-mul 2 \
        --gen-tokens 512 --prompt-file "$PROMPT" --csv "$OUT/decode-$TAG-p$PASS.csv" >> "$WLOG" 2>&1
    echo "  decode rc=$? $(date +%H:%M:%S)" | tee -a "$WLOG"
    SW=$(swap_used)
    [ "${SW%.*}" -gt $((SWAP0 + 6144)) ] && abort "swap ${SW}M grew >6G over SWAP0=${SWAP0}M after arm $TAG"
    echo "== arm $TAG done $(date +%H:%M:%S) swap=${SW}M ==" | tee -a "$WLOG"
}
test -x ./ds4-bench || abort "ds4-bench missing"
test -r "$V41" || abort "model missing"
# 2026-10-12: 8000/10000 already adjudicated by the 2026-10-04 rescan
# (8000 non-monotonic, 10000 physically unselectable on 128G). Tonight is the
# ABBA confirmation of the leading arm: two passes of {auto, 9000}.
SWAP0=$(swap_used); SWAP0=${SWAP0%.*}
echo "SWAP0=${SWAP0}M $(date +%H:%M:%S)" | tee -a "$WLOG"
for PASS in 1 2; do
    arm auto  ""    74
    arm 9000  9000  84
done
echo "=== M3-4 RESCAN DONE $(date +%H:%M:%S): $OUT ===" | tee -a "$WLOG"
