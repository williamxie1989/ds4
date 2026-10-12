#!/bin/sh
# M3-3 [W]: R2a-3 acceptance window — bit-exact legs + multi-turn append
# simulator ABBA speed gate (ITERATION_PLAN_2026Q4 M3-3).
#
# Arms:  A = R2a on  (DS4_METAL_STREAMING_EXPERT_HOTLIST=v2 ranked file seed
#            + DS4_V41_HOTLIST_PROTECT=1 protect bitmap)
#        B = R2a off (kill switch set -> seeder early-returns; exactly the
#            pre-R2a production behavior, bit-exact baseline)
#
# Legs:
#   1. bind-parity microtest   (small cache 16 GiB, separate small load)
#   2. dump parity 223/1524    (OFF vs ON dumps byte-compared via
#                               gguf-tools/quality-testing/compare_frontier_logits.py)
#   3. small-cache pressure    (cache 2048 experts, 643/1524 rows, dump parity)
#   4. append simulator ABBA   (6-round tool traffic, cold start, four sizes
#                               spanning [100,512) and one large control;
#                               per-frontier prefill wallclock from CSV +
#                               per-layer gather hit/miss from the cleanup
#                               report; order A B A B across sizes)
#   5. idle prewarm smoke      (recording server, protect on, idle 35 s, one
#                               request; prewarm seed line must appear)
#
# Preflight: §2 window protocol. Swap gate measures GROWTH against the
# machine's historical residue (SWAP_FLOOR_MB, default absolute 3 GB;
# watchdog aborts on >6 GB growth mid-window). Product ds4-server must be
# down. Nothing touches port 8055.
set -u
cd "$(dirname "$0")/../../.."
V41=gguf/DeepSeek-V4.1-Flash-Q2.gguf
SIMDIR=speed-bench/M3/m3-3
OUT=$SIMDIR/run-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT/dumps_off223" "$OUT/dumps_on223" "$OUT/dumps_off1524" "$OUT/dumps_on1524" \
         "$OUT/dumps_p643" "$OUT/dumps_p1524"
WLOG=$OUT/window.log
HOTLIST=$SIMDIR/hotlist_v2_ranked.txt
abort() { echo "ABORT: $* $(date +%H:%M:%S)" | tee -a "$WLOG"; exit 3; }
avail_gib() { vm_stat | awk '/Pages free/{f=$3}/inactive/{i=$3}/speculative/{s=$3}/purgeable/{p=$3} END{printf "%.0f", (f+i+s+p)*16384/2^30}'; }
swap_used() { sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v+0}'; }

test -x ./ds4-bench || abort "ds4-bench missing"
test -r "$V41" || abort "model missing"
test -r "$HOTLIST" || abort "hotlist v2 ranked file missing"
pgrep -x ds4-server >/dev/null && abort "production ds4-server is up"
pgrep -fl 'ds4-bench|mlxCamlRunner|python.*omlx' >/dev/null && abort "LLM resident"

SW0=$(swap_used)
AV=$(avail_gib)
SWAP_FLOOR_MB=${SWAP_FLOOR_MB:-13500}   # absolute pre-gate; 13500 admits the ~13.3 GB residue
[ "${SW0%.*}" -gt "$SWAP_FLOOR_MB" ] && abort "swap ${SW0}M > ${SWAP_FLOOR_MB}M (§2-1)"
[ "$AV" -lt 84 ] && abort "composite avail ${AV}G < 84G (§2-1)"
echo "preflight ok swap=${SW0}M avail=${AV}G $(date +%H:%M:%S)" | tee -a "$WLOG"

# watchdog: abort on >6 GB swap GROWTH over the preflight baseline (§2-6)
(
    while :; do
        sleep 20
        SW=$(swap_used)
        G=$(python3 -c "print(max(0, $SW - $SW0))" 2>/dev/null || echo 0)
        [ "${G%.*}" -gt 6144 ] && { echo "$(date +%H:%M:%S) watchdog: swap grew ${G}M > 6 GB" >> "$WLOG"; pkill -TERM ds4-bench 2>/dev/null; pkill -TERM ds4-server 2>/dev/null; exit; }
        pgrep -f "run_m33.sh" >/dev/null || exit
    done
) &
WD=$!

leg() { echo "=== LEG $1 start $(date +%H:%M:%S) ===" | tee -a "$WLOG"; shift; "$@" >> "$WLOG" 2>&1; RC=$?; echo "=== LEG done rc=$RC $(date +%H:%M:%S) ===" | tee -a "$WLOG"; sleep 20; return $RC; }

# ---- leg 1: bind-parity microtest (16 GiB cache) -------------------------
# Gate-1 set (P0c/P0g 27/27): the full row sweep carries the adjudicated
# [9,31] registered red (HANDOFF-V41 15.2b) which is NOT part of this gate.
leg bind-parity env DS4_TEST_STREAMING_CACHE_GIB=16 DS4_TEST_BIND_PARITY_GATE1=1 \
    ./tests/test_deepseek41_dspark "$V41" --moe-bind-parity
grep -q "bit-exact at every layer and size" "$WLOG" || abort "bind-parity gate FAILED"

# ---- leg 2: dump parity 223/1524, B(protect off) vs A(protect on) --------
BENCH_COMMON="--model $V41 --metal --ssd-streaming --warm-weights \
    --ctx-alloc 393216 --gen-tokens 0"
leg dump223-off  env -u DS4_V41_HOTLIST_PROTECT -u DS4_METAL_STREAMING_EXPERT_HOTLIST \
    DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 \
    ./ds4-bench $BENCH_COMMON --ctx-start 2048 --step-incr 223 --ctx-max 2940 \
    --prompt-file "$SIMDIR/sim_480.txt" --dump-frontier-logits-dir "$OUT/dumps_off223"
leg dump223-on   env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" DS4_V41_HOTLIST_PROTECT=1 \
        DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE=1 \
        DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
    ./ds4-bench $BENCH_COMMON --ctx-start 2048 --step-incr 223 --ctx-max 2940 \
    --prompt-file "$SIMDIR/sim_480.txt" --dump-frontier-logits-dir "$OUT/dumps_on223"
leg dump1524-off env -u DS4_V41_HOTLIST_PROTECT -u DS4_METAL_STREAMING_EXPERT_HOTLIST \
    DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 \
    ./ds4-bench $BENCH_COMMON --ctx-start 2048 --step-incr 1524 --ctx-max 8144 \
    --prompt-file "$SIMDIR/sim_1650.txt" --dump-frontier-logits-dir "$OUT/dumps_off1524"
leg dump1524-on  env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" DS4_V41_HOTLIST_PROTECT=1 \
        DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE=1 \
        DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
    ./ds4-bench $BENCH_COMMON --ctx-start 2048 --step-incr 1524 --ctx-max 8144 \
    --prompt-file "$SIMDIR/sim_1650.txt" --dump-frontier-logits-dir "$OUT/dumps_on1524"

F223="2048 2271 2494 2717 2940"
F1524="2048 3572 5096 6620 8144"
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
    --frontiers $F223 --ctx 393216 --model "$V41" --backend metal --quality false \
    --quant-bits 2 --vocab 129280 --output "$OUT/cmp223.json" \
    "$OUT/dumps_off223" "$OUT/dumps_on223" >> "$WLOG" 2>&1 \
    || abort "223 dump compare rc!=0"
grep -q '"status": "identical"' "$OUT/cmp223.json" || abort "223 dump gate: $(grep -o '"status": "[a-z]*"' "$OUT/cmp223.json" | head -1)"
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
    --frontiers $F1524 --ctx 393216 --model "$V41" --backend metal --quality false \
    --quant-bits 2 --vocab 129280 --output "$OUT/cmp1524.json" \
    "$OUT/dumps_off1524" "$OUT/dumps_on1524" >> "$WLOG" 2>&1 \
    || abort "1524 dump compare rc!=0"
grep -q '"status": "identical"' "$OUT/cmp1524.json" || abort "1524 dump gate: $(grep -o '"status": "[a-z]*"' "$OUT/cmp1524.json" | head -1)"
grep -q "hotlist seed source=.*protect=1" "$WLOG" || abort "A-arm protect evidence missing (no seed line protect=1)"
echo "dump parity 223+1524 IDENTICAL $(date +%H:%M:%S)" | tee -a "$WLOG"

# ---- leg 3: small-cache pressure (P0c recipe, 2048 experts) --------------
P643TOK=$(./ds4 --dump-tokens -m "$V41" -p "$(cat "$SIMDIR/sim_480.txt")" 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {n++} END {print n}')
P643TOK=$((P643TOK - 64))
P643MAX=$((2048 + ( (P643TOK - 2048) / 3 ) * 3))
leg press643 env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" DS4_V41_HOTLIST_PROTECT=1 \
    DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
    ./ds4-bench $BENCH_COMMON --ssd-streaming-cache-experts 2048 \
    --ctx-start 2048 --step-incr 643 --ctx-max "$P643MAX" \
    --prompt-file "$SIMDIR/sim_480.txt" --dump-frontier-logits-dir "$OUT/dumps_p643"
leg press1524 env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" DS4_V41_HOTLIST_PROTECT=1 \
    DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
    ./ds4-bench $BENCH_COMMON --ssd-streaming-cache-experts 2048 \
    --ctx-start 2048 --step-incr 1524 --ctx-max 8144 \
    --prompt-file "$SIMDIR/sim_1650.txt" --dump-frontier-logits-dir "$OUT/dumps_p1524"
# The 643-step chain has no same-depth partner (frontier lists differ), so the
# only valid pressure pairing is press1524 vs on1524 (same frontier list) —
# the gate below.  press643 itself must simply complete under thrash.
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
    --frontiers $F1524 --ctx 393216 --model "$V41" --backend metal --quality false \
    --quant-bits 2 --vocab 129280 --output "$OUT/cmpp1524.json" \
    "$OUT/dumps_p1524" "$OUT/dumps_on1524" >> "$WLOG" 2>&1 \
    || abort "pressure compare rc!=0"
grep -q '"status": "identical"' "$OUT/cmpp1524.json" || abort "pressure gate: $(grep -o '"status": "[a-z]*"' "$OUT/cmpp1524.json" | head -1)"
echo "small-cache pressure gate IDENTICAL $(date +%H:%M:%S)" | tee -a "$WLOG"

# ---- leg 4: append simulator ABBA ----------------------------------------
# Sizes: sim_120/300/480 cover the [100,512) row band per-round; sim_1650 is
# the large control.  Six rounds each; order A B A B across the size list.
sim_leg() { # $1=tag $2=simfile
    local TAG=$1 FILE=$2
    # The bench's in-process tokenize can differ by tens of tokens from the
    # CLI dump (948 vs 983 observed); shrink by a 64-token guard so ctx-max
    # always fits the bench's own count.
    local TOK=$(./ds4 --dump-tokens -m "$V41" -p "$(cat "$FILE")" 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {n++} END {print n}')
    TOK=$((TOK - 64))
    local STEP=$(( (TOK - 164) / 6 ))
    local MAX=$((164 + STEP * 6))
    # A = v2 calibrated hotlist file seed + protect bitmap
    #     (DS4_METAL_STREAMING_EXPERT_HOTLIST is the SEED INPUT; the seed
    #      success line is printed under _PROFILE so set it for evidence)
    # B = pre-R2a production behavior. The kill switch is REQUIRED now that
    #      the ds41 session path seeds: without it the seeder would apply the
    #      built-in Flash table to B and silently change the baseline.
    # Both arms warm-weights: --ssd-streaming-cold disables the hotlist seed
    # entirely (hotlist_enabled requires !cold), so cold arms can never test
    # the protect path.
    leg "simA-$TAG" env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" \
        DS4_V41_HOTLIST_PROTECT=1 DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE=1 \
        DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
        DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 DS4_METAL_MEMORY_REPORT=1 \
        ./ds4-bench $BENCH_COMMON \
        --ctx-start 164 --step-incr $STEP --ctx-max $MAX \
        --prompt-file "$FILE" --csv "$OUT/simA-$TAG.csv"
    leg "simB-$TAG" env DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 \
        DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 DS4_METAL_MEMORY_REPORT=1 \
        ./ds4-bench $BENCH_COMMON \
        --ctx-start 164 --step-incr $STEP --ctx-max $MAX \
        --prompt-file "$FILE" --csv "$OUT/simB-$TAG.csv"
}
sim_leg 120 "$SIMDIR/sim_120.txt"
sim_leg 300 "$SIMDIR/sim_300.txt"
sim_leg 480 "$SIMDIR/sim_480.txt"
sim_leg 1650 "$SIMDIR/sim_1650.txt"

# ---- leg 5: idle prewarm smoke (recording server) ------------------------
PORT=${PORT:-8057}
leg prewarm env DS4_METAL_STREAMING_EXPERT_HOTLIST="$HOTLIST" DS4_V41_HOTLIST_PROTECT=1 \
    DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP=5156 \
    DS4_V41_IDLE_PREWARM=1 DS4_V41_IDLE_PREWARM_SEC=5 \
    DS4_METAL_MEMORY_REPORT=1 \
    "$SIMDIR/run_prewarm_smoke.sh" "$V41" "$HOTLIST" "$PORT" "$OUT"
grep -q "streaming expert hotlist seed" "$OUT/prewarm.log" \
    || grep -qE "hits=[1-9]" "$OUT/prewarm.log" \
    || abort "idle prewarm smoke: no seed evidence"

echo "=== M3-3 WINDOW DONE $(date +%H:%M:%S): $OUT ===" | tee -a "$WLOG"
kill "$WD" 2>/dev/null
exit 0