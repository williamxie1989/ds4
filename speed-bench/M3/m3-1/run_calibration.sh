#!/bin/sh
# M3-1 (R2a-1) offline calibration window — [W] class.
# Window protocol: ITERATION_PLAN_2026Q4 §2 (preflight, product down, one
# build one pass, abort lines).  Requires an explicit user-opened window:
# this script refuses to start with any LLM process resident, but opening
# the window itself is the user's call.
#
# What it does (single model load):
#   1. starts a recording ds4-server (V4.1 Flash Q2, --ssd-streaming, plain
#      decode = the production posture; selected-id hotlist counters +
#      500ms snapshots on) on its own disk-KV directory,
#   2. replays a recorded DSH session turn by turn (tools/prefix_reuse_replay.py),
#      archiving one cumulative snapshot per turn boundary,
#   3. SIGTERMs the server (atexit writes the final hotlist + snapshot),
#   4. runs tools/hotlist_v2_from_turns.py + tools/hotlist_v2_calibrate.py
#      over the turn deltas -> hotlist v2 candidate .inc + paper hit-gain
#      calibration report (gate >= 25 pp aggregate).
#
# Usage:
#   SESSION=/path/session.jsonl[.zst] [PORT=8055] [V41=gguf/...] \
#   [CACHE_EXPERTS=6959] [LAYERS=40] \
#     speed-bench/M3/m3-1/run_calibration.sh
# Outputs: speed-bench/M3/m3-1/run-<stamp>/  (+ notes.md)
set -u
cd "$(dirname "$0")/../../.."
OUT=speed-bench/M3/m3-1
SESSION=${SESSION:?need SESSION=/path/to/recorded DSH session jsonl(.zst)}
PORT=${PORT:-8055}
V41=${V41:-gguf/DeepSeek-V4.1-Flash-Q2.gguf}
CACHE_EXPERTS=${CACHE_EXPERTS:-6959}
LAYERS=${LAYERS:-40}
AVAIL_FLOOR_GIB=${AVAIL_FLOOR_GIB:-84}   # V4.1 streaming ~70 GiB x 1.2 (§2-1)

STAMP=$(date +%Y%m%d-%H%M%S)
RUN=$OUT/run-$STAMP
mkdir -p "$RUN/snapshots" "$RUN/turns"
WLOG=$RUN/window.log
LOG=$RUN/server.log
SRV=""
WD=""

abort() {
    echo "ABORT: $* $(date +%H:%M:%S)" | tee -a "$WLOG"
    [ -n "$WD" ] && kill "$WD" 2>/dev/null
    [ -n "$SRV" ] && kill -TERM "$SRV" 2>/dev/null
    exit 3
}

# ---- §2 preflight ------------------------------------------------------
test -x ./ds4-server || abort "ds4-server binary missing (build first)"
test -r "$V41" || abort "model not readable: $V41"
test -r "$SESSION" || abort "session not readable: $SESSION"
python3 -c 'import sys; sys.exit(0)' || abort "python3 missing"
python3 -B -m py_compile tools/prefix_reuse_replay.py \
    tools/hotlist_v2_from_turns.py tools/hotlist_v2_calibrate.py \
    || abort "tools do not compile"
pgrep -fl 'ds4-server|ds4-bench|mlxCamlRunner|python.*omlx' > "$RUN/pgrep.txt" 2>/dev/null
if [ -s "$RUN/pgrep.txt" ]; then
    cat "$RUN/pgrep.txt"
    abort "LLM runtimes resident; window preflight refuses (§2-1/2-2)"
fi
SWAP=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v}')
case "$SWAP" in ''|*[!0-9.]*) abort "swap parse failed: $(sysctl -n vm.swapusage)";; esac
[ "${SWAP%.*}" -gt 3072 ] && abort "swap ${SWAP}M > 3 GB (§2-1)"
FS=$(vm_stat | awk '/Pages free/{f=$3}/inactive/{i=$3}/speculative/{s=$3}/purgeable/{p=$3} END{printf "%.0f", (f+i+s+p)*16384/2^30}')
case "$FS" in ''|*[!0-9]*) abort "avail parse failed";; esac
[ "$FS" -lt "$AVAIL_FLOOR_GIB" ] && abort "composite avail ${FS} GiB < ${AVAIL_FLOOR_GIB} GiB (§2-1)"
echo "$(date +%H:%M:%S) preflight ok swap=${SWAP}M avail=${FS}G" | tee -a "$WLOG"

# ---- session input ------------------------------------------------------
case "$SESSION" in
    *.zst|*.zstd)
        command -v zstd >/dev/null || abort "zstd needed for $SESSION"
        zstd -dc "$SESSION" > "$RUN/session.jsonl" || abort "zstd decode failed"
        SESS=$RUN/session.jsonl ;;
    *) SESS=$SESSION ;;
esac
if [ "${MAX_STEPS:-0}" -gt 0 ]; then
    # trim to the first MAX_STEPS step/end events (keeps the replay leg of
    # a multi-hour session inside one window; prefixes stay sequential)
    python3 - "$SESS" "$RUN/session.trimmed.jsonl" "$MAX_STEPS" <<'PYEOF'
import json, sys
src, dst, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
steps = 0
with open(src, encoding='utf-8') as fi, open(dst, 'w', encoding='utf-8') as fo:
    for line in fi:
        fo.write(line)
        try:
            if json.loads(line).get('type') == 'step/end':
                steps += 1
        except Exception:
            pass
        if steps >= cap:
            break
print('trimmed to %d steps' % steps, file=sys.stderr)
PYEOF
    [ $? -eq 0 ] || abort "session trim failed"
    SESS=$RUN/session.trimmed.jsonl
fi

# ---- recording server ---------------------------------------------------
echo "$(date +%H:%M:%S) starting recording ds4-server (V4.1 streaming, plain decode)" | tee -a "$WLOG"
DS4_MOE_RECORD_SELECTED_HOTLIST="$RUN/hotlist_final.txt" \
DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT="$RUN/snapshots/current.txt" \
DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT_MS=500 \
./ds4-server -m "$V41" --metal -c 393216 --ssd-streaming \
    --kv-disk-dir "$RUN/kv-cache" --kv-disk-space-mb 102400 \
    --warm-weights --port "$PORT" > "$LOG" 2>&1 &
SRV=$!

# watchdog: §2-6 abort lines while the model is up
(
    while kill -0 "$SRV" 2>/dev/null; do
        sleep 20
        SW=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v}')
        case "$SW" in *[!0-9.]*) SW=0;; esac
        if [ "${SW%.*}" -gt 6144 ]; then
            echo "$(date +%H:%M:%S) watchdog: swap ${SW}M > 6 GB -> TERM server" >> "$WLOG"
            kill -TERM "$SRV" 2>/dev/null
            break
        fi
    done
) &
WD=$!

# wait for the port (model load is minutes-class)
i=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null \
   || curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" 2>/dev/null; do
    kill -0 "$SRV" 2>/dev/null || { tail -20 "$LOG"; abort "server died during load"; }
    i=$((i+1)); [ "$i" -gt 900 ] && abort "server not up in 900s"
    sleep 2
done
echo "$(date +%H:%M:%S) server up (pid $SRV); auto-cache check:" | tee -a "$WLOG"
grep -iE "streaming.*(cache|experts)" "$LOG" | tail -3 | tee -a "$WLOG"

# ---- replay with per-turn snapshot capture ------------------------------
python3 tools/prefix_reuse_replay.py \
    --session "$SESS" --replay \
    --base "http://127.0.0.1:$PORT" \
    --out "$RUN/replay_report.json" \
    --snapshot-file "$RUN/snapshots/current.txt" \
    --snapshot-dir "$RUN/turns" \
    2>&1 | tee -a "$WLOG"
RC=$?
echo "$(date +%H:%M:%S) replay rc=$RC" | tee -a "$WLOG"

# ---- stop server (atexit: final hotlist + snapshot) ---------------------
kill -TERM "$SRV" 2>/dev/null
wait "$SRV" 2>/dev/null
kill "$WD" 2>/dev/null
echo "$(date +%H:%M:%S) server stopped" | tee -a "$WLOG"

# ---- offline: candidates + calibration ----------------------------------
ls "$RUN"/turns/turn_*.txt 2>/dev/null | sort > "$RUN/files.txt"
N=$(wc -l < "$RUN/files.txt" | tr -d ' ')
echo "turn snapshots: $N" | tee -a "$WLOG"
[ "$N" -ge 2 ] || abort "need >= 2 turn snapshots, got $N"
python3 tools/hotlist_v2_from_turns.py --files-from "$RUN/files.txt" \
    --cache-experts "$CACHE_EXPERTS" --layers "$LAYERS" \
    --out-inc "$RUN/ds4_streaming_hotlist_v2.inc" \
    --out-json "$RUN/hotlist_v2_tables.json" 2>&1 | tee -a "$WLOG"
python3 tools/hotlist_v2_calibrate.py --files-from "$RUN/files.txt" \
    --cache-experts "$CACHE_EXPERTS" --layers "$LAYERS" \
    --out-json "$RUN/calibration.json" --out-md "$RUN/calibration.md" \
    2>&1 | tee -a "$WLOG"

# ---- notes.md ------------------------------------------------------------
{
    echo "# M3-1 calibration run $STAMP"
    echo
    echo "- session: $SESSION"
    echo "- model: $V41  (V4.1 Flash Q2, --ssd-streaming, plain decode)"
    echo "- replay: tools/prefix_reuse_replay.py --replay --snapshot-file ... (max_tokens=1)"
    echo "- engine env: DS4_MOE_RECORD_SELECTED_HOTLIST=$RUN/hotlist_final.txt"
    echo "  DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT=$RUN/snapshots/current.txt (500 ms)"
    echo "- cache experts: $CACHE_EXPERTS (verify against server log line above);"
    echo "  layers: $LAYERS"
    echo "- turn snapshots: $N"
    echo "- server rc / replay rc: see window.log; swap+process watchdog armed"
    echo "- products: hotlist_v2_tables.json, ds4_streaming_hotlist_v2.inc,"
    echo "  calibration.json, calibration.md (gate >= 25 pp)"
} > "$RUN/notes.md"
echo "=== M3-1 CALIBRATION DONE $(date +%H:%M:%S): $RUN ===" | tee -a "$WLOG"
