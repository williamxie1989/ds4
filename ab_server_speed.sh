#!/usr/bin/env bash
# One server-speed A/B arm. Usage: ab_server_speed.sh <label> <worktree-dir>
# Starts ds4-server with a fixed small-context config, sends 1 warmup +
# 3 timed completions, prints wall times, stops the server, checks swap.
set -uo pipefail
LABEL="$1"; WT="$2"
MODEL="/Users/xieyongliang/ds4/gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf"
PORT=8123
LOG="/tmp/ab_${LABEL}_server.log"
cd "$WT"

swap_used_gb() { sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | awk '{printf "%.1f", $1/1024}'; }
free_gb() { vm_stat | awk '/Pages free/{f=$3} /Pages purgeable/{p=$3} /page size/{ps=$8} END{gsub(/\./,"",ps); printf "%.0f", (f+p)*ps/1073741824}'; }

echo "[ab:$LABEL] preflight free=$(free_gb)GB swap=$(swap_used_gb)GB"
./ds4-server -m "$MODEL" --metal -c 32768 --port $PORT >"$LOG" 2>&1 &
SRV=$!
echo "[ab:$LABEL] server pid=$SRV"

# wait for readiness (mmap + kernel compile can take minutes); abort if the
# server dies or swap blows past the guard rail
for i in $(seq 1 120); do
  sleep 5
  if ! kill -0 $SRV 2>/dev/null; then echo "[ab:$LABEL] DIED early; log tail:"; tail -5 "$LOG"; exit 1; fi
  SW="$(swap_used_gb)"
  if awk "BEGIN{exit !($SW > 16)}"; then
    echo "[ab:$LABEL] SWAP GUARD: swap=${SW}GB > 16GB — killing server immediately"; kill -TERM $SRV; exit 2
  fi
  if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models"; then
    echo "[ab:$LABEL] ready after ~$((i*5))s"; break
  fi
  [ "$i" = 120 ] && { echo "[ab:$LABEL] TIMEOUT waiting for readiness"; kill -TERM $SRV; exit 3; }
done

send() { curl -s -o /dev/null -w "%{time_total}" --max-time 900 -H 'Content-Type: application/json' \
        -d @/tmp/ab_payload.json "http://127.0.0.1:$PORT/v1/completions"; }
echo "[ab:$LABEL] warmup: $(send)s"
for r in 1 2 3; do echo "[ab:$LABEL] run$r wall: $(send)s"; done

kill -TERM $SRV; wait $SRV 2>/dev/null
echo "[ab:$LABEL] stopped. post free=$(free_gb)GB swap=$(swap_used_gb)GB"
echo "[ab:$LABEL] --- decode/prefill log lines ---"
grep -iE "prefill.*done|gen[ .]*[0-9]+.*t/s|decode.*tps|avg" "$LOG" | tail -12
