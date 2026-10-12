#!/bin/sh
# Idle-prewarm smoke: start a recording server with protect+hotlist+prewarm,
# send one short request, then idle past the prewarm delay and check the log
# for seed evidence.  Usage: run_prewarm_smoke.sh MODEL HOTLIST PORT OUTDIR
set -u
MODEL=$1; HOTLIST=$2; PORT=$3; OUT=$4
LOG=$OUT/prewarm.log
./ds4-server -m "$MODEL" --metal --ssd-streaming --warm-weights \
    --kv-disk-dir "$OUT/prewarm-kv" --kv-disk-space-mb 8192 \
    --port "$PORT" > "$LOG" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
# wait for listen
for _ in $(seq 1 240); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
    sleep 2
done
curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"reply with the single word ok"}],"max_tokens":4,"temperature":0}' \
    -o "$OUT/prewarm_req.json"
sleep 8   # idle past the 5 s prewarm delay
kill -TERM $SRV 2>/dev/null
wait $SRV 2>/dev/null
trap - EXIT
grep -E "streaming expert hotlist|streaming expert cache budget|hits=" "$LOG" | tail -8 > "$OUT/prewarm_evidence.txt"
exit 0