#!/bin/bash
# One-shot end-to-end verification for the prefix-replay-cache fix.
# Run INSIDE the ds4 repo root, on the machine that hosts the model.
#
#   MODEL=/path/DeepSeek-V4.1-Flash-Q2.gguf SESSION=./session.v4.jsonl \
#     bash tools/run_verification.sh
#
# Gates:
#   1) build from fix/prefix-replay-cache (set GIT=0 to skip checkout)
#   2) ./ds4_test --server                    (in-process, no model)
#   3) --control-token-roundtrip --logprob-vectors (loads model once)
#   4) patched ds4-server on :PORT, replay recorded DSH session twice
#      (clean + --apply-prunes), assert cache-hit ratio per step
#   5) trace grep: requires-rebuild / token-mismatch counts
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
MODEL=${MODEL:?set MODEL=/abs/path/to.gguf}
SESSION=${SESSION:-session.v4.jsonl}
PORT=${PORT:-18080}
BRANCH=${BRANCH:-fix/prefix-replay-cache}
MIN_FREE_GIB=${MIN_FREE_GIB:-60}
TRACE=/tmp/ds4_verify_trace.txt; SLOG=/tmp/ds4_verify_server.log
SSD=${SSD:-}
fail=0

echo "== preflight"
[ -f "$SESSION" ] || { echo "FATAL: session transcript not found at $SESSION (copy it here first)"; exit 2; }
[ -f "$MODEL" ]  || { echo "FATAL: model not found at $MODEL"; exit 2; }
if ps aux | grep -iE "ds4-server|ds4_test|llama-server|ollama|vllm" | grep -v grep >/dev/null && [ "${ALLOW_BUSY:-0}" != 1 ]; then
  echo "FATAL: another model process is resident (iron rule #1). Move it or set ALLOW_BUSY=1"; exit 2
fi
if [ "$(uname)" = Darwin ]; then
  FREE=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages purgeable/{p=$5} END{gsub(/\./,"",f);gsub(/\./,"",p); printf "%d",(f+p)*ps/2^30}')
  SW=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$5); print int($5)}')
  echo "free+purgeable=${FREE}GiB swapused=${SW}MiB (need >= ${MIN_FREE_GIB}GiB)"
  [ "$FREE" -lt "$MIN_FREE_GIB" ] && { echo "FATAL: below MIN_FREE_GIB (iron rule #2). Free RAM or raise it deliberately."; exit 2; }
  [ -z "$SSD" ] && SSD=1
fi
[ "${GIT:-1}" = 1 ] && { git status --porcelain | grep -q . && { echo "FATAL: dirty tree; commit or stash first"; exit 2; }; git checkout "$BRANCH" || exit 2; }

echo "== [1/5] build"
make -j8 ds4_test ds4-server || { echo "FAIL build"; exit 1; }

echo "== [2/5] ds4_test --server (no model)"
./ds4_test --server 2>&1 | tail -2
[ "${PIPESTATUS[0]:-0}" = 0 ] || { echo "FAIL server unit"; fail=1; }

echo "== [3/5] model-level tests (one load)"
DS4_TEST_MODEL="$MODEL" ${SSD:+DS4_TEST_SSD_STREAMING=1} ./ds4_test --control-token-roundtrip --logprob-vectors 2>&1 | tail -6
[ "${PIPESTATUS[0]:-0}" = 0 ] || { echo "FAIL model-level gate"; fail=1; }

echo "== [4/5] launch patched server on :$PORT"
pkill -f "ds4-server.*--port $PORT" 2>/dev/null
rm -f "$TRACE"
./ds4-server -m "$MODEL" --port "$PORT" --trace "$TRACE" ${SSD:+--ssd-streaming} > "$SLOG" 2>&1 &
SRV=$!
ready=0
for i in $(seq 1 180); do
  sleep 5
  curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null && { ready=1; break; }
  kill -0 $SRV 2>/dev/null || { echo "FATAL: server died, see $SLOG"; tail -20 "$SLOG"; exit 2; }
done
[ $ready = 1 ] || { echo "FATAL: server not ready in 15min"; kill $SRV; exit 2; }
echo "server up (pid $SRV)"

echo "== [5/5] replay: clean, then with prunes"
python3 tools/prefix_reuse_replay.py --session "$SESSION" --replay \
  --base "http://127.0.0.1:$PORT" --out /tmp/replay_clean.json | tee /tmp/replay_clean.log
python3 tools/prefix_reuse_replay.py --session "$SESSION" --replay --apply-prunes \
  --base "http://127.0.0.1:$PORT" --out /tmp/replay_pruned.json | tee /tmp/replay_pruned.log
kill $SRV 2>/dev/null; sleep 2; kill -9 $SRV 2>/dev/null

echo "== verdict"
python3 - <<'EOF' || fail=1
import json,sys
clean=json.load(open('/tmp/replay_clean.json'))
pruned=json.load(open('/tmp/replay_pruned.json'))
miss=[r for r in clean if r['step']>0 and (r['ratio'] or 0)<0.95]
print('clean : %d steps, %d misses (<0.95)'%(len(clean),len(miss)))
for r in miss[:10]: print('   MISS',r)
# pruned: dips allowed only around prune positions; require recovery to >=0.95 afterwards
dips=[i for i,r in enumerate(pruned) if r['step']>0 and (r['ratio'] or 0)<0.95]
ok_recov=all(any((pruned[j]['ratio'] or 0)>=0.95 for j in range(i+1,min(i+6,len(pruned)))) for i in dips if i+1<len(pruned))
print('pruned: %d steps, %d dips, recover-after-dip=%s'%(len(pruned),len(dips),ok_recov))
sys.exit(0 if (not miss and (not dips or ok_recov)) else 1)
EOF
[ $fail = 0 ] || echo "REPLAY GATE FAILED"
RB=$(grep -c "requires rebuild" "$TRACE" 2>/dev/null || echo 0)
TM=$(grep -c "reason=token-mismatch" "$TRACE" 2>/dev/null || echo 0)
echo "trace: requires-rebuild=$RB token-mismatch=$TM  (full log $SLOG, trace $TRACE)"
if [ $fail = 0 ]; then echo "VERIFICATION PASS"; else echo "VERIFICATION FAIL"; exit 1; fi
