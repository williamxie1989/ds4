#!/bin/sh
# Server-side probe: why does production show ~18 t/s when ds4-bench shows ~22.8?
# Arms differ only by --vision / --warm-weights. Each arm: fresh server load,
# three identical greedy completions (~8K prompt tokens, 256 generated).
set -eu
PORT=8056
DIR=speed-bench/v41_m5max_ab
PROMPT_JSON=$(python3 - <<'EOF'
import json
text = open("speed-bench/promessi_sposi.txt", encoding="utf-8", errors="replace").read()[:28000]
print(json.dumps({"model": "ds4", "prompt": text, "max_tokens": 256, "temperature": 0}))
EOF
)

FREE_PCT=$(memory_pressure -Q | grep -o '[0-9]*%' | head -1 | tr -d '%')
if [ "$FREE_PCT" -lt 70 ]; then echo "abort: only ${FREE_PCT}% free"; exit 1; fi

arm() {
    TAG=$1; shift
    LOG=$DIR/server-$TAG.log
    echo "=== arm $TAG $(date +%T) :: $*"
    ./ds4-server -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -c 393216 --port "$PORT" \
        --ssd-streaming --power 100 "$@" >"$LOG" 2>&1 &
    SPID=$!
    i=0
    until nc -z 127.0.0.1 "$PORT" 2>/dev/null; do
        i=$((i+1))
        if [ "$i" -gt 240 ]; then echo "TIMEOUT waiting for server ($TAG)"; kill "$SPID" 2>/dev/null; return 1; fi
        sleep 5
    done
    sleep 3
    for r in 1 2 3; do
        RESP=$(curl -s -w '\n%{time_total}' "http://127.0.0.1:$PORT/v1/completions" \
            -H 'Content-Type: application/json' -d "$PROMPT_JSON")
        T=$(printf '%s\n' "$RESP" | tail -1)
        CT=$(printf '%s\n' "$RESP" | head -1 | python3 -c \
            'import json,sys; d=json.load(sys.stdin); print(d.get("usage",{}).get("completion_tokens","?"))' \
            2>/dev/null || echo "?")
        python3 -c "print(f'arm=$TAG req=$r wall=${T}s tokens=$CT decode_tps={$CT/$T:.2f}')" 2>/dev/null \
            || echo "arm=$TAG req=$r wall=${T}s tokens=$CT (parse failed: $(printf '%s\n' "$RESP" | head -1 | head -c 120))"
    done
    kill "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    sleep 12
}

arm user --vision gguf/DeepSeek-V4.1-Flash-Vision.gguf
arm warm --vision gguf/DeepSeek-V4.1-Flash-Vision.gguf --warm-weights
arm novis --warm-weights
echo "=== PROBE DONE $(date +%T)"
