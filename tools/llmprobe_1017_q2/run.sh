#!/usr/bin/env bash
# Reproduce the llmprobe 10-question reasoning eval against the already-running
# ds4-server (DeepSeek V4.1 Flash 1017 Q2). This script does NOT load weights:
# it only sends requests to the server and writes a JSON report.
#
# Default target: http://127.0.0.1:8055/v1, model deepseek-v4.1-flash.
# Default eval: --eval-only --eval-questions 10 --eval-suite core,
# reasoning medium, greedy sampling, llmprobe@0.6.17.
#
# Usage:
#   ./run.sh --dry-run          # check endpoint/model/memory, print command, no eval
#   ./run.sh                    # run the 10-question eval + summarize
#   ./run.sh --baseline b.json  # also compare against a saved baseline report
#
# Outputs (default /tmp/ds4-llmprobe-1017q2):
#   report.json   llmprobe --save JSON report
#   run.log       llmprobe stdout
#   run.err.log   llmprobe stderr
#   summary.txt   summarize.py output
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_URL="${LLMPROBE_BASE_URL:-http://127.0.0.1:8055/v1}"
MODEL="${LLMPROBE_MODEL:-deepseek-v4.1-flash}"
OUTDIR="${LLMPROBE_OUTDIR:-/tmp/ds4-llmprobe-1017q2}"
QUESTIONS="${LLMPROBE_QUESTIONS:-10}"
EVAL_SUITE="${LLMPROBE_EVAL_SUITE:-core}"
REASONING="${LLMPROBE_REASONING:-medium}"
SAMPLING="${LLMPROBE_SAMPLING:-}"
BASELINE="${LLMPROBE_BASELINE:-}"
DRY_RUN=0
NO_MEMORY_CHECK=0

usage() {
    cat <<'EOF'
Usage: run.sh [options]

  --dry-run              Check endpoint/model/memory, print command, do not run.
  --out DIR              Output directory (default: /tmp/ds4-llmprobe-1017q2).
  --base URL             OpenAI-compatible base URL (default: http://127.0.0.1:8055/v1).
  --model NAME           Model id (default: deepseek-v4.1-flash).
  --questions N          First N questions (default: 10).
  --eval-suite S         core (default), hard, or code. Do not mix suites.
  --reasoning E          off, low, medium (default), high, default, xhigh.
  --sampling P           precise, balanced, or creative. Empty = greedy (default).
  --baseline FILE        Compare against a saved llmprobe report.
  --no-memory-check      Skip the free-RAM/swap warning (still checks endpoint).
  -h, --help             Show this help.

The model must already be running. This script never starts a second model.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --out) OUTDIR="$2"; shift ;;
        --base) BASE_URL="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        --questions) QUESTIONS="$2"; shift ;;
        --eval-suite) EVAL_SUITE="$2"; shift ;;
        --reasoning) REASONING="$2"; shift ;;
        --sampling) SAMPLING="$2"; shift ;;
        --baseline) BASELINE="$2"; shift ;;
        --no-memory-check) NO_MEMORY_CHECK=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "$QUESTIONS" in
    ''|*[!0-9]*) echo "ERROR: --questions must be a positive integer" >&2; exit 2 ;;
esac
[ "$QUESTIONS" -ge 1 ] || { echo "ERROR: --questions must be >= 1" >&2; exit 2; }
case "$EVAL_SUITE" in core|hard|code) ;; *) echo "ERROR: --eval-suite must be core|hard|code" >&2; exit 2 ;; esac
case "$REASONING" in off|low|medium|high|default|xhigh) ;; *) echo "ERROR: --reasoning must be off|low|medium|high|default|xhigh" >&2; exit 2 ;; esac
case "$SAMPLING" in ''|precise|balanced|creative) ;; *) echo "ERROR: --sampling must be precise|balanced|creative or empty" >&2; exit 2 ;; esac

# ds4-server exposes /v1; accept a bare host:port and normalize to /v1.
BASE_URL="${BASE_URL%/}"
case "$BASE_URL" in
    */v1) ;;
    *) BASE_URL="$BASE_URL/v1" ;;
esac

if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node not found (llmprobe needs Node >= 22.19.0)." >&2
    exit 1
fi
if ! command -v npx >/dev/null 2>&1; then
    echo "ERROR: npx not found." >&2
    exit 1
fi
echo "node: $(node -v 2>/dev/null || echo unknown)"
echo "npx:  $(npx -v 2>/dev/null || echo unknown)"

mkdir -p "$OUTDIR"
models_json="$OUTDIR/models.json"
if ! curl -sS --max-time 10 "$BASE_URL/models" -o "$models_json"; then
    echo "ERROR: cannot reach $BASE_URL/models." >&2
    echo "Start ds4-server first (the script does not start a model)." >&2
    exit 1
fi

python3 - "$models_json" "$MODEL" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception as exc:
    print(f"WARN: cannot parse {path}: {exc}")
    raise SystemExit(0)

ids = [item.get("id", "") for item in data.get("data", []) if isinstance(item, dict)]
print("model ids: " + (", ".join(ids) if ids else "(none)"))
if expected and expected not in ids:
    print(f"WARN: expected model '{expected}' not in /v1/models; pass --model to match.")
PY

server_line="$(ps aux | grep '[d]s4-server' | head -n 1 || true)"
if [ -n "$server_line" ]; then
    echo "ds4-server: $server_line"
    case "$server_line" in
        *DeepSeek-V4.1-Flash-1017-Q2.gguf*) ;;
        *) echo "WARN: ds4-server command line does not show DeepSeek-V4.1-Flash-1017-Q2.gguf." ;;
    esac
    case "$server_line" in
        *DeepSeek-V4.1-Flash-1017-Vision.gguf*) ;;
        *) echo "WARN: ds4-server command line does not show DeepSeek-V4.1-Flash-1017-Vision.gguf." ;;
    esac
else
    echo "WARN: no ds4-server process found; the eval will fail unless another OpenAI server is on $BASE_URL."
fi

other_llm="$(ps aux | grep -Ei 'llama-server|llama-cli|mlx_lm|ollama|vllm|lmstudio' | grep -v 'grep -Ei' | grep -v 'ds4-server' | head -n 5 || true)"
if [ -n "$other_llm" ]; then
    echo "WARN: other LLM-ish processes found (do not start another model):"
    echo "$other_llm"
fi

if [ "$NO_MEMORY_CHECK" -eq 0 ]; then
    echo "--- memory ---"
    mem_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    awk -v b="$mem_bytes" 'BEGIN { printf "hw.memsize: %.2f GiB\n", b / 1024 / 1024 / 1024 }'
    vm_stat 2>/dev/null | sed -n '1,8p' || true
    sysctl vm.swapusage 2>/dev/null || true

    free_pages="$(vm_stat 2>/dev/null | awk '/Pages free/ { gsub(/\./, "", $3); print $3 }' || true)"
    if [ -n "$free_pages" ] && [ "$free_pages" -lt 65536 ]; then
        echo "WARN: free pages < 65536 (~1 GiB at 16 KiB pages); memory is tight."
    fi
    swap_line="$(sysctl vm.swapusage 2>/dev/null || true)"
    swap_used_mb="$(printf '%s\n' "$swap_line" | sed -nE 's/.*used = ([0-9.]+)M.*/\1/p')"
    if [ -n "$swap_used_mb" ]; then
        awk -v u="$swap_used_mb" 'BEGIN { if (u + 0 > 1024) print "WARN: swap used > 1 GiB; watch it while the eval runs." }'
    fi
    echo "NOTE: the model is already loaded; do not start a second ds4-server."
    echo "NOTE: watch swap; if it grows fast, Ctrl-C the eval."
fi

if [ -n "$SAMPLING" ]; then
    sampling_label="sampling ${SAMPLING}"
else
    sampling_label="greedy"
fi
label="1017 Q2 HF file; ${QUESTIONS}q ${EVAL_SUITE} eval; reasoning ${REASONING}; ${sampling_label}"

cmd=(npx --yes llmprobe@0.6.17 "$BASE_URL" \
    --model "$MODEL" \
    --eval-only \
    --eval-questions "$QUESTIONS" \
    --eval-suite "$EVAL_SUITE" \
    --reasoning "$REASONING" \
    --concurrency 1 \
    --save "$OUTDIR/report.json" \
    --no-save \
    --no-color \
    --label "$label")
if [ -n "$SAMPLING" ]; then
    cmd+=(--sampling "$SAMPLING")
fi

echo "--- command ---"
printf '%q ' "${cmd[@]}"
echo
if [ "$DRY_RUN" -eq 1 ]; then
    echo "dry-run: not executing."
    exit 0
fi

log="$OUTDIR/run.log"
errlog="$OUTDIR/run.err.log"
set +e
"${cmd[@]}" >"$log" 2>"$errlog"
status=$?
set -e

echo "llmprobe exit: $status"
echo "stdout: $log"
echo "stderr: $errlog"
echo "report: $OUTDIR/report.json"
if [ -s "$errlog" ]; then
    echo "--- stderr tail ---"
    tail -n 20 "$errlog"
fi
if [ -s "$log" ]; then
    echo "--- stdout tail ---"
    tail -n 40 "$log"
fi

if [ -f "$OUTDIR/report.json" ]; then
    echo "--- summary ---"
    set +e
    if [ -n "$BASELINE" ]; then
        python3 "$SCRIPT_DIR/summarize.py" "$OUTDIR/report.json" --baseline "$BASELINE" | tee "$OUTDIR/summary.txt"
    else
        python3 "$SCRIPT_DIR/summarize.py" "$OUTDIR/report.json" | tee "$OUTDIR/summary.txt"
    fi
    summary_status=$?
    set -e
    if [ "$summary_status" -ne 0 ]; then
        echo "WARN: summarize.py failed; see stderr above." >&2
    fi
else
    echo "WARN: no report written; see logs above." >&2
fi

exit "$status"
