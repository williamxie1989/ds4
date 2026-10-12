#!/usr/bin/env bash
# ds4-bench A/B: menubar branch vs origin/main baseline, same machine/backend.
# Loads a model -> RUN ONLY ON A QUIET MACHINE. Preflight aborts if the box
# is not safe. Usage: ./speed-bench/menubar_ab.sh [MODEL] [CTX_MAX]
set -euo pipefail

MODEL="${1:-ds4flash.gguf}"
CTX_MAX="${2:-32768}"
BASE_WT="${BASE_WT:-$HOME/ds4-bench-base}"      # origin/main worktree
PR_WT="${PR_WT:-$HOME/ds4-menubar}"
OUT="${OUT:-/tmp/menubar-ab}"
mkdir -p "$OUT"

# ---- memory preflight (abort unless headroom is clearly sufficient) ----
resident_llm() { pgrep -lf 'ds4-server|omlx-server|ollama|llama-server|vllm|lmstudio' | grep -v "$0" || true; }
RES="$(resident_llm)"
FREE_GB="$(vm_stat | awk '/Pages free/{f=$3} /Pages purgeable/{p=$3} /page size/{ps=$8} END{gsub(/\./,"",ps); print int((f+p)*ps/1073741824)}')"
MODEL_GB="$(du -g "$MODEL" 2>/dev/null | cut -f1 || echo 0)"
echo "[preflight] free+purgeable=${FREE_GB}GB model=${MODEL_GB}GB"
if [ -n "$RES" ]; then
  echo "[preflight] REFUSING: LLM process resident:"; echo "$RES"
  echo "Stop it first (or run this on a quiet machine), then re-run."
  exit 1
fi
NEED=$(( MODEL_GB + 20 ))   # weights mmap + KV/graph buffers, overestimate
if [ "$FREE_GB" -lt "$NEED" ]; then
  echo "[preflight] REFUSING: need ~${NEED}GB headroom, have ${FREE_GB}GB"
  exit 1
fi

# ---- build baseline in a clean origin/main worktree ----
if [ ! -d "$BASE_WT" ]; then
  git -C "$PR_WT" worktree add "$BASE_WT" origin/main
fi
echo "[build] baseline"; make -C "$BASE_WT" -j8 ds4-bench >/dev/null
echo "[build] PR";       make -C "$PR_WT"  -j8 ds4-bench >/dev/null

run() { # $1=label $2=bench-binary
  echo "[bench] $1"
  "$2" -m "$MODEL" \
    --prompt-file "$PR_WT/speed-bench/promessi_sposi.txt" \
    --ctx-start 2048 --ctx-max "$CTX_MAX" --step-incr 2048 \
    --gen-tokens 128 --csv "$OUT/$1.csv"
}
# order both directions once to cancel thermal drift
run baseline "$BASE_WT/ds4-bench"; run menubar "$PR_WT/ds4-bench"
run menubar "$PR_WT/ds4-bench";    run baseline "$BASE_WT/ds4-bench"

echo "[compare] prefill_tps / gen_tps (menubar vs baseline)"
python3 - "$OUT" <<'PY'
import csv, sys, os
out = sys.argv[1]
def load(lbl):
    rows={}
    with open(f"{out}/{lbl}.csv") as f:
        for r in csv.DictReader(f):
            ctx=int(float(r.get("ctx_size") or r.get("ctx") or 0))
            rows.setdefault(ctx, []).append((float(r["prefill_tps"]), float(r["gen_tps"])))
    return {c:(sum(p for p,_ in v)/len(v), sum(g for _,g in v)/len(v)) for c,v in rows.items()}
b, m = load("baseline"), load("menubar")
print(f"{'ctx':>8} {'pre_base':>9} {'pre_men':>9} {'Δ%':>6} {'gen_base':>9} {'gen_men':>9} {'Δ%':>6}")
for ctx in sorted(set(b)&set(m)):
    pb,gb=b[ctx]; pm,gm=m[ctx]
    print(f"{ctx:>8} {pb:>9.1f} {pm:>9.1f} {100*(pm-pb)/pb:>+6.1f} {gb:>9.2f} {gm:>9.2f} {100*(gm-gb)/gb:>+6.1f}")
print(f"CSVs in {out}")
PY
