#!/bin/sh
# M0-4 resume: remaining legs after the strict-gate abort (g2 + MTP smoke).
# Gate fixed: composite avail (free+inactive+spec+purge) — post-model-exit
# file pages sit in inactive and are instantly reclaimable.
set -u
cd "$(dirname "$0")/../../.."
OUT=speed-bench/M0/m0-4
LOG=$OUT/logs
GLM=gguf/GLM-5.3-Flash-Q2.gguf
PROMPT=speed-bench/promessi_sposi.txt
abort() { echo "ABORT: $*" | tee -a "$OUT/window.log"; exit 3; }
gate() {
    SWAP=$(sysctl -n vm.swapusage | awk '{for(i=1;i<=NF;i++) if($i=="used") {v=$(i+2)} gsub(/M/,"",v); print v}')
    case "$SWAP" in ''|*[!0-9.]*) abort "swap parse failed";; esac
    [ "${SWAP%.*}" -gt 6144 ] && abort "swap ${SWAP}M > 6 GB"
    pgrep -x ds4-server >/dev/null && abort "ds4-server (production) came up mid-window"
    FS=$(vm_stat | awk '/Pages free/{f=$3}/inactive/{i=$3}/speculative/{s=$3}/purgeable/{p=$3} END{printf "%.0f", (f+i+s+p)*16384/2^30}')
    case "$FS" in ''|*[!0-9]*) abort "avail parse failed";; esac
    [ "$FS" -lt 80 ] && abort "composite avail ${FS} GiB < 80 GiB"
    echo "$(date +%H:%M:%S) gate ok swap=${SWAP}M avail=${FS}G" | tee -a "$OUT/window.log"
}
leg() {
    NAME=$1; shift
    echo "=== LEG $NAME start $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
    "$@" > "$LOG/$NAME.log" 2>&1
    RC=$?
    echo "=== LEG $NAME rc=$RC end $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
    sleep 25
    gate
}
gate
leg g2 ./ds4-bench --model "$GLM" --metal --ctx-start 2048 --ctx-max 12288 --step-mul 2 \
    --gen-tokens 512 --prompt-file "$PROMPT" --csv "$OUT/glm/decode-G2.csv"
leg m1_smoke ./ds4 -m "$GLM" --metal -c 32768 --mtp --mtp-timing -n 256 -p "$(cat "$OUT/mtp_prompt.txt")"
echo "=== RESUME DONE $(date +%H:%M:%S) ===" | tee -a "$OUT/window.log"
