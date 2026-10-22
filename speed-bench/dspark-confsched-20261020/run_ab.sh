#!/bin/bash
# A/B for confidence-scheduled DSpark verification (2026-10-20).
#   A  baseline (no support model)
#   C  --dspark-strict   (support resident, never drafts: control arm)
#   B0 --dspark with DS4_DSPARK_CONF_SCHED=0 (fixed threshold + feature ring)
#   B1 --dspark with the cost-curve scheduler (default)
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
TOKENS=${TOKENS:-128}

run() {
  local name=$1 ctx=$2 prompt=$3 arm=$4
  local extra="" conf=1
  case $arm in
    C) extra="--dspark --dspark-strict --mtp-model $SUP";;
    B0) extra="--dspark --mtp-model $SUP"; conf=0;;
    B1) extra="--dspark --mtp-model $SUP";;
  esac
  echo "[$(date +%H:%M:%S)] start $name ctx=$ctx arm=$arm conf_sched=$conf"
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=$conf \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c $ctx -n $TOKENS --temp 0 --prompt-file $prompt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  local rc=$?
  local rate
  rate=$(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1)
  echo "[$(date +%H:%M:%S)] done  $name rc=$rc ${rate:-no-rate} md5=$(md5 -q $OUT/$name.out)"
}

for spec in "2k:4096:$OUT/p2k.txt" "8k:16384:$OUT/p8k.txt"; do
  tag=${spec%%:*}; rest=${spec#*:}; ctx=${rest%%:*}; prompt=${rest#*:}
  for arm in A C B0 B1; do
    run ${tag}-$arm $ctx $prompt $arm
  done
done
echo "ALL DONE"
