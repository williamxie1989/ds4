#!/bin/bash
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
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=$conf \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c $ctx -n $TOKENS --temp 0 --prompt-file $prompt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for spec in "2k:4096:$OUT/p2k.txt" "8k:16384:$OUT/p8k.txt"; do
  tag=${spec%%:*}; rest=${spec#*:}; ctx=${rest%%:*}; prompt=${rest#*:}
  for round in r2 r3; do
    for arm in A B1; do run ${tag}-${arm}-$round $ctx $prompt $arm; done
  done
done
echo ALL DONE
