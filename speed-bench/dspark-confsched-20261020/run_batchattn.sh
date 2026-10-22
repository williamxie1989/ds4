#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 n=$2 fast=$3
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=0 DS4_DSPARK_VERIFY_BATCH_ATTN=$fast \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 4096 \
    -c 4096 -n $n --temp 0 --prompt-file $OUT/p2k.txt \
    --dspark --mtp-model $SUP > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name fast=$fast rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for round in b1 b2; do
  run 2k-ba-slow-$round 192 0
  run 2k-ba-fast-$round 192 1
done
echo ALL DONE
