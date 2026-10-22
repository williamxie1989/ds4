#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
for cfg in "4096 5 sp-cap5" "4096 1 sp-cap1" "1024 5 sp-cap5-c1024"; do
  set -- $cfg; cache=$1; cap=$2; name=$3
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=0 DS4_DSPARK_VERIFY_CAP=$cap \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c 4096 -n 192 --temp 0 --prompt-file $OUT/p2k.txt \
    --dspark --mtp-model $SUP > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1)"
done
echo ALL DONE
