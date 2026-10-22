#!/bin/bash
# Verify-row cost curve: fixed-threshold drafting (scheduler off), sweep the
# block width with DS4_DSPARK_VERIFY_CAP and the expert cache.
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 cache=$2 cap=$3
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=0 DS4_DSPARK_VERIFY_CAP=$cap \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c 4096 -n 192 --temp 0 --prompt-file $OUT/p2k.txt \
    --dspark --mtp-model $SUP > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name cache=$cache cap=$cap rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1)"
}
run vc-cap1-1024  1024 1
run vc-cap2-1024  1024 2
run vc-cap3-1024  1024 3
run vc-cap5-1024  1024 5
run vc-cap5-4096  4096 5
run vc-cap1-4096  4096 1
echo ALL DONE
