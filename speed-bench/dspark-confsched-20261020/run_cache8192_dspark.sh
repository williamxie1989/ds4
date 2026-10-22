#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 cache=$2 cap=$3 conf=$4
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=$conf DS4_DSPARK_VERIFY_CAP=$cap \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c 4096 -n 192 --temp 0 --prompt-file $OUT/p2k.txt \
    --dspark --mtp-model $SUP > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name cache=$cache cap=$cap conf=$conf rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for round in m1 m2; do
  run 2k-cap5-8192-$round 8192 5 0
  run 2k-sched-8192-$round 8192 5 1
done
echo ALL DONE
