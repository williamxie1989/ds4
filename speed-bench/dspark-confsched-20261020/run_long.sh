#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 arm=$2
  local extra=""
  case $arm in B1) extra="--dspark --mtp-model $SUP";; esac
  DS4_DSPARK_STATS=1 ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 4096 -n 1024 --temp 0 --prompt-file $OUT/p2k.txt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for round in L1 L2; do for arm in A B1; do run 2k-$arm-$round $arm; done; done
echo ALL DONE
