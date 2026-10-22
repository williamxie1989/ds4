#!/bin/bash
# Production context (384K) with the automatic expert-cache budget:
# baseline vs the confidence-scheduled DSpark, which is what ds4-server would run.
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 arm=$2
  local extra=""
  case $arm in B1) extra="--dspark --mtp-model $SUP";; esac
  DS4_DSPARK_STATS=1 ./ds4 -m $MODEL --ssd-streaming \
    -c 393216 -n 128 --temp 0 --prompt-file $OUT/p2k.txt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out) $(grep -o 'dynamic cache ([0-9]* experts' $OUT/$name.err | head -1)"
}
for round in p1 p2; do
  for arm in A B1; do run 384k-$arm-$round $arm; done
done
echo ALL DONE
