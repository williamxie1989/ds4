#!/bin/bash
# Baseline (no DSpark) vs expert cache size: the one lever that moved the
# streaming decode number in this investigation.
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
run() {
  local name=$1 cache=$2
  ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c 4096 -n 192 --temp 0 --prompt-file $OUT/p2k.txt \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name cache=$cache rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for round in k1 k2; do
  for cache in 1024 4096 8192; do run 2k-base-$cache-$round $cache; done
done
echo ALL DONE
