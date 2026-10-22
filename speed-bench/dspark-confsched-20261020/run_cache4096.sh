#!/bin/bash
# The cache sweep said verify cost is dominated by the streaming expert cache,
# not by row count. Re-run the A/B at cache 4096 and compare against a baseline
# measured at the same cache size.
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 arm=$2 n=$3
  local extra="" conf=1
  case $arm in B0) extra="--dspark --mtp-model $SUP"; conf=0;; B1) extra="--dspark --mtp-model $SUP";; esac
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=$conf \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 4096 \
    -c 4096 -n $n --temp 0 --prompt-file $OUT/p2k.txt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
for round in c1 c2; do
  for arm in A B1 B0; do run 2k-$arm-$round-c4096 $arm 192; done
done
echo ALL DONE
