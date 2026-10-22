#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=speed-bench/dspark-confsched-20261020
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 ctx=$2 n=$3 prompt=$4 arm=$5
  local extra="" conf=1
  case $arm in
    B0) extra="--dspark --mtp-model $SUP"; conf=0;;
    B1) extra="--dspark --mtp-model $SUP";;
  esac
  DS4_DSPARK_STATS=1 DS4_DSPARK_CONF_SCHED=$conf \
    ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c $ctx -n $n --temp 0 --prompt-file $prompt $extra \
    > $OUT/$name.out 2> $OUT/$name.err
  echo "[$(date +%H:%M:%S)] $name rc=$? $(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1) md5=$(md5 -q $OUT/$name.out)"
}
# n=384: the discovery probe (~1-2 s) has to amortize before the steady state shows.
for spec in "2k:4096:$OUT/p2k.txt" "8k:16384:$OUT/p8k.txt"; do
  tag=${spec%%:*}; rest=${spec#*:}; ctx=${rest%%:*}; prompt=${rest#*:}
  for round in r4 r5; do
    for arm in A B1; do run ${tag}-${arm}-${round}-n384 $ctx 384 $prompt $arm; done
  done
done
echo ALL DONE
