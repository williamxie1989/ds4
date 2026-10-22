#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab
MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
SUP=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
run() {
  local name=$1 ctx=$2 prompt=$3 arm=$4
  local extra=""
  case $arm in
    C) extra="--dspark --dspark-strict --mtp-model $SUP";;
    B) extra="--dspark --mtp-model $SUP";;
  esac
  echo "[$(date +%H:%M:%S)] start $name ctx=$ctx arm=$arm"
  DS4_DSPARK_STATS=1 ./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 1024 \
     -c $ctx -n 256 --temp 0 --prompt-file $prompt $extra \
     > $OUT/$name.out 2> $OUT/$name.err
  local rc=$?
  local rate=$(grep -o 'generation: [0-9.]* t/s' $OUT/$name.err | tail -1)
  echo "[$(date +%H:%M:%S)] done  $name rc=$rc ${rate:-no-rate}"
}
for spec in "2k:4096:$OUT/p2k.txt" "8k:16384:$OUT/p8k.txt"; do
  tag=${spec%%:*}; rest=${spec#*:}; ctx=${rest%%:*}; prompt=${rest#*:}
  run ${tag}-r1-A $ctx $prompt A
  run ${tag}-r1-C $ctx $prompt C
  run ${tag}-r1-B $ctx $prompt B
  run ${tag}-r2-B $ctx $prompt B
  run ${tag}-r2-C $ctx $prompt C
  run ${tag}-r2-A $ctx $prompt A
done
echo "ALL DONE"
