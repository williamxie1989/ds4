#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab; M=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf; S=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
pair() {
  local name=$1 cache=$2 ctx=$3 prompt=$4; shift 4
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c $ctx -n 96 --temp 0 --prompt-file $prompt > $OUT/p5-$name-A.out 2>/dev/null
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts $cache \
    -c $ctx -n 96 --temp 0 --prompt-file $prompt --dspark --mtp-model $S > $OUT/p5-$name-B.out 2>/dev/null
  local ha=$(shasum -a 256 $OUT/p5-$name-A.out | cut -c1-12)
  local hb=$(shasum -a 256 $OUT/p5-$name-B.out | cut -c1-12)
  if [ "$ha" = "$hb" ]; then echo "$name MATCH ($ha)"; else echo "$name DIFF A=$ha B=$hb"; fi
}
pair ctx4k 1024 8192 $OUT/p4k.txt
pair cache4k 4096 16384 $OUT/p8k.txt
echo BISECT5 DONE
