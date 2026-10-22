#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab; M=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf; S=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
pair() {
  local name=$1; shift
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt > $OUT/p4-$name-A.out 2>/dev/null
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt --dspark --mtp-model $S > $OUT/p4-$name-B.out 2>/dev/null
  local ha=$(shasum -a 256 $OUT/p4-$name-A.out | cut -c1-12)
  local hb=$(shasum -a 256 $OUT/p4-$name-B.out | cut -c1-12)
  if [ "$ha" = "$hb" ]; then echo "$name MATCH ($ha)"; else echo "$name DIFF A=$ha B=$hb"; fi
}
pair attn DS4_METAL_DISABLE_V41_BATCH_ATTN=1
pair core DS4_METAL_DISABLE_V41_BATCH_CORE=1
pair compress DS4_METAL_DISABLE_V41_BATCH_COMPRESS=1
pair head DS4_METAL_DISABLE_V41_BATCH_HEAD=1
echo BISECT4 DONE
