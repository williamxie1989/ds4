#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab; M=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf; S=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
probe() {
  local name=$1; shift
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt \
    --dspark --mtp-model $S > $OUT/bis2-$name.out 2> $OUT/bis2-$name.err
  echo "$name rc=$? -> $(grep -o 'analyze, [a-z]*' $OUT/bis2-$name.out | head -1)"
}
probe hcfuse DS4_METAL_DISABLE_V41_HC_FUSE=1
probe batchhc DS4_METAL_DISABLE_V41_BATCH_HC=1
probe batchattn DS4_METAL_DISABLE_V41_BATCH_ATTN=1
probe batchcore DS4_METAL_DISABLE_V41_BATCH_CORE=1
echo BISECT2 DONE
