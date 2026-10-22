#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab
M=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf
S=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
probe() {
  local name=$1; shift
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt \
    --dspark --mtp-model $S > $OUT/bis-$name.out 2> $OUT/bis-$name.err
  local rc=$?
  local mark=$(grep -o "analyze, [a-z]*" $OUT/bis-$name.out | head -1)
  echo "$name rc=$rc -> ${mark:-no-marker}"
}
probe engram DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP=1
probe shared DS4_METAL_DISABLE_V41_SHARED_FUSE=1
probe moefuse DS4_METAL_DISABLE_V41_MOE_FUSE=1
probe router DS4_METAL_DISABLE_V41_ROUTER_FUSE=1
probe batchmoe DS4_METAL_DISABLE_V41_BATCH_MOE=1
probe gathermm DS4_METAL_DISABLE_V41_STREAM_GATHER_MM_ID=1
echo BISECT DONE
