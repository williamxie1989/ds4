#!/bin/bash
set -u
cd /Users/xieyongliang/ds4
OUT=/tmp/dspark-ab; M=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf; S=gguf/DeepSeek-V4.1-Flash-DSpark.gguf
pair() {
  local name=$1; shift
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt > $OUT/p3-$name-A.out 2>/dev/null
  env "$@" ./ds4 -m $M --ssd-streaming --ssd-streaming-cache-experts 1024 \
    -c 16384 -n 96 --temp 0 --prompt-file $OUT/p8k.txt --dspark --mtp-model $S > $OUT/p3-$name-B.out 2>/dev/null
  local ha=$(shasum -a 256 $OUT/p3-$name-A.out | cut -c1-12)
  local hb=$(shasum -a 256 $OUT/p3-$name-B.out | cut -c1-12)
  if [ "$ha" = "$hb" ]; then echo "$name MATCH ($ha)"; else echo "$name DIFF A=$ha B=$hb"; fi
}
pair index DS4_METAL_DISABLE_V41_BATCH_INDEX=1
pair topk DS4_METAL_DISABLE_V41_BATCH_TOPK=1
pair packed DS4_METAL_DISABLE_V41_PACKED_INDEX=1
pair engramread DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP=1 DS4_DISABLE_V41_ENGRAM_STEP_READERS=1
echo BISECT3 DONE
