# V4.1 decode: where the 43 ms actually goes (2026-10-06)

Measured on a window, `gguf/DeepSeek-V4.1-Flash-Q2.gguf`, SSD-streaming, ctx 4096,
prefix 1024, expert cache 8078 (D = 40.20 ms = **24.9 t/s**, production parity).
Instruments: `tests/test_deepseek41_dspark.c --verify-scan-ssd` (now accounting the
process's own disk reads via `proc_pid_rusage`), `--stage-timing`,
`--weight-inventory`.

## 1. The disk is not involved. At all.

```
disk read per decode step: 40.1 MB (cold) / 0.8 MB (steady)
```

Steady state is **0.8 MB per token against 9.78 GiB of weight traffic = 0.02%**.
The expert cache covers the hot set completely; going from 958 to 8078 experts
moves `D` by 35% only because the *cold* phase changes, not the steady state.

**Consequences — these routes are dead, do not spend on them:**

- Bigger expert cache / more RAM. `--ssd-streaming-cache-experts 10000` is already
  past the point where it does anything.
- SSD tuning, prefetch, early load (card H was already negative; now we know why).
- Reducing the model's disk footprint.

## 2. What a token actually reads

`--weight-inventory` (per decoded token, all 40 layers):

| group | bytes/token | weight touches | quant |
|---|---|---|---|
| routed experts (6 of 384) | 2.224 GiB | **8.494 G** | Q2_K (0.281 B/w) |
| attention output (o_a, o_b) | 2.988 GiB | 3.020 G | Q8 (1.06 B/w) |
| attention projections (q_a, q_b, kv) | 2.024 GiB | 2.045 G | Q8 |
| shared expert | 1.401 GiB | 1.416 G | Q8 |
| output head + norm | 0.655 GiB | 0.662 G | Q8 |
| indexer, hc mixes, norms, router | 0.483 GiB | 0.180 G | — |
| **total** | **9.78 GiB (10.50 GB)** | **15.816 G** | |

An earlier estimate of "~3.5 GB per token" was 3× low: it forgot that the
attention projections and output are **Q8** and together outweigh the Q2 routed
experts. The model file is 340.6 GB but its tensors sum to 155.87 GiB (167 GB) —
2.03× unexplained. That does not affect decode (0.02% disk) but is worth
understanding for disk footprint.

## 3. Every stage runs at the same *weight* rate

`--stage-timing` (streaming, same config), against the inventory above:

| stage | ms/token | G weights | **G weights/s** | GB/s |
|---|---|---|---|---|
| attention projections | 4.65 | 2.045 | 440 | 467 |
| attention output | 6.00 | 3.020 | 503 | 535 |
| shared expert | 3.34 | 1.416 | 424 | 450 |
| routed experts (by subtraction) | ~20.5 | 8.494 | **~414** | **116** |
| whole step | 40.20 | 15.816 | 393 | 261 |

**This is the finding.** The Q2_K gather path and the Q8 dense paths differ by
**4.6× in bytes/s** and agree within **±10% in weights/s**. A bandwidth-bound
engine cannot do that: the Q2 path moves a quarter of the bytes per weight and
would be correspondingly faster. So:

> Decode is bound by **weights processed per second (~400–500 G/s)**, not by
> bytes per second.

Everything else measured today is consistent with that and nothing else:

- **Batching ≤8 rows does not help** (`a = 0.68`): each row pays its own
  weight-touch cost on the same path.
- **Locality does not help**: verifying the same token eight times (minimum
  expert union, maximum TLB reuse) gives the same slope, 0.666 vs 0.681.
- **Context length does not matter** (user's data: 22.3 → 21.8 t/s from short
  context to 250–300K).
- **Cache size does not matter above ~6000 experts** (steady state).

**Consequence — the last bytes-based route is also dead:** requantizing the
experts to fewer bits will **not** speed decode up, because the Q2 path is
already as fast per weight as the Q8 path. Fewer bytes per weight buys nothing
when the limit is weights per second.

## 4. So the only lever left is the weight rate itself

Two ways to move it, and only one is available:

1. **Touch fewer weights per token.** The routed experts are already 6 of 384;
   attention is architectural. Nothing to win here.
2. **Process more weights per instruction.** The batch-1 decode path runs
   matrix-*vector* kernels (`g_moe_mul_mv_id_q2_k*`), which is roughly one weight
   per lane-op. The tensor units process tiles instead — and per
   `V41_STREAMING_ENGRAM_SPEED_PLAN.md` they need **B ≥ 16/32/512 rows**, which a
   single-token decode does not have. Prefill does use them (the log shows
   `kernel_dsv4_indexed_mixed_attention_heads16_nax`); decode cannot.

So the whole problem reduces to: **at batch = 1 the engine is locked to the
matvec path at ~450 G weights/s, and there is no way to give it more rows
without batching.**

## 5. The one unmeasured cliff, and why it may matter

`DS4_TP_BATCH_MAX_ROWS = 8` (`ds4_tp.h:34`). Every measurement above, including
`a = 0.68`, was taken at **≤ 8 rows** — entirely inside the matvec regime. The
tensor units want **≥ 16**. **The batching curve above 8 rows has never been
measured, because the row ceiling forbids it.**

If the tensor path engages at 16 rows and lifts the weight rate several-fold, the
marginal row cost collapses and speculative decoding's economics flip — the
`a = 0.68` that killed it was measured only where the slow path lives.

That is a bounded experiment: raise `DS4_TP_BATCH_MAX_ROWS` to 16/32, extend
`--verify-scan` past 8, measure. It touches `ds4_tp.h`, the `ds41_spec` buffer
sizing, and the bit-contract surface, so it needs the user's adjudication before
anyone starts. Two cautions:

- The E0.3 registered red ("contract does not exist above eight rows") is about
  the *append* path, not the verify path — but the verify path's exactness must
  be re-established above 8 rows, not assumed.
- The V4.1 "exact" paths are gated by `ds4_gpu_dsv41_exact_admitted()`, which
  returns false whenever `g_ssd_streaming_mode` is set — and streaming is the
  only mode this model fits in on a 128 GiB host. The tuned paths also require
  `ds4_gpu_device_is_m3_ultra()`. **On M5 Max in streaming mode the engine is
  structurally excluded from those paths**, which is a second reason to expect
  headroom, and a second thing to check before betting on row count alone.

## Reproduction

```sh
make tests/test_deepseek41_dspark
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --verify-scan-ssd tests/long_context_security_prompt.txt
DS4_TEST_SSD=1 DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --stage-timing tests/long_context_security_prompt.txt     # routed stage aborts in streaming mode
DS4_TEST_SSD=1 DS4_TEST_STREAMING_CACHE_GIB=16 DS4_TEST_STREAMING_CACHE_EXPERTS=958 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf --weight-inventory
```

`--stage-timing` aborts at the routed-expert stage in streaming mode
("Metal model range ... is not covered by mapped model views"): it binds
whole-map views that streaming does not map. The routed number above is therefore
D minus the four stages that did measure, and the other three are by subtraction
too. Making that stage measurable directly is a prerequisite for trusting the
`~414 G/s` figure to better than ±10%.
