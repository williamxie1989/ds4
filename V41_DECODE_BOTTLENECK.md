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

So the problem reduces to the weight rate. Section 5 tests the one way to give
the engine more rows -- and finds it does not help either.

## 5. The batching cliff does not exist (measured 2026-10-06, second window)

`DS4_TP_BATCH_MAX_ROWS` was raised from 8 to 32 for one experiment and
`--verify-scan` extended to 12/16/24/32 rows. **The per-row cost does not move:**

| rows | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| V/D | 2.186 | 2.739 | 3.456 | 4.839 | 5.936 | 9.047 | 11.428 | 16.849 | 22.393 |
| marginal per row | – | – | – | – | 0.705 | 0.778 | 0.595 | 0.678 | 0.693 |

Fitting the large end, `V(n) = 0.451 + 0.686·n` in decode steps — a fixed batch
overhead of ~18 ms plus ~27.6 ms per row, against 40.3 ms for a standalone step.
So a batched row costs 0.69 of a standalone row, and that ratio is **flat from 8
to 32 rows**. The tensor units either do not engage above 8 rows on this path or
buy nothing when they do.

**The ceiling was reverted to 8** and the scan now skips rows above it rather
than clamping. Nothing else was kept: raising the constant globally would also
widen the batch path for 9..32-row callers, which is exactly the E0.3
registered-red territory, and there is no payoff to buy with that risk.

This closes the last candidate for a step change. Speculative decoding is dead
for a stronger reason than the earlier `a = 0.68`: it is not a small-batch
artifact that a larger verify would fix.

## 6. A dense kernel on the same machine reaches 700 GB/s

`--stage-timing`'s row-scaling tail times 40 layers of the `q_b` projection (Q8,
1.68 G weights) at increasing row counts:

| rows | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| ms for 40 layers | 3.25 | 5.02 | 9.73 | 14.47 | 19.17 |
| ms per row | 3.25 | 2.51 | 2.43 | 2.41 | 2.40 |

Marginal row: 1.68 G weights in 2.40 ms = **700 G weights/s = 700 GB/s**. So the
machine's unified memory delivers at least 700 GB/s on a dense Q8 matvec, and the
dense attention stages (450–535 GB/s) are at 65–75% of that, not at a wall.

The routed-expert path's **116 GB/s is therefore 6× off what the same machine
does on a dense kernel**, and it is ~50% of the decode step.

## 7. The routed dispatch is not in the same command buffer as the layer

Stage 4 of `--stage-timing` used to abort under streaming because the harness
hardcoded the last argument of `ds4_gpu_routed_moe_one_tensor` to `true`. That
argument is `force_resident`; the real call in `ds41_moe_partial` passes
`!g->streaming` (`ds4.c:41095`). Fixing it removes the abort -- and the stage
then reports **0.33 ms per token for all 40 layers**, which is 8 µs per layer.

2.224 GiB of expert weights cannot move in 0.33 ms. So the encode path is not
where the routed cost lives: **the gather is dispatched in a different command
buffer from the rest of the layer, after the host has prepared state for it.**

That is the real shape of the problem, and it is why the stage cannot be
isolated by calling encode helpers out of sequence -- the streaming gather
depends on host-side work done *between* command buffers, so any call that skips
that work measures a gather with an unpopulated address table.

The interleaving lives in `metal_graph_encode_decode_layer_phase`
(`ds4.c:24104`), the per-layer decode encoder, which calls
`metal_graph_decode_set_hash_selected_override` (`ds4.c:23101`, the paging) and
`metal_graph_decode_selected_readahead_override` (`ds4.c:23240`, the readahead)
at `ds4.c:26252` and `ds4.c:26765`.

**So "slow gather kernel" and "slow host paging loop" are not separable by
construction in the current code** -- the paging is a precondition for the
dispatch and they interleave at the command-buffer level. Splitting them means
instrumenting that loop, not isolating the kernel. The routed figure above stays
D minus the other six stages, all six measured through the real decode path
(4.62 + 3.27 + 5.98 + 3.37 + 1.75 + 1.20 = 20.19 ms of 40.32).

One caution that survives from the hypothesis this section tested: the V4.1
"exact" paths are gated by `ds4_gpu_dsv41_exact_admitted()`, which returns false
whenever `g_ssd_streaming_mode` is set -- and streaming is the only mode this
model fits in on a 128 GiB host. They also require
`ds4_gpu_device_is_m3_ultra()`. **On M5 Max in streaming mode the engine is
structurally excluded from those tuned paths**, which is a separate reason to
expect headroom and is not addressed by anything measured here.

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
whole-map views that streaming does not map. The routed number is therefore D
minus the other six stages, all six of which do measure through the real decode
path. See section 7 for why that stage resists isolation, and section 5 for the
row-ceiling experiment (the ceiling is back at 8; the scan skips rows above it).
