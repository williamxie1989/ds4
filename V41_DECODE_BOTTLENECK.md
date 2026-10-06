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

Section 8 does exactly that, and the answer is that neither of the two is the
problem. Note also that the V4.1 decode step does not go through
`metal_graph_encode_decode_layer_phase` at all: `ds41_graph_step` (`ds4.c:41834`)
drives `ds41_graph_layer` -> `ds41_moe` -> `ds41_moe_partial` (`ds4.c:41029`),
which calls `ds4_gpu_routed_moe_one_tensor` directly. The paging is therefore
*inside* that function, not in the encoder's override calls.

One caution that survives from the hypothesis this section tested: the V4.1
"exact" paths are gated by `ds4_gpu_dsv41_exact_admitted()`, which returns false
whenever `g_ssd_streaming_mode` is set -- and streaming is the only mode this
model fits in on a 128 GiB host. They also require
`ds4_gpu_device_is_m3_ultra()`. **On M5 Max in streaming mode the engine is
structurally excluded from those tuned paths**, which is a separate reason to
expect headroom and is not addressed by anything measured here.

## 8. Neither the kernel nor the paging loop: it is the per-layer drain (2026-10-06, third window)

The section-7 conclusion was that the two costs cannot be separated. That is
true of the real decode path, but not of the kernel: the kernel *can* be timed
if its address table is populated first. `tests/test_deepseek41_dspark.c
--routed-split-ssd` does that. It runs 72 real decode steps (which is also what
fills the cache), reads the per-(layer, expert) hit snapshot the streaming cache
writes while it runs, and then binds, for each of the 40 layers, the six experts
*that decode actually selected* -- so the table is valid and the experts are
resident -- before timing the dispatch. Arms, best of four, per token:

| arm | ms/token | GB/s | of which drain |
|---|---|---|---|
| A  gather only, one command buffer | **6.76** | **353** | 6.61 |
| B  gather only, flush between layers | 7.05 | 339 | 6.52 |
| C  A + per-layer `begin_selected_load` | **6.75** | **354** | 6.60 |
| E  per-layer drain only, nothing dispatched | 0.87 | -- | 0.01 |
| D  gather + per-layer drain (what the engine does) | **13.62** | 175 | 0.36 |

The routed path moves 2.22 GiB per token (6 of 384 experts, 9.49 MiB each:
gate 2.90 + up 2.90 + down 3.69 MiB), so arm A is **8.494 G weights in 6.61 ms
= 1285 G weights/s** -- the highest weight rate anywhere in this engine, 1.8x
the dense Q8 `q_b` matvec's 700 G/s. The earlier "116 GB/s, 6x off the dense
kernel" was a bytes-based reading of a weights-bound engine; measured alone,
the Q2 gather is the *fast* one.

**Arm C is the answer to the question.** Adding the paging decision
(`ds4_gpu_stream_expert_cache_begin_selected_load`, the residency check and slot
setup the host runs per layer) changes arm A by **-0.01 ms**. The host paging
loop is free. So is the readback copy itself: over 64 decode steps the streaming
cache reports `read_avg 0.823 ms` of which `copy_avg 0.000` -- every microsecond
of it is `sync_avg`, i.e. `ds4_gpu_end_commands()`.

And that sync is the cost. `ds4_gpu_routed_moe_one_tensor` needs the six expert
ids **on the host** to pick cache slots, so it drains the command queue, reads
them back and reopens a buffer -- per layer, 40 times a token. Arm E puts the
empty round trip at 0.87 ms/token; arm D, the same dispatch with that drain in
place, costs **6.86 ms more than arm A** -- almost exactly the kernel's own GPU
time, which the drain now fully exposes instead of overlapping. This is what
`DS4_METAL_DISABLE_V41_DECODE_QUEUE` and the per-layer `flush_commands` in
`ds41_graph_step` exist to avoid; the drain inside the dispatch defeats them.
The host profile agrees: `DS4_METAL_V41_DECODE_HOST_PROFILE=1` puts **43.401 of
43.01 ms** of the step inside the layer loop, with 1969.3 compute encodes and
139.1 blit copies per step.

**Where the ~20.13 ms actually goes.** The six non-routed stages sum to
20.19 ms and the routed bucket is D minus those, so the bucket decomposes as:

| term | ms/token | how measured |
|---|---|---|
| gather kernel | 6.6 | arm A |
| per-layer drain / lost overlap | 6.9 | arm D - arm A |
| expert-cache misses | 6.0 | 386 loads / 64 tokens at ~1.0 ms (`load_prepare` 0.385 + `load_pread` 0.611 + readahead) |
| router, glue, readahead residue | ~0.6 | remainder |
| **total** | **~20.1** | |

Two cautions on the miss term: it is *host* time inside the same dispatch
(`bind_avg` 0.149 ms over 2560 calls, rising to ~1.6 ms on the 15% of calls with
a miss), and it is logical `pread` traffic, not physical disk -- `miss_pread` is
57 MiB/token while the process's own `ri_diskio_bytesread` stays at 0.8 MB/token.
The page cache is absorbing it. Section 1's "the disk is not involved" stands;
the *path* to the disk still costs.

**Consequences.**

- **Do not touch the gather kernel.** It is the fastest kernel in the engine per
  weight, and it is only 6.6 of the 20 ms. Anything spent on IQ2_XXS/Q2_K gather
  micro-optimisation buys at most a third of the bucket.
- **Do not touch the paging decision.** `begin_selected_load` costs nothing
  (arm C).
- The two real targets are the **per-layer readback drain** (6.9 ms) and the
  **miss path** (6.0 ms). The first already has two implementations in the tree
  that move the readback off the critical path: the generic decode path's
  `metal_graph_selected_async_load_*` worker (`ds4.c:23532`), which the V4.1
  step does not use, and card I's GPU-binding service thread (PR #1178,
  measured +2.7-3.2%). Both keep the drain *as a drain* -- they only remove the
  host from the critical path around it -- which is consistent with card I's
  small gain and leaves the 6.9 ms figure as the ceiling for that shape of fix.
- `DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` looks like the readback-free
  answer (`selected_ids_available = false`, no drain) but it is not: it binds the
  *whole layer's* expert tensors out of the model map (3.64 GiB per layer), which
  is a resident-mode feature and exactly what streaming cannot map.

The D measured in this window is 43.01 ms (23.25 t/s), against 40.2-40.3 ms in
the first two windows at the same 82 GiB / 8078-expert configuration; the arms
are self-consistent within a run and are compared only against that run's D.

## Reproduction (additions)

```sh
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt
```

`--routed-split-ssd` prints D, the arms above, and (with the two env vars) the
streaming cache's own per-layer counters and the V4.1 host profile. It needs
~93 GiB of process footprint at that cache size; the run above peaked at 75.1
GiB resident with swap unchanged.

## 9. The drain, priced in the real step: 6.3-6.5 ms of pure serialization (2026-10-06, fourth window)

Section 8's D-arm gap (6.86 ms) was an isolated measurement: an empty command
buffer holding one gather per layer, drained per layer. In the real step the
drain window contains GPU work that has to finish anyway, so the *net* loss
had never been measured -- and the one structural attempt so far (card I /
PR #1178: move the readback onto a GPU-event-gated service thread) recovered
only +2.7-3.2% (~1.3 ms), 5x less than the isolated gap. The two numbers had
to be reconciled.

The probe: a test-only, never-committed ablation inside the selected-id
readback branch of `ds4_gpu_routed_moe_one_tensor` (`ds4_metal.m`, file-scope
statics + the final `else`). It replays a **stale per-layer id set**: the
first token records each layer's real ids, every later token rebinds that
frozen set through the same `ds4_gpu_stream_selected_ids_prepare` path the
override uses, so host and kernel see the same ids and the cache converges to
240 resident experts. `DS4_METAL_V41_PROBE_STALE_IDS=2` keeps the drain (the
readback real ids land in a scratch buffer and are discarded); `=1` skips the
`end_commands`/`begin_commands` pair entirely and the command buffer stays
open across layers -- exactly the chaining `ds41_graph_step` was designed
for. Output is wrong by construction; only the timing is the point.

Legs at 82 GiB / 8000 experts, one process per leg, decode-window deltas only
(64 tokens, 2560 selected calls), same session:

| leg | layers (host, ~= D) | D wall | drain/layer (sync_avg) | load_calls | bind_avg | window cache |
|---|---|---|---|---|---|---|
| live routing (no probe) | **44.962** | 44.74 | 0.855 ms | 386 (6.03/tok) | 0.150 ms | 2174 all_resident + 386 mixed, 0 all_missing |
| PROBE=2 stale ids, drain kept | **38.900** | 37.94 | 0.848 ms | **0** | 0.002 ms | 2560/2560 all_resident |
| PROBE=1 stale ids, drain skipped | **32.417** | 31.63 | **0.000 ms** | **0** | 0.002 ms | 2560/2560 all_resident |

Both probe legs are identical workloads -- 1969.3 compute encodes and 139.1
blits per step in both, `bind_avg` 0.002 in both, zero loads in both. The
*only* variable is the 40-per-token drain. The two readings:

- **The drain costs 6.48 ms/token net** (layers 38.900 -> 32.417; wall D
  37.94 -> 31.63). Per layer the host waits 0.848 ms inside
  `ds4_gpu_end_commands` (33.9 ms/token), of which 6.5 ms is exposed
  serialization -- GPU idle while the host reopens and re-encodes -- and the
  rest is work that would overlap anyway. Section 8's isolated 6.86 ms was
  *not* inflated by the empty-CB geometry: the real step loses the same
  6.3-6.5 ms.
- **The steady-state miss cost is 6.1-6.8 ms/token** (live 44.96 -> stale
  38.90 in layers; 44.74 -> 37.94 in D), arriving from exactly the 6.03
  loads/token of section 8 and vanishing to zero when the ids stop moving:
  pinned ids give 2560/2560 all-resident hits, `bind_avg` collapses 0.150 ->
  0.002 ms, and `load_calls`, `reuse_scan_calls` and the whole prune/pread
  machinery go silent. This also reconciles the T0 cross-check:
  `bind_avg x 40 = 5.95 ms` ~= `load_calls x ~1.0 ms = 6.03` -- the 0.148 ms
  of bind between the frozen and live legs *is* the miss-path bookkeeping
  (slot prune + addr prepare), so the 6.0 ms miss term is real at this cache
  size and it lives inside bind, not beside it.

**Consequences.**

1. **T3 (remove the drain) is worth >= 4 ms and card I had ~20% of it.**
   Card I's shape -- host off the critical path, drain still a drain --
   recovers the host sync time but not the GPU bubble at the 40 command-buffer
   boundaries per token. The 6.5 ms is only recoverable by shape (b) of
   section 8's option table (per-layer address table over cache buffers +
   on-GPU id->mask kernel), which touches the registered red-line bind
   surface (E0.3, DRIFT <= 7.6e-06). **User adjudication 2026-10-06: T3 is
   closed -- the recoverable shape does not justify the red-line surface.**
2. **T2 (miss path) stands: 6.03 loads/token confirmed in-session**, ~6 ms,
   and its accounting is now bracketed: `load_prepare 0.383 + load_pread
   0.613 + install 0.003 + reuse_scan 0.073 ms` per call accounts for most of
   each ~1.0-1.16 ms load; the residue inside `bind` is where T2.0's prune
   probes (`prune_layer`, `prune_global`) still have to look.
3. Steady-state at this cache size is *both* penalties at once: 6.5 ms drain
   + 6.1 ms misses ~= 12.6 of the ~21 ms routed bucket, leaving the kernel's
   6.6 ms and ~0.6 ms of glue -- section 8's decomposition, now independently
   confirmed by ablation rather than by subtraction.

Caveats: the probe is deleted and the engine is back to the committed bytes
(full rebuild + zero-model suite green); the legs are sequential runs of the
same clean binary, and as always only same-round numbers are compared (this
round's live leg measured D 44.74 vs 43.01 last round, within the known
inter-round spread). Mid-window resident peaked at 84.4 GiB (cleanup report
75.0), swap unchanged throughout.

```sh
# legs (probe existed only in a scratch build; logs: /tmp/t1_{base,drain,nodrain}.log):
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
DS4_METAL_SELECTED_PROFILE=1 DS4_METAL_SELECTED_PROFILE_LAYER=0 \
[DS4_METAL_V41_PROBE_STALE_IDS=2|1] \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt
```

## 10. The miss path, accounted to the last 0.1 ms: it is the serial pread wall (2026-10-06, fifth window, T2.0)

Section 9 left 6.03 loads/token (~5.7 ms) as the second target, with 0.29 ms of
each load unattributed and two untimed prune suspects. Permanent instrumentation
was added to the streaming-expert timing system (commit `c306365`, env-gated,
default off): residency-scan time, `prune_layer`/`prune_global` time + evictions
+ scan entries, pread-pool submit + sync-fallback counts, and a split of the
load wait into overlap (engine work while reads run) versus block. Then three
legs of the identical 64-token window (logs `/tmp/t2_{attr,no_ra,thr1}.log`):

| leg | load_prepare | load_pread | per-load | readahead | prune (layer/global) | D |
|---|---|---|---|---|---|---|
| default (threads=9, readahead on) | 0.353 | 0.588 | 0.944 ms | 1287 calls x 0.084 | 2565 calls each, **0.000 ms, 0 evictions, 0 scans** | 43.73 |
| readahead off | **0.078** | **0.933** | 1.014 ms | 0 | same, 0 | 44.91 |
| pread threads=1 | 0.344 | **0.695** | 1.041 ms | 1332 x 0.079 | -- | 45.16 |

**The composition of the 5.7 ms/token (386 loads x per-load, window delta):**

| term | ms/call | ms/token | verdict |
|---|---|---|---|
| serial page-cache pread (~3 tasks, pool) | 0.588 | **3.55** | the term. 16.4 GB/s with 3 threads; 18.7 GB/s/thread single -- scaling gave only 0.107 ms for 3x parallelism, so it is a per-copy-path ceiling, not disk latency |
| readahead `F_RDADVISE` (3.3 ranges/expert, per-16K-page fcntl) | 0.28 | 1.69 | near self-financing: costs 0.28, saves 0.345 of pread -> net **-0.07**; keep on |
| buffer prep (`prepare_load_buffers`, incl. `take_reusable` scans) | 0.07 | 0.42 | includes reuse_scan 0.071/call |
| batch reuse scan | 0.009 | 0.06 | |
| install | 0.003 | 0.02 | |
| `prune_layer` | ~0 | ~0 | **exonerated with evidence**: `effective_cap(layer, 384, 6) = 384` and the layer cache is indexed by expert, so the eviction loop can never execute |
| `prune_global` | ~0 | ~0 | **exonerated**: in steady state `entry_count == budget` -- every call took the trivial-return path (2565 calls, 0 scans); evictions happen via buffer reuse, which does not grow `entry_count` |

The two views reconcile exactly: `bind_avg 0.142 ms x 2560 = 5.68 ms/token` is
the same cost spread over all selected calls, and 5.7 matches the frozen-ids
ablation's live-minus-stale 6.1 ms.

**What the accounting kills and what it leaves.** Every knob inside cache
management moved the per-load cost by <= 0.11 ms: threads 1 -> 3 (+0.107),
readahead on -> off (-0.070 net), prune does not exist as a cost. The `0.29 ms`
mystery was readahead. The dominant 3.55 ms is the **exposed serial wall of
preads issued after the drain** -- ids are only known after the per-layer sync,
so by design the host pays copy time with the GPU idle (section 9 closed the
only shape that removes the sync). Hiding it inside the current contract needs
the split/masked-gather path (dispatch the resident experts while the missing
ones stream) -- that changes accumulation shape (E0.3 family) and only pays on
the 15% mixed layers. The remaining honest lever is not code: at this cache
size misses are rotation, and the production sizing (10000 experts) is
untested -- if its hot set fits, the miss term is zero where it matters and T2
closes without a line of optimization. No engine default was touched.

```sh
# the three legs (identical except the two env vars; instrumentation commit c306365):
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
[DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1 | DS4_METAL_STREAMING_EXPERT_PREAD_THREADS=1] \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt
```

## 11. The 10000-expert sizing leg: it fits, misses halve, D drops to 40.8 -- but the hot set is still bigger (2026-10-06, sixth window)

Section 10 left one honest lever: the production sizing, untested. Ran the identical
64-token window with `DS4_TEST_STREAMING_CACHE_EXPERTS=10000` (plus `GIB=100` so the
bytes side did not clamp first; log `/tmp/t2_10k.log`). The machine said yes:
budget requested 10000, granted **9861** (the harness caps 100 GiB to 99 to stay under
graph working-set pressure; 99 GiB / 9.49 MiB = 9861), **mlock locked 91.48 GiB with
zero failures** in 1555 ms, swap never moved, window peak lived inside the box.

| metric (64-token window delta) | 8078 experts | 9861 experts |
|---|---|---|
| loads/token | 6.03 | **3.13** |
| fully-resident layers | 2174/2560 (85%) | **2360/2560 (92.2%)** |
| missing experts/window | 435 | 207 |
| miss cost | 5.7 ms/token | **3.85 ms/token** |
| bind_avg | 0.142 | 0.098 |
| D | 43.73 (family 43.7-45.5) | **40.82** |

Per-load cost crept up (pread 0.588 -> 0.847 ms: with 92.7 GiB of cache buffers
mlocked, the page cache that used to absorb reads is squeezed, so more of each read
goes to real disk) -- but load count halved, so the term nets ~1.85 ms/token off and
D lands ~3-4 ms under the 8k family: **-7 to -8%**. That is real, contract-clean,
zero-code.

**The sizing answer is therefore "mostly, not fully":** the 128 GB machine locks the
production cache and takes the win, but the hot set of this long-prompt workload
still rotates ~3.1 misses/token at 9861 entries -- the miss term is smaller, not zero,
and the remaining ~3.9 ms miss + ~6.5 ms drain cost stays closed under the current
contract shape (sections 9-10). Anything larger than ~10000 does not fit this machine.
Recommendation: **run production at 10000 experts on 128 GB boxes** and close T2 here --
what is left is only reachable through the two shapes the project has already closed
(T3 drain removal, split masked gather).

Housekeeping: the leg's `layers 82.666` profile line double-counts the before/after
arms of the DSpark test inside the 64-step profile window (D 40.82, 24.50 t/s, and
`bind_avg 0.098 x 2560 = 250 ms` are mutually consistent); read D, not that line.

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
