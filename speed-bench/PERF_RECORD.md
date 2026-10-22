# Performance record (ds4)

> **Hardware.** Every number here was measured on one machine: an Apple **M5 Max
> with 128 GB** of unified memory, with the DeepSeek V4.1 Flash **1017 Q2** GGUF
> streamed from the internal SSD (`--ssd-streaming`). Absolute values depend on the
> page cache, the room the machine has and its power state; the **gain** is the
> figure to read, not the absolute number.

This is the ledger of speed work on this tree, ported in shape from
`Chida82/sf-ds4-1flash` (`speed-bench/perf-record.md`, 42bc593) and adapted to this
tree's harness and gates. Each row is one `speed-bench/ab_bench.py` run, pasted from
the `record row:` line of its summary. A cell reads `B median (gain)`: the candidate's
absolute median under the frozen streaming configuration, and its gain against the
segment's start build, measured in the same session; positive is faster for every
metric (tokens/s up, seconds down).

## The harness and its frozen configuration

| Piece | Where | What it enforces |
|---|---|---|
| A/B harness | `speed-bench/ab_bench.py` | A B B A quads per kind, warm-up pair per kind, median gain over valid pairs with a bootstrap 95% CI, token-identical gate, optional `--bitwise` prefill-logit gate, cache-counter drift check |
| README tables | `speed-bench/readme_bench.py` | one block per build, discarded warm-up per block, cool-down target from the chip's settled floor +2 °C (mactop), rotated columns, median + spread |
| Unit tests | `tests/test_ab_bench.py` (`make test-ab-bench`, also in `make test`) | the parsers, schedule, statistics, correctness gates and the logits validator, on synthetic data; model-free |

Frozen configuration of every run in this record:

```
--ssd-streaming --ssd-streaming-cache-experts 72GB --ctx-alloc 65536
--prompt-file speed-bench/promessi_sposi.txt
greedy, no EOS, token ids compared run to run
```

**Why 72GB and not auto.** Auto tracks free RAM: on an idle machine it now asks
80.5 GiB dynamic, and page cache left by the previous run moves it, so an A/B pair
would compare two cache sizes. `72GB` is the flag; the loader takes its 7.12 GiB
prefill headroom and reports **64.88 GiB dynamic / 6999 experts**, the size the
historical auto cache settled on (`speed-bench/v41_m5max_ab/auto-1/2.csv`: 64.51 GiB,
6959 experts). The harness refuses a run whose dynamic cache leaves 64.0-65.5 GiB or
whose slot count differs from the first run of the session.

**Why 65536 context.** The sweep kind measures up to 32768 tokens and the bench needs
`ctx_alloc > ctx_max + gen_tokens`; 65536 covers every kind with one allocation.

**Gates for a change in this record** (in order): token ids identical → `--bitwise`
prefill logits when the change claims bit-identical output → `./ds4_test
--logprob-vectors` with the matching vectors → DSpark streaming short-prefill and
verify parity → `./ds4_test --long-context` → SSD streaming + resident both ways →
Metal default. Negative results go to the table below, not to a silent revert.

## Rejected ideas

Read this before proposing one again: a retry needs a reason the measurement does not
cover. Rows marked *fork* are the fork's own measured rejections, re-checked against
our tree; rows marked *ours* are this tree's measurements.

| Idea | Why it does not help here | Source |
|---|---|---|
| `--boost` / fanboost | Root fan daemon at full speed, not a software change; the fork's main table excludes it; hardware risk. | fork, assessment §5 |
| Dual-drive (TB5) prefill (`140`) | Needs a second drive; the fork's own upstream port measured **-13..-38%** (routed `up` runs ~2x slower from the copy's mapping). | fork, assessment §5 |
| Pruning the tree to one model / Metal-only | Pruning is not a runtime speed source; it would break multi-model, CUDA/ROCm and the agent. | fork, assessment §5 |
| Blindly raising the expert cache (82 GB+) | The fork's own 86/88/90 GB data: first token after 8192 goes 134 → 157 → 213 → 301 ms; page compression starts at 86. Our 2K first token at 64.88 GiB is already 62-77 ms. | fork, assessment §5 |
| Route-prediction prefetch, global LRU, CPU/GPU keep-alive, more Engram readers | Measured and dropped in the fork (`perf-record.md`, "Rejected ideas"); our plan §12.6 reached the same conclusion for the read path: misses are already hidden (0.004 ms exposed wait). | fork, assessment §5 |
| Judging a change from back-to-back single-timeline runs | Thermal drift is larger than the effect (the same baseline read 29 and 39 ms per dispatch in the fork). Use A B B A and the CI. | fork |
| Treating the README's +62~78% decode as our target | That is against ds4 `0aaea5a` at 16.6 t/s; this tree already measures 21.7-23.8 t/s steady at 64.88 GiB. The remaining honest gap is +18~30% decode / +20~84% prefill. | assessment §3 |
| `270` lock the expert cache at engine open (fork `5b6482a`) | At 64.88 GiB the lazy mlock never fails, so there is no shrink to prevent; decode +0.6..+0.8% over the decode-kind control (inside noise) and +1.7 s of engine open. The combined run's 2K first-token -6.5% is -2.4% once the control's own -4.1% order bias is removed. Re-apply only where a run prints `using locked cache cap`. | Measured and rejected |
| `131`/`132`/`210` decode readback work (fork `f8dd83e`/`67b75b8`/`37a5d1e`) | All three optimize a per-layer host read of selected ids. This tree has none: `ds41_moe_partial` keeps ids on the GPU (`begin_selected_load_tensor`), `ds41_graph_step` commits per layer without waiting, and the profile shows 0 lines from the two readback functions. Ported 131 was a no-op; 132/210 have nothing to overlap. | Structural finding |

## Segment 1 — frozen baseline (2026-10-21, `89e068c` + working tree)

The segment starts with an **A/A run**: both sides are the same build (the baseline
binary at `/tmp/ds4-base`, byte-identical to `./ds4-bench`), so every "gain" below is
run-to-run noise. Nothing smaller than that noise is believed later.

Run: `speed-bench/v41_m5max_ab/aa-sweep-20261021`, both sides the same binary
(`/tmp/ds4-base` is a byte-identical copy of `./ds4-bench`), `--kinds sweep`, 128
greedy tokens, three complete A B B A quads → 4 valid pairs per metric. No `mactop`
on this machine, so no GPU-clock gating; the run was stopped after three quads to
bound the session, which is why the table has 4 pairs and not 8.

| metric | A median | B median | A/A gain | 95% CI | n |
|---|---:|---:|---:|---|---:|
| prefill 2048 t/s | 133.7 | 133.3 | -0.3% | -1.5% .. +3.4% | 4 |
| prefill 4096 t/s | 188.2 | 187.6 | -0.5% | -1.0% .. +1.0% | 4 |
| prefill 8192 t/s | 224.6 | 225.2 | +0.3% | -1.1% .. +2.2% | 4 |
| prefill 16384 t/s | 383.3 | 380.9 | -0.8% | -2.3% .. +1.6% | 4 |
| prefill 32768 t/s | 516.3 | 516.5 | -0.4% | -1.2% .. +4.2% | 4 |
| steady decode 2048 t/s | 22.5 | 22.4 | +0.0% | -1.0% .. +0.8% | 4 |
| steady decode 4096 t/s | 23.5 | 23.5 | +0.0% | -0.4% .. +0.9% | 4 |
| steady decode 8192 t/s | 22.9 | 22.9 | +0.5% | -0.7% .. +2.1% | 4 |
| steady decode 16384 t/s | 22.3 | 22.3 | +0.2% | -0.2% .. +1.3% | 4 |
| steady decode 32768 t/s | 21.4 | 21.5 | -0.2% | -1.4% .. +1.2% | 4 |
| first token 2048 ms | 67.4 | 66.0 | -0.4% | -1.1% .. +4.5% | 4 |
| first token 4096 ms | 67.5 | 68.2 | -0.4% | -1.4% .. -0.2% | 4 |
| first token 8192 ms | 201.2 | 186.9 | +7.8% | -6.0% .. +28.1% | 4 |
| first token 16384 ms | 184.0 | 227.7 | -14.6% | -27.5% .. -11.1% | 4 |
| first token 32768 ms | 236.9 | 278.4 | -14.7% | -37.0% .. -2.0% | 4 |

**What the A/A says, and what it forbids.**

- **Prefill and steady decode are measurable.** The median A/A gain is within ±0.8%
  and every CI sits inside ±2.3% (prefill) / ±2.1% (decode). A change in this record
  must beat that band on its target metric to count.
- **First token at 2048/4096 is measurable** (median ±0.4%, CI ±4.5%): it is the
  prefill-to-first-decode gap with a warm expert cache.
- **First token at 8192/16384/32768 is not measurable with this protocol.** The A/A
  itself reports -14.6% and -14.7% with CIs that exclude zero, on the *same binary*:
  the page-cache state left by the previous leg dominates the 180-280 ms gap, and the
  A B B A rotation does not cancel it because every process starts cold. Any
  first-token claim at ≥8K needs a different protocol (cold-cache control per leg or
  a page-cache flush between legs), not more pairs.
- **Frozen baseline** (the A medians above, 64.88 GiB dynamic cache, ctx 65536):
  prefill 2048/8192/32768 = **133.7 / 224.6 / 516.3 t/s**; steady decode
  2048/8192/32768 = **22.5 / 22.9 / 21.4 t/s**; first token 2048/8192/32768 =
  **67.4 / 201.2 / 236.9 ms**. Compare `auto-1/2.csv` (auto 64.51 GiB, 512 greedy,
  ctx 393216): prefill 131.6-135.5 / 216.4-228.7 / 507.3-515.9, steady decode
  22.8-23.0 / 22.5-22.8 / 22.1-22.3 — the frozen configuration reproduces the
  historical baseline within the A/A band, so numbers taken here are comparable with
  the earlier record.

Record row (A/A, no change to attribute):

```
| aa | 2026-10-21 | 89e068c | DeepSeek-V4.1-Flash-1017-Q2.gguf | 4 | PASS (A/A) | 133.3 (-0.3%) | 225.2 (+0.3%) | 516.5 (-0.4%) | 22.4 (+0.0%) | 22.9 (+0.5%) | 21.5 (-0.2%) | 66.0 (-0.4%) | 186.9 (+7.8%) | 278.4 (-14.7%) |  |  |
```


### Noise bands by metric class (measured once, reuse them)

An A/A is worth running **once per metric class per segment**, not before every
change. These are the bands this machine produced (same binary on both sides); a
change only needs to beat the band of the metric it targets, and a first read with
2-3 pairs is enough unless the result lands inside the band and the decision matters.

| metric class | kind | A/A median | 95% CI | pairs | source |
|---|---|---:|---|---:|---|
| incremental prefill (2K-32K) | `sweep` | ≤0.8% | ≤±2.3% | 4 | `aa-sweep-20261021` |
| steady decode (2K-32K) | `sweep` | ≤0.5% | ≤±2.1% | 4 | same |
| steady decode (2K/8K) | `decode` | -0.3..+0.5% | ±2-4% | 6-10 | `p1p2-decode`, `p2-only-decode` |
| first token 2K/4K | `sweep` | ±0.4% | ±4.5% | 4 | `aa-sweep-20261021` |
| first token 2K | `decode` | -4.1% | -6.8..-0.0 | 6 | `p2-only-decode` (A/A control) |
| cold ttft 5000 | `cold-5000` | -0.4% | -4.8..+0.3 | 8 | `aa-cold-20261021` |
| cold ttft 7500 | `cold-7500` | +0.6% | +0.2..+4.3 | 6 | same |
| cold first token 5000/7500 | `cold-*` | +1.1/+2.4% | ±3-6% | 6-8 | same |
| first token ≥8K | `sweep` | ±15% | excludes zero | 4 | `aa-sweep-20261021` — **not usable** |

Reading: cold prefill (`ttft 5000/7500`, the P3 target) carries a **±5% band**, wider
than the incremental sweep's ±2%, because it pulls the whole layer set from disk and
depends on what the previous leg left in the page cache. P3's claimed +8.5..+24% is
above that band except for the +8.5% point, which needs 4-6 pairs. Do not re-run a
full A/A to find this out again; extend the existing band with pairs only when a
decision sits inside it.

## Measured and rejected: P1 `270` and P2 `131` (2026-10-21)

Both changes were ported, measured against the frozen baseline, and **rejected**. The
patch is archived as `speed-bench/sf-port-p1p2-20261021.patch` (195 lines, 5 files);
reverse-applying it to the snapshot reproduces the current tree byte for byte, and
`patch -p1 < speed-bench/sf-port-p1p2-20261021.patch` re-applies it.

### P1 — `270` expert cache locked at engine open (fork `5b6482a`)

- **Ported:** `ds4_gpu_stream_expert_cache_lock_ahead()` allocates every slab and
  mlocks every slot at engine open, before the first prefill, and pushes the slots on
  the free list; `DS4_METAL_DISABLE_STREAMING_CACHE_LOCK_AHEAD` restores the lazy
  path. Single-box streaming only (`!load_slice && !tp_shard`).
- **Mechanism check:** `streaming expert cache locked ahead: 6999 of 6999 slots, 17
  slabs, 64.93 GiB in 1656 ms`; engine open grows by ~1.7 s.
- **Gates:** token ids identical at 2K; `--bitwise` prefill logits identical over the
  10-pair decode run.
- **Result (P1+P2 run, 10 pairs):** decode 2048 +0.3% (CI -0.5..+2.9), decode 8192
  +0.5% (CI -0.5..+1.9), first token 2048 -6.5% (CI -9.9..-4.1), first token 8192
  +5.1% (CI -1.5..+15.0). **Read that first-token figure with the control below**: the
  "P2-only" run turned out to be an A/A (P2 never executes on this path, see next
  section) and it showed -4.1% on the same metric, so the order bias of the decode
  kind is about -4% and P1's incremental effect is about -2.4%, inside the noise.
  Decode is +0.6..+0.8% over that same control, also inside noise.
- **Why it does not help here:** at 64.88 GiB the lazy mlock does **not** fail — no
  `using locked cache cap` line appears — so there is no cache shrinkage to prevent.
  Locking 64.93 GiB at open only spends wire memory before the first prefill, and the
  2K first token pays for it. The fork's gain came at 82 GB, where lazy locking hit
  `vm.global_user_wire_limit`.
- **Verdict: rejected for this tree.** The measured decode effect is neutral within
  noise and the only certain cost is +1.7 s of engine open; the first-token signal is
  confounded by the decode kind's own order bias. Re-apply only for a configuration
  shown to shrink the cache below the flag (the harness refuses such a run by name).

### P2 — `131` bounded status poll at the decode readback (fork `f8dd83e`)

- **Ported (wrong site):** a 2 ms status poll in `ds4_gpu_wait_command_buffer`,
  scoped to `metal_graph_decode_cpu_router` and
  `metal_graph_decode_selected_readahead` through `ds4_gpu_set_readback_poll`, with
  `DS4_METAL_DISABLE_V41_READBACK_POLL` to roll it back.
- **The profile says those two functions do not run here.** V4.1 streaming decode
  goes `ds41_graph_decode_layer` → `ds41_graph_layer` → `ds41_moe_partial`; a run with
  `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE=1` and
  `DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE=1` prints **zero** profile lines, so the
  poll was never armed on the measured path. The "P2-only" run is therefore an **A/A
  for the decode kind**, and its numbers are the control: decode 2048 -0.3% (CI
  -4.4..-0.1), decode 8192 -0.0% (CI -3.1..+0.8), first token 2048 -4.1% (CI
  -6.8..-0.0), first token 8192 +1.7% (CI -11.8..+6.9).
- **Verdict: not applicable as ported, and the port is dropped.** The DS41 readback
  lives inside `ds41_moe_partial`/`ds41_graph_layer`, a different structure from the
  fork's `routed_moe_one_tensor`; a real 131/210 port has to land there, and the A/A
  control above says the 2K first token already carries a ~-4% order bias that any
  such claim must beat.

The control also weakens the P1 first-token figure: -6.5% combined minus -4.1% order
bias leaves about -2.4%, inside the noise, so P1's verdict rests on "neutral decode +
1.7 s open", not on a proven first-token regression.

### Why 131/132/210 have no site in this tree (structural, not a measurement)

The fork's three readback changes all optimize one structure: a per-layer host read of
the selected expert ids, which forces a command-buffer commit and wait before the
routed pass. **This tree no longer has that read.**

- `ds41_moe_partial` encodes the router, the shared expert and the routed MoE in one
  batch and passes `g->selected` (a GPU tensor) to
  `ds4_gpu_stream_expert_cache_begin_selected_load_tensor`; the function contains
  **zero** `ds4_gpu_tensor_read` and **zero** `ds4_gpu_end_commands` calls.
- `ds41_graph_step` commits each layer's command buffer **without waiting**
  (`ds4_gpu_flush_commands` only commits and appends to the pending list), so the GPU
  starts layer N while the CPU encodes layer N+1.
- The only host read in the step is the final logits read, once per token.
- Profile confirmation: a 2K decode with
  `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE=1` and
  `DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE=1` prints **0 lines** — the two
  functions the fork's 131 would scope are not on the path.

Consequence: `131` (poll), `132` (router alone) and `210` (id mailbox) all assume a CPU
wait for ids that this tree has already removed (the #1041/#1090 decode queue plus the
GPU-side selected load). **Do not port them** unless a future change reintroduces a
host readback; the thing to measure first in that case is the per-layer commit/wait
count, not the wake latency.

### Attribution: which switch owns the prefill gap (2026-10-21)

Each arm is the fork with one optimization disabled, prefill-only, two reps,
`--frontiers 2048,8192 -n 0`. Numbers are prefill t/s (2048 / 8192-incremental).

| arm | 2048 | 8192 | reading |
|---|---:|---:|---|
| fork default | 180 / 171 | 539 / 470 | — |
| no layer pread | 173 / 170 | 527 / 489 | **no effect** — the read path is not the cause |
| no short sweep | 173 / 171 | 573 / 477 | no effect at these frontiers (S1a needs a 3072-8191 tail) |
| no decoder suffix | 173 / 172 | **401 / 380** | **-23%** — the suffix is the biggest identifiable piece |
| no batch index | 170 / 170 | 471 / 436 | no effect |
| no batch index proj | 171 / 172 | 507 / 484 | no effect |
| **ours** | **136 / 134** | **324 / 284** | — |

Our own suffix, same frontiers (`DS4_V41_DECODER_SUFFIX_MIN_ROWS=1e9` disables it):

| arm | 2048 | 8192 |
|---|---:|---:|
| ours default | 138 / 135 | **316 / 307** |
| ours no suffix | 139 / 133 | **280 / 287** |

**Decomposition at the 8192 frontier:**

- the suffix is worth **+10% to us** (311 → 283) and **+29% to the fork** (505 → 390);
- with the suffix off, the fork is still **+25%** above our suffix-on number
  (390 vs 311), so a second, larger piece exists that none of the switches touch;
- the 2048 frontier is **+27%** and *none* of the arms moves it, so that piece is
  active there too (the fork's S3 single-chunk buffers have no switch; kernels are
  the other candidate).

The earlier statement that "the mechanism is already present" is therefore only
true of the *existence* of a suffix, not of its quality: the fork's S1b (tile-rounded
starts, 128 warm rows, 2541 floor) is a materially better implementation of the same
idea, and the residual 25% is not the suffix at all.

**Stage profile at 2048 (DS4_METAL_V41_STAGE_PROFILE, 40 layers, sums):**

| run | `shared/routed ffn` | all sections | prefill t/s |
|---|---:|---:|---:|
| ours (gather path) | **12955 ms** (324 ms/layer) | 15470 ms | 136 |
| ours, page cache warm | 12426 ms | 14936 ms | — |
| ours, `DS4_METAL_DISABLE_V41_STREAMING_SWEEP_GATHER=1` | **3092 ms** (77 ms/layer) | 6530 ms | 132 |
| fork | **1810 ms** (45 ms/layer) | 2728 ms + ~7 s unaccounted | 176 |

Reading:

- Warm page cache changes the FFN section by 4%, so the 324 ms/layer is **not disk
  wait inside the section** — it is the gather path's dispatch cost (2048 rows
  select nearly every expert, so the per-expert gather degenerates).
- Disabling the gather cuts the section 4.2x (12955 → 3092 ms) but **not the wall
  time** (136 → 132 t/s): the whole-layer path moves the same wait into the layer
  page-in, outside the timed sections. The fork shows the same shape (~7 s outside
  its sections) and still wins by ~4 s of wall time.
- So the fork's advantage at 2048 is **overlap**: it reads the next layer into
  explicit buffers while the current layer computes, and runs the whole-layer
  kernels on those buffers. Our gather path pays dispatch; our whole-layer path
  pays a synchronous page-in; neither overlaps at 2048 rows.

This is the S3 half of the fork's `60` (single-chunk sweeps of 32-2048 rows, two
locked Metal buffers inside the 7.12 GiB reserve, next layer's read overlapping
compute, F_NOCACHE descriptor, purgeable discard; fork measured ttft 2500 +14.2%,
append +1500 +42.6%). It is a smaller port than the wide-sweep half and now has a
direct instrument: the FFN section must fall from ~324 ms/layer toward the fork's 45.


## S3 port: explicit single-chunk expert buffers (2026-10-21, opt-in, regresses)

Ported the fork's `60` S3 half: `ds4_gpu_stream_prefill_bind_layer` /
`ds4_gpu_stream_prefill_discard_buffer` (ds4_metal.m, ds4_gpu.h) and the engine
side (`ds41_prefill_expert_*`: two slots, 16 pread workers, `F_NOCACHE` descriptor,
mlock inside the loader's 7.12 GiB prefill reserve, purgeable discard), wired into
`ds41_graph_prefill_sweep` for single-chunk sweeps of 32..2048 rows.

Measured on 2048 rows, same flags as the baseline:

| arm | prefill t/s | FFN section | token ids |
|---|---:|---:|---|
| S3 off (default) | **140** | 12.9 s (gather) / 3.1 s (gather disabled) | reference |
| S3 on, gather still admitted | **87** | 20.5 s | identical |
| S3 on, gather disabled | **170** | — | identical |

The first S3 attempt (87 t/s) was not the buffers: `gather_sweep` was still true, so
the MoE dispatch took the selected-address gather branch against bound buffers, which
is the slow path. Forcing the whole-layer bind when the explicit buffers are active
(`gather_sweep = !explicit_experts && ...`) gives **170 t/s vs 140 (+21%)** with
identical tokens — the S3 mechanism works and matches the fork's 176.

Cache seed (`896785c`) ported on top: `ds4_gpu_stream_prefill_copy_cached` blits the
decode cache's ready experts into the slot before the disk readers start, and the
readers skip them (merged runs). It recovers steady decode (25.0 t/s single run) but
not the first token: paired A/B against S3-off, `decode` kind, 2 pairs —

| metric | S3 off | S3 on + seed | gain |
|---|---:|---:|---:|
| decode 2048 | 22.4 t/s | 21.7 t/s | **-3.1%** (CI -3.7..-2.4) |
| first token 2048 | 68.3 ms | 423.2 ms | **-84%** |
| decode 8192 | 22.2 t/s | 22.5 t/s | +1.1% (CI +0.4..+1.9) |
| first token 8192 | 206.2 ms | 197.1 ms | +6.2% (noise) |

Cache hit rate drops (0.974 -> 0.909), so the seed is not populating the cache the
way the mmap path does: the seed copies from the bound slot while the next layer's
read is about to overwrite it, and only `copy_wait` (cache->slot) is ordered, not the
slot->cache seed. S3 therefore stays **opt-in** and is a prefill lever only; the
decode target needs the fork's `70` decode fusions, not this path.

Kept **opt-in** (`DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT`) so the default stays on the
mmap path. Next step is to compare the MoE dispatch between the two memory sources —
the fork's S3 relies on the same bind, so either our routed path takes a different
(gather) branch once the decode-span map is installed, or the bound buffer needs the
same residency/tracking attributes as the mmap view.

## Cross-tree check: the fork is really faster (2026-10-21)

The campaign above rejected every ported mechanism on our tree. That is not the same
as the fork's numbers being wrong, so the fork was built (`/tmp/sf-ds4-1flash` @
`42bc593`, `make sf-ds4-1flash-bench`) and run against our tree on the same machine,
model, prompt, cache flag and context allocation.

Config: `--ssd-streaming --ssd-streaming-cache-experts 72GB --ctx-alloc 65536`,
promessi_sposi, both loaders report **64.88 GiB dynamic / 6999 experts** — identical
cache, so this is a like-for-like comparison.

**Prefill-only** (`-n 0`), two runs each, alternating order:

| frontier | fork | ours | fork / ours |
|---|---:|---:|---:|
| 2048 | 177.0 / 175.5 | 135.8 / 134.3 | **+30%** |
| 8192 | 531.3 / 451.3 | 324.0 / 284.4 | **+59~86%** |
| 32768 | 742.8 / 701.4 | 554.0 (1 run) | **+30%** |

**Decode** with `--teacher-forced-decode` (the trees emit different greedy tokens from
token 8 on, so the forced stream is the only fair comparison), 1 A B B A quad = 2
pairs, cache-drift gate off because the two trees account cache reads differently:

| metric | ours | fork | fork advantage |
|---|---:|---:|---:|
| decode 2048 | 22.0 t/s | 26.3 t/s | **+19.5%** (CI +18.6..+20.5) |
| first token 2048 | 65.5 ms | 60.7 ms | **+8.3%** (CI +0.6..+16.0) |
| decode 8192 | 21.8 t/s | 25.3 t/s | **+16.3%** (CI +16.2..+16.3) |
| first token 8192 | 189.3 ms | 59.8 ms | **+216%** (CI +199..+234) |

**What this changes.** The fork's data is credible and the gap is real: prefill
+30~60%, steady decode +16~20%, first token after an 8K prefill ~3x. The per-mechanism
rejections above therefore do **not** mean "no gap"; they mean the mechanisms I tested
or reasoned about are not the ones that produce it. In particular the earlier "P3
mechanism already present, not adopted" verdict was **wrong at the outcome level** —
the fork's wide-sweep read path is measurably not equivalent to our prepare ring. The
fork also runs at a *lower* cache hit rate (0.901 vs 0.973) and is still faster, so
its advantage is in the read/buffer path, not in cache hits.

Caveats: 2 pairs per decode metric; teacher-forced decode is a proxy (same token
stream, but the trees' internal routing can still differ slightly); one machine.

**Next: attribution, not more comparison.** The fork ships a disable switch per
optimization (`DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD`,
`..._V41_DECODER_SUFFIX`, `..._V41_BATCH_INDEX_PROJ`, ...). Running the fork with one
switch off at 2048+8192 prefill-only, two reps, isolates which one owns the gap; then
that piece gets ported for real.

## P3 upper-bound probe: deeper prepare-ahead is neutral (2026-10-21)

Before porting the fork's explicit expert buffers (300-400 lines), the cheap question
is whether the wide sweep has **exposed read wait** at all. The wide branch pins
`prepare_ahead = 1`; the env knob was wired to it (default stays 1, so behaviour is
unchanged unless set) and depth 2 was probed on `cold-7500`, 6 pairs:

| metric | A (depth 1) | B (depth 2) | gain | 95% CI |
|---|---:|---:|---:|---|
| ttft 7500 | 19.5 s | 19.5 s | +1.0% | -0.9% .. +2.8% |
| first token 7500 | 684.9 ms | 693.7 ms | -1.2% | -6.4% .. +4.0% |

**Neutral.** Two readings, and this probe cannot separate them:

1. the one-layer page-in already hides the reads, so P3's overlap half has no
   headroom; or
2. the prepare ring's admission kept one slot for wide sweeps (the comment says
   whole-layer sweeps deliberately keep a single slot because two in-flight layers
   pin two full expert working sets), so depth 2 silently degraded to 1.

What is *not* ambiguous is the arithmetic: `cold-7500` takes 19.5 s for 40 layers,
i.e. ~142 GiB of expert bytes at **~7.3 GiB/s**. A bare F_NOCACHE pread of one layer
runs at the drive's rate (the fork measured 245.7 ms for 3.82 GB ≈ 15.5 GB/s), and the
sweep's GPU compute is roughly 100 ms/layer. So the sweep is neither at the drive
floor (~9 s) nor at a compute floor: **there is time between them that this probe did
not explain**. P3's explicit buffers are the candidate explanation (the GPU reading
through the mmap view instead of a shared buffer), but proving it needs the port, not
another env knob.

**Status: not adopted — the mechanism is already present.** The wide sweep already
runs a one-layer-ahead `pread` ring (`prepare_ahead = 1`,
`metal_graph_stream_prefill_layer_pagein_start(..., pread_only=true)`), which is the
same overlap the fork's two explicit slots provide. On unified memory the GPU then
reads the same pages through the mapping, so the explicit buffers would read the same
bytes from the same memory; the only remaining delta is `F_NOCACHE` page alignment
(the fork's `268`), which needs the explicit driver to exist first and only removes
page-cache pollution, not read time. The fork's +19~24% was measured against its own
tree, whose wide sweeps still paged in synchronously through mmap — not against a
prepare ring. **Do not port the 300-400 lines**, and do not cite the fork's number as
this tree's expected gain.

## P4 probes: S1b threshold neutral, S2 needs new kernels (2026-10-21)

**S1b decoder suffix threshold (fork `0d88af9` S1b).** `ds41_decoder_suffix_min_rows()`
already accepts `DS4_V41_DECODER_SUFFIX_MIN_ROWS` (minimum 2542), so the threshold was
probed with no code change: A = default 4096, B = 2542, same binary, `cold-7500`,
6 pairs. Result: **ttft 7500 +0.8% (CI -1.4..+2.9), first token 7500 +1.1% (CI
-2.1..+3.9)** — neutral. The fork's +30.2% at 7500 came from the *short_sweep* tile
alignment (round `first` down to a 2048 tile, warm 128 rows), not from the threshold;
that part is a code change and carries the DSpark 128-row parity constraint. Verdict:
**keep the default at 4096**; a retry has to port short_sweep and pass DSpark
short-prefill/verify parity first.

**S2 index batch (fork `0d88af9` S2).** The remaining delta is
`ds41_candidates_batch`: batching the il==20 block selection and the il>20 candidate
filter across rows. This tree already has `candidate_blocks`, `candidate_filter` and
the select-all short-row case (`top == blocks` fills the mask), but it runs them
per row; the fork's batch form needs two kernels this tree does not have
(`ds4_gpu_dsv41_candidate_topk_batch`, `ds4_gpu_dsv41_candidate_mask_batch`), with the
top-k tie order preserved bit for bit. Fork gain: +2.3/+1.69/+4.3% ttft at 7500, i.e.
inside this machine's ±5% cold-ttft band. Verdict: **not adopted** — two new kernels
for a sub-band effect; revisit only if a larger prefill change makes 7500-row index
work the dominant term.

## S3 explicit expert buffers extended to wide sweeps (2026-10-22)

The S3 port (fork `0d88af9`/`896785c`) was gated to single-chunk sweeps
(`count 32..2048 && !wide`). The wide half is one predicate: a slot holds a whole
layer, so it can stay bound through every tile of that layer exactly as it does for a
single tile. Change: `ds41_prefill_expert_sweep_supported` drops `count <= 2048` and
`!wide` (still opt-in behind `DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT`).

Paired A/B, `--kinds decode`, 1 quad = 2 pairs, 256 gen. Two rounds; the second one
(`wide-explicit-bitwise-20261022`, with `--bitwise`, machine settled) is primary and the
first (`wide-explicit-20261022`, machine loaded by ~10 preceding model runs) reproduces
the direction. A = default, B = env on:

| metric | A | B | gain | 95% CI |
|---|---:|---:|---:|---|
| prefill 2048 t/s | 133.6 / 135.4 | 174.8 / 180.4 | **+31%** | — |
| prefill 8192 increment (6144 rows) t/s | 319.1 / 320.8 | 430.1 / 444.3 | **+36%** | — |
| first token 8192 ms | 209.0 | 167.0 | **+25.3%** | +22.5 .. +28.1 |
| decode 2048 t/s | 22.3 | 22.4 | +0.7% | +0.6 .. +0.8 |
| decode 8192 t/s | 23.1 | 22.3 | **-3.4%** | -3.5 .. -3.3 |
| first token 2048 ms | 68.4 | **405.2** | **-83.1%** | -83.7 .. -82.5 |

`correctness PASS (tokens; bitwise)`: the mmap path and the explicit-slot path produce
bit-identical prefill logits at both frontiers. The first round (loaded machine) reads
prefill 2048 +27%, prefill 8192 +32%, first token 8192 +21.7% (CI +14.2..+29.3), decode
8192 -0.6%, decode 2048 -0.9%, first token 2048 -82.2%; the two rounds agree everywhere
except decode 8192 (-0.6% vs -3.4%), so **record the conservative -3.4%**.

Cold 8K single run (`--frontiers 8192 -n 16`, one 8192-row sweep = 2 x 4096 tiles):
prefill **397.5 -> 525.0 t/s (+32%)**, first token **749.4 -> 377.2 ms (-50%)**,
steady decode 21.9 -> 23.0 t/s. 32K prefill-only (`--frontiers 2048,32768 -n 0`):
**654.8 -> 750.7 t/s (+14.6%)**, so the wide path does not regress long context even
though it reads whole layers for the decoder-suffix layers. The wide path is a win on
prefill, first token and long context; only the 2048 single-tile frontier regresses on
first token, and decode 8192 pays ~3.4%.

**Why the old gate looked neutral-to-negative at 8192.** `--frontiers 2048,8192 -n 0`
with the old gate: A 319.8 / B 284.9 t/s. The 2048 leg took the explicit path with
`F_NOCACHE`, so the following 6144-row wide leg lost the page cache the mmap path would
have warmed, and the wide leg itself still ran on mmap. With wide admitted the same pair
is A 324.8 / B 444.9.

**2048 first token: the "seed ordering" hypothesis is refuted.**

1. The seed blit is encoded into the current batch command buffer and `ds41_prefill_seed`
   ends with `ds4_gpu_end_commands()`, which waits (`ds4_gpu_finish_command_buffer(cb, 1)`
   -> `waitUntilCompleted`). The next layer's `read_start` cannot overwrite the slot the
   seed is still reading.
2. Disabling the seed (`DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED=1`, 2048 + 2
   tokens, single runs) moves first token 360 -> 307 ms and prefill 171.7 -> 179.2 t/s.
   The seed costs 2.45 s of profiled time but only ~0.5 s of wall, and it is not the
   first-token cause.
3. The actual shape (`DS4_METAL_CB_TIMES=1` + `DS4_METAL_DISABLE_V41_DECODE_QUEUE=1`):
   ~19 command buffers in the first token with `commit->done 20 ms`, `gpu span 0.2 ms`,
   `encode 0.0 ms`; the default path has 1 such buffer. No expert pread is printed in
   that window (`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`), and `miss_pread` is
   0.96 GiB / 32 ms for the whole run. So the host is stalled between commit and wait,
   at the scale of cache-entry eviction (`setPurgeableState` + release) rather than I/O.
4. Consistent with that: 8K does *not* regress. Its decoder suffix shrinks layers 20-39
   to a tail (layer 39 has `count=1`), so the seed's `target` and the churn shrink with it.

**Verdict: wide half accepted behind the existing opt-in switch; 2048 first token is the
one hard blocker before the switch can default on.** Next instrument: per-layer evict
count + time inside `prune_layer`/`prune_global`/`setPurgeableState` during the first
decode token.

## Fork decode glue: the fork does have switches, and the effect is an interaction (2026-10-22)

The premise "the fork's decode glue fusions have no disable switch" is false: the fork
ships 43 `DS4_METAL_DISABLE_V41_*` switches, 15 of them on the decode path. Running the
fork against itself (`--frontiers 2048 -n 128`, one run each, same cache) gives the
attribution directly instead of porting blind:

| fork config | steady decode t/s | first token ms | prefill t/s |
|---|---:|---:|---:|
| default | **27.29 / 25.28** (two runs) | 55.0 / 56.9 | 182.1 / 189.0 |
| all 15 glue switches off | **22.94 / 22.11** (two runs) | 106.7 / 65.6 | 182.8 / 187.7 |
| group A off (router_one, rope_quantize, expand, hc_block, linear_bf, packed_m, topk_prefix) | 29.08 | 53.3 | 184.8 |
| group B off (topk_shuffle, router_mm, router_batch, fast_lookup, packed_index, readback_*) | 25.56 | 62.3 | 183.8 |

Group B add-back (baseline = all 15 off, turn exactly one back on, one run each):

| add back | decode t/s | vs all-off |
|---|---:|---:|
| `ROUTER_MM` | 24.19 | +5.4% |
| `PACKED_INDEX` | 24.16 | +5.3% |
| `DECODE_PIPELINE_FAST_LOOKUP` | 24.11 | +5.1% |
| `TOPK_SHUFFLE` | 22.59 | -1.6% |
| `READBACK_POLL` | 22.45 | -2.2% |
| `READBACK_MAILBOX` | 22.41 | -2.4% |
| `READBACK_SPLIT` | 22.18 | -3.4% |
| `ROUTER_BATCH` | 22.02 | -4.0% |

- The glue family is worth **12-16%** of decode (default 25.3-27.3 vs all-off 22.0-22.9,
  reproduced twice), the same order as the cross-tree gap (ours 22.0 vs fork 26.3), so
  goal 1's direction is confirmed.
- **The fork's own decode varies 4-7% between single runs** (default 27.29 -> 25.28,
  all-off 22.94 -> 22.11, same machine and flags), so the per-item +-5% ranking above is
  indicative only. Next session must pair the runs
  (`ab_bench --a /tmp/sf-tree --b /tmp/sf-tree --b-env ...`, 2 pairs each) before acting.
- All-off is far below "only group A off" (29.08): the switches interact strongly. Group B
  fully on gives 29.08, but its three best items are +5% each, so ~12% is combination
  effect, not a sum of kernels.
- Turning group A off is *faster* than default, so at least one of those fusions is a net
  loss on this machine; do not copy them wholesale.
- `READBACK_*` is structurally inapplicable here (no per-layer host readback in this
  tree, see LOCAL_INVENTORY §D) and each is negative when added back: do not port. They
  still participate in the interaction, i.e. the fork's decode is a pipeline.
- Single best first port: `DECODE_PIPELINE_FAST_LOOKUP` (+5.1%, a structure this tree has
  no equivalent of and it does not depend on the fork's readback).

## Decode attribution: the gap is dispatch count, and it is all in attention (2026-10-22)

Goal 1 (decode 22.5 -> 26 t/s) was instrumented instead of guessed. A dispatch
counter was added to **both** trees (`ds4_gpu_compute_encoder` + atexit print, fork
tree only; removed from this tree again), same flags, `--frontiers 2048 -n 128`:

| | total dispatches | prefill 2048 alone | decode per token |
|---|---:|---:|---:|
| ours | 236,682 | 51,825 | **1,444** (35.9/layer) |
| fork | 122,062 (128 tok) / 65,109 (64 tok) | — | **890** (22.3/layer) |

So the fork issues **554 fewer dispatches per token**, and at the measured ~13 us
per dispatch that is ~7 ms/token — the whole observed gap (ours 42.5 ms = 23.5 t/s,
fork 36-38 ms = 26-27 t/s). Decode is host-encode bound, not GPU bound: GPU busy is
~15-23 ms/token while the wall is 42 ms.

Per-layer split in this tree (env-gated counters, removed after measuring):

| section | dispatches/layer |
|---|---:|
| `ds41_graph_before_attention` | 2.1 |
| **`ds41_attention`** | **25.8** |
| `ds41_graph_after_attention` | 3.0 |
| `ds41_moe` (router + shared + routed + sum) | 4.0 |
| `ds41_graph_after_moe` | 1.0 |
| total | 35.9 |

Inside attention: `project 10.0`, `rope+quantize+window store 3.0`, `select 2.9`,
`gather+decode 1.0`, `output 5.0`. The projection is five calls and **every one of
them is exactly two dispatches**: q_a 2, norm_q 2, q_b 2, kv 2, norm_kv 2 (all Q8_0,
type 8). The pair is "MPP matvec writes f32" + "separate BF16 round".

Why the fused path does not engage: `ds4_gpu_dsv41_matmul_q8_0_bf16` is refused by
`ds4_gpu_mpp_available()` on M5 (verified by admitting it under streaming on M5 —
the dispatch count stayed at 2.0 per projection, and decode stayed 23.5 t/s), so the
M5 path always pays the separate rounding. The fork's `project_pair_q8` /
`norm_pair_rows` / `project_q` fuse exactly these pairs.

**Roadmap (ordered by dispatches saved, all bit-exact by construction):**
1. MPP matvec + BF16 store fusion: project 10 -> 5, output 5 -> 3, indexer -1
   (~8/layer = 320/token = ~4 ms = +10%).
2. `rope_quantize`: kv rope + quantize + window store in one dispatch (3 -> 1,
   ~1 ms = +2.5%).
3. norm pair: norm_q + norm_kv in one dispatch (-1/layer).
Together those reach the fork's ~890/token.

**Negative result: the fused one-dispatch matvec is slower than MPP + round.** The
fused kernel already exists (`ds4_gpu_dsv41_matmul_q8_0_bf16`, one dispatch, bf16 store)
but is refused on M5 because `ds4_gpu_mpp_available()` is true. Forcing it on (probe
env) gives **15.64 t/s against 23.37** (-33%): the legacy non-MPP geometry loses far
more than the saved dispatch. So the fusion must be built on top of MPP (a bf16-store
variant of the MPP matvec), not by switching kernels.

**Negative result: the allocation-free pipeline cache is not worth porting here.**
The fork's `DS4_METAL_DISABLE_V41_DECODE_PIPELINE_FAST_LOOKUP` add-back showed +5.1%,
so it was ported (fast path in `ds4_gpu_get_pipeline`, armed for the decode step).
Measured in this tree: 23.65 t/s with it, 23.66 without (three back-to-back runs).
Neutral, so it was reverted; the fork's +5% was single-run noise, consistent with the
fork's own 4-7% run-to-run decode variance (default 27.29 then 25.28, all-off 22.94
then 22.11).

**Our own decode knobs, for the record** (2048, 128 tokens, single runs): default
23.70; `DS4_METAL_DISABLE_V41_DECODE_PIPELINE_PORTS=1` 22.75 (-4.0%);
`..._DECODE_FLUSH=1` 22.52 (-5.0%); `..._DECODE_PIPELINE=1` 22.74 (-4.1%). All three
are active and each is worth ~4-5%, so the queue structure is already doing its job;
what is left is dispatch count.

## BF16-store fusions: the first two dispatch cuts land (2026-10-22)

Following the dispatch attribution, two kernels gained a "round to BF16 on store"
variant, so a projection or norm that always ends at a BF16 boundary costs one
dispatch instead of two. Both are bit-exact by construction (same walk, same
accumulation, same reduction; only the store rounds) and both are gated by their
own rollback env.

| change | dispatches/token | paired A/B (A = fused) | gates |
|---|---:|---|---|
| baseline | 1,444 | — | — |
| `kernel_mul_mv_q8_0_f32_bf16` (Q8_0 matvec stores BF16) | 1,284 (-160) | **+2.8%** (CI +0.9..+4.7) at 2048; 8192 -0.8% (CI -2.6..+0.9) | tokens + bitwise PASS |
| + `kernel_rms_norm_mul_f32_4_bf16` (weighted RMS norm stores BF16) | 1,166 (-278 total) | +0.7% / +0.9% (CI -1.0..-0.3, i.e. A faster) at 2048 and 8192 | tokens + bitwise PASS |

Rollbacks: `DS4_METAL_DISABLE_V41_MATVEC_BF16_STORE`,
`DS4_METAL_DISABLE_V41_NORM_BF16_STORE`.

Calibration this gives: the first 160 dispatches were worth 1.1 ms/token, so a
dispatch costs ~7 us of host encode — but the second cut (118 dispatches) only
moved the needle by ~0.8%, so the cost per dispatch is not uniform (the removed
BF16 converts are small kernels). The two runs also disagree in magnitude
(+2.8% vs +0.7%) because the machine drifted ~10% between them (decode 23.0 ->
21.2 for the same binary); the sign is consistent and both gates pass, so both
fusions are kept, with the conservative number quoted.

**Where the remaining gap is.** With 1,166 dispatches/token we are still ~19%
above the fork's 890, but dispatch count alone no longer explains the 15%: the
GPU busy time per layer CB measures 0.356 ms here against the fork's 0.231 ms,
so the rest is kernel efficiency (the fork's fused decode attention, pair
projection and expand-row kernels). Next targets, in dispatch terms: attention
output 5/layer, rope+quantize+window store 3/layer, indexer select 2.9/layer.

## Porting the fork's two paired kernels: both rejected (2026-10-22)

The handoff's port list started with the fork's `ds4_gpu_dsv41_project_pair_q8`
(q_a and kv over one activation row in one dispatch) plus
`ds4_gpu_dsv41_norm_pair_rows` (the two weighted RMS norms in one dispatch).
Both were ported on this tree's plain bf16-store matvec and one-threadgroup
norm, gated on `ds4_gpu_dsv41_activation_exact_admitted` with their own
rollbacks (`DS4_METAL_DISABLE_V41_{PROJECT,NORM}_PAIR`), and measured three
times.

| sample | kind | pairs | A (paired) | B (separate) | gain |
|---|---|---:|---:|---:|---:|
| 1 | decode | 2 | 23.5 | 23.6 | -0.4% |
| 2 | decode | 6 | 23.7 | 23.0 | +3.0% |
| 3 | decode2k | 16 | 22.5 | 22.4 | +0.4% |

The 16-pair run is the one to believe: the harness's own 95% CI is -1.5..+2.7%
and the point estimate is +0.4%, i.e. **0**. Samples 1 and 2 disagree because a
`decode` quad puts B's second run right after B's 8192 sweep, which leaves the
expert page cache cold for that run (the same quad shows a 25% first-token
spread that no attention kernel can cause); `decode2k` repeats one frontier per
run so every sample starts from the same cache state, and its per-run spread
collapses from +-5% to +-1%.

Mechanism, and the reason this is a *reject* rather than a "keep it, it is free":
both halves remove dispatches that were already overlapped.

- The paired matvec merges 640 + 256 threadgroups into one 896-threadgroup grid.
  Wave count is identical (896/40 vs 640/40 + 256/40), and two back-to-back
  dispatches in one encoder already overlap at the boundary, so there is nothing
  to win. Measured: nothing.
- The paired norm is worse than free: each norm is **one threadgroup**, i.e. one
  GPU core. Two separate dispatches run on two cores at once; the paired kernel
  puts 10 + 4 simdgroups into a single threadgroup on **one** core and
  serializes them behind two threadgroup barriers. That is the sign of the
  sample-2 measurement and the reason the pair must not be revived as-is.

Calibration this adds to the dispatch model: a removed dispatch is worth ~2.2 us
when it is a dependency stall (the bf16-store matvec, +2.8%) and ~0 us when it is
already pipelined behind a neighbour. **Do not size a fusion by dispatch count
alone.**

### The second kernel, `rope_quantize`: also zero

The next item on the handoff's list was the fork's `rope_store` /
`ds4_gpu_dsv41_rope_quantize`: KV RoPE, the Q8_0 quantize and the window store in
one dispatch instead of three. It looked like the better candidate because the
three kernels are *serially dependent*, so unlike the projection pair there was a
real stall to recover.

| kind | pairs | A (fused) | B (three dispatches) | gain |
|---|---:|---:|---:|---:|
| decode2k | 16 | 22.6 | 22.6 | **0.0%** (CI -1.1..+1.2) |

Rejected and reverted. All three kernels are 1-16 threadgroups, so the fused
version is still one threadgroup wide at each stage and the dependency chain is
unchanged; there was no stall to remove, only a launch.

**Third item, never ported.** The fork's `ds4_gpu_dsv41_attention_decode` (fused
gathered decode attention) was dropped rather than measured: this tree already
has `ds4_gpu_encode_flash_attention_gathered_heads`, which takes the same
pre-gathered `comp_kv`, and the fork's extra `comp_ids` gather only removes a
~116 MiB/token f32 round-trip through `selected_kv` plus one dispatch. At the
measured cost of a non-stall dispatch (~0) and ~450 GB/s, that is ~1%, inside the
noise this harness can resolve. The mailbox below was worth 14x more and was
found instead.

Reverted: `metal/dense.metal`, `metal/norm.metal`, `ds4_metal.m`,
`ds4_deepseek41_gpu.h`, `ds4.c` are back to the bf16-store-only state.

## The decode gap was the per-layer selected-id readback (2026-10-22)

The two ported kernels above are neutral, so the handoff's premise ("dispatch
count is the decode gap") was re-measured instead of trusted. A dispatch is
worth ~0 unless it is a dependency stall, and the earlier +2.8% matvec cut was
exactly that: a stall, not a launch count.

**Measurement that found it.** `DS4_METAL_CB_TIMES=1` on a 2048/64-token decode
splits the token into 43.3 ms of wall = 21.9 ms GPU span + 7.2 ms host encode +
~14 ms of per-command-buffer latency, over **41 command buffers per token**
(`decode_cbs = 2624 / 64`). The fork runs **3**. The cause is
`ds4_gpu_routed_moe_one_tensor`: the streaming routed MoE needs the router's
selected expert ids on the host to build the expert address table and to decide
the SSD reads, and it gets them by calling `ds4_gpu_end_commands()` (commit +
`waitUntilCompleted`) and reading the tensor back — once per layer, 40 times per
token. Host and GPU therefore alternate one layer at a time.

**Upper bound, measured.** `DS4_MOE_REPLAY_SELECTED_IDS` feeds the same ids from
the host and skips the readback entirely (record with `DS4_V41_TRACE_EXPERTS`,
strip the 8-byte pos/layer header of each 32-byte record into 24-byte entries):

| 2048, 32 tokens | decode t/s | first token ms | token ids |
|---|---:|---:|---|
| default | 22.52 | 71.2 | (reference) |
| replay, no readback | **31.86** | 47.6 | identical |

Same tokens, +41%. So the whole gap to the fork, plus more, is this sync.

**The fix (`DS4_METAL_DISABLE_V41_SELECTED_MAILBOX` rolls it back).** A mailbox
kernel (`kernel_dsv41_selected_publish`) writes the router's ids, a checksum and
a monotonically increasing tag into a CPU-visible shared ring (64 slots), tag
last. The host takes them with `ds4_gpu_dsv41_selected_take`, which commits the
open buffer *without waiting* and polls the tag for at most 20 ms; a timeout or
a checksum mismatch falls back to the existing blocking readback, so a stall can
never hang and the worst case is today's behaviour. With the ids in hand the
host keeps encoding into the same pipeline: **41 commit-and-waits per token
become 41 commits and 1 wait** (the token's final drain).

| 2048, 32 tokens | decode t/s | first token ms |
|---|---:|---:|
| before | 22.52 | 71.2 |
| mailbox | **26.69** | 64.0 |

Paired A/B, `--kinds decode2k --bitwise`, 16 pairs, A = mailbox, B =
`DS4_METAL_DISABLE_V41_SELECTED_MAILBOX=1`:

| metric | A | B | change | 95% CI |
|---|---:|---:|---:|---|
| decode 2048 | 25.1 | 22.0 | **+14.1%** | +12.4 .. +15.7 |
| first token 2048 ms | 63.3 | 66.8 | **+5.2%** (faster) | +2.7 .. +9.8 |

`correctness PASS (tokens; bitwise)`. Both arms report the same expert-cache
counters, and the run's own token string matches the replay arm's md5, so the
ids the host acts on are the router's own.

**Why it stops at +14% and not the replay's +41%.** The take still waits for the
GPU to *reach* the publish, which is the layer's last encoded dispatch, so the
pipeline is depth one: the host's encode of layer N+1 overlaps the GPU's layer N,
but the host cannot run a whole token ahead. Going further needs the expert
address resolution itself on the GPU (the fork's `ROUTER_MM` / `PACKED_INDEX`
structure, i.e. a device-side id -> address table), which this tree has only for
the MXFP4/IQ2-both paths, not for V4.1's IQ2_XXS-gate / Q2_K-down recipe.

**Correction to the previous session's attribution.** `PERF_RECORD` claimed "GPU
busy/layer 0.356 ms here vs the fork's 0.231". That comparison is not
apples-to-apples: the fork's decode buffers are never waited on individually, so
`GPUEndTime - GPUStartTime` reads back as 0 for them (its 64-token run shows
189.6 ms of GPU span for 64 tokens, i.e. 3.0 ms/token, which cannot be the same
math). Measured on this tree alone, the token is 21.9 ms GPU + 7.2 ms encode +
14 ms of buffer latency. Kernel efficiency was never the lever; the sync was.

## After the mailbox: the pipeline is saturated, the lever moved to GPU work (2026-10-22)

`DS4_METAL_GPU_BUSY_PROFILE=1`, 2048, mailbox on (`-n 0` subtracted from `-n 32`):

| | wall/token | GPU busy/token | GPU duty |
|---|---:|---:|---:|
| before mailbox | 43.3 ms | 22.3 ms | 52% |
| mailbox | 36.2 ms | **35.1 ms** | **97%** |
| replay (no host at all) | 32.5 ms | 34.5 ms | ~100% |

The same work now takes 35.1 ms of GPU time instead of 22.3 ms, so the wall gain
(+17%) is much smaller than the idle it removed (21 ms of 43.3). Sustained load
is either clocking the GPU down or contending for bandwidth; there is no
frequency counter on this machine (`mactop` absent) so the split is not measured.
That reading is **inflated**: the replay arm reports *less* wall than GPU busy,
which cannot happen for sequential buffers, so CB spans overlap somewhere in the
sum (the expert-cache blits and the pipeline's unwaited buffers). Do not size the
remaining gap from it.

**Sized correctly, by interleaved A/B (3 pairs, same binary, same session):**

| 2048, 32 tokens | decode t/s | wall/token |
|---|---:|---:|
| mailbox | 27.33 / 27.66 / 27.54 | 36.3 ms |
| replay (no host wait at all) | 31.88 / 32.15 / 31.32 | 31.4 ms |

So the whole remaining sync prize is **+15.8%**, and it is **not** the poll: the
two arms report byte-identical expert-cache counters (495,147 hits, 11,013
misses, 102.09 GiB pread, 8.11 s pread) and identical decode-phase preads
(179 misses, 1.66 GiB, 118 ms over 32 tokens). It is the **40 Metal command
buffers per token**: `ds4_gpu_flush_commands` allocates and commits one buffer
per layer so the GPU can reach that layer's publish, ~0.12 ms each. The
bookkeeping around it is a handful of integer ops (`note_batch_committed`), so
the cost is the driver's buffer creation + commit.

**What removing it would take, and why it is not a small change.** Fewer buffers
per token means the host no longer needs the ids between the router and the
gather, i.e. expert addresses resolved on the device. But the gather also must
not read a slot before its pread lands, so the port is three coupled pieces:
(1) a per-layer device id->slot address table plus a resolve kernel, (2) the
pread issued asynchronously and its completion signalled to the GPU through a
CPU-signalled `MTLSharedEvent` with `encodeWaitForEvent` before the gather, and
(3) the host allowed to encode several layers ahead. Correctness then rests on
the event wait, not on a "misses are rare" assumption — decode still takes 5.6
misses per token, so a blocking fallback per miss would cost more than the 4.9 ms
it saves. The async machinery partly exists (`g_stream_expert_pread_pool_threads`,
`g_stream_expert_pending_load`, `begin_selected_load`/`pending_load_finish`), but
the device table has to be maintained at every cache install/evict point.

**Past 31.4 ms the lever is GPU work per token.** The routed experts read
6 x 9.49 MiB per layer (2.28 GB/token) and the shared expert 37 MB/layer
(1.5 GB/token); at ~450 GB/s that is ~8 ms, against ~35 ms measured, so the
single-token kernels run at roughly a third of the machine's bandwidth —
occupancy, not bandwidth.

For scale: the routed experts read 6 x 9.49 MiB per layer (2.28 GB/token) and the
shared expert 37 MB/layer (1.5 GB/token), which at ~450 GB/s would be ~5 + 3 ms.
Measured GPU work is 35 ms, so the single-token kernels are running at roughly a
third of the machine's bandwidth — occupancy, not bandwidth. That is where a
further decode gain has to come from (MPP/tensor-unit single-row shapes, or the
fork's `matmul_expand_rows` style widening), and it is a kernel project, not a
scheduling one.

## Why the last 15.8% is not reachable, and where the GPU time is (2026-10-22)

**The sync path is closed.** The replay arm is not a valid upper bound: it reads
the ids from a file, so the host is free to run ahead of the GPU and the
synchronous expert pread is hidden behind GPU work (both arms report identical
cache counters: 495,147 hits, 11,013 misses, 102.09 GiB, 8.11 s). In real
inference the ids only exist once the GPU's router has run, and the pread must
land before the gather that uses them, so the host can lead by at most one layer.
Two ways to spend that one layer of slack, measured:

| variant (2048, 32 tokens) | decode t/s | what it removes |
|---|---:|---|
| replay: ids for free, 3 CB/token | 31.7 / 32.2 | the poll *and* the commit |
| replay + one forced commit per layer | 28.8 / 31.7 | the commit only (~1-2 ms) |
| mailbox (landed) | 26.9 / 27.3 | nothing: 1 commit + 1 poll per layer |

So the poll is ~3-4 ms and the commit ~1-2 ms. Replacing the poll with a
GPU-side event wait moves that 3.6 ms onto the GPU as pread stall (5.6 misses
per token, 0.66 ms each = 3.7 ms) — an even trade, no net gain. **Device-side
address resolution alone cannot pay for itself; the alternation is forced by
"GPU picks the experts, host must do the I/O".**

**Where the GPU time actually is.** `DS4_METAL_V41_SECTION_PROFILE=1` (added;
cuts one waited command buffer per section so `DS4_METAL_CB_TIMES` can attribute
GPU spans) on a 2048/16-token decode:

| section | GPU per layer | share |
|---|---:|---:|
| attention (HC mix + 5 projections + flash + output) | 0.225-0.244 ms | 22-24% |
| router + shared expert | 0.225-0.231 ms | 22-23% |
| routed gather + sum | 0.231-0.300 ms | 23-30% |
| after_moe (HC post) | 0.233-0.300 ms | 22-30% |

Four nearly equal buckets, each only ~0.14 ms of real GPU work (the waited
buffer adds ~0.1 ms of latency per section). **There is no hotspot to fix.** The
"1/3 of peak bandwidth" reading is dozens of small kernels each losing to launch
and occupancy, which is why the next decode step is a kernel-efficiency project
(fewer, wider kernels per token) and not a patch.

## S3-wide explicit prefill is now the default: +18% / +34% (2026-10-22)

The explicit expert sweep was opt-in behind `DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT`
because of its first-token cost. Both halves are now measured, and the tokens are
identical, so it is default-on with a profitability floor.

Interleaved A/B, 3 pairs, same binary, `--ssd-streaming --ssd-streaming-cache-experts 72GB`:

| prefill | default (sweep) | disabled (mmap) | change |
|---|---:|---:|---:|
| 2048 t/s | 202.2 / **161.3** / 160.4 | 137.9 / 136.5 / 136.6 | **+18.1%** (median) |
| 8192 t/s | 483.5 / **372.2** / 366.9 | 280.9 / 270.4 / 278.3 | **+33.7%** (median) |

Single 8192/16-token confirmation: 504.4 vs 424.3 (+19%), steady decode 28.61 vs
26.89. Decode is not regressed by the change (it is slightly better, which is
most likely the different expert-cache state the sweep leaves behind).

**Correctness**: `--show-output` token ids are identical with the sweep on and
off at 2048 (`e36dac1fe120`) and at 8192 (`cedd70f358ac`). The sweep reads the
same weights through a whole-layer pread instead of mmap paging.

**The first-token cost, and why the floor exists.** After a sweep the first decode
step costs ~0.3 s instead of ~0.06 s. It is one-time (steady decode is unchanged)
and it is *not* I/O: the sweep leaves the cache mostly bypassed but the first
token still hits 237 of 240 expert lookups, `miss_pread` is 0.03 GiB, and the
stalls show as command buffers with `gpu span 0.00` and 20-130 ms of host encode.
It is present with the prefill seed off and with or without the decode mailbox.
Not root-caused. Because it is a fixed cost, the sweep only pays for itself past
roughly `count x (1/137 - 1/167) = 0.3 s`, i.e. ~230 rows, so:

- floor `DS4_V41_PREFILL_EXPLICIT_MIN_ROWS` (default 256; 128-row prompts keep
  the old mmap behaviour and the old first-token latency)
- `DS4_METAL_DISABLE_V41_PREFILL_EXPLICIT=1` rolls back to mmap prefill
- `DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1` still forces it on (floor 32), so the
  previous opt-in workflow is unchanged

At 2048 rows the trade is still a clear win end to end: 12.4 s + 0.40 s = 12.8 s
to first token, against 14.4 s + 0.06 s = 14.5 s on mmap.

Guards already in `ds41_prefill_expert_buffers_init` keep the sweep off the
resident path, TP, quality mode and encoder-resident sweeps, and it falls back to
mmap when the two layer slots do not fit the streaming prefill budget or cannot
be mlocked.

## Step log

| Step | Change | Verdict | Where |
|---|---|---|---|
| P0 | A/B harness + frozen baseline (this document) | accepted, tooling only | `speed-bench/ab_bench.py`, `speed-bench/readme_bench.py`, `tests/test_ab_bench.py`, `ds4_bench.c` (`--frontiers`, `--cache-stats`, token ids), `ds4_engine_memory_report` |
| P1 | `270` expert cache locked at engine open | **rejected**: decode neutral within noise (+1.7 s open); first-token signal is the decode kind's own -4% order bias plus ~-2.4%; no lazy-mlock failure to prevent at 64.88 GiB | `speed-bench/sf-port-p1p2-20261021.patch` |
| P2 | `131` bounded status poll at the decode readback | **not applicable as ported**: the edited functions never run on the V4.1 streaming path (0 profile lines); the run doubles as the decode-kind A/A control | same patch |
| P4a | S1b decoder suffix threshold 4096 → 2542 | **neutral**: ttft 7500 +0.8% (CI -1.4..+2.9), 6 pairs; the fork's gain was short_sweep, not the threshold | env-only probe |
| P4b | S2 batched candidate selection | **not adopted**: needs two new Metal kernels for a +2~4% ttft effect inside the ±5% cold band | static analysis |
| P3a | wide-sweep prepare depth 1 → 2 | **neutral** (ttft 7500 +1.0%, 6 pairs), but admission may have kept one slot; not a proof that reads are hidden | `prepare-depth2-probe` |
| S3-wide | explicit expert buffers admitted to wide sweeps (drop `count<=2048 && !wide`) | **accepted behind the opt-in switch**: prefill 8192 increment +36% (2 pairs), first token 8192 +25.3% (CI +22.5..+28.1), 32K +14.6%, decode 2048 +0.7%, decode 8192 -3.4%; bitwise PASS. **2048 first token still -83%** (68 -> 405 ms), so the switch cannot default on yet | `ds4.c` `ds41_prefill_expert_sweep_supported`; `wide-explicit-bitwise-20261022` |
| S3-wide/TTFT | hypothesis "seed slot->cache copy is unordered against the next layer's slot overwrite" | **refuted**: the seed blit is inside the batch CB and `end_commands` waits; seed off still leaves 2048 first token at 307 ms. New lead: ~19 x 20 ms host stalls (commit->done 20 ms, GPU 0.2 ms, no pread) = cache-entry eviction churn | `HANDOFF-S3-WIDE-20261022.md` §3 |
| decode-bf16-store | Q8_0 matvec + weighted RMS norm store BF16 directly (2 dispatches -> 1 each) | **accepted**: 1,444 -> 1,166 dispatches/token; matvec alone +2.8% (CI +0.9..+4.7), both together +0.7..0.9%; tokens + bitwise PASS; rollback envs `DS4_METAL_DISABLE_V41_{MATVEC,NORM}_BF16_STORE` | `metal/dense.metal`, `metal/norm.metal`, `ds4_metal.m`, `ds4_gpu.h`, `ds4.c`; `matvec-bf16-store-20261022`, `bf16-store-fusions-20261022` |
| decode-dispatch | dispatch counter on both trees: ours 1,444/token vs fork 890; attention 25.8/layer of 35.9, project 10.0 = 5 calls x 2 (MPP matvec + separate BF16 round) | **accepted as the decode roadmap**: host-encode bound, ~13 us/dispatch; the 554 extra dispatches are the 7 ms/token gap. MPP+BF16 store fusion is step 1 (+10% projected) | counter + per-section counters, removed after measuring; `PERF_RECORD` section above |
| decode-fast-lookup | allocation-free pipeline cache in `ds4_gpu_get_pipeline`, armed per decode step | **rejected: neutral** (23.65 vs 23.66 t/s, 3 runs). The fork's +5.1% add-back was single-run noise; the fork's own decode varies 4-7% between runs | reverted; no code kept |
| decode-attr | fork decode glue ablation ("fork has no switches" premise) | **premise false**: 15 decode switches exist; all-off -16% decode (27.29 -> 22.94), group A off is *faster* (29.08), so the effect is an interaction. Port in marginal order after "all off + add back one" | `HANDOFF-S3-WIDE-20261022.md` §4 |
| decode-pair-project | fork's `project_pair_q8` + `norm_pair_rows` ported (q_a+kv in one dispatch, both norms in one) | **rejected: neutral** (+0.4%, CI -1.5..+2.7%, 16 decode2k pairs; bitwise PASS). Both halves merge work that already overlapped: the matvec pair has the same wave count, and the norm pair serializes two one-threadgroup kernels onto one GPU core. Calibration: a removed dispatch is worth 2.2 us only when it is a dependency stall | reverted; `project-norm-pair-2k-20261022`, section above |
| decode-rope-quantize | fork's `rope_store` / `ds4_gpu_dsv41_rope_quantize`: KV RoPE + Q8_0 quantize + window store in one dispatch (3 -> 1) | **rejected: neutral** (0.0%, CI -1.1..+1.2, 16 decode2k pairs, bitwise PASS). All three stages are 1-16 threadgroups, so fusing them keeps one threadgroup wide per stage and the serial chain is unchanged: there was a launch to remove, not a stall | reverted; `rope-quantize-2k-20261022`, section above |
| s3-wide-default | `ds41_prefill_expert_sweep_supported` default-on with a 256-row profitability floor (was opt-in) | **accepted**: prefill 2048 **161.3 vs 136.6 (+18.1%)**, 8192 **372.2 vs 278.3 (+33.7%)** (3 interleaved pairs, medians); tokens identical at 2048 (`e36dac1fe120`) and 8192 (`cedd70f358ac`); steady decode not regressed (28.61 vs 26.89). The sweep's one-time ~0.3 s first-token cost is **not root-caused** (not I/O: 237/240 lookups hit, `miss_pread` 0.03 GiB) so the floor keeps sub-256-row prompts on mmap; rollback `DS4_METAL_DISABLE_V41_PREFILL_EXPLICIT=1`, floor `DS4_V41_PREFILL_EXPLICIT_MIN_ROWS` | `ds4.c`; section above |
| decode-q8-nsg | sweep `DS4_METAL_Q8_MV_NSG` 1/2/3/6/8 against the default 4 | **rejected: default already optimal**. 2048/32tok decode: 4 -> 27.57, 2 -> 26.87, 3 -> 26.18, 6 -> 26.09, 1 -> 25.62, 8 -> 23.94. No free win in the shape knob; the Q8_0 matvec geometry is where it should be | one run each; do not re-run |
| decode-selected-mailbox | publish the decode router's ids into a GPU-written shared ring; the host polls instead of ending the command buffer | **accepted**: decode 2048 22.0 -> 25.1 (**+14.1%**, CI +12.4..+15.7, 16 decode2k pairs), first token 66.8 -> 63.3 ms (+5.2%); single runs 22.5 -> 26.7 t/s; 41 commit-and-waits/token -> 1; tokens + bitwise PASS; rollback `DS4_METAL_DISABLE_V41_SELECTED_MAILBOX`; 20 ms poll deadline falls back to the blocking readback | `metal/dsv41.metal`, `ds4_metal.m`, `ds4_deepseek41_gpu.h`, `ds4.c`; `selected-mailbox-2k-20261022`, section above |
| decode-replay-bound | `DS4_MOE_REPLAY_SELECTED_IDS` supplies the ids from the host (no readback at all) | **measured as the upper bound**: 22.52 -> 31.86 t/s (+41%) with identical tokens. Not a shipping path; it is what sizes the mailbox (+14%) against what a device-side expert address table would be worth | `selected-mailbox-20261022` section above |
| harness | new `decode2k` kind (one 2048 frontier, 128 gen) for sub-1% gates | **accepted, tooling only**: repeating one frontier per run keeps the expert page cache in the same state at every sample, which removes the `decode` kind's position bias (B's second run follows B's 8192 sweep) and buys ~2x the pairs per minute; 16 pairs at +-1% per run instead of 6 at +-5% | `speed-bench/ab_bench.py`; `tests/test_ab_bench.py` 42/42 |

## Adding a row

1. Build the start build into its own tree (the A side): `cp ds4-bench /tmp/ds4-base/`
   plus a no-op `Makefile` and a `metal` symlink, or a real `git worktree`.
2. `python3 speed-bench/ab_bench.py --a /tmp/ds4-base --b . --kinds <kinds of the
   change> --guards <guards>` exits 0: tokens identical to the start and every
   headline metric with two pairs or more. Add `--bitwise` when the change claims
   bit-identical output.
3. Run the repo gates above for the change's risk class.
4. Paste the printed `record row:` under the current segment and add a step-log line.
   A cell of a kind that was not run stays empty; do not fill it from another session.

### Reading the `gain` column

The harness prints **B relative to A**: `B/A` for every rate and `A/B` for every
time (`ab_bench.py` `gain()` / `unit()`, design D9). So a printed **+1% always
means B is better than A**, whichever metric it is. This tree runs the candidate
on A and the rollback on B (`--b-env DS4_METAL_DISABLE_...=1`), so the number to
quote as "the change is worth" is the **negation** of the printed gain, and a
printed negative gain is a win. Every `+x%` in the tables above is already in
"the change is worth +x%" form; `matvec-bf16-store-20261022/summary.txt` prints
`-2.8%` for the same measurement.

## Segments

A segment starts with an A/A run on its start build. A change that alters greedy
output makes the token gate against the old start fail by design; it then opens a new
segment with a new start row, and the previous segment's rows stay as they are.
