# sf-ds4-1flash 提速宣称 → ds4 可借鉴性评估

> 评估日 2026-10-21。只做静态核对 + 既有 CSV 粗比：**不加载模型、不做新 A/B**。
> fork：`https://github.com/Chida82/sf-ds4-1flash` @ `42bc593`（2026-10-12）；README 性能表对应
> `fa51b38`（2026-10-11），baseline 是上游 ds4 `0aaea5a`（2026-09-20）；fork 之后 sync 到
> `fc80bd6`，但性能表早于该 sync。
> 我们树：`HEAD 89e068c`（2026-10-21），M5 Max 128 GB，V4.1 Flash Q2，SSD streaming。
> 当前 8055 `ds4-server` 占约 85 GB RSS，所以本次没有重跑 fork 的 `readme_bench.py`；
> 下面的“我们树”数字来自 `speed-bench/v41_m5max_ab/auto-1.csv`、`auto-2.csv` 与
> `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1 的既有记录。
>
> **勘误（相对上一轮口头结论）**：我上一轮说“`ds4_gpu_stream_prefill_bind_layer` API 和测试已在树内”
> 是误把 fork 的 `tests/test_deepseek41_graph.c` 当成本地文件。逐文件核对后：**本地没有**
> `ds4_gpu_stream_prefill_bind_layer` / `ds4_gpu_stream_prefill_discard_buffer` /
> `streaming_prefill_bytes` / `ds41_prefill_expert` / `ds41_prefill_expert_pread`；
> 这套 explicit prefill 基础设施是 fork 的 `60/100` 新增，需要整块移植（见 §4 P3）。
> 本地已有的是 18-thread `pread` pool、slab 分配器与 mlock relief，不是这套 explicit buffer driver。

---

## 0. 一页结论

**fork 的宣称在其测量口径下基本属实，但不能把 README 的百分比直接当成我们当前树的提升空间。**

- fork 自报：prefill **+45.0%~+99.4%**、steady decode **+61.7%~+78.3%**、
  generation（含首 token）**+109.3%~+129.2%**、“first token after prefill” **2.565 s → 0.185 s**。
  它的 `perf-record` 给出 baseline 分解、A/B 轮转、thermal/page-cache 控制、parity/bitwise 闸门；
  **没有发现计时作弊**。
- 三个限定：① “first token 185 ms”是 **prefill 完成后的第一个 decode token**，不是总 TTFT；
  ② “generation”含首 token，`steady decode` 才是每 token 速度；
  ③ baseline 是 **`0aaea5a`（2026-09-20）**，不是我们当前树。
- 按我们树内 auto-cache CSV（64.5 GiB、512 greedy）粗比，fork 的领先只有
  **decode +18%~+30%、prefill +20%~+84%**，不是 README 的 +62~78% / +45~99%。
- 首 token 不是全面领先：**2K/4K 我们更快**（77/75 ms vs fork 185/144 ms），
  **8K/32K fork 更快**（152/194 ms vs 我们 368/279 ms）。这主要是 fork 的 82 GB cache
  与显式 expert buffer / 页对齐 reads 的权衡，不是通用优势。
- 真正值得借的：**P0 A/B harness 方法论；P1 `270` cache lock；`268` 页对齐 reads（条件：先有 P3 driver）；
  P2 `131/132/210` readback；P3 `60/100` explicit expert buffers（最大 prefill 项）；
  P4 `60` S1b/S2、`#1125`、`#1073` 缺口**。不要借 `--boost`/fanboost、dual-drive、
  整树 prune、盲目 82 GB cache。

一句话：**值得借鉴，但“+78% decode”是相对旧 ds4 的数字；我们树已经吃掉了 #1041/#1042/#1090/#1120/#1192
等大头，剩余肉在 prefill 的 explicit buffers/页对齐 reads 和 decode 的 readback/mailbox。**

---

## 1. 口径与证据

### 1.1 fork 的测量口径

| 维度 | fork 值 |
|---|---|
| 机器 | Apple M5 Max，128 GB unified memory |
| 模型 | DeepSeek V4.1 Flash Q2 GGUF |
| 模式 | `--ssd-streaming --ssd-streaming-cache-experts 82GB` |
| prompt | `speed-bench/promessi_sposi.txt`（I Promessi Sposi） |
| 采样 | greedy，128 tokens；`generation` 含首 token，`steady decode` 去掉首 token |
| 上下文 | 2048/4096/8192/16384/32768（`--step-mul 2`） |
| 对比 | ds4 `0aaea5a` vs fork `fa51b38`；A B B A 轮转、median、bootstrap 95% CI |
| 热/页缓存 | 每列前冷到 43.2~46.5 °C；ds4 先跑、sf 后跑；sf 首轮丢弃；page-cache 不对称在文档中明说 |
| 输出 | parity oracle 10 prompts token-identical；每个 step token gate；声称 bitwise 的过 `--bitwise` |

证据文件：`README.md`、`docs/PERFORMANCE.md`、`speed-bench/perf-record.md`、
`docs/upstream-prs.md`、`speed-bench/README.md`。

### 1.2 我们树的口径

| 维度 | 我们树 |
|---|---|
| HEAD | `89e068c`（2026-10-21） |
| 机器 | 同一台 M5 Max 128 GB（`Mac17,6`） |
| 模型 | `gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf`（生产 server 使用） |
| 模式 | `--ssd-streaming`，auto cache：**64.51 GiB dynamic**（6959 experts） |
| 采样 | greedy，512 tokens（`auto-1/2.csv`） |
| 上下文 | 2048/4096/8192/16384/32768 |
| 已有速度记录 | `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1：steady decode 中位 2K 22.82 / 4K 23.66 / 8K 22.31 / 16K 23.33 / 32K 22.02 t/s（n=4） |

**这不是严格 A/B**：cache 大小（64.5 vs 82 GB）、生成 token 数（512 vs 128）、时间点（10-21 vs 10-11）
都不同；但方向足够用来判断“剩余空间量级”。

### 1.3 fork 的代码是“单模型 + Metal-only”的裁剪树

fork 从 ds4 0aaea5a 起做了 deep prune：删掉 CUDA/ROCm、多模型（GLM/Qwen）、agent、部分 DSpark/MTP，
只保留 V4.1 Flash + Metal + SSD streaming。裁剪本身不是运行时速度来源，但它让 fork 可以
把 cache/buffer/readback 做成单一路径；移植到我们多模型/多后端树时，必须重新做准入与回归。

---

## 2. fork 的数字与拆解

### 2.1 宣称数字（README “Speed” 表）

| 指标（ctx） | ds4 `0aaea5a` | sf | 增益 |
|---|---:|---:|---:|
| prefill 2048 | 121.1 t/s | 175.7 | +45.0% |
| prefill 4096 | 113.9 | 227.0 | +99.4% |
| prefill 8192 | 202.8 | 398.5 | +96.4% |
| prefill 16384 | 354.2 | 559.5 | +58.0% |
| prefill 32768 | 397.7 | 702.8 | +76.7% |
| generation 2048 | 12.5 | 28.7 | +129.2% |
| generation 8192 | 13.3 | 27.8 | +109.3% |
| generation 32768 | 11.7 | 25.7 | +118.8% |
| steady decode 2048 | 16.6 | 29.6 | +78.3% |
| steady decode 8192 | 17.7 | 28.6 | +61.7% |
| steady decode 32768 | 16.1 | 26.3 | +63.1% |
| first token after prefill 2048 | 2565 ms | 185 ms | -92.8% |

`first token after prefill` 是 `gen_first_ms`：prefill 结束后到第一个生成 token 的时间，**不含 prefill**。
2048 上下文总 TTFT ≈ `2048 / 175.7 ≈ 11.7 s` + 0.185 s。`generation` 的 +129% 大部分来自
2.565 s → 0.185 s 的首 token；`steady decode` 的 +78% 才是每 token 的真实提升。

### 2.2 为什么 baseline 首 token 要 2.5 s

`perf-record` 的分解（8K 上下文、一个 decode token ≈ 56 ms）：

- **78% 在 expert path**，不是“读权重慢”；
- expert path 里最大头是 **selected-id readback 的 `sync` ≈ 38 ms**；
- wide prefill sweep 后 decode cache 冷、page cache 被 evict，首 token 要 2.1~2.4 s；
- ds4 的 prefill 还会把 13 GiB 更多 GGUF 留在 page cache，后续 sf run 会因此丢 4~6%。

fork 的对应改动：

| 改动 | 机制 | 主要实测 |
|---|---|---|
| `30` #1041 | 单盒 decode layer queue，逐层 commit 不等待 | decode 2048 +13.1%、8192 +13.3%；ttft 2500 +8.4% |
| `40` #1125 + pool | expert read path、slot leak/in-flight 修复、slab residency、18 threads、split 4 | slab residency first token 2048 +82.6%、8192 +34.8%；decode 8192 -0.37%（CI -0.84..-0.10） |
| `60` | one-sweep tail、decoder suffix 2541、index batches、单 chunk 显式 expert buffers | ttft 3500 +79.3%、7500 +39.4%（S1a）；7500 +30.2%（S1b）；7500 +2.3/+1.69/+4.3%（S2）；2500 +14.2%、append+1500 +42.6%（S3） |
| `70` #1073 | decode glue fusions（HC、q_a/kv、RoPE/quantize、attention gather、shared gate/up、Engram workers） | 累计 decode 2048 +5.7%、8192 +6.2%；S2 +1.15%、S3 +1.14% |
| `100` | wide sweep 显式 expert buffers + RAM-before-disk（decode cache 的 ready experts 先 blit 到 slot） | ttft 3500 +23.3%、5000 +8.5%、7500 +19.0%、10000 +23.9%；append+1500 +34.8%；首 token 3500 1373→258 ms、7500 1372→297 ms、10000 164→125 ms |
| `131/132` | bounded status poll（2 ms 再阻塞）、router 单独 commit | +2.8~3.1% / +1.6~2.2% decode |
| `210` | selected-id mailbox：router 最后一个 one-thread kernel 把 6 个 id + checksum + seq 写进 64 B box，CPU 轮询；命中只等更早的 CB | decode 2048 +5.5%、8192 +6.2%；append+300 +5.8%；prefill 5000 +4.7% |
| `268` | `F_NOCACHE` 读按 16 KiB 页对齐（partial first page 单独读） | first token 8192 1230→150 ms；ttft 10000 +5.7%；append+1500 +12.5% |
| `270` | engine open 时提前分配并 mlock 所有 cache slot | first token 2048 209→190.5 ms、8192 157.5→140.1 ms；prefill 5000 +1.8%；decode 2048 +2.4%（CI -0.6..+3.5） |

### 2.3 证据链的强弱

**强**：每个 step 有 token gate，bitwise step 有 `--bitwise`；A/B 是 A B B A 轮转；
parity oracle 10 prompts token-identical；`--boost` 单列，主表不含 boost；page-cache/thermal 不对称在
`perf-record` 里明说并做控制；`perf-record` 有 “Rejected ideas” 表，负结果保留。

**弱/需注意**：

- 只有一台机器、一个模型、一个量化、一个 cache 档、一条 prompt；不能外推到其他 Mac/模型/驻留态。
- baseline `0aaea5a` 是 2026-09-20；fork 之后 sync 了 `fc80bd6`，但性能表是 sync 前的 `fa51b38`。
- 首 token 是最噪声的指标，且依赖 page cache；它自己的 README 也说 “first token depends on which
  expert pages earlier processes left in the page cache”。
- `--boost` 是 fanboost（全速风扇），不是软件加速；主表没有它。
- fork 的 82 GB cache 比我们 auto 64.5 GiB 大；它自己的 `270` 数据显示 86/88/90 GB 会让首 token
  明显变差（86 GB：8192 首 token 134→157 ms；90 GB：121→301 ms），所以 cache 大小是权衡，不是越大越好。

---

## 3. 对我们当前树的实际对比

### 3.1 粗比表

我们树 `auto-1.csv` / `auto-2.csv`（auto cache 64.51 GiB，512 greedy）vs fork（82 GB，128 greedy）：

| 指标（ctx） | 我们当前树 | fork | fork vs 我们 |
|---|---:|---:|---:|
| prefill 2048 | 131.6–135.5 t/s | 175.7 | +30~34% |
| prefill 4096 | 188.7–190.8 | 227.0 | +19~20% |
| prefill 8192 | 216.4–228.7 | 398.5 | +74~84% |
| prefill 16384 | 368.6–392.8 | 559.5 | +42~52% |
| prefill 32768 | 507.3–515.9 | 702.8 | +36~39% |
| steady decode 2048 | 22.77–23.02 | 29.6 | +29~30% |
| steady decode 8192 | 22.45–22.78 | 28.6 | +26~27% |
| steady decode 32768 | 22.05–22.25 | 26.3 | +18~19% |
| first token 2048 | 76.95–78.45 ms | 185 | 我们更快 |
| first token 8192 | 278.6–368.3 ms | 152 | fork 更快 |
| first token 32768 | 259.4–279.3 ms | 194 | fork 更快 |

结论：**decode 剩余空间约 +18~30%，prefill 约 +20~84%；首 token 2K/4K 我们更快，8K/32K fork 更快。**
README 的 +62~78% decode 是相对 16.6 t/s 的旧 ds4，不是相对我们的 22~24 t/s。

### 3.2 我们树已经有什么

| fork 改动 | 我们树状态 | 证据 |
|---|---|---|
| `30` #1041 单盒 decode queue | **已有，默认开** | `LOCAL_INVENTORY.md` #1041 行（1/2）；`V41_M5MAX_BASELINE_SPEED_PLAN.md` §1 |
| `70` #1073 decode glue | **大部分被 #1042/#1090 覆盖** | `LOCAL_INVENTORY.md` #1042 6/7、#1090 3/7；`V41_M5MAX_BASELINE_SPEED_PLAN.md` §1 说 #1073 不再单独取 |
| `300` #1192 top-k | **已 pick** | `LOCAL_INVENTORY.md` #1192 行 |
| `40` 18-thread pread pool / slab / mlock relief | **已有类似物**；#1125 四项修复不在 ledger | `ds4_metal.m:15626`（18 threads）、`ds4_metal.m:1161`（mlock relief） |
| `60` S1a one-sweep tail | **已有类似逻辑**：`ds41_prefill_count` 的 medium sweep（2048<remaining<8192，tail>=256） | `ds4.c:43118` |
| `60` S1b decoder suffix | **有，但默认 4096**，没有 fork 的 short_sweep tile 对齐/128 warm；本地有 DSpark 128-row parity 约束 | `ds4.c:43424`、`ds4.c:43663` 附近的注释 |
| `60` S2 index batch | **有较旧的 `ds41_index_batch`**；没有 fork 的 candidates_batch / select-all / ratio-1 publish | `ds4.c:42482` |
| `60/100` explicit expert buffers | **没有**：本地没有 `ds41_prefill_expert`、`ds41_prefill_expert_pread`、`ds4_gpu_stream_prefill_bind_layer`、`streaming_prefill_bytes` | 逐文件核对；上一轮误把 fork 的文件当成本地 |
| `131/132/210` readback | **没有**：本地没有 `g_wait_poll_ns`、`g_readback_box`、split readback | `ds4_metal.m:43649` 的 `ds4_gpu_routed_moe_one_tensor` 路径与 fork 不同 |
| `268` 页对齐 uncached reads | **没有**（依赖 explicit prefill driver） | 无 `ds41_prefill_expert_pread` |
| `270` cache lock at open | **没有**；有 lazy mlock + relief | 无 `ds4_gpu_stream_expert_cache_lock_ahead` |
| `#1125` slot leak/in-flight/prune 修复 | **不在 ledger**；需单独评估 | `LOCAL_INVENTORY.md` 未列 #1125 |
| `#1073` 缺口 | 可能缺 q_a/kv、RoPE/quantize、attention gather 等小片 | #1042 6/7 未取 attention glue；#1090 3/7 |

---

## 4. 可借鉴清单（按优先级）

### P0：先借方法论，不先写运行时

**`ab_bench.py` + `readme_bench.py` + `perf-record.md` 纪律**。我们树目前只有
`speed-bench/v41_m5max_streaming_ab.sh` 和 CSV，没有 `ab_bench.py`。fork 的 harness 强制：
同 token gate、bitwise logits、A B B A 轮转、thermal/GPU clock 门控、cache-policy-change、
bootstrap 95% CI、`--sections` GPU section time。没有这套，后面的“+X%”很容易被路径漂移/热漂移污染。

**建议**：把 `ab_bench.py`（799 行）和 `tests/test_ab_bench.py` 作为独立工具移植，不碰引擎；
先冻结我们 HEAD 的基线（同 GGUF/flags/prompt、auto cache、512/128 tokens），再谈改动。

### P1：小改动、独立性强

**`270` cache lock at engine open**（fork `ds4_metal.m` +62 行，`ds4.c` +1 行）

- 机制：engine open 时按 `ds4_gpu_stream_expert_cache_configured_budget()` 分配所有 slab、mlock
  所有 slot，放进 free list；避免首次 prefill 的 explicit buffer/scratch 在 `vm.global_user_wire_limit`
  下把最后几个 slot 的 mlock 挤掉，导致 cache 缩到 flag 以下。
- fork 实测：first token 2048 209→190.5 ms、8192 157.5→140.1 ms；prefill 5000 +1.8%；
  decode 2048 +2.4%（CI -0.6..+3.5）；engine open 约多 2 s。
- 我们树：有 lazy mlock + relief（`g_stream_expert_cache_mlock_relief_applied`），没有 lock-ahead。
- 风险：engine open 内存占用提前；82 GB 以上才明显；需要按我们 cache 预算和 macOS wire limit 复测。

**`268` aligned uncached reads**（fork `ds41_prefill_expert_pread` +8 行）

- 机制：APFS `F_NOCACHE` 读不按 16 KiB 页对齐时，约 1/4 范围会漏进 page cache；
  expert range 每 3,041,280 B 起，partial first page 单独读，其余从页边界开始。
- fork 实测：first token 8192 1230→150 ms；ttft 10000 +5.7%；append+1500 +12.5%；bitwise。
- 我们树：**没有** `ds41_prefill_expert_pread`；只有先移植 `60/100` 的 explicit buffer driver，
  这个 8 行才有落点。若我们继续走 mmap page-in，这个 fix 不直接适用。
- 风险：低（若 driver 存在）；需要复现 APFS 行为并确认没有把页缓存行为改成负收益。

### P2：decode 剩余空间的主要候选

**`131/132/210` readback**（fork `ds4_metal.m` + `metal/dsv41.metal` 的 mailbox kernel）

- `131` bounded status poll：`waitUntilCompleted` 唤醒约 72 µs，poll status 2 ms 后回退，
  唤醒约 48 µs；+2.8~3.1% decode。
- `132` router 单独 commit：shared expert 不喂 selected-id readback，router CB 先 commit，
  CPU 只等更早的 CB；+1.6~2.2% decode。
- `210` selected-id mailbox：router batch 最后一个 one-thread kernel 把 ids/checksum/seq
  写进 64 B shared box；CPU 轮询，命中只等更早的 CB，20 ms 超时回退；+5.5~6.2% decode。
- 我们树：没有这三个机制；`ds4_gpu_routed_moe_one_tensor` 路径与 fork 不同（我们有多条
  Q4/IQ2/MXFP4 selected slot 路径）。**不能整包复制**；先把 fork 的 `ds41_moe_partial`/readback
  结构与我们 `ds41_moe_partial` 对齐，再选一条路径做 opt-in + 回退开关。
- 风险：中；命令缓冲/共享事件/内存序容易出错；必须有 token-identical + bitwise + DSpark
  streaming verify 闸门。

**`#1125` 四项修复**（fork `40` 的一部分）

- fork 说 #1125 修 slot leak / in-flight recycling / prune hang / cache 计数；并让 2048 首 token
  从 4.4 s 降到 2.4 s（+82.6%），但 decode 8192 -0.37%。
- 我们树：18-thread pool、slab、mlock relief 已有；但 #1125 不在 `LOCAL_INVENTORY.md`。
- **建议**：先把 #1125 作为独立 upstream PR 评估（它比 fork 的 440 行 ds4_metal.m 改动更可审计），
  只在我们能复现同类 slot leak/prune hang 时取；不要把它当纯速度项。

### P3：prefill 最大项

**`60/100` explicit expert buffers + RAM-before-disk**

- 机制：在 prefill reserve 内分配两套完整的 expert layer buffer（gate/up/down），
  `F_NOCACHE` pread 填进 GPU 消费的 storage，而不是先 mmap 预热再被下一次读 evict；
  下一层读与当前层 compute 重叠；wide sweep 也走这套；decode cache 的 ready experts 先 blit 到 slot。
- fork 实测：ttft 3500 +23.3%、5000 +8.5%、7500 +19.0%、10000 +23.9%；append+1500 +34.8%；
  首 token 3500 1373→258 ms、7500 1372→297 ms、10000 164→125 ms。
- 我们树：**没有**这套 API/driver；`ds41_graph_prefill_sweep` 的 wide sweep 走
  `metal_graph_stream_prepare_join_layer` 的 mmap page-in；small append 走
  `ds41_streaming_sweep_moe_gather_admitted` 的 selected-address gather（已比整层读省）。
- **建议**：先只做 wide sweep（>=2048 行）的 explicit buffer，避免和我们的 gather 路径重复；
  port 约 300–400 行（`ds41_prefill_expert_*` + `ds4_gpu_stream_prefill_bind_layer/discard` +
  `streaming_prefill_bytes` + `ds41_prefill_expert_pread`），复用 fork 的测试思路；
  必须过 bitwise、DSpark streaming、长上下文、SSD streaming + resident、Metal default。
- 风险：高（内存预算、mlock、F_NOCACHE、cache seed 交互）；收益也最大。

### P4：小-中，先对照再取

**`60` S1b decoder suffix 2541 / short sweep**

- fork：默认从 2541 行起用 decoder suffix；`short_sweep` 把 `first` 向下取整到 2048 tile，
  warm 128 行；ttft 3500 +2.6%、7500 +30.2%。
- 我们树：`ds41_decoder_suffix_min_rows()` 默认 **4096**，且没有 `short_sweep`/128 warm；
  本地注释明确说 DSpark 的 128-row 特征窗会改变 `first` 的 GEMM 分区，破坏 `--dspark` 的
  bitwise prefill parity。**不能直接改默认**；先重验 DSpark streaming short-prefill/verify，
  再决定是否接受 4096→2541 的收益。

**`60` S2 index batch**

- fork 的三项：ratio-1 batch publish、batched candidate blocks（`ds41_candidates_batch`）、
  select-all short index rows；ttft 7500 +2.3%、+1.69%、+4.3%。
- 我们树：`ds41_index_batch` 较旧（`ds4.c:42482`），没有这三项；`ds41_attention_batch`
  还有 `ratio == 2u` 的 Apple gate。可借，约 100 行；先做 ratio-1 publish 与 select-all，
  再评估 candidates_batch。

**`70` #1073 缺口**

- 我们树 #1042 6/7 已覆盖 decode MoE/HC/router/expand 的大部分；#1090 3/7 覆盖流水线。
- 可能缺：q_a/kv pair、RoPE/quantize/store、attention selected rows gather；单项约 0.4~0.8% decode。
- 建议：先按 fork `70` 的 S4a-g 与我们 `ds41_moe_partial`/`ds41_graph_layer` 逐项对照，
  只补缺口，不整包 800 行；先做 q_a/kv 与 RoPE/quantize 两个低风险项。

### 已具备 / 不需要动

- `30` #1041 queue：已取（1/2），默认开。
- `300` #1192：已 pick；DSv4.1 文本 prefill 主路无新增，decode 无感。
- `#1042` 6/7、`#1090` 3/7、`#1120` 5/5：已在树。

---

## 5. 不建议借 / 明确不要重开

| 项 | 原因 |
|---|---|
| `--boost` / fanboost | root fan daemon，全速风扇；主表不含 boost；不是软件加速；有硬件风险 |
| `140` dual-drive prefill | 需要第二块 TB5 盘；fork 的 upstream port 实测 **-13~38%**，不划算 |
| 整树 prune 成一模型/Metal-only | 裁剪不是运行时速度来源；会破坏我们多模型/CUDA/ROCm/agent |
| 盲目 82 GB cache | fork 自己的 86/88/90 GB 数据：首 token 明显变差、内存压缩；我们 auto 64.5 GiB 在 2K 首 token 77 ms，比 fork 185 ms 更快 |
| route-prediction prefetch / global LRU / CPU-GPU keep-alive / 更多 Engram readers | fork `perf-record` 的 “Rejected ideas” 已实测否掉；别重复 |
| 把 README 的 +78% 当我们的目标 | 那是相对 16.6 t/s 的旧 ds4；我们树已经是 22~24 t/s |

---

## 6. 验证闸门与建议顺序

**任何模型级测试前**：先停掉 8055 `ds4-server`（当前约 85 GB RSS），确认 free+inactive 内存
大于本次峰值；不要并发跑两个 huge model 进程。

**合入顺序**：

1. P0：移植 `ab_bench.py`/`readme_bench.py`，冻结我们 HEAD 基线；先跑 A/A。
2. P1：`270` cache lock；若 P3 先落地，再 `268` 页对齐 reads。
3. P2：`131/132/210` readback（先一条 opt-in 路径，带回退开关）。
4. P3：`60/100` explicit expert buffers（wide sweep 优先）。
5. P4：`60` S2 index batch；`60` S1b decoder suffix（先过 DSpark parity）；`#1125`；`#1073` 缺口。

**每项闸门**：

- token-identical（同 prompt、greedy）；
- `--bitwise` logits（声称 bitwise 的 step）；
- `./ds4_test --logprob-vectors`（匹配 checkpoint）；
- DSpark streaming `--short-prefill-ssd` 与 verify parity（`tests/test_deepseek41_dspark`）；
- `./ds4_test --long-context`；
- `tests/test_metal_ssd_experts`、`test-metal-command-memory`；
- SSD streaming + resident 两态；Metal default；有 CUDA/ROCm 访问时再跑对应后端。
- 负结果按 `LOCAL_INVENTORY.md` §D 登记，别重复。

**验收数字**：以我们 auto cache 64.5 GiB 的 `auto-1/2` 为基线，记录
prefill 2048/8192/32768、steady decode 2048/8192/32768、first token 2048/8192/32768；
任何改动只有在其目标指标上超过 bootstrap CI 且不拖累其他指标时才保留。

---

## 7. 参考

### fork

- 仓库：`https://github.com/Chida82/sf-ds4-1flash`
- `README.md`（Speed 表、first token 表、second drive）
- `docs/PERFORMANCE.md`（context sweep、A/B、记录基线）
- `speed-bench/perf-record.md`（Situation 0、Read path、Decode kernels、Prefill、Readback、Mailbox、Uncached reads、Cache above cap、Rejected ideas）
- `docs/upstream-prs.md`（78 PR registry、verdict、#1041/#1042/#1073/#1125/#1192 等）
- 关键提交：
  - `5410123`（30 decode layer queue）
  - `b044217`（40 SSD expert reads）
  - `0d88af9`（60 prefill sweeps/index batches/explicit buffers）
  - `6da9533`（70 decode glue #1073）
  - `896785c`（100 explicit expert buffers / RAM-before-disk）
  - `f8dd83e`（131 bounded status poll）
  - `67b75b8`（132 commit router alone）
  - `37a5d1e`（210 selected-id mailbox）
  - `220dd7d`（268 aligned uncached reads + 265 replica stamp）
  - `5b6482a`（270 cache lock at open）
  - `42bc593`（HEAD，300 intake）
- 性能表对应：`fa51b38`；baseline：`0aaea5a`；sync tag：`sync-fc80bd6`。

### 我们树

- `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1（steady decode 22.0–24.4 t/s；§1 #1041/#1090/#1035/#1073 盘点）
- `LOCAL_INVENTORY.md` #1041（L37）、#1042（L38）、#1090（L19）、#1120（L21）、#1192（L44）、#1073（L236）
- `speed-bench/v41_m5max_ab/auto-1.csv`、`auto-2.csv`（prefill/steady/first token）
- 代码坐标：
  - `ds4.c:42482` `ds41_index_batch`
  - `ds4.c:43118` `ds41_prefill_count`
  - `ds4.c:43424` `ds41_decoder_suffix_min_rows`
  - `ds4.c:43526` `ds41_graph_prefill_sweep`
  - `ds4_metal.m:43649` `ds4_gpu_routed_moe_one_tensor`
  - `ds4_metal.m:15626` 18-thread pread pool
  - `ds4_metal.m:1161` mlock relief
- 生产 server：`./ds4-server -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf -c 393216 --port 8055
  --ssd-streaming --power 100 --warm-weights --vision ... --dspark --mtp-model ...`
  （当前约 85 GB RSS；测试前先停）。

---

## 8. 交接提示词（可直接粘贴到下一个 session）

> 你是 ds4 仓库（`/Users/xieyongliang/ds4`）的 coding agent。上一个 session 完成了对
> `Chida82/sf-ds4-1flash` 的调研，记录在 `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md`，
> 交接在 `HANDOFF-SF-DS4-1FLASH-20261021.md`。当前 HEAD `89e068c`，工作区有未提交的
> DSpark/ledger 改动，**不要覆盖**；本次只做调研，未改代码。
>
> 目标：验证并选择性移植 fork 的可借鉴项，不是追 README 的 +78%。
> 关键事实：fork 的 +45~99% prefill / +62~78% decode 是相对 ds4 `0aaea5a`；我们树已有
> #1041/#1042/#1090/#1120/#1192，按 `auto-1/2.csv` 粗比只剩 **decode +18~30%、prefill +20~84%**；
> 首 token 2K/4K 我们更快、8K/32K fork 更快；fork 的 82 GB cache 与 explicit buffers 是主因之一。
>
> 先做 P0：借 fork 的 `ab_bench.py`/`readme_bench.py` 与 perf-record 纪律，冻结我们 HEAD 基线
> （同 GGUF/flags/prompt、auto 64.5 GiB、512/128 greedy），跑 A/A。然后按 P1→P4：
> P1 `270` cache lock、`268` aligned uncached reads（后者依赖 explicit buffer driver）；
> P2 `131/132/210` readback；P3 `60/100` explicit expert buffers（wide sweep 优先）；
> P4 `60` S2 index batch、`60` S1b decoder suffix（先过 DSpark 128-row parity）、`#1125`、`#1073` 缺口。
>
> 不要借：`--boost`/fanboost、dual-drive、整树 prune、盲目 82 GB cache。
> 任何模型级测试前先停掉 8055 `ds4-server`（当前约 85 GB RSS），确认内存余量；不要并发跑两个 huge model。
> 闸门：token-identical → `--bitwise` → `--logprob-vectors` → DSpark streaming short-prefill/verify →
> long-context → SSD streaming + resident → Metal default；负结果按 `LOCAL_INVENTORY.md` §D 登记。
> 先读 `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md` 与 `HANDOFF-SF-DS4-1FLASH-20261021.md`，再动手。
