# HANDOFF：sf-ds4-1flash P0 落地 + P1/P2 判负（2026-10-21 第二段）

> 承接 `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md` 与 `HANDOFF-SF-DS4-1FLASH-20261021.md`。
> 本段**改了代码**：只保留 P0（A/B harness + bench 字段 + memory-report API）；P1/P2 实测判负后已回退，
> 补丁存档 `speed-bench/sf-port-p1p2-20261021.patch`。工作区原有的 DSpark/ledger 改动未被覆盖。

---

## 0. 一页结论

| 项 | 结果 |
|---|---|
| P0 harness | **采纳**：`ab_bench.py` / `readme_bench.py` / `tests/test_ab_bench.py`（42 例）+ `--frontiers`/`--cache-stats`/token ids + `ds4_engine_memory_report()` |
| 冻结基线 A/A | prefill **133.7 / 224.6 / 516.3 t/s**，steady decode **22.5 / 22.9 / 21.4 t/s**，first token **67.4 / 201.2 / 236.9 ms**（2K/8K/32K） |
| A/A 噪声底 | prefill/decode 中位 ≤0.8%、CI ≤±2.3%；first token 2K/4K ≤0.4%；**first token ≥8K 本协议不可测**（同二进制 A/A 自身 −14.6%/−14.7%，CI 排除 0） |
| P1 `270` cache lock | **判负**：decode +0.3/+0.5%（CI 跨 0，扣对照后 +0.6..+0.8% 仍噪声内）、open +1.7 s；2K first token −6.5% 须扣 decode kind 自身 −4.1% 顺序偏差，增量约 −2.4%（噪声内）。64.88 GiB 下 lazy mlock 不失败，没有可防的缩容 |
| P2 `131` readback poll | **落点错误（no-op）**：profile 证明 `metal_graph_decode_*` 不在 V4.1 streaming decode 路径上（0 行）；该腿实为 decode kind 的 A/A 对照：decode −0.3%/−0.0%、2K first token −4.1%（CI −6.8..−0.0）。真正的 readback 在 `ds41_moe_partial` 内 |
| P2 其余 / P3 / P4 | **未动**，port map 见 §5 |

一句话：**fork 的 harness 值得借（已借），`270`/`131` 这两个 decode 侧“速赢”在本树 64.88 GiB 档不成立，已用 10 对 / 6 对 A/B + bitwise 证伪并回退；下一个该测的是 `132`（router 单独 commit），而不是继续堆 poll。**

---

## 1. 环境与冻结配置

- 机器：Apple M5 Max，128 GB unified；模型 `gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf`（365.7 GB，SSD streaming）。
- 冻结配置（每条腿都相同）：

```
--ssd-streaming --ssd-streaming-cache-experts 72GB --ctx-alloc 65536
--prompt-file speed-bench/promessi_sposi.txt
greedy, no EOS；token ids 逐腿比对
```

  loader 报 **64.88 GiB dynamic / 6999 experts**（`72GB - 7.12 GiB prefill headroom`），
  与历史 auto（`auto-1/2.csv`：64.51 GiB / 6959 experts）同档。
- 为什么不用 auto：auto 跟空闲内存走（本机空闲时给 80.5 GiB），且上一腿留下的 page cache 会改变它，
  A/B 会拿两个 cache 尺寸比。72GB 钉住后 A/A 的 prefill/decode 中位 ≤0.8%。
- 为什么 ctx 65536：sweep 到 32768，bench 要求 `ctx_alloc > ctx_max + gen`。
- 模型级测试前已确认：8055 无 ds4-server、无其它 >1 GiB LLM 进程、free ≈86 GiB、swap 未增长
  （全程 2.88→2.89 GB，未触发压力）。**没有并发跑两个 huge model。**
- 本机没有 `mactop`/`fanboost`：harness 以 `monitor none` 运行，**没有 GPU 频率/温度门控**；
  A/A 的噪声底就是在该条件下测的。要发布级数字先 `brew install mactop`。

---

## 2. P0 交付物

| 文件 | 内容 |
|---|---|
| `speed-bench/ab_bench.py` | fork 的 A/B harness 适配本树：`ds4-bench`、`--frontiers`、pin 72GB、`--bitwise` 只比 prefill logits、mactop 可选、auto cache 尺寸跨腿一致性检查 |
| `speed-bench/readme_bench.py` | README 表：每 build 一块 + 丢弃 warm-up、mactop 冷却目标（无 mactop 用固定 settle）、旋转列、median/spread；去掉 fanboost / 第二块盘列 |
| `tests/test_ab_bench.py` | 42 例单测（解析、调度、统计、correctness、logits validator），`make test-ab-bench`，并已加进 `make test` |
| `speed-bench/PERF_RECORD.md` | 协议、冻结基线、A/A 噪声底、rejected ideas、P1/P2 判负、逐项 step log |
| `ds4_bench.c` | `--frontiers`（严格递增列表）、`--cache-stats`（每 frontier 打 memory/cache 报告）、`gen[ctx=N] token ids:` 行 |
| `ds4.c` / `ds4.h` | `ds4_engine_memory_report(e,label)`（bench 不碰张量内部） |
| `ds4_help.c` / `Makefile` | 新 flag 帮助；`test-ab-bench` 目标 + `make test` 一行 |

验证：`make ds4-bench -j8` 无警告；`python3 tests/test_ab_bench.py` 42/42 绿；
`./ds4-bench --help` 显示新 flag；2K smoke 的 token ids 与回退后二进制逐 token 相同。

---

## 3. A/A 结果（噪声底，`speed-bench/v41_m5max_ab/aa-sweep-20261021`）

两侧同一二进制（`/tmp/ds4-base` 为 `./ds4-bench` 的逐字节拷贝），3 个完整 A B B A quad → 4 对/指标。

| 指标 | A 中位 | B 中位 | A/A 增益 | 95% CI |
|---|---:|---:|---:|---|
| prefill 2048 / 8192 / 32768 t/s | 133.7 / 224.6 / 516.3 | 133.3 / 225.2 / 516.5 | −0.3% / +0.3% / −0.4% | ≤±2.3% |
| steady decode 2048 / 8192 / 32768 t/s | 22.5 / 22.9 / 21.4 | 22.4 / 22.9 / 21.5 | +0.0% / +0.5% / −0.2% | ≤±2.1% |
| first token 2048 / 4096 ms | 67.4 / 67.5 | 66.0 / 68.2 | −0.4% / −0.4% | ±4.5% |
| first token 8192 / 16384 / 32768 ms | 201.2 / 184.0 / 236.9 | 186.9 / 227.7 / 278.4 | +7.8% / **−14.6%** / **−14.7%** | 宽，且后两者排除 0 |

**关键教训**：first token ≥8K 在同二进制 A/A 上就能“测出” −15%，说明它由上一腿留下的 page cache 主导，
A B B A 不抵消（每腿冷启）。**以后任何 ≥8K first-token 的宣称，必须先有每腿冷 cache 控制，否则不算证据。**
prefill 与 steady decode 是可信指标；2K/4K first token 可用（CI ±4.5%）。

---

### 闸门口径说明（本轮实际用了哪些）

- 采纳的只有 P0（工具层：bench 字段 + memory-report API + harness），**不改推理**，所以它的闸门是
  `make` 无警告 + 42 例单测 + 2K token ids 与回退后二进制逐 token 相同。
- P1/P2 在被回退前跑了 **token-identical + `--bitwise`**（10 对 / 6 对）。`--logprob-vectors` 在本机
  **无法有效执行**：可用 fixture 只有 `flash-0731` / `flash-pre-0731`，而生产 GGUF 是 1017，按
  `tests/test-vectors/README.md` 这属于“checkpoint 与 fixture 不匹配 = 无效测试”。对纯调度改动，
  同 checkpoint 的 token-identical + prefill logits 逐位是更强的闸门。
- DSpark streaming/verify、long-context、resident 三态没有跑：P1/P2 已回退，没有可验的改动；
  下一轮若有推理改动，这三项按 §5 的顺序补。

## 4. P1/P2 判负证据

### P1 `270`（fork `5b6482a`）— 移植后判负

- 实现：`ds4_gpu_stream_expert_cache_lock_ahead()`（62 行，engine open 时分配全部 slab + mlock 全部 slot
  并压入 free list；`DS4_METAL_DISABLE_STREAMING_CACHE_LOCK_AHEAD` 回退；`!load_slice && !tp_shard`）。
- 机制生效：`locked ahead: 6999 of 6999 slots, 17 slabs, 64.93 GiB in 1656 ms`，open +1.7 s。
- 闸门：token ids 2K 相同；`--bitwise` 在 10 对 decode 腿上 PASS（prefill logits 逐位）。
- 实测（P1+P2，10 对）：decode 2048 **+0.3%**（CI −0.5..+2.9）、decode 8192 **+0.5%**（CI −0.5..+1.9）、
  first token 2048 **−6.5%**（CI −9.9..−4.1）、first token 8192 +5.1%（CI −1.5..+15.0）。
  **first token 要扣对照**：P2-only 腿实为 decode kind 的 A/A（见下），同指标 −4.1%，故 P1 增量约 −2.4%；
  decode 相对同一对照 +0.6..+0.8%，均在噪声内。
- 机理：64.88 GiB 下 lazy mlock **没有失败**（无 `using locked cache cap` 行），没有缩容可防；
  open 期锁满 64.93 GiB 只是在首 prefill 前占 wire。fork 的收益来自 82 GB 档
  `vm.global_user_wire_limit` 咬住。**裁决依据 = decode 中性 + open +1.7 s**，不是已证实的 first-token 回归。

### P2 `131`（fork `f8dd83e`）— 移植后判负

- 实现：`ds4_gpu_wait_command_buffer` 内 2 ms status poll（`g_wait_poll_ns`），scope 到两处 decode readback
  （`metal_graph_decode_cpu_router`、`metal_graph_decode_selected_readahead`）；`DS4_METAL_DISABLE_V41_READBACK_POLL` 回退。
- **落点错误**：V4.1 streaming decode 走 `ds41_graph_decode_layer → ds41_graph_layer → ds41_moe_partial`；
  带 `DS4_METAL_STREAMING_SELECTED_READAHEAD_PROFILE=1` 与 `DS4_METAL_STREAMING_IQ2_CPU_ROUTER_PROFILE=1`
  的 2K 腿打印 **0 行** profile，说明我改的两处 readback 不在该路径上，poll 从未生效。
- 该腿因此是 decode kind 的 **A/A 对照**（6 对）：decode 2048 **−0.3%**（CI −4.4..−0.1）、
  decode 8192 **−0.0%**（CI −3.1..+0.8）、first token 2048 **−4.1%**（CI −6.8..−0.0）、
  first token 8192 +1.7%（CI −11.8..+6.9）。这组数就是该 kind 的顺序偏差。
- 真正的 DS41 readback 在 `ds41_moe_partial`/`ds41_graph_layer` 内，结构不同于 fork 的
  `routed_moe_one_tensor`；131/210 要重做必须落在那里，且先拿到唤醒暴露的 profile。

### 处置

- 两处代码已**逐字节回退**（P2 是落点错误，P1 是“中性 + 1.7 s open”；两者的回退决定不变）（reverse-apply 补丁后 5 个文件与快照完全一致），工作树只剩 P0。
- 补丁：`speed-bench/sf-port-p1p2-20261021.patch`（195 行 / 5 文件），`patch -p1 < ...` 可重放。
- 裁决已登记 `LOCAL_INVENTORY.md` §D 与 `speed-bench/PERF_RECORD.md`（Rejected ideas）。

---

## 5. 未动项与精确 port map（下一段直接用）

行号会随未提交的 DSpark 改动漂移，**以函数名为锚**。

| 项 | fork 规模 | 本树落点 / 阻塞 |
|---|---|---|
| `132` router 单独 commit（`67b75b8`） | metal +33、ds4.c +6、gpu.h +1 | **关闭（结构不适用）**：本树 `ds41_moe_partial` 不把 ids 读回主机（`g->selected` 直接给 GPU 侧 `begin_selected_load_tensor`），`ds41_graph_step` 逐层 flush 只 commit 不等待；没有“CPU 读 ids 时让 shared expert 先跑”的重叠对象。见 PERF_RECORD「Why 131/132/210 have no site」 |
| `210` selected-id mailbox（`37a5d1e`） | metal +110、dsv41.metal +16、ds4.c +2 | **关闭（同因）**：没有 CPU 等 ids 的等待可消除。除非将来重新引入 host readback，否则不要开 |
| `268` 页对齐 uncached reads（`220dd7d`） | `ds41_prefill_expert_pread` +8 行 | **阻塞**：本树没有 explicit prefill driver / `ds41_prefill_expert_pread`；只有先落 P3，这个 8 行才有落点。继续走 mmap page-in 时**不适用** |
| P3 `60/100` explicit expert buffers（`0d88af9`/`896785c`） | 300–400 行 | **不取：机制已在本树**。宽 sweep 已有提前一层 `pread` 环（`prepare_ahead=1` → `metal_graph_stream_prefill_layer_pagein_start(pread_only=true)`），即 fork 两槽 buffer 提供的 overlap；统一内存下 GPU 读同一批页，显式 buffer 不减少字节。depth 2 探针中性（ttft 7500 +1.0%，6 对）佐证。剩余只有 `F_NOCACHE` 页对齐（268），需先有 explicit driver 且只减 page cache 污染。fork 的 +19~24% 是对其“wide sweep 仍同步 mmap”的树测的。HANDOFF §9 的配方仅作存档 |
| P4 `60` S2 index batch | ~100 行 + 2 kernel | **不取**：本树已有 candidate_blocks/filter 与 select-all，但按行；fork 的批量形态要新增 `candidate_topk_batch`/`candidate_mask_batch` 两个 kernel，收益 +2.3/+1.69/+4.3% ttft 在 ±5% 带内 |
| P4 `60` S1b decoder suffix | 小 | **阈值单独无效**（2542 vs 4096：ttft 7500 +0.8%，6 对）；fork 的收益在 short_sweep tile 对齐 + 128 warm，需连 DSpark parity 一起做，维持默认 4096 |
| `#1125` 四项修复 | 上游 PR | **不取**：四项均为健壮性修复（错误路径还槽、GPU-copy seed in-flight、逐层计数、prune 跳过 in-flight），fork 自述无 headline 指标为负；本树未复现 slot leak / prune hang，触发条件不成立 |
| `#1073` 缺口 | 小片 | **不取**：按名核对缺 `rope_quantize`/`qkv_norm_kv_tail`/`router_one` 等具名融合（#1042/#1090 已覆盖大部分），单项 0.4~0.8% decode，低于 ±2% 带，需 4+ 对——ROI 不成立 |

---

## 6. 复现命令

```sh
# 基线树（A 侧）
mkdir -p /tmp/ds4-base && cp ds4-bench /tmp/ds4-base/ && ln -sfn "$PWD/metal" /tmp/ds4-base/metal
printf 'ds4-bench:\n\t@true\n' > /tmp/ds4-base/Makefile

# A/A 噪声底（约 12 min/quad；budget 按需）
python3 speed-bench/ab_bench.py --a /tmp/ds4-base --b . --kinds sweep --budget 900 \
    --out speed-bench/v41_m5max_ab/aa-$(date +%Y%m%d-%H%M)

# decode 侧改动（P1/P2 类）
python3 speed-bench/ab_bench.py --a /tmp/ds4-base --b . --kinds decode --bitwise --budget 1400

# 单测
make test-ab-bench
```

---

## 7. 收口：本轮移植战役的结论

每一项都有结论，只有 P0 值得留：

| 项 | 结论 | 重开条件 |
|---|---|---|
| P0 harness | **采纳** | — |
| P1 `270` | 不取（decode 中性 + open 1.7 s） | 某配置打印 `using locked cache cap` |
| `268` | 不适用（依赖 explicit driver） | P3 重开且 driver 落地 |
| P2 `131/132/210` | 无落点（本树无 host readback） | 重新引入逐层 ids 回读 |
| P3 `60/100` | 不取（机制已在本树：提前一层 pread 环） | 出现“宽 sweep 同步 page-in”的回归，或实测读时间 > 盘底 2× 且排除计算 |
| P4 S1b | 阈值中性；short_sweep 未验 | 先过 DSpark 128-row parity |
| P4 S2 | 不取（需 2 个新 kernel，收益在带内） | index 工作成为 7500 行 sweep 的主项 |
| `#1125` | 不取（健壮性，无复现缺陷） | 复现 slot leak / prune hang |
| `#1073` | 不取（单项 <1%，低于带） | 需要 ≥4 对的判定预算时 |

**没有遗留的实现工作**。如果还要继续提速，方向应该换：本轮的证据指向“宽 sweep 的
~7.3 GiB/s 与盘底 ~15.5 GB/s 之间有时间没被解释”，而那个问题不是 fork 的任何一项能回答的；
下一个仪器应该把 wide sweep 的 wall time 拆成“等 pread / 等 GPU / 主机编码”三段，再决定动哪里。

---

## 8. 参考

- `speed-bench/PERF_RECORD.md`（协议、基线、A/A 噪声底、P1/P2 判负、rejected ideas）
- `speed-bench/v41_m5max_ab/aa-sweep-20261021/`（A/A 原始 CSV/err）
- `speed-bench/v41_m5max_ab/p1p2-decode-20261021/summary.txt`（P1+P2，10 对，bitwise）
- `speed-bench/v41_m5max_ab/p2-only-decode-20261021/summary.txt`（P2 only，6 对）
- `speed-bench/sf-port-p1p2-20261021.patch`（回退的 P1/P2）
- `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md`、`HANDOFF-SF-DS4-1FLASH-20261021.md`（上一段调研）
- `LOCAL_INVENTORY.md` §D「sf-ds4-1flash 可借鉴项移植」

---

## 9. 附录：P3 explicit expert buffers 移植配方（下一轮直接开工）

来源：fork `0d88af9`（60，单 chunk explicit buffers + S1a/S1b/S2/S3 混合）与 `896785c`（100，wide
sweep 两缓冲 + RAM-before-disk）；`220dd7d` 的 `268` 页对齐只改 `ds41_prefill_expert_read` 里那 8 行。
**本树这些符号全无**（`ds4_gpu_stream_prefill_bind_layer` / `discard_buffer` /
`streaming_prefill_bytes` / `ds41_prefill_expert*`）。

### 预算（已核对，不需要动 dynamic cache）

- loader 报 `7.12 GiB prefill headroom` = 2 × 3.56 GiB；一层 routed experts =
  384 × 9.49 MiB = **3.56 GiB**。即 fork 的“两套完整 layer buffer”正好吃满 reserve。

### 引擎侧（ds4.c，新增 ~250 行）

1. `ds41_gpu_graph` 增 `uint64_t streaming_prefill_bytes;`，由 loader 的 prefill headroom 赋值
   （`ds4_engine_open_internal` 里已有 `prefill_headroom_bytes`，传给 graph）。
2. `struct ds41_prefill_expert_slot`：每 slot 3 个 `ds4_gpu_tensor *`（gate/up/down）、
   `bytes[3]`、`locked[3]`、`ds4_gpu_stream_expert_table table`、`read_fd`（`F_NOCACHE`）、
   `pthread_t threads[16]`、`ds41_prefill_expert_worker workers[16]`、`started`、`ok[16]`、`dst[3]`。
3. `ds41_prefill_expert_open_nocache_fd(source_fd)`：`fcntl(fd, F_NOCACHE, 1)` 的副本 fd。
4. `ds41_prefill_expert_read`：每个 tensor 的字节区间按 16 线程切分，循环 `pread` 到
   `ds4_gpu_tensor_contents(dst[j])`；**268 的改动就在这一行**——起点按 16 KiB 页对齐，首段 partial
   page 单独读，避免 APFS `F_NOCACHE` 漏进 page cache。
5. `read_join` / `buffers_free`（munlock + `discard_buffer` + free）/ `buffers_init`
   （2 slot × 3 tensor，mlock，总量 ≤ `streaming_prefill_bytes`）。
6. `ds41_graph_prefill_sweep` 的 wide 分支：每层 `bind_layer(slot)` → （100 的
   `copy_cached` 把 decode cache 里 ready 的专家 blit 进 slot）→ 启动 workers → `join` → 编码该层 →
   下一个 slot 预读；deferred-decoder sweep 仍走 mmap page-in。

### Metal 侧（ds4_metal.m，新增 ~120 行）

- `ds4_gpu_stream_prefill_bind_layer(table, gate, up, down)`：把显式 buffer 注册成该层的专家张量视图，
  让 routed 派发从 buffer 读而不是 model map（与现有 `ds4_gpu_stream_expert_cache_*` 的地址表路径同构）。
- `ds4_gpu_stream_prefill_discard_buffer(tensor)`：释放注册视图。
- `ds4_gpu_stream_prefill_copy_cached(table, gate, up, down, ready, bytes)` +
  `ds4_gpu_stream_prefill_copy_wait()`：从 decode cache blit 命中专家，**不更新 hit/recency/hotness**。
- `ds4_gpu_cleanup` 里补 `copy_wait` + `bind_layer(NULL,...)` 的收尾（fork 在 896785c 里就是这两行）。

### 顺序与闸门

- 先只做 **wide sweep**（`total_count > encoder_chunk`，≥8192 行）与 **单 chunk**（`!wide` 但
  `total_count` 一个 chunk 能装下）；small append 继续走 `ds41_streaming_sweep_moe_gather_admitted` 的
  gather 路径，避免重复。
- 每步：token-identical → `--bitwise`（prefill logits）→ DSpark streaming short-prefill/verify →
  长上下文 → streaming 两态。P3 只动 prefill 读路径，decode 指标应保持中性。
- 验收指标：`ttft 3500/5000/7500/10000` 与 `append +300/+1500`（本附录 §8 的噪声带）。
- 负结果（例如本机 SSD 已跑满、显式 buffer 无收益）按 §D 登记，不要静默回退。
