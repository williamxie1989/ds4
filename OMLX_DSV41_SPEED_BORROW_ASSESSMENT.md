# omlx DSV4.1 提速经验 → ds4 可借鉴性评估

> 评估日 2026-10-18。只做方案/代码层评估：**不加载模型、不做新 A/B**。
> omlx 侧证据：`~/omlx-notes/dsv41_offload_state_and_levers_2026-10-17.md`、
> `~/omlx-notes/dsv41_decode_regression_2026-10-18.md`、`~/omlx` 2026-10-14..18 的
> `perf(deepseek_v41)` 提交；ds4 侧证据：本仓库路线图/交接文档、`speed-bench/M3/`、
> `LOCAL_INVENTORY.md`。当前 ds4 的旧 checkpoint 已删除、新 checkpoint 未下载，
> 因此本文所有 ds4 数字都来自既有文档/日志，**不代表当前二进制复测**。

---

## 0. 一页结论

**这轮 omlx 优化里，没有值得直接移植到 ds4 的新速度杠杆。**

最像大杠杆的 **prefill 热集 pin**，ds4 已经在 2026-10-12 用同思路做过 A/B（M3-3）：
静态 hotlist + `protected` 位，结果 **慢 13.6%–28.4%、命中率反而更低、驱逐 4–8×**，
代码按预注册回退条款整体撤回。根因不是实现，而是：

| | omlx pin 的基线 | ds4 M3-3 的基线 |
|---|---|---|
| prefill 命中率 | **29.7%**（K=0） | **0.967–0.996** |
| miss 形态 | 99.9% 是“读过又被驱逐” | 自适应 `ds41_prefill_seed` 已在会话内学会热集 |
| 缓存组织 | 每层固定容量（146 槽） | 全局共享池（7822 专家） |
| LRU 是否病态 | 是（每层 working set 170–314 > 146） | 否（全局 LFU + 频率播种） |

omlx 的 pin 是在 **29.7% 命中率的 LRU 病态**上救火；ds4 的 warm agent replay 已经
是 **0.996 命中率**，pin 的理论上限只有 0.4pp，而保护 66% 容量会把动态池压到 2666 槽，
直接制造 churn。两者问题不同构。

**可借鉴的只有 3 条方法论 + 1 条待证伪的冷域假设：**

1. **prefill 几何自报**：每次请求记录 `path/rows/chunks/cache_experts/ced/encoder_resident`。
   omlx 实测同一 8.4K prompt 会因 chunk 数 2 vs 5 产生 **2× TTFT** 差异；ds4 的
   `ds41_prefill_count` 同样有多条阈值路径，A/B 不钉几何就会把路径漂移记到改动头上。
2. **离线回放目标改成 `misses + λ×load_calls`**，而不是只看字节/命中率。M3-3 的
   static pin 正是“字节可能降、load/驱逐爆炸”的形态；ds4 现有
   `speed-bench/mtp_ledger_replay.py` 只记 GPU-cache miss 与 SSD 字节，缺 load-call 惩罚。
3. **首轮丢弃**：omlx 的 Engram 行缓存冷启动让首轮慢 15–18%；ds4 的 Engram/页缓存
   同样有冷启动，A/B 必须丢 run0。
4. **冷域假设**：多会话轮换 + 小 cache 下，跨会话标定的 hotlist 是否值得。omlx 的
   跨语料 first-exposure 仍有 −10%–−23%，这是唯一还开着的门；但 ds4 M3-3 的首个
   append 已经 A≈B（−6.6%..+3.5%），所以预期只有 **0–10%**，且必须先纸面回放
   （≥5pp 才动引擎）。

**明确不要重开**：MTP/DSpark 流式（`a≈0.68`，oracle 上界 1.387×）、decode overlap /
更深流水（ds4 是 host-bound，已有负结果）、prefill read-ahead（P0g 负）、staging
（ds4 已是零拷贝）、容量扩容（8000≈auto，128 GB 加不了 23 GB）、dense prefix
（indexer 已 clamp）、affine8 / block packing / 读速优化。

---

## 1. 口径差异：为什么 omlx 的百分比不能直接引用

| 维度 | omlx | ds4 |
|---|---|---|
| 引擎 | Python + MLX，vendor fork | C + Metal/CUDA，单模型引擎 |
| 专家缓存 | 每层固定 slot 池（146/384） | 全局共享池（生产 7822/15360） |
| 驻留 | `resident_fraction` + Engram SSD offload | `--ssd-streaming` + expert cache + Engram 盘读 |
| prefill 主路径 | MLX block loop，wide step 8192，CED | layer-major sweep / gather [32,2048] / wide ≥4096 |
| decode 墙钟 | SSD 34 ms + GPU 27 ms 串行（63 ms/token） | 44 ms/token，GPU 只忙 13.3 ms，host 指令量主导 |
| 量化/权重 | oQ2.5e（MLX 布局） | V4.1 Q2 GGUF（IQ2_XXS + Q2_K） |

同模型不同引擎、不同量化、不同驻留态，**omlx 的 CED +16.6%、pin −26% 都是
MLX 驻留/offload 域的相对值**。ds4 文档里已经写过这条纪律
（`V41_STREAMING_ENGRAM_SPEED_PLAN.md` §0 第 4 条），本文沿用：
任何 omlx 数字只作为“方向证据”，不作为 ds4 的预期值。

---

## 2. omlx 这轮做了什么（2026-10-14..18）

### 2.1 prefill

| 改动 | 实测 | 机制 |
|---|---|---|
| **pin 7/8 专家槽**（`96601d89`） | 8.4K prompt TTFT 21.85→16.19 s（**−26%**）；fetched 155.6→112.1 GB（−28%）；hit 29.7→49.5%；跨语料 −19%–−28%；decode hit 87.3→84.9%、fetched +25%，但 `expert_call` 8.44→8.37 s 基本平 | 每层池 146 槽，保护 128 个静态热专家，留 18 槽流式窗口；`refresh_pins` 每个 layer-chunk 按全史引用频率重排。K=138 已反转，K=145（1 槽窗口）崩到 16,761 blocks / 31.66 s（+45%）。默认改为容量比例 0.875，池 <128 槽时关闭 |
| **wide step 8192**（`38659bc5`/`eb4a0fca`） | pp8192 +10.5%，pp4096 +14.9%；16384 被内存 guard 拒绝 | scheduler 给 deepseek_v41 接 8192 chunk，要求 native kernel + ≥64 GB |
| **CED**（`58d92fae`/`669fb469`） | pp8192 +16.6% over wide-step，+28.8% vs stock；greedy diff 通过（math 相同，summary 1-token tie-flip） | decoder 半只送 window=128 尾行，midpoint CSA2 层产 global KV |
| **dense prefix 旁路** | 推导后仅 ~0.5%，放弃 | v41 indexer 已 clamp，前缀行没有白算 |
| **prefill 跨块 read-ahead**（`dd33a431`） | 默认开，next chunk 的读在本块 eval 期间发出 | 读只提交、不改驻留；未消费的 future 在 `finally` 取消 |
| **staging 零拷贝池**（`53ac7916` 等） | 把 bytes→mx.array 的 convert-copy 从 28% 压下去；后来相位账显示稳态没有 host 肥肉 | 读直接落进 MLX 分配器；ds4 没有这次拷贝 |
| **block-loop ledger**（`268a22f0`） | K=128 下 prefill 16.43 s：`eval` 6.87 + `route_sync` 5.79 + `fetch` 2.89；循环体 self 仅 0.13 s | pin 把瓶颈从盘换到 GPU；`route_sync` 是设备同步，不是 Python |
| **route_sync numpy 化**（`96601d89` 附带） | 合成微基准 302→39 ms，但端到端只 0.5% | 相位里 Python 只占 ~6%，大头是 attention+router 的 GPU 时间 |

### 2.2 decode

| 改动 | 实测 | 机制 |
|---|---|---|
| **async_eval 流水深度 3→6**（`f0097064`） | `expert_call` 9.09→8.14 s（**−10.5%**）；depth 12 回到 8.29 s | MLX `async_eval` 在途层数；decode-only，prefill 仍 per-chunk eval |
| **forward_overlap** | 9.09 vs 9.07 s，**判死** | 32% 调用真的触发，但它要填的 GPU 空窗已被 depth 3 流水线填掉；两者是替代品，不是叠加品 |
| **Engram 精确行读 + 行缓存**（`02f032ee`） | merged exact-row 116–131k IOPS vs 16 KiB page 113k；一次 8K chunk 的 page 预取 15.4 s vs 7.9 s；行缓存让同语料 engram 6.67→0.31 s | 行 ≤512 B 直接精确读；1 GiB FIFO 行缓存；预取门槛 128→1 行 |
| **prefill 后热集播种**（`fdc8383e`） | K=4 −1.9% s/tok，K=8 持平；默认 0 | 按 prefill 引用次数重装 top-K，走现有 IO 池 |
| **decode route episode**（`c67da396`/`d991983b`） | 纯记录，无行为变化 | 每次 prefill 后快照 residency，decode 每次调用记 route/miss，供离线回放 |
| **容量 146→203** | 外推 decode −25% | 需 +23.3 GB wired；用户决策，非代码 |

### 2.3 omlx 自己排掉的死路

- `FORWARD_OVERLAP`：与流水深度重复。
- 更深流水 depth 12：比 6 差。
- “让读更早发出”：读走 IO 线程池，不被 GPU drain 阻塞；跨层路由不可知。
- 块打包：blocks 967→504，墙钟 −0.2%（噪声），因为成本是 O(references) 不是 O(blocks)。
- Engram worker 48→16：只值 ~4%（engram 的 4% ≈ prefill 的 1%）。
- router bf16 / gate 融合：1.3–3.8%，且破位精确。
- 读速优化：prefill 读已 10.5–12.5 GiB/s，对 12.36 GiB/s 上限 85–101%，**只有“少读”没有“读快”**。

---

## 3. ds4 对应机制与已有证据

| omlx 机制 | ds4 对应物 | 状态 |
|---|---|---|
| 每层 pin 7/8 | 全局 cache + `protected` 位（M3-2） | **已实现、已 A/B、已撤回**；patch 存档 `speed-bench/M3/m3-3/r2a-reverted-20261012.patch` |
| `refresh_pins` 按全史频率 | `ds41_prefill_seed`（`ds4.c:42849`）：每层 sweep 后按本层引用频率 + 最近 32 token 加权，seed `capacity/40` 个专家 | **已在树上**；M3-3 的 B 臂就是它，命中率 0.967–0.996 |
| 静态 hotlist 文件 | `ds4_streaming_hotlist.inc` + `DS4_METAL_STREAMING_EXPERT_HOTLIST` | 已在树上；M3-3 A 臂用它 + protect，负 |
| 容量比例/下限 | M3-1 标定 60/75/85% 三档 | 工具链在 `tools/hotlist_v2_from_turns.py`；M3-3 用 5156/7822（66%） |
| wide 8192 | `ds41_prefill_count` wide ≥4096；`carry_cap` | 已在树上 |
| CED | `decoder_suffix`，`DS4_V41_DECODER_SUFFIX_MIN_ROWS` 默认 4096（`ds4.c:43210`） | 已在树上；流式域 +9.3% steady |
| dense prefix | Metal indexer 已 clamp（P3-B 裁掉） | 死路，双方一致 |
| prefill read-ahead | P0g prepare ring 1→N | **实测负**，默认 1，opt-in 休眠 |
| staging 零拷贝 | expert cache 直接 pread 进 slot buffer | ds4 没有 bytes→array 拷贝，不适用 |
| Engram 精确行读 | `DS4_ENGRAM_ROW_BYTES=264`，`ds4_engram_read` 逐行 pread，`F_NOCACHE`，decode 48 行并发 | **ds4 已经是精确行读**；没有 page 浪费 |
| Engram 行缓存 | 无 | 可试，但预期 <1–2%（见 §5.4） |
| decode 深流水 | `#1041` decode queue + P1-A pipeline ports | 已在树上；M5 流式 +2.5–4% |
| decode overlap | 卡 H / mailbox / P1-C | 已实测负/中性 |
| MTP/DSpark 流式 | 硬门 + P2 管线 | **已关账**：`a≈0.68`，oracle 1.387×，实测 1.000×（见 §6.1） |
| 图回放 | 卡 I-2/I-3 | macOS 26.5.1 无 MTLGraph capture，M1 搁置 |

### 3.1 M3-3 的精确数字（为什么 pin 在 ds4 上不成立）

M3-3 窗 `speed-bench/M3/m3-3/run-20261012-010604/`，V4.1 Q2 streaming，cache 7822：

| 臂 | 配置 | hit_rate（120/300/480/1650 行） | 驱逐 | 相对 B |
|---|---|---|---|---|
| A | hotlist v2 5156 条全 40 层 seed + `protect=1` | 0.951 / 0.969 / 0.980 / 0.992 | 6511 / 8644 / 9838 / 12432 | **慢 13.6%–28.4%** |
| B | 生产基线（`ds41_prefill_seed` 自适应频率播种） | **0.967 / 0.981 / 0.988 / 0.996** | 777 / 1467 / 2100 / 3451 | — |

- A 把 5156/7822（66%）钉死，动态池只剩 2666 槽；`ds41_prefill_seed` 的自适应播种
  和小池互相踩，驱逐 4–8×。
- 首个 append（最接近冷启动）A≈B（−6.6%..+3.5%），**冷启动假设也没有可见收益**。
- bit-exact 全绿（bind-parity 27/27、223/1524 dump IDENTICAL、小 cache 压力腿 IDENTICAL），
  所以这不是正确性问题，是纯速度/容量问题。
- 结论：**暖回放域 unachievable-by-design；暖负载不开 protect=1；冷域假设保留但未证。**

---

## 4. 为什么 omlx 的 pin 不能直接搬

1. **基线不同构。** omlx 的 29.7% 是 LRU 在“每层 working set 170–314 > 146 槽”下的
   病态命中率；ds4 的 0.967–0.996 是全局 LFU + 每层频率播种的结果。pin 在 omlx 能
   把命中率抬 20pp，在 ds4 的理论上限只有 0.4pp。
2. **缓存组织不同。** omlx 每层固定 146 槽，pin 128 留 18 槽窗口；ds4 是全局 7822 槽，
   没有“每层窗口”概念。M3-3 的 66% 静态保护已经把动态池压到 2666；若照 omlx 的
   87.5% 保护，只剩 ~978 槽（每层平均 24），比 M3-3 更糟。
3. **动态性不同。** omlx 的 pin 每个 layer-chunk 按全史频率重排，是“动态保护”；
   M3-3 的 A 臂是**静态文件 + 固定保护**。ds4 的 `ds41_prefill_seed` 已经做了
   “每层按本层频率重排并播种”，只是没有保护位；保护位在 M3-3 被证明会减少动态池，
   而不是增加命中。
4. **冷域证据弱。** omlx 跨语料 first-exposure 有 −10%–−23%，但 ds4 M3-3 的首个
   append 是 −6.6%..+3.5%，没有可见收益。冷域唯一还开着的门是“多会话轮换 + 小 cache +
   跨会话标定”，不是“把 M3-3 的 patch 再 apply 一次”。

---

## 5. 真正值得借鉴的（按价值排序）

### 5.1 prefill 几何自报（零速度收益，但防止假 A/B）

omlx 的教训：同一个 8.4K prompt，一次 2 chunk（29.35 s）、一次 5 chunk（60.86 s），
**TTFT 差 2.07×，全部来自 chunk 数**；而 chunk 数由 wide-step 门、`prefill_step_size`、
admission guard 三处联动决定，A/B 日志里不记录。ds4 的 `ds41_prefill_count`
同样有多条路径：

- `remaining < minimum` → token-major（minimum 32/256/1024，取决于 cache 是否 ≥7680）；
- `remaining < 2048` 且 gather 准入 → gather sweep；
- `remaining > 2048 && < 8192` 且 `carry_cap` → medium sweep；
- `remaining ≥ 4096` → wide sweep；
- `encoder_resident`（≥16384）→ encoder 常驻；
- `decoder_suffix`（≥4096）→ CED。

建议加一行 env-gated 日志：

```
prefill geometry: path=gather|medium|wide|encoder|token rows=831 chunks=1
  cache=6959 gather=1 wide=0 ced=0 encoder=0
```

以及“为什么没走某条路”的 reason（cache < 384、rows < 32、type 不匹配、env 关闭）。
这不是速度改动，但能让下一次 A/B 的噪声从 2× 降到可读。

### 5.2 离线回放目标加 load-call 惩罚

omlx `§4.1` 的结论：pin 的收益函数必须是 `misses + λ×blocks`，不能只看命中率；
K=138 时字节还在降，但 blocks 从 967 涨到 2124，墙钟反转。

ds4 的 `speed-bench/mtp_ledger_replay.py` 目前只记：
- GPU-cache misses（DRAM 字节）；
- SSD pread 字节。

它**不记** `load_calls` / `pread_ranges` / 驱逐次数。M3-3 的 A 臂正是“命中率略低、
驱逐 4–8×”的形态；如果回放目标里有 load-call 惩罚，这个方案在纸面就会被否，
不必上机。建议给回放器加：

```
score = ssd_bytes + λ_load * load_calls + λ_evict * evictions
```

λ 用一次小规模实测标定（例如 1 次 load ≈ 多少 GiB 等价），然后所有 pin/hotlist/
容量策略先用这个目标筛。**这是当前 checkpoint 缺失时最值得做的代码工作之一。**

### 5.3 首轮丢弃 + 冷/热分离

omlx 实测首轮慢 15–18%（Engram 行缓存冷），且历史“engram 占 25%”全是冷启动数；
稳态 engram 只占 1.4%。ds4 的 Engram 是 `F_NOCACHE` 逐行读，没有行缓存，但页缓存/
专家缓存同样有冷启动。A/B 纪律：

- 每个臂至少 4 轮，**丢 run0**；
- 冷/热分开报，不混中位数；
- 同 prompt 连跑 3 次，chunk 数一致且墙钟极差 <5% 才开测。

### 5.4 Engram 行缓存（可选，低预期）

ds4 已经精确读 264 B 行，没有 omlx 的 page 浪费；每 token 24 行 × 2 层 × 264 B
≈ 12.7 KiB，15 tok/s 下约 190 KiB/s，带宽可忽略。行缓存能省的只是随机读延迟，
而 decode 的 Engram 读已经被 P1-A 的异步 step 读/流水线覆盖，prefill 有专用预取线程。
omlx 的行缓存收益来自“同语料重复读同一批行”，ds4 的 n-gram 行在 multi-turn 里会重复，
但命中率取决于 token 局部性。**预期 <1–2%，不建议在 checkpoint 到之前投入；
如果要做，先加一个只读计数器统计 (table,row) 复用率，再决定。**

### 5.5 “读已饱和，只有少读”作为路线确认

omlx 的 prefill 读 10.5–12.5 GiB/s，对 12.36 GiB/s 上限 85–101%；ds4 的 sweep
10.9 GiB/s、顺序读上限 12.3 GiB/s，同样 ~89%。两边的结论一致：

- 继续优化 pread 线程数/range 合并/布局，收益上限是个位数；
- 真正的杠杆是**减少 miss 字节**（命中率/容量/量化），不是“读得更快”。

这直接支持 ds4 路线图把 R2（命中率/保温）排在 R5（expert-contiguous 布局）之前；
R5 的预期也应从“+5–10%”下调到“微测定，0 或个位数”。

### 5.6 pin 之后瓶颈会换人（预测，不是当前收益）

omlx K=128 后：fetch 10.27→3.64 s，eval 4.74→5.50 s，prefill 从“磁盘 47%”变成
“GPU 71%”；小 M gather 成为新墙。ds4 如果未来在冷域真的做 pin，应该同步监控：
`encodes/step`、gather 时间、`pread_ranges`、驱逐次数；不要假设“字节降 = 墙钟降”。

---

## 6. 明确不借鉴 / 不要重开

### 6.1 MTP/DSpark 流式（最重要的一条）

omlx 的生产形态是 **MTP + expert offload + Engram offload 同开**：adaptive depth ≤3，
accept 79–87%，tok/cycle 2.76–3.09。看起来 ds4 应该照搬，但 ds4 的成本结构不同：

- ds4 实测 **一行 verify ≈ 0.68 decode 步**（`a≈0.68`，四配置一致：cache 958→8078
  不变、8 行同 token 亦不变）；8 行 verify = 5.77 步，**没有可摊薄的固定成本**。
- 盈亏平衡需要 avg_accept ≈ 2.96；实测 1.33–1.91；oracle 上界 1.387×；最优策略
  端到端 1.000×。
- omlx 的 6 行 verify 便宜，是因为 MLX `gather_qmm` 把 rows 批在一个 kernel 里
  （rows=24 时 462 GB/s，按 token 比 rows=6 还便宜 10%），且 decode 不是 host-bound；
  ds4 的 streaming verify 是逐行 host paging + 专家并集，没有这个摊薄。

**结论：omlx 的 MTP 成功不构成 ds4 重开的证据。** 只有等 ds4 的 host 成本结构性下降
（M2 图回放，当前 macOS 基底缺失），先重测 `a`；若 `a≤0.17`（+10% 门槛）才谈得上
MTP。否则维持 `LOCAL_INVENTORY.md` D 段的“投机整类关闭”。

### 6.2 decode overlap / 更深流水

ds4 decode 墙钟 = host 指令量（44 ms/token，GPU 13.3 ms，~750–900 dispatch，
26 CB 轮转）。omlx 的 `ASYNC_EVAL_DEPTH` 是 MLX 提交队列的深度，不是 Metal host
编码；ds4 的对应物是 `#1041` decode queue + P1-A pipeline ports，已经拿到 +2.5–4%。
卡 H、mailbox、P1-C 都已实测负/中性。omlx 的 `forward_overlap` 判死还提供了一个
好论据：**overlap 是流水的替代品，不是叠加品**。不要重开。

### 6.3 prefill read-ahead

ds4 的 P0g 已经测过 prepare ring 1→N：24 runs 无差/更差，假设证伪；默认 1，
opt-in 休眠。omlx 的 read-ahead 是**同一层内、router 已算出 ids 之后**的块间预读；
ds4 的 gather 路径一次加载整层 selected 集，没有 omlx 的 block loop，所以连对应物
都不成立。不要按 omlx 的结论去调 ds4 的 prepare 深度。

### 6.4 staging / 零拷贝

ds4 的 expert cache 直接 pread 进 slot buffer（slab/direct），没有 `bytes→mx.array`
的整块拷贝；omlx 的 convert-copy 28% 是 MLX 特有的。不适用。

### 6.5 容量扩容

omlx 146→203 外推 decode −25%，但需要 +23.3 GB wired；ds4 的 cache 8000≈auto
（BASELINE §12.1），128 GB 机器也加不了 23 GB。`--ssd-streaming-cache-experts 9000`
的单腿 +6–8% 是 decode 域，且 10-12 ABBA 复测被预检拦下；维持现状。不要拿 omlx
的容量结论来推 ds4。

### 6.6 dense prefix / affine8 / block packing / 读速

- dense prefix：omlx 推导后 ~0.5% 放弃；ds4 P3-B 已裁（Metal indexer clamp）。
- affine8：MLX `switch_layers` 门控；ds4 是 GGUF IQ2/Q2_K，不适用。
- block packing：omlx 中性（O(references) 不是 O(blocks)）；ds4 P0f2 gather >2048
  也证平手。
- 读速：见 §5.5，已饱和。

---

## 7. 唯一还开着的技术假设：冷域多会话 + 小 cache

**假设**：生产 server 在多个会话间切换（disk KV cache），专家缓存跨会话变冷；
如果 hotlist 是在**其他会话**上标定的，而不是当前会话，冷启动第一轮可能受益。
omlx 的跨语料 first-exposure −10%–−23% 是这条假设的外部证据。

**但 ds4 的现有证据是弱的**：

- M3-1 的 hotlist v2 标定用的是**单个会话**（60 轮），纸面增益 −1.1..+0.3pp，
  因为单会话内 expert 需求平稳、LRU 近最优；25pp 门隐含的是“多会话轮换 + 小缓存”。
- M3-3 的 A 臂用同会话标定的 hotlist + protect，首个 append A≈B；暖域整体负。
- `ds41_prefill_seed` 在会话内第一轮就学到热集，冷启动窗口很短。

**如果要做，顺序必须是：**

1. **纸面回放（零模型加载）**：用 `tools/hotlist_v2_from_turns.py` 在 208 条 ≥8 轮
   agent 会话上做 leave-one-session-out 标定；用 M3-1 的 `hotlist_v2_calibrate.py`
   仿真“跨会话 hotlist + 小 cache + 冷启动”，目标函数 `misses + λ×load_calls`。
2. **门槛**：相对 `ds41_prefill_seed` 基线，首个 append 的纸面命中增益 ≥5pp，
   或 `misses + λ×loads` ≥5%；否则不做。
3. **只有过门槛才** apply `speed-bench/M3/m3-3/r2a-reverted-20261012.patch`，
   并且**不要默认开 protect**；先只 seed、不 protect，做 A/B。
4. 上机 A/B：多会话轮换（冷启动起）+ 小 cache（1500/3000）+ ABBA ≥4 轮，
   bit-exact 门照 M3-3 全套。

**预期**：0–10%，中心估计 0–5%；如果纸面 <5pp，直接关账，不要上机。

---

## 8. 如果 checkpoint 到了：最小实验清单

> 前提：先跑 §7 的纸面回放；如果不过门槛，**不需要任何引擎改动**。

| # | 实验 | 目的 | 成本 |
|---|---|---|---|
| 1 | 基线 + 几何日志 | 钉住 path/chunks/cache，建立可复现基线 | 1 窗 |
| 2 | 跨会话 hotlist 纸面回放 | 决定冷域假设是否值得上机 | 零模型，1–2 天 |
| 3 | 若过门槛：seed-only（不 protect）A/B | 验证跨会话 hotlist 是否真的帮冷启动 | 1 窗 |
| 4 | 若 seed-only 有效：小 cache + protect 小比例（25–50%） | 找保护比例/窗口下限 | 1 窗 |
| 5 | Engram (table,row) 复用率计数器 | 决定行缓存是否值得 | 零模型/短窗 |
| 6 | 不做：MTP、overlap、read-ahead、staging、容量、dense prefix | 已证负/不适用 | — |

**A/B 纪律**（沿用 ds4 现有）：

- 同 build、同机、同几何；ABBA 交错 ≥2 对（差异 <15% 时 4 对）；
- 丢 run0；冷/热分开；
- bit-exact：`--moe-bind-parity` 27/27 + 223/1524 dump IDENTICAL + 小 cache 压力腿；
- 指标：TTFT、`miss_pread` 字节、`load_calls`、驱逐、decode steady t/s；
- 负结果按 `LOCAL_INVENTORY.md` §D 登记“别再试”。

---

## 9. 预期收益台账（诚实区间）

| 项 | 预期 | 依据 |
|---|---|---|
| prefill pin（暖域） | **0 或负** | M3-3：−13.6%–−28.4%；基线命中率 0.967–0.996 |
| prefill pin（冷域多会话） | **0–10%**，中心 0–5% | omlx first-exposure −10–23% vs ds4 首个 append A≈B；需纸面 ≥5pp |
| 几何日志 | 0（测量收益） | omlx 2× chunk 漂移 |
| 回放 load-call 惩罚 | 0（避免负收益） | M3-3 驱逐 4–8× |
| Engram 行缓存 | <1–2% | 读量 12.7 KiB/token，已被预取/异步覆盖 |
| MTP/DSpark 流式 | 0（维持关闭） | `a≈0.68`，oracle 1.387×，实测 1.000× |
| decode 深流水/overlap | 0（已有等价物/负结果） | ds4 host-bound；P1-A 已 +2.5–4% |
| CED/wide | 0（已在树上） | ds4 CED 4096 +9.3% 已收 |
| dense prefix | 0（已裁） | 双方一致 |
| 容量扩容 | 0（关闭） | 8000≈auto；128 GB 加不了 23 GB |
| 读速/布局 | 0–个位数 | 读已 89% 上限；R5 微测定 |

**一句话**：这轮 omlx 经验对 ds4 的**净速度收益约等于 0**；最大价值是
**避免重开已证负的 pin / MTP / overlap**，以及把 A/B 方法论补强。

---

## 10. 参考

### omlx

- `~/omlx-notes/dsv41_offload_state_and_levers_2026-10-17.md` §0.1/§4.1/§9.3/§9.6/§9.8–9.14
- `~/omlx-notes/dsv41_decode_regression_2026-10-18.md`
- `~/omlx-notes/dsv41-flash-prefill-speed-handoff.md`
- `~/omlx` commits：`96601d89`（pin 7/8）、`f0097064`（depth 6）、`fdc8383e`（seed）、
  `02f032ee`（Engram exact row）、`dd33a431`（read-ahead）、`268a22f0`（block ledger）、
  `58d92fae`/`669fb469`（CED）、`38659bc5`/`eb4a0fca`（wide step）

### ds4

- `V41_SSD_STREAMING_ROADMAP.md` §1.1/§1.3/§2.2/§3 R2
- `V41_STREAMING_ENGRAM_SPEED_PLAN.md` §0/§6.5 P3-A2/§9
- `HANDOFF-V41-STREAMING-SMALL-PREFILL.md` §2.3/§14/§17
- `ITERATION_PLAN_2026Q4.md` M3-1/M3-2/M3-3 行
- `LOCAL_INVENTORY.md` D 段 M3-3 长条（2026-10-12）、MTP/投机整类关闭条
- `speed-bench/M3/m3-3/run-20261012-010604/`、`speed-bench/M3/m3-3/r2a-reverted-20261012.patch`
- `tools/hotlist_v2_from_turns.py`、`tools/hotlist_v2_calibrate.py`、
  `speed-bench/mtp_ledger_replay.py`
- 代码坐标：`ds4.c:42849`（`ds41_prefill_seed`）、`ds4.c:43210`（CED 阈值）、
  `ds4.c:43243`（gather 准入）、`ds4_metal.m:1083`（hotness 16-token 衰减）、
  `ds4_metal.m:15217`（全局 cache 策略）、`ds4_engram.h`（264 B 行）
