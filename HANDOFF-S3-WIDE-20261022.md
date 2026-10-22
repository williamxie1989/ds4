# HANDOFF：S3 显式专家 buffer 扩到 wide sweep（2026-10-22 凌晨）

> 承接 `HANDOFF-SF-DS4-1FLASH-P0P2-20261021.md` 与上一轮的 S3 移植（未提交，默认关）。
> 本段**只改了一处**：`ds41_prefill_expert_sweep_supported` 放开 wide 限制（删掉
> `count <= 2048 && !wide`）。工作区原有的 DSpark/ledger 改动未被触碰。
> 结论：**8192 达到目标 2（prefill 增量 +36%、首 token +25%），2048 首 token 退化仍未解决（目标 3）。**

---

## 续接（2026-10-22 白天，优先读这一节）

> 本节覆盖 §4b/§4c 的归因结论。§1–§7 保留为上一段的历史记录，其中
> "decode 差距 = dispatch 数" 与 "GPU busy 0.356 vs 0.231" **已被证伪**。

### 结果

| 项 | 结果 |
|---|---|
| **decode 2048** | **22.9 → 26.7 t/s（配对 +16.6%，6 对，`decode` kind）**；decode2k 16 对 +14.1%（CI +12.4..+15.7） |
| **decode 8192** | **23.2 → 27.9 t/s（配对 +20.3%）** |
| 首 token 2048 | 66.8 → 64.1 ms（+5.2% 更快，CI +1.2..+8.9） |
| 首 token 8192 | 247.5 → 238.4 ms（CI 跨 0） |
| 闸门 | `correctness PASS (tokens; bitwise)`；`make test-metal-command-memory` / `./ds4_test --server` / `./ds4_agent_test` 全绿 |
| 改动 | 默认开，回退 `DS4_METAL_DISABLE_V41_SELECTED_MAILBOX` |

### 三项 kernel 移植：两项实测判负、第三项未移植（别再重开）

| 项 | 配对结果 | 机制 |
|---|---|---|
| `project_pair_q8` + `norm_pair_rows` | +0.4%（16 对 decode2k，CI -1.5..+2.7） | 成对 matvec 的 wave 数不变；成对 norm 把两个**各占一个 GPU 核**的单 threadgroup kernel 塞进一个 threadgroup 串行化 |
| `rope_quantize` | 0.0%（16 对，CI -1.1..+1.2） | 三个 kernel 都是 1~16 threadgroup，串行依赖也没有可回收的停顿 |
| `attention_decode`（fused decode attention） | **未移植** | 本树已有 `ds4_gpu_encode_flash_attention_gathered_heads`（吃同样的预 gather `comp_kv`）；fork 版多出来的 `comp_ids` 内 gather 只省 ~116 MiB/token 的 f32 往返 + 1 个 dispatch，按非停顿 dispatch ≈0 且 ~450 GB/s 折算约 **1%**，在噪声内。同等工时的 mailbox 拿到 14% |

**校准**：一个 dispatch 只有在它是**依赖停顿**时值 ~2.2 µs（上一轮 bf16-store matvec +2.8%），
否则 ~0（上一轮 norm bf16-store 只 +0.7%，本轮三项 0.0%）。**不要按 dispatch 数估收益。**

### 真实归因：逐层 selected-id 回读

`DS4_METAL_CB_TIMES=1`（2048 / 64 token）把 43.3 ms/token 拆成 **21.9 ms GPU span +
7.2 ms host encode + ~14 ms 的 per-CB 延迟**，而 **CB 数是 41/token**（fork 是 3）。
来源：`ds4_gpu_routed_moe_one_tensor` 为拿 router 的 6 个 expert id 建专家地址表，
每层调一次 `ds4_gpu_end_commands()`（commit + `waitUntilCompleted`）再读回张量 →
host 与 GPU 逐层交替。

上界实测：`DS4_MOE_REPLAY_SELECTED_IDS` 把同样的 id 从 host 喂进去、完全不回读
（先用 `DS4_V41_TRACE_EXPERTS` 录，再把 32 B 记录的 8 B pos/layer 头剥成 24 B 条目）：
**22.52 → 31.86 t/s（+41%），token 完全相同**。

### 落地方案（已收）

`kernel_dsv41_selected_publish` 把 id + 校验和 + 单调 tag 写进 CPU 可见共享环（64 槽，
tag 最后写）；host `ds4_gpu_dsv41_selected_take` **只 flush 不 wait**，然后轮询 tag
（每 1024 次查一次时钟，20 ms 上限）。超时或校验和不符 → 回落原阻塞回读，最坏等于今天。
效果：**41 次 commit-and-wait/token → 41 次 commit + 1 次 wait**（token 末尾那一次）。

### 为什么停在 +14~17% 而不是 replay 的 +41%

`take` 仍要等 GPU **到达** publish（该层最后编码的 dispatch），所以流水线深度是 1：
host 编码第 N+1 层能与 GPU 跑第 N 层重叠，但 host 不能领先一整个 token。要再往前必须把
**专家地址解析搬到 GPU**（fork 的 `ROUTER_MM` / `PACKED_INDEX` 结构，即 device 侧 id→地址表）。
本树只有 MXFP4 与 IQ2/IQ2 两种配方有 device 侧地址表（`use_iq2_stream_addr_table` 要求
gate+down 都是 IQ2_XXS），**V4.1 的 IQ2_XXS-gate / Q2_K-down 没有**——这是下一段的入口。

### 更正 §4c

"GPU busy/layer 我们 0.356 vs fork 0.231" **不可比**：fork 的 decode CB 从不逐个等待，
`GPUEndTime - GPUStartTime` 对它读回 0（fork 64 token 只报 189.6 ms GPU span = 3.0 ms/token，
不可能是同一份数学）。单看本树：21.9 ms GPU + 7.2 ms encode + 14 ms CB 延迟。
**kernel 效率从来不是杠杆，同步才是。**

### 新工具

`speed-bench/ab_bench.py` 新增 `decode2k` kind（单 frontier 2048 / 128 gen）：每次 run 重复同一个
prefill，专家 page cache 状态一致，消除 `decode` kind 的位置偏差（B 的第二个 run 紧跟 B 的 8192
sweep），同样时间内拿到 ~2 倍配对数、单跑离散度从 ±5% 降到 ±1%。**亚 1% 效应必须用它。**

### S3 wide 显式 prefill 已改为默认开（2026-10-22）

配对 3 组（同二进制、交替）：prefill **2048 → 161.3 t/s（+18.1%）**、**8192 → 372.2 t/s（+33.7%）**；
单跑 8192/16tok 是 504.4 vs 424.3（+19%），稳态 decode 28.61 vs 26.89 不退化。
**token 逐位相同**：2048 `e36dac1fe120`、8192 `cedd70f358ac`（开/关一致）。

首 token 的一次性成本 ~0.3 s 仍在，且**不是 I/O**（首 token 命中 237/240、miss_pread 仅 0.03 GiB；
停顿表现为 `gpu span 0.00` + 20–130 ms host encode；关掉 seed、有无 mailbox 都在）。**未定位到根因。**
因为它是一次性固定成本，按 `count×(1/137−1/167)=0.3 s` 得平衡点 ~230 行，所以设了门槛：

- `DS4_V41_PREFILL_EXPLICIT_MIN_ROWS`（默认 256；128 行 prompt 仍走 mmap、首 token 不变）
- `DS4_METAL_DISABLE_V41_PREFILL_EXPLICIT=1` 回退
- `DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1` 仍强制开（门槛降到 32），旧用法不变

2048 行时端到端仍是净赢：12.4 s + 0.40 s = 12.8 s 到首 token，mmap 是 14.4 s + 0.06 s = 14.5 s。
驻留路径 / TP / quality / encoder-resident 由 `ds41_prefill_expert_buffers_init` 的既有护栏挡住，
预算不够或 mlock 失败会回落 mmap。

### prefill 有没有被我改坏？没有

同机同 flags、`-n 0` 隔离（mailbox 开/关，各 2 跑）：

| | 2048 t/s | 8192 t/s |
|---|---:|---:|
| mailbox 开 | 178.7 / 179.7 | 478.9 / 489.4 |
| mailbox 关 | 178.6 / 178.5 | 425.8 / 448.5 |

**2048 完全一致，8192 是已知的 I/O 方差。** 交接里 134/300 与 175/430 的差是
`DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT`（S3 wide 显式 buffer）**默认关**造成的，
不是回归。当前实测：S3 关 = 135/300，S3 开 = 178/479。

S3 开仍然卡在首 token：**2048 首 token 349 ms**（S3 关 58–62 ms），mailbox **没有**修好它
（349 vs 交接记录的 405，只是顺带好了一点）。要让 S3 默认开，还得解决那 ~19 个
20 ms 的 commit→done 停顿（§3 的淘汰风暴假设）。

### 剩下的 41%：现在卡在 GPU，不在同步

`DS4_METAL_GPU_BUSY_PROFILE=1`，2048，`-n 0` 从 `-n 32` 里减掉：

| | wall/token | GPU busy/token | GPU 占用 |
|---|---:|---:|---:|
| mailbox 之前 | 43.3 ms | 22.3 ms | 52% |
| mailbox 之后 | 36.2 ms | **35.1 ms** | **97%** |
| replay（完全无 host） | 32.5 ms | 34.5 ms | ~100% |

同样的活现在要 35.1 ms GPU 时间而不是 22.3 ms，所以墙钟只降了 7.1 ms 而空闲被填掉了 21 ms。
**流水线已经吃满：剩下的杠杆是每 token 的 GPU 工作量，不是同步。**

- **更正**：上面那组 GPU busy 读数偏高（replay 的 wall 小于 GPU busy，说明 CB span 有重叠），
  不能用来估剩余空间。用同 session 交替 3 对测：mailbox 27.33/27.66/27.54，replay
  31.88/32.15/31.32 → **剩余同步空间正好 +15.8%（36.3 → 31.4 ms）**。
- 这 4.9 ms **不是轮询**：两臂的专家 cache 计数器逐字节相同（495147 hits / 11013 misses /
  102.09 GiB pread / 8.11 s pread），decode 段 pread 也相同（179 misses、1.66 GiB、118 ms）。
  它是**每 token 40 个 Metal command buffer**：`ds4_gpu_flush_commands` 为让 GPU 能跑到该层
  的 publish，每层分配并提交一个 buffer，~0.12 ms/个；周边记账只是几个整数运算。
- 要拿掉它 = host 不再需要在 router 与 gather 之间拿 id，即 **device 侧地址解析**；但 gather
  也不能在 pread 落地前读 slot，所以是三件耦合的事：①每层 device 侧 id→slot 地址表 + resolve
  kernel；②pread 改成异步，完成用 CPU 侧 `MTLSharedEvent` 信号、gather 前
  `encodeWaitForEvent`；③host 允许提前编码多层。正确性靠 event 等待保证，**不能**靠"miss 很少"
  ——decode 仍有 5.6 misses/token，逐 miss 阻塞回退会比省下的 4.9 ms 更贵。异步机器的骨架已有
  （`g_stream_expert_pread_pool_threads`、`pending_load`、`begin_selected_load`），但 device 表
  要在每个 cache install/evict 点维护。**这是一条独立工作流，不是收尾小改。**
- 超过 replay 的 32.5 ms：只能减少 GPU 工作。量级参照：routed 专家 6×9.49 MiB/层
  （2.28 GB/token）+ shared 37 MB/层（1.5 GB/token），按 450 GB/s 应是 ~8 ms，实测 35 ms
  → **单 token kernel 只跑到机器带宽的约 1/3，是 occupancy 不是带宽**。这是 kernel 项目
  （MPP/张量单元的单行形状，或 fork 的 `matmul_expand_rows` 式加宽），不是调度项目。

### 剩下的 15.8% 为什么拿不到，以及 GPU 时间花在哪

**同步这条路已经关死。** replay 不是有效上界：它的 id 从文件读，host 因此能跑在 GPU 前面，
同步 pread 被 GPU 工作掩盖（两臂 cache 计数完全一致）。真实推理里 id 只能由 GPU router 产生，
pread 又必须在 gather 之前落地，所以 host 最多领先一层。这一层余量只有两种花法，都测了：

| 变体（2048/32tok） | decode t/s | 去掉了什么 |
|---|---:|---|
| replay：id 白送，3 CB/token | 31.7 / 32.2 | poll + commit 全部 |
| replay + 每层强制 commit | 28.8 / 31.7 | 只去掉 poll（commit 值 1–2 ms） |
| mailbox（已落地） | 26.9 / 27.3 | 什么都没去：每层 1 commit + 1 poll |

poll 值 3–4 ms，commit 值 1–2 ms。把 poll 换成 GPU 侧 event 等待，只是把 3.6 ms 挪成 GPU 的
pread 停顿（5.6 misses/token × 0.66 ms = 3.7 ms）——等价交换。**只做 device 地址解析赚不回来。**

**GPU 时间分布**（`DS4_METAL_V41_SECTION_PROFILE=1`，新增，2048/16tok）：attention 22–24%、
router+shared 22–23%、routed gather+sum 23–30%、after_moe 22–30%，**四段几乎均分，没有热点**；
每段真实 GPU 工作只有 ~0.14 ms。下一步 decode 只能走 kernel 效率（更少更宽的 kernel），不是补丁。

### 未跑（诚实边界）

resident（非 streaming）不受影响（publish 只在 `g_ssd_streaming_mode` 生效）；DSpark/verify 走
batch 路径不 publish；TP 未测；`--logprob-vectors` 本机不可用（fixture 只有 flash-0731）。

---

## 0. 一页结论

| 项 | 结果 |
|---|---|
| wide sweep 显式 buffer | **已打通**：8K prefill 增量 **319/321 → 430/444 t/s（+36%，2 对，bitwise 腿）**；冷 8K 单跑 **397.5 → 525.0 t/s**；32K **655 → 751 t/s（+14.6%）** |
| 8K 首 token | **变好**：增量 209 → 167 ms（+25.3%，CI +22.5..+28.1）；冷 8K 单跑 749 → 377 ms |
| 8K decode | **小退化**：23.1/23.0 → 22.3/22.3（**−3.4%**，CI −3.5..−3.3）；第一轮只 −0.6%，按保守值记；冷跑 21.9 → 23.0 |
| 2048 prefill | +31%（134/135 → 175/180 t/s，2 对） |
| 2048 decode | +0.7%（22.2/22.3 → 22.4/22.5） |
| 2048 首 token | **仍坏**：68 → 405 ms（CI −83.7..−82.5），这是默认开 S3 的唯一硬阻塞 |
| 上一轮“seed 排序”假设 | **证伪**（见 §3）：seed 不排序不是原因；关掉 seed 后 2048 首 token 仍 307 ms |
| fork decode 归因 | **有新证据**（§4）：fork 的 15 个 glue 开关**全部关掉** decode 27.29 → 22.94（−16%），首 token 55 → 107 ms；用户前提“fork 无 disable 开关”不成立 |
| 闸门 | token-identical **PASS** + `--bitwise`（prefill logits 逐位）**PASS** |

---

## 1. 改动（唯一一处）

`ds4.c` `ds41_prefill_expert_sweep_supported`：

```c
/* Expert storage depends on layer bytes, not token count: one slot holds a
 * whole layer (gate/up/down), and it stays bound through every tile of that
 * layer, so a wide sweep reuses the same two slots as a single-chunk one and
 * only joins/restarts reads at layer boundaries (fork 896785c). Deferred
 * encoder sweeps stay on the mmap path. */
if (!getenv("DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT")) return false;
(void)wide;
return count >= 32u && batch_hc && batch_core && !encoder_only && !resume_encoder;
```

**不需要其它改动**：`ds41_graph_prefill_sweep` 的显式分支本来就把 `bind_layer` /
`read_start` 放在 tile 循环**外面**，slot 在整层所有 tile 之间保持绑定；tile 循环只重复
`ds41_moe_batch(..., gather_sweep=false)`。`decoder_suffix` 的 `ds41_decoder_prepare` 不碰
model span 映射，不会顶掉 slot 视图。

**已核对的不变量**：
- `gather_sweep` 仍被 `!explicit_experts` 排除（显式 buffer 已持有整层，混用 gather 会掉到 87 t/s）。
- 每层 `read_join(slot[il&1]) → bind_layer → read_start(slot[(il+1)&1])`；`read_start` 写的是
  **另一槽**，且 `read_join` 先等 `copy_wait`（cache→slot 的 blit）再 join 线程。
- 尾部 `buffers_free` 的 `synchronize + bind_layer(NULL) + munlock + discard + free` 未改。
- 预算不变：2 槽 × 3.56 GiB = 7.12 GiB = loader 的 prefill headroom，与行数无关。

---

## 2. 实测（同机 1017 Q2 / `--ssd-streaming-cache-experts 72GB` = 64.88 GiB / ctx 65536）

### 2.1 配对 A/B（`ab_bench --kinds decode`，1 quad = 2 对，256 gen）

A = 默认，B = `DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1`。跑了两轮，**以第二轮（带 `--bitwise`、
机器已回稳）为主**，第一轮作为独立复现：

`speed-bench/v41_m5max_ab/wide-explicit-bitwise-20261022/`（主，`correctness PASS (tokens; bitwise)`）：

| metric | A | B | gain | 95% CI |
|---|---:|---:|---:|---|
| prefill 2048 t/s | 133.6 / 135.4 | 174.8 / 180.4 | **+31%** | — |
| prefill 8192 增量（6144 行）t/s | 319.1 / 320.8 | 430.1 / 444.3 | **+36%** | — |
| first token 8192 ms | 209.0 | 167.0 | **+25.3%** | +22.5 .. +28.1 |
| decode 2048 t/s | 22.3 | 22.4 | +0.7% | +0.6 .. +0.8 |
| decode 8192 t/s | 23.1 | 22.3 | **−3.4%** | −3.5 .. −3.3 |
| **first token 2048 ms** | **68.4** | **405.2** | **−83.1%** | −83.7 .. −82.5 |

`speed-bench/v41_m5max_ab/wide-explicit-20261022/`（第一轮，机器上刚跑完 ~10 腿大模型，绝对值整体低 ~20%）：
prefill 2048 +27%（125/128 → 160/165）、prefill 8192 **+32%**（247/261 → 325/333）、
first token 8192 **+21.7%**（CI +14.2..+29.3）、decode 8192 −0.6%、decode 2048 −0.9%、
first token 2048 −82.2%（75 → 426）。两轮在“8K 大赢、2048 首 token 大输、decode 小幅”上一致；
decode 8192 的 −0.6% vs −3.4% 是两轮唯一的量级分歧，**按更保守的 −3.4% 记**。

prefill 行取自 `samples.csv`（ab_bench 只把本 kind 的 RECORD 指标打进表）。

### 2.2 冷 8K / 32K

冷 8K 单跑（`--frontiers 8192 -n 16`，一次 8192 行 sweep = 2 × 4096 tile）：

| | prefill t/s | first token ms | steady decode t/s |
|---|---:|---:|---:|
| A 默认 | 397.5 | 749.4 | 21.9 |
| B 显式 | **525.0 (+32%)** | **377.2 (−50%)** | 23.0 |

冷 8K 是**双赢**：A 在 wide 下只给 decoder suffix 的尾巴 page-in 专家页，首 token 要吃 749 ms；
B 整层读进 slot 再由 seed 填 cache，首 token 反而只要 377 ms。

32K prefill-only（`--frontiers 2048,32768 -n 0`，30720 行增量，单跑）：A **654.8** → B **750.7 t/s（+14.6%）**。
显式路径每层读整层（含 decoder-suffix 层），本来担心 32K 会亏，实测没有：**长上下文不退化**。

### 2.3 上一轮门槛下 8K 为什么“没走 S3”

`--frontiers 2048,8192 -n 0`（旧 gate）：A 319.8 / B **284.9** t/s。B 反而更慢，因为 2048 那一腿
走了显式路径（`F_NOCACHE`），**8192 那一腿的整层 sweep 少了前一层留下的 page cache**，而它自己
又走不到 S3。放开 wide 后同一对变成 A 324.8 / B **444.9**。

---

## 3. 2048 首 token：上一轮假设证伪，新证据

上一轮假设是“`ds41_prefill_seed` 的 slot→cache 拷贝没有和下一层读覆盖同一 slot 排序”。
**不成立**：

1. `ds41_prefill_seed` 的 blit 编在**当前 batch CB** 里（`gpu_copy_source=1` →
   `ds4_gpu_stream_expert_cache_load_selected_missing_with_source` 用 `g_blit_encoder(g_batch_cb)`），
   函数尾部 `ds4_gpu_end_commands()` → `ds4_gpu_finish_command_buffer(cb, 1, ...)` → `waitUntilCompleted`。
   **seed 返回时 blit 已完成**，下一层 `read_start` 不可能覆盖它。
2. 关掉 seed 重测（`DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED=1`，2048 + 2 token，单跑）：

| 配置 | prefill t/s | first token ms |
|---|---:|---:|
| A 默认 | 136.3 | 73 |
| A + 关 seed | 136.2 | 63 |
| B 显式 | 171.7 | **360** |
| B + 关 seed | 179.2 | **307** |

   关 seed 只把 2048 首 token 从 360 降到 307 —— **seed 不是主犯**，而且关 seed 只省 0.5 s prefill
   （profile 里 seed 合计 2.45 s，说明其中约 2 s 本来就和别的工作重叠）。

3. 真正的形态（`DS4_METAL_CB_TIMES=1` + `DS4_METAL_DISABLE_V41_DECODE_QUEUE=1`，首 token 窗口）：
   ~19 个 CB 呈 **commit→done 20 ms、GPU span 0.2 ms、encode 0.0 ms**；A 同配置只有 1 个这样的 CB。
   同期 `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` **一行 pread 都没打**，
   `miss_pread` 全程 0.96 GiB / 32 ms —— **不是专家 pread**。
   即：主机在 commit 与 wait 之间被 ~20 ms 的同步操作卡住，量级像**淘汰/释放 cache entry 时的
   `setPurgeableState` + 释放**（B 的 seed 会一次性换掉整份 64.9 GiB cache，A 的 sweep 是逐层
   admission，稳态无 churn）。
4. 旁证：**8K 反而不退化**（§2.2 首 token 749→377）。8K 的 decoder suffix 让 20–39 层只算尾巴
   （最后一层 `count=1`），seed 的 `target` 也跟着小 → 几乎没有 churn。

**下一步（按代价排序）**：
- a) 给 seed 加“churn 计数 + 每层 evict 数”打印，确认 20 ms 卡在哪（`prune_layer` / `prune_global` /
  `setPurgeableState` / `tensor_free`）。
- b) 若确认是淘汰风暴：让 seed 只**补**不换（保留 decode 真正要用的 entry，用 `protect` 列表把
  最近一层选中的 expert 钉住），或把 seed 的 eviction 改成延后到 sweep 结束后一次做。
- c) 另一条更省事的路：**只对 wide 开显式路径**（`wide` 时 `supported=true`，单 chunk 2048 仍走 mmap）。
  数据支持这个切法：wide 双赢、单 chunk 只有 prefill 赢而首 token 输。代价是多一个语义分支，
  与 `AGENTS.md`“no permanent semantic variants behind flags”冲突，**要么修好 2048，要么默认保持关**。

---

## 4. fork decode 归因（目标 1 的新证据，下一段直接用）

用户前提“fork 的 decode glue fusions 没有 disable 开关”**不成立**：fork 有 43 个
`DS4_METAL_DISABLE_V41_*`，其中 decode 相关 15 个。直接用 fork 自己 ablation（同机、同 cache、
`--frontiers 2048 -n 128`，各 1 跑）：

| fork 配置 | steady decode t/s | first token ms | prefill t/s |
|---|---:|---:|---:|
| fork 默认 | **27.29 / 25.28**（两次） | 55.0 / 56.9 | 182.1 / 189.0 |
| 15 个 glue 开关全关 | **22.94 / 22.11**（两次） | 106.7 / 65.6 | 182.8 / 187.7 |
| 关 A 组（router_one/rope_quantize/expand/hc_block/linear_bf/packed_m/topk_prefix） | 29.08 | 53.3 | 184.8 |
| 关 B 组（topk_shuffle/router_mm/router_batch/fast_lookup/packed_index/readback_*） | 25.56 | 62.3 | 183.8 |

**B 组逐项加回**（基线 = 15 个全关；每次只把一项打开，单跑）：

| 加回项 | decode t/s | 相对全关 |
|---|---:|---:|
| `ROUTER_MM` | 24.19 | +5.4% |
| `PACKED_INDEX` | 24.16 | +5.3% |
| `DECODE_PIPELINE_FAST_LOOKUP` | 24.11 | +5.1% |
| `TOPK_SHUFFLE` | 22.59 | −1.6% |
| `READBACK_POLL` | 22.45 | −2.2% |
| `READBACK_MAILBOX` | 22.41 | −2.4% |
| `READBACK_SPLIT` | 22.18 | −3.4% |
| `ROUTER_BATCH` | 22.02 | −4.0% |

读法（**单跑，须谨慎**）：
- **glue 家族确实值 12~16% decode**（默认 25.3~27.3 vs 全关 22.0~22.9，两次复现），
  与跨树 decode 差（我们 22.0 vs fork 26.3）同量级 —— 目标 1 的方向确认。
- **fork 自己的 decode 单跑方差就有 4~7%**（默认 27.29→25.28、全关 22.94→22.11，同机同参数），
  所以上面逐项 ±5% 的排名只是**指示性**的；下一段必须用 `ab_bench --a /tmp/sf-tree --b /tmp/sf-tree`
  配对跑（每项 2 对），别再拿单跑排名当结论。
- 全关（22.0~22.9）比“只关 A 组”（29.08）低得多 → **强交互项**；B 组 8 项全开 29.08，
  但三项最好的单项各只 +5%，剩下的 +12% 是组合效应，不是单项相加。
- 关 A 组反而更快（29.08 > 27.29）→ A 组里至少一项在本机（M5 Max）是负收益。
- `READBACK_*` 三项目前与本树结构不兼容（本树无逐层 host readback，见 LOCAL_INVENTORY §D），
  且单项加回都是负的 → **不移植**；但它们参与交互项，说明 fork 的 decode 是**整条流水线**。
- 若只能挑一个先移植：`DECODE_PIPELINE_FAST_LOOKUP`（+5.1%，且是本树完全没有的“解码 pipeline
  快速查找”结构，不依赖 fork 的 readback）。

---

## 4b. decode（目标 1）：差距 = dispatch 数，且全在 attention

本段没有改 decode 代码，但把差距量化到了“少几个 dispatch”的粒度。做法：给**两棵树**都加
dispatch 计数器（`ds4_gpu_compute_encoder` + atexit 打印；fork 树临时加，本树测完已删），
同 flags、`--frontiers 2048 -n 128`：

| | dispatch 总数 | 其中 prefill 2048 | decode 每 token |
|---|---:|---:|---:|
| 我们 | 236,682 | 51,825 | **1,444**（35.9/层） |
| fork | 122,062(128 tok) / 65,109(64 tok) | — | **890**（22.3/层） |

差 **554 dispatch/token**；实测每 dispatch ≈ **13 µs** → ≈ **7 ms/token**，正好是
42.5 ms（23.5 t/s）对 36–38 ms（26–27 t/s）的差。**decode 是主机编码受限，不是 GPU 受限**
（GPU busy 只有 15–23 ms/token）。

本树每层拆分（env 计数，测完已删）：

| 段 | dispatch/层 |
|---|---:|
| `ds41_graph_before_attention` | 2.1 |
| **`ds41_attention`** | **25.8** |
| `ds41_graph_after_attention` | 3.0 |
| `ds41_moe` | 4.0 |
| `ds41_graph_after_moe` | 1.0 |
| 合计 | 35.9 |

attention 内部：`project 10.0`、`rope+quantize+window store 3.0`、`select 2.9`、
`gather+decode 1.0`、`output 5.0`。projection 五次调用**每次恰好 2 个 dispatch**
（q_a 2、norm_q 2、q_b 2、kv 2、norm_kv 2，全 Q8_0）：MPP matvec 写 f32 + 单独一轮 BF16 round。
fused 路径不生效的原因已定位：`ds4_gpu_dsv41_matmul_q8_0_bf16` 在 M5 上被
`ds4_gpu_mpp_available()` 拒掉（在 streaming+M5 下放行它，dispatch 仍是 2.0、decode 仍 23.5）。

**移植顺序（按省下的 dispatch 排，全部可 bit-exact）：**
1. **MPP matvec + BF16 store 融合**：project 10→5、output 5→3、indexer −1（≈8/层 = 320/token ≈ 4 ms ≈ **+10%**）；
2. **`rope_quantize`**：kv rope + quantize + window store 合一（3→1，≈1 ms ≈ +2.5%）；
3. **norm pair**：norm_q + norm_kv 合一（−1/层）。
三项合起来就是 fork 的 ~890/token。

**负结果：单 dispatch 的 fused matvec 反而更慢。** 现成的
`ds4_gpu_dsv41_matmul_q8_0_bf16`（一次 dispatch、直接写 bf16）在 M5 上被
`ds4_gpu_mpp_available()` 拒；强行放行后 **15.64 t/s vs 23.37（−33%）**——legacy 非 MPP 几何
亏掉的远多于省下的那一次 dispatch。**结论：融合必须建在 MPP 路径上（给 MPP matvec 加 bf16 输出），
不能换 kernel。**

**负结果：pipeline fast lookup 不值得移植。** fork 的 add-back 显示 +5.1%，本树移植后实测
23.65 vs 23.66 t/s（三次背靠背）→ 中性，已回退；fork 那个 +5% 是单跑噪声（fork 自己
decode 单跑方差 4–7%）。

**本树自己的 decode 开关（记录）**：默认 23.70；`..._DECODE_PIPELINE_PORTS=1` 22.75（−4.0%）；
`..._DECODE_FLUSH=1` 22.52（−5.0%）；`..._DECODE_PIPELINE=1` 22.74（−4.1%）——三个都在生效，
队列结构已经吃满，剩下的是 dispatch 数。

## 4c. decode：两个 BF16-store 融合落地（目标 1 首次正向）

**改动（3 处，全部默认开、各自带回退 env）：**
1. `metal/dense.metal`：`kernel_mul_mv_q8_0_f32_bf16` —— Q8_0 matvec 的归约结果直接按 BF16 存
   （模板里本来就有 `BF16_STORE`，只是没导出）。
2. `metal/norm.metal`：`kernel_rms_norm_mul_f32_4_bf16` —— weighted RMS norm 存储时舍入。
3. `ds4_metal.m` / `ds4_gpu.h` / `ds4.c`：两个新入口（`ds4_gpu_matmul_q8_0_bf16_tensor`、
   `ds4_gpu_rms_norm_weight_bf16_tensor`），`ds41_matmul`/`ds41_norm` 先试融合、失败回落原链。

**效果**：dispatch **1,444 → 1,166/token（−19%）**；配对 A/B（A=融合，B=`..._DISABLE_V41_MATVEC_BF16_STORE`
+`..._DISABLE_V41_NORM_BF16_STORE`）：

| 轮次 | decode 2048 | 结论 |
|---|---|---|
| matvec 单项 | **+2.8%**（CI +0.9..+4.7）、8192 −0.8%（CI 跨 0） | 正向 |
| 两项合计 | +0.7% / +0.9%（两轮，CI 紧） | 正向但小 |

闸门：`correctness PASS (tokens; bitwise)` —— prefill logits 逐位相同（store 的舍入与
`ds41_bf16` 完全一致，norm 变体也只是把同一舍入提前到存储时）。

**校准**：160 个 dispatch ≈ 1.1 ms/token（≈7 µs/个），但第二刀 118 个只值 ~0.8%（被删的是小 kernel）。
**靠 dispatch 融合最多再拿 2~3%，到不了 26 t/s。**

**剩余差距的真实位置**：GPU busy/layer 我们 **0.356 ms** vs fork **0.231 ms**（同 flags、
`DS4_METAL_GPU_BUSY_PROFILE=1`）；prefill 也慢 25%（134 vs 176 t/s）。→ **kernel 效率**，
对应 fork 的 `ds4_gpu_dsv41_attention_decode`（fused decode attention）、`project_pair_q8` /
`norm_pair_rows`、`matmul_expand_rows`。**下一步 port 这三个，不要再数 dispatch。**

融合后每层 dispatch 分布（诊断用，测完已删）：project 5.0、rope+store 3.0、select 2.8、
gather+decode 5.9、output ~3、before_attn 2.1、moe 4.0、after_moe 1.0。

## 5. 复现命令

```sh
# 配对 A/B（本段主证据）
python3 speed-bench/ab_bench.py --a . --b . --kinds decode \
    --b-env DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1 --no-cache-check --budget 520 \
    --out speed-bench/v41_m5max_ab/wide-explicit-20261022

# 冷 8K 单跑
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf --ssd-streaming \
    --ssd-streaming-cache-experts 72GB --ctx-alloc 65536 \
    --prompt-file speed-bench/promessi_sposi.txt --frontiers 8192 -n 16 --cache-stats \
    [env DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1]

# 首 token 卡点定位
DS4_METAL_ENABLE_V41_PREFILL_EXPLICIT=1 DS4_METAL_CB_TIMES=1 \
    DS4_METAL_DISABLE_V41_DECODE_QUEUE=1 ./ds4-bench ... --frontiers 2048 -n 3

# seed 成本
DS4_METAL_GRAPH_PREFILL_PROFILE=1 ./ds4-bench ... --frontiers 2048 -n 16
# → 每层 `seed=%.3f ms`；A 合计 6.4 ms，B 合计 2453 ms

# fork ablation
cd /tmp/sf-tree && ./ds4-bench -m <gguf> --ssd-streaming --ssd-streaming-cache-experts 72GB \
    --ctx-alloc 65536 --prompt-file <promessi> --frontiers 2048 -n 128 --cache-stats \
    [env DS4_METAL_DISABLE_V41_<NAME>=1 ...]
```

---

## 6. 闸门状态（诚实边界）

- **token-identical：PASS**（ab_bench 两侧 256 token × 2 frontier，token sha256 相同）。
- **`--bitwise`（prefill logits 逐位）：PASS**——`wide-explicit-bitwise-20261022` 报
  `correctness PASS (tokens; bitwise)`：mmap 路径与显式 slot 路径在 2048/8192 两个 frontier 上
  prefill logits 逐位相同（同一份 routed 派发只换了源 buffer）。
- **长上下文：PASS**——32K prefill-only +14.6%（§2.2）。
- DSpark streaming/verify、SSD streaming 两态：**未跑**。改动默认关，且只在
  `!encoder_only && !resume_encoder && batch_hc && batch_core` 时生效；开默认前必须补。
- `--logprob-vectors` 在本机不可用（fixture 只有 flash-0731，模型 1017）。

## 7. 收口

- 本段唯一代码改动：`ds41_prefill_expert_sweep_supported` 放开 wide（1 处，编译无警告）。
- 目标 2（8K prefill 追 fork）：**8K 增量 319/321 → 430/444 t/s（+36%）、冷 8K 397 → 525（+32%）、
  32K +14.6%**。对 fork 的 505 还差 ~12%（增量口径），但 fork 那组数是上一轮单独测的，
  要定论得用同一协议重跑跨树对比。
- 目标 3（TTFT）：**2048 仍坏**（68 → 405 ms），但已证伪上一轮假设并给出新落点（cache 淘汰风暴，
  §3）；8K 首 token 反而 **+25%**。
- 目标 1（decode 26 t/s）：**未动代码**，但拿到了 fork 自己的 ablation 数据（§4），下一段可按
  “全关 + 加回一项”精确定位。当前 S3 的代价是 **decode 8192 −3.4%**，与“prefill +36% / 首 token +25%”
  是一笔明账。
