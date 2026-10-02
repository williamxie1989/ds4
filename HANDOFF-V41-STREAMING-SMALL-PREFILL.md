# HANDOFF: DeepSeek V4.1 Flash × `--ssd-streaming` — 小 chunk prefill / TTFT

> 新 session 入口。**只管一件事**：SSD streaming 下多轮对话里续写小 append 的 TTFT。
> 上一轮已把根因定位到代码行并实测；P1（Metal row-batch prefill）已实现、实测、**被仓库自带闸门否决并回滚**。
> 本文记录：根因、实测基线、否决证据、修正后的路线、测量工具、坑、建议 goal 文本。
> 分析日 2026-10-01。机器 = 本机 M5 Max / 128 GB（NAX 域）。
>
> **2026-10-02 P0a 已执行：parity 不通过，按纪律停在 P0a。** 全部证据与归因状态见 **§12**。
> **§12.5 单层判别已执行（见 §12.7）：gather 地址表 bind 逐位清白，ULP 源在 MoE 上游的 batch 共享算子——P0b 放行。**
> **2026-10-02 P0b 已实现并实测：速度门达标（831-token 续写 17.5 s → 9.6 s，sweep I/O 151→42 GiB），但逐位对拍 DRIFT——sweep 行数下 gather 走 `mul_mv_addr` pair（f32 mid），整层臂走 `mm_id` tile（f16 mid），不同内核族不可能逐位。按 §7.2 DRIFT 即否决；机制以 opt-in（`DS4_METAL_ENABLE_V41_STREAMING_SWEEP_GATHER`）保留，默认行为与开工前逐位一致。全过程与后续坐标见 §13。**

---

## 0. 一句话目标

`./ds4-server -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming`（约 `-c 393216`）下，
**续写 append < 1024 token 时 TTFT ≈ 15–17 s**。把它降到 ~7 s 量级，且**不改变任何 greedy 输出**。

---

## 1. 症状（用户原始日志）

```
20:01:20 ds4-server: chat ctx=37187..40619:3432 gen=829 TOOLS ... finish=tool_calls 62.687s
20:01:23 ds4-server: chat ctx=41448..42379:931 TOOLS prompt start
20:01:40 ds4-server: chat ctx=41448..42379:931 TOOLS prefill chunk 931/931 (100.0%) chunk=0.00 t/s avg=54.91 t/s 16.955s
20:01:40 ds4-server: chat ctx=41448..42379:931 TOOLS prompt done 16.988s
```

用户口径：大块 ~500 t/s，小块掉到 ~50 t/s。

**日志本身已给出路径判据**：`chunk=0.00 t/s` 说明这是该请求的**第一条**进度事件；且只有一条（`931/931`）。
- sweep 路径：V4.1 会话循环每次迭代只发一次 `prefill_chunk`（`ds4.c:77554`），一次 931 sweep → 只发一条 `931/931` ✅
- token-major 路径：每 token 发一条，首条应为 `1/931`

→ **用户的 931-token 是"单次 layer sweep"**，不是 token-major。这条判据后面反复用到。

---

## 2. 根因（已确认，带代码坐标）

### 2.1 V4.1 有独立的 session sync 分支

`ds4.c:77491` `if (ds4_session_is_ds41(s)) { ... }` —— **不走** 通用的 `ds4.c:78324` 续写分支。
每次迭代选 chunk 大小（`ds4.c:77515-77519`）：

```c
const uint32_t short_count = decoder_pending ? 0u :
    ds41_short_prefill_count(g, &e->weights, remaining);
const uint32_t count = short_count ? short_count : g->encoder_resident ?
    (remaining - 512u < g->prefill_cap ? remaining - 512u : g->prefill_cap) :
    ds41_prefill_count(g, remaining);
const bool layer_major = count > 1u;
```

`count == 1` → 每 token 一次 `ds41_graph_step`（decode 路径）；`count > 1` → layer sweep。

### 2.2 `ds41_prefill_count()`：1024 的硬门（`ds4.c:42093`）

```c
uint32_t minimum = 256u;
#if defined(__APPLE__) && !defined(DS4_NO_GPU)
    /* With at least half the experts cached, warm short appends beat a full
     * disk sweep. Leave a token-major tail to warm the following decode too. */
    if (g->streaming && g->pos &&
        ds4_gpu_stream_expert_cache_configured_count() >= DS4_N_LAYER * DS4_N_EXPERT / 2u)
        minimum = 1024u;            /* ds4.c:42110 */
#endif
    if (remaining < minimum) return 1;   /* ds4.c:42112  ← 一次一个 token */
```

`DS4_N_LAYER * DS4_N_EXPERT / 2 = 40 * 384 / 2 = 7680` experts。
- auto cache ≥ 7680 → `minimum = 1024` → < 1024 走 token-major
- auto cache < 7680 → `minimum = 256` → [256,1024) **走 sweep**

**用户是后者**：`-c 393216` 把 auto cache 压到 6959 experts（64.51 GiB，见 §3），所以 931 ≥ 256 → sweep。

### 2.3 sweep 的固定成本 = 整个 routed-expert 集合

`ds41_graph_prefill_sweep`（`ds4.c:42460-42467`）：

```c
const uint32_t first_count = total_count < encoder_chunk ? total_count : encoder_chunk;
if (!g->encoder_resident || il >= 20)
    ok = metal_graph_stream_prepare_join_layer(NULL, m, w, il, first_count,
            false, true, false, /*decode_only=*/false, &prepare, 1);   /* ds4.c:42464 */
if (ok) ok = metal_graph_stream_map_layer(m, w, il);                    /* ds4.c:42467 */
```

`decode_only=false` → `metal_graph_stream_prefill_layer_pagein_start`（`ds4.c:21215`）选

```c
const bool spans_ok = decode_only ?
    weights_model_map_decode_layer_spans(weights, il, &spans) :   // 只 static，专家交给 cache
    weights_model_map_spans(weights, il, il, false, &spans);      // 整层，含全部 384 专家
```

**对照**：decode 版 span 图 `model_map_span_vec_include_layer_decode`（`ds4.c:8437`）**明确排除** routed-expert blob —— 那些本该由 expert cache 供给。sweep 用的是包含版。

**实测（`DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1`）**：

```
Metal streaming prefill layer pread layer=0  tokens=2048 threads=8 ranges=104 bytes=4.96 GiB wait=275.489 ms thread=275.503 ms
Metal streaming prefill layer pread layer=1  tokens=2048 threads=8 ranges=96  bytes=4.02 GiB wait=0.033 ms   thread=264.475 ms
... (layer 2..39 各 ~3.73 GiB)
→ 40 层合计 151.1 GiB，Σ thread 时间 13.8 s，≈10.9 GiB/s（本机 SSD 顺序读实测 12.3 GiB/s）
```

**这就是那 ~15 秒。** 与 chunk 大小无关，所以：

```
吞吐 = chunk 大小 / ~15 秒
```

### 2.4 反证：sweep 完全不看 expert cache

`ds4-bench` 同配置只改 cache 预算，跑 2048-token 续写 append：

| cache | append 2048 t/s |
|---|---|
| auto（75 GiB，8145 experts） | 114.13 |
| `--ssd-streaming-cache-experts 16GB` | **130.02** |

**cache 变小反而快 14%** —— 因为 75 GiB 的 mlock 挤掉了真正喂 sweep 的 page cache。
若 sweep 吃 cache，变小只会更慢。→ **sweep 与 expert cache 完全无关。**

### 2.5 为什么用户配置下引擎"选对了"

在用户的 cache < half 配置下，931 token 走 sweep（~15 s），若改走 token-major 会更慢
（本机实测 token-major 42 ms/token，见 §3）。**两条路都烂**，不是阈值选错：

- sweep：固定 149 GiB I/O
- token-major：每 token 一次 decode step（18–42 ms）

真正的病是 **"每次 prefill 调用都要把整个专家集合摸一遍，而 cache 里已经躺着一半"**。

---

## 3. 实测基线（M5 Max / 128 GB / `gguf/DeepSeek-V4.1-Flash-Q2.gguf`）

模型事实：40 层 / 384 experts / top-6 / 文件 340.60 GiB / 主权重 152 GiB / Engram 188.83 GiB 只从盘读。

命令模板：

```sh
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max <MAX> --step-incr <APPEND> --gen-tokens 0 \
  --csv /tmp/out.csv
```

`--step-incr` 就是 append 大小；bench 只计最新一段 prefill（`prefill_tokens = frontier - previous`）。

| 场景 | 路径 | t/s | 墙钟 |
|---|---|---|---|
| 初次 prefill 2048 | sweep | 125–140 | 14.6–16.4 s |
| 续写 append 256 | token-major | 23 | 11.2 s |
| 续写 append 512 | token-major | 23 | 22.3 s |
| 续写 append **1023** | token-major | 23.9 | **42.7 s** |
| 续写 append **1024** | sweep | **63.1** | **16.2 s** |
| 续写 append 2048 | sweep | 109–120 | 18 s |
| 续写 append 8192 | wide sweep | ~450–500 | ~17 s |
| `--ctx-alloc 393216` + append 831 | sweep | **46.05** | 18.1 s |

→ **1024 处有一个 2.6× 的断崖**（本机）；但用户机器上两条路等值，断崖不是他的主要痛点。

auto cache 实测：

| 上下文 | cache target | experts |
|---|---|---|
| ctx 8193 | 72.51 GiB | 7822 |
| ctx 8193（另一次） | 75.50 GiB | 8145 |
| **ctx 393216（用户配置）** | **64.51 GiB** | **6959** ← < 7680，所以 `minimum=256` |

per-expert = 9.49 MiB。全专家工作集 = 40 × 384 = 15360 experts = 142.4 GiB。

---

## 4. 已否决：P1 Metal row-batch prefill

### 4.1 动机与改法

`ds41_cuda_row_batch_supported`（`ds4.c:42742`）在 Apple 上硬编码 `return false`，
所以 `ds41_short_prefill_count`（`ds4.c:42762`）在 Metal 上永不生效 —— 小 append 只能一 token 一步。
批处理机制本身是跨平台的：`ds41_graph_short_prefill`（`ds4.c:43285`）→ `ds41_graph_step_batch`
（`ds4.c:43012`）→ `ds41_graph_step_batch_spec`（`ds4.c:42810`）。

上一轮实际改了三处（**已全部回滚**）：

1. `ds41_cuda_row_batch_supported` 加 Apple 分支：`streaming && tp_world==1 && !quality/imatrix/image_count`
   + 每层 IQ2_XXS/IQ2_XXS/Q2_K + `configured_count() >= DS4_N_EXPERT` + kill switch。
2. `gather_moe`（`ds4.c:42850`）去掉 `spec &&`，让**纯 prefill batch** 也走流式 gather bind。
3. `ds41_short_prefill_count` 窗口 256 → 1024（Apple）。

### 4.2 实测：只值 1.4×

128-token 续写 append：

| | t/s | 墙钟 |
|---|---|---|
| row batch ON | 30.2 | 4.24 s |
| kill switch（一 token 一步） | 21.8 | 5.87 s |

**8 行 batch = 265 ms，单 token decode step = 46 ms** —— 8 行花掉 5.7 个 decode step 的时间，摊薄极差
（每行每层都要重建一次 selected-address 表）。远低于开工前估的 3–4×。

### 4.3 否决证据：仓库自带比较器判 DRIFT

```sh
# ON/OFF 两腿都加 --dump-frontier-logits-dir
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
  --frontiers 2048 2176 2304 2432 2560 2688 2816 2944 3072 \
  --ctx 3073 --model gguf/DeepSeek-V4.1-Flash-Q2.gguf --backend metal --quality false \
  --quant-bits 2 --vocab 129280 --output /tmp/cmp.json <OFF_DIR> <ON_DIR>
```

```
DRIFT: 9 expected frontiers
{"status": "drift", "strict_float32_identity": false}
```

129,280 个 logit **全部**不同，max |Δ| = 0.5–1.5。这不是归约顺序的 1 ulp 抖动。
仓库对续写 prefill 的契约是 `tests/test_deepseek41_dspark.c::check_short_prefill`
（`tests/test_deepseek41_dspark.c:144`）里的 `worst == 0`。

### 4.4 为什么仓库里从没发现

`check_verify_parity`（`tests/test_deepseek41_dspark.c:29`）与 `check_short_prefill`
（`:144`）用的 `ds4_engine_options`（`:40` / `:151`）**都没有 `.ssd_streaming`**。
于是 `gather_moe = spec && g->streaming && ...` 恒为假 ——
**Metal 上"流式 + 专家 cache gather"这条路从来没有被任何测试覆盖过。**
上一轮是第一次点亮它，它当场就漂了。

⚠️ **这条直接改变修复排序**：P0（让 sweep 吃 cache）用的是**同一套 address-table gather 内核**，
会继承同一个漂移。**不先解决 parity，P0 不能开工。**

### 4.5 回滚状态（务必确认）

- `ds4.c` / `ds4_metal.m`：**已完全回到用户基线**，上一轮一字未留。
- 只保留 3 行测试改动，修的是**本来就红**的闸门：用户本地把 medium sweep 从 CUDA-only 放开到
  Apple（`ds4.c:42113` 的 `#if !defined(DS4_ROCM_BUILD)`，判据在 `ds4.c:42119`），但
  `tests/test_deepseek41_prefill.c:37-43` 的 Apple 期望表没跟着改。现在：

```sh
make tests/test_deepseek41_prefill && ./tests/test_deepseek41_prefill --dispatch
# → V4.1 cold/warm and TP prefill dispatch, tile boundaries and debug/imatrix fallbacks: PASS
```

- `make` 全绿。

---

## 5. omlx 对照（用户问过"是否有帮助"）

| omlx 近期工作 | 对 ds4 是否可搬 |
|---|---|
| **expert-boundary prefill chunking**（按专家排序切块，"每专家每次调用最多读一次"） | ❌ ds4 sweep 本来就每层每专家读一次，无此缺口 |
| **CED**（decoder 半段 prefill 只算尾部窗口） | ✅ **ds4 已有**：`decoder_suffix`（`ds4.c:42416`，门 `ds41_decoder_suffix_min_rows()` `ds4.c:42385` 默认 4096）。但 ds4 只用它省算力，**仍然把整个 decoder 层 pread + map**（layer 39 只算 1 行却照读 3.73 GiB）→ 可回收 |
| **wide-step 8192** | ✅ ds4 已有（`carry_cap` / `DS4_METAL_DISABLE_V41_WIDE_PREFILL`）—— 这是"大块 500 t/s"的来源 |
| **TTFT 那段**（"long prefill routes to most experts per layer … re-fetched an expert … TTFT 16.60 s → 0.97 s"） | ✅ **这才是对应项**：omlx 的 before 状态 = "重读常驻 cache 里已有的东西"，ds4 sweep 正是如此 |
| 不 pin 热专家（保精度） | ds4 没 pin，OK |
| LRU→optimal +17pp 命中率 | ds4 已有 route-hotness |

参考文档：`~/omlx/docs/MoE_Expert_Offload.md`（该段落在 "Performance" 与
"Why not pin the hot experts instead?" 之间）。

---

## 6. 修正后的路线

### P0a（前置，必做）· 给 Metal 流式 gather 建立 parity —— **已执行：不通过，见 §12；§12.7 判别后 gather 清白，P0b 放行**

**这是唯一的第一步。** 在它通过之前，任何"让 prefill 吃 expert cache"的改动都不能上。

1. 把 `check_verify_parity` / `check_short_prefill` 参数化成 `ssd_streaming = true`
   （加 `--short-prefill-ssd` / `--verify-parity-ssd` 模式，或 env 开关），
   `opt.ssd_streaming_cache_bytes` 给一个能容纳一层专家的预算（≥ `DS4_N_EXPERT * 9.49 MiB` ≈ 3.6 GiB）。
2. 用现成的二分入口缩小范围：`DS4_TEST_VERIFY_CASE=PREFIX,ROWS`（`tests/test_deepseek41_dspark.c:56`）。
3. 判据：先确认 **rows=2** 就漂，还是只有 rows>1 的 batch attention 漂。
   - 若 rows=2 就漂 → 问题在 **address-table gather MoE**（`ds4_metal.m:44740` `use_iq2_batch_selected_addr`
     那条路）本身，需要核对其归约序与 `ds4_gpu_routed_moe_one` 是否一致。
   - 若只有多行漂 → 问题在 **batch attention / KV 写入调度**（`ds41_graph_step_batch_spec` 的
     `prefill_only` 分支，`ds4.c:42956` `first_output = count - 1`）。
4. 注意 CUDA 侧的契约写在 `tests/test_deepseek41_dspark.c:142` 的注释里
   （"The production continued prefill of two to eight rows is one-token decode: every routed recipe
   keeps scalar reductions through eight rows"），并由 `tests/test_deepseek41_prefill.c:50-53`
   的期望表与 `ds41_short_prefill_count` 共同约束 ——
   Metal 上这条**不成立**，要么修内核，要么明确记录为设备差异并放弃该路径。

### P0b · 让 sweep 读 miss 集而非整层

parity 修好之后：

- `ds4.c:42464` 的 `decode_only` 改 `true`（→ `weights_model_map_decode_layer_spans`，只 static）
- `ds4.c:42467` 的 `metal_graph_stream_map_layer` 改 `..._decode`
- `ds4.c:42653` 的 `ds41_moe_batch(..., false, false)` 最后一个参数改 `true`（gather bind）
- 可能还要放开 `ds4_metal.m:14826` `ds4_gpu_stream_prefill_batch_selected_addr_auto_max`
  （384 experts 时 auto max = 800；sweep 的 `n_tokens` 会更大）

预期：sweep 149 GiB → ~74 GiB（cache 覆盖 ~50%），14 s → ~7 s；
2048 chunk 18 s → ~11 s；8192 chunk 500 → ~900 t/s。

> **2026-10-02 已执行**：四处改动全部落地，速度超预期（831 append 46→86 t/s，I/O 151→42 GiB），
> 但 §7.2 逐位门 **DRIFT 否决**——sweep 行数下 gather 与整层 arm 分属不同内核族
> （`iq2_batch_stream_addr` mv-pair/f32-mid vs `mm_id` tile/f16-mid）。
> 机制 + 同 build 运行时开关保留为 **opt-in**。全部数据与后续坐标见 **§13**。

### P2 · CED 路径下不读 decoder 半段专家

`decoder_suffix` 生效时（`total_count >= 4096`），layer 20–39 只算收缩尾部（layer 39 只算 1 行），
但每层照读 3.73 GiB。8192 chunk 下约一半 sweep I/O 花在这。只影响 ≥4096 的宽步，小 append 无收益。

### 零代码现状旋钮

- `--ssd-streaming-cache-experts 16GB`：2048 sweep +14%（130.0 vs 114.1 t/s），
  但拖慢 decode，**只当测量旋钮**，不是修复。
- `DS4_METAL_DISABLE_V41_SSD_MEDIUM_SWEEP=1` / `DS4_METAL_DISABLE_V41_WIDE_PREFILL=1`：把 append 打回 2048 步长，用于隔离。

---

## 7. 测量方法（现成工具，别再自己造）

### 7.1 分阶段 profile 环境变量

| 变量 | 作用 |
|---|---|
| `DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1` | **逐层 pread 字节/毫秒**（本次定位的关键证据） |
| `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` | selected/split/load 分项计时 + 命中率 |
| `DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1` | 逐层 hits/misses/evictions/pread |
| `DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA=1` | 同上，增量 |
| `DS4_METAL_GPU_BUSY_PROFILE=1` | 每 64 CB 累计 GPU busy |
| `DS4_METAL_CB_TIMES=1` | 每 CB driver/queue-wait/gpu/gap |
| `DS4_METAL_GRAPH_PREFILL_PROFILE=1` | 逐层 encode/execute |
| `DS4_METAL_MOE_STAGE_PROFILE=1` + `DS4_METAL_MOE_STAGE_PROFILE_LAYER=0` | 打印 `path=mm_id / iq2_batch_stream_addr / ...`，**判定实际 MoE 路径** |
| `DS4_METAL_ENCODER_TIMELINE=<file>` | 逐 kernel GPU 时间线 |

⚠️ `DS4_METAL_GRAPH_PREFILL_PROFILE` 在 V4.1 上实测**没有输出**（`profile` 谓词与 ds41 分支不匹配），
别指望它；用 pread profile + MoE stage profile。

### 7.2 逐位对拍（唯一的正确性口径）

```sh
mkdir -p /tmp/A /tmp/B
DS4_BENCH ... --dump-frontier-logits-dir /tmp/A
DS4_BENCH ... --dump-frontier-logits-dir /tmp/B      # 另一腿
python3 gguf-tools/quality-testing/compare_frontier_logits.py \
  --frontiers <全部 frontier> --ctx <ctx+1> --model <gguf> --backend metal \
  --quality false --quant-bits 2 --vocab 129280 --output /tmp/cmp.json /tmp/A /tmp/B
# 期望 IDENTICAL；DRIFT 即否决
```

注意：`--output` 文件**必须不存在**；`--frontiers` 要列**全部**（含首个 frontier）；
`--ctx` 要传 dump 元数据里的 **alloc ctx**（如 `--ctx-alloc 393216` 时传 393216），不是 frontier+1，否则 INVALID（§13.5）。

⚠️ **这道门度量的是可复现性，不是数值正确性**。契约原文在
`tests/test_deepseek41_dspark.c::check_short_prefill`：同一 token 序列走不同执行路径
（batched append vs 逐 token decode）必须 `REQUIRE(worst == 0)`——logits 逐位全同。
它的职责是保证输出不随"怎么算"（分块、绑定方式、内核选择）改变，server 的 KV checkpoint/resume、
test-vectors 回归都建立在这上面。**DRIFT 否决的是"行为变了"，不代表新路径算错**——
例如 P0b 的 gather 臂 f32 mid 其实比现状 mm_id 的 f16 mid 更接近 CPU 参考（§13.3）。
若要回答"哪边更准"，需另立第三臂（CPU/scalar 参考）对比；那属于"要不要放宽契约"的政策决策（§11 决策分叉），不是这道门能回答的。

### 7.3 仓库自带单测闸门

```sh
make tests/test_deepseek41_prefill && ./tests/test_deepseek41_prefill --dispatch   # 快，无模型
./tests/test_deepseek41_dspark <MODEL> --short-prefill <PROMPT>                    # 需要模型
./tests/test_deepseek41_dspark <MODEL> --verify-parity <PROMPT>
```

⚠️ 这两个 dspark 闸门当前**只能用常驻引擎**；本机 128 GB 跑不了常驻 V4.1（
`V4.1 needs 155.87 GiB before the expert cache; safe budget 107.52 GiB`）。P0a 的一部分工作就是把它们改成可流式跑。

---

## 8. 坑

1. **`DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX` 同时被两个门读**：
   `ds4.c:20024` 的 `metal_graph_stream_prefill_batch_selected_addr_auto_max`（决定 prepare 的 span 集）
   与 `ds4_metal.m:14826` 的 `ds4_gpu_stream_prefill_batch_selected_addr_auto_max`（决定 MoE 内核分派）。
   两者默认值不同（FLASH 760 vs 384-expert 800）。改一个必须同时想另一个。
2. **`--ctx-alloc` 会改变路径**：ctx 越大 → auto cache 越小 → 跨过 7680 阈值 → `minimum` 从 1024 掉到 256。
   A/B 必须锚定 ctx。
3. **expert cache 变大不一定快**：sweep 场景下它会抢 page cache（§2.4）。
4. **`chunk=0.00 t/s` 是第一事件的标志**，可用来从日志反推走了哪条路（§1）。
5. **`gather_moe` 现在是 `spec &&`**（`ds4.c:42850`）：纯 prefill batch 在流式下拿不到 gather bind。
   这是 P1 的改动点，也是 P0b 必须处理的点。
6. **`ds41_graph_step_batch_spec` 不设置 `g->valid`**，靠 `ds41_graph_reset`（`ds4.c:40411`）设 true；
   `ds41_graph_short_prefill` 入口要求 `g->valid`，所以 fresh session 第一条路径必须是 reset 之后。
7. **静态权重图**：流式下 `metal_graph_stream_map_decode_static_all`（`ds4.c:21710`）只在 sweep 末尾
   （`ds4.c:42711`）与 `layer_resident`（streaming && quality，`ds4.c:42003`）时安装。
   token 路径不装图 —— 靠引擎加载时装的 static decode map。改动前先确认当前 map 状态。
8. **工作区有未提交改动**：`ds4.c`（+207）、`ds4_metal.m`（+28）、`tests/test_deepseek41_attention.c`、
   `tests/test_deepseek41_graph.c` 都是用户的在制品。**动手前 `git status` + `git diff` 确认，不要 checkout 覆盖。**

---

## 9. 环境速查

- 本机：Apple M5 Max / Mac17,6 / 128 GB（NAX 域，MPP tensor API 可用）
- 模型：`gguf/DeepSeek-V4.1-Flash-Q2.gguf`（340.60 GiB）；`gguf/DeepSeek-V4.1-Flash-DSpark-ds4.gguf`（7.97 GiB）
- 本机 SSD 顺序读实测 **12.3 GiB/s**
- 构建：`make`（Metal）；测试：`make test`、`./ds4_test --all`、`make test-glm53-kda`
- 用户服务启动模板：`start.md` 第 149-153 行
  （`DS4_METAL_DISABLE_V41_ROUTER_FUSE=1 ./ds4-server -m gguf/DeepSeek-V4.1-Flash-Q2.gguf -c 393216 --port 8055 --ssd-streaming`）
- codegraph 传 `projectPath=/Users/xieyongliang/ds4`
- 上一轮遗留的 bench 脚本与日志：`/tmp/ds41bench/`（`A_2048.csv` `B_256.csv` `C1.csv` `P.log` `Q.log` `rb_on*` `rb_off*` `cmp.json`）
  —— `/tmp` 重启即清；§7 的命令可完整重建，脚本本身无独有信息。

---

## 10. 文件索引

| 文件 | 作用 |
|---|---|
| `ds4.c:77491` | V4.1 session sync 分支（**改动的落点**） |
| `ds4.c:77515-77519` | chunk 大小选择 |
| `ds4.c:42093` / `:42110` / `:42112` | `ds41_prefill_count`、1024 硬门、`return 1` |
| `ds4.c:42159` | `ds41_encoder_acquire`（encoder 半常驻） |
| `ds4.c:42385` | `ds41_decoder_suffix_min_rows`（CED 门，默认 4096） |
| `ds4.c:42416` | `decoder_suffix`（= omlx 的 CED） |
| `ds4.c:42464` / `:42467` | sweep 的 prepare（`decode_only=false`）/ map（**P0b 落点**） |
| `ds4.c:42653` | sweep 的 `ds41_moe_batch(..., false, false)`（**P0b 落点**） |
| `ds4.c:42742` / `:42762` / `:43285` | row batch 门 / 窗口 / `ds41_graph_short_prefill` |
| `ds4.c:42810` / `:42850` | `ds41_graph_step_batch_spec` / `gather_moe`（**P1 落点**） |
| `ds4.c:21215` | prepare 的 span 集二选一 |
| `ds4.c:8437` / `:8579` | decode span 图（排除专家）/ `weights_model_map_decode_layer_spans` |
| `ds4.c:21710` | `metal_graph_stream_map_decode_static_all` |
| `ds4_metal.m:44740` / `:44756` | `use_iq2_batch_selected_addr` / `use_iq2_cached_batch`（GLM-only） |
| `ds4_metal.m:14826` | `ds4_gpu_stream_prefill_batch_selected_addr_auto_max` |
| `ds4_metal.m:45214` / `:45605` | `prepare_selected_batch` / `mul_mm_id_mapped_tile_resources(stream_resources, overflow)` |
| `tests/test_deepseek41_prefill.c:9` | `--dispatch` 分发表（快，无模型） |
| `tests/test_deepseek41_dspark.c:29` / `:144` | `check_verify_parity` / `check_short_prefill`（**P0a 落点**） |
| `gguf-tools/quality-testing/compare_frontier_logits.py` | 逐位对拍权威口径 |
| `docs/SSD_STREAMING.md` | 流式设计说明 + M5 Max 参考数 |
| `OMLX_WAVE_PORT_ANALYSIS.md` §5 | 五桶分解 / roofline / A/B 纪律（本仓方法论） |
| `~/omlx/docs/MoE_Expert_Offload.md` | omlx 侧 TTFT 战例（对照） |

---

## 11. 新 session 建议 goal 文本

> **状态（2026-10-02 之后）**：P1 已回滚（§4）；P0a 已做，ULP 源在 MoE 上游 batch 算子（§12/§12.7：
> gather bind 本身逐位清白）；P0b 已实现、速度达标（831 append 17.5→9.6 s）、被 §7.2 门 DRIFT 否决，
> 机制 opt-in 保留（§13）。**<1024 append 的"调度层"方案已穷尽**——根因是内核族差，不是调度。
>
> **开工前先让用户在决策分叉上选，不要直接动手：**
>
> - **分叉 A（默认：维持逐位契约）→ P0c 内核移植**。唯一能同时拿到 ~9 s 与逐位不变的路。
>   坐标与首个闸门见 **§13.4**：先给 `--moe-bind-parity` 微测加"同 kernel 名、addr vs whole-map、
>   rows 扫到 1024"的用例并加 IQ2_XXS 的 mm-id addr 模板实例化（`metal/moe.metal:8239/8931`，
>   GLM 驱动 `ds4_metal.m:40678` 可抄），**微测逐位通过后**才把 sweep gather 接到该族，
>   最后过 §7.2 对拍 + 同 build forward/reverse bench 门。目标：831/931 append ≤ ~8 s（冷 cache 口径）。
> - **分叉 B（放宽契约）→ 什么都不用开发**。§13 的 gather 就是现成 opt-in
>   （`DS4_METAL_ENABLE_V41_STREAMING_SWEEP_GATHER=1`，9–10 s）。代价必须写明：
>   top5 已见次序互换（§13.3），长链 greedy 迟早分叉；temperature>0 采样流直接不同；
>   test-vectors 需重定基线；与已存盘 KV checkpoint 的前后一致性故事要重建。用户明确接受才走。
> - **不要重测/重试任何 P0b 式调度方案**：DRIFT 机理已定（§13.3：`mm_id` f16-mid vs
>   `mul_mv_addr` f32-mid，族差），再排一次队只会再撞同一次 DRIFT。
>
> 闸门口径不变（§7）：`--dispatch`、`--moe-bind-parity`、§7.2 对拍（注意 `--ctx` 传 alloc ctx）、
> bench 同 build 同机 forward+reverse、每臂记 `[cache target]` 与 `-c`。
> 提不动就用数据证明无效并写回本文档。

---

## 12. P0a 实测（2026-10-02 session）：Metal 流式 gather parity **不通过**

### 12.1 交付物（保留在工作区）

`tests/test_deepseek41_dspark.c`（+40/-13，只动测试，`ds4.c`/`ds4_metal.m` 一字未动）：

- 新模式 `--verify-parity-ssd` / `--short-prefill-ssd`：同一套检查跑 `opt.ssd_streaming = true`。
- cache 预算 helper `ds41_test_streaming_cache_bytes()`：默认 **16 GiB**，`DS4_TEST_STREAMING_CACHE_GIB` 覆盖。
  ⚠️ **坑（新增）**：`ssd_streaming_cache_bytes` 是"prefill headroom + 动态 cache"的**总预算**。
  ctx 4096 下 headroom 固定吃掉 ~7.12 GiB；给 8 GiB 只剩 **95 experts** 的动态 cache，
  低于 gather 准入门 `configured_count() >= DS4_N_EXPERT (384)` → gather 静默不启用。
  **预算必须给到 ≥12 GiB 左右才真正测到 gather**；确认手段：`DS4_METAL_MOE_STAGE_PROFILE=1`
  看 `path=iq2_batch_stream_addr` 是否出现（本次就是这么确认的，不是猜的）。
- `make tests/test_deepseek41_dspark && ./tests/test_deepseek41_prefill --dispatch` 全绿；`make` 全绿。

### 12.2 结果矩阵（Q2 + Metal + streaming，16 GiB 预算 → 958 experts 动态 cache，gather 确认在跑）

| fixture | case | 结果 |
|---|---|---|
| security | `--short-prefill-ssd`（token-major 续写 2–8 行，10 组） | **全部 max diff = 0** ✅ |
| security | verify rows=2 / 3 / 4（prefix 40） | exact ✅（**后证是 fixture 幸运**） |
| security | verify rows=5/6/8（prefix 40） | row≥4 漂，max 0.879/1.106/0.408/0.560；**row 4 值跨 batch 大小逐位相同** |
| security | verify rows=5（prefix 38、100） | **exact**（排除了"行号窗口"与"绝对位置窗口"两个假设） |
| **story** | verify **rows=2**（prefix 40） | **row 0 就漂，max 0.180**；rows=3/4/5/8 全部行漂，值逐位不变，随行号级联放大（0.18→0.73） |

结论：`check_verify_parity` 的 rows≤8 逐位契约（`:142` 的标量归约保证）**在 Metal streaming gather 上不成立**。
批次内每行对单 token decode 存在**数据依赖的 ULP 级差异**，只有命中 bf16/router 量化边界才放大成可见漂移——
security prompt 在 rows≤4 的窗口恰好一次没放大（所以之前"≤4 精确"是运气），story 从 rows=2 起每行都放大。
**P1 当时的 DRIFT（幅度 0.5–1.5、全部 logit 不同）与此同机制，P1 不用重测。**

### 12.3 两个工程事实（此前没人验证过，实测坐实）

1. **streaming 下 gather 是 batch MoE 的唯一 bind**：`DS4_METAL_DISABLE_V41_STREAMING_VERIFY_GATHER=1`
   后 verify 在 layer 1 直接失败（`Metal model range ... not covered by mapped model views`，
   `ds41_moe_batch` 最后一个参数 `!gather_streaming_experts` → `force_resident=true`）。
   → **§6 P0a 设想的"kill switch 二分对照组"在 streaming 下不可用**，归因不能靠这个开关。
2. `ds41_streaming_verify_moe_gather_admitted` 是 `#ifdef __APPLE__` **Metal-only**；
   生产里 verify-gather 挂在 `DS4_EXPERIMENTAL_DSPARK_STREAMING` 后（`ds4.c:86283`）。
   ⚠️ 本结果说明该实验开关**当前产出的 verified 行不逐位等于 decode**——若有人开着它，greedy 会漂。建议在代码注释/发布说明里点名。

### 12.4 归因状态（诚实地说：还没定谁）

ULP 源有两个候选，现有证据**分不开**：

- **(a) batch 共享算子**（`ds41_matmul_batch`、`ds41_route_batch`/`router_select_batch`、rows 版 quantize 等）
  与标量版归约序不同 → route top-6 或 bf16 边界翻转。
- **(b) gather MoE 内核本身**（`iq2_batch_stream_addr` 家族：addr pair-swiglu + addr q2_k sum6）与 mul_mm_id/标量内核归约序不同。

已排除的：行号 tile 窗口、绝对位置窗口、地址表 host 回读截断（`prepare_selected_batch` 全量读
`n_tokens*n_selected` 个 id，`ds4_metal.m:17804`）、gate/down 内核选路在 4→5 token 间翻转
（stage profile 实测 rows=4 与 rows=5 都走 `iq2_batch_stream_addr`，无翻路）、expert 驱逐竞态（漂移逐位确定）。
旁证：security 38,5/100,5 全行逐位 exact 证明 batch 管线**可以**逐位等于 decode——ULP 差不是恒有的。

### 12.5 下一步（P0b 的生死判别，先做这个再谈 P0b）——**已执行，结果见 §12.7：逐位相同**

**单层强制对拍 microtest**（预计只动测试文件）：
在 streaming 引擎里 `metal_graph_stream_map_layer` 把**一层**的 384 专家全部映射进视图（~3.65 GiB），
同一份 x/ids/weights 分别调 `ds4_gpu_routed_moe_batch_tensor(..., force_resident=true)`（整层映射版内核）
与 `force_resident=false`（gather 地址表版内核），逐位比对 routed 输出。

- **若逐位相同** → ULP 源在 batch 共享算子（a），与 P0b 的改动无关；
  P0b 的 A/B 门（同 prompt、同 router、同 pre-MoE 管线，只换 MoE bind）对它天然不敏感，**P0b 可以开工**。
- **若逐位不同** → gather 内核与整层内核归约序有差（b）；P0b 换 bind 必然改变 sweep 数值，
  只能先修内核（核对 addr pair-swiglu/sum6 与 `ds4_gpu_routed_moe_one`/mul_mm_id 的归约序），修不齐就到此为止。

判据工具沿用 §7.2 的逐位口径；microtest 判 `memcmp == 0`。

### 12.6 遗留物

`/tmp/p0a/`：`short_ssd.log` `parity_40_2.log` `parity_full.log`（矩阵在 40,6 中止）`p44/p45/p48.log`
`phase_38,5.log` `phase_100,5.log` `story_40_{2,3,4,5,8}.log` `prof_4.log` `prof_5.log`。重启即清，§7+§12.1 的命令可完整重建。
新 fixture 提醒：`tests/long_context_story_prompt.txt` 比 security prompt 严苛得多，**归因/验收请两个都用**。

### 12.7 §12.5 判别已执行（2026-10-02）：**gather 地址表 bind 逐位清白 → P0b 放行**

实现：`tests/test_deepseek41_dspark.c --moe-bind-parity`（测试内新增，+198/-6 累计；生产代码零改动）。
流程 = `metal_graph_stream_map_layer` 映射整层专家 → 同一份随机 f32 激活 + 每批 48 个互异专家 id + 随机路由权重 →
`ds4_gpu_routed_moe_batch_tensor` 各跑一遍 `force_resident=true`（整层映射 bind）与 `false`（streaming 地址表 gather），
输出 `memcmp` 逐位比对。层 {0, 20, 39} × rows {2, 5, 8}：

- **9/9 全部 `memcmp == 0`。** 结论：地址表取数与 mul_mv 内核算术**不引入任何位差**；
  §12.4 的候选 (b) 排除，ULP 源收敛到候选 (a)——**MoE 上游的 batch 共享算子**
  （`ds41_matmul_batch`、`ds41_route_batch`/router-select、batch attention 等），这些 P0b 一概不碰。
- **对 P0b 的意义**：P0b 的 A/B 是同 batch 管线的两臂对拍，上游共享算子在两臂中逐位相同，
  所以 §12.2 的 verify 漂移**不构成** P0b 的阻塞；P0b 只需过自己的 frontier-logits 逐位门。
- **诚实边界**：本判别覆盖 n_tokens≤8（两臂同走 mul_mv tiny-pair 家族）。sweep 大 n_tokens 下 gather 臂
  走哪条内核家族、与整层 `mul_mm_id` tile 家族是否同归约序，是**另一个**问题——那正是 P0b 自己的
  `compare_frontier_logits.py` 门要回答的，不能拿本结果顶替。overflow→映射视图分支未直接执行
  （cache 预算充足），依据是 §12.5 引用的内核侧注释：两分支读同一份专家字节、地址表由同一 host 路径拼装。
- **踩坑记录（写测试的人看）**：症状"selected id 读到浮点位型"看着像生产内核覆写输入，
  实为测试自身把 `int32_t *host_ids` 混进了 `float *` 声明列表——`host_ids[p]=id` 写进去的是 `float(id)`，
  外加一处 `malloc(n)` 漏 `*sizeof(float)` 的堆溢出。两次误判都因为先怀疑引擎不先怀疑自己；
  echo 回读法（写后立即读回比对）三轮定位。`--moe-bind-parity` 的最终版已不含这些脚手架。
- 闸门复验：`make` 全绿、`./tests/test_deepseek41_prefill --dispatch` PASS、`--moe-bind-parity` EXIT=0。
  日志：`/tmp/p0a/bind_parity_final.log`（干净 9/9）、`bind_parity{,2,3,4}.log`（排查现场）。

---

## 附：上一轮已确认、可直接引用的事实（勿重测）

| 事实 | 值 | 条件 |
|---|---|---|
| sweep 的逐层 pread | 3.73 GiB/层 × 40 = 151 GiB，13.8 s | V4.1 Q2，tokens=2048 |
| 本机 SSD 顺序读 | 12.3 GiB/s | 4 GiB 中段顺序读 |
| per-expert 字节 | 9.49 MiB | V4.1 Q2 |
| 全专家工作集 | 15360 experts / 142.4 GiB | 40 层 × 384 |
| `minimum` 阈值 | 7680 experts = `DS4_N_LAYER*DS4_N_EXPERT/2` | `ds4.c:42109` |
| auto cache @ ctx 8193 | 72.51–75.50 GiB / 7822–8145 experts | 默认 |
| auto cache @ ctx 393216 | 64.51 GiB / 6959 experts | 用户配置 |
| cache 16GB vs 75GiB（append 2048） | 130.02 vs 114.13 t/s | 同 build 同机 |
| 1023 vs 1024 断崖 | 23.9 vs 63.1 t/s | 同 build 同机 |
| P1 row batch（append 128） | 30.2 vs 21.8 t/s，且 DRIFT | 已回滚 |

---

## 13. P0b 执行记录（2026-10-02）：速度达标，逐位 DRIFT，opt-in 保留

### 13.1 实现（已在工作区，未提交；默认行为逐位不变）

- `ds4.c`：新增 `ds41_streaming_sweep_moe_gather_admitted(g, w, rows)`（在 `ds41_graph_prefill_sweep` 之前）。
  单一谓词同时决定 ①层 map（`map_layer_decode` + pagein `decode_only=true`，join 与
  **前瞻启动**两处都要传）②`ds41_moe_batch` gather 参数。非 wide、非 encoder_only、
  rows=min(total,encoder_chunk)≤上限、cache≥384、recipe=IQ2_XXS/Q2_K、Apple、tp=1、非 quality/imatrix。
  **运行时开关：默认关；`DS4_METAL_ENABLE_V41_STREAMING_SWEEP_GATHER=1` 开，
  `DS4_METAL_DISABLE_V41_STREAMING_SWEEP_GATHER=1` 否决一切** → A/B 永远同 build。
- `ds4.c` sweep 内 `gather_sweep` 常量：`!wide && !encoder_only && !resume_encoder && 谓词(min(total,chunk))`；
  三处使用：join prepare、前瞻 `prepare_start_if_needed(il+1…)`、map 三元、`ds41_moe_batch` 末参。
- `ds4_metal.m`：`ds4_gpu_stream_prefill_batch_selected_addr_auto_max` 384-expert 档 800→1024
  （覆盖 931/1023 目标行数；只影响 force_resident=false 的调用方，即 V41 gather 族）。
  ds4.c 谓词默认 1024 与它镜像；`DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX` 两边共读。

### 13.2 速度门（同 build、同 cache 配置 6959 experts、双向 A/B、`--ctx-alloc 393216`）

append 831（story fixture，冷启动 worst case）：

| 方向 | 整层臂 t/s（墙钟） | gather 臂 t/s（墙钟） |
|---|---|---|
| 反向 B→A | 49.59/47.47/46.95（16.8–17.7 s） | 86.22/95.03/91.74（8.7–9.6 s） |
| 正向 A→B | 47.68（17.4 s） | 82.30（10.1 s） |
| 终验 | 49.52 | 85.97 |

I/O（tokens=831 那一次 sweep，profile 求和）：整层 146.33 GiB → **7.49 GiB（decode span）+ 34.93 GiB（gather miss）= 42.4 GiB**。
注意 miss 是冷 cache 数字；续写场景 cache 已热，只会更少。2048 初始 prefill（>1024 上限）两臂都走整层，路径不变。

### 13.3 逐位门：DRIFT，按 §7.2 否决

frontiers 2048/2879/3710/4541，`compare_frontier_logits.py --ctx 393216`（compare 用 alloc ctx，不是 frontier+1——§7.2 命令模板在这里会 INVALID，dump 元数据记的是 alloc）：
`2048 strict=True`（锚点），三个 gather append 全部 `strict=False`：n_diff=全词表，max|Δ|=3.145/1.809/1.469，**argmax 三处全同**，但 top5 内出现次序互换。

**根因（stage profile 实锤，非 bug）**：831 行时整层臂 `path=mm_id, mid=f16`，gather 臂
`path=iq2_batch_stream_addr, mid=f32`。§12.7 微测只覆盖 rows≤8——那里两臂选**同一内核族**（tiny pair/mv），
所以逐位同；sweep 行数（≥256）整层走 mm_id tile 族，gather 走 mul_mv_addr pair 族，**族不同 + mid 精度不同**，
逐位无解。cache/view 数据本身无罪（无 overflow-view 报错，全部 cache 供给）。

### 13.4 若要继续：P0c = 内核移植（不是调度活）

> 本节是 §11 **分叉 A**（用户维持逐位契约）的执行细节；若用户选分叉 B（放宽契约），
> 直接用 §13.1 的 opt-in 开关即可，无需本节工作。

逐位方案的唯一路径：给整层 sweep 实际选中的 mm-id 管线做一个**源指针间接寻址（addr 表）实例化**，
tile 迭代/归约/mid 精度与现管线逐字节一致。现有坐标：

- `metal/moe.metal:8239` `kernel_mul_mm_id_addr`（模板源指针版），host 名 **仅 q2_K/q4_K/mxfp4**（:8931–8936）——**缺 IQ2_XXS**；
  整层 V41 用的族是 `kernel_mul_mm_id_iq2_xxs_*`（:8915/:8923）、`_cached_` 变体（:8924，模板尾参 `false,true`）、
  `mpp_packed`（:9190/:9403，M5 上 mm_id 实际可能选它——先 profile 确认 831 行整层臂的真实 pipeline 再动手）。
- GLM 已有"cache 资源 + overflow view 喂 mm tile"的完整驱动可抄：
  `ds4_gpu_glm_routed_moe_batch_grouped_addr_tensor` + `ds4_gpu_encode_mul_mm_id_addr_mapped_tile`（调用点 `ds4_metal.m:40678`）。
- 先行门：把"同 kernel 名、addr vs whole-map"加进 `--moe-bind-parity` 微测（rows 扫到 1024），先证逐位再谈速度。

### 13.5 本轮踩到的新坑

- **前瞻 prepare 抢跑**：sweep 每层开头会 `prepare_start_if_needed(il+1,…,decode_only=false,…)`（`ds4.c` Apple 分支）
  预启动**整层** pagein 塞进 slot；只改 join 处不改前瞻，layer 1–39 照读整层（症状：layer0=0.17 GiB、layer1+=4 GiB）。
  两处必须传同一个 gather 谓词。
- `compare_frontier_logits.py --ctx` 要传 **alloc ctx**（dump 元数据记 alloc），§7.2 模板里 `<ctx+1>` 只适用于不显式 alloc 的跑法。
- `| grep -m1` 会 SIGPIPE 掐死 ds4-bench 导致 CSV 不落盘——重定向后再 grep。

### 13.6 终验状态（同一最终 build）

- `make` 无 warning；`--dispatch` PASS；`--moe-bind-parity` 9/9 bit-exact；
- 默认（不设 env）831 append = 49.52 t/s ≈ 开工前基线 46–50 t/s，**行为逐位不变**；
- `DS4_METAL_ENABLE_V41_STREAMING_SWEEP_GATHER=1` = 85.97 t/s（§7.2 门在此开关下才会 DRIFT，勿在生产开）。
- 测量产物：`/tmp/p0b/`（bench_A/B*.csv+log、A/B logits dump、cmp.json=drift、prof_*.log=路径证据、microtest_final.log）。

---

## 14. P0c 执行记录（2026-10-02）：内核移植完成，逐位门全绿，gather 翻为默认

> §11 分叉 A 的闭环。两条腿：addr 表镜像内核 + 主机接线；全部门闸通过后按 §13 计划
> 把 sweep gather 从 opt-in 翻为默认（TensorOps parity 域内）。**§0 目标达成，逐位契约保持。**

### 14.1 实现坐标

- `metal/moe.metal`：`kernel_mul_mm_id_mpp_packed` 加 `EXPERT_ADDRESSES` 模板参（`wsrc`/`wbase`
  二段基址，同模板同 tile 同累加）；新实例化
  `kernel_mul_mm_id_iq2_xxs_mpp_packed_cached` / `kernel_mul_mm_id_q2_K_mpp_packed_cached`（≥512 行 packed 镜像）、
  `kernel_mul_mm_id_q2_K_cached_f16_mpp`（[32,512) down 镜像；gate/up 的 `iq2_xxs_cached_f32_mpp` 已存在）。
  均在同一 `#ifdef DS4_METAL_HAS_TENSOR` 域。
- `ds4_metal.m`：新谓词 `use_v41_stream_gather_addr`（mask==7 + IQ2_XXS/Q2_K + rows∈[32,8192) +
  selected-addr 门 + 4 个镜像 pipeline 非 nil + kill `DS4_METAL_DISABLE_V41_STREAM_GATHER_MM_ID`）；
  激活时压制 `use_iq2_batch_selected_addr`（mul_mv 漂移族），rows≥32 落 `use_mm_id` 家族；
  `use_packed_mpp` 放行；pipeline 按整层臂逐档选 `_cached_` 镜像名；encode src0 经
  `use_mm_id_stream_addr` 绑 addr 表 + cache resources + overflow view（GLM 驱动复用，零新调度）。
  导出 `ds4_gpu_v41_stream_gather_parity_supported()`（声明 `ds4_gpu.h`），供 ds4.c admission 硬门。
- `ds4.c`：`ds41_streaming_sweep_moe_gather_admitted` rows≥32 + parity 查询硬门（pre-M5 mask≠7 →
  admission 假 → 默认整层，行为逐位不变）+ **翻为默认 opt-out**（kill：
  `DS4_METAL_DISABLE_V41_STREAMING_SWEEP_GATHER`）。`DS4_METAL_ENABLE_...SWEEP_GATHER` 现为 no-op。
  rows [2,32) 默认仍整层（<32 整层臂走 mul_mv 族，addr 表不镜像该族 sweep 尺寸；≤8 的 addr pair
  逐位事实见 §12.7，未接入 sweep）。

### 14.2 门闸（全绿，产物 `/tmp/p0c/`）

1. **§13.4 先行门** `--moe-bind-parity`：layers {0,20,39} × rows {2,5,8,32,256,511,512,831,1024}
   = 27/27 bit-exact；ids 改 token 内去重（跨 token 重复，贴近真实路由）。`GATE1_EXIT=0`。
2. **§7.2 逐位对拍 831 append**（frontiers 2048/2879/3710/4541，`--ctx 393216`，gather ON vs 整层 OFF，
   同 build）：**IDENTICAL**，strict=True，4/4（`cmp_831.json`）。
3. **§7.2 931 append**（2048/2979/3910/4841）：**IDENTICAL**，strict=True，4/4（`cmp_931.json`）。
4. **内核族实锤**（stage profile layer20）：gather 臂 tokens=831 **`path=v41_gather_packed_mpp mid=f16`**
   （P0b 根因是 `iq2_batch_stream_addr mid=f32`）；2048 锚点两臂同 `path=mm_id`（>1024 不 gather，路径不变）。
5. **翻默认复验**（最终 build）：新默认 dump（`E831`）vs 翻默认前 build 的旧默认 dump（`A831`）
   4-frontier **IDENTICAL** strict=True（`cmp_default.json`）；kill switch 下 append 臂回 `path=mm_id`（`prof_kill.log`）。

### 14.3 速度（TTFT = append rows ÷ prefill_tps）

| 场景 | 旧整层默认 | 新 gather 默认 |
|---|---|---|
| 831 首 append（每 run 冷） | 52.0 t/s → **16.0 s** | 135.7 t/s → **6.1 s** |
| 831 热 append（后续层 cache 已热） | 46.7–47.6 → 17.4–17.8 s | 164.6–168.3 → **4.9–5.0 s** |
| 931 首 append | 55.9 → 16.7 s | 141.0 → **6.6 s** |
| 2048 初始 prefill（两臂同路） | 131–136 | 128–135（噪声域，回归干净） |

§0 目标（≤~8 s）达成；比 P0b 的 gather（85.97 t/s，9.6 s）再快 ~1.6×——计算侧同步从
mul_mv pair 升到 packed TensorOps 镜像，不止省 I/O。

### 14.4 遗留

- 未提交改动：`metal/moe.metal`、`ds4_metal.m`、`ds4.c`、`ds4_gpu.h`、`tests/test_deepseek41_dspark.c`。
- rows (8,32) 区间不进 gather（整层臂在该区间是 tiny pair/mv 混合，addr 镜像未覆盖）——留作后续，非 §0 目标。
- CUDA / quality / TP 路径不受影响（谓词全部要求 `tp_world==1 && !quality`，内核在 TensorOps 域）。

### 14.5 P0d：`minimum=256` 地板在 gather 域内降到 32（已接受，默认生效）

生产日志暴露：真实多轮对话的 ~223 行续写走 17 t/s 单 token 步进，而 360 行走 54.7 t/s sweep。
根因与 P0c 无关：`ds41_prefill_count`（`ds4.c`，P0d 前 `ds4.c:42112`）有既有地板
`remaining < minimum(256) → return 1`（单 token 步进），地板年代 = sweep 读整层的时代。
断崖实测（同 ctx 2048 锚点后）：**223 行 = 21.4 t/s，256 行 = 89.4 t/s**。

- 改动：`ds41_prefill_count` 加 `w` 参数；在 gather parity 域内
  （`ds41_streaming_sweep_moe_gather_admitted(g,w,remaining)` 真）把 minimum 压到 32，
  与内核镜像的自有地板对齐；rows 2..31 维持单 token（addr 表不镜像该族的批量路）。
  受 `DS4_METAL_DISABLE_V41_STREAMING_SWEEP_GATHER` 统一 kill。
- 速度：223 行 append 21.4 → **81.6 t/s**（`/tmp/p0c/p0d_223.csv`）。
- **逐位对拍（跨路）不过——定性为既存双路分歧，非任何路径的数值改动**：
  新默认（223 行走 sweep 批量）vs 旧默认（223 行走单 token 步进），2048 锚点
  `strict=True changed_count=0`，2271 全词表 drift（max|Δ|=1.77，argmax 相同）。
  两条路各自的内核/数值一行未动；分歧本体 = scalar 家族（tiny pair/mv，f32 mid）与
  批量家族（mm-id tile，f16 mid）从来不等价。该分歧 P0d 之前就在生产里：
  同一续写按不同块尺寸送达（如 583=360+223）时，各块本来就分属两个数值家族。
  rows [256,1024] 无"sweep 223 行"基线可破坏——契约基线在该行数上不存在。
- **契约裁定（用户，2026-10-02）**：接受 [32,256) 行数的 append 收敛到批量家族。
  登记偏差：① 该区间轮次的 greedy logits 与 scalar 时代不同（抽查点 argmax 同，不保证恒同）；
  ② scalar 时代存盘的 KV checkpoint resume 后为两族数值拼接（版本升级同性质）。
- 产物：`/tmp/p0c/{F223,G223b}/`、`cmp_223b.json`、`p0d_223*.csv`。
- 部署：server 进程需重启才吃到 P0d（build 时间晚于进程启动即是旧进程）。

---

## 15. P0e 路线设计（未开工）：行数分层与跨路契约

### 15.1 问题精确化

V4.1 Metal 流式下 MoE 存在**两个数值家族**，按 `n_tokens` 分层（本机 mask==7 域）：

| n_tokens | 内核族（`ds4_gpu_routed_moe_batch_tensor`） | mid 精度 | 数值系 |
|---|---|---|---|
| 1..8（decode/verify/标量步进） | tiny pair fused（`v41_decode_batch`，`ds4_metal.m` ~44931） | f32 | scalar |
| 9..31（真空档：无批量 gather，整层 mv 族） | 通用 mul_mv | f32 | scalar?（未测 fused vs 非 fused） |
| ≥32 | mm-id tile：[32,512) plain mpp / ≥512 packed（P0c 后 addr 镜像逐位同整层） | f16 | batch |

`v41_decode_batch ≤8` 是引擎作者登记的**有意标约边界**（注释原话：V4.1 需要 scalar
reduction order "through eight rows"）。两族从来不等价（f32 vs f16 mid + 归约序），
所以"同一 token 序列、不同送达切块 ⇒ 同一 logits"这条契约在 ≥9 行的批量 vs 步进上
**从来不存在**，非流式也一样（`check_short_prefill` 只测 ≤8 所以一直绿）。
P0c/P0d 未引入也未扩大这个矛盾；[32,1024] 的批量段内部经 P0c 门已证互洽。
rows 9..31 是真空档：批量走整层 mv（scalar 系?）vs 步进 tiny pair fused——两值是否相同**无人测过**。

### 15.2 E0 —— 三个先行微测（决定走哪条，全在现有 `--moe-bind-parity` 框架上扩）

1. **E0.1 mv 族行数独立性**：整层绑、同 `*_f16/f32` mv 管线，臂 A=`n_tokens=N` 批量
   vs 臂 B=`n_tokens=1` 连跑 N 次（N∈{2,5,9,16,31}），比对每行输出切片。
   过 ⇒ rows [2,32) 存在"保 scalar 数的批量路"的数学基础 → E1 可开工；不过 ⇒ E1 死，只剩 E2。
2. **E0.2 mm-id 行数独立性**：整层绑 mm-id 族，臂 A=`n_tokens=N`（N∈{32,256,831}）vs
   臂 B=`n_tokens=1`×N。过 ⇒ decode/verify 理论上可统一走 mm-id → E2 有基础；不过 ⇒ E2 死。
3. **E0.3 streaming 版 `check_short_prefill`（§4.4 的遗留 TODO，P0e 的终极闸门）**：
   `tests/test_deepseek41_dspark.c:144` 的 options 加 `.ssd_streaming=true` +
   streaming cache 配置（仿 `check_moe_bind_parity` 的引擎开启），行集 {1,8,32,223,831}，
   批量 append vs 逐 token 步进 `REQUIRE(worst==0)`。**今天预期 rows≥32 判 DRIFT——把它
   先落成登记红灯**（复现既存违约），E1/E2 收敛后转断言。

### 15.3 E1 —— 低行数批量省（保 scalar 数值）：首选，成本≈P0c 的十分之一

前置：E0.1 过。目标：rows [2,32) 也吃 gather 批量，且数值 = 今天的标量步进（契约零变化）。

- 内核：**不需要新移植**。P0b 家族（mul_mv addr pair，`use_iq2_batch_selected_addr`）就是
  mv 系的 addr 镜像，§12.7 已证 ≤8 逐位同；E0.1 把证明扩到 9..31。
- 改动：`ds4.c` P0d 的 minimum 覆盖从 32 放到 2（admission 同步：`rows<32` 时
  metal 侧 `use_v41_stream_gather_addr` 自然假、落 `use_iq2_batch_selected_addr`——谓词已自洽，
  关键是 admission 的 decode-span map 前提在 rows<32 同样成立）；
  `--moe-bind-parity` row_sets 补 {9,16,31}（现有 addr vs whole-map 双臂即覆盖 E0.1 的绑定维度）。
- 门：① E0.3 断言下 rows {2,9,31} 批量==步进 IDENTICAL（数值不变，只提速）；
  ② rows [256,1024] 门（§14.2 全套）零回归；③ 16/31 行 append 提速实测。
- 风险：`ds4_gpu_stream_prefill_batch_selected_addr_enabled` 的 [min,max] 门与 P0d 域交界
  （32 两侧内核族不同但各自逐位），E0.3 的 {31,32} 相邻行集对拍专测该缝。

### 15.4 E2 —— 单一数值家族（根治，重决策，可选）

前置：E0.2 过 + decode 吞吐损失实测可接受。方向：`v41_decode_batch` 的行数带从 ≤8 扩到
全带（1 行也走 mm-id：单行 tile 权重带宽同量级，代价是 tile 尾浪费 + map0/sum 固定开销——
**先 profile 1-row mm-id decode 的 ms/step 回退再谈**）。收益：跨路契约真正成立
（E0.3 全绿）、tiny 族 addr 镜像免维护。代价：**decode/verify/步进全线 logits 变更**
= 一次数值版本升级（向批量家族全域收敛，比 P0d 面大得多），需要 golden vectors 全量
重生成 + 用户单独拍板。除非 E0.3 的登记红灯成为实际痛点（server checkpoint 混用、
test-vectors 锚定失败等），不主动推。

### 15.5 决策树 / 非目标

- E0.1 过 → 做 E1（半天级，含门）；E0.1 不过且 E0.2 过 → 低行数提速只剩 E2（重）或放弃；
  都不过 → 分层维持现状，E0.3 红灯作为登记偏差长期保留。
- 非目标：批量家族数值变更（禁）；decode 步进吞吐回退；CUDA 侧同构（本机域外）；
  quality/imatrix/TP 路径。
