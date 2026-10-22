# HANDOFF：V4.1 DSpark 速度与长上下文分叉（2026-10-19）

> 交接给新 session。先读 §6 的路线判断题，再决定是否继续 §5 的排查。
> 本文只记录事实与证据；未做的门禁在 §9 列明。

## 0. 一句话结论

上游 DSpark 已在 `rebase/20261019` 上跑通，**短上下文（≤2K）与普通 decode 逐位相同**，但：

1. **速度**：`--dspark` 比基线慢 **7.4–7.9%**（2K 与 8K 一致）。
2. **正确性**：**≥4K 上下文贪心输出与基线分叉**，首个差异在单个近 tie 的词上，确定性复现。

速度慢的机理已查清（草稿器 254 个周期只起草 2 次，那 2 次各花 ~1.2 s）；分叉的机理**未定位**，已排除一批嫌疑，剩余假设见 §5。

---

## 1. 仓库状态

| 项 | 值 |
|---|---|
| 分支 | `rebase/20261019` |
| HEAD | `c1b4aaa tests: retire the fork drafter's verify-cost harness` |
| 与上游 | 领先 `origin/main`（`5379c1c`）**140 笔** |
| 备份 | tag + branch `pre-rebase-20261019` → 旧 `main` `ae45e32`；`main` **未动** |
| 工作区 | 只剩用户自己的 6 个 docs 改动 + 若干未跟踪文件（未提交） |

关键提交：

```
c1b4aaa tests: retire the fork drafter's verify-cost harness
df0334e v41: retire the fork drafter and run upstream DSpark
eb65075 rebase: reconcile V4.1 DSpark, engram overlap and Metal verify-row conflicts
（以下 137 笔为本地历史重放；e5b19e28 已作为 0b793da 吸收，被跳过）
```

模型资产：

| 文件 | 说明 |
|---|---|
| `gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf` | 主模型 365 GB；元数据确认是上游新默认配方（`IQ2_XXS gate/up; Q2_K down; Q8_0 attn/shared/head`，imatrix `v41-1017-262144.dat`，revision `1bae0e10…`） |
| `gguf/DeepSeek-V4.1-Flash-DSpark.gguf` | **本次新转**，8.04 GiB，81 张量，`--verify` PASS；`architecture=deepseek41-dspark`、`target_architecture=deepseek41`、layers `[37,38,39]`、block 5、128 experts/3 per tok |
| `~/models/DeepSeek-V4.1-Flash-1017/` | 原生分片：`config.json`、`model.safetensors.index.json`、`model-00044/45/46-of-00048`（7.93 GB，含全部 2401 个 `mtp.*`） |

---

## 2. 问题陈述

**P1（速度）**：`--dspark` 在 M5 Max / 128 GiB / SSD streaming / cache 1024 上比不带的基线慢 7.4–7.9%。

**P2（正确性）**：`--temp 0` 贪心，8K 上下文下 `--dspark` 的输出与基线在字符 289 分叉：

```
A（基线）: …summarize, analyze, continue? …
B（dspark）: …summarize, analyze, translate, continue? …
```

两轮 B 结果相同（确定性）；`--dspark-strict`（drafter 常驻但不投机）与 A 逐字节相同。

---

## 3. 已完成的实验与结果

### 3.1 速度矩阵（A/B/C，两轮，A→C→B→B→C→A）

`--temp 0 -n 256`，prompt 为 `speed-bench/promessi_sposi.txt` 截断。

| 上下文 | A 基线 | C strict | B dspark | B vs A | C vs A |
|---|---|---|---|---|---|
| 2K（`-c 4096`） | 13.47 t/s | 13.24 | **12.40** | **−7.9%** | −1.7% |
| 8K（`-c 16384`） | 12.48 t/s | 12.41 | **11.55** | **−7.4%** | −0.6% |

### 3.2 DSpark 统计（`DS4_DSPARK_STATS=1`）

| 轮次 | cycles | proposed | accepted | no_draft | net_saved | propose ms | verify ms |
|---|---|---|---|---|---|---|---|
| 2k-r1 | 254 | 2 | 2 | 253 | **−3834** | 1617 | 599 |
| 2k-r2 | 254 | 2 | 2 | 253 | **−727** | 123 | 481 |
| 8k-r1 | 254 | 2 | 2 | 253 | **−4061** | 1693 | 675 |
| 8k-r2 | 254 | 2 | 2 | 253 | **−4008** | 1680 | 648 |

读法：**254 个周期只有 2 个真起草**；那 2 次共 ~2.4 s，换 2 个 token，等于普通 decode 步（~80 ms）的 15 倍。整轮生成 ~21 s，这 2.4 s 就是那 7–8%。

`prop_stage0…prop_confidence`、`verify_upload/layer/head/read` 全为 `0.000` —— 这些细分字段原本由已退役的本地草稿器填写，现在是死字段（见 §9）。

### 3.3 输出等价性

| 上下文 | A | C | B |
|---|---|---|---|
| 2K | `2deb7ada…` | 同 A ✓ | 同 A ✓ |
| 8K | `9644e880…` | 同 A ✓ | **`a320267a…` ✗** |
| 4K | `149fdfa4…` | — | **`db7a3951…` ✗** |

**阈值在 2K 与 4K 之间。**

### 3.4 进程内对照（上游自带）

```
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark.gguf \
DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 \
./ds4_test --dspark-verify-depth
```

结果：`nspec=128 max_chunk=4 draft_depth=5 worst_argmax_gap=0.000 at=-1`

即**短上下文（ctx 250/378）下 verify 与 decode 逐位相同**。注意该测试同时报了一个失败：
`tests/ds4_test.c:7306: assertion failed: ds4_test_dspark_prefix_capture(engine, &prompt)`（第一套 ctx=250 配置），
第二套 ctx=378 通过并给出上面的 gap。**这个失败本身也需要解释**（可能是 ctx 太小放不下 prefix capture）。

### 3.5 二分实验（全部无效）

单腿（B 加开关 vs 无开关的 A）与配对（同开关下 A vs B）两种都做过。

| 开关 | 结果 |
|---|---|
| `DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP` | 仍分叉 |
| `DS4_DISABLE_V41_ENGRAM_STEP_READERS`（配对） | 仍分叉 |
| `DS4_METAL_DISABLE_V41_SHARED_FUSE` | 仍分叉 |
| `DS4_METAL_DISABLE_V41_MOE_FUSE` | 仍分叉 |
| `DS4_METAL_DISABLE_V41_ROUTER_FUSE` | 仍分叉 |
| `DS4_METAL_DISABLE_V41_HC_FUSE` | 仍分叉 |
| `DS4_METAL_DISABLE_V41_STREAM_GATHER_MM_ID` | 仍分叉 |
| `DS4_METAL_DISABLE_V41_BATCH_INDEX`（配对） | 仍分叉（且该开关对输出**零影响**） |
| `DS4_METAL_DISABLE_V41_BATCH_TOPK`（配对） | 同上 |
| `DS4_METAL_DISABLE_V41_PACKED_INDEX`（配对） | 同上 |
| `DS4_METAL_DISABLE_V41_BATCH_COMPRESS`（配对） | 同上 |
| `DS4_METAL_DISABLE_V41_BATCH_HEAD`（配对） | 同上 |
| `DS4_METAL_DISABLE_V41_BATCH_ATTN`（配对） | 仍分叉（且改变 prefill 输出） |
| `DS4_METAL_DISABLE_V41_BATCH_CORE`（配对） | 仍分叉（且改变 prefill 输出） |
| `DS4_DISABLE_V41_BATCH_OUTPUT=1` | 仍分叉（**排除批量输出投影**） |
| `DS4_METAL_DISABLE_V41_BATCH_MOE=1` | 直接失败（rc=1） |
| `DS4_METAL_DISABLE_V41_STREAMING_VERIFY_GATHER=1` | **直接失败**：`Metal model range … is not covered by mapped model views` |

### 3.6 缓存假设（已排除）

8K + `--ssd-streaming-cache-experts 4096`（4 倍缓存）与 1024 的 A/B 输出 hash **完全相同**（`e1dda6a9f6cc` / `4d34ee8182cf`）。

→ 不是流式专家缓存驱逐/装载的 bug，输出对缓存大小不敏感（这本身是好事）。

---

## 4. 代码层面的关键事实

1. **`ds41_graph_step_batch` 用 `g->verify_prefix` 区分 verify 行**（上游设计）。本地原实现用 `spec` 参数；退役时已翻译成 `verify_prefix`，并修了两个真 bug：
   - `prefill_only` 必须 `&& !g->verify_prefix`（否则 verify 只推进最后一行）；
   - 收尾两个 per-row 循环**必须无条件执行**（它们写 `history` 与 `pos`），原先本地用 `!spec` 表达"非草稿验证"，直译成 `!verify_prefix` 会漏掉上游的 verify 行。
2. **`gather_moe` 是本地独有的**（上游没有）：`g->verify_prefix && g->streaming && ds41_streaming_verify_moe_gather_admitted(...)`。流式 verify **必须**走它，关掉即失败。
3. **`ds41_hc_fused` 只要求单卡 + kernel 可用，没有 streaming 门** → 普通 decode 走融合 HC；而批处理（verify 行）走的是未融合的 `ds4_gpu_hc_expand_split_tensor` 序列。关掉 HC 融合后仍分叉，所以不是它（但"decode 融合 / batch 不融合"这个结构差异仍然值得记住）。
4. **`ds4_gpu_dsv41_exact_admitted` 在 streaming 下恒 false**（`g_ssd_streaming_mode` 直接短路），所以本地那批"精确"融合在本机 streaming 上**全部不生效**；能生效的是新增的 `ds4_gpu_dsv41_activation_exact_admitted`（纯激活域，streaming 可用，M5 类设备）。
5. 模型元数据：`sliding_window=128`、`index_topk=512`、`candidate_topk_blocks=2048`（块 8）、`compress_ratios=[0,0,2×18,1×20]`、`kv_source_layer_ids=[2,8,14,20]`。

---

## 5. 剩余假设（按可能性排序）

| # | 假设 | 为什么可疑 | 怎么验 |
|---|---|---|---|
| H1 | **128 槽 KV 环在 verify 行的环绕/保存恢复** | `sliding_window=128`，verify 批的 3 行占连续槽位；跨 128 边界时 `slot=(pos+i)%128` 可能与 seed 行冲突，或 commit 的 old_raw 恢复错位。上下文越长越容易踩到 → 与 2K/4K 阈值吻合 | 进程内 harness：在 `pos%128` 分别取 0/1/126/127 的位置发起 count=3 的 verify，比 `verified[i].logits` 与单 token decode |
| H2 | **压缩层索引（ratio 2，top-512）在 verify 行的多行选择** | 2K 时压缩条目 1024、4K 时 2048，都 >512，理论上 2K 就该走 top-k；但 2K 不分叉，说明可能是"候选块全保留"路径与"真 top-k"路径的切换 | 用 `DS4_DSPARK_VERIFY_TOPS_CHECK` / 索引相关 profile 环境变量；或 dump 每层 `index_scores` 对比 |
| H3 | **多行 MoE gather 的归约顺序** | 无法单独关（关掉即失败），是唯一没法用开关排除的大段 | dump `routed/shared/block` 中间张量，逐行比 decode |
| H4 | **上游自身在长上下文不保证逐位** | 上游文档只说 "one-token arithmetic"，没有承诺长上下文逐位；可能只是近 tie 翻转 | 直接问/查 upstream issue；或把同一实验跑在纯 `origin/main`（注意纯上游在 streaming + DSpark 下可能根本跑不起来，因为缺本地 gather） |

**建议先验 H1**：它最符合"2K 好、4K 坏、缓存无关、开关切不开"的全部特征。

---

## 6. 新 session 要先确认的路线判断题

1. **验收标准**：DSpark 是否接受"短上下文逐位、长上下文近似"？如果不接受（本仓库的纪律是 bit-exact 或解释清楚漂移），那么 P2 不解决就不能启用。
2. **是否继续查**：收益已经实测为负（−7.4~−7.9%），查 P2 只对"将来换工作负载/换机器"有价值。**建议先做 H1 的一次性 harness，无论结果都收口**。
3. **engram 二选一**：上游 `ds4_engram_reader` 仍编译未接线，本地 `pipeline_layers` 在用。是否要做 A/B（见 §8 的 engram 段）。
4. **本地 #1090 剩余部分**：融合（HC/MoE/shared/router）+ 流式 gather + engram 重叠都还在用。是否继续保留？（注意：`exact` 族在 streaming 下根本不生效，只有 `activation` 族与 gather 有效。）
5. **分支去向**：`rebase/20261019` 是否合回 `main`、是否推 `fork/main`。

---

## 7. 下一步具体动作

### 7.1 先做的（H1）

写一个进程内 harness（建议 `tests/test_dspark_verify_parity.c`，`#include "../ds4.c"`，参照 `tests/test_deepseek41_dspark.c` 的骨架）：

```
engine_open(model, mtp=dspark support, ssd_streaming, cache=1024, ctx=16384)
s1 = session_create; s1 同步 8K prompt;  t0 = argmax(s1->logits)
s1: eval(t0)                      → L1 = copy(s1->logits)        # 单 token decode
s2 = session_create; s2 同步同一 prompt
s2: ds4_test_ds41_verify_tokens(s2, {t0, t1}, 2, L2)            # verify 两行
比较 L1 vs L2[0]：max |Δ|、argmax、top-2 gap
```

- `t1` 取 `argmax(L1)`。
- 先跑 `pos%128 ∈ {0,1,126,127}` 四个相位；若某一相位 Δ≠0 → H1 成立。
- 需要 `-DDS4_TEST_HOOKS`（钩子在 `#ifdef DS4_TEST_HOOKS` 内）。

### 7.2 已有的可复用命令

```sh
# 速度三腿（每轮 ~31 s @2K / ~75 s @8K）
DS4_DSPARK_STATS=1 ./ds4 -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 1024 -c 16384 -n 96 --temp 0 \
  --prompt-file /tmp/dspark-ab/p8k.txt [--dspark --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark.gguf]

# 逐周期日志
DS4_DSPARK_SPEC_LOG=1 ...

# 进程内等价（短上下文）
DS4_TEST_MODEL=... DS4_TEST_DSPARK=... DS4_TEST_SSD_STREAMING=1 \
DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 ./ds4_test --dspark-verify-depth
```

### 7.3 实验脚本位置

`/tmp/dspark-ab/`：`run_ab.sh`（速度矩阵）、`bisect*.sh`（各轮二分）、`summarize.py`（解析 stats）、
`p2k/p4k/p8k.txt`（截断后的 prompt）、各轮 `*.out/*.err`。
**`/tmp` 会被清空，若要长期保留请先拷进仓库**（例如 `speed-bench/dspark-20261019/`）。

---

## 8. 本地 DSV4.1 提速开发：环境变量与关键参数清单

> 这是本次交接的重点背景。绝大多数开关是 **kill switch**（默认开，设了才关）。

### 8.1 准入机制（决定其余开关是否生效）

| 机制 | 条件 | 对本机（streaming） |
|---|---|---|
| `ds4_gpu_dsv41_exact_admitted(test_flag, rollback_env)` | `!quality && !ssd_streaming && tp_world==1`，且（test flag 或（`g_v41_measured_config` 且 **M3 Ultra**）），且 rollback env 未设 | **恒 false** → 所有"精确"融合在本机不生效 |
| `ds4_gpu_dsv41_activation_exact_admitted(...)` | 纯激活域；streaming 也可用；M5 类设备 | **可用**，这是本地在 streaming 上唯一有效的精确加速入口 |
| `g_v41_measured_config` | `tp_world==1 && !streaming && !imatrix && !image && !quality` 且 40 层路由专家类型一致（Q4_K 或 MXFP4） | false（streaming） |

### 8.2 融合族（kill switch，默认开）

| 环境变量 | 关掉什么 |
|---|---|
| `DS4_METAL_DISABLE_V41_HC_FUSE` | 每层 19→6 dispatch 的 hyper-connection 融合（`ds41_hc_mix_fused` / `ds41_hc_collapse_norm_fused` / `ds4_gpu_dsv41_hc_expand4`） |
| `DS4_METAL_DISABLE_V41_MOE_FUSE` | FFN tail 融合（shared down 折进 tail、`ds41_moe_finish`） |
| `DS4_METAL_DISABLE_V41_ROUTER_FUSE` | router matvec + select 合成一个 dispatch |
| `DS4_METAL_DISABLE_V41_SHARED_FUSE` | shared gate/up/SwiGLU 合成一个 dispatch |
| `DS4_METAL_V41_SHARED_DOWN_EARLY` | **正开**诊断：shared down 提前到 routed 之前单独跑 |
| `DS4_METAL_DISABLE_V41_SHARED_FUSE`（同 8.2 第 4 行） | 同上 |
| `DS4_METAL_DISABLE_BATCH_HC_NORM_FUSION` | 批处理里的 HC norm 融合 |
| `DS4_METAL_DISABLE_HC_EXPAND_PRODUCER_FUSE` | expand 的 producer 融合 |
| `DS4_METAL_DISABLE_M5_ROUTER_PROJECT_SELECT_FUSE` | M5 上 router project+select 融合 |
| `DS4_METAL_DISABLE_PRE_M5_*`（若干） | 前 M5 机型专用融合（本机 M5 不适用） |

### 8.3 decode 排队 / 流水

| 环境变量 | 关掉什么 |
|---|---|
| `DS4_METAL_DISABLE_V41_DECODE_QUEUE` | 单机 decode 每层入队后不等待（跨层流水） |
| `DS4_METAL_DISABLE_V41_DECODE_PIPELINE` | 整个 decode 流水 |
| `DS4_METAL_DISABLE_V41_DECODE_PIPELINE_PORTS` | 流水端口复用 |
| `DS4_METAL_DISABLE_V41_DECODE_FLUSH` | 流水中的 flush 策略 |
| `DS4_METAL_DISABLE_V41_RESIDENT_DECODE_QUEUE` | 驻留模式的 decode 队列 |
| `DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE` | TP 模式的 decode 队列 |

### 8.4 流式专家 gather（本机唯一在跑的大杠杆）

| 环境变量 | 作用 |
|---|---|
| `DS4_METAL_DISABLE_V41_STREAMING_VERIFY_GATHER` | **不可关**：verify 行会退化成整层 map → `model range not covered` 直接失败 |
| `DS4_METAL_DISABLE_V41_STREAM_GATHER_MM_ID` | gather 的 mm-id kernel 族（默认关掉会走另一族） |
| `DS4_METAL_DISABLE_V41_STREAMING_SWEEP_GATHER` | prefill sweep 的 gather |
| `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR` | 流式 prefill 批量 selected-address |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE` | 流式专家地址表 |
| `DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_MAX` | 该路径允许的最大行数，**默认 800** |

### 8.5 Engram 读重叠（本地实现）

| 环境变量 | 作用 |
|---|---|
| `DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP` | 关掉 `pipeline_layers`（层流水：flush 在 `il==0/4`、drain 在 `il==13`） |
| `DS4_DISABLE_V41_ENGRAM_STEP_READERS` | 关掉 step reader 双缓冲 |
| `DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE` | 编码阶段的 engram 流水 |
| `DS4_METAL_DISABLE_V41_ENGRAM_PREFETCH` | engram 预取 |

> 上游 `ds4_engram_reader`（4 个常驻 pthread + 每表 start/wait）**已编译但未接线**。二选一待裁决。

### 8.6 批量分段（verify/prefill 共用，改动会同时影响 prefill）

`DS4_METAL_DISABLE_V41_BATCH_ATTN` / `_CORE` / `_COMPRESS` / `_HEAD` / `_HC` / `_INDEX` / `_INDEX_PROJ` / `_MOE` / `_PUBLISH` / `_TOPK` / `_ENGRAM`，
以及上游的 `DS4_DISABLE_V41_BATCH_OUTPUT`（关掉批量的注意力输出投影，回到逐行）。

> 实测：`_INDEX/_TOPK/_PACKED_INDEX/_COMPRESS/_HEAD` 对本机 8K 输出**零影响**；`_ATTN/_CORE` 会改变 **prefill** 结果（做二分时必须配对跑 A）。

### 8.7 prefill / 调度

`DS4_METAL_DISABLE_V41_LAYER_PREFILL`、`_WIDE_PREFILL`、`_WIDE_CHUNK`、`_PREFILL_ALIAS`、`_PACKED_INDEX`、`_SESSION_BATCH`、`_ENCODER_RESIDENCY`、`_SSD_MEDIUM_SWEEP`、`_TP_SMALL_PREFILL`、`_TP_SHARED_OWNER`、`_COMPACT_CARRY`、`_DEFER_DECODER`、`_DECODER_SUFFIX`。

### 8.8 流式专家缓存（生产参数，影响最大）

| 参数 | 说明 |
|---|---|
| `--ssd-streaming-cache-experts N\|NGB` | 生产手动档 **9000**（10-04 单腿 +6.0~7.8% decode）；本机实验用 1024 |
| `--ssd-streaming-preload-experts N` | 预载数量 |
| `--ssd-streaming-cold` | 跳过热度预载 |
| `--ssd-streaming-full-layers N` | 前 N 个 routed 层整层驻留（GLM 用） |
| `DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP` | 自动预载上限，**默认 4096**（账本记：hotlist 5156 条会被截到 32 层） |
| `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST` | B 臂对照**必须**设，否则内置表会静默变基线 |
| `DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE` | 打印 seed 证据行 |

### 8.9 DSpark（本轮新增）

| 参数 / 环境变量 | 说明 |
|---|---|
| `--dspark` | 启用（需 `--mtp-model`） |
| `--dspark-strict` | 加载 support 但只跑目标 decode（**最干净的对照组**） |
| `--dspark-confidence F` | 置信度剪枝。Metal 默认 **0.6**；exact sampling 默认 0.8 |
| `--mtp-model FILE` | support GGUF 路径 |
| `DS4_DSPARK_STATS=1` | 会话销毁时打印整行统计（见 §3.2） |
| `DS4_DSPARK_SPEC_LOG=1` | 每周期实时行 + scheduler skip 行 |
| `DS4_DSPARK_SCHEDULER` | 退避调度总开关，**默认开**（`=0` 关） |
| `DS4_DSPARK_SCHEDULER_WINDOW` | 判决窗口，默认 4（Spark TP/ROCm 16） |
| `DS4_DSPARK_SCHEDULER_SKIP` / `_SLOW_SKIP` | 跳过周期数 |
| `DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP` | 连续无草稿后跳几次，默认 3 |
| `DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP` | 默认 4 |
| `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP` | 默认 7 |
| `DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI` | 默认 500（=0.5） |
| `DS4_DSPARK_SCHEDULER_MIN_AVG_MILLI` / `_MAX_MS_PER_ACCEPT_MILLI` / `_MAX_EXTRA_SAVED_RATIO_MILLI` / `_BREAK_EVEN_WINDOW` / `_TAIL_MIN_TOKENS` | 其它退避阈值 |
| `DS4_DSPARK_VERIFY_CAP` | 强制 verify 行数上限，默认 0（=按 block 5） |
| `DS4_DSPARK_EXEC_TIER` | 执行档位选择 |
| `DS4_DSPARK_SEED_BATCH` | seed 批处理开关 |
| `DS4_DSPARK_EXPERT_OVERLAP` | 专家装载重叠 |
| `DS4_DSPARK_VERIFY_SPLIT_HEAD` / `_VERIFY_HEAD_NO_TP` | verify 头拆分 |
| `DS4_DSPARK_VERIFY_TOPS_CHECK` | verify 的 top-k 校验（H2 可能用得上） |
| `DS4_DSPARK_*_PROFILE`（PROP/STAGE/VERIFY/CYCLE_TRACE） | 各段计时 |
| `DS4_DSPARK_NO_MARKOV` / `_NO_GPU_MARKOV` / `_DISABLE_FUSED_CPU_MARKOV_ARGMAX` / `_DISABLE_REUSE_CONFIDENCE0_MARKOV` | Markov 头路径选择 |
| `DS4_DSPARK_FAKE_ARGMAX_PROPOSAL` | 诊断：假草稿 |
| `DS4_DSPARK_PROBE` | 探针 |

### 8.10 其它相关

- `DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS` / `_MXFP4_SELECTED_EXPERT_VIEWS` / `_IQ2_SELECTED_SHARED_OVERLAP`：流式专家视图与 shared 重叠。
- `DS4_METAL_DISABLE_KV_NORM_DEFER`：KV norm 延迟。
- `DS4_METAL_DECODE_STAGE_PROFILE`：decode 分段计时。

---

## 9. 未完成事项

1. **`LOCAL_INVENTORY.md` 未更新**（用户要求全部做完再更新）。需要登记：
   - 19 笔上游吸收、`e5b19e28` 退役（`0b793da`）；
   - 本地 `ds41_draft`/`ds41_spec` 退役、`tests/test_deepseek41_verify_cost.c` 删除；
   - 新资产 `gguf/DeepSeek-V4.1-Flash-DSpark.gguf` 与其转换命令；
   - 已知红：上游 `test_deepseek41_metal` 的 `quantization mode=3 index=5428`（**纯 origin/main 同红**，已用干净 worktree 验证）。
2. **模型级门未跑**：`--logprob-vectors`、`--local-golden-vectors`、`--tool-call-quality`、`ds4-bench`。
3. **`fork/main` 未推**（140 笔只在本地）。
4. **死代码**：`ds4_metal.m` 里 `ds4_gpu_dsv41_verify_oa_rows` / `verify_head_rows_admitted` / `verify_attn_out_rows_admitted` 已无调用者（5 处）。
5. **stats 死字段**：`prop_stage0…prop_confidence`、`verify_upload/layer/head/read` 需接上游计时点（用户已确认放到最后）。
6. **`--dspark-verify-depth` 的 prefix_capture 断言失败**需解释（§3.4）。

---

## 10. 已验证的门禁（本轮）

```
make                          零 error 零 warning
./ds4_agent_test              ok
./ds4_test --server           ok
tests/test_session_state_gpu  PASS
tests/test_glm53_kda          PASS
tests/test_mxfp4_metal        PASS
DSpark 短上下文等价            worst_argmax_gap=0.000（nspec=128）
DSpark support GGUF           --verify PASS（81 张量）
```
