# HANDOFF-DSPARK P2 结案：V4.1 DSpark verify/decode 分叉根因与修复（2026-10-19）

> 承接 `HANDOFF-DSPARK-20261019.md`。该文档的 H1/H2/H3 三条假设**全部不成立**。
> 本文记录实际根因（两条，互相独立）、修复、证据与仍然开放的事项。

## 0. 结论

| 项 | 状态 |
|---|---|
| P2（≥4K 输出分叉） | **已定位并修复**。根因是两条独立的 ulp 级路径分歧，不是 KV 环、不是压缩索引、不是多行 MoE 归约 |
| 验收（`--dspark` 与原 decode 逐字节相同） | **通过**：2K / 4K / 8K 三个上下文 A==B |
| P1（速度 −7.4~−7.9%） | **机理已定**：草稿器在 decode 阶段从未被喂特征，只能在 prompt 前沿起草一次。修复后起草次数为 0，因此 `--dspark` 现在是**性能中性**，而不是慢 7% |

修复后 `--dspark` 与基线**逐字节相同**，但**也不再有任何加速**——因为在当前实现下它根本没有可用的草稿。真正的性能修复见 §5。

## 1. 根因一：融合 router 少了 softplus 的小 exp 多项式

**症状**：verify 批的行 logits 与同一位置的逐 token decode logits 不等。绝大多数位置逐位相同，少数位置差 1 个 BF16 ulp，再经 40 层放大成 0.3 量级的 logit 差，最终翻掉近 tie 的词。

**定位链**（可复现，全部有确定值）：

1. 进程内 parity harness 在 512/1024/1536/2048/2560/3072/3584/4096 各取一个 5 行 verify 块：
   除 `len=1024 row=4` 外**全部 EXACT**。→ 不是上下文阈值，是稀疏事件。
2. 该点宽度无关（rows=2/3/4/5 都是 row 0 先分叉）→ 不是批量行数问题。
3. 两侧 session 的**全部隐藏状态逐字节相同**（`previous_kv/previous_score` ×4、
   `compressed[]`、`index_cache[]`、`window[0..39]`、`history`）→ **H1（128 槽 KV 环）不成立**。
4. 逐层探针：前 15 层 residual 逐位相同；`layer=15` 的 residual 有 **2/28672 个元素差 1 个 BF16 ulp**
   （0.00195312 = 2⁻⁹），而 layer=15 的 pre-attention `norm`/`q`/`kv` 仍逐位相同
   → 分歧发生在 attention 之后，且是**量化边界**事件，不是状态错乱。
5. 逐个 kill switch：`DS4_METAL_DISABLE_V41_ROUTER_FUSE=1`、`_MOE_FUSE=1`、`_HC_FUSE=1`
   **三者任一都能让 parity 变 EXACT**；`_SHARED_FUSE=1` 无效。
   三者的公共交集是 `ds41_router_fused` 被关掉 → 嫌疑收敛到融合 router。
6. `DS4_METAL_DISABLE_V41_ROUTER_SINGLE_DISPATCH=1`（禁用融合 matvec）**无效**
   → 不是 matvec 的归约顺序，是 select 段。
7. 对照两份实现的概率计算：

   | 位置 | softplus 表达式 |
   |---|---|
   | `metal/unary.metal`（参考 unary） | `select(select(log(1+ex), poly, ex < 1/32), x, x > 20)` |
   | `metal/dsv4_misc.metal`（两处） | 同上 |
   | `kernel_dsv41_router`（verify 用的批处理 router） | 同上 |
   | `kernel_dsv41_router_top6` | 同上 |
   | **`kernel_dsv41_router_select`（decode 融合 router）** | `select(log(1+exp(v)), v, v > 20)`，**缺 poly** |

   decode 走 `ds41_router_fused` → 缺多项式的那个；verify 走批处理 select → 有多项式。
   两者对小 logit（`< log(1/32)`）的专家概率差 1–2 ulp，正好解释"只在小 logit 专家上行分叉"。

**修复**（`metal/dsv41.metal`，`kernel_dsv41_router_select`）：把 decode 融合 router 的 softplus
改成与参考 unary 完全一致（含小 exp 多项式）。

**副作用（需知悉）**：旧基线自己用的是缺多项式的版本，所以**基线贪心输出会变**。
实测 2K 不变（`2deb7ada…`），8K 从 `9644e880…` 变为 `a320267a…`（即原来 dspark 那一侧的值）。
这是一次对齐参考实现的正确性修复，但它确实是用户可见的行为变化。

## 2. 根因二：drafter 的存在改变了 prefill 的分块，从而改变 prefill 数值

根因一修好后，4K 仍然 A≠B。进一步二分：

| 对比 | 结果 |
|---|---|
| A（无 drafter） vs C（`--dspark-strict`，drafter 常驻但零投机） | **仍需区分** |
| 2K prompt（1871 tokens）A vs C | **逐位相同**（0 个 logit 不同） |
| 4K prompt（5013 tokens）A vs C | 全部 logit 不同，中位相对差 1.3% → prefill 就走岔了 |
| 4K + `DS4_V41_DECODER_SUFFIX_MIN_ROWS=8192`（关掉 decoder suffix）A vs C | **逐位相同** |

→ 分歧在 prefill 的 **decoder suffix 依赖窗口**：

`ds4.c:ds41_graph_prefill_sweep`

```c
const uint32_t needed = (g->dspark ? 128u : 1u) + (DS4_N_LAYER - 1u - il) * 127u;
first = total_count - needed;
```

有 drafter 时窗口多 127 行 → `first` 平移 → suffix 的 chunk 边界全部改变 →
GEMM tiling 重新切分 → 舍入不同。这正是仓库自己在
`ds41_decoder_suffix_min_rows` 注释里承认的
"batch-on sweeps change GEMM partitioning and so carry the same tiling-rounding class"。
而 2K 低于 4096 的 suffix 门槛，所以 2K 一直是好的——与观测完全吻合。

**修复**（`ds4.c`）：让依赖窗口与 drafter 无关，恒为 `1 + (DS4_N_LAYER-1-il)*127`。
基线一个字节都没动（2K 哈希不变），`--dspark` 侧对齐到基线。
代价：drafter 初始特征窗口变短（见 §5）。

## 3. 验证矩阵（修复后）

```
A/B 端到端（--temp 0 -n 256，SSD streaming，cache 1024）
  2k:  A=2deb7ada95ac  B=2deb7ada95ac  IDENTICAL
  4k:  A=6147bc3c03db  B=6147bc3c03db  IDENTICAL
  8k:  A=a320267a4b67  B=a320267a4b67  IDENTICAL

进程内 parity（verify 行 vs 逐 token decode，全部 dmax=0）
  len 1024/1536/2048/2560/3072/3584/4096/6144/8192    PARITY OK
  sweep 80 块 × 5 行 from 1024   （400 个连续位置）   PARITY OK
  sweep 80 块 × 5 行 from 4096   （400 个连续位置）   PARITY OK
  sweep 60 块 × 5 行 from 8192   （300 个连续位置）   PARITY OK

门禁
  make                      零 warning 零 error
  ./ds4_agent_test          ok
  ./ds4_test --server       ok
  make dspark-verify-parity PARITY OK（新增）
  make dspark-verify-depth  仍红：tests/ds4_test.c:7306 prefix_capture 断言（**修复前即红**，
                            见 §6；同一次运行里 worst_argmax_gap=0.000 正常）
```

## 4. 为什么 H1/H2/H3 全错

- **H1（128 槽 KV 环）**：状态逐字节相同，且分叉点位于 attention 之后。环本身没问题。
- **H2（压缩层 top-512 索引）**：`compressed[]`/`index_cache[]` 逐字节相同；
  且 `_BATCH_INDEX/_TOPK/_PACKED_INDEX` 一类开关对结果零影响（与旧文档一致）。
- **H3（多行 MoE gather 归约）**：分歧源头在 router 概率，不在 routed 专家归约；
  逐层探针显示前 15 层 residual 逐位相同。
- 旧文档 §3.5 的"关掉仍分叉"是因为在 8K 端到端上看 token 流：那时**两侧各有一个不同的 bug**，
  关掉一侧的开关只会把分叉点挪走。只有把验证降到"单块 verify vs 单步 decode 的 logits"才看得清。

## 5. P1：草稿器为什么从来不起草（这才是 −7.4% 的真因）

`ds41_dspark_capture` 全仓只有**一个**调用点：`ds4.c:43580`，位于
`ds41_graph_prefill_sweep`（prefill）。decode 路径（`ds41_graph_layer`）从不调用它。

于是 `d->target[]` / `d->positions[]` 只装过 prefill 的行。
`ds41_session_speculate` 的 `ready` 判据要求最近 128 个位置的特征都在环里：

```c
for (uint32_t pos = g->pos > 128u ? g->pos - 128u : 0; pos < g->pos; pos++)
    for (unsigned stage = 0; stage < 3; stage++)
        ready = ready && d->positions[stage][pos % 128u] == pos;
```

生成出第一个 token 之后该判据必然失败 → **整轮最多在 prompt 前沿起草一次**。
实测正是 `cycles=255 no_draft=254`，那 1 次还花了 1.7 s 起草 + 0.8 s verify，
换 1 个 token → 净亏 4336 ms ≈ 就是那 7–8%。

修复后（§2 让窗口与 drafter 无关）：`cycles=256 proposed=0 no_draft=256`，
速度 12.66 t/s ≈ 基线 12.75 t/s——**变中性了，但也没有收益**。

**把 DSpark 变成真正的加速，需要的是**：在 decode 的层循环里（target 层 37/38/39）
调用 `ds41_dspark_capture`，让特征环随生成滚动预热；随后 §2 那个
(drafter 依赖窗口) 也就完全没必要存在了。这是**功能补全**，不是 bugfix，
且会改变速度画像，建议单独立项。

## 6. 模型级门禁（§9.2）实跑结果

| 门禁 | 结果 | 归因 |
|---|---|---|
| `./ds4_test --tool-call-quality` | **红**：exact 路径 `V4.1 layer 1 failed at position 0`，5 条断言 | **既有红，与本次改动无关**。用 `git worktree add /tmp/ds4-head-check HEAD`（不含本次改动的干净 HEAD `c1b4aaa`）重建 `ds4_test` 复跑，**同样 5 条、同样日志**。fast 路径正常（`finish_reason=tool_calls` → `stop`，replay_mem=1） |
| `./ds4_test --logprob-vectors` | **不可用**：本机无匹配 fixture | 向量按 checkpoint 分目录（`tests/test-vectors/README.md`：*"A mismatch between checkpoint and fixture is an invalid test, not a model-quality result"*），现有只有 `flash-0731`/`flash-pre-0731`，**没有 1017 的向量**。要么 `fetch_official_vectors.py --checkpoint 1017`（需 API key），要么由 1017 自身重新 capture——后者会把本次改动烘进基线，属于再基线化决策 |
| `./ds4_test --local-golden-vectors` | 同上 | 同上 |
| `./ds4_test --server` / `ds4_agent_test` | 绿 | 本次已跑 |

**§9.6 的 prefix_capture 断言已解释清楚**（不是模型问题，是**测试脚手架过期**）：

`ds4_test_dspark_prefix_capture` 调 `metal_graph_verify_suffix_tops(&spec->graph, …)`，
而该函数第一道守卫是
`if (n_tokens == 0 || n_tokens > g->prefill_cap || !g->spec_logits) return false;`
（`ds4.c:39157`）。DeepSeek V4.1 的 session 在 `ds4_session_create` 里**只分配
`s->ds41_graph`**（`ds4.c:76312`），从不分配 fork 时代的 `s->graph`，
所以 `spec_logits`/`prefill_cap` 恒为 0/NULL → 直接返回 false → 断言必红。
它测的是已退役的 fork 草稿器设施（`spec_frontier_commit_prefix`），与 DSpark verify 路径无关。

顺带纠正旧文档 §3.4 的读法：**不存在"两套配置"**。`test_dspark_verify_depth` 只有一个配置，
`TEST_ASSERT` 只记失败计数、不中断（`test_failures`，见 `tests/ds4_test.c:7431`），
所以同一次运行会先打印这条断言失败，再打印 `worst_argmax_gap=0.000` 并让其余检查通过。
旧文档把它读成"第一套失败、第二套通过"。

## 7. 仍然开放

1. **`LOCAL_INVENTORY.md` 未更新**。台账按提交记 SHA，本次改动尚未提交，
   所以要先落提交再登记。待登记：`metal/dsv41.metal` 的 softplus 对齐、
   `ds4.c` 的依赖窗口去 drafter 化、`tests/test_dspark_verify_parity.c` +
   `make dspark-verify-parity`、以及"基线 8K 输出变化"这一行为变更。
2. **`ds4.c` 里新增的测试探针**（`ds41_test_probe_*`，`#ifdef DS4_TEST_HOOKS`，25 行）：
   是这次二分的关键工具，编译不进生产二进制。保留还是移除，请裁决。
3. **`--tool-call-quality` 的既有红**（exact 路径 position 0 失败）需要单独查；
   它在干净 HEAD 上就红，属于另一条在途问题。
4. **向量 fixture**：1017 checkpoint 缺 official/local 向量，`--logprob-vectors` 与
   `--local-golden-vectors` 目前是无效测试。再基线化需拍板。
5. **`fork/main` 未推**（`main` 未动，140 笔在 `rebase/20261019`）；分支去向见旧文档 §6.5。

## 8. 复现与工具

```sh
# 本次新增的回归门禁（需要主模型 + DSpark support）
make DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
     DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4.1-Flash-DSpark.gguf \
     DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 \
     dspark-verify-parity

# 二分工具（单独使用）
tests/test_dspark_verify_parity --lengths 1024,4096,8192 --rows 5      # 每长度一块
tests/test_dspark_verify_parity --lengths 4096 --rows 5 --sweep 80     # 连续 400 个位置
tests/test_dspark_verify_parity --lengths 1029 --rows 2 --state        # 两侧隐藏状态逐张量比对
tests/test_dspark_verify_parity --lengths 1029 --rows 2 --trace        # 逐层激活比对，定位首个分叉层
tests/test_dspark_verify_parity --lengths 1029 --rows 2 --fresh        # 每次重建 session（慢但无复用假设）
```

实验产物已从 `/tmp/dspark-ab` 归档到 `speed-bench/dspark-20261019/`。
