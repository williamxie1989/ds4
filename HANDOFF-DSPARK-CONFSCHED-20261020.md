# HANDOFF：DSpark confidence-scheduled varlen verification（2026-10-20）

> 承接 `HANDOFF-DSPARK-20261019.md`（P1 速度）与 `HANDOFF-DSPARK-P2-20261019.md`（P2 分叉 + §5 草稿器不喂特征）。
> 本文记录：本次做的东西、为什么必须连带做前置补全、实测数字、以及仍然亏在哪。

## 0. 结论

| 项 | 状态 |
|---|---|
| 草稿器在 decode 阶段被喂特征（P2 §5 的功能补全） | **已做**。`ds41_graph_layer` 与 verify/prefill 批都调用 `ds41_dspark_capture`，特征环随生成滚动 → `ready` 判据不再在 prompt 前沿之后失效 |
| 起草次数 | `--dspark` 从 **proposed=0** 变成每轮都起草（`max_chunk=6`，5 draft + verify） |
| 置信度调度的变长验证（DeepInfer `mtp_varlen` 借法） | **已做**。按生存概率 × 实测成本决定块长；并据此决定“这一轮要不要起草” |
| 输出等价性 | **保持**。2K/4K/8K 以及所有 A/B 分支逐字节相同（md5 一致） |
| 速度 | **仍然是亏的**，但只剩“探测税 + 特征环税”：n=1024 时 −4.4%（其中一次性冷探测 ~2.2 s、其余 ~1%/73 s）；同样条件下固定阈值 −23% |
| 根因（实测，不再是猜） | 本机 **verify 一行 ≈ 1.8~2.3 倍 decode 一步**（145~175 ms vs 73~79 ms）。一行验证要换来一个 token 才划算，acceptance 不可能到这个量级，所以 DSpark 在这台机器的流式路径上**结构性不划算** |

一句话：**机制做完了、跑通了、输出没变；“让 DSpark 变快”在这台机器上不成立，原因是 verify 行比 decode 步还贵，不是调度没做好。**

## 1. 为什么必须先补特性环

`ds41_session_speculate` 的 `ready` 判据要求最近 128 个位置的三层特征都在环里。环只有 `ds41_graph_prefill_sweep` 会写（`ds4.c:43588` 旧行号），decode 从不写 → 生成一个 token 之后判据必然失败。P2 §5 已实测 `proposed=0`。

所以“confidence-scheduled varlen”单独做是**没有输入的**。本次先把环补上，再让调度去决定长度：

- `ds41_graph_layer`：单 token 解码层尾（以及 prefill 的非 batch 回退路径、CUDA decode 回退）每个 target 层（37/38/39）capture 1 行。
- `ds41_graph_step_batch`：当 `verify_prefix`（verify 批）或 `prefill_only`（单会话短 prefill）时 capture `rows` 行。`ds41_dspark_verify` 已经在批前保存了被覆盖的 slot（`old_features`/`old_positions`），`commit()` 会把被拒行放回去，所以环里最终只剩“被接受的前缀”——这正是那两个备份缓冲区原本的用途。
- **不**在会话批（`native_ds41`，多会话共享 workspace）里 capture：那里的行属于不同 session/position。
- `--dspark-strict` 用 `ds41_gpu_graph.dspark_capture=false` 完全跳过（控制臂不能为不存在的投机付 9 个 dispatch/token）。

`capture` 只读 residual、只写 drafter 私有缓冲区，所以对目标模型的数值零影响——A/B 的 md5 全一致可以证明。

## 2. 调度器

### 2.1 长度规则（纯函数，`ds4_dspark_conf_len` / `ds4_dspark_conf_probe_step`）

第 k 个 draft token 只有在“它之前整段前缀都被接受”时才会变成 token，概率是 `prod_{i<=k} p_i`（`p_i = sigmoid(confidence_i)`）。它能替代的是一个 decode 步；代价是一条顺序 draft 行（Markov 偏置 + 全词表回读）加一条 verify 行。于是：

```
prod_{i=1..k} p_i * decode_ms >= safety * (draft_row_ms + verify_row_ms)
```

**生存概率、不是单 token 置信度**，才是该停的位置：p=0.8 时第一行值 0.8 个 decode 步，第五行只值 0.33。固定阈值（`--dspark-confidence`，Metal 默认 0.6）仍然是硬下限，调度只可能把块**变短**。

### 2.2 周期准入（`ds4_session_dspark_conf_admit`）

drafter 的 5 行块在第一条 Markov 行之前就已经付掉了，所以“要不要起草”必须单独判：用**上一轮置信度链**的期望 token 数（`sum_{j} prod_{i<=j} p_i`）乘 decode 一步，与实测 `propose+verify` 周期成本比较。低于 break-even 就整轮走普通 decode。

- 重探默认 1024 个 skip，且只有成本朝有利方向变化（decode 变慢 25% 或 verify_row 便宜 20%）才提前到 8。
- 一次探测本身就是一次冷 verify（本机 ~1 s），所以不能频繁探。

### 2.3 成本从哪来

| 量 | 来源 | 说明 |
|---|---|---|
| `decode_ms` | `ds4_session_eval_internal` 的 V4.1 单 token 路径（`ds41_graph_step` 前后计时），4 样本 EMA | 没测到时先用 `verify_row` EMA 顶替：一行 verify 与一步 decode 是同一份单行模型计算，而“每轮都成功起草”的 session 永远不会跑普通 decode，否则准入永远无法启动 |
| `draft_row_ms` | `ds41_session_speculate` 里每条 `ds41_dspark_markov` 的耗时 | 实测 ~0.5 ms，可忽略 |
| `verify_row_ms` | `ds41_session_verify` 总耗时 / `count` | **本机 145~175 ms**，是全部问题的所在 |
| `cycle_ms` | 每轮 `propose + verify` | 准入用 |

EMA 都是 4 样本记忆（`ema += 0.25*(x - ema)`），因为流式专家缓存会变。

## 3. 开关与统计

| 开关 | 含义 |
|---|---|
| `DS4_DSPARK_CONF_SCHED=0` | 回到固定阈值 / 固定块宽（A/B 用） |
| `DS4_DSPARK_CONF_SCHED_WARMUP` | 多少样本后启用规则，默认 1 |
| `DS4_DSPARK_CONF_SCHED_SAFETY_MILLI` | 安全系数，默认 1000（=1.0，纯粹 break-even） |
| `DS4_DSPARK_CONF_SCHED_REPROBE` | skip 多少轮才重探一次，默认 1024 |
| `DS4_DSPARK_CONF_DECODE_MS_MILLI` / `_DRAFT_ROW_MS_MILLI` / `_VERIFY_ROW_MS_MILLI` | 用离线曲线驱动规则（回放用） |

`DS4_DSPARK_STATS=1` 新增字段：

```
conf_sched=N conf_stops=N conf_blocked=N conf_skips=N
cost_ms decode=.. draft_row=.. verify_row=.. cycle=..
conf_len_hist=..            # 规则给出的块长
```

`conf_sched` 是“规则真的生效了多少轮”；`conf_blocked` 是“规则把第 0 行也否掉了”；`conf_skips` 是“准入否掉了整轮”。

## 4. 验收

### 4.1 模型门禁（本机，M5 Max 128 GiB，SSD streaming cache=1024）

```
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark.gguf \
DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 \
./ds4_test --dspark-verify-depth
  → nspec=128 max_chunk=6 draft_depth=5 worst_argmax_gap=0.000 at=-1   OK

make DS4_TEST_MODEL=... DS4_DSPARK_SUPPORT=... DS4_TEST_SSD_STREAMING=1 ... dspark-verify-parity
  → lengths 2048/4096/8192 全 EXACT dmax=0；sweep 16 blocks from 4096 PARITY OK
```

`max_chunk=6` 是本次的关键证据：修复前 `proposed=0`，现在每轮都能起草满 5 个 draft 并整套 verify。

### 4.2 `--dspark-verify-depth` 的门禁改动

原来第一行断言 `ds4_test_dspark_prefix_capture(...)`，而它调
`metal_graph_verify_suffix_tops(&spec->graph, …)`；V4.1 session **从不分配 `s->graph`**（只分配 `ds41_graph`），所以这条断言在这个模型族上永远为假，测的是已退役的 fork 前缀机制（P2 §6 已解释）。

本次把它删掉（连同 `ds4_test_dspark_prefix_capture` 这个已无调用者的 hook），改用活的路径做门禁：
- `max_chunk > 1`：真的发生了多行验证；
- `worst_argmax_gap <= 2.0`：提交的 token 仍是目标 argmax。

并且在这个测试里设 `DS4_DSPARK_CONF_SCHED=0`：**这条门禁测机制，不测经济性**，否则成本曲线可能判定“今天不值得验证”而让门禁随机变红。

## 5. 速度（`--temp 0`，2K prompt，M5 Max / SSD streaming cache=1024，两轮/组）

| 组 | n | A 基线 | B1 调度（默认） | 差 |
|---|---|---|---|---|
| 2K | 384 | 13.4 | 12.1 | −9.4% |
| 8K | 384 | 13.2 | 11.9 | −9.1% |
| **2K** | **1024** | **14.0** | **13.4** | **−4.4%** |

（B1 的中间版本在 n=128 上是 2K −6.9% / 8K −20%；版本间差异来自准入/回退的几次修正，上表只列最终二进制。）

| 对照（n=128） | 2K | 8K |
|---|---|---|
| B0 固定阈值（`DS4_DSPARK_CONF_SCHED=0`） | 9.01 (−22.9%) | 8.79 (−23.8%) |
| C `--dspark-strict`（控制臂） | 11.92 (+2.0%) | 11.37 (−1.4%) |

所有分支输出 md5 相同（逐个上下文比对：2K `3bb1662b…`、8K `faead71c…`；n=1024 为 `7e855a34…`）。

读法：
- **C 中性**：`dspark_capture=false` 让 strict 完全不为投机付费 —— 这条是本次特意加的，否则控制臂会变慢。
- **B0 是“把草稿器修好”的直接后果**：固定阈值 + 真草稿 = 每轮起草并验证，−23%。这是 P2 之后没人观察到的真实成本，也说明“先把草稿器修好”本身是个负收益动作。
- **B1 的亏随长度收敛**：n=1024 的 805 轮里只有 **1** 轮起草（`conf_skips=804`），那轮是发现成本：`propose=1708 ms`（drafter 5 行块 + 5 行词表投影首次触碰专家）+ `verify=479 ms`。一次性 ~2.2 s，占 73 s 的 3%；剩下的 ~1% 是特征环 capture（9 dispatch/token）的固定税。
- 也就是说：**调度器一旦知道“不划算”，后面 804 轮一分钱不花。** n=384 组看起来更差，纯粹是因为那 2.2 s 摊不开。

## 6. 为什么还是亏（真正的结论）

实测成本（`DS4_DSPARK_STATS=1`）：

```
cost_ms decode=73~79  draft_row=0.5  verify_row=145~175  cycle=1000~1400
```

- verify 一行 = 1.8~2.3 个 decode 步。要回本，第 k 行的生存概率得 ≥ ~2，不可能。
- 即便 acceptance 是 100%，一轮 2 行的验证：成本 ≈ 1700（5 行块，冷）+ 480（verify）ms，换来 2 个 token ≈ 140 ms 的 decode。**十几倍亏**；预热后块成本降下来也仍是 3~8 倍亏。
- 所以调度器的正确行为就是“别起草”，它也确实这么做了；剩下的成本只是**为了知道这件事而付的一次探测**。

要让 DSpark 在这台机器上真的赚，必须让 verify 行**显著便宜于** decode 步，方向有：
1. verify 行批走真正的流式 expert gather 复用（当前每行仍是一次 gather，行间不共享 IO）；
2. draft 块宽可变（`ds41_dspark_forward` 现在固定算 5 行 + 5 行词表投影，探测时也在为不用的行付费）；
3. 或者在 verify 便宜得多的形态上再用（常驻、大 cache、或未来的 expert-parallel）。

这三条都不是调度能解决的，属于 §7 的后续。

## 7. 未决 / 后续

1. **`LOCAL_INVENTORY.md` 未登记**（本轮未提交；按仓库规矩要先落提交）。
2. **可变 draft 块宽**：`ds41_dspark_forward` 固定 5 行。按 `cap` 收窄能直接削掉探测成本的一大半，是下一个最高性价比的点。
3. **探测成本**：一次 ~2.2 s（drafter 5 行块 + verify 块的冷 expert 装载），已经是“最少一次”的代价。可以再压：用第一次探测结果跨 session 缓存（server 场景），或让 drafter 块宽可变（见上一条）。
4. **特征环税 ~1%**：`--dspark` 期间每 token 多 9 个 dispatch（3 层 × weighted-sum/quantize/copy）。想省掉它需要“停采 → 重探前再预热 128 token”的状态机；1% 不值这个复杂度，但记录在案。
5. **`net_saved` 口径**：本轮顺手修了两个问题（`spec_total` 不再重复计 propose；ds41 路径补上 `saved_ms` 记账），但 `saved_ms` 依赖 `dspark_last_target_eval_ms`，前两轮可能还是 0 用 EMA 顶替。别拿它当唯一验收。
6. **`DS4_DSPARK_SCHEDULER_*`（退避调度）在 V4.1 路径上从未接线**，本次也没有接：V4.1 的经济性判断由新的成本准入负责。两套策略不要同时开。
7. **其他后端**：本次只在 Metal + SSD streaming 上实测。CUDA/ROCm/常驻路径的 `verify_row` 可能远低于 decode，届时同一套规则会自动放行（这正是它该有的行为），但没有实测证据。

## 8. 复现

```sh
# 机制门禁（需要主模型 + DSpark support）
DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
DS4_TEST_DSPARK=gguf/DeepSeek-V4.1-Flash-DSpark.gguf \
DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 \
./ds4_test --dspark-verify-depth

make DS4_TEST_MODEL=gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
     DS4_DSPARK_SUPPORT=gguf/DeepSeek-V4.1-Flash-DSpark.gguf \
     DS4_TEST_SSD_STREAMING=1 DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=1024 \
     dspark-verify-parity

# A/B（脚本在 speed-bench/dspark-confsched-20261020/）
bash speed-bench/dspark-confsched-20261020/run_ab7.sh    # A vs B1，n=384
bash speed-bench/dspark-confsched-20261020/run_ab.sh     # A/C/B0/B1，n=128

# 单跑，看成本曲线
DS4_DSPARK_STATS=1 ./ds4 -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 1024 -c 16384 -n 384 --temp 0 \
  --prompt-file speed-bench/dspark-confsched-20261020/p8k.txt \
  --dspark --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark.gguf
```

产物目录：`speed-bench/dspark-confsched-20261020/`（`run_ab*.log`、`*.err` 里的 stats 行、`verify-depth.log`、`verify-parity.log`）。

## 9. 本次跑过的门禁

```
make                                  零 error 零 warning
./ds4_test --dspark-conf-sched        OK（纯函数真值表，无模型）
./ds4_test --metal-kernels            OK
./ds4_test --server                   OK
./ds4_agent_test                      OK
./ds4_test --dspark-verify-depth      OK（max_chunk=6，gap=0.000）
make dspark-verify-parity             PARITY OK（2048/4096/8192 + sweep 16）
A/B 输出等价                          md5 全一致（2K/8K，A/B0/B1/C）
```

未跑：`--logprob-vectors` / `--local-golden-vectors`（本机无 1017 的向量 fixture，见 P2 §6）、`--tool-call-quality`（P2 记录为既有红）、`ds4-bench`。

## 10. 修正（同日更晚）：成本分解，以及“还是有提速空间”在哪

§5/§6 的第一版结论（“verify 一行 = 1.8~2.3 个 decode 步，所以结构性不划算”）**把固定开销摊进了行成本**，方向对但不够准。用 `DS4_DSPARK_VERIFY_CAP` 扫块宽（fixed-threshold，cache 1024/4096，2K，n=192）可以把 verify 成本拆成“每轮固定 + 每行边际”：

| cache | verify 固定/轮 | verify 边际/行 | decode 一步 | 边际行 / decode |
|---|---|---|---|---|
| 1024 | ≈ 57 ms | ≈ 50 ms | 86.5 ms | 0.58 |
| 4096 | ≈ 49 ms | ≈ 36 ms | 52.2 ms | **0.69** |

（拟合点：cap1 = 2 行/轮，cap5 = 2.69 行/轮；cap5 的 `verify_row` EMA 54.2 是含固定摊销的平均值，不是边际值 —— 上一版把它当边际值用了。）

顺带一个与 DSpark 无关但很重要的发现：**`--ssd-streaming-cache-experts` 从 1024 提到 4096，基线快 20%**（13.4 → 16.2 t/s），DSpark 臂快 30%（10.6 → 13.9）。所以之前拿 cache=1024 的数字下“结构性”结论，有一半是缓存不够。

### 10.1 周期的账（cache 4096，实测）

```
每轮产出   1（seed）+ 2.2（accepted）           = 3.2 token
每轮成本   propose 24 ms + verify(49 + 2.69×36) = 170 ms
同等 token 的普通 decode                        = 3.2 × 62 ms = 198 ms
```

咦，这么算是**赚的**？实测却是 −14%。差在**每轮不是 2.2 个 accepted**：61 轮 verify 只接受 73 个 draft（1.2/轮），而 61 轮里还有大量 0-accept 轮；把 58 个纯 decode 轮和历史一起摊，平均每轮 2.2 token 是对的，但**成本按轮固定、收益按命中分布**，`saved=4549 ms / propose+verify=10297 ms` 的实测口径是 −5.7 s，与墙钟 −14% 一致。所以：

- **每轮固定开销 73 ms（propose 24 + verify 固定 49）≈ 1.2 个 decode 步**，这是 DSpark 的“入场费”。
- **边际行只要 0.69 个 decode 步**，所以“多验证几行”本身不贵 —— 之前的调度器因为把边际值当成了平均值（0.84~1.0），过早判死，**这个判断需要修正**。

### 10.2 因此可动的三个杠杆（按实测头寸排序）

| # | 杠杆 | 现在的数 | 目标 | 预期 |
|---|---|---|---|---|
| 1 | **drafter 块宽按调度长度收窄**（`ds41_dspark_forward` 恒算 5 行 + 5 行词表投影） | ~24 ms/轮 | L=2~3 时 ~10 ms | 省 ~14 ms/轮 ≈ 8% 周期 |
| 2 | **verify 的每轮固定开销**（每行 40 层 `old_raw`/`old_features` 预拷贝、每行 8 个 view、全词表回读） | ~49 ms/轮 | 一半 | 省 ~25 ms/轮 ≈ 15% 周期 |
| 3 | **acceptance**（0.76/행 实测） | 73/103 = 0.71 | 0.85+ | 每 0.1 接受率 ≈ +0.35 token/轮 |

把 1+2 都做到（73 → ~35 ms/轮），在 cache 4096 上大致回到 break-even；再叠加 3 就是净赚。**这三条都不是“调度”能解决的，都需要动 drafter/verifier 的编码。**

### 10.3 为什么边际行只有 0.69 却仍然不够

边际行便宜是因为 verify 的批处理在共享权重读取上有部分收益；但**每行仍选自己的 6 个专家**（5 行 ≈ 26 个不同专家 vs decode 的 6 个），所以收益止步于 30%。要再往下压只能做“行间专家集合共享的 grouped GEMM”，那要求路由结果可共享 —— 模型层面不成立。

### 10.4 结论修订

- ❌ 旧结论：verify 行 ∀ 便宜不了 → 结构性不划算（**过强**，且被 cache=1024 放大）。
- ✅ 新结论：**每轮 73 ms 固定开销 + 0.69×decode 的边际行**，在 acceptance ≈0.7~0.76 下净亏 ~34 ms/轮；把固定开销减半、块宽收窄，就能把 DSpark 拉回 break-even 附近，acceptance 再涨一点就是净赚。
- 调度器本身没有错，但它的**成本模型用了含固定摊销的平均行成本**，会过早判死。修法：`verify_row_ms` 改记**边际**（用 `verify_ms` 对 `count` 做在线回归，或简单用“两档块宽的差”），并把每轮固定成本单独建模。

这就是下一步该做的两件事：**改成本模型（边际 + 固定两项），再动 verify 固定开销和 drafter 块宽。**

## 11. 真正的大杠杆：verify 的 97% 是 host encode（实测，不是猜）

在 `ds41_dspark_verify` 里加了三段计时（`verify_prepare_ms` = 状态拷贝与行图 setup，
`verify_layer_ms` = `ds41_graph_step_batch`，`verify_read_ms` = 等 GPU 的回读），
跑 cap5 / cap1（cache 4096，2K，n=192）：

| 组 | verify 总 | prepare | layer(encode) | read(wait) | 轮数 | 行/轮 |
|---|---|---|---|---|---|---|
| cap5 | 8602 ms | **4.0 ms** | **8371 ms (97%)** | 213 ms | 61 | 2.69 |
| cap1 | 8848 ms | 4.1 ms | 8629 ms (97%) | 199 ms | 75 | 2.00 |

两个结论，都推翻前面 §10.2 表格里的猜测：

1. **状态拷贝 / 行图 setup 几乎免费**（4 ms / 61 轮 = 0.065 ms/轮）。原始设想里“40 层 old_raw 预拷贝很贵”是错的。
2. **verify 的成本 = 40 层 × 行数 的 host encode**：8371/61 = 137 ms/轮，而普通 decode 一步（40 层 + GPU 等待）是 52 ms。137 / 2.69 行 ≈ 51 ms/行 ≈ 一步 decode 的 encode 量。

也就是说：**当前的 verify 批在 encode 层面根本不是“批”，它是“rows 个单 token decode 串起来”** —— `ds41_graph_step_batch` 里每行各自跑 `ds41_attention(&row,…)`，MoE 又走 per-row 的 streaming gather，所以 encode 开销严格随行数线性，一点都没摊掉。GPU 反而是闲的（read 只有 200 ms）。

### 11.1 这解释了全部现象

- 为什么边际行 ≈ 0.69 个 decode 步：那是“批里多一行”的 GPU 成本，而 encode 是 1:1 摊不下来。
- 为什么加 cache 只帮到一定程度：cache 减少的是 SSD 读（GPU 侧），encode 一分不减。
- 为什么 cap1/cap5 差不多：轮数不同、行数不同，但每行都要付一次 encode。

### 11.2 杠杆与预期（按头寸）

| # | 做什么 | 现在的数 | 做完的预期 |
|---|---|---|---|
| 1 | **让 verify 的 encode 真的批起来**：给 attention core 与 streaming expert gather 做 “exact rows” 变体（沿 `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` 既有模式：一次 dispatch 出 N 行、逐行逐位等于单行 decode） | 137 ms/轮 | 一次 ~55~70 ms/轮 |
| 2 | drafter 块宽按调度长度收窄（词表投影 5 行 → L 行） | ~24 ms/轮 | ~10 ms/轮 |
| 3 | acceptance 0.71 → 0.85 | — | 每 +0.1 ≈ +0.35 token/轮 |

只要第 1 条落地到 70 ms/轮：周期 170 → 94 ms，产出 2.2 token（= 136 ms 的普通 decode）→ **DSpark 从 −14% 变成约 +30%**；只做到一半（110 ms/轮）也仍是 **+15~20%**。第 1 条是**纯 encode 侧工程**，不改采样、不改模型语义；能不能保持逐位一致取决于新 kernel，`tests/test_dspark_verify_parity` 就是它的验收门。

### 11.3 修正后的结论

- ❌ “DSpark 在这台机器上结构性不划算” —— **错**。是 **verify 的 encode 没有批处理** 造成的，属于实现缺陷，不是物理限制。
- ✅ 现在的准确说法：verify 一轮 137 ms 的 encode 把 0.69×decode 的 GPU 边际优势吃光了；把 encode 批起来，DSpark 有 20~30% 的头寸。
- 调度器仍然是对的（它只按实测下结论），但它现在读到的 `verify_row_ms` 是“已被 encode 污染”的平均值；等 encode 批起来，它会自动放开块宽。

## 12. 追到底：87% 是专家搬运，不是 encode 结构（2026-10-21 凌晨）

§11 说“97% 在 host encode”，进一步把它劈开（新增 `verify_attn` / `verify_moe` 两个计时）：

| 组（2K, n=192） | verify | attn | **moe** | 其余 | 轮数 | 行/轮 |
|---|---|---|---|---|---|---|
| cap5, cache 4096 | 8703 ms | 268 ms (3%) | **7546 ms (87%)** | ~890 ms | 61 | 2.69 |
| cap1, cache 4096 | 8887 ms | 274 ms | 7766 ms | ~850 ms | 75 | 2.00 |
| cap5, cache 1024 | 10967 ms | 279 ms | 9808 ms | ~880 ms | 61 | 2.69 |

所以 §11.2 那条“批 encode”的杠杆被实测否掉了：

- 我把 `ds41_attention_batch`（prefill 的批处理注意力）接到 verify 上（`DS4_DSPARK_VERIFY_BATCH_ATTN`），
  **速度没动**（13.90/14.46 vs 12.05/14.32 t/s），因为 attention 只有 3%；
  而且它让 parity 变成 `PARITY BROKEN (15)`（dmax 0.2~3.1，4096 处一次 argmax 翻转）—— 因为批处理与增量 decode 的归约顺序不同。
  该实验已删除，只留下“attention 占 3%、不值得批”这条结论。

真凶是 MoE：`ds41_moe_batch` → `ds4_gpu_routed_moe_batch_tensor` 的流式 gather，
每层要把本块**所有行选中的、去重后的**专家从 SSD 页进来。**行数越多，去重后的专家越多**：
1 行 6 个、2.69 行 ≈ 16 个。所以“每 token 的专家字节数”在 verify 侧**永远不会比 decode 低**，
而这正是流式路径的全部成本来源。

## 13. 真正的提速在那儿：专家缓存（实测，+54%）

顺手把基线（不带 DSpark）按 cache 扫了一遍（2K, n=192, 两轮）：

| `--ssd-streaming-cache-experts` | 基线 t/s | 相对 1024 |
|---|---|---|
| 1024 | 13.7 | — |
| 4096 | 16.7 | **+22%** |
| 8192 | 21.0 | **+54%** |

之前所有 A/B 都用 1024（沿用 P2 的测试档），这是那批数字整体偏低的另一半原因。

同一 cache（8192）上 DSpark：

| 组 | t/s | 相对基线 21.0 |
|---|---|---|
| 固定阈值 cap5 | 14.96 / 16.89 | −24% |
| **置信度调度（默认）** | **19.43 / 19.31** | **−8%** |
| 探测开销 | verify 245 ms 总计（1 轮），`verify_moe=223 ms` | — |

即：调度器在 8192 上把损失压到 −8%（1 次 245 ms 的探测 + 特征环 capture），
但**没有任何配置能让它在流式下超过基线** —— 一个 verified row 必须为自己的 6 个专家付 SSD 字节，
decode 一个 token 也是 6 个，speculation 只可能让这个数变多（draft rows 各带 6 个），不可能变少。

## 14. 最终结论（三次修正后）

1. **流式（SSD/专家不驻留）下 DSpark 不可能正收益**，原因是每 token 的专家字节数不变、且 verify 侧按行累加（去重后更多）。
   这不是调度、不是 encode、不是 kernel 形状的问题，是数据搬运。
2. **调度器的价值是止损**：固定阈值 −24%，调度后 −8%，剩下的 8% 是 1 次探测 + 特征环(~1%)。
3. **真正能让这台机器变快的是专家缓存**：1024 → 8192 基线 +54%（13.7 → 21.0 t/s），一个参数。
4. DSpark 的正收益区间是**专家常驻**的配置（M3 Ultra/Spark/512 GB），那里的 verify 行几乎不搬盘；
   仓库已有那侧的正面记录，不在本机可测范围。
5. 因此：**不要把“流式 DSpark 提速”当目标**；把 `--ssd-streaming-cache-experts` 调到 8192 左右，
   或者去做流式 decode 的每 token 专家字节/队列（V41 roadmap 的 encode 与 gather 项）——那才是这台机器的速度来源。

## 15. ds4-server 生产配置实跑（384K ctx，自动缓存预算）

生产命令（用户机器上的实际形态）不带 `--ssd-streaming-cache-experts`，走自动预算：

```
ds4: metal SSD streaming cache target 83.62 GiB; effective 83.62 GiB =
     7.12 GiB prefill headroom + 76.50 GiB dynamic cache (8253 experts, 9.49 MiB each)
ds4: V4.1 static context buffers 12717.91 MiB (ctx=393216)
```

即 §13 里 1024/4096 只是**测试档**，生产早就在 8253 专家；把它调到 8192 没有意义，**别再动 cache 参数**。

同一配置下的 A/B（2K prompt，n=128，两轮）：

| 组 | t/s | 相对基线 |
|---|---|---|
| 基线（无 `--dspark`） | 19.62 / 20.41 | — |
| `--dspark`（调度器默认开） | 14.68 / 14.53 | **−27%** |

原因（`DS4_DSPARK_STATS=1`）：一次冷探测 = `propose 1582 ms`（drafter 5 行块首次装载）+ `verify 887 ms`（2 行，`verify_moe=866`），
合计 ~2.5 s，占 128 token 生成的 38%（实测净损 27%）。这一笔**与上下文长度无关、只发生一次**，长生成会摊薄；
稳态剩下的是特征环 capture（≈1~3%）。输出 md5 与基线一致。

结论：**生产上不要为了速度开 `--dspark`**。它可以开、不会崩、输出不变，但只会更慢。

### 开关速查（都给的是环境变量，命令行只加 `--dspark --mtp-model`）

| 目的 | 用法 |
|---|---|
| 调度器（默认） | 不加任何变量；`DS4_DSPARK_CONF_SCHED=0` 关闭 |
| 看成本曲线 | `DS4_DSPARK_STATS=1`（`cost_ms decode=… draft_row=… verify_row=… cycle=…`、`conf_skips=…`、`verify_moe=…`） |
| 固定阈值对照（−24% 那档） | `DS4_DSPARK_CONF_SCHED=0` |
| 最干净的控制臂 | `--dspark-strict`（加载 drafter、不投机、也不 capture） |
| 强制块宽 | `DS4_DSPARK_VERIFY_CAP=N`（N = draft 行数上限） |
| 用离线曲线驱动 | `DS4_DSPARK_CONF_DECODE_MS_MILLI` / `_DRAFT_ROW_MS_MILLI` / `_VERIFY_ROW_MS_MILLI` |

注意：`--batched-session N` 会绕过投机（会话批走普通 decode），要和 DSpark 一起用就别开它。
