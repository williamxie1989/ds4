# GLM-5.3-Flash × M5 Max 128 GB：下一轮 prefill / decode 提速方案调研

> 调研日 2026-10（当前 main 含 `0743c7f` 合并 + `1e09244` MLA-NAX + `9bc0bd7` 调优门控 NAX 放开）。
> **本轮为纯案头调研，未跑任何加载权重的 bench/fixture**（本机当前按"不可加载大模型"处理）。
> 所有实测动作都排在"空机执行窗口"清单里（§5），动手前必须停掉其它常驻 runtime。
> 域：M5 Max = NAX 域（tensor units 可用）。GLM-5.3-Flash-Q2 全驻留（~96.5 GB）。
> 现状锚点：prefill 5.2k ~514 t/s、41k ~480–486 t/s；decode ~38–39 t/s（短 ctx 冒烟）/
> ds4-bench ctx12288 ~33.6–33.8 t/s；MTP 净亏（66% 接受率、cycle 成本 ~1.8× 单 token）。

---

## 0. 证据基座（决策只用这三份数据）

1. **prefill @41k post-MLA-NAX encoder timeline**（`SPARSE_MLA_NAX_PLAN.md` §9，GPU 合计 85.2 s）：

| 占比 | kernel | 随 ctx 增长？ |
|---|---|---|
| 25.3% | `mul_mm_id_iq2_xxs_f32_mpp`（MoE routed） | 否 |
| 14.2% | `mul_mm_q8_0_f32_nax_direct_rhs_n128` | 否 |
| 12.1% | `mul_mm_id_q2_K_f16_mpp`（MoE routed） | 否 |
| 10.3% | 本 fork 的 MLA-NAX 核（改前 27.4%） | 是 |
| 6.8% | `mul_mm_q4_K_f32_nax_direct_rhs_n128` | 否 |
| 6.7% | `glm53_kda_prefill_*` | 是（线性） |
| 4.7% | `glm_indexer_scores_tiled` | **是（二次趋势）** |

2. **prefill @2k timeline**（`OMLX_WAVE_PORT_ANALYSIS.md` S0-3）：MoE 37.3%、KDA 链 8.1%、
   QK low-rank 7.7%、indexer **0.1%**、HC 2.2%。短 prompt 与长 prompt 是两个不同的优化域。
3. **decode timeline**（同前，32 token 段，**系合并前 main 所采**）：Q8_0 matvec 34.9%
   （328 次/token）+ Q4_K dense 8.8% + BF16 7.0% ≈ **58% 权重流式 matvec**，MoE 22.1%；
   有效带宽 ~500 GB/s，接近本机 roofline；attention 仅 2.0%。
   ⚠️ 调优家族（bf16 pair、wide head groups 等）合入后构成已变，**复采一次是 §5 的第一项**。

结构性结论（S0-1/S0-3 实测，不是推测）：
- GLM 路径 GPU busy 96–99%，host 编码被逐层 async flush 藏住 → **调度/排队/减 dispatch 家族在 GLM 上没有可回收 headroom**（#1041/#1042 那族机制不适用，那是 DeepSeek decode 的病）。
- prefill 提速 = 让 GPU 把同样算术干得更快（kernel 级）。
- decode 提速 = ①摊薄每 token 的权重搬运（MTP 每 cycle 多吐 token），或 ②把 ~640 次/token 的小 matvec 做大做少，逼近 roofline 上限。

---

## 1. Prefill 路径（按 ROI 排序）

### PF-1 · KDA prefill 递推 blocked / per-core（6.7–8.1%，线性涨）
- **现状**：`kernel_glm53_kda_prefill_recurrence`（`metal/glm53_kda.metal:249`）= 上游 #3984
  修复**前**形态：1 simdgroup/值行、逐 token 设备读 q/k/decay、每 token 两次 `simd_sum`、
  无寄存器态、无块化、无预取。
- **供体机制**（omlx #3984，机制可搬、码不可搬）：① blocked：每线程寄存器持 2 行×16 通道
  fp32 态 + 16-token 块 stage/预取 + 门在核内算 → **~2.3×/层**；② per-core：64×128 值行按
  GPU 核数切段 + 12-token 块 → 再 ~1.15×。
- **收益上界**：6.7% × (1−1/2.3) ≈ **+3.8% @41k**；2k 场景 ≈ +2.5%。不高但全 ctx 段通吃，
  且是唯一"线性增长"的非注意力项——ctx 越大越值。
- **落点**：重写 recurrence（先照 omlx 的 blocked 形态，per-core 二期）；`_prefill_prepare`/
  `_prefill_output` 的 conv4+SiLU+L2 融合另立小项。
- **风险与闸门**：fp32 累加序会变 → logprob-vectors + GLM53 双 fixture + long-fixture
  scalar control 对照（§4 新门）；**KDA 的 conv_state / recurrent_state 两个持久态跨 chunk、
  跨 session 一致性要单独回归**，这是这项独有的坑。
- **工作量**：1 个执行窗口内可完成 blocked 版。

### PF-2 · MoE 专家 GEMM packed 路径开门（37.4% @41k，最大单项；常数项）
- **2026-10 静态勘察修正（推翻"无蓝本"假设）**：GLM 走的是**非 packed** 的 mpp mm-id
  （`iq2_xxs_f32_mpp` 25.3% + `q2_K_f16_mpp` 12.1%）。V4.1 的 packed TensorOps 家族
  （`kernel_mul_mm_id_*_mpp_packed`：先 `pack_rhs` 把激活按专家聚成连续行，GEMM 全合并读）
  **kernel 侧对专家数完全运行时通用**——`map0_ne20_8` 实例存在、`pack_rhs`/`packed_offsets`/
  packed GEMM 全走运行时 args，且 IQ2_XXS+Q2_K 恰是 packed 支持的量化对。挡住 GLM 的
  只有 host gate 的 `n_expert == 6u`（ds4_metal.m）。GLM 几何全部过限：288≤384、
  4096≤5120、2048≤4096、2048 行 chunk packed RHS=128 MiB≤256 MiB 限。
- **已落地（本仓库，默认 OFF 零行为）**：门臂 `glm53_top8_packed`——
  `DS4_METAL_ENABLE_GLM53_ROUTED_MPP_PACKED=1` 且几何精确等于 288/8/4096/2048 才放行，
  现有 kill `DS4_METAL_DISABLE_ROUTED_MPP_PACKED` 仍叠加生效；quality 模式不受影响。
- **空机窗口 A/B**：`ds4-bench` @41k prefill，臂 ON/OFF 各 2 轮（间隔 ≥25s）+
  `DS4_METAL_ENCODER_TIMELINE` 分列 `*_packed` vs `*_f32_mpp` 每次调用 ms 与 pack_rhs 新增段。
  **诚实预期**：V4.1 收益发生在 ≥128 行/专家；GLM top-8×288 只有 ~57 行/专家
  （N=32 tile 2 块），pack 的 128 MiB 写流量可能摊不平 → 区间 **-2% ~ +4%**，以数据定去留；
  留档则顺手补 1 行 engage 日志锚点，不留就删臂。
- **风险**：gate 改动 env 默认 OFF = 零行为；ON 腿的位域差异走标准 fixture 闸门
  （packed 与非 packed 的累加序是否位一致**未证明**，按惯例必须过 100-case@4096）。
- **这是短 prompt TTFT 的第一杠杆**：2k prefill 里 MoE 占 37.3% 且完全不随 ctx 变
  （但注意 chunk=2048 时 2k prompt 只有一层 chunk 的 packed 收益）。
- **执行窗口结果（2026-10-03，判负）**：41k prefill ON/OFF 各 5 腿，OFF 均值 483.6 /
  ON 均值 479.4 t/s，**平均 -0.9%、区间完全重叠**。engage 已被 timeline 证实
  （`kernel_mul_mm_id_{iq2_xxs,q2_K}_mpp_packed` + `pack_rhs` 全出现），但 kernel 级
  packed+pack 合计 ≈ 非 packed 的 37.4% —— ~57 行/专家下 pack 流量与 tile 收益互抵，
  诚实预期带的下端（-2%）兑现。**裁决：门臂保持默认 OFF，不推广；MoE GEMM 若要再挤
  只能动 tile 形状/核本身，那是另一个提案，本项关账。**

### PF-3 · indexer 打分 NAX 化（4.7% @41k / 0.1% @2k，增速最快的增长项）
- **现状与蓝本**：`kernel_glm_indexer_scores_tiled` 全 SIMD；同仓库
  `kernel_dsv4_indexer_scores_nax`（`metal/dsv4_misc.metal:6587`）几何几乎相同（32 heads×128
  dim），门控/降级/测试骨架照搬。
- **决策口径**（`SPARSE_MLA_NAX_PLAN.md` §9 已定死）：理论上界 ~4%，**41k 端到端 <+5% 才留**
  → 实质上是 **≥64k 场景才合算**的投资。若当前生产 ctx ≤32k，此项缓行。
- **数值风险与 MLA 那次不同，更要紧**：打分输出决定 top-k **选择集合**，边界平票换行 = 语义
  变化。开工第一步是 kernel 级 **"选中 id 集合与经典核一致率 100%"** 检查，不到 100% 不往下
  走 fixture。（MLA 那次 P 残差换半精度教训在此复用：先找对精敏感量，再谈瓦片化。）

### PF-4 · 宽步 chunk 8192（结构项，收益 +2~6% 供体口径，须本机验证）
- **现状三门**：`DS4_GLM53_PREFILL_CHUNK_TOKENS 2048`（`ds4.c:39042`）、workspace rows>2048
  拒绝（`ds4.c:47645`）、score scratch 256 MB（`ds4.c:39040`）。
- **机制**：MLA-NAX 落地后，每 query 注意力成本已与 chunk 无关（瓦片化，正是本项目拆掉的
  病理）→ 8192 chunk 让 MoE 每专家行数翻倍、摊薄 per-chunk 固定开销。
- **前置判据（先证再改）**：①确认每 query 成本与 chunk 无关（大概率已成立，一条 A/B 即可
  定）；②**长上下文下真约束是 score scratch**：64k 时 256 MB 只够 ~4096 行，提到 1 GB 并
  纳入 memory guard；③内存账：96.5 GB 驻留 + ~3.6 GB workspace（450 KB/行 × 8192）+ KV，
  128 GB 机器余量吃紧，**必须实测 wired 峰值**并确认 memory guard 日志零 throttle。
- **回滚**：`DS4_GLM53_PREFILL_CHUNK=2048`。

### PF-5 · MLA 残余：选择列表压缩（先测空洞率再动手）
- 2048 选择槽里无效行（padding/内部空洞）仍进瓦片 P=0 白算，`SK=32` 固定步进。
- **先测**：indexer 输出侧统计一次无效槽占比；**空洞率 <5% 就降级**为常数优化
  （SK 32→64 摊薄 softmax/步进开销）。这是 buffer 契约变更（每 token 加 `n_valid`），
  改动面大于收益面，除非数据撑腰否则不做。

### 明确不做（有实测/裁决在案，别再开）
- HC pre 融合（实测 1.9–4.7% 占比，预期 1–3% e2e，omlx 的 +22% 是 MLX 框架税，禁止跨域引用）→ P3。
- causal 稠密前缀变体（<1%）、`qk_rope=64`（本模型 rope=0）、短尾 <32 token（门已挡）、
  **Q 残差双趟**（实测零效果已回退，MLA §8.2）。
- 把 omlx / M3 Ultra / MLX 的任何百分比当预期。

---

## 2. Decode 路径

### DC-1 · MTP 经济性专项（decode 唯一的大杠杆，P0）
- **成本解剖**（合并前实测，构成不变的部分先复算）：单 token 29.1 ms；MTP cycle =
  draft head **15 ms** + verify2 **37.5 ms** = 52.5 ms ≈ **1.8×** 单 token。
  期望收益 = 1 + p₂（第二 token 接受率），**回本线 p₂ > 0.8**；实测 66% → 净亏 8%。
  对照：omlx 同模型接受率 **79.7–84.6%、2.0–2.3 tok/cycle** → 差距在实现，不在模型。
- **2026-10 开工勘察结论（先于任何编码，修正了原假设）**：
  - **GLM cycle 结构性 depth-1，`--mtp-draft` 对 GLM 是静默 no-op**。
    `ds4_engine_mtp_draft_tokens()` 对 GLM 家族硬编码返回 2（=首 token+1 draft，
    与 `--mtp-draft` 值无关）；`ds4_session_glm_spec_cycle_impl` 只有一个 pending draft
    （`glm_mtp_draft`）、verify 固定 2 行。`glm_mtp_draft2/have2` 与链式起草
    （`qwen4_graph_mtp_chain_step`、snap2、3 行 verify）**只在 Qwen3.8 cycle 使用**，
    字段名 `glm_mtp_*` 被 Qwen cycle 复用是误导来源。所以"深度未生效"不是开关坏了，
    是 GLM 侧未实现——DC-1.1 的真实工作 = **把 Qwen 的链式起草机制移植到 GLM-5.3**
    （功能开发，含 3 行 verify 的 spec-state 快照：GLM 侧已有 2 行版
    `glm53_graph_copy_spec_state`，扩到 snap_after_second/snap2 语义）。
  - **诊断已落地（本仓库，`--mtp-timing` 门控，零行为改动）**：session 关闭时打印
    GLM MTP 经济性决策表——verify cycles、接受率、tokens/cycle、verify2/head+draft/plain
    三段均耗时、effective t/s vs plain t/s（原 `--mtp-timing` 只有逐行打印，统计要人肉数）。
    下次空机窗口直接 `--mtp-timing` 跑 5-10k token 即得回本判定。
  - **2026-10-03 执行窗口实测（greedy，17k ctx，216 verify cycles）**：接受率 **70.8%**、
    1.537 tok/cycle；verify2 **50.0 ms** vs plain 30.9 ms（verify2 成本随 ctx 涨：12k 时 37.5 ms，
    17k 已 50 ms）、head+draft 13.9 ms → **effective 27.5 vs plain 32.3 t/s = 净 -15%**。
    比 12k 口径的 -8% 更差：ctx 越长 MTP 越亏（verify 2 行与 draft 全窗都随 ctx 涨）。
    **裁决：MTP 保持默认关闭；任何"开 MTP 提速"的提案必须先过链式起草+draft 减半两道门槛。**
- **三条子路径（修正后）**：
  1. **链式起草移植（原"修深度"）**：Qwen3.8 的 depth-3 机制（chain_step + 3 行 verify +
     snap2）搬到 GLM-5.3 cycle。成本：1-2 窗。收益：接受率 80% 时 2.3 tok/cycle 的来源。
     先用已落地的经济性表量化"接受率高但只有 2-token 上限"还剩多少空间，再决定投入。
  2. **draft head 15 ms 砍半**：15 ms = 单 token 成本的一半，直接抬高回本线。查 draft 段是否
     吃在调优家族外（未走 bf16 pair / fused 核）；draft head 一层，融合空间大。
  3. **提接受率 66%→80%+**：先按 位置×prompt 类型（代码/长文/tool 上下文）分解 53/80 样本，
     判断是"第 2 token 天然难"还是"draft 分布偏移"。greedy verify 对齐 serial（omlx #4050
     机制）是**行为项**，验收按"greedy MTP == MTP-off"而不是"输出不变"。
- **成功判据**：端到端 decode ≥ +15%，且质量 fixture 不劣化；失败判据：三项子路径都做完
  仍 <78% 接受率 → 关账写进偏差日志，避免下轮再试。
- **前提**：`--batched-session` 与 MTP 互斥（server 日志自证），A/B 时两条腿都要非 batched，
  且与 34.4/39.2 这类 batched 口径**不可直接比**。

### DC-2 · decode matvec 做大做少（roofline 差距 ~≤10% 的收口项）
- ~640 次/token 的 matvec（Q8_0 328 次/token 占 34.9%）是"带宽被小 kernel 切碎"形态；
  有效 ~500 GB/s。调优家族已吃掉一大块（bf16 pair、wide head groups），**复采 decode timeline
  后再决定投哪**：若 Q8_0 matvec 占比仍 >30%，参照 V4.1 侧先例
  （`2db1152` M5 Q2 sum6 两 simdgroup 专用核）给 GLM decode 形状做 M5 特化 matvec；
  行拼接（同形状 K 行合并成一次 matvec）是供体 #3990 机制。
- **判据先行**：`DS4_METAL_DECODE_STAGE_PROFILE` + 每 matvec 尺寸分布；预期上限就是 roofline
  余量（~10%），承诺值取一半。
- **明确不做**：给 GLM 上逐层排队/单次 CB wait（#1041/#1067 家族）——GLM decode 99% GPU busy，
  没有调度空隙；机制不同病。
- **复采结论（2026-10-03 执行窗口；含一次自我纠错）**：decode@12288/256 基线 32.0 t/s。
  timeline 按**时间窗**分离 decode pass 的真实构成（每 token）：q8_0 matvec 4.8 ms、MoE
  pair-swiglu 4.1 ms、q4_K dense 2.7 ms、HC mix 2.6 ms、KDA output 2.4 ms、
  **router 已走融合核 `glm53_router_shared_exact` 2.1 ms**、q2_K 2.0 ms、HC expand4 2.4 ms、
  **LM head 已是 Q8_1（GGUF type 8，实测 1.16 ms/tok，无需再量化）**——全是调优家族形态，
  无单点大头；decode busy 87%，空隙 ~4.4 ms/tok。**DC-2 维持"roofline ≤10% 收口项"定性。**
  ~~"f32 router matvec 6.53 ms/tok = 墙钟 21%"~~ 为聚合窗错误：`mul_mv_f32_f32_4` 的 252 次里
  241 次落在 prefill 段（≈42 次/chunk × 6.3 ms = 41k prefill 的 7.3%），decode 段仅 11 次。
  `mul_mv` 前缀 ≠ decode 专属，归因必须按时间窗切，不能按 kernel 命名猜。
- **`DS4_METAL_PLAIN_MV_NR0=2` A/B：判负（同窗追加，ABBA + 核级证据）**：e2e
  34.40/34.44 vs 33.72/33.75（-2%，噪声带内），带 timeline 对腿 29.78 vs 29.78（0%）；
  两臂全部 kernel 的 dispatch 次数与耗时逐项相同——GLM decode 没有 kernel 走
  `plain_mv_single_row` 分支，该 env 对本模型是 no-op，**别再排窗口测它**。
- **新 PF 线索（P1，静态可查）**：prefill 段那个每 chunk×42 次、每次 6.3 ms 的
  `mul_mv_f32_f32_4`（= 41k 的 7.3%，6.5 s/89.4 s）调用点未定位；查
  `ds4_gpu_matmul_f32_tensor` n_tok>1 分支的全部调用者，若属可减半的 f32 读，收益上限 ~3.5%。

### DC-3 · 长上下文 decode attention（缓议）
- 32-token 段 attention 仅 2.0%，但 GLM 长 ctx decode 走 `..._indexed_decode_exact_*`（MLA-NAX
  计划**完全没碰** decode 路径）。≥64k 生产化时另开计划；测法已有（S8：µs ÷ KV 字节 → 有效带宽，
  <40% roofline 且占比 >15% 才立项）。

---

## 3. 排序与预期（诚实版）

| 优先级 | 项 | 域 | 预期（本机口径，非承诺） | 成本 |
|---|---|---|---|---|
| P0 | DC-1.1 打通 draft2 + 逐位置接受率 trace | decode | 诊断（决策输入） | 半天 |
| P0 | DC-1.2 draft head 融合/轻量化 | decode | 回本线 0.8→~0.6 | 1 天 |
| P0 | PF-2 MoE tile 扫参（首个 kernel 专项） | prefill（全 ctx） | e2e +2~4% | 1 窗 |
| P1 | PF-1 KDA blocked 重写 | prefill | +2.5~4%（全 ctx，随 ctx 微增） | 1 窗 |
| P1 | DC-1.3 接受率分解与提升 | decode | 若 ≥78%：有效 **45–55 t/s** | 1–2 窗 |
| P2 | PF-4 宽步 8192 | prefill 长文 | +2~6%（须先过两判据） | 1 窗 |
| P2 | PF-3 indexer NAX | prefill ≥64k | ≤4% 上界，64k+ 才合算 | 1–2 窗 |
| P2 | DC-2 decode matvec 特化 | decode | ≤5%（预期上限的一半） | 1 窗 |
| P3 | PF-5 选择列表压缩 | prefill 长文 | 取决于空洞率实测 | 先测 |
| 不做 | HC 融合 / causal 前缀 / Q 残差 / 排队家族搬 GLM | — | — | — |

叠加口径提醒：这些是乘法且互相在分母上，**prefill 现实总预期 +8~15%**（41k ~500→550~575）；
decode 若 MTP 三项都成，按 2.0 tok/cycle × 接受率 80% 保守算是 **~45 t/s 有效**，matvec 特化再
加零头。达不到就按项关账，不硬凑。

---

## 4. 闸门与纪律（沿用仓库既有制度，两条修订提案）

沿用：同 build 同机 ABBA、区间不重叠才宣称；kill switch + engage 日志锚定
（`grep -c engaged` 为 0 = 假腿）；kernel 线一次只开一条；`score_official` 必须显式重链
（不在默认 make 目标，MLA 那次就是这样拿到过假过）；两腿间留 ≥25 s。

**修订提案（需与仓库维护者确认后入 QA 文档）**——frontier 门从单一 `max|Δlogit|<2` 改为三条：
1. top-1 逐位相同（0 容差）；
2. baseline top-20 内 `max|Δlogit|` 有界（建议 3）；
3. **scalar control（`--quality`）必跑并入库**，漂移不得超过 scalar control（QA 已证同一探针
   它漂 9.8 还翻 top-1，单阈值界已被证伪）。

## 5. 执行窗口清单（所有涉及加载模型的动作，本调研均未执行）

1. 空机前置：停掉其它常驻 runtime（本机此前 omlx-server 常驻 ~41 GB）；free 回 ≥85% 才起下一腿。
2. 复采三件（一次窗口全带走）：post-merge **decode timeline**、`DS4_METAL_DECODE_STAGE_PROFILE`、
   GLM53 prefill 分阶段 trace。**env 钩子已接好**（原 `glm_graph_indexed_prefill_trace_*`
   三根桩硬编码 `return false`，现走 env）：`DS4_GLM_INDEXED_PREFILL_TRACE=1` 开分阶段
   （ensure_cache/upload/indexer/attention… 各段 ms，只打 ≥`DS4_GLM_INDEXED_PREFILL_TRACE_SLOW_MS`
   默认 100 ms 的慢段），`=…ALL=1` 打全量。这是 PF-2/PF-1/DC-2 排序的最后输入。
3. 之后每项 kernel 改动按各自 §2/§1 的闸门跑 A/B + fixture，一个窗口只归因一项。

## 6. 一句话

**prefill 吃 kernel：MoE tile（最大常项）→ KDA blocked（增长项）→ 宽步/indexer（长文专属）；
decode 只有一条大路：把 MTP 从净亏修成净赚（先修 draft2 未生效，再砍 draft head 15 ms，
再提接受率），matvec 特化只是 roofline 收口。** 所有百分比预期只认本机 ABBA，供体数字一律不进决策。

---

## 附：开工日志（第一批，编码日 2026-10；全部零行为改动，未加载模型）

| 改动 | 文件 | 验证 |
|---|---|---|
| GLM53 prefill trace 三桩接 env（`DS4_GLM_INDEXED_PREFILL_TRACE[_ALL][_SLOW_MS]`，未设=原行为） | ds4.c `glm_graph_indexed_prefill_trace_*` | `make -j8` 干净，`-Wall -Wextra` 零告警 |
| GLM MTP 经济性汇总（cycles/接受率/tokens-per-cycle/三段均耗时/effective-vs-plain t/s，`--mtp-timing` 门控，session 关闭时打印） | ds4.c session 字段 + `ds4_session_glm_spec_cycle_impl` + free 路径 | 同上 |
| `test_glm53_router_shared` 补 `ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL)`（上游缺陷：非 M3-Ultra/非 NAX 主机必红，现跨机可跑） | tests/test_glm53_router_shared.c | PASS（240 poisoned draws） |

kernel 级回归全绿：`test-glm53-fork`（router_shared/topk/q8_inputs）、`test-glm-attention`、
`test-glm53-kda`。发现与裁决见 §2 DC-1 修正（GLM cycle 结构性 depth-1，`--mtp-draft` 静默 no-op）。

**未做（按纪律留给空机窗口）**：一切 bench/fixture/timeline 实测；MTP 经济性表的真实数字；
链式起草移植（等表出来再立项）。

### 开工日志（第二批，同编码日）

| 改动/发现 | 文件 | 验证 |
|---|---|---|
| PF-2 门臂：`DS4_METAL_ENABLE_GLM53_ROUTED_MPP_PACKED`（288/8/4096/2048 精确几何 pin，默认 OFF） | ds4_metal.m `glm53_top8_packed` | `make -j8` 零告警；fork 三件套重跑 PASS |
| DC-1.2 静态解剖 draft step（`glm_graph_mtp_step`）：一层完整 nextn 层（enorm/hnorm→eh_proj→q_a/q_b→kv→**全窗 dense attention（无 indexer，窗口=[min_pos..pos] 只随 decode 长度增长，有界）**→o_proj→router+top8 MoE+shared→head_norm→**全 vocab 输出头 + 600KB logits 回读 + host argmax（154880 项循环）**）；accept 路径连跑**两次** draft step，`--mtp-timing` 的 "head+draft 15 ms" 是 2 步合计（≈7.5 ms/步） | ds4.c 只读勘察 | 无代码改动 |
| DC-1.2 杠杆排序（静态）：①GPU argmax+单 int 回读替代全 vocab 回读+host argmax（Qwen `qwen4_mtp_draft_head_load` 有现成先例）②draft FFN 改吃 fused router/shared exact（#1090 家族，输出不变，draft 目前走 untuned 链）③selected 数组每步 malloc→静态复用。draft attention 无 top-k 问题（窗口有界） | — | 待经济性表定优先级 |
| 模型几何核实（GGUF 头解析，零内存压力）：46 块（45 trunk+1 nextn）/embd 4096/vocab 154880/experts 288 top-8 shared-1/expert FFN 2048/dense FFN 12288/indexer 32h×128d top_k 2048/KDA 64h×128d conv4/rope_dim=0/ctx 1M | gguf/GLM-5.3-Flash-Q2.gguf 头部 | PF-2 pin 与 DC-1.2 数字的事实来源 |

**空机窗口命令单（顺序执行，均 non-batched，腿间隔 ≥25s）**：
1. `ds4-bench … -c 12288`（decode 基线复采，tuning ON，对照 33.6-33.8）
2. `ds4-bench --mtp … --mtp-timing` 5-10k token → 读 session-free 经济性行定 MTP 去留
3. PF-2 A/B：`ds4-bench … -c 40960` prefill ON/OFF 各 2 轮 + `DS4_METAL_ENCODER_TIMELINE=1` 一次
4. prefill 分阶段：`DS4_GLM_INDEXED_PREFILL_TRACE=1 … -c 40960` 一次

### 开工日志（第三批：2026-10-03 执行窗口，21:36–22:03 CST）

机器全空窗（开跑前 74 GB 空闲、无 LLM 驻留），9+7 腿全部后台 job，swap 全程稳定 ≤2.8 GB、
free ≥17%，内存监视器每 10s 采样（`/tmp/glm53win/mem.log`），未触阈值。

| 腿 | 结果 | 判定 |
|---|---|---|
| PF-2 packed A/B @41k ×5 腿/侧（ABBA+） | OFF {489.6,449.8,489.0,493.3,496.7} / ON {428.8,476.1,465.2,509.1,518.1}，均值 -0.9% 区间重叠 | **判负关账**（engage 已由 timeline 证实，见 §PF-2） |
| MTP 经济性 `--mtp-timing` greedy 17k | 70.8% 接受、1.54 tok/cycle、27.5 vs 32.3 t/s | **净 -15%，MTP 默认关**（见 §DC-1） |
| decode 基线 12288/256 | 31.97/32.04 t/s | 噪声带内，无回归信号 |
| decode + timeline | f32 router matvec 6.53 ms/tok=墙钟 21%（新第一杠杆） | 见 §DC-2 |
| prefill staging trace @41k | host 侧全 <1ms，chunk=GPU drain | **staging 方向判负**（见上） |
| engaged 锚点 | 每条 prefill/decode 腿 ≥1 | 无假腿 |

**方法学记录**：① ds4-bench 对 prompt 按 ctx 截断，ds4 CLI 不截断（413k token 全文首腿即拒），
MTP 腿用 55 KB 切片（≈17k tok，意大利语文本 ≈3.2 B/tok）。② 编码器 timeline 自身 ~8% 开销
（DEC 腿 29.6 vs 32.0），**带 timeline 的腿不进 e2e 对比，只用于 kernel 分列**。③ 本机 e2e
噪声实测可达 ±8%（首几条腿偏慢，页缓存爬坡），单靠 ABBA 4 腿不够，PF-2 用了 5+5 才压出结论。
④ GGUF 解析两个坑（下次别踩）：ARRAY 值先读 **elem_type 再读 count**；tensor-info 条目是
name + ndim(u32) + dims(u64[]) + **type(u32)** + offset(u64)，type 在 dims 之后。
⑤ 全部原始数据在 `/tmp/glm53win/`（csv/log/timeline/mem.log），重启机器前可捞进仓库归档。

**下一步（P0→P2 重排后，2026-10-03 NR0 追加实验后更新）**：
1. ~~DC-2 零代码腿：`DS4_METAL_PLAIN_MV_NR0=2` decode A/B~~ **同日实测判负（见 §DC-2），移除**；
2. DC-1.1 链式起草移植编码（窗口外可做，验收仍靠经济性表）；
3. PF-1 KDA blocked 递推（7.1% @41k，线性涨，唯一还没关账的 prefill 增长项）；
4. **PF-6 已定位（窗口外编码候选）**：prefill router 投影走通用 f32 matmul——
   `ds4.c:52873` `ds4_gpu_matmul_f32_tensor(g->batch_router_logits, ffn_gate_inp, n_tokens)`，
   每 chunk×每 MoE 层一次，实测 6.3 ms/次（2048×288×4096 ≈ 2.4 GFLOP，6.3 ms = 0.76 TFLOP/s，
   对比调优 bf16 家族应有 ~1 ms）@41k 合计 6.5s = 7.3%。改走 `glm53_graph_matmul`
   调优家族（decode 侧同名函数已走融合 router；两路 logits 位域可能变，top-8 选择属行为项，
   验收 = 100-case@4096 + 长文 fixture）。收益上限：prefill +3.5~5%（短 prompt 也吃得到）。
