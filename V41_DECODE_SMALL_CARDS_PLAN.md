# V4.1 流式 decode 小项打包方案（卡 C/B/E/F + 驻留权重融合的流式域准入）

> **立项背景（2026-10-12）**：契约内的大块主线已全部关闭——Metal 捕获基底缺失（M1 搁置）、
> 卡 I-1/I-2 的速度前提被上游撤回（[#1178](https://github.com/antirez/ds4/pull/1178) 于 2026-10-05 关闭；
> 热态复测 decode 均值 −0.8%、冷态 −31%，见 `PR1178_SESSION2_RESULTS.md`）、T3 逐层排空与 T2 未命中经用户裁决关闭、
> 投机解码整类 / NAX decode / 捕获路线关闭。本轮"继续探索"的裁决（用户，2026-10-12）：
> **走小项打包**，且**先出方案、窗口以后再定**。
>
> **诚实上限（先读）**：打包合计约 **+2~3% 墙钟**——卡 C ≤+1.5%（§14.6.3 改判）、卡 B/E/F 各 <1.5%（§14.6.4）、
> 融合重新准入未测但同族 §12.4 已证 ≈中性。本方案不追求大杠杆；价值在**可评审、可一次窗跑完、把 §14.3/§14.6.4 的卡片普查封档**。
> 墙钟真身（host 往返段 ~19 ms = 43%，§14.6.1/14.6.2）由卡 H（负结果）与 T3（用户裁决关闭）两条封住，**本方案不重开**。
>
> 本文档**零模型加载**（纸面 + 零模型套件设计）；所有加载类步骤集中在 §3 的一次窗口内。

---

## 1. 逐项方案

| 项 | 落点 | 改法 | 逐位论证 | 预期上限 | 回滚 env |
|---|---|---|---|---|---|
| **A1 卡 C** logits 508 MiB matvec tiling | `kernel_mul_mv_q8_0_f32`（`metal/dense.metal:210`）；dispatch 构造 `ds4_gpu_make_q8_0_mv_args`（`ds4_metal.m:5838`，nr0=2）/`ds4_gpu_make_q8_0_mv_dispatch`（`:5925`，nsg 默认 4） | **先纸面写 tiling 方案**（destination staging：129280 行目的缓冲按 U 形循环切瓦；单行 K 走序与 reduction tree 不变），再 worktree 移植为独立 kernel 变体 | tiling 只改目的缓冲写序与 threadgroup 布局，**逐行归约序不变** ⇒ 位等价；nsg/nr0/ILDG 扫描属诊断（nsg 改 K 分块 = 改归约序，必被 frontier logits 门抓住） | **+1.5%**（§14.6.3：1.36 ms/token = 3.1% 墙钟，即便 kernel 提速 50% 也只有 +1.5%） | 新增 `DS4_METAL_DISABLE_V41_LOGITS_TILING`（env 正开、默认关，验收后再议默认位） |
| **A2 卡 E** indexer topk fast 准入 | `ds4_metal.m:21395-21397`（`v41_fast` 走 `ds4_gpu_dsv41_exact_admitted` ⇒ 流式恒关）；kill env 已在树 | 准入改走 **`ds4_gpu_dsv41_activation_exact_admitted`**（`:12318`）——它**不读权重**（13-bit histogram + 候选分位数筛，纯激活），正合该函数注释的豁免语义；域开关复用 | 选择语义不变（同候选集、同 top-k 次序）；门 = `test_deepseek41_topk`（600 poisoned）+ `--verify-parity-ssd` + frontier logits | **~0.5%** decode（0.014→0.0014 ms/层） | `DS4_METAL_DISABLE_V41_TOPK_FAST=1`（既有） |
| **A3 卡 F** compute-copy 融合准入 | `ds4_metal.m:9948-9958`（`kernel_ds4_copy_words` 融合，~139 blits/token；准入走 exact ⇒ 流式恒关）；kill env 已在树 | 同 A2 改走 activation 准入（纯字节搬运、同序） | 字节与顺序不变（源码注释已写 "in the same order"）；门 = 零模型融合套件 + CLI 贪心逐字节 + frontier logits | **<1.5%**（§12.4 同族已证 ≈中性；仅当 cb 分解显示 encoder 边界占比意外才值） | `DS4_METAL_DISABLE_V41_COMPUTE_COPY=1`（既有） |
| **A4 驻留权重融合的流式域准入** hc_norm / shared_fusion / shared gate-up-swiglu | `hc_norm` `:54978`、`shared_fusion` `:55062`、shared gate/up/swiglu（`ds4.c:41870` mid-only + `ds4.c:41774` `ds41_shared_mid`）；均走 exact ⇒ 流式恒关 | **逐条、不许打包**：每条先跑零模型位等价套件 → 改走 activation 准入（或新设"驻留权重融合"准入）→ 路径门 → 一腿 ABBA | 这些融合读的是**静态映射的 dense 权重**（流式只 gather routed 专家），正合 activation 注释的豁免语义；但历史教训（p42-attn-glue 裸移植折叠未认证权重读路径 → logprob 24 红）要求**每条自带位证据 + 自带 kill env** | **未测**；上游 #1204 的 1.5×（2.1 vs 3.2 ms/40 层）是复用权重的 dispatch/激活往返数字，真实路径 DRAM 流量不变 ⇒ 估 <1.5% | `DS4_METAL_DISABLE_V41_HC_NORM` / `DS4_METAL_DISABLE_V41_SHARED_FUSION` + 域开关 `DS4_METAL_DISABLE_V41_STREAMING_ACTIVATION_FUSIONS` |

> **A4 的准入边界（写死，不许放宽）**：只放宽**读静态映射 dense 权重**的融合；凡读专家缓存槽（`use_iq2_selected_slots`/地址表/裸地址）的融合继续禁入流式域——
> 那条路径的标记缺口正是 #1178 两个静默 bug 的根因（`use_slabs()` 编码期快照 vs 执行期建 slab）。
> A4 每条改动**只允许一行准入函数替换 + 自带 kill env**，禁止顺带改默认位。

---

## 2. 闸门（照 §2.3/§2.5 全套；每条独立过）

1. **零模型**（窗口外可跑，但**会争 GPU**，需先问）：`make tests/test_deepseek41_fusions tests/test_deepseek41_metal tests/test_deepseek41_topk`
   → 三套全绿（含 `DS4_GPU_TEST_V41_SHARED_ROWS` 支、`DS4_GPU_TEST_V41_FUSIONS` 支；`test_deepseek41_fusions.c` 的 shared 段 = 120 全形状 Q8 用例含 NaN/Inf 臂）。
2. **CLI 贪心逐字节** ON/OFF（短 prompt + 128 token）。
3. `DS4_TEST_SSD_STREAMING=1 ./ds4_test --logprob-vectors` ON==OFF。
4. **`ds4-bench --dump-frontier-logits-dir`** 五 frontier（2048/10240/18432/26624/32768）**全量 logit 逐字节**（与 #1178 修复验收同一门）。
5. `test_metal_ssd_experts` 4/4；`--verify-parity-ssd` + `--short-prefill-ssd`（6000 B prompt）。
6. **ABBA ≥2 对** ×{2K, 32K} ×512 tok（差别预算 <±15% 时 4 对起步）；同窗采 `DS4_METAL_V41_DECODE_HOST_PROFILE=1`
   前后对照（`encodes/step`、`layers` 桶、`logits` 桶）——用于判断"省下的 dispatch 是否真落到墙钟"。

---

## 3. 窗口规程与腿单（**一次构建、一次跑完**）

- 前置：`ITERATION_PLAN_2026Q4.md` §2 全套（`pgrep` 无 ds4/ds4-server/ds4-bench/mlx/omlx 残留；swap <3 GB 且稳定；
  可用+purgeable ≥ 目标峰值 1.2×；**停生产 8055**；单实例 flock）。
- 构建：一个二进制（所有新路径 **env 正开、默认关**），一次建成跑完；禁止"跑 5 分钟→失败→改码→重跑"循环。
- 腿单（~12–16 腿，按 §2 的 ABBA 协议）：

| # | 腿 | 目的 |
|---|---|---|
| 0 | OFF 基线 ABBA ×2 对（2K/32K ×512） | 本窗基线 |
| 1 | A2 单开 | topk fast 的独立读数 |
| 2 | A3 单开 | compute-copy 的独立读数 |
| 3 | A1 单开（tiling 变体） | 卡 C 的真实上限 |
| 4 | A4 逐条（hc_norm / shared_fusion / shared gate-up） | 每条一腿，互不叠加 |
| 5 | 组合臂（仅当 1–4 各自 ≥0 才跑） | 打包总量 |
| 6 | profile 腿（OFF 与组合臂各一） | 归因 |

- 产物：`speed-bench/M5-CEBF/run-<时间戳>/`（命令、时间戳、CSV、量化、cold/warm、失败项）+ `notes.md`（方法学）。
- 预算：一窗 **~2.5–3 h**（零模型套件 10 min；闸门 25 min；ABBA 4×2 前沿 ~50 min；profile 20 min）；
  **A1 若需先纸面设计 + 新 kernel 编码，另计 1–2 天（窗口外）**。

---

## 4. 判定线与止损

- 每条：**逐位全绿 = 前置**；增益 <+0.5% → 记中性、默认关；≥+0.5% → 收账（**默认位仍待用户点头**）。
- 打包：合计 <+1% ⇒ 封档为"M5 流式 decode 在契约内已封顶"，回填 §14.6.4 表 + 台账。
- 止损：任一 DRIFT → 撤该条，**两次未愈即停线归因**（不许连改连跑）；任一 ABBA 腿绝对值掉出历史区间 ±15% → 中止并记录。

---

## 5. 风险与禁令

- **不碰** E0.3 红灯绑定面（T3 形状的登记红线）；**不动默认位**（测量窗内只开 env）。
- **不设 `DS4_METAL_Q8_MV_NSG`** 于 ABBA 腿：它会关掉 `parallel_full_ffn_eligible`（`ds4.c:24183`），且改 Q8 K 分块（改归约序）；诊断扫完必须 unset。
- **ABBA 腿禁设 `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`**：该探针自身把 id 回读变成强制路径（probe-observer 效应，§14.6.2）。
- A4 的准入放宽是本方案的**唯一政策动作**：需用户裁决 + 每条自带位证据；任何"裸移植"（p42 教训）直接否决。
- 加载类一律后台 job 盯内存；窗口前必查三样（可用内存/swap/驻留进程）。

---

## 6. 不做项（本方案边界）

- T3 排空（用户已关；唯一形状动 E0.3 红灯面）、T2 未命中（已关）、投机整类 / NAX decode / 捕获路线（已关）。
- 不改 cache 预算档位（用户已按流量定为 auto 7822；9000 档 = decode +6~8% 换整层 prefill −30~48%，见 M3-4 prefill 补账）。
- **不做 dense 重量化**（Q8→Q5_K/Q6_K）：估 +6~13% + 约 4 GiB cache 余量，但破逐位/黄金向量契约、需新内核 + 工具 + 质量重基线 ⇒ **另案、待产品裁决**（roadmap R8 "备案，默认不做"）。
- 不重开 `up/pr/1204`（已评估未取：与本地 `10b59ae` 同目标、内核名撞车、本机流式两条路都不执行）。

---

## 7. 台账联动与产出

- 完成后按 `LOCAL_INVENTORY.md` 纪律落行：A 段若涉及上游 PR 片按"评估未取/吸收"处置（#1204 勿重开）；
  D 段补本方案的四条判定。
- `V41_M5MAX_BASELINE_SPEED_PLAN.md` §14.6.4 表回填状态列（B/E/F/C 的实测读数）。
- `ITERATION_PLAN_2026Q4.md` M3 段可挂"卡 C/B/E/F 收尾"一行；M2 段的 I-2 已按 §3 M2 重锚注搁置。
- 若 A4 准入被用户否决：把"驻留权重融合在流式域"的否决证据写进台账，防止下个会话重开。
