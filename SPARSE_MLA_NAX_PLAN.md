# GLM-5.3-Flash sparse MLA → tensor units（NAX）实施计划

> 立项依据：S0 实测（`DS4_METAL_ENCODER_TIMELINE` 采的 41k prefill encoder timeline）。GLM53 prefill 的
> `kernel_glm_attention_indexed_batch_lora_group8_vec` 在 **41k 上下文占 28.3%**
> （220 次调用 × 143.6 ms），是长上下文 prefill 的第一大项；在 2048 上下文也占 6.6%。
> 上游 oMLX 的对应项（#3986/#3996）用 tensor units 把同一段做到 2.8–3.3×。
> 本仓库已有可抄的模板：**#1149 的 `kernel_qwen4_attn_mm_nax`**（已并入 main）。

## 1. 现状（本仓库，已核对）

| 维度 | 值 | 证据 |
|---|---|---|
| 目标 kernel | `kernel_glm_attention_indexed_batch_lora_group8_vec_impl` | `metal/dsv4_misc.metal:3789` |
| 变体族 | `_vec` / `_vec_valid` / `_vec_valid_fullheads` / `_vec_causal{,_fullheads}` / `_vec_prefix_fullheads`(#1147) / `_group16_vec_fullheads`(#1090, heads_per_sg=2) | `ds4_metal.m:9018-9034`、`metal/dsv4_misc.metal` |
| 线程组 | 32×8 = 256 线程（8 simdgroups，**1 head/simdgroup**，group_heads=8） | `ds4_metal.m:38556` 附近 dispatch |
| 网格 | `(n_head/8, n_tokens)` | 同上 |
| 每 token 工作 | 遍历自己的 selection list（≤ `n_indexer_top_k`=2048 个 pooled 行），staging 16 行/步；每行每 head 做 `dot(float4)×4 + simd_sum`（ALU-bound） | kernel body |
| 输出 | `lora_out[token][head][512]`（fp32） | kernel body |
| 精度结构 | fp32 online softmax（m/l 每行），P 无 half 舍入 | kernel body |
| 几何 | 64 heads、qk_nope=256、latent 512、n_rot=0、kv_lora 512 | `ds4.c` DS4_SHAPE_GLM53 |
| 选择列表语义 | `selected[]` 可能含 `UINT32_MAX` 填充（尾部/内部空洞）；`last_indexer_guaranteed_prefix` 区分"前缀有效"与"全有效" | `ds4.c:55688-55695`、#1147 |

**为什么现在慢**：每个 (query, head, key) 都要一次 32-lane `simd_sum`，与 oMLX §1.5 诊断的
MLX vector SDPA 同构（ALU-bound，DRAM 带宽用不满）。

## 2. 模板（已在 main，可直接对照）

`kernel_qwen4_attn_mm_nax`（`metal/qwen4.metal:2590`，来自 #1149，源头 oMLX #4020）：

- 一 (kv head, token) 一 threadgroup，**2 simdgroups**；组的 query heads 填**一个 16 行张量瓦片**
  （`fm = (qid & 4) | ((lane >> 1) & 3)`，行 ≥ group 的槽位闲置）；
- head dim 256 劈成 128/128 给两个 simdgroup，partial scores 走 `threadgroup float xchg[2][8*32]`；
- key 循环按 query 自己的 selection list **32 keys/步**（dense 模式走因果位置），
  `16x32x32 matmul2d`：Q·Kᵀ 与 P·V（P 沿 K 拼两段）；
- **O 常驻 cooperative tensor** 跨整个 key 循环（无每瓦片 store/load），Q 每步从 device 重读，
  行最大值变化时才 rescale O；
- masked 槽位读 cache row 0（P=0 不产生 NaN），只有"越过列表尾部"和"越过 query"两种掩码；
- 标量 softmax（m/l 每行，P 舍入到 half）与 normalize + sigmoid-gate epilogue 照抄经典核；
- 门控：`n_tokens > 8 && head_dim == 256 && n_splits == 1 && ds4_gpu_mpp_available()`，
  `DS4_QWEN4_NO_ATTN_MM_NAX=1` 回退；短尾（T≤8）、key split、quality 模式、pre-M5 保持经典核。

**数值**：#1149 报 kernel 级 max |Δ| ≤ 1.3e-4（对经典核），e2e logits max |Δlogit| 1.14/0.91/1.68
@8k/32k/64k，**top-1 全同**；不是逐位一致（累加树不同）。

## 3. 数值契约（本仓库的取舍）

GLM53 目前有 **strict float32 逐位** 的验收基线（`compare_frontier_logits.py`，
`--tolerate-drift` 才允许漂移）。所以两条路：

- **A 路（推荐先做）**：接受 ulp 级漂移，照 #1149 的验收方式——
  kernel 级 `max|Δ| ≤ 4e-3*scale`（对 decode 路径）+ e2e `top-1 全同` + `max|Δlogit| < 2`
  （模型自身 chunking 漂移 0.85–1.4 的同一量级），并在 A/B 时用
  `--tolerate-drift`；GLM53 100-case 与 long fixture 必须落在参考带内
  （0.458030/89/7.37 与 long 的 0.668884120/6/2.500）。
- **B 路（若要保 bit-exact）**：把经典核的归约序逐位转录进张量路径——即用 `matmul2d` 但
  让每行 K 的累加顺序与经典 `dot(float4)+simd_sum` 树一致，代价是放弃大部分 tensor 收益。
  **不推荐**：oMLX 自己也没有做到（#3989 的教训是"融合核要跟着 eager 的算法走"，但那是
  sampler 对齐问题，不是 attention 累加树）。

先做 A 路；若质量闸门不过再退回经典核（fail-closed）。

## 4. 落点

1. **新 kernel**：`kernel_glm_attention_indexed_batch_lora_group8_nax`（`metal/dsv4_misc.metal`），
   按 GLM 几何实例化：
   - 16 行瓦片覆盖 **16 个 query head**（64 heads / 16 = 4 threadgroups/token），
     或保守先做 **8 heads + 半瓦片**（照 #1149 的 `row_ok` 处理闲置行）——先用后者降低风险；
   - latent 512 劈 2×256 或 4×128 给 simdgroups，partial scores 走 threadgroup 交换；
   - key 步长 32（selection list 2048 → 64 步/token）；
   - **必须处理的语义**：`UINT32_MAX` 填充（读 row 0 + P=0）、内部空洞、
     `_valid` 与 `_valid_fullheads` 的差异、`_causal` 变体（dense 前缀 2051 行）、
     `head_base` 偏移、RoPE 尾（GLM53 `n_rot=0`，但 GLM 5.2 有 64，需按 `qk_rope` 分支）。
2. **分派**：`ds4_metal.m` 在现有变体选择处加 NAX 分支，门控
   `ds4_gpu_mpp_available() && n_tokens >= 32 && cache_f16 && kv_lora_dim == 512 &&
   (qk_rope == 0 || qk_rope == 64) && !quality_mode`；pipeline 解析失败或首用对拍失败即
   永久回退经典核（fail-closed）。
3. **Kill switch**：`DS4_METAL_DISABLE_GLM53_MLA_NAX=1`；NAX 门同时纳入既有的
   `DS4_METAL_DISABLE_GLM53_PREFILL_INDEXED_ATTN`（indexed-attn A/B 基线腿）与分支级
   `DS4_METAL_DISABLE_GLM53_FLASH_TUNING`（一次性关掉所有 GLM 5.3 Flash 调优），保证
   请求经典基线的 A/B/回滚轮不会让 tensor tiles 抢跑。
4. **Engage 锚定**：仿 #1149 在 `g_test_flags` 加一位（如 `DS4_GPU_TEST_GLM53_MLA_NAX`），
   测试里断言新核真被派发（否则"测试通过"可能只是回退路径在跑）。

## 5. 验证阶梯（每步带闸门，不过就停）

| 步 | 动作 | 闸门 |
|---|---|---|
| V1 | `tests/test_glm_attention.c` 加 NAX-vs-classic 用例（2051/512/33/16/1 行、UINT32_MAX 尾部与内部空洞、head_base 偏移、causal/dense 前缀） | kernel 级 `max|Δ| ≤ 4e-3*scale`；T≤8 与 key-split 断言**逐位相等**（证明回退路由） |
| V2 | `make test-glm-attention` + `make test-metal-kernels` | 全绿 |
| V3 | GLM53 100-case fixture（FORCE 调优 + 新核） | avg_nll ≤ 0.459、first_match ≥ 89、avg_lcp ≥ 7.3（参考 0.458030/89/7.37） |
| V4 | GLM53 long 8-case fixture（8192 ctx，跨 2051 边界） | avg_nll ≤ 0.6695、first_match ≥ 6、avg_lcp ≥ 2.4（main 0.668884120/6/2.500） |
| V5 | frontier logits 2k..12k 对拍 | `--tolerate-drift` 下 top-1 全同、max|Δlogit| < 2（当前 main 是 strict identity，新核会破，需在报告里显式记录） |
| V6 | ds4-bench A/B（同 build，`DS4_METAL_DISABLE_GLM53_MLA_NAX` 双腿，ABBA，≥2 轮） | 41k 上下文 prefill 提升区间不重叠；kernel 级 ≥1.5× 才宣称 |

## 6. 预期与风险

- **预期**：上游 #3986/#3996 在同类 kernel 上拿到 2.8–3.3×；ds4 侧 attention 占 28.3% @41k，
  若 kernel 提 2×，端到端 prefill ≈ **+14%**（41k 域）；2048 上下文 ≈ +3%。
- **风险 1**：选择列表的 2048 槽里有大量 padding/空洞，张量瓦片对 masked 行是"白算"——
  oMLX #4020 的教训（并集在 8K 是 1.3×、64K 2.0× 的纯掩码浪费）要求按**升序块表**走，
  而不是按块并集。GLM 的 list 是 pool 行号，需要先确认其有序性/去重性。
- **风险 2**：`_valid` 语义（只有前缀 [0, indexer_top_k) 保证有效）在张量路径上要显式
  钳制，否则会读到未写过的 cache 行。
- **风险 3**：质量闸门若不过（NLL 超出参考带），按 A 路回退经典核，并把结论按域入档
  （"GLM53 MLA 在 M5 上 tensor-unit 化不可行/收益不足"），避免下个会话重复试。

## 7. 参考实现清单（都在本仓库，可直接对照）

- 模板：`metal/qwen4.metal:2590 kernel_qwen4_attn_mm_nax`（+ 其 host 侧门控与
  `tests/test_qwen4_kernels.c` 的 NAX-vs-classic 边界用例）
- 目标：`metal/dsv4_misc.metal:3789 kernel_glm_attention_indexed_batch_lora_group8_vec_impl`
- 既有 tensor-unit 索引/打分先例：`kernel_dsv4_indexer_scores_nax`（`dsv4_misc.metal:6587`）
- 测试入口：`tests/test_glm_attention.c`（`ds4_gpu_glm_attention_indexed_batch_lora_tensor`）、
  `make test-glm-attention`
- 质量与对拍：`gguf-tools/quality-testing/score_official`、
  `gguf-tools/quality-testing/compare_frontier_logits.py`

---

## 8. 实测结论（2026-09-30，M5 Max / 128 GB，`gguf/GLM-5.3-Flash-Q2.gguf`）

**结论：稀疏 MLA 已在 M5 tensor units 上跑通并默认启用（形状门 fail-closed +
`DS4_METAL_DISABLE_GLM53_MLA_NAX` 开关）。前提是 P 走 half+残差两趟；只把 Q/K/V
直接喂 half 的第一版过不了质量闸门。** kernel 级 **3.0–3.4×**，41k prefill 端到端
**+15.4%**，attention 占 41k GPU 时间从 **27.4% 降到 9.9%**。

### 8.1 落地形态与对计划的偏差

- kernel：`metal/dsv4_misc.metal` 里 `#ifdef DS4_METAL_HAS_TENSOR` 下的
  `glm_attention_indexed_batch_lora_group8_nax_body` + 四个入口
  `..._group8_nax{,_valid,_valid_fullheads,_prefix_fullheads}`（buffer 顺序与经典核一致）。
- 瓦片：一个 threadgroup = 4 个 simdgroup = **16 个 q head × 1 个 token**，
  512 latent（K 和 V 同一个缓冲）按 4×128 分给 4 个 simdgroup；每步 SK=32 个 key，
  `matmul2d 16×32×32`；每步 4 次 QK op + 8 次 PV op（P 双趟）。threadgroup 内存只剩
  4 KB 分数交换缓冲（经典核要 ~16 KB staging）。网格 `(head_count/16, n_tokens)`，
  线程组 `(32, 4, 1)`。
- **偏差 1（瓦片宽度）**：计划让 8 head/threadgroup 直接照抄 qwen4；实测按 **16 head**
  切才能让 64 头在 4 个 simdgroup 上全部瓦片饱满，且 latent 切 4 份正好喂 32 列 op。
- **偏差 2（V2 目标不存在）**：仓库里没有 `make test-metal-kernels` 这个 target；
  V2 用 `make test-glm-attention` + `make test-glm53-kda` 覆盖。
- **偏差 3（causal 不转）**：`slice_pos < dense_limit`（前 2051 位置）仍走经典 causal 核。
  41k 实测 11 次 × 0.7 ms 对比稀疏路径 209 次 × ~40 ms，占比 <1%，不值得。
- **偏差 4（`qk_rope==64` 不转）**：那是 GLM 5.2 形状；GLM-5.3-Flash 的
  `glm5-next.attention.rope_dimension_count = 0`，本模型不受影响。
- **偏差 5（多一个门）**：门里加了 `n_tokens >= 32`。短 prefill 尾部 threadgroup 太少，
  张量路径不划算，且质量闸门覆盖不到那段。

### 8.2 精度：误差源是 P，不是 Q（这一节是本次最贵的结论）

| 版本 | V1 kernel 级 max\|Δ\| vs 经典核 | V4 长文 avg_nll（基线 0.668884，门 ≤0.6695） |
|---|---|---|
| Q/K/V 直接 half（照抄 qwen4） | 1.22e-5（≈输出量级的 1e-3 相对） | 0.6725 **不过**（+0.53%，逐例 ±3% 双向摆动） |
| + Q 残差（hi/lo 两趟 QK，按 op 数估 +50% 时间，未单独测速） | 1.22e-5（**没变**） | 0.6725 **不过** → 证明 Q 的 half 舍入不是误差源，已回退 |
| + P 残差（hi/lo 两趟 PV，kernel 时间 27.9–37.4 → 38.2–42.1 ms/次） | **6.9e-8**（计划界 4.4e-5） | **0.665861 过**（8 例 6 例变好，first_match/greedy_lcp 8/8 不变） |

原因：K/V 本来就是 f16 cache（两条路径读同一份），Q 舍入到 half 在实测里几乎不贡献误差；
**P（softmax 权重）是这个 kernel 自己造出来的、经典核唯一保持 fp32 的量**，5e-4 的相对舍入
乘上 O(10) 的 latent 幅值、再经 out_proj 和 45 层放大就足够把 NLL 推 0.5%。
补上残差后：NAX vs CPU fp32 oracle = **1.7e-8**，NAX vs 经典核 = **6.9e-8**
（三角不等式 ⇒ 经典核 vs oracle ≥ 5.2e-8，也就是说这一档残余误差不再来自 NAX）。

### 8.3 闸门记分

| 闸门 | 结果 |
|---|---|
| V1 kernel 级 | **PASS** — `./tests/test_glm_attention` 10 个 NAX 用例全绿（padding 尾洞、内部空洞、全无效、partial step、48 头、rope=64 关门）；启用例 max\|Δ\| 6.9e-8，门 ≤4e-3·scale；三个关门用例与经典核 **逐位相同**（Δ=0）且 engage 位缺失 |
| V2 套件 | **PASS** — `make test-glm-attention`、`make test-glm53-kda`（后者 IA_TOKENS=8 → 门关闭，断言不受影响）；`make -j8` 干净 |
| V3 100 例 @4096 | **PASS（但是空腿）** — avg_nll **0.458177** ≤ 0.459、first_match **90** ≥ 89、avg_lcp **7.390** ≥ 7.3；日志 engagement 计数 **0** → 这批用例整轮没走稀疏路径，只证明短上下文无回退（详见 §8.6） |
| V4 长文 8 例 @8192 | **PASS** — avg_nll **0.665861** ≤ 0.6695，first_match **6** ≥ 6，avg_lcp **2.500** ≥ 2.4，日志有 engagement 行；同 fixture 的 scalar control 是 **0.672288**（见下） |
| V5 frontier logits 2k..12k | **破坏 strict identity（预期内，必须记录）**：comparer 状态 `drift`、finite 全覆盖、**top-1 6/6 全同**；整表 max\|Δlogit\| 最坏 6.22（限定 baseline top-20 内 ≤2.34） |
| V6 ds4-bench A/B（40960 prefill，ABBA，带 encoder timeline） | **达到宣称线** — kernel 级 **38.2/42.1 ms vs 127.5/128.2 ms = 3.0–3.4×**（门槛 1.5×）；端到端 **479.32 / 485.76 vs 416.07 / 419.93 t/s = +15.4%**，两腿区间不重叠 |

V5 的 "max|Δlogit| < 2" 这条界比仓库自己的容差还紧。按 QA 文档要求做 **scalar control**
（`--quality`，同一份数学只换算术顺序）对同一批 frontier 探针的漂移：

| 探针 | scalar control top-20 max\|Δ\| / 整表最坏 | 本次 NAX top-20 max\|Δ\| / 整表最坏 |
|---|---|---|
| 2048 | 0.77 / 1.63 | 0（该点 NAX 不启用） |
| 4096 | 1.56 / 3.65 | 1.92 / 1.98 |
| 6144 | 2.64 / **9.81** | 1.55 / 6.22 |
| 8192 | 0.65 / 1.46 | 2.34 / 6.10 |
| 10240 | 1.10 / 3.10，**top-1 翻转** | 2.03 / 2.62，top-1 保持 |
| 12288 | 2.33 / 2.79 | 2.17 / 3.41 |

即：这批 12k 拼接新闻探针的 logits 对"任何算术重排"都高度敏感，scalar control 漂得比
NAX 更多、还翻掉一个 top-1；NAX 在决策相关的指标（top-1、greedy lcp、fixture NLL）上
不劣于仓库已接受的变体。逐 case 的 V4 漂移是双向的（6 好 2 坏，first_match/lcp 全不变），
是噪声签名而不是系统性退化。

QA 要求的 **long-fixture scalar control**（同一 8 例、`--quality`）也跑了：
avg_nll **0.672288** / first_match 6 / avg_lcp 2.500。也就是仓库自己认可的标量控制路径
在这份 fixture 上就漂 **+0.51%**——和本次第一版未补偿 NAX（+0.53%）同一量级；
补偿后的 NAX 是 **-0.45%**。所以 V4 的门（≤0.6695）本质上是在问"你有没有比 scalar
control 更靠近 bit-exact 基线"，而答案是明确的是。

### 8.4 生产启用证据（跨会话锚点，下次别再重新发现）

- 一次性 stderr：`ds4: GLM53 prefill MLA engaged on tensor-unit tiles (<variant>)`。
  **跑 V3/V4/V5/V6 时必须 grep 这一行**，没有就是根本没上张量路径。
- 测试期另有 engage 位 `DS4_GPU_GLM53_PREFILL_INDEXED_ATTN_NAX`（只在 test-flag 模式记）。
- 想知道门为什么关：`DS4_METAL_DEBUG_GLM53_MLA_NAX=1` 会打印前 6 次派发的门控输入。
- ⚠️ **坑**：`gguf-tools/quality-testing/score_official` 不在默认 make 目标里。
  改完内核必须 `make gguf-tools/quality-testing/score_official` 重链，否则跑的是旧二进制。
  本次第一轮 V4 就是这样拿到一份"和基线一模一样但其实没启用 NAX"的假过。
- 单跑一次 fixture 前确认模型形状前提：GLM-5.3-Flash `rope_dimension_count=0`、
  `dense attention limit=2051`、`indexer_layers=11`（45 层里只有 11 层走这个核）。

### 8.5 还没做的

见 §8.7 移交清单（带优先级、判据和复测动作）。一句话版：风险 1（选择列表空洞白算）
原样存在；≤12k 端到端只有 +2~3%；decode 路径没动；Q 残差那条路已证明无用。

### 8.6 V3（100 例 @4096）

`summary cases=100 tokens=11559 avg_nll=0.458177271 first_match=90 avg_lcp=7.390`
—— 三项都在门内（≤0.459 / ≥89 / ≥7.3），对比记录的 Q2 参考
0.458030488 / 89 / 7.37：nll +0.000147、first_match 反而 90>89。

**但日志里 engagement 行计数为 0**：4096 上下文下这批 continuation 的
prompt+target 基本不跨 dense limit 2051，稀疏路径整轮没被调用，所以这一腿
**没有检验 NAX**，它的含义只是"没有把短上下文路径改坏"（那 +0.000147 的差来自跨机参考本身，
参考数是 M3 Ultra 上录的）。真正检验数值的是 V4（8 例全部跨 2051）和 V5。
下次要在 V3 上覆盖 NAX，得把 ctx 提到 ≥4096 且 prompt 跨过 2051——那就是 V4 那份 fixture。

---

## 9. 移交清单（下一轮优化的目标与动作）

排好优先级的靶子来自 NAX 之后重采的 41k prefill encoder timeline（GPU 合计 85.2s）。
**这张表就是决策依据**，别再凭猜：

| 占比 | 秒 | 次数 | kernel | 随上下文增长？ |
|---|---|---|---|---|
| 25.3% | 21.59 | 1680 | `kernel_mul_mm_id_iq2_xxs_f32_mpp` | 否（MoE 常数项） |
| 14.2% | 12.07 | 8340 | `kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128` | 否 |
| 12.1% | 10.31 | 840 | `kernel_mul_mm_id_q2_K_f16_mpp` | 否（MoE） |
| 10.3% | 8.80 | 209 | 本计划的 NAX MLA 核（**改前 27.4%**） | 是 |
| 6.8% | 5.76 | 1360 | `kernel_mul_mm_q4_K_f32_nax_direct_rhs_n128` | 否 |
| 6.7% | 5.68 | 680 | `kernel_glm53_kda_prefill_*` | 是（线性） |
| **4.7%** | 4.04 | 209 | **`kernel_glm_indexer_scores_tiled`** | **是（每 token 扫全部历史）** |

### P0 — `kernel_glm_indexer_scores_tiled` 张量化

- **为什么是它**：attention 被打下去之后，它是唯一"量级可观 + 随上下文增长"的项。
  它和 MLA 一样是每 token 扫全部历史位置（整体二次），ctx 越大占比越高；41k 已经 4.7%。
- **先例就在同一文件**：`kernel_dsv4_indexer_scores_nax`（`metal/dsv4_misc.metal:6587`）——
  DSV4 的打分核已经张量化过，形状门/降级/测试那套照搬即可。
- **动作**：完全复用本计划的流程（NAX 变体 + 形状门 + engage 位 + 双腿对拍测试 +
  V1/V4/V6 闸门），不要发明新流程。
- **⚠️ 这条的数值风险和本计划不同**：打分核的输出决定 top-k **选择集合**，分数只要边界
  平票就可能换掉被选中的行 → 那是语义变化，不是 ulp 漂移，V4 会直接红。所以开工第一步是
  kernel 级加一条 **"top-k 选中 id 集合与经典核一致率 100%"** 检查（比 max|Δscore| 有意义得多），
  集合一致率不到 100% 就不要往下跑 fixture。
- **判据**：41k 端到端 <+5% 才留（理论上界只有 ~4%，还要扣掉自己的开销；不到就不合算）。

### P1 — 选择列表压缩（计划风险 1，仍未做，但先测再写）

- 现状：每个 token 的 2048 个选择槽里，无效行（padding、内部空洞）照样进瓦片，只是
  `P=0` 白算；步进还是固定 `SK=32`。理论上"压成升序有效行紧凑列表 + `n_valid`，按
  `ceil(n_valid/SK)` 步进"能直接省掉白算的步数。`_valid` 系列已经保证了跳过语义，
  但**没有省步数**。
- **先测收益再动手**：长上下文里 padding 往往很少，真实收益完全取决于空洞率。
  先在 indexer 输出侧统计一次"无效槽占比"（或在 timeline 里加一次计数），**空洞率 <5%
  就把这项降级**，改成常数优化方向（例如 SK 从 32 提到 64 摊薄 softmax 与步进开销）。
  不要先写核再找收益。
- 若决定做：这是 buffer 契约变更（每 token 多一个 `n_valid`，选择列表要么紧凑存要么带
  偏移表），会同时影响 `_valid` 判定、`selected_rows_valid` 门、以及 §8.4 那条 engagement
  日志的语义，改动面比 P0 大。

### P2 — MoE 专家 GEMM（占比最大，但不随上下文增长）

- `mul_mm_id_iq2_xxs_f32_mpp`（25.3%）+ `mul_mm_id_q2_K_f16_mpp`（12.1%）是新的第一名，
  加上稠密侧 `..._nax_direct_rhs_n128` 两个变体合计 ~21%。
- 它是常数项：短上下文也照吃，收益对 TTFT 和 TPOT 都有效，从"每 token 成本"看性价比最高；
  但它不属于稀疏 MLA，**要单独立项**（本计划的门/闸门不通用）。

### P3 — 已经量化过、不值得再开

- causal 变体（前 2051 位置）：<1%（11 次 × 0.7ms vs 209 次 × ~40ms）。
- `qk_rope == 64`（GLM 5.2 形状）：GLM-5.3-Flash 是 `rope_dimension_count=0`，本模型用不到。
- 短 prefill 尾部（<32 token）：门已挡，见 §8.1 偏差 5。
- **Q 残差双趟**：V4 实测零效果并已回退，见 §8.2，别再试。

### 闸门定义的决策项（是决策，不是代码）

计划 §5 的 V5 界 `max|Δlogit| < 2` 已被实测证伪：仓库自己认可的 scalar control 在同一批
探针上漂到 **9.8** 且翻掉一个 top-1（见 §8.3）。建议改成三条，并且改之前要和仓库维护者确认：

1. top-1 必须逐位相同（0 容差）；
2. 限定在 baseline top-20 内的 `max|Δlogit|` 有界（建议 3）；
3. 漂移不得超过 scalar control 在同样探针上的漂移——即 `QA_BEFORE_RELEASES.md`
   里"compare the default path against the scalar control"从"也要跑"升级为**必跑并入库**。

另外：decode 路径（`..._indexed_decode_exact_*`）本计划完全没碰，服务端稳态 decode 与它无关；
要提 TPOT 得另开计划。

### 体感预期（免得下一轮又被问"怎么没变快"）

真实服务端请求（30k prompt + 2350 token 生成）里 prefill 只占 **44%** 墙钟时间，
本计划摊到整请求约 **+5%**。所以验证只看两个地方：**跨过 2051 之后每个 chunk 的边际 t/s**，
或者 encoder timeline。整请求平均 t/s 不是这项改动的观测口径。

### 复测最小动作集（避免重复踩坑）

```sh
make -j8 && make gguf-tools/quality-testing/score_official   # 后者不在默认目标里，必须显式重链
# 每个 fixture/bench 跑完都要：
grep -c engaged <log>        # 必须 ≥1，为 0 说明整轮没走张量路径，结果是假的
# bench 采样：ABBA 夹逼（NAX 腿夹住两条经典腿），单腿噪声 ±5%，区间不重叠才算增益
DS4_METAL_ENCODER_TIMELINE=/tmp/tl_x.txt ./ds4-bench ... --ctx-start 40960 --ctx-max 40960 --gen-tokens 0
```

两条腿之间留 ≥25s；`score_official` 连着跑会撞 memory guard（日志里
`memory already wired at startup`），等内存 free 回到 ≥85% 再跑下一轮。
不要用 100 例 @4096 那份 fixture 验证稀疏路径——整轮不启用，见 §8.6。
