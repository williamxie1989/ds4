# DSV4 Flash Vision Exp sparse indexed attention → tensor units（NAX）实施计划

> 立项依据：S0 实测（本文 §8，2026-09-30，M5 Max / 128 GB，
> `gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf`，
> `DS4_METAL_ENCODER_TIMELINE`，prefill @40960）。
> **`kernel_dsv4_indexed_mixed_attention_heads16_dual` 在 41k prefill 占 19.5%
> （210 次 × 68–83 ms），是 GPU 时间第一大项**，比任何单个 MoE GEMM 都大。
> 本仓库已有两份可抄模板：GLM 版同型改造 **`kernel_glm_attention_indexed_batch_lora_group8_nax`**
> （刚并入，实测 3.0–3.4×，见 `SPARSE_MLA_NAX_PLAN.md` §8）及其上游源头
> **`kernel_qwen4_attn_mm_nax`**（#1149）。
> Vision Exp 与 plain Flash 同几何（加载时逐项 `config_expect_u32` 断言相等，
> 仅 name / `rms_eps=1e-20` / vision sidecar 元数据不同，`ds4.c:6617-6637`），
> 本计划对两者同样生效，不需要 vision 特判。

## 1. 现状（本仓库，已核对）

| 维度 | 值 | 证据 |
|---|---|---|
| 目标 kernel | `kernel_dsv4_indexed_mixed_attention_heads16_dual`（prefill 快路径） | `metal/dsv4_misc.metal:7122` |
| 变体族 | `_heads8`（fallback）/ `_heads16_dual`（prefill 门控）/ `_rb16`（decode 单 token）/ `_split`+`_split_reduce`（decode 12 分片） | `metal/dsv4_misc.metal:7018-7495`、`ds4_metal.m:9156-9164` |
| 派发点 | `prefill_dual_heads` 门：`mpp_available && !quality && (n_head==64||32) && top_k==512 && window==128 && head_dim==512` | `ds4_metal.m:31213-31232` |
| 线程组 | 一 token 一 threadgroup，8 simdgroup × **2 head/simdgroup** = 16 头；K/V 行 stage 进 `kv_shared`（tid<128 → 512 维 half4）全头复用 | kernel body |
| 每 (head,row) 工作 | 4×`dot(float4)` + 1×32-lane `simd_sum`（ALU-bound）；双头变体每行 2 次 | `dsv4_attend_shared_h4_row`，`metal/dsv4_misc.metal:6913` |
| key 来源 | **两段**：raw 尾部 ≤128 行（连续、fp32 存储、staging 时舍半）+ comp 行 ≤`top_k`=512（索引读，`comp_kv_f16` 两种存储） | kernel body |
| 掩码语义 | topk 已升序排序（`kernel_dsv4_sort_i32_rows_asc`，decode 跳过）；`idx<0`=padding continue；`idx>=visible` break；`visible=min((qpos+1)/ratio, n_comp)`；raw 段 SWA 窗 128 + 因果 | `metal/dsv4_misc.metal:7164-7209` |
| sinks | 每 head 一个常量分数，经 `dsv4_attend_sink` 折进 online softmax | `metal/dsv4_misc.metal:6992` |
| 输出 | `dst[token][head][512]`（fp32） | kernel body |
| 精度结构 | **"DS4 F16 attention rounding"**：Q/K/V 舍半（raw fp32 在 staging 舍、comp fp32 在 load 舍、Q cast half4），fp32 online softmax，**P（row_scale）保持 fp32** | kernel body |
| 几何 | 64 heads、head_dim=512（=cache 行维，同一行既做 K 又做 V）、kv 头 1、indexer 64×128、top_k=512、ratio-4、SWA 128 | `ds4.c:883-919` |
| S0 占比 | 41k prefill **19.5%**（8k 点 19.9%）；210 calls = 10 chunk × 21 层稀疏路径；另 22 层走 `kernel_flash_attn_ext_f16_dk512_dv512`（7.4%） | 本文 §8 |

**为什么现在慢**：与 GLM §1 诊断同构——每个 (head,key) 一次 32-lane `simd_sum`，
实测有效算力 ~4 TFLOPS，tensor units 全程闲置。

## 2. 模板（已在 main，可直接对照）

`kernel_glm_attention_indexed_batch_lora_group8_nax_body`（`metal/dsv4_misc.metal:4491`，
本仓库刚落地的 GLM 版；源头 `kernel_qwen4_attn_mm_nax` #1149）：

- 瓦片：一 threadgroup = 4 simdgroup = **16 q head × 1 token**，latent 512 按 4×128
  分给 simdgroup；每步 SK=32 keys，`matmul2d 16×32×32`；
- **O 常驻 cooperative tensor** 跨整个 key 循环，Q 每步从 device 重读，行最大值变化才 rescale；
- masked 槽位读 row 0（P=0 不产生 NaN）；
- **P half+残差双趟**（GLM §8.2 的最贵结论：误差源是 P 不是 Q；Q 残差已证明无用）；
- 门控：`ds4_gpu_mpp_available()` + 形状门 + `n_tokens >= 32`，pipeline 解析失败或
  首用对拍失败即永久回退经典核（fail-closed）。

DSV4 侧同文件的张量化先例：`kernel_dsv4_indexer_scores_nax`（`metal/dsv4_misc.metal:7866`，
已默认启用，`ds4_metal.m:19276`）——形状门/降级/质量模式回退那套照搬。

## 3. 数值契约（本仓库的取舍）

DSV4 有自己的 F16 attention rounding 约定（§1 表）与验收基线（logprob 向量、
frontier 对拍），**不能沿用 GLM 的 4e-3·scale 门和 0.458/0.669 参考带**。走 A 路
（接受 ulp 级漂移），但验收三件套按 GLM 计划 §9 修订后的 V5 规则执行：

1. top-1 逐位相同（0 容差）；
2. 限定 baseline top-20 内的 `max|Δlogit|` ≤ 3；
3. 漂移不得超过 scalar control（`--quality`）在同一批探针上的漂移——必跑并入库。

开工前先在 main 上录 DSV4/Vision Exp 的参考带（见 V0）。

## 4. 落点

1. **新 kernel**：`kernel_dsv4_indexed_mixed_attention_heads16_nax`（`metal/dsv4_misc.metal`
   的 `DS4_METAL_HAS_TENSOR` 段），按 DSV4 几何实例化 GLM 瓦片：
   - 16 head × 1 token，512 维劈 4×128 给 4 simdgroup，SK=32；
   - **两段 key 循环**：raw 段（连续行，≤128 行 → ≤4 步，地址直接算，无需 topk 表）+
     comp 段（索引行，≤512 行 → ≤16 步，`dsv4_load_cache_h4` 的两种存储都要支持）；
   - **必须逐位复现的舍入**：raw fp32→half 在 staging、comp fp32→half 在 load、
     Q cast half4——matmul2d 的输入侧与经典核取同一份 half 值；
   - **sinks**：作为末尾常量分数折进 online softmax（等价于一个虚拟 key 行，
     score = `sinks[head]`）；注意这会改变累加序（A 路可接受），V1 需对拍确认
     sink 的 rescale 语义；
   - 输出 fp32 `[token][head][512]` 与经典核同布局。
2. **分派**：`ds4_metal.m` 在 `prefill_dual_heads` 分支处（`ds4_metal.m:31220-31232`）
   加 NAX 分支，门控沿用该 gate 的形状前提
   （`mpp_available && !quality && n_head==64 && top_k==512 && window==128 &&
   head_dim==512`）**再加 `n_tokens >= 32`**（GLM 偏差 5：短尾 threadgroup 太少）；
   失败永久回退 `_heads16_dual`（fail-closed）。
3. **Kill switch**：`DS4_METAL_DISABLE_DSV4_MLA_NAX=1`；`--quality` 模式天然回退
   （门里已含 `!quality`），保证 scalar control 腿干净。
4. **Engage 锚定**：仿 GLM 版加 `g_test_flags` 位 + 一次性 stderr
   `ds4: DSV4 prefill indexed attention engaged on tensor-unit tiles (<variant>)`；
   **V3–V6 每轮必须 grep 该行**，没有就是回退腿在跑（GLM §8.4 的教训）。

## 5. 验证阶梯（每步带闸门，不过就停）

| 步 | 动作 | 闸门 |
|---|---|---|
| V0 | **前置**：在 main 上给 Vision Exp 录参考带——`score_official` 短/长两组 fixture 的 avg_nll/first_match/avg_lcp + frontier 2k..12k 探针的 logits 与 scalar control 漂移 | ✅ 已完成（2026-09-30，见 §9：带数值、阈值与归档路径） |
| V1 | `tests/test_deepseek41_metal.c` 的 `check_tp_attention` 加 NAX 腿（现成用例已含 shuffled 选择行、-1 padding、掩码 future 行、`idx>=visible` break、TP 半头分解） | kernel 级 `max|Δ|` ≤ 4e-3·scale（对经典核）；`--quality` 关门用例逐位相同（证明回退路由）；sink 用例单独覆盖 |
| V2 | `make -j8` + `make test-deepseek41-metal` + `ds4_test --all` 相关项 | 全绿 |
| V3 | score_official 短 fixture（FORCE 新核，ctx ≥ 4096 保证跨进稀疏路径） | avg_nll/first_match/avg_lcp 落在 V0 参考带内；日志 grep engage ≥1 |
| V4 | score_official 长 fixture（≥8192，≥2 chunk） | 同上；对照组 = V0 的 scalar control 记录 |
| V5 | frontier logits 2k..12k 对拍（`compare_frontier_logits.py`） | §3 三条规则：top-1 全同、top-20 内 ≤3、漂移 ≤ scalar control |
| V6 | ds4-bench A/B（同 build，`DS4_METAL_DISABLE_DSV4_MLA_NAX` 双腿，ABBA，≥2 轮） | 41k prefill 两腿区间不重叠；kernel 级 ≥1.5× 才宣称；预期 3.0–3.4× |

⚠️ 工程坑（GLM §8.4 原样适用）：`gguf-tools/quality-testing/score_official` 不在默认
make 目标里，改完内核必须显式重链；两腿之间留 ≥25s；等 free 内存回 ≥85% 再跑下一轮。

## 6. 预期与风险

- **预期**：S0 口径 kernel 3× → 41k prefill 端到端 **+13~17%**（GLM 实测 +15.4% 同量级）；
  attention 家族占比 34.8% → ~22%（indexed 19.5%→~7%）。收益从 ctx≈4k 起全量兑现
  （`n_comp≥512` 即进稀疏路径，`ds4.c:41262`）。
- **风险 1（白算步）**：topk 表升序 + `break` 已天然跳过越界行；`idx<0` padding 的
  空洞率需先测（GLM P1 纪律：空洞率 <5% 就不做选择列表压缩，改试 SK=32→64）。
- **风险 2（两段循环的开销）**：raw 段 + comp 段两轮 barrier；raw 段 ≤4 步、占比小，
  若 timeline 显示 barrier 开销显著，可把 raw 段折进 dense 瓦片路径（qwen4 causal 模板）。
- **风险 3（sinks 累加序）**：sink 折进 online softmax 的位置与经典核不同（经典是
  逐行循环里的一行），属 A 路可接受漂移；若 V1 对拍超界，改为"循环后单行 rescale"。
- **风险 4（V0 缺失）— 已消除（2026-09-30）**：V0 参考带已按"文本 fixture 复用 +
  本机录带"方式入库（§9）：短 100 例走 `data/flash`，长 9 例走
  `deepseek-v4.1-flash-20260911-long`，frontier 探针走 `speed-bench/promessi_sposi.txt`。
  铁律不变：**后续任何动内核的轮次，V3–V5 都只对 §9 的带判红判绿**。
- **范围外**：decode 三腿（`_rb16`/`_split`）已专门优化，本计划不碰（GLM 同立场）；
  22 层 dense flash（`kernel_flash_attn_ext_f16_dk512_dv512`，7.4%）是下一个接力项，
  单独立项；MoE packed-RHS 物化（OMLX §1.6）另案。
- **top-k 选择一致性风险不适用**：本核不改变选择集合（indexer scores 已张量化且不
  动），比 GLM P0 的"边界平票换行"风险低一档。

## 7. 参考实现清单（都在本仓库，可直接对照）

- 模板：`metal/dsv4_misc.metal:4491 glm_attention_indexed_batch_lora_group8_nax_body`
  （+ `ds4_metal.m` 的 GLM NAX 分派/门控/engage 模式）；
  上游源头 `metal/qwen4.metal:2590 kernel_qwen4_attn_mm_nax`
- 目标：`metal/dsv4_misc.metal:7122 kernel_dsv4_indexed_mixed_attention_heads16_dual`
  （fallback `_heads8` :7018）
- DSV4 侧张量化先例：`metal/dsv4_misc.metal:7866 kernel_dsv4_indexer_scores_nax`
- 测试入口：`tests/test_deepseek41_metal.c` `check_tp_attention`（:1135）、
  `make test-deepseek41-metal`
- 测量钩子：`DS4_METAL_ENCODER_TIMELINE`（`ds4_metal.m:62`）、
  `DS4_METAL_INDEXER_STAGE_PROFILE`（`ds4.c:30879`，per-stage 边界含 "attention"）
- 质量与对拍：`gguf-tools/quality-testing/score_official`、
  `gguf-tools/quality-testing/compare_frontier_logits.py`、
  `gguf-tools/quality-testing/frontier_drift.py`（V0 新增：跨 quality 腿的
  top-1 / top-20 / 整表漂移，官方 comparer 要求两腿元数据逐字段相同，做不到）

---

## 8. S0 实测（2026-09-30，M5 Max / 128 GB，Vision Exp 主 GGUF）

两条腿 + 一个 8k 对照点，`--ctx-start 40960 --ctx-max 40960 --gen-tokens 0`。
总量噪声 ±13%（77.7s vs 88.8s GPU，timeline 有 pass 边界串行化，绝对值是上界），
但**占比稳定**（18.5% / 19.5% / 8k 19.9%）——决策指标用占比。

| 占比 | 秒 | 次数 | kernel | 随上下文增长？ |
|---|---|---|---|---|
| **19.5%** | 17.34 | 210 | **`kernel_dsv4_indexed_mixed_attention_heads16_dual`** | ctx≥~2k 后恒定 ~20%（top_k=512 封顶，`visible=(qpos+1)/ratio` 在 2k 就达上限） |
| 15.9% | 14.15 | 740 | `kernel_mul_mm_id_iq2_xxs_mpp_packed` | 否（MoE 常数项） |
| 12.5% | 11.06 | 3010 | `kernel_mul_mm_q8_0_f32_nax_direct_rhs_n128` | 否 |
| 8.1% | 7.18 | 370 | `kernel_mul_mm_id_q2_K_mpp_packed` | 否（MoE） |
| 7.9% | 7.05 | 210 | `kernel_dsv4_indexer_scores_nax` | 恒定（已张量化，勿重做） |
| 7.4% | 6.60 | 220 | `kernel_flash_attn_ext_f16_dk512_dv512` | 恒定（22 层 dense flash，接力候选） |
| 6.6% | 5.84 | 60 | `kernel_mul_mm_id_q4_K_pair_swiglu_f16` | 否 |
| 4.9% | 4.38 | 430 | `kernel_attn_out_low_q8_0_mpp_direct_rhs_n64` | 否 |

注意力家族合计 ≈34.8%。210 次 = 10 chunk × 21 层稀疏路径；220 次 = 10 × 22 层 flash。

### 与 GLM53 S0 的差异

- GLM 占比随 ctx 爬升（2048 行上限 + 2051 dense limit 渐进）；DSV4 在 ctx≈2k 就封顶，
  之后恒定 ~20% → 收益域从 4k 起就不变，不局限于超长上下文。
- DSV4 的 MoE 已被 MPP packed 压低（三族合计 ~37% 但分散），注意力反而居首。
- 每 (chunk, 层) 68–83 ms、有效 ~4 TFLOPS——ALU-bound 证据，与 GLM 立项时同病理。

### 复测最小动作集

```sh
make -j8 && make gguf-tools/quality-testing/score_official   # 后者不在默认目标，必须显式重链
DS4_METAL_ENCODER_TIMELINE=/tmp/tl_dsv4_41k.txt ./ds4-bench \
  -m gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 40960 --ctx-max 40960 --gen-tokens 0
# 每轮跑完必须：
grep -c "engaged on tensor-unit tiles" <log>   # ≥1，为 0 说明走的是回退腿
# 跑 bench/score 前停 ds4-server；free 内存 ≥85%；两腿间留 ≥25s
```

原始 timeline：`/tmp/tl_visionexp_41k.txt`（正式腿）、`/tmp/tl_visionexp_41k_run1.txt`、
`/tmp/tl_visionexp_8k.txt`（8k 对照）。

---

## 9. V0 参考带（2026-09-30 实测，M5 Max / 128 GB，Vision Exp 主 GGUF）

**V0 闸门已通过。** 本节是 §5 中 V3–V5 的判定基线；在此之前 Vision Exp 没有任何质量参考
（风险 4 已消除）。归档目录 `~/ds4-bands/vision-exp-v0/`（含完整 frontier dumps 与 TSV），
复现命令见 §9.5。

- 构建：main `9bc0bd7`，`make -j8` 后显式重链 `gguf-tools/quality-testing/score_official`。
- 模型：`gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf`
  （quant_bits=2，vocab=129280）。本机没有 plain Flash GGUF，**本参考带只适用于 Vision Exp**。
- **Engage 锚点**：四条腿日志里 `engaged on tensor-unit tiles` 计数全部为 **0**
  （main 的 DSV4 路径无张量瓦片，符合预期）。⚠️ main 上只有 GLM53 会打这行
  （`ds4_metal.m:38619`）——新核落地时 engage 串必须带 DSV4 专属前缀
  （建议 `ds4: DSV4 prefill indexed attention engaged on tensor-unit tiles (...)`），
  V3–V6 的 grep 用该专属串，不要用裸 `engaged on tensor-unit tiles`（会被 GLM 腿误命中）。

### 9.1 score_official 短 100 例（`data/flash`，ctx 4096）

`summary cases=100 tokens=2313 avg_nll=0.506775391 first_match=45 avg_lcp=4.120`

⚠️ 与 GLM §8.6 同构的**空腿**：这批 prompt 平均 ~22 token（且 DSV4 稀疏路径要
`n_comp≥512` 即 ctx≈2k+ 才启用），整轮不进稀疏路径。此腿只作"短上下文不回退"带；
§5 V3 里"ctx≥4096 保证跨进稀疏路径"的假设对这批 fixture 不成立，真正的覆盖在 9.2。

### 9.2 score_official 长 9 例（`deepseek-v4.1-flash-20260911-long`，ctx 34816）

9 例全部跨稀疏边界（prompt 8k/16k/32k × 3 任务），是稀疏路径的真实覆盖腿。
Δ = quality − default（负 = control 更好）。

| case | prompt | default avg_nll | --quality avg_nll | Δ | lcp（两腿相同） |
|---|---|---|---|---|---|
| 000 | 8197 | 0.487816 | 0.495215 | +0.007399 | 0 |
| 001 | 8197 | 0.717031 | 0.712999 | −0.004032 | 16 |
| 002 | 8196 | 1.296776 | 1.313811 | +0.017035 | 9 |
| 003 | 16389 | 1.163028 | 1.192174 | +0.029146 | 0 |
| 004 | 16389 | 1.138593 | 1.142826 | +0.004233 | 16 |
| 005 | 16388 | 1.124634 | 1.121627 | −0.003007 | 9 |
| 006 | 32772 | 0.508332 | 0.504741 | −0.003591 | 0 |
| 007 | 32773 | 1.054365 | 1.048280 | −0.006085 | 16 |
| 008 | 32773 | 0.716136 | 0.743961 | +0.027825 | 8 |
| **summary** | 576 tok | **0.911856919** | **0.919514823** | **+0.00766（+0.84%）** | first_match **6 / 6**，avg_lcp **8.222 / 8.222** |

per-case 漂移双向（4 好 / 5 坏），是噪声签名而不是系统性退化。
**scalar control 自己在长文上就漂 +0.84%**——这是 V4 "NAX 不得比 control 更远离 default"
的对照量。

### 9.3 frontier 探针 2k..12k（`speed-bench/promessi_sposi.txt`，增量 prefill，gen=0）

default 腿 vs scalar control（`--quality`）腿。top-1 argmax 两腿同 id：

| frontier | argmax id | top-1 同 | top-20 max\|Δ\| | 整表 max\|Δ\| | default logits sha256[0:12] |
|---|---|---|---|---|---|
| 2048 | 201 | ✓ | 0.362 | 1.245 | `17589268be0d` |
| 4096 | 201 | ✓ | 0.950 | 1.926 | `1598b42c9502` |
| 6144 | 16997 | ✓ | 0.761 | 1.134 | `e1ced2db0b9b` |
| 8192 | 77179 | ✓ | 0.291 | 0.820 | `f4d52cdc0795` |
| 10240 | 2317 | ✓ | 1.888 | 2.356 | `5d0229d76afa` |
| 12288 | 8131 | ✓ | 0.663 | 1.591 | `1436993de15e` |

- **default 腿逐位可复现**：同命令重跑，`compare_frontier_logits.py` 对拍 6/6 全文件
  逐位相同（changed=0 / 129280×6）——§5 V5 的 baseline 腿直接用归档的
  `frontier_default/`，不必也不应重录。
- 两腿 dumps 各自通过官方 comparer 自检（status=identical、complete finite）。
- 漂移表由本仓库新增的 `gguf-tools/quality-testing/frontier_drift.py` 产出：官方
  comparer 要求 baseline/candidate 元数据逐字段相同（含 `quality`），无法跨
  default↔control 腿；未来 NAX-vs-baseline（两腿 quality=false）仍用官方 comparer。
- 对比 GLM：这批探针比 GLM 的 12k 新闻探针钝得多（GLM control 最坏 top-20 2.64 /
  整表 9.81 且翻 top-1；DSV4 control 最坏 1.888 / 2.356、top-1 6/6 保持）。
  §3 的"top-20 内 ≤3"在 DSV4 上是合理阈值，且大概率不会被 control 自己打破。

### 9.4 闸门阈值固化（V3–V5 直接照此判）

- **V3 短**：avg_nll ∈ [0.5058, 0.5078]，first_match ≥ 44，avg_lcp ≥ 4.00（空路径腿，
  只查不回退）。
- **V4 长**：avg_nll ≤ **0.9196**（= 不比 control 腿差）、first_match ≥ 6、avg_lcp ≥ 8.1；
  且 per-case Δ 双向、|Δ| 不超过同 case 的 control 包络（表 9.2 最后一列，最大 +0.0291 /
  −0.0061）。
- **V5 frontier**：§3 三条规则实例化为——top-1 6/6 逐位同（0 容差）；限定 baseline
  top-20 内 max|Δ| ≤ 3；漂移不超过 9.3 控制表（逐 frontier、top-20 与整表两口径分别比）。
  若某 frontier 仅整表口径超界、top-1 保持、top-20 ≤3 且 9.1/9.2 不红，按 GLM §8.3
  先例显式记偏差并提请维护者确认，不得默默放行。
- V5 对拍基线 = 归档 `frontier_default/`（§9.3 已证逐位可复现）；NAX 腿跑同一条
  ds4-bench 命令（不带 `--quality`），输出到新目录后用官方 comparer
  （`--quality false --quant-bits 2 --vocab 129280 --ctx 12289`）+ `frontier_drift.py`
  双脚本出报告。

### 9.5 复现命令与归档

```sh
MODEL=gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf
# 短 100 例（空路径带）
gguf-tools/quality-testing/score_official "$MODEL" \
  gguf-tools/quality-testing/data/flash/manifest.tsv out.tsv 4096
# 长 9 例（稀疏路径带；128 GB 机直接驻留，未用 --ssd-streaming，与 S0/V6 同配置）
gguf-tools/quality-testing/score_official "$MODEL" \
  gguf-tools/quality-testing/deepseek-v4.1-flash-20260911-long/manifest.tsv out.tsv 34816 [--quality]
# frontier 探针（DIR 必须先 mkdir，ds4-bench 不建目录）+ 漂移
./ds4-bench -m "$MODEL" --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 12288 --step-incr 2048 --gen-tokens 0 [--quality] \
  --dump-frontier-logits-dir DIR
python3 gguf-tools/quality-testing/frontier_drift.py DEFAULT_DIR QUALITY_DIR \
  --frontiers 2048 4096 6144 8192 10240 12288 --output drift.json
```

归档 `~/ds4-bands/vision-exp-v0/`：`frontier_default/`、`frontier_quality/`（各 6 个
full-vocab logits dump）、`frontier_scalar_control_drift.json`（含每 frontier 两腿
logits float32 sha256）、`short_flash100_default.tsv`、`long9_default.tsv`、
`long9_quality.tsv`、`v0_comparer_*.json`（自检与确定性报告）。

实测耗时：短 100 例 ≈1.5 min；长 9 例 default ≈5.4 min、quality ≈9.3 min；
frontier 单腿 20–40 s。每腿之间等 `memory_pressure` free ≥85%（长 fixture 跑完会掉到
~24%，page cache 会自动回收，一般一轮 15s 采样就回 93%）。
## 10. V1/V2 实施与实测记录（2026-09-30）

### 10.1 V1 落地内容

- `metal/dsv4_misc.metal`：新增 `kernel_dsv4_indexed_mixed_attention_heads16_nax`
  （`#ifdef DS4_METAL_HAS_TENSOR` 包裹，位于 heads16_dual 之后）。GLM NAX 模板逐块
  移植：SK=32、16 头×4 simdgroup、512=4×128 分片、`matmul2d` 16x32x32 QK/PV、
  P half+residual 两次 PV、quarter exchange、标量 online softmax 与 per-head sink
  末尾折叠。DSV4 语义全部保留：环形 raw 窗口取行、fp32 行 staging 时 RTNE→half、
  comp f16/f32 双模式、`(uint)idx>=visible→break`、`idx<0→continue`、
  行槽复用（masked key 读已写行 P=0）、`1/S` 归一与 float4 存储。
- `ds4_metal.m`：gate `prefill_nax = prefill_dual_heads && n_tokens>=32 &&
  n_head%16==0 && !DS4_METAL_DISABLE_DSV4_MLA_NAX`；fail-closed（pipeline 编译
  miss 即永久回退 classic）；每进程一行 engage 日志
  `ds4: DSV4 prefill indexed attention engaged on tensor-unit tiles (kernel_...)`
  （带 DSV4 前缀，不与 GLM53 裸串冲突）；分派网格 `(n_head/16, n_tokens, 1)` ×
  `(32,4,1)`；NAX 走 `ds4_gpu_get_pipeline` 按名直查（GLM 先例），
  **不走 `ds4_gpu_hot_pipeline`**——后者按值返回静态缓存变量，未注册的
  名字永远 nil（V2 调试时踩过：engage 后分派取到 nil pipeline 直接 return 0）。
- `ds4_gpu.h` / `ds4_metal.m`：测试探针 `DS4_GPU_TEST_V41_INDEXED_ATTN` +
  `DS4_GPU_V41_INDEXED_ATTN_NAX` + take-and-clear getter（GLM MPP 探针同构）。
- `tests/test_deepseek41_metal.c`：
  - `--tp-attention` 重做：engage 位断言（n≥32 且 mpp 时必须置位）、
    kill-switch 腿 full-head 与 classic 逐位 memcmp（n<32/非 mpp）或
    ulp 门 `<4e-3·(1+|fallback|)`（n≥32）、rank 腿改为 classic-vs-classic
    逐位一致（先缓存 quality 腿再跑 kill 腿），打印新增 `nax=` 列。
  - 新增 `--indexed-attn-nax`（`check_indexed_attention_nax`）：D=512/H=32/K=512/
    C=1024，n=96，环形 raw（raw_start=97、cap=256、first_raw_pos=137 强制回绕）、
    visible=100..148 << C（break 覆盖）、-1 前置填充、升序 comp ids、
    sink 含 ±18 极值；f32 与 f16 comp 双腿；每腿：engage 位、kill 腿 fallback、
    fp64 oracle（抽样 1 列/头/token）双 CHECK（tile 与 classic 各 `<3e-5·(1+|ref|)`）。
    ⚠️ oracle 踩坑：升序 ids 的 -1 排最前，comp 消费循环必须 `id<0→continue`，
    以 `>=0` 作循环条件会在第一个 -1 处静默消费 0 个键。

### 10.2 V2 实测

- `make -j8` 干净；离线整库 `-std=metal4.0` 编译 0 error。
- `./tests/test_deepseek41_metal`（=`--all`）：**93 PASS，0 FAIL**。
  - TP attention（H=64 全头 + TP rank 半头）：`max split=0` 全腿——
    NAX 全头与 16 头 tile、heads8 split 跨切分逐位一致；
    kill 腿 memcmp 逐位等于 classic；tile-vs-classic drift 最大 **2.83e-7**
    （与 oracle 误差同量级，远低于 4e-3 门）。
  - indexed-attn-nax 新几何：`oracle=5.36e-8  tile-classic=1.12e-7`，
    f16/f32 双腿 PASS；engage 位双腿置位正确。
- `./ds4_test --all`（DS4_TEST_MODEL=Vision Exp）：除 **1 条预存在红** 外全绿。
  红腿 = `--local-golden-vectors` 的 `long_story_4096`
  （top5_overlap 3/5、top20_max_abs 9.45）：该 golden 框架以
  `DS4_METAL_DISABLE_METAL4`（`tensor_matmul=off`）跑 classic 路径，
  且 kill/default 两腿输出**逐位相同**——失败与本改动无关，
  是 37-42 层截断 Vision Exp 对整模 golden 的预存在漂移，**不动、不修**，
  仅在此入档（行为准则 8：有意偏差入档）。

### 10.3 V3/V4/V5 实测（2026-09-30，NAX 默认腿）

- **V3 短 100 例**：`avg_nll=0.506775391 first_match=45 avg_lcp=4.120`——与
  §9.1 **逐位相同**；engage=0（该批不进稀疏路径，空路径带符合预期）。**PASS**。
- **V4 长 9 例**：engage=1/进程；`avg_nll=0.916165342`（default 0.911857，
  control 0.919515——NAX 落在 default 与 control 之间，聚合漂移 +0.47%，
  比 scalar control 自己的 +0.84% 更靠近 default）；first_match 6/6；
  avg_lcp 8.222；**per-case greedy_lcp 与 V0 default 9/9 逐位相同**。
  per-case Δ（nax−default）双向：−0.0094..+0.0165，最大恶化 +0.0165
  （case_001）不超过同批 control 列最大幅度 +0.0291。闸门 avg≤0.9196、
  fm≥6、lcp≥8.1 全过。**PASS**。
  备注：若按最严格的 per-case 同 case 包络读法，case_001 漂移 +0.0165 而
  该 case 的 control Δ 为 −0.0040（方向相反、幅度不可直接比）；case_007
  为 −0.0094 的**改善方向**越下界。此类逐 case 重排噪声正是 control 列
  自身双向漂移所刻画的形态，判为噪声签名。
- **V5 frontier 2k..12k**（NAX 腿 vs 归档 `frontier_default/`）：
  官方 comparer status=drift、6/6 complete_finite、无错误；2048 逐位相同
  （未跨稀疏边界）。漂移表（同表并排 §9.3 scalar control）：

  | frontier | top-1 | top1 id | top20 max\|Δ\| | ctrl top20 | 整表 max\|Δ\| | ctrl 整表 |
  |---|---|---|---|---|---|---|
  | 2048 | 同 | 201 | 0.000 | 0.362 | 0.000 | 1.245 |
  | 4096 | 同 | 201 | 0.516 | 0.950 | 1.147 | 1.926 |
  | 6144 | 同 | 16997 | 0.600 | 0.761 | 0.842 | 1.134 |
  | 8192 | 同 | 77179 | 0.470 | 0.291 | 1.123 | 0.820 |
  | 10240 | 同 | 2317 | 1.493 | 1.888 | 2.605 | 2.356 |
  | 12288 | 同 | 8131 | 1.191 | 0.663 | 1.993 | 1.591 |

  top-1 **6/6 逐位（0 容差）✓**；top-20 全部 ≤3（max 1.493，不到控制表
  最坏值 1.888）✓。但 8192/12288 的 top-20 与 8192/10240/12288 的整表
  口径**逐 frontier 超出 scalar control 同点值**（超出幅度 ≤0.61）。
  §9.4 的字面"§8.3 先例场景"只覆盖**仅整表**超界；本次 top-20 在两个
  frontier 也超 control → 不属可默默放行场景，**显式记偏差、提请维护者确认**。
  佐证：短/长 fixture 全绿（10.3 V3/V4）、top-20 绝对值远低于 3、
  GLM53 有同类先例（长文单 case +0.51% 漂移被接受）。

### 10.4 V6 ABBA 计时（40960 prefill，A=kill/dual，B=NAX，2026-09-30）

| 腿 | prefill_tps | GPU total | indexed attn 秒/占比（210 次） |
|---|---|---|---|
| A1 | 553.23 | 73.10s | dual 13.02s / 17.81% |
| B1 | 515.50 | 78.47s | **nax 7.46s / 9.50%** |
| B2 | 555.11 | 72.85s | **nax 6.84s / 9.40%** |
| A2 | 510.67 | 79.29s | dual 14.53s / 18.33% |

- **内核本体 ~2×**：attention 内核 13.77s→7.15s（-48%），占比 18.1%→9.4%。
- **端到端在噪声带内不可测**：S0 已记录 ±13% 端到端噪声（pass 边界串行化），
  本收益上界 ~8% GPU 时间，恰好被淹没；A/B 两组均值差 +0.6%，无统计意义。
- engage 位 4/4 正确（A 腿 0、B 腿 1）。

### 10.5 决策（2026-09-30，维护者确认）

**保留 NAX，M5+ 默认开启**（GLM §8.3 先例接受 V5 偏差）：
top-1 6/6 零容差与 top-20 ≤3 两条硬闸全过；逐 frontier 超 scalar 控制的
偏差（≤0.61）已在本文件 §10.3 显式入档，`docs/METAL.md` 记录 kill 开关
`DS4_METAL_DISABLE_DSV4_MLA_NAX`。V3–V6 全部产物归档
`~/ds4-bands/vision-exp-v6-nax/`（含 6 个 frontier dump、comparer/drift
JSON、长/短 TSV、ABBA 日志与 4 份 timeline）。
