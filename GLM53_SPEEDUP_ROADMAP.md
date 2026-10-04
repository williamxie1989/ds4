# GLM-5.3 Flash（M5 Max 128 GB）：现状分析与下一步提速路线（2026-10-05）

> 范围：GLM-5.3-Flash-Q2（~96.5 GB，**本机全驻留**，非 ssd-streaming）× M5 Max 128 GB。
> 与 [V41_SSD_STREAMING_ROADMAP.md](V41_SSD_STREAMING_ROADMAP.md) 同套路：现状综合 + llama.cpp/vLLM/SGLang 调研 + 路线方案。
> **本轮零模型加载、零推理访问**（纯案头）；所有实测动作留给空机窗口。
>
> 源文档（细节证据都在那里，本文不重复推导）：
> [GLM53_SPEEDUP_NEXT_PLAN.md](GLM53_SPEEDUP_NEXT_PLAN.md)（PF/DC 全部裁决与协议）、
> [OMLX_WAVE_PORT_ANALYSIS.md](OMLX_WAVE_PORT_ANALYSIS.md)（五桶 S0、附二/附三）、
> [SPARSE_MLA_NAX_PLAN.md](SPARSE_MLA_NAX_PLAN.md)（MLA 上 NAX：27.4%→10.3% @41k）、
> [V41_M5MAX_BASELINE_SPEED_PLAN.md](V41_M5MAX_BASELINE_SPEED_PLAN.md) §16–17（卡 I，G1 的上游工程）、
> [LOCAL_INVENTORY.md](LOCAL_INVENTORY.md)。

---

## 0. 一页结论

- **GLM 的病与 V4.1 流式是两个域**：GLM GPU busy 96–99%（合并波后 12k 重测 87% + ~4.4 ms/token 空隙），
  host 编码被逐层 async flush 藏住——**调度/排队/减 dispatch 家族在这里没有可回收空间**（#1041/#1042 是 DeepSeek 的病）。
  prefill 提速 = 同样的算术让 GPU 干得更快（kernel 级）；decode 提速 = ①每 token 摊薄权重搬运（MTP 阵营），
  ②把 ~640 次/token 的 matvec 做大做少逼近 roofline（当前有效 ~500 GB/s）。
- **唯一一块"非 kernel"的肉是 decode 图回放空隙**：`ds4_gpu_decode_graphs_supported()` 目前 **只有 CUDA 实现**
  （ds4_cuda.cu:964），GLM 的 FFN-tail 逐层 island 接线在树里（ds4.c:58571+）却在这台 Mac 上永远返回不支持。
  卡 I-2 把 Metal graph capture 地基修好后，**GLM 驻留 decode 是最简单的首验收场景**（指针全静态、无专家槽、无 service 线程），
  预期把 4.4 ms 空隙+漂移的 host 段收回来：**decode 32→35–36 t/s（+10~14%）**，且为 MTP verify 摊薄解锁（DC-1.2 明说必须配套）。
- **MTP 判决（净亏 −15% @17k）不用推翻，但它的两个失效前提正在被拆除**：①起草头 16.2 ms 里一半是图外碎片
  （GPU argmax + 回读链 + 每层 CB 尾巴，DC-1.2 解剖已定位）；②verify/seed 没有图回放。
  外部佐证：llama.cpp 原生 MTP（PR #22673）72% 接受率 2.4×——CUDA 侧的配方正是"图回放让 verify 近似免费"。
  **G4 = 把起草侧砍半 + 回放就位，再用已写一半的 `--mtp-timing` 账本重测**；翻转点 = 金样接受率 ≥80%。
- **prefill 剩下的都是有界的 kernel 活**：PF-6 已兑现（router f32→BF16 tensor units，+7.3% @41k，上游 PR #1177 在途）；
  PF-2 packed 判负关闭；头号未关账项是 **PF-1 KDA blocked 递推**（7.1% @41k、全 ctx 通吃、线性涨；
  外部 FLA `chunk_kda`/vLLM FlashKDA 验证该形态是业界标准）。其后是 dense direct_rhs 26.3% 的 tile 扫描、MLA 残段微调。
- **新增两块案头才看得见的板**：①**M5 tensor-API 寻址审计**（llama.cpp upstream #28748：≥2 GiB slice 偏移寻址错——
  我们 NAX 内核全部在 96.5 GiB mmap 大 buffer 上按 offset 取 slice，S11 修过同类雷，需要一次系统审计 + ≥2 GiB 偏移回归）；
  ②**KV 内存制度**（FP32 无量化，128k 全注意力头 ≈135 GB > 本机 RAM——≥96k 域的一切收益都要先过 KV Q8_0/refit 关）。

---

## 1. 现状快照（锚定数字）

### 1.1 稳态

| 指标 | 数字 | 出处 |
|---|---|---|
| prefill 5.2k / 41k | 514 / 480–486 t/s（合并波 +19~23% 后） | OMLX 附二、NEXT_PLAN 头部 |
| decode 短 ctx / 12k | 38–39 / 32.0 t/s（10-03 复测；短 ctx 旧法高估 4–6%） | NEXT_PLAN §DC-2 |
| decode GPU 构成 @12k | busy 87%；q8_0 matvec 4.8 + MoE 4.1 + q4_K 2.7 + HC 2.6 + KDA 2.4 + router 2.1 + q2_K 2.0 + expand4 2.4 + LM head 1.16 | DC-2 五桶复测 |
| 权重流量 roofline | 每 token ~454 MiB/层 ⇒ 24 t/s 口径 + 30–40% 批内摊薄 | DC-2 NR-1 |
| MTP | 默认 OFF（接受率 70.8%、cycle ~2.0× 单步 ⇒ 净 −15%@17k；+4%@0/−21%@17k 按 ctx 变号） | DC-1 |
| 长 ctx 参考 | indexer 打分 41k=35 s、128k=171 s（二次趋势，@128k 占 64%） | PF-3 |

### 1.2 已关账（本文不再开题）

PF-2 packed-RHS（判负，默认关臂保留）、NR0 通用 nr0（判负）、NR-1 全内核行复用（模型否决：桶收益≈0.5~2.5% 低于门槛）、
Q 残差 FMA 折叠 / HC 两比例 / causal view / batch=1 ns=1 式 K-split / rope 循环 K 复用（六位点等价改造全负——**残差逐位必然改拓扑，判死**）、
MTP 深度 1/2/3 现行机制（关闭；per-sequence 批量是另一题）、#1127（GLM 短 prompt −19%）、
`--mtp 0` 解析陷阱（SR-1，脚本一律用裸 `--mtp`）。

### 1.3 在途与待还账

| 项 | 状态 | 处置 |
|---|---|---|
| PF-6 router→BF16 tensor units | **已实现**（d09ca3b，+7.3% @41k）；上游 [PR #1177](https://github.com/antirez/ds4/pull/1177) OPEN | 等 review；GLM 域本机关账 |
| 主树 WIP：`--mtp-timing` GLM 经济账本 + indexed-prefill trace env | 半成品（session 计数 + 结算打印已写） | **G4 第一子项**：补完、窗口内落账，然后决定去留 |
| `tests/test_glm53_router_shared.c` 未提交小补丁 | 未提交 | 随手提交（附三已证本机转绿） |
| 长 fixture / 100-case / 2k manifest | **已采齐**（阶段 A 完成，sr3） | 锁为数值总门；G6 引用 |

---

## 2. 外部调研：谁在走哪条路，与我们的差集

### 2.1 图回放 / host 开销（对应 G1）

- **llama.cpp CUDA Graphs**（[NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)）：
  bs=1 decode 默认开、图段执行 −40%、动态标量 staging buffer 化——I-2 已按此设计；GLM 驻留域连"动态"都更少。
- **vLLM V1**（[CUDAGraph in V1](https://discuss.vllm.ai/t/cudagraph-in-v1/1016)、[V1 架构](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture)）：
  piecewise（动态 op 切图外）与 full-graph（纯 decode 批 + paged 间接寻址）都已是量产形态；spec decode 的 verify 也吃 full-graph
  （llama.cpp MTP 线 [PR #24261](https://github.com/ggml-org/llama.cpp/pull/24261) 为 verify 修 cudagraph、重排采样防 UAF——**verify 摊薄的外部配方**）。
- **ggml-metal 现状**（[c363256](https://github.com/ggml-org/llama.cpp/commit/c363256839fdffae184279ad8e5f0f3775c48b77)）：MTLGraph capture env 化、默认实验——Metal 侧没人做成，卡 I-2 做成即差异化。

### 2.2 MTP / 投机（对应 G4）

- **llama.cpp 原生 MTP**（[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)，2026-05 合并）：
  draft head 随 GGUF 自带、自持 KV；72.18% 接受率下 draft×3 = **2.4×**、draft×2/82.58% = 2.2×（Qwen3.6-27B，CUDA+graph replay）。
  后续 GLM4MoE/DeepSeek2 的 MTP KV 分配修复（b10907）——方向成熟、有先例可抄语义。
- **Ollama v0.32.6**：MTP 自动检测，Apple Silicon 走 **MLX runner**——Mac 上投机可行性的直接旁证。
- **ngram 系起草**（llama.cpp `--spec-type ngram-simple/ngram-mod`，ngram-mod 为 MoE 推荐；server 级共享 hash 池）：
  **零起草成本**（无 draft head、无 seed cycle），只付 verify k 行；实测随流量方差大（重复 prompt 放大、多样流量归零）。
- **DSpark**（arXiv 2607.05147；[vLLM 解读](https://www.spheron.network/blog/dspark-explained-deepseek-s-new-speculative-decoding-in-vllm)）：
  置信度调度 + 并行半自回归起草，vLLM 实现 = 复用稀疏注意力后端 + 单 graph 圈住 draft-verify 环。
  我们的裁决不变（V4.1 流式域验证成本 ≥ 整步 ⇒ 否决）；对 GLM 驻留域，它是 **G4b 翻正后的 adaptive-depth 选项**，且 GLM53/DSpark 资产"未验证"一笔仍在账上。

### 2.3 注意力/内核（对应 G2/G3）

- **KDA（Kimi Delta Attention）**：[vLLM Kimi K3 博文](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)——
  prefill 收编 [FlashKDA](https://github.com/QwenLM/FlashKDA)/FLA `chunk_kda`；**decode 用 conv+状态更新+门+norm 的单融合核**
  （原话：KDA 层多，每层一个小 launch/内存税很快变成 TPOT 大税）；hybrid（递归态+paged KV）prefix 缓存细粒度命中。
  对我们的意义：PF-1 的 blocked 形态 = 业界标准（omlx #3984 供体继续有效）；**decode 侧 KDA 融合是新的小项**；
  conv/recurrent 双态一致性 = 业界也当验收点（SGLang KDA issue 的验收式 `prefill(T) == prefill(T−k)+decode(k)` 与我们 PF-1 闸门同款）。
- **DSA/indexer**（[tensoreconomics 深拆](https://www.tensoreconomics.com/p/deepseek-sparse-attention-from-first)、vLLM DSA 线程）：
  decode 域稀疏注意力定成本（top-k 2048），indexer 全序列 O(S) 但每 token 只读 132 B——**长 ctx decode 的注意力成本结构与我们一致**；
  GLM-5.2/5.3 模型侧自带 **IndexShare**（full→shared→shared→shared，1 个 indexer 喂 4 层；量化器已按此归主，
  `gguf-tools/glm53_full_quantize.py:indexer_owner`），所以 PF-3 的 indexer 二次增长是"共享后的残量"，长 ctx 才疼。
- **M5 tensor API 的社区雷区**（[modelfit 综述](https://modelfit.io/blog/m5-mac-metal-tensor-api-llama-cpp-fix)）：
  upstream **#28748**：`kernel_mul_mm` 在 slice−base 偏移 ≥2 GiB 时算错地址（M5，修复审中）；#27473/PR #27461 languageVersion=4.0 修复；
  Ollama #15594 cooperative tensor 类型断言。我们已经在 NAX 域压了 26.3%（direct_rhs）+17%+（MoE）+10.3%（MLA）的时间在 96.5 GiB 大 buffer 上
  ——**必须做一次同型审计**（S11 tile 寻址越界就是前车之鉴）。
- **llama.cpp Metal 近况**（[release 流](https://freedom.tech/project/llama-cpp)）：0.5.0 带 "Metal MoE/SSM fusion"、
  b9952 DeepSeek V4 f16 KQ mask/去冗余 cache repeat——方向参考，码不互通。
- **KTransformers**（[官网](https://kvcache-ai.github.io/ktransformers)：GLM-5 Day-0、SOSP'25 Expert Deferral 5+3 拆分）：
  CPU 混合线不适用驻留 Mac；deferral 思想在 V4.1 路线 R6 已挂账，GLM 驻留域 GPU 96%+ 没有 overlap 空间，**不适用**。

### 2.4 调研总结论

GLM 域外部给不了"整搬"的方案（我们 prefill 内核已在本机口径领先公开 Metal 栈半代），给的是三样：
①**图回放的工程配方与收益锚**（G1）；②**MTP 经济翻正的标准配方 = 起草砍半 + verify 回放化**（G4），外加 ngram 这个零成本草案源；
③**两枚免费的板**：KDA decode 融合核（业界当季做法）与 M5 tensor-API 大偏移审计（别人替我们踩过的雷）。

---

## 3. 路线方案（G1–G6，各带落点/闸门/预期）

### G1 · decode 图回放（主线；与卡 I-2 并线，GLM 是首个验收域）

- **事实基**：island 机制 + GLM FFN-tail 接线在树（ds4.c:58571+，`glm53_graph_encode_ffn_tail_one`）；
  `ds4_gpu_decode_graphs_supported()` 仅 CUDA 有实现（ds4_cuda.cu:964）、Metal 返回 0 → 本机零收益。
- **做法**：把卡 I-2 的 Metal 地基（MTLGraph capture、keyed 图缓存、动态标量 staging、`useResource` 规则、flush/CB 轮换三铁律）
  按"**驻留单卡 GLM 先、流式 V4.1 后**"的顺序落地：GLM 域无路由回读、无专家槽、无 service 线程，是地基的最纯净试验场；
  GLM 侧从 FFN-tail island 扩到整层（attention/KDA/router 各岛），全部静态指针。
- **收益假设**：回收 4.4 ms 空隙 + host 漂移 → **decode +10~14%**（32→35–36）；顺带兑现 MTP verify 摊薄（G4 前置）
  与 V4.1 卡 I-2 的复用。上游 PR #1178（I-1）的评审结论会反向校准这里的实现细节。
- **闸门**：GLM53 100-case + 长 fixture + frontier logits identical（比 V4.1 门更轻，因为指针全静态）→ clean ABBA ×{短,12k}。
- **风险**：capture scope 与 async flush 的关系（GLM 现依赖 flush 藏 host）；图缓存 keyed 于形状、ctx 变更要 retired 重建；
  MTLGraph 与共享事件/定时器未见公开先例——**保底交付 = 每层一岛的 piecewise**，全图 I-3 另案。

### G2 · prefill kernel 续耕（GPU-bound 域的正路）

- **G2a · PF-1 KDA blocked 递推**（头号未关账，优先级不变）：
  供体 omlx #3984（1 simdgroup/值行 → 每线程 2 行×16 通道 fp32 态 + 16-token 块预取，~2.3×/层；per-core 二期）；
  外部验证 FLA `chunk_kda`/FlashKDA 同形态。预期 **+2.5~4%（全 ctx 通吃、越长越值）**。
  独有闸门：conv_state/recurrent_state 跨 chunk 跨 session 一致性 + `prefill(T)==prefill(T−k)+decode(k)` 对拍。半天到一天。
- **G2b · MoE 主件收尾（37.4% @41k，packed 判负后只留两条）**：
  ①tile 形 vs 真实专家分布扫描（top-8×288 ⇒ ~57 行/专家的低并行段，28/64/128 档实例 + `*_packed` 判别日志锚）；
  ②上游 "Metal MoE/SSM fusion"（llama.cpp 0.5.0）思路对照。限时各一个窗口，负即收（PF-2 的诚实门槛照用）。
- **G2c · dense direct_rhs 26.3% tile 扫描**：`mul_mm_q8_0/q4_K_f32_nax_direct_rhs_n128` 的 n256/2m 分块可用性一次 A/B；
  n256 系已有 13% 打平先例——纯便宜实验。
- **G2d · MLA 残段（10.3%）**：group_vec m128 21.7ms 距版本A参考 14.6 ms；`SK32→64`/residual 微调，排 G2a 后。
- **G2e · 维持 deferred**：PF-3 indexer NAX（**仅当出现 ≥64k 生产域**）、PF-4 wide 8192（先量 `write_kv`/`select_ids` @1024 两前置）、
  PF-5 空洞率（30–60 min profile 决定立项与否）。
- **G2f · M5 tensor-API 大偏移审计（新，正确性优先）**：
  枚举全部 NAX 内核（direct_rhs/mpp/packed/mla/indexer）的 rhs slice 地址构造，对照 upstream #28748 的 ≥2 GiB 偏移模式逐一核查；
  给每个 family 补一条 **offset ≥ 2 GiB 的单元测试**（`test_metal_nax_tensor_units.c` 有框架）；
  发现同型问题按 S11 先例修 + 回归向量。产出物即使"全部清白"也入档（一劳永逸排除长 ctx 错数类鬼故事）。

### G3 · decode kernel 小刀（respect roofline，不碰已判死的方向）

- **G3a · q8_0 matvec 特化一次**：4.8 ms@12k、桶内 ~55% roofline（全局 87%）——参照 #1120 先例
  （sum6 2-simdgroup Q8_0 形）对 `K=4096 rows=1 Q8_0` 做一次 shape 特化，限时一个窗口，目标该核 +30%（decode +1.3%），
  <20% 即停。此后 kernel 级 decode 按"桶占比 >40% roofline"纪律收兵。
- **G3b · KDA decode 融合核**：vLLM Kimi K3 同款——conv+递归更新+输出门+norm 并进 `glm53_kda_decode` 单核
  （现在 decode 上 KDA 段多核拆分，KDA output 2.4 ms@12k）；顺带就是 G4b 起草步的种子融合件（E1 fused seed 同一把刀）。
  预期 decode +2~3%；数值走标准门（KDA 状态一致性测式已存在）。
- **G3c · 长 ctx 注意力（≥64k）**：先复采（S8 五桶在 NAX 域 + 调优家族下的 ≥64k 版）；
  若注意力桶 <40% roofline 再上 vec 核改造/K-split（`decode_attention_scan_vec` 复活评估）；
  indexer decode 成本是 O(S) 定值（外部结构同 DeepSeek），不是本期靶子。

### G4 · MTP 第二春：前提补齐包（不是重开，是"把判负的两个前提拆掉再测"）

- **G4a · 经济账本关账**（半天）：主树 `--mtp-timing` WIP 补完（verify/plain/head-draft 三段 + 接受率 rollup 已写一半），
  窗口内落一版决策表——DC-1 的 ctx 变号规律（0k +4% / 17k −21%）从此有持续可观测性。
- **G4b · 起草头砍半**（DC-1.2 的解剖照做，全部已定位）：
  ①CPU argmax + 逐 token logits 回读 → **GPU argmax，回读只留一个 int**；
  ②`ffn_one`/`router_one` 现绑 prefill 融合版 → **给 draft 绑专属 fused-FFN 内核**（字母表换一下的事）；
  ③第 46 层 selected/ids/bias 每步 cudaMemcpyAsync 重灌 → **常驻 only 下"写一次长期有效"静态数组**（门控：驻留 only，streaming/共享一律不开）；
  ④conv+递归+gate+norm 种子融合 = 复用 G3b 产出。
  **起草 16.2→~8–9 ms/cycle**；verify/seed 侧的回放摊薄 = G1 产出。
- **G4c · 复测翻转判定**（G1+G4b 都齐后）：G4a 账本 + 金样 manifest 重跑经济表；
  翻转条件 = 工作流量接受率 ≥80%（金样 76.3–86.5% 区间）且 cycle 成本 <1.35× 单步。翻正才谈 DC-1.1 链式起草（2 头共享权重串行，省 7ms/草稿）；
  再谈 DSpark 式置信度自适应深度（GLM53/DSpark 资产届时一并验证）。
- **G4d · ngram 自起草（新，旁路小项）**：llama.cpp `ngram-mod` 同型（上下文 hash 匹配，MoE 推荐形）；
  **零起草成本、零新权重**，只付 verify k 行——tool/JSON/代码类高重复流量可能白捡 +5~15%，多样流量趋零（外部实测方差大）。
  实现 = sampler 挂钩 + 跨请求 hash 池（几百行）；验收直接用 G4a 账本 + 现网流量影子测；负即弃。
- **纪律**：`--mtp` 一律裸写（SR-1 陷阱）；采样率类结论必须显式标 ctx 档位（DC-1 的 0k/17k 变号教训）。

### G5 · ≥96k 域前置：KV 内存制度（新，域决策）

- 现状：KV FP32 无量化；全注意力 KV 线性涨，128k ≈ **135 GB > 本机 RAM**（DC-1.2 实测口径）。
  128k 域的一切（PF-3 二次项、PF-4、长 ctx 复采）都建在"KV 放得下"上。
- 做法：**KV Q8_0 路在 GLM53 上过门**（kill 已有、refit 门已修）+ indexer 桶优化一起做（kill 状态尚未弄清）；
  验收 = 长 fixture identical 或漂移入账 + frontier logits 过门 + 128k 冒烟内存表。
- 产出：一份"本机能跑的 ctx 档位表"（FP32 ≤96k / Q8 ≤~220k 的口径落成文档），之后 PF-3/PF-4 的立项申请必须引用它。

### G6 · 回归与口径（随各刀携带）

- 数值总门已锁：**100-case@4096（3 次批量）+ 2k manifest（52 min×2）+ 长 fixture manifest（10 min×2）**；
  top-8 选择/权重/量化级改动**必过全套**，只动 attention/indexer 的只需长 fixture（PF-2 条款 5 推广为通则）。
- 计时三坑 + 单变量两则照抄（NEXT_PLAN §1.3）：共享机 ±30%、冷启动、ABBA×2 起步；PF-2 教训加一条——
  **差别小于 ±15% 时配对数不足会把中位差洗成 0**。
- 执行窗口铁律不变：加载类后台 job、预检（swap/pgrep/preflight）、测量要停其它 runtime、单实例 flock。

---

## 4. 执行顺序与依赖

```
0) 还账：router_shared 测试补丁提交；--mtp-timing WIP 定性（完/弃）
1) G1（与 V4.1 卡 I-2 并线，GLM 驻留域为地基首验收）——主线
2) 穿插（互不冲突）：G2a（KDA blocked）、G2c（tile 扫描）、G2f（审计，纯静态+单测可窗口外做）、G3a/G3b
3) G4a 账本窗口（半天）→ 基线落袋
4) G1 落地 → G4b 全套 → G4c 经济复测（翻转/再关账，都诚实记录）
5) G5 独立窗口（域决策）→ 翻 64k+ 的门后再谈 PF-3/PF-4/G3c
6) G4d 蹭任意 decode 窗口（影子测）
```

与 V4.1 路线的耦合点只有一个：**G1 = I-2 的 Metal 地基**。排期上地基先行，GLM 驻留域先跑通（改动面最小），
V4.1 流式岛随后（带路由/槽位动态性）；两边各自保留独立 env 与 ABBA 台账。

## 5. 预期台账（诚实区间）

| 路线 | 对象 | 预期 | 依据 |
|---|---|---:|---|
| G1 | decode @短–12k | **+10~14%（32→35–36）** | DC-2 空隙 4.4 ms/31.3；CUDA graphs 外部 −40% 图段 |
| G2a | prefill 全 ctx | +2.5~4% | omlx 实测 2.3×/KDA 核占比 7.1% |
| G2b/G2c | prefill @41k | −5%~+8%（扫描定夺） | packed 先例：低并行段可能摊不平 |
| G3a/G3b | decode | +1~3% | 桶占比 × roofline 缺口 |
| G4（翻正情形） | decode 工作流量 | 再 +10~20%，且净亏转正 | llama.cpp MTP 2.2~2.4×（CUDA）；我们 verify 摊薄打折 |
| G4d | 高重复流量 decode | +5~15%（流量敏感） | ngram-mod 外部实测区间 |
| G5 | ≥96k 域 | 可行性前置，不计速 | KV 135 GB > RAM |

无硬件条件合计目标：**decode 32→35–36（G1）+ kernel 小刀 2~5% + MTP 翻正则另加一档**；
prefill 480–514 → +4~8%（G2a+c），128k 域解锁后二次评估 PF-3/PF-4。
再往上，驻留域的顶是 DRAM roofline（有效已 ~500/546 GB/s）——**40 t/s 以上的稳态只能从"每 token 多吐 token"（G4）来，
不能从搬运来**；这句话写死，防止下一轮又去挖 matvec。

## 6. 参考出处

内部：GLM53_SPEEDUP_NEXT_PLAN（PF/DC 全部裁决）、OMLX_WAVE_PORT_ANALYSIS（S0 五桶、附二/附三）、
SPARSE_MLA_NAX_PLAN（§7–§9 定版与 §8 遗留）、V41_M5MAX_BASELINE_SPEED_PLAN §16–17、V41_SSD_STREAMING_ROADMAP（R1 共用地基）、LOCAL_INVENTORY。
外部：
- llama.cpp：CUDA Graphs（NVIDIA blog）、issue #6763、MTP PR #22673 / Gemma4 #23398 / verify cudagraph #24261、
  DSA 支持 issue #20363、b10907（MTP KV 修复）、release 流（Metal MoE/SSM fusion@0.5.0）、
  M5 tensor API 雷区：upstream #28748（≥2 GiB slice 寻址）、#27473 + PR #27461（languageVersion）、Ollama #15594；
  `--spec-type ngram-*` 文档；优化经验帖 #21112（MTP+ub/b 组合，域=CUDA 多卡，仅方向参考）
- vLLM：V1 架构（Red Hat）、CUDAGraph in V1、Kimi K3/KDA 融合内核博文（2026-07-22）、DSpark 解读（Spheron）与 arXiv 2607.05147
- DSA/IndexShare：tensoreconomics《DeepSeek Sparse Attention from First Principles》、Raschka IndexShare 条目、vLLM DSA 线程
- KDA：FLA `chunk_kda`、FlashKDA（CUTLASS）、sglang-jax #948（验收式 `prefill(T)==prefill(T−k)+decode(k)`）
- KTransformers：官网 updates（GLM-5 Day-0）、SOSP'25 Expert Deferral
- Gemma4-MTP/Ollama 生态佐证链：freedom.tech release 史、tech-insider 2026 投机部署综述
