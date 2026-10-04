# ds4 ← oMLX 提速波：移植可行性分析

> 分析日 2026-09-29。供体：`~/omlx-notes/upstream_speedup_methods_survey.md`（上游 `5ea8cfc1`，
> 2026-09-26~29 提速波）+ 同目录 `perf_lessons.md` / `glm53_flash_speedup.md` /
> `glm53_decode_speedup_handoff.md` / `dsv41_upstream_perf_port_analysis.md`。
> 受体：本机 ds4（DwarfStar 4）检出 `~/ds4`，目标模型 GLM-5.3-Flash-Q2 与 DeepSeek V4 Flash / V4.1 Flash。
> 本机 = **M5 Max / Mac17,6 / 128 GB = NAX 域**（与 Studio M3 Ultra 的"非 NAX"结论域相反，
> 上游 NAX 条目在本机**成立**）。
> 域标注约定：**NAX** = M5 系 tensor units；**SIMD** = 经典 simdgroup 矩阵/FMA；
> **框架** = 只在 MLX 图/调度语义下成立的机制。

---

## 0. 结论速览

1. **代码不可搬运，机制可搬运，且 ds4 已自带该波的大部分机制。**
   omlx 侧是 Python + MLX 的图级融合（把 6600 次 dispatch 融成精确大核）；ds4 是手写
   C/ObjC/Metal 引擎（`ds4.c` 8.5 万行、`ds4_metal.m` 5.0 万行、26 个 `.metal` 共 22 万行），
   融合本来就是逐核手写的。逐条比对后：survey 的 26 个条目里，
   **约 12 条 ds4 已有等价物**（NAX dense/MoE GEMM、DSV4 indexer NAX、逐层 async flush、
   GPU keep-warm、融合 decode 核族、链式 MTP…），**约 6 条不适用**（纯 MLX 框架件或家族件），
   **约 8 条是真缺口**。
2. **真缺口高度集中在 GLM-5.3 的三个专用核**，而这台机器是 NAX 域——正是上游这波收益最大的地方：
   - GLM53 **indexer 打分**仍走经典 SIMD 8×8，而同仓库的 DeepSeek indexer 已有 MPP/NAX 版
     （几何几乎相同：都是 32 heads × 128 dim）→ **最优先，蓝本在仓库内**；
   - GLM53 **KDA prefill 递推**是上游 #3984 修复前的形态（1 simdgroup/值行、逐 token 设备读、
     两次 `simd_sum`、无寄存器/时间块、无预取）；
   - GLM53 **sparse MLA**（`kernel_glm_attention_indexed_*`）全 SIMD，无 tensor-unit 版。
3. **其余缺口是策略/调度级小项**：GLM53 宽步被 2048 硬门 + 256 MB score scratch 卡住；
   GLM53 的 HC pre 是 4 dispatch 链（DeepSeek 已有 n=16384 的融合核可抄）；
   DeepSeek layer-major prefill 仍逐阶段阻塞 `end_commands`（GLM/Qwen4 路径已是 async flush）；
   DeepSeek MoE 的 packed-RHS 物化可参照 #4029 消除；greedy verify 对齐（#4050）是行为项。
4. **上游所有数字都是 MLX 基线相对值，一律不得跨框架引用。** 例如 §2.1 HC 融合在 omlx 是
   pp4096 +22.2%（因为 MLX 规范路径有 256 行分片拷贝），而 ds4 的 HC 链只有 4 个 dispatch、
   sinkhorn+mix+collapse 早已融成一核，预期量级完全不同。**每个候选都必须先交 ds4 侧测量。**

---

## 1. 为什么不能复制代码：两个项目的结构性差异

| 维度 | omlx | ds4 |
|---|---|---|
| 语言/运行时 | Python + MLX（图构建 + 懒求值） | C / Objective-C++ / Metal，手写调度 |
| 调度单位 | `mx.eval` / `async_eval` / MLX command buffer | `ds4_gpu_begin_commands` / `end_commands`（阻塞）/ `flush_commands`（异步），CB 由引擎显式管理 |
| 量化布局 | oQ4e：312×mxfp8 + 129×mxfp4，**MLX affine block 布局** | GGUF：IQ2_XXS / Q2_K / Q4_K / Q5_K / Q6_K / MXFP4 / Q8_0，**ggml 布局** |
| 注意力 | `mx.fast.scaled_dot_product_attention` + MLX 融合核 | 手写 `metal/flash_attn.metal` + `kernel_glm_attention_indexed_*` / `kernel_dsv4_indexed_mixed_attention_*` |
| MoE | MLX `gather_qmm` + 家族门控 | 手写 `metal/moe.metal`（84 核）+ MPP packed 族 |
| MTP | 共享 `batch_generator` + `_DepthController` | 自研 `glm_mtp_*` cycle + DSpark |
| 测试 | pytest + bit-exact 融合套件 | `ds4_test` + logprob 向量 + quality fixture |
| 硬件选择 | MLX 自动挑 NAX | `ds4_gpu_mpp_available()` 显式门 |

**推论**：上游 kernel 里的 `mxfp8`/`mxfp4` 载入、`relaxed_precision`、cooperative tensor 形状、
`dextents` 直读，全部绑死 MLX 布局与算子语义；ds4 侧对应物必须按 ggml 布局重写。
**能搬的是**：算法（分块/寄存器复用/预取/掩码折进 epilogue/共享一遍 KV）、
门控与回退设计（fail-closed + kill switch + engage 锚定）、测量与 A/B 纪律。

---

## 2. 逐条判定矩阵

判定记号：**已有**（ds4 已实现等价机制，勿重复造）／**缺口**（值得立项）／**待测**（先量再定）／
**不适用**（框架件或家族件）。

### 2.1 硬件域（NAX tensor units）

| 供体 | 机制一句话 | ds4 现状（证据） | 判定 |
|---|---|---|---|
| §1.1 #3985 GLM DSA indexer→tensor units | 每 query 打全池 key O(L×P)；32×32 输出块、bf16×bf16→fp32、mask 折进 epilogue | GLM53 只有 `kernel_glm_indexer_scores_tiled{,_f32}`（经典 `simdgroup_float8x8`，`metal/dsv4_misc.metal:2070/2203`）；**同仓库** DeepSeek 已有 `kernel_dsv4_indexer_scores_nax`（`metal/dsv4_misc.metal:6587`，门 = `mpp_available && n_tokens>=16`，`ds4_metal.m:18906`） | **缺口（最高 ROI）** |
| §1.2 #3986/#3996 GLM sparse MLA→tensor units | 一 threadgroup 管 (query,32 heads)，key 行按索引直读、fp32 online softmax；P×V 操作数瘦身 | `kernel_glm_attention_indexed_decode/batch*` 全 SIMD（`metal/dsv4_misc.metal`），无 MPP 变体；DeepSeek 侧 `kernel_dsv4_indexed_mixed_attention_*` 同样 SIMD | **缺口（大工程）** |
| §1.3 #4020/#3934 Qwen4 QSA | 两步 tensor-unit flash + 升序块表 | ds4 有 `metal/qwen4.metal`（含 `kernel_qwen4_moe_mm_*_nax_t`），但本次目标模型不含 Qwen4 | 域外（可留档） |
| §1.4 #3994 JIT NAX attention 192/128 | JIT 生成 NAX 核、stride 直读、fail-closed | ds4 走预编译 kernel 集（`metal/flash_attn.metal` 模板实例化），无 JIT | 不适用（家族/几何不同）；**"fail-closed + 首用对拍"机制可搬** |
| §1.5 #4033 长上下文 decode split-key matrix 核 | MLX vector SDPA 每 (row,head,key) 一次 simd_sum → ALU-bound；改 8×8 fp32 simdgroup product、verify 行共享一遍 KV | ds4 **已有 split-key 结构**（`kernel_glm_attention_indexed_decode_split_group8_partial/reduce`、`kernel_dsv4_indexed_mixed_attention_heads8_split`），flash 用 `simdgroup_matrix`（`metal/flash_attn.metal:520/569`） | **待测**（是否 ALU-bound 未知；测法见 §5） |
| §1.6 #3954/#3995/#4022/#4029/#3993 MoE gather_qmm NAX 家族 | 128 行 tile 实例、SwiGLU 融进 epilogue、sorted row→token 原地读、gate/up 拼接 | **已自带大半**：MPP dense/MoE 族（`kernel_mul_mm_mpp_direct_rhs` n64/n128，`metal/dense.metal:2394`；`kernel_mul_mm_id_*_mpp*`）、pair-SwiGLU 融合（`kernel_mul_mm_id_pair_swiglu_f16*`）、M32N128 宽块（`ds4_metal.m:43371`）。**缺口**：`ds4_gpu_encode_moe_packed_rhs`（`ds4_metal.m:33175`）把激活按专家序物化成 f16 32 行 tile（8192 chunk×6 专家×5120 宽 ≈ 503 MB/层/次），正是 #4029 删掉的那类拷贝 | **部分缺口**（仅 DeepSeek；GLM53 走 GLM 专用核族，不消费 packed 路径） |
| §1.7 #3988/#3973 宽 chunk/宽步 | tensor-unit 稀疏注意力让每 query 成本与 chunk 无关后，8192 chunk 让 MoE 每专家行数翻倍 | GLM53 chunk 硬门 2048（`ds4.c:39042`）+ workspace 硬门（`ds4.c:47645`）+ score scratch 256 MB（`ds4.c:39040`）；DeepSeek 默认 4096，但 MPP MoE 已有 ≥8192 的 M32N128 路径 | **缺口（GLM53 侧，低风险试）** |

### 2.2 prefill 侧融合与调度

| 供体 | 机制 | ds4 现状（证据） | 判定 |
|---|---|---|---|
| §2.1 #3983 GLM HC prefill 批不变融合核 | fp32 RMS+mix+sinkhorn+collapse 一核、expand 一核，固定 per-row 归约序 | GLM53 是 4 dispatch 链：`glm53_graph_hc_pre_rows`（`ds4.c:47831`，prefill）/`glm53_graph_hc_pre`（`ds4.c:48017`，decode）= RMS → matmul → `ds4_gpu_hc_split_weighted_sum`（sinkhorn+mix+collapse **已融合**）→ RMS。**蓝本**：DeepSeek decode 用 `kernel_dsv4_hc_rms_norm_mix_f16`（`metal/dsv4_hc.metal`）把 fp32 RMS+mix 融成一核，注释写明按 **n=16384** 设计——GLM53 的 hc_dim = 4×4096 = 16384，**同几何**（调用点 `ds4.c:24387/26019`） | **部分缺口（低风险）** |
| §2.2 #3984 GLM KDA 递推 blocked→per-core | 寄存器持 2 行×16 通道 fp32 态、16-token 块 stage+预取、门在核内算 → ~2.3×/层；per-core 再 1.15× | `kernel_glm53_kda_prefill_recurrence`（`metal/glm53_kda.metal:249`）= **上游修复前形态**：一 simdgroup 一值行、逐 token 设备读 q/k/decay、每 token 两次 `simd_sum`、无块化/预取/寄存器复用 | **缺口** |
| §2.3 #4025 GLM 稠密前缀因果行块 | 前 2051 行跳 indexer 走稠密；MLX 无融合核 → 拆 8 个因果行块，6.17→3.91 ms/层 | ds4 稠密前缀走 flash attention（`glm_graph_use_flash_attention_prefill`，≥24 行，`ds4.c:49534`）+ 稠密区间用 causal range select（`ds4.c:53451`）；**没有** [rows,rows] 非融合全阵 | **已有等价物** |
| §2.4 #3982/#4006 Qwen4 prefill / GDN 流水线 | MLP 残差折进 HC norm、depthwise 核、`gated_delta_pipelined` | ds4 对应物是 `metal/qwen4.metal` 的 GDN/HC 核族；非本次目标模型 | 域外；**"12-token 块 + 软件流水线"的 KDA 移植可复用其思路** |
| §2.5 #3971 per-layer async 流水线 + lazy_last | 逐层 blocking eval → `async_eval` 深度 1，GPU 不再等 host 建图 | **GLM/Qwen4 已自带**：GLM prefill 每层 `flush`（异步提交，继续编码）+ 周期 `drain`（阻塞）（`ds4.c:53151-53200`）；GLM decode 每 4/32 层 flush（`ds4.c:55540`、`56525`，`DS4_GLM_DECODE_FLUSH_INTERVAL`）；Qwen3.8 prefill 有注释自证同款模式（`ds4.c:58819-58822`）。**窄缺口**：DeepSeek layer-major prefill 在 split 路径仍 attn/ffn 各自 `begin→encode→end_commands`（`ds4.c:37837-37886`），而 `end_commands` = commit + wait（`ds4_metal.m:1568-1573`） | **已有（GLM）+ 窄缺口（DeepSeek）** |

### 2.3 decode / verify 融合栈

| 供体 | 机制 | ds4 现状 | 判定 |
|---|---|---|---|
| §3.1 #3989/#4019/#4026 GLM decode 融合栈 | 6600 dispatch/token → 精确 JIT 大核；每 8 层 `async_eval`；verify 融合 | **已自带大半**：融合核族（QKV+norm+rope+KV store `kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32`、router project+select `kernel_dsv4_router_project_select_fused`、GLM53 KDA decode `kernel_glm53_kda_decode`、FFN tail `glm53_graph_encode_ffn_tail_one`、HC expand 延迟 `ds4.c:24041`）+ 每 4/32 层 async flush。**剩余**：host 编码时间是否在关键路径上（用 `DS4_METAL_CB_TIMES` 的 encode/commit→done 分列直接量） | **已有 + 待测** |
| §3.2 #4024/#4038/#4039/#4041 Qwen4 decode | 两 launch HC、one-launch MoE combine、行精确 verify 窗 | ds4 有对应 qwen4 核族；非目标模型 | 域外 |
| §3.3 #3990/#4033 MiMo decode fast | 行拼接权重单 matvec、router fp32 副本消除、combine+residual+norm 单核 | 机制命中 DeepSeek：ds4 的 router/combine 已融合（`kernel_dsv4_router_project_select_fused`、`kernel_dsv4_moe_sum6/8_f32`），**但"每步 cast 权重到 fp32"的副本是否存在未查**（MiMo 供体每步省 188 MB） | **待测（小项）** |

### 2.4 MTP / scheduler / 方法论

| 供体 | 机制 | ds4 现状 | 判定 |
|---|---|---|---|
| §4 #4050 greedy verify 对齐 serial 采样 | verify 行 argmax 跑在与 serial 相同的算术上（logits−logsumexp 在 logits dtype 舍入后）→ greedy MTP == MTP-off | `docs/SPECULATIVE_DECODING.md` 明说"accepted tokens 保留 batched verifier 的状态；归约序可与单 token decode 不同，长 greedy 续写不必逐位一致"——**正是 #4050 之前的形态** | **缺口（行为/可复现性，非速度）** |
| §4 #4026/#4041 verify 融合 / 行精确窗 | 融合 verify 核；一行窗直接跑 serial decode 步 | ds4 有 `v41_decode_batch`（≤8 行走融合 decode 核，`ds4_metal.m:43379`）+ GLM verify2 cycle（`ds4.c:73923`） | **已有等价物** |
| §4 #4031 late-join hand off | 共享批里完成的请求交接而非 replay | ds4 是单会话 + 磁盘 KV checkpoint 架构（无共享批），语义不同 | 不适用 |
| §5 #3974 GPU keep-warm | Apple GPU ~1s park，空闲 2s 首 CB +996 ms | **已自带**：`ds4_gpu_queue_keepalive_thread`（`ds4_metal.m:11331`）每 `DS4_METAL_QUEUE_KEEPALIVE_MS`（默认 1000 ms）提交 1 threadgroup 空核 | **已有**（周期可从 1000→500 ms 微调，需带 gap 实测） |
| §5 #3975 detokenizer 原型复用 | 每请求重建 250k 词表 ~88-90 ms | ds4 tokenizer 是共享 C 结构，无每请求重建（`ds4_server.c` 无 tokenizer 初始化路径） | 不适用 |
| §5 #3854 首 chunk 提前交付 | burst decode 先释放首 chunk | ds4 直接从生成循环流式吐 token | 不适用 |
| §5 #4012 command-buffer cap | MLX 绑定字节超限即终结 CB，空核 launch 28→12 µs | ds4 显式管理 CB 边界（无 MLX 的隐式绑定膨胀） | 不适用（框架件） |
| §6 bit-exact 融合套件 | 逐位转录 + fp32 对拍 + canary + fail-closed + kill switch + engage 锚定 | ds4 已有 `ds4_test --logprob-vectors`、quality fixture、`DS4_*_DISABLE_*` kill switch 传统 | **全可搬（纪律层）** |
| §6 A/B 纪律 / 域标注 | 同 build 同机、fwd+rev、区间不重叠、NAX 命题上 M5 | 与 perf_lessons 一致；本机即 NAX 域 | **全可搬** |

---

## 3. 真缺口详述（按 ROI 排序）

### S1 · GLM53 indexer 打分 → NAX tensor units（§1.1）

- **现状**：`kernel_glm_indexer_scores_tiled` / `_tiled_f32`（`metal/dsv4_misc.metal:2203/2070`）用经典
  `simdgroup_float8x8`；decode 走 `kernel_glm_indexer_score_one{,_direct}`。无 MPP 变体。
- **蓝本（同仓库）**：`kernel_dsv4_indexer_scores_nax`（`metal/dsv4_misc.metal:6587`）——
  MPP tensor ops、16 token × 32 行 tile、causal mask 折进 epilogue（不可见行写 `-INFINITY`）、
  门控 `ds4_gpu_mpp_available() && n_tokens >= 16`（`ds4_metal.m:18906-18910`）。
- **几何对照**：两者 indexer 都是 **32 heads × 128 dim**；差异在
  GLM53 的 ratio-4 pooled cache（f16 compact）+ `index_top_k = 2048` + `expand_pool_selection`
  的尾部语义（`last_indexer_guaranteed_prefix`，见 `ds4.c:55688-55695` 注释）。
- **落点**：新增 `kernel_glm53_indexer_scores_nax`（沿用 DSV4 骨架 + GLM53 的 pool/掩码语义）
  → `ds4_metal.m` 分派（照 18906 的模式）→ `DS4_GLM53_INDEXER_NAX=0` kill switch
  → 首次 dispatch 与 `_tiled_f32` 对拍，不过永久回退（fail-closed）。
- **预期**：供体报 3.36/4.05/10.94 → 0.64/1.12/3.43 ms/层 @池 1k/4k/16k（**MLX/NAX 域**）。
  ds4 域未测——先量 indexer 占 chunk 比例。**测量钩子已存在但被硬编码关掉**
  （`glm_graph_indexed_prefill_trace_enabled()` 直接 `return false`，`ds4.c:39248/39264`），
  打开即可得到 per-stage prefill 时间。
- **风险**：GLM53 pool 是 ratio-4 压缩 + f16 + 尾部补齐；掩码/top-k 选择必须逐位一致，
  否则 2048 选 512 的行集合会漂移（下游稀疏注意力误差被放大）。

### S2 · GLM53 KDA prefill 递推 blocked / per-core（§2.2）

- **现状**：`kernel_glm53_kda_prefill_recurrence`（`metal/glm53_kda.metal:249-287`）：
  `head = tgpig.x; value = tgpig.y*4 + sg;` 后逐 token 循环，每 token 从 device 读 q/k/decay
  float4，`h *= decay4; hk = simd_sum(dot(h,k4)); h = fma(k4, delta_v, h); simd_sum(dot(h,q4))`。
  **逐 token 两次 simd_sum + 无预取 + 无寄存器块**。
- **供体机制**（#3984）：① blocked 版：每线程寄存器持 2 行×16 通道 fp32 态、16-token 块
  stage + 预取、forget 门在核内按原表达式树算（门位一致）→ ~2.3×/层；② per-core 版：
  64×128 值行按 GPU 核数切连续段、8 lanes/行、12-token 块展开、四步 q·S 归约批量后置
  → 再 1.15×；组合树上 +6.4-6.9%（GLM）。
- **落点**：重写 recurrence（伴生核 `_prefill_prepare` / `_prefill_output` 可另立项融合
  conv4+SiLU+L2 与 norm-gate，对应 omlx P3 的两个融合核）。
- **预期**：ds4 域 KDA 占比未知（GLM53 46 层中 KDA 层数需实测日志确认）。先 profile 再定目标倍数。
- **风险**：递推的 fp32 累加序改变 → 必须跑 `ds4_test --logprob-vectors` + GLM53 quality fixture；
  KDA 状态跨 chunk/跨 session 的一致性要单独回归（有 conv_state / recurrent_state 两个持久态）。

### S3 · GLM53 宽步 8192（§1.7 / #3988）

- **现状**：三处硬门 —— `DS4_GLM53_PREFILL_CHUNK_TOKENS 2048`（`ds4.c:39042`）、
  prefill workspace `rows > 2048 → false`（`ds4.c:47645`）、score scratch 256 MB
  （`DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB`，`ds4.c:39040`，决定长上下文下的 score 行数上限）。
  workspace 约 **450 KB/行**（4×16384 HC + 6×8192 KDA 投影缓冲等，`ds4.c:47656-47680`）
  → 2048 行 ≈ 0.9 GB，8192 行 ≈ 3.6 GB。
- **供体机制**：tensor-unit 稀疏注意力让每 query 成本与 chunk 无关后，8192 chunk 使 MoE
  每专家行数翻倍（omlx：pp16384 +3.4%、pp65536 +6.0%，门 ≥128 GB；本地非 NAX 强制宽步
  在 P1+P3 之上 +2.5% pp8192）。
- **落点**：把 2048 变成"默认值 + env 覆盖"（cap 提到 8192），score scratch 提到 1 GB 并纳入
  memory guard；保留 `DS4_GLM53_PREFILL_CHUNK=2048` 回退。
- **前置判据**：**先证明 GLM53 的每 query 成本与 chunk 无关**（上游门控存在的理由）。
  ds4 的 indexed attention 已是"每行独立 split-key"形态，大概率满足；但 score scratch 与
  pool 列数的乘积在 64k 上下文时只够 ~4096 行（`glm_graph_indexed_prefill_score_tokens`），
  即**长上下文才是宽步的真正约束**。
- **风险**：96 GB 模型 + KV + 4 GB workspace 在 128 GB 机器上的内存余量；chunk 变化会改变
  MoE 每专家行数 → 可能触发不同 tile 路径（`packed_m32n128` 只在 DeepSeek 侧）。

### S4 · GLM53 HC pre 链融合（§2.1 / #3989）

- **现状**：4 dispatch（RMS → matmul(16384→24) → sinkhorn+mix+collapse → RMS），
  见 `glm53_graph_hc_pre_rows`（`ds4.c:47831`）/`glm53_graph_hc_pre`（`ds4.c:48017`）。
- **蓝本（同仓库）**：`kernel_dsv4_hc_rms_norm_mix_f16`（`metal/dsv4_hc.metal`）把 fp32 RMS 与
  mix matvec 融成一核（Phase A 逐位复刻 `kernel_rms_norm_f32_4` 归约树，Phase B 复刻
  `kernel_mul_mv_f16_f32_4`），注释写明 NSG/NR0 按 **n=16384** 的 dispatch 推导 ——
  GLM53 hc_dim = 4×4096 = **16384**。调用点 `ds4.c:24387/26019`（DeepSeek decode）。
- **落点**：GLM53 的两条路径（prefill rows 版 + decode 版）改走融合核；原链保留为 kill switch。
- **预期**：**不要引用 omlx 的 +22.2%**（那是 MLX 规范路径带 256 行分片拷贝的结果）。
  ds4 侧先量 HC 占 prefill kernel 时间的比例（`DS4_METAL_GRAPH_PREFILL_PROFILE` 的
  encode/execute 分列 + layer stage profile），估 1-3%。
- **风险**：融合核的 RMS 归约序与 `ds4_gpu_rms_norm_plain_tensor` 必须逐位一致，
  否则 HC 权重漂移会经 sinkhorn 迭代放大。

### S5 · DeepSeek layer-major prefill 的逐阶段阻塞（§2.5 窄缺口）

- **现状**：`metal_graph_prefill_layer_major` 在 split 路径里
  `begin → encode attn → end_commands → begin → encode ffn → end_commands`（`ds4.c:37837-37886`）；
  `end_commands` 会 `commit` 后 `wait`（`ds4_metal.m:1568-1573`），host 编码时间直接加在关键路径上。
  触发 split 的条件含 `n_tokens > 2048`、`ssd_streaming`、`callback_split`（`ds4.c:37489-37493`），
  即 DeepSeek 默认 4096 chunk 一定走这条路。
- **蓝本（同仓库）**：GLM prefill 的 `flush`（异步）/ `drain`（阻塞）双动作
  （`ds4.c:53151-53200`）与 Qwen3.8 的 `flush_layer`（`ds4.c:58819-58822` 有注释自证
  "Submit this prefix while the host encodes the remaining layers"）。
- **落点**：把 split 路径的 `end_commands` 换成 `flush_commands` + 周期 drain；
  **注意数据依赖**：`ds4_gpu_signal_selected_readback_ready` / `ds4_gpu_wait_selected_readback_ready`
  的 shared-event 语义（`ds4.c:26886`、`ds4_metal.m:11411`）与 transient buffer 生命周期
  （`finish_command_buffer` 里 `[g_transient_buffers removeAllObjects]`）都绑在"等待"上。
- **预期**：`DS4_METAL_GRAPH_PREFILL_PROFILE` 直接打印每层
  `attn encode=.. execute=.. ffn encode=.. execute=..`；encode 占比 >2% 才动手。

### S6 · DeepSeek MoE packed-RHS 消除（§1.6 / #4029）

- **现状**：`ds4_gpu_encode_moe_packed_rhs`（`ds4_metal.m:33175`）把激活按专家序物化为
  f16、32 行 tile（`kernel_moe_pack_rhs_{f16,f32}` + `kernel_moe_packed_offsets`），
  8192 chunk × 6 专家 × 5120 宽 ≈ **503 MB/层/次**（写+读 ≈ 1 GB 流量）；
  门控 `use_packed_mpp`（`ds4_metal.m:43359`，n_tokens∈[512,8192]，512 MB 上限）。
- **供体机制**（#4029）：核经 sorted row→token map **原地读行**，大拷贝直接消失
  （omlx：8192 chunk 每层 419-537 MB、0.65-0.9 ms GPU；服务器 +1.2-2.3%，位一致）。
- **落点**：MPP MoE 核加索引化 B 载入（threadgroup 内 gather → tensor load），
  保留 packed 路径为回退；canary 用乱序 row map 先验。
- **预期**：+1-2% 量级（带宽域，与 M5 Max ~500-600 GB/s 成比例）。**只命中 DeepSeek**
  （GLM53 top-8 不走 packed 路径：`use_packed_mpp` 硬要求 `n_expert == 6`）。

### S7 · greedy verify 对齐（§4 / #4050）——行为项，不是提速

- **现状**：`docs/SPECULATIVE_DECODING.md` 明确"accepted tokens keep the state produced by the
  batched verifier. Floating-point reduction order can differ from one-token decode, so long
  greedy continuations need not be byte-identical"。
- **供体机制**：让 verify 行的 argmax 跑在与 serial sampler **相同**的算术上
  （`logits - logsumexp` 在 logits dtype 上舍入后再 argmax，tie 归低 token id），
  于是 greedy MTP 输出 == MTP-off。
- **落点**：ds4 的 GLM MTP cycle（`ds4.c:73800-73930`）与 DSpark verify 的 argmax 路径。
- **验收**：按"对齐 serial"而非"输出不变"（会改变文本）。ds4 已有 `--quality` / `--dspark-strict`
  作为对照臂，logprob 向量回归可复用。
- **价值**：可复现性/可验收性（同一 prompt 在不同执行配置下 greedy 一致），间接减少"提速后
  文本变了"的验收成本。

### S8 · 长上下文 decode attention 是否 ALU-bound（§1.5）——先测

- **现状**：ds4 已有 split-key（partial + reduce）与 classic `simdgroup_matrix` flash。
  供体测到 MLX vector SDPA 在 GQA16 下只有 ~1/3 DRAM 带宽（350-390 GB/s vs 1.1 TB/s roofline），
  verify 多行时进一步掉到 112-134 GB/s。
- **测法**：`DS4_METAL_DECODE_STAGE_PROFILE` 拿 attn 阶段的 µs，除以该阶段 KV 字节数，
  得有效带宽；若 <40% roofline 且 attn 占比 >15%，则立项"split-key fp32 matrix 核
  （verify 行共享一遍 KV）"。
- **预期**：不确定；ds4 的 split 结构本来就是"共享一遍 KV"形态，可能只需替换内积实现。

---

## 4. 已自带清单（**不要重复造**）

| 能力 | ds4 证据 |
|---|---|
| NAX/MPP dense GEMM 家族（n64/n128、q4_0/q4_K/q8_0…） | `metal/dense.metal:2394` `kernel_mul_mm_mpp_direct_rhs` + 2494-2503 实例化表 |
| MPP MoE + pair-SwiGLU 融合 + packed RHS + M32N128 宽块 | `metal/moe.metal` 的 `kernel_mul_mm_id_*_mpp*`；门控 `ds4_metal.m:43359-43379` |
| DeepSeek indexer NAX | `metal/dsv4_misc.metal:6587`；门控 `ds4_metal.m:18906` |
| GLM 逐层异步 flush（prefill）+ 周期 drain | `ds4.c:53151-53200` |
| GLM decode 每 4/32 层异步 flush | `ds4.c:55540`、`56525`；`DS4_GLM_DECODE_FLUSH_INTERVAL` |
| Qwen3.8 prefill 的 flush_layer 模式 | `ds4.c:58819-58822` |
| GPU keep-warm（Apple park 对策） | `ds4_metal.m:11331`；`DS4_METAL_QUEUE_KEEPALIVE_MS`（默认 1000 ms） |
| 融合 decode 核族（QKV+norm+rope+KV store / router+select / KDA decode / FFN tail） | `kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32`、`kernel_dsv4_router_project_select_fused`、`kernel_glm53_kda_decode`、`glm53_graph_encode_ffn_tail_one` |
| HC expand 延迟执行 | `ds4.c:24041-24067` |
| 链式 MTP（depth-3 cycle）+ verify2 | `glm_mtp_draft2`（`ds4.c:60248`）、`ds4.c:73923` |
| DSpark 5-token + ≤8 行融合 decode 批 | `ds4_metal.m:43379` `v41_decode_batch` |
| 测量工具 | `DS4_METAL_CB_TIMES`（每 CB encode/commit→done/gpu span）、`DS4_METAL_GPU_BUSY_PROFILE`、`DS4_METAL_GRAPH_PREFILL_PROFILE`（逐层 encode/execute）、`DS4_METAL_DECODE_STAGE_PROFILE`、`DS4_METAL_FLASH_ATTN_STAGE_PROFILE` |

---

## 5. 测量方案（动手前强制）

1. **五桶分解**（`perf_lessons §1`）：用现成 env 拿
   `总时间 = kernel busy + GPU idle(等 host 编码/提交) + CB commit + 前后处理 + 电源态`。
   - `DS4_METAL_CB_TIMES=1` → 每 CB 的 `encode X ms, commit->done Y ms, gpu span Z ms`，
     以及每批 transient buffer 字节数（判断"减 launch"还是"减拷贝"）。
   - `DS4_METAL_GRAPH_PREFILL_PROFILE=1` → DeepSeek layer-major 逐层
     `attn encode=/execute=、ffn encode=/execute=`。
   - `DS4_METAL_DECODE_STAGE_PROFILE=1` → decode 逐阶段（GLM/DeepSeek 共用 macro）。
2. **打开 GLM53 prefill trace**：`glm_graph_indexed_prefill_trace_enabled()` 目前硬编码
   `return false`（`ds4.c:39248/39264`）——接一个 env 即可得到 indexer/KDA/attention 各阶段
   的 prefill 时间，这是 S1/S2/S3 排序的前提。
3. **带 gap 的交互吞吐**（`perf_lessons §1`）：连跑 benchmark 测不到 GPU park 惩罚。
   用 2 s / 6 s 间隔发请求，对照 `DS4_METAL_QUEUE_KEEPALIVE_MS=1000` vs `500`。
4. **roofline 行**（`perf_lessons §3`）：每个候选热点给"flop 下界 / 当前耗时 / 差距倍数"，
   >2× 的项即使无供体也立项（S1/S2/S8 都要这一行）。
5. **A/B 纪律**：同 build 同机、forward+reverse 双腿、区间不重叠才宣称；
   kill switch 与 engage 日志锚定；每次 A/B 记录量化路径/residency（同模型不同配置差 5-12× 的老坑）。
6. **内存前提**：本机 128 GB，加载 GLM53-Q2（96.5 GB）或 DSV4 Vision-Exp（97.6 GB）时
   必须空机（当前 `omlx-server` 常驻 ~41 GB，实测前需停）。S3 的宽步会额外要 ~3 GB workspace。

---

## 6. 否决 / 不适用清单（带域）

| 项 | 判定 | 域 |
|---|---|---|
| 直接搬 omlx 的 mxfp8/mxfp4 NAX kernel 源码 | 不可行 | 布局 + 量化 + 框架三重不兼容 |
| #3975 detokenizer 原型复用 | 不适用 | ds4 是 C 引擎，无每请求重建 |
| #3854 首 chunk 提前交付 | 不适用 | ds4 直接流式吐 token |
| #4012 command-buffer caps | 不适用 | ds4 显式管理 CB，无 MLX 绑定膨胀 |
| #4031 late-join hand off | 不适用 | ds4 单会话 + 磁盘 KV，无共享批 |
| #3994 JIT NAX attention(192/128)、#4020 QSA、#3990 MiMo decode fast、#4006 GDN、#4024-#4041 Qwen4 | 域外 | 家族（ds4 目标模型不含该结构）；**机制**可按 §2/§3 借用 |
| 在非 NAX 机器上给 NAX 条目下"不适用"结论 | 禁止 | perf_lessons §5 前车之鉴（本机是 NAX） |
| 把 omlx 数字当 ds4 预期 | 禁止 | 基线不同（MLX 图开销 vs 手写引擎） |

---

## 7. 建议执行顺序

| 步 | 动作 | 前置 | 验收 |
|---|---|---|---|
| S0 | 交五桶分解表（§5.1+5.2）+ 带 gap 吞吐 | 空机 + 停 omlx-server | 表 + 每热点 roofline 行 |
| S1 | GLM53 indexer NAX 核（抄 DSV4 蓝本） | S0 证明 indexer 占比 | `_tiled_f32` 对拍 ≤1 ulp、top-k 逐位一致；A/B 区间不重叠 |
| S2 | GLM53 KDA prefill blocked/per-core | S0 证明 KDA 占比 | logprob 向量 + quality fixture 全绿；A/B |
| S3 | GLM53 宽步 8192（参数化三门） | S0 证明每 query 成本与 chunk 无关 | pp4096/16384/65536 A/B；内存 guard 日志零 throttle |
| S4 | GLM53 HC pre 走融合核（n=16384 同几何） | S0 量出 HC 占比 | RMS 归约序对拍；A/B |
| S5 | DeepSeek layer-major prefill 换 async flush | `DS4_METAL_GRAPH_PREFILL_PROFILE` encode 占比 >2% | CB 时间分列改善；readback 语义回归 |
| S6 | DeepSeek MoE packed-RHS 索引化 | S0 量出 pack 阶段耗时 | 乱序 row-map canary；位一致 |
| S7 | greedy verify 对齐 serial（行为项） | 无 | greedy MTP == MTP-off（长总结题） |

**不建议同时开两条以上 kernel 线**：S1/S2/S4 都改 GLM53 数值路径，需要各自独立的
对拍与 A/B 窗口，混在一起无法归因（`perf_lessons §9` 的域标注纪律）。

---

# 附：S0 实测结果 + 上游 PR 清点（2026-09-30）

> 机器：M5 Max / 128 GB（NAX）。模型 `gguf/GLM-5.3-Flash-Q2.gguf`（96.5 GB）。
> 配置：`./ds4-server -m gguf/GLM-5.3-Flash-Q2.gguf --metal -c 256000 --port 8055
> --batched-session 2 --mixed-prefill-quantum 4096 --mtp`（注意：`--batched-session` 会
> **关闭 MTP**，日志自证 "MTP speculative decoding is disabled while native session batching is active"）。
> 工具：`DS4_METAL_CB_TIMES` / `DS4_METAL_GPU_BUSY_PROFILE` / `DS4_METAL_ENCODER_TIMELINE`
> （逐 kernel GPU 时间线，自带，无需 Metal System Trace）。

## S0-1 五桶分解：GLM53 在 ds4 上是 **GPU-bound**

| 指标 | prefill | decode |
|---|---|---|
| GPU busy（进程级累加） | 96–99% | ~99%（6264 ms busy / 6386 ms wall，两次请求合计） |
| GPU idle（CB 间 gap） | Σ84 ms / 400 CB（最大 29 ms，>5 ms 仅 3 次） | 同量级 |
| host encode（阻塞批） | 88 ms / 81 CB（~0.8%） | 116 ms / 128 token（0.9 ms/token） |

**结论**：ds4 的 GLM 路径（逐层 async `flush_commands` + 周期 `drain`，见 `ds4.c:53151`）
已经把 host 编码/提交开销藏到 GPU 后面。survey 里"砍 launch / 异步流水线"那一族
（#3971/#4019/#3989）**在 ds4 上没有可回收的 headroom**；要提速只能让 GPU 把同样的活干得更快
（更好的 kernel），或**每个 backbone cycle 吐更多 token**（MTP）。

补充：
- **GPU park 无惩罚**：空闲 0/0.5/2/6/12 s 后 TTFT 恒定 1.25–1.46 s（= 370 token 的纯 prefill
  时间），keepalive 线程（`DS4_METAL_QUEUE_KEEPALIVE_MS` 默认 1000 ms）有效 → §5 #3974 已闭环。
- `--batched-session` 的 `decode_coalesce_us=2000` 在单请求下值 ~3%（35.1 → 34.0 t/s，`DS4_SERVER_DECODE_COALESCE_US=0` 可关）。

## S0-2 吞吐基线（main @428e191）

| 场景 | 结果 |
|---|---|
| prefill 2616 / 5111 / 10080 token | 467 / 410 / 372 t/s（单 chunk 2048，steady ~350–360 t/s） |
| prefill 41237 token（长上下文） | 371 t/s 平均，每 chunk 348–400 t/s（**到 41k 都不衰减**） |
| decode（batched-session，无 MTP） | **34.4 t/s**（29.1 ms/token） |
| decode（MTP depth-1，单会话） | 31.7 t/s；接受率 53/80 = **66%**，verify2 37.5 ms + head/draft 15 ms |
| decode（`--mtp-draft 2`） | 33.0 t/s（深度未生效，接受统计与 d1 相同） |

**MTP 在当前形态下是零收益甚至负收益**（verify2 = 1.28× 单 token 成本，接受率仅 66%）。
对照 omlx 的 GLM53：接受率 79.7–84.6%、2.0–2.3 tok/cycle。**decode 的唯一大杠杆在这里**，
不是 kernel 融合（见 S0-3）。

## S0-3 逐 kernel 解剖（`DS4_METAL_ENCODER_TIMELINE`）

**prefill @2048 token（Σ 5115 ms）**
| 家族 | 占比 |
|---|---|
| MoE routed（iq2_xxs MPP 24.1% + q2_K MPP 11.7%） | **37.3%** |
| attention（sparse MLA 6.6% + flash 5.9%） | 12.8% |
| Q8_0 NAX GEMM（`..._nax_direct_rhs_n128`） | 9.5% |
| **KDA prefill 链**（prepare+recurrence+output 一 dispatch 组，6.1 ms/次） | **8.1%** |
| QK low-rank 投影（`kernel_glm_qk_lowrank_q8_0_batch`，17.8 ms/次） | 7.7% |
| `kernel_mul_mv_f32_f32_4`（HC mix / router fp32） | 5.7% |
| indexer 打分（`kernel_glm_indexer_scores_tiled`） | **0.1%** |
| HC | 2.2% |

**prefill @41237 token（Σ 111.8 s）**
| 家族 | 占比 |
|---|---|
| **sparse MLA attention**（`..._lora_group8_vec`，143.6 ms × 220 次） | **28.3%** |
| MoE routed | 28.5% |
| Q8_0 NAX GEMM | 10.1% |
| KDA prefill 链 | 7.0% |
| QK low-rank 投影 | 6.3% |
| `mul_mv_f32_f32_4` | 4.9% |
| q4_K NAX GEMM | 4.6% |
| **indexer 打分** | **3.3%**（429 次 × 8.65 ms） |
| HC | 1.9% |

**decode（32 token，剔除重 prefill 批后 Σ 960.8 ms）**
| 家族 | 占比 | 说明 |
|---|---|---|
| `kernel_mul_mv_q8_0_f32` | **34.9%** | 10496 次 / 32 token ≈ **328 次/token** |
| MoE routed | 22.1% | pair_swiglu 14.1% + q2_K down 6.9% |
| `kernel_mul_mv_q4_K_dense_f32` | 8.8% | 2176 次 |
| `kernel_glm53_mul_mv_bf16_f32` | 7.0% | 3584 次 |
| MoE shared/dense FFN | 7.1% | |
| HC | 4.7% | |
| norm | 3.0% | |
| KDA decode | 2.2% | |
| attention | 2.0% | |
| router | 1.6% | |
| indexer | 0.3% | |

**decode 结论**：~58% 是**权重流式 matvec**（Q8_0 + Q4_K + BF16）+ 22% MoE →
**DRAM 带宽域**（≈13–15 GB/token @ ~30 ms/token ≈ 500 GB/s，接近本机 roofline）。
每 token ~640 次 matvec dispatch（~30 µs 级），是"带宽被小 kernel 切碎"的形态。
→ 提 decode 只有两条路：**每 cycle 多吐 token（MTP）**，或让 matvec 更大更少（row 拼接等）。

## S0-4 对原计划的修订（按实测 ROI 重排）

| 原项 | 实测证据 | 修订 |
|---|---|---|
| S1 indexer NAX | 2048 ctx 仅 **0.1%**，41k ctx **3.3%**（线性涨） | **降级为 P2**（只在 ≥64k 上下文有意义） |
| S8 sparse MLA → tensor units | **28.3% @41k**（2048 ctx 也有 6.6%），纯 SIMD（无 simdgroup_matrix/MPP） | **升为 P0** |
| MoE（#3995/#4029 类） | 28.5–37.3%，但 ds4 已用 MPP | **P0/P1**：剩下的是 tile/调度与 packed-RHS |
| S2 KDA prefill blocked | **7.0–8.1%**，现为上游 #3984 修复前形态 | **P1**（保持） |
| S4 HC pre 融合 | **1.9–4.7%**（远低于 omlx 的 12–22%） | **P3**（降级） |
| S5 DeepSeek async flush | GLM 已证 GPU-bound 99% | **P3**（先测 DeepSeek） |
| — | **MTP 零收益**（66% 接受率，verify2 1.28×） | **新增 P0（decode 侧）** |

## S0-5 上游 PR 清点（`github.com/antirez/ds4`，未合并）

已 cherry-pick 到分支 **`perf/picks-plus-1090`**（17 commits，基于 main `428e191`）：

| PR | 内容 | 判定 | 本机实测 |
|---|---|---|---|
| **#1147** | GLM prefill 恢复（validated selection prefixes） | ✅ 取 | 长 prefill **+10~13%** |
| **#1120** | M5 resident decode shapes（含 GLM53 KDA Q8） | ✅ 取 | decode 小幅 + |
| **#1093** | GLM live checkpoint 跨 tool turn | ✅ 取 | 消除每轮 re-prefill（#1092） |
| **#1135** | batched compressed attention 未来 token 泄漏 | ✅ 取（正确性） | 用户跑 `--batched-session`，直接相关 |
| **#1126** | keepalive pipeline 竞态（NSMutableDictionary 并发） | ✅ 取（稳定性） | |
| **#1122** | GLM memory guard 按 wired 内存封顶 | ✅ 取（安全） | 128 GB 机器 + 其他 runtime 场景 |
| **#1149** | Qwen3.8 QSA prefill → tensor units（oMLX #4020 移植） | ✅ 取（+模板） | 上游 M5 Max 实测 +2.8~7.4% |
| **#1150** | Qwen4 HC 残差写延迟（oMLX #3982A） | ✅ 取 | 上游 M5 Max +1.3~2.4% |
| **#1090** | **bit-exact GLM5.3 + DSV4.1 加速**（3 个代码 commit） | ✅ 取（2 处冲突手工合并） | 见下 |
| **#1127** | callback prefill layer overlap | ❌ **剔除** | GLM 短 prompt（470 tok）**-19%**，用其 kill switch 复现确认 |
| #1067 | V4.1 decode 单次 CB wait | ⛔ 不适用 | 硬门控 `pre_m5_apple_silicon()`，M5 上不生效；且冲突 |
| #1062 | Qwen3.8 batched decode + MTP（16 流 203 t/s） | ⏸ 待办 | Qwen3.8 专用 + 冲突，需手工 rebase |
| #1073 | DeepSeek V4.1 Metal + DSpark | ⏸ 待办 | 28 文件，M3 Ultra 域，未验证本机 |

**#1090 的关键发现**：它的 GLM53 调优被硬门控在 **M3 Ultra**：
`ds4_metal.m:36928 ds4_gpu_glm53_tuning_available()` 返回
`... || [g_device.name isEqualToString:@"Apple M3 Ultra"]`。
本机 M5 Max 默认**完全不生效**。加 `DS4_METAL_FORCE_GLM53_TUNING=1` 覆盖后实测：

| 场景 | main | 分支（门控关） | 分支 + FORCE | vs main |
|---|---|---|---|---|
| prefill 5204 tok | 410 t/s | 463 t/s | **495 t/s** | **+20.7%** |
| prefill 9952 tok | 372 t/s | 425 t/s | **448 t/s** | **+20.4%** |
| decode | 34.4 t/s | 34.9 t/s | **37.9 t/s** | **+10.2%** |
| 短 prompt 470 tok | 1.318 s | ~1.32 s | 1.266 s | +4% |
| 质量（1977 tok prompt + 96 greedy） | — | — | **与门控关逐字节一致** | bit-identical |

验证：`make test-glm53-kda` **PASS**（含 #1149+#1090 合并区的 two-head dispatch 逐位一致性、
UINT32_MAX/越界 ID 回退）；`tests/test_glm53_router_shared` 在 M5 Max 上 FAIL，原因是该测试
**没有**像 `test_glm53_kda.c` 那样调 `ds4_gpu_test_set_flags(DS4_GPU_TEST_GLM53_PREFILL)`，
在非 M3 Ultra 上按门控必然拿不到 fused dispatch（上游测试在非 M3 Ultra 主机的缺陷，非合并问题）。

## S0-6 下一步（建议顺序）

1. ✅ **验证并放开 GLM53 调优门控**（已完成，见文末「附三」）：门控已在 NAX 域
   （`ds4_gpu_mpp_available()`，M4/M5）默认放开，输出逐字节不变；实测本机 decode
   **+20%**、prefill **+5~12%**。详见「附三：GLM53 调优门控在 NAX 域默认放开」。
2. **sparse MLA 注意力 tensor-unit 化**（28.3% @41k）：以 #1149（Qwen4 QSA NAX）为模板，
   GLM 侧目标 `kernel_glm_attention_indexed_batch_lora_group8_vec`（纯 SIMD、ALU-bound）。
3. **MTP 经济性**（decode 唯一大杠杆）：66% → 80%+ 接受率 / 更深起草 / verify 成本（37.5 ms）。
4. KDA prefill blocked（7%）；MoE tile/调度；indexer NAX（仅超长上下文）。
5. 待办 cherry-pick：#1073（DeepSeek V4.1）与 #1062（Qwen3.8 batched+MTP）需手工 rebase。

---

# 附二：合并前的完整验证 + 合并记录（2026-09-30）

合并 commit：`0743c7f Merge upstream perf PRs for GLM-5.3-Flash and DeepSeek (M5 Max validated)`
（源分支 `perf/picks-plus-1090` 已删除）。所有数据：M5 Max / 128 GB / GLM-5.3-Flash-Q2。

## 验证闸门

| 闸门 | 结果 |
|---|---|
| `make test`（DS4_TEST_MODEL=GLM53-Q2） | 仅剩 `server` 顺序性 flake；**该 flake 在合并前的 main 上同样失败**（同一断言）；合并还**修好了** main 上的 `local-golden-vectors` / `logprob-vectors` / `metal-ssd-streaming-cache-pressure` 三个失败 |
| `tests/test_glm53_kda` | PASS（two-head dispatch 逐位、UINT32_MAX/越界回退） |
| GLM53 100-case fixture | **avg_nll 0.458177 / first_match 90 / avg_lcp 7.390**；文档参考 0.458030 / 89 / 7.37 |
| GLM53 long 8-case fixture（8192 ctx，全部跨 2051 边界） | **与 main 逐位一致**（0.668884120 / 6 / 2.500） |
| frontier logits 2k..12k（`compare_frontier_logits.py`） | **status: identical，strict float32 identity，全词表有限覆盖** |
| 吞吐（同次 ds4-bench 扫描，FORCE 调优 vs main） | prefill **+19~23%**、decode **+21~23%** |
| 冒烟（合并后 main，-c 65536） | prefill 5200 tok / 10.12 s（514 t/s）、decode 39.2 t/s、日志 0 error |

## 合并内容

- 上游 PR：#1147、#1120、#1093、#1135、#1126、#1122、#1149、#1150、#1090（3 个代码 commit）
- 手工解决的两处冲突：`metal/dense.metal`（Q8_0 matvec 模板参数 `GLM53_KDA_OUTPUT` ×
  `BF16_STORE` 合并）、`metal/dsv4_misc.metal`（attention 实例化：两侧都是新增，取并集）
- 新增修复（针对 #1122 在本机的误拒）：
  - wired 基线**每进程只采一次**（`ds4_test --all` 在一个进程里开多个 engine，后开的会把
    本进程已映射的 90 GB 当成"别的 runtime"）；
  - 将要拒绝时**重采样**（跨进程快速接力时，前一个进程的 wired 页在 <0.5 s 内释放）。
- 剔除：#1127（GLM 短 prompt -19%，其 kill switch 复现确认）。

## 仍未做（下次）

- ~~GLM53 调优门控默认仍只在 M3 Ultra 生效；本机需 `DS4_METAL_FORCE_GLM53_TUNING=1`。~~
  **已放开**：NAX 域（M4/M5）现已默认生效，见「附三」；`DS4_METAL_DISABLE_GLM53_TUNING`
  为总回滚开关。M3 Ultra / test / `DS4_METAL_FORCE_GLM53_TUNING` 三条旧路径不变。
- 上游测试缺陷：`tests/test_glm53_router_shared.c` 未设 `DS4_GPU_TEST_GLM53_PREFILL`，
  在门控未生效的主机必然失败（`test_glm53_kda.c` 有设）。放开 NAX 域后，本机（M5）该测试
  已随门控默认生效而**转绿**——反而成了门控生效的旁证；更老的非 NAX 域主机仍建议补设该 flag。
- 待手工 rebase 的 PR：#1073（DeepSeek V4.1 Metal + DSpark）、#1062（Qwen3.8 batched+MTP）。
- 下一步工作项：**sparse MLA → tensor units**，计划见 `SPARSE_MLA_NAX_PLAN.md`。

---

# 附三：GLM53 调优门控在 NAX 域默认放开（2026-09-30，main）

把 #1090 的 GLM53 调优家族（fused router/shared、DSA top-k、mixed Q8/BF16 输入、KDA two-head、
BF16 pair matmul、qk-lowrank tiled、wide head groups 等）从"仅 M3 Ultra"放开到 **NAX 张量域默认生效**。

## 改动（`ds4_metal.m` · `ds4_gpu_glm53_tuning_available()`）

门控谓词新增一条 `ds4_gpu_mpp_available()`（= `g_metal4_tensor_api_enabled && !g_quality_mode`，
与 MLA-NAX / DSV4-NAX / Qwen4-NAX 同一条设备判据），并新增总回滚开关 `DS4_METAL_DISABLE_GLM53_TUNING`：

```c
return !g_ssd_streaming_mode && g_tp_split_world == 1 &&
    getenv("DS4_METAL_DISABLE_GLM53_TUNING") == NULL &&
    ((g_test_flags & DS4_GPU_TEST_GLM53_PREFILL) != 0u ||
     [g_device.name isEqualToString:@"Apple M3 Ultra"] ||
     ds4_gpu_mpp_available() ||
     getenv("DS4_METAL_FORCE_GLM53_TUNING") != NULL);
```

- **fail-closed 不变**：TP（`g_tp_split_world != 1`）、SSD streaming、quality mode 一律关。
- 三条旧路径（test flag / M3 Ultra / `DS4_METAL_FORCE_GLM53_TUNING`）原样保留。
- 下游各 per-kernel 开关（`_M3_ULTRA_GLM53_DECODE` / `_BF16_NSG4` / `_ROUTER_TOP8` /
  `_ROUTER_SHARED` / `_FLASH_TUNING`）叠在本门之上，行为不变。
- 这些内核都是**输出保持**的速度变体（各自 `return 0 if !tuning` 回退到未调优链），
  精确性与设备无关，被扫的只是性能——性能在 M5 上已扫过。

## 验证闸门（M5 Max / 128 GB / GLM-5.3-Flash-Q2，二进制 = main(ff5b4e0)+本改动）

| 闸门 | 结果 |
|---|---|
| 构建 | `make -j8` + `score_official` 干净通过，无 warning |
| `test_glm53_router_shared` | PASS（240 投毒 draw，fused 路径 memcmp == 未调优参考）——门控在 M5 生效的旁证 |
| `test_glm53_topk_fast` / `_q8_inputs` / `test_glm53_kda` | 全部逐位 PASS |
| `test_glm_attention` | PASS |
| decode A/B（ds4-bench ctx12288/256tok，ON/OFF/ON） | ON 33.77 / 33.58 vs OFF 28.14 t/s → **+20%**，两条 ON 腿非重叠地高于 OFF |
| prefill A/B（ds4-bench 2048..32768，ABBA 各两条） | ≤8k **+11.8%**、32k **+5~6%**，全前沿 ON>OFF，无一条为负 |
| **GLM53 100-case @4096（门控 ON）** | **avg_nll 0.458177271 / first_match 90 / avg_lcp 7.390**（= 0743c7f 合并时 FORCE-on 记录，band 内） |
| **GLM53 long 8-case @8192（ON vs OFF）** | **TSV 完全相同**（0.668884120 / 6 / 2.500） |
| **first-logits ON vs OFF（`--dump-first-logits` + `cmp`）** | **逐字节一致** |

## 决策与归属

- 该改动与 feat 分支的 sparse-MLA NAX **正交**（MLA-NAX 走 `mpp_available` 判据，与本门独立），
  且 main（含 0743c7f 的 #1090）已带全套调优内核 + 字节精确测试，故落在 **main** 上验证/落地，
  比纠缠 MLA-NAX 更干净（此时任何 ON/OFF fixture 差异都只可能来自调优本身，而实测为零）。
- `SPARSE_MLA_NAX_PLAN.md` §9 的"P0=indexer NAX"其 keep-bar(+5% e2e) 超过其 ~4% 天花板，
  且 OMLX §S0-4 已按实测 ROI 把 indexer NAX 降级到 P2；本项（调优门控）是 §S0-6 #1 的最高 ROI、
  最低风险项，故优先。
- 状态：已 commit 到独立分支 `feat/glm53-tuning-gate-nax-domain`（`808800b`，仅 `ds4_metal.m`），
  已 push 到 fork。**上游 PR 挂起**：上游 `antirez/ds4:main` 尚不含 #1090 tuning 基础设施
  （整条在我本地 ahead-23 里），门控改动依赖它才成立，故等该 tuning 栈进入 `origin/main` 后，
  gate 与 sparse-MLA 各开一个 base=`origin/main` 的干净小 PR（各 1–2 commit），**不折叠进 MLA PR**。
  就绪判据：`git fetch origin && git grep -q ds4_gpu_glm53_tuning_available origin/main -- ds4_metal.m`。
  stash@{0} 仍保留原 feat 分支版本，未 pop。回滚：`DS4_METAL_DISABLE_GLM53_TUNING=1` 或 revert 本谓词。
