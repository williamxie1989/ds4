# V4.1 Flash × SSD Streaming：现状分析与下一步路线（2026-10-05）

> 范围：DeepSeek V4.1 Flash Q2（151.77 GiB 主权重 + 188.83 GiB FP8 Engram）× M5 Max 128 GB × `--ssd-streaming`，
> 覆盖 prefill（TTFT）与 decode（steady t/s）两条线。
> 本文 = 现状综合分析 + llama.cpp / vLLM / 研究系统调研 + 路线方案。
> **本轮零模型加载、零推理访问**；所有执行项留给未来空窗，门规沿用各源文档。
>
> 源文档（细节与实测证据都在那里，本文不重复推导）：
> [V41_M5MAX_BASELINE_SPEED_PLAN.md](V41_M5MAX_BASELINE_SPEED_PLAN.md)（BASELINE，decode 域 §12–17）、
> [HANDOFF-V41-STREAMING-SMALL-PREFILL.md](HANDOFF-V41-STREAMING-SMALL-PREFILL.md)（HANDOFF，prefill sweep/gather P0 系列）、
> [V41_STREAMING_ENGRAM_SPEED_PLAN.md](V41_STREAMING_ENGRAM_SPEED_PLAN.md)（ENGRAM，瓶颈模型与 P1-A/P3-A2）、
> [LOCAL_INVENTORY.md](LOCAL_INVENTORY.md)（台账 A–E）。

---

## 0. 一页结论

- **decode 的病已定性并只剩一味药**：miss 读已被完全隐藏（wait 4 µs），GPU 只忙 30%，墙钟 = 主线程 Metal 指令量
  （~1000+ API 调用/token、26 次 CB 轮转）。异步/重叠类药方（卡 H、mailbox、validator）全部实测无效或为负；
  **唯一对口的是图回放（卡 I-2/I-3，Metal graph capture）**，I-1 已落地并 +3.2%/+2.7%（bit-exact，上游 PR #1178 在途）。
  外部调研证实这是所有主流引擎都走过的路（llama.cpp CUDA Graphs 图段 −40%/整体 +10~15%、vLLM V1 piecewise/full CUDA graphs），
  且 ggml-metal 自己也停在同一道坎前（capture 默认关、实验性）——我们做成了就是差异化。
- **物理顶要先讲清楚**：warm decode 的 DRAM 域流量 ≈11.6 GiB/token（ENGRAM §2 实测），
  ~546 GB/s 峰值 ⇒ **本模型 Q2 在本机的架构顶 ≈45 t/s**（驻留参考 39–45 t/s 吻合）。
  I-2 达标线 ~28 t/s 距顶还有 60% 空间；**45 t/s 以上只有改权重字节数（量化/减层）才有**。
- **prefill 的调度层已闭环**（P0c/P0d/P0f，gather 镜像内核默认生效，831 行 append 16 s→5–6 s）；
  剩余肉集中在 **[100,512) 行段的命中率/保温**（生日数学下 gather 省不了字节，瓶颈是 miss 换入）与
  **E1 低行数批量**、**CED 段 decoder 半段专家白读** 两件小事。
- **新肉（外部带来的）**：热专家**校准钉住**（llama.cpp 社区实测文本校准 42% vs 随机 6% 命中）、
  **expert-contiguous 布局重排**（页错误 ×36 改善，对我们是 pread range 数与冷域带宽）、
  **KTransformers 式 expert deferral**（命中专家先算、miss 专家边读边算——即 §7 挂起的 down-proj overlap）。
- **先还账再开新路**：I-1 合入裁定、p42-attn-glue 合/弃、主树 WIP 两个 env 的 A/B 关账，都是一次执行窗口内的事。

---

## 1. 现状快照（锚定数字，勿用旧文档数）

### 1.1 稳态与顶

| 指标 | 数字 | 出处 |
|---|---|---|
| decode steady（fork，auto cache 6959 experts） | 2K 22.8 / 32K 22.0 t/s（ABBA OFF 腿 22.44/21.82） | BASELINE §12.1/§17.3 |
| decode steady（I-1 开） | +3.2% / +2.7%（23.17/22.40），bit-exact | BASELINE §17.3 |
| 上游基线（PR 树，无 fork 融合线） | 18.1 / 17.0 t/s | BASELINE §17.6 |
| DRAM 流量 roofline | ≈11.6 GiB/token ⇒ **顶 ≈45 t/s**（全驻留参考 39–45） | ENGRAM §2 |
| 首 token（12K） | TTFT ~130 ms（I-1 略降） | BASELINE §17.3 |
| 本机 SSD 顺序读 | 12.3 GiB/s（单盘，副本盘已裁决放弃） | HANDOFF 附录 |

### 1.2 prefill（TTFT）现状

| 段 | 路径 | 数字 |
|---|---|---|
| 初始 2048 | 整层 sweep | 14.6–16.4 s（125–140 t/s） |
| append [32, 2048] | **gather 镜像内核，默认开**（P0c/P0d/P0f） | 831 行：冷 6.1 s / 热 4.9 s（P0c 前 16–17.5 s，**×2.2–3**）；1524 行 194–248 t/s |
| append 100 行 | gather（plain mpp 带） | 生产冷域 30.2 t/s（3.3 s），warm 域 100 t/s |
| append [2,32) | 单 token 步进 | ~21–24 t/s ← **E1 靶子** |
| append >2048 | 整层臂（P0f2 已证 gather 平手，帽 2048 勿动） | 129–137 t/s |
| 宽步 ≥4096（CED 默认 4096） | wide sweep + decoder_suffix | 8192 段 ~450–500 t/s；CED 阈值下移 steady +9.3% |

### 1.3 归因终局（decode 域，三条已实锤）

1. **读不背锅**：cache 命中 0.974，miss 读 `missing_wait_avg=0.004 ms`，readahead 池 121 GiB 超读在跑；
   cache 扩容在拐点平台右侧（8000≈auto，4000 −15%）。
2. **等待不背锅**：每层 selected-ids 回读的 0.88 ms "sync" 是 GPU 换出口的假象（validator 免回读实测 −2.7%）；
   卡 H 把等待搬去 worker，墙钟纹丝不动。
3. **主犯 = 主线程指令量**：44 ms/token 墙钟里 GPU 忙 13.3 ms、host 编码 ~10%、CB 完成链外 host 串行 ~43%；
   ~750–900 dispatch + ~90 blit + 26 CB + 数千参数调用，×25–30 µs/次自洽。
   **I-1 只砍了 40 个回读边界就 +3%，验证了模型；剩下的全在 encode/回放里（I-2/I-3）。**

### 1.4 已关账（本文一律不再开题）

cache 扩容 / mailbox+flush+prescan / 副本盘 / NAX-decode matmul（卡 A）/ logits matvec tiling（卡 C，≤3% 可选）/
attention mmp decode（卡 D）/ compute-copy 融合（卡 F 挂起）/ 卡 H early-load / P1 row-batch /
P0f2 窗口 >2048 / P0g 环深度 / E2 单一数值家族（封存）/ 流式 MTP-DSpark（**但见 R3 复议条款**）/
GLM MTP / #1127 layer overlap。证据链见 LOCAL_INVENTORY §D。

### 1.5 在途与待还账（开新方案前先清）

| 项 | 状态 | 处置建议 |
|---|---|---|
| 卡 I-1（`p4-cardI-graph-capture` 376687e+434dd55） | bit-exact 全过 +3%，默认关；上游 [PR #1178](https://github.com/antirez/ds4/pull/1178) OPEN | 等上游 review；fork 侧可先默认开（ABBA 已独立成立） |
| PF-6 上游 PR [#1177](https://github.com/antirez/ds4/pull/1177) | OPEN（GLM 域 +5.2%） | 只等合并，不投入 |
| p42-attn-glue worktree | ≈中性（≥8K +2%），全部未提交 | 一个窗口定夺：合入取位等价，或弃 |
| 主树 WIP（`STREAM_GATHER_SPEC_ROWS`/`GLM53_ROUTED_MPP_PACKED`/`--mtp-timing`） | A/B 未定 | 一次空窗跑完 A/B 并关账 |
| `tests/test_glm53_router_shared.c` 小补丁 | 未提交 | 随手提交 |
| E0.3 契约卫生（streaming 版 `check_short_prefill` 登记红灯） | **已做并实测**（2026-10-04 窗，HANDOFF §15.6） | 新红灯 [9,31] bind 轴待裁决，E1 按纪律停线 |

---

## 2. 外部调研：谁在走哪条路，与我们的差集

### 2.1 图回放 / CPU 开销（decode 域，对应卡 I）

- **llama.cpp CUDA Graphs**（[NVIDIA blog](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs)、[issue #6763](https://github.com/ggml-org/llama.cpp/issues/6763)）：
  bs=1 decode 默认开；图内执行段 −40%、端到端 +10~15%；动态标量走 staging buffer、图内读固定地址——
  **这就是 I-2 §16.4 已计划的抄法，外部数字（−40% 图段）支撑 I-2 的 +25% 线**。
- **ggml-metal MTLGraph capture**（[commit c363256 / PR #20398](https://github.com/ggml-org/llama.cpp/commit/c363256839fdffae184279ad8e5f0f3775c48b77)）：
  `GGML_METAL_CAPTURE_COMPUTE` env，默认关、定位实验（gputrace 采集向）。**同类引擎都停在动态入参/缓冲生命周期这道坎上**；
  我们的动态性来源已被 I-1 收敛成"图外一条 service 线程 + 两个事件 + 一张 addr 表"，比 ggml 的处境好。
- **vLLM V1**（[V1 架构](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture)、[CUDAGraph in V1](https://discuss.vllm.ai/t/cudagraph-in-v1/1016)）：
  - **piecewise CUDA graphs**：注意力等动态 op 切图外、其余分段捕获——与 I-2"每层一岛 + 事件桥"完全同构；
  - **full CUDA graphs**（`full_cuda_graph=True`，FA3/Triton/FlashMLA，纯 decode 批）：注意力也进图，靠 paged-KV 间接寻址——
    我们的 **expert addr 表就是"paged experts"**，I-3（全 token 一图）方向被外部量产实现验证；
  - **Persistent Batch**：输入张量常驻、每步只写差量——I-2 每层动态更新应当做成"写 param buffer 差量"而非重建；
  - **EngineCore 进程隔离**：tokenize/detoken/streaming 与执行环 overlap——server 外围（SSE、tool 解析、checkpoint 落盘）离开关键路径的思路。

### 2.2 MoE 磁盘流式（我们主战场的同类问题）

- **llama.cpp Discussion [#23324](https://github.com/ggml-org/llama.cpp/discussions/23324)（RFC+PoC）**：
  负结果清单与我们高度互证（大 slab 缓存反而慢、低内存机 pin 崩溃、**零拷贝 slot buffer 正确**——正是 ds4 路线）；
  新知识的两点：**expert-contiguous 重布局把页错误降 36×**、**流水化异步读把 SSD 藏在计算后**；
  "bytes/token 是墙"的结论与我们 roofline 一致。PR **#25294 `--moe-stream-cache`**（slot cache + CPU id 重映射）与 I-1 同形态——I-1 的上游化正当其时。
- **llama.cpp Discussion [#27149](https://github.com/ggml-org/llama.cpp/discussions/27149)（Expert-Aware SSD Streaming，Apple Silicon）**：
  **text-calibrated pinning：真实文本 10 token 校准 ⇒ 42% 命中（随机 6%，oracle 42%）；校准+LRU=56%**；
  Q4 小专家（2.2 MB）在 0% 命中下也能被计算全隐藏。对 ds4 的含义：**从生产日志/录制 session 离线标定每层热集并钉住**
  是社区验证过的最高性价比招（我们有现成素材：`e4488c1` 录制回放器 + `ds4_streaming_hotlist.inc` 种子机制 + 1002 生产日志）。
- **KTransformers**（[SOSP'25 论文](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)、[官网](https://kvcache-ai.github.io/ktransformers)）：
  AMX/CPU 部分不适用（统一内存无此分层）；可搬的是 **Expert Deferral**：top-6 拆"立即 5 + 递延 3"，
  递延专家的计算与另一路 overlap——映射到 ds4 = **命中专家先绑先算，miss 专家读完即算**（即 BASELINE §7 挂起未查实的
  "down-proj overlap"子项）；以及它的 CPU-GPU 双流 overlap 编排思想在 I-2 之后（GPU 占比升高时）才值得借鉴。
- **EdgeMoE**（[arXiv 2308.14352](https://huggingface.co/papers/2308.14352)）/ **SP-MoE**（[arXiv 2510.10302](https://arxiv.org/html/2510.10302v2)）/ AdapMoE：
  预测式专家预取 + 投机解码联动。对 ds4：decode 域 miss 已隐藏（无肉）；**prefill sweep 的 miss 被 router 数据依赖锁死**（P0g 结论），
  而 [100,512) 行段生日数学下预测集≈全集——**精确预取结构性不可行，只有"热集猜读"型近似预取**，低预期、限时试错（见 R2b）。
- **MLX 生态**（[mlx-lm #1438](https://github.com/ml-explore/mlx-lm/issues/1438) OPEN、mlx-moe PoC、SharpAI/SwiftLM）：
  同形态竞品在 Apple Silicon 上做 SSD 专家流式 + KV 压缩；无可直接复用的代码（布局/量化/框架三重不兼容，OMLX_WAVE_PORT_ANALYSIS §6 同理）。
- **LMCache / vLLM OffloadingConnector**（[LMCache MP](https://blog.lmcache.ai/en/2026/04/03/lmcaches-new-architecture-boosts-moe-inference-performance-by-10x)）：
  KV 缓存 CPU/盘分层与跨进程池化。ds4 的 disk KV checkpoint + tool-id 精确重放已是单机等价物；
  跨会话/跨进程 KV 池不是单 session server 的目标，**只借鉴其"checkpoint 读写彻底异步化、绝不站在 TTFT 上"** 一条。

### 2.3 调研总结论

外部没有一条能"整搬"的路超过卡 I-2/I-3；ds4 在流式 MoE 的工程完成度（gather 镜像内核、addr 表 GPU 绑定、
逐位契约体系）已领先这批项目半代。**可搬的四块新砖**：①CUDA-graph staging/piecewise 的工程细节（I-2 直接受益）；
②文本校准热专家钉住（R2a）；③expert-contiguous 布局（R5）；④deferral/overlap 的拆分时机（R6，等 I-2 之后）。

---

## 3. 路线方案（R1–R8，各带落点/闸门/预期/风险）

### R1 · 卡 I-2/I-3：整层 encode 进 Metal graph capture（主线，动因与准入已立项）

- **依据**：BASELINE §16（普查、可捕获性分类）、§17.5（I-2 go 判定与三条硬准入）；外部验证见 §2.1。
- **改法**（沿用已立项设计，不重开）：`ds41_decode_island` + `ds4_gpu_decode_graph_begin/end`
  （ds4.c:41539 一带，现门控 tp_world==2 && !streaming）推向 **单卡 + streaming**；每层"1 次回放 + param buffer 差量更新"；
  router ids / expert addr 表留在 GPU（I-1 机制即"paged experts"）；service 线程 + 事件桥留在图外。
- **三条硬准入（§17.4 用真金白银买的）**：动态标量全部 staging buffer 化（token id/position 写固定 buffer）；
  经 addr 表取权重的 dispatch 必须把槽 buffer `useResource` 进图（绑 slab 全集，同 I-1）；
  capture/begin 前先 `ds4_gpu_commands_active()`（flush 会轮换 CB）。
- **顺路子项**：logits 头 argmax 进 GPU（每 token 只回读 token id + 请求的 top-k），把最后一次 host 大回读也关进图。
- **闸门**：§2.3 全套（CLI 贪心逐字节 / logprob ON==OFF / dspark 两测 / test_metal_ssd_experts）→ clean ABBA ≥2 对（2K/32K × 512）。
- **预期/判定**：I-2 目标 steady **≥ +25%**（→ ~28 t/s）；未达标（<+10%）先归因再续 I-3，负结果诚实关账（卡 H 先例）。
- **风险**：capture 与 MTLSharedEvent 的相互作用未证（先做"事件在岛外、岛=单计算段"的最小形态）；
  graph 回放后 GPU 占比从 30%→55%+，新瓶颈会换人——**I-2 落地后必跑一轮 §14.6 式再归因，作为 R3/R6 的开工判据**。

### R2 · 命中率/保温工程（prefill [100,512) 行段的真靶子）

生产冷域 100 行 30 t/s vs warm 100 t/s 的差全部是 miss 换入。三件独立小事：

- **R2a · 热集校准钉住 + 空闲预热（新，外部验证过的最高性价比）**
  - 素材已备：录制 session 回放器（`e4488c1`）、1002 生产日志、`ds4_streaming_hotlist.inc` 种子、route-hotness 衰减半衰机制（715ed7a）。
  - 做法：离线统计每层跨轮共现 top-K 专家（按部署流量画像出 hotlist v2）；引擎侧给 hotlist 槽加 **protected 优先级位**（LRU 先逐非保护槽，
    容量仍 auto，不加总预算）；server 空闲窗（两请求之间）service 线程按 hotlist 预热每层 K 个专家。
  - 外部锚：文本校准 42% vs 随机 6%（#27149）。
  - **bit-exact 论证**：缓存进出策略只改字节的物理位置不改字节（同 P0g 论证），gate 用 §7.2 dump 门构造性证明。
  - 门：`--moe-bind-parity` 27/27 + 223/1524 dump IDENTICAL + 小 cache 压力腿；速度 ABBA 用"多轮 append 模拟"（agent 型固定内容多轮，冷启动），
    判据 100–500 行段 ≥+15%。预期这是 [100,512) 段最大的一块肉（KTransformers 论文的 prefix-cache 三层里这也是打头项）。
- **R2b · 层间热集猜读（预取）——限时试错，低预期**
  - P0g 已证"真实 ids 的跨层预取结构性不可行"；唯一残路是**用 hotlist/上一轮 ids 猜 L+1 层的候选集提前 pread，ids 到达后核对纠正**。
    猜中省下的正是每层 ~2.9 GiB 的串行等待；猜错 = 白读挤占带宽。
  - 纪律：先纸面标定猜中率（用录制 session 重放每层 top-32 候选命中率），**猜中率 <60% 不开工**；开工则一天出原型 + ABBA，负即弃。
- **R2c · gather 域 cache 容量复扫（一次严谨 ABBA 结账）**
  - P0f 的 C/D 单腿其实给出 9000-experts + gather = 211–248 vs auto + gather = 194–212（区间不重叠），当时按噪声记。
    且 cache 与 page cache 抢内存的账（§2.4）在 gather 域换了一套 I/O 模式，值得一次 ABBA 级复扫
    （auto / 8000 / 9000 / 10000 四臂 × {831, 1524} 行 + decode 2K/32K 不回退约束）。
  - 注意 argo/llama.cpp 双方面的内存崩盘教训：≥10000 臂必须盯 swap，预检不过不跑。

### R3 · 流式 MTP/DSpark 裁决复议（不是重开，是改判前置）

- 放弃依据（ENGRAM §9：verify 93 ms/轮 ≈ 单步成本、盈亏平衡 accept 2.96 vs 实测 1.33–1.91）的成立前提 =
  **host 指令量主导的 44 ms 域**。I-2 落地把墙钟压到 ~20 ms 后，verify 批行的边际成本构成会变
  （host 编码不再随行数线性放大；Engram/static/attention 摊薄占比上升；routed 并集 ~4.7× 的账不变）。
- 做法：**I-2 收账后用新成本模型重推一次盈亏平衡**（纯桌面工作 + 现成 `--mtp-timing` 与 mtp_ledger_replay 工具）；
  平衡点 ≥ accept 上限才申请重启管线，否则把 ENGRAM §9 的裁决补上"适用域 = host-bound"四个字后继续封存。
- 不抢 R1 窗口，顺序上必须在 R1 之后。

### R4 · prefill 小项打包（各半天级，互不冲突可穿插）

- **E1 · [2,32) 行批量走 gather**（HANDOFF §15.3 已设计完）：前置 E0.1 微测（mv 族行数独立性，addr-vs-whole-map 现成框架）；
  过则 admission 32→2，数值落 scalar 家族（契约零变化）；收益 = 20–31 行 append 从 ~21 t/s 步进到批量带（参照 223 行 81 t/s，预期 ×2–3.5）。
- **P2-CED · decoder 半段专家不白读**：wide/CED 下 layer 39 只算 ~128 行仍照读 3.73 GiB/层——
  prepare 的 spans 改 suffix-aware（decoder_suffix 生效时 layer 20–39 用收缩行数决定 gather miss 集）。
  只影响 ≥4096 宽步；粗账 = wide sweep I/O 省三到四成，TTFT(8192) 预期 −15~25%。落点：`decoder_suffix` 段 ×
  `metal_graph_stream_prefill_layer_pagein_start` 的 spans 选择；数值不动，dump 门照过。
- **E0.3**：streaming 版 `check_short_prefill` 红灯登记（契约卫生）——已落（2026-10-04）：新臂 `--short-prefill-ssd-rows` + bind-parity 行集补 9/16/31；实测见 HANDOFF §15.6。

### R5 · 读路径：expert-contiguous 布局重排（新，先微测本说话）

- 现状：单专家 9.49 MiB 分布在 gate/up/down 三张量（3–4 个 pread range/专家）；sweep 每层 ~100 ranges；
  实测 10.9/12.3 GiB/s——**顺序带宽接近饱和但 range 粒度粗**。
- 外部证据：#23324/#27149 的 contiguous 重布局把页错误降 36×、冷读吞吐 3.0+ GB/s（在 PCIe 3.0 弱盘上是决定性项）。
- 我们的适用性打折（NVMe 快、无 mmap 分页、已有线程池），**所以第一步不是改引擎，是 200 行离线工具**：
  `gguf-tools/` 出一个"布局转换 + 读微测"——把 V4.1 的 routed 专家改写成 per-expert 连续块，
  用现有 pread 池微测 1-range vs 3–4-range 的实际 GiB/s 与延迟差。**微测 ≥+8% 才立项改 loader**（GGUF 加 vendor 元数据表 +
  双布局读取分支；不改任何字节数值，canonical GGUF 保持可跑）。
- 收益方向：prefill 冷域（miss 换入）与 R2 全族的放大器；decode 域读已隐藏，预期≈0。

### R6 · Expert deferral / down-proj overlap（等 R1 之后）

- BASELINE §7 唯一未查实的读路径子项："常驻命中专家的 down 投影与 miss pread overlap"。
  KTransformers 的 5+3 immediate/deferred 拆分给了结构参考。
- 前提：R1 后主线程不再是 44 ms 主犯，重排 dispatch 顺序的收益才会露出来；现状下它和卡 H 同墓（重叠解决不了指令量）。
- 触发条件：I-2 收账且归因显示 GPU 关键路径上存在"等 expert ready 事件"的成串空泡时才做。

### R7 · server 外围 overlap（小，蹭窗口）

- vLLM EngineCore 思想的最小单机化：把 detokenize、tool-call 解析、SSE flush、KV checkpoint 落盘从请求关键路径挪进
  独立线程/延后提交（checkpoint 已是单 session 常驻 + 落盘，确认落盘不与下一请求 prefill 抢盘即可）。
- 审计先行：现网日志找 TTFT 里的非引擎成分（历史上 18 t/s 冤案就是平均口径）；**很可能审计结果是"没肉"，那就记档收兵**。

### R8 · 不动 bit-exact 契约就不许碰的清单（备案，默认不做）

- 降权重量化（IQ1*/混合位宽、共享专家再压缩）：roofline 11.6→~7 GiB/token，顶可抬到 ~70 t/s 级，
  但输出改变、golden vectors 全量重生成、test-vectors 重定基线——**产品决策，不是工程决策**。
- 第二块 NVMe 副本（基线 §8 语义已设计完，等有盘）；双机 TP（island 机制已在，跨机收益未验）。

---

## 4. 执行顺序与依赖

```
0) 还账窗口（半天级）：I-1 fork 侧翻默认 / p42-attn-glue 定夺 / 主树 WIP A/B 关账 / E0.3
1) R1  I-2 工作树（基线 = I-1 合并后的 main）——主线，两段闸门
2) 穿插（与 R1 无代码冲突）：R2a hotlist v2 + 保护位、R2c 容量复扫、R4 三小项、R5 布局微测工具
3) R1 收账 → 立即做一轮 §14.6 式再归因（新瓶颈画像）
4) 按再归因放行：R3（MTP 复议）/ R6（deferral）/ R5 引擎侧 / R2b（仅当猜中率达标）
```

- 每项独立 worktree + 独立 env（默认关）+ §2.3 全套 bit-exact → clean ABBA ≥2 对，CSV/日志入 `speed-bench/`；
  负结果按 LOCAL_INVENTORY §D 惯例登记"别再试"。
- **两个域的门不要串线**（§11 红线 + P0g 教训）：decode 域看 steady t/s，prefill 域看 append 墙钟 + 逐层 pread profile。
- 逐位契约一切照旧：改数值家族的（无）必须先报备裁定；缓存/调度/图形态的靠构造性 dump 门。
- 执行窗口铁律：模型加载类一律后台 job + 预检（swap/pgrep/preflight）；单实例 flock，测量要 server 停窗。

## 5. 预期台账（诚实区间）

| 路线 | 对象 | 预期 | 依据 |
|---|---|---:|---|
| R1（I-2，目标线） | decode steady | **+25%（23→28）** | §16.1 上限分析 + CUDA graphs 外部 −40% 图段 |
| R1（I-3，若达标续） | decode steady | 另评（目标 33+） | 墙钟逼近 max(GPU, DRAM) |
| R2a | prefill [100,512) 段 TTFT | +15~30%（冷域） | #27149 校准命中 6→42% 社区实测 |
| R2c | prefill 1524 段 | 0~+8%（定夺性实验） | P0f C/D 区间不重叠的疑点 |
| E1 | append [2,32) | ×2–3.5 | 223 行 21.4→81.6 t/s 先例 |
| P2-CED | TTFT(≥4096) | −15~25% | decoder 半段白读字节占比 |
| R5 | prefill 冷域 | 微测定（0 或 +5~10%） | 本机 NVMe 已近饱和 |
| R3/R6 | — | 待 R1 归因后评估 | — |

无硬件条件下的合计目标：**decode +25%（R1 兑现），[32,2048] append 段 TTFT 再 −20~30%（R2/R4 叠加）**。
距 DRAM roofline（~45 t/s）的剩余差距，非模型侧（R8）不可越。

## 6. 参考出处

内部：BASELINE §12–17、HANDOFF §12–17、ENGRAM §2/§9、[OMLX_WAVE_PORT_ANALYSIS.md](OMLX_WAVE_PORT_ANALYSIS.md)、[LOCAL_INVENTORY.md](LOCAL_INVENTORY.md)。
外部：
- NVIDIA《Optimizing llama.cpp AI Inference with CUDA Graphs》 developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs
- llama.cpp issue #6763；commit c363256（ggml-metal capture env，PR #20398）
- llama.cpp Discussions #23324（MoE offload RFC+负结果清单）、#27149（Expert-Aware SSD Streaming：contiguous 布局 ×36、校准 pinning 42%）；PR #25294（--moe-stream-cache）
- vLLM V1（Red Hat 博文：Persistent Batch / piecewise CUDA graphs / EngineCore）；discuss.vllm.ai "Cudagraph in V1"（full_cuda_graph 支持面）
- KTransformers SOSP'25（Expert Deferral 5+3、CPU-GPU overlap 编排）；ktransformers 官网 updates
- EdgeMoE（arXiv 2308.14352）；SP-MoE（arXiv 2510.10302）
- mlx-lm issue #1438（Apple Silicon 专家流式 FR，引 mlx-moe / SwiftLM）
- LMCache MP 架构博文（KV 盘分层 / checkpoint 异步化参照）
