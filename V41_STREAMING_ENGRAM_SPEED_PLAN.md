# DeepSeek V4.1 × ds4 × SSD 流式 + Engram 磁盘驻留：prefill/decode 提速方案

> 设计日 2026-09-30。机器：**M5 Max / 128 GB（NAX 域）**，本机 `gguf/DeepSeek-V4.1-Flash-Q2.gguf`（341 GiB，
> 主权重 ≈151.77 GiB + FP8 Engram 表 ≈188.83 GiB，层 1/14，rows 384,006,168 / 384,016,682，
> `e4m3_e8m0_32_row264`，见 [ds4.c:6935-6952](ds4.c)）。
> 姊妹篇（先读）：[V41_M5MAX_SPEED_RESEARCH.md](V41_M5MAX_SPEED_RESEARCH.md)（上游 PR 全景与实测口径）、
> [OMLX_WAVE_PORT_ANALYSIS.md](OMLX_WAVE_PORT_ANALYSIS.md)（omlx 机制清点与五桶方法论）、
> [SPARSE_MLA_NAX_DSV4_PLAN.md](SPARSE_MLA_NAX_DSV4_PLAN.md)（NAX 注意力移植）。
> omlx 侧证据：`~/omlx-notes/dsv41_flash_speedup.md`、`dsv41_upstream_perf_port_analysis.md`、
> `dsv41-flash-prefill-speed-handoff.md`。
> **本文不重复上列文档已核对的数字，只引用并标注来源。**

---

## 0. 结论速览

1. **ds4 的 V4.1 流式栈比想象中完整**：宽步 layer-major sweep、CED 等价物（`decoder_suffix`，
   [ds4.c:42335](ds4.c)）、encoder 常驻（`ds41_encoder_acquire`，[ds4.c:42109](ds4.c)）、
   Engram 预取线程 + 异步 step 读、expert cache seed/hotness、decode queue（#1041 已并）都在。
   **不要重复造这些**。
2. **真正的三大缺口，全部集中在"流式模式被一刀切关掉的优化"上**：
   - **MTP/DSpark 与 `--ssd-streaming` 硬性互斥**（[ds4.c:72519-72522](ds4.c) 引擎门 +
     [ds4.c:86116](ds4.c) `drafting = !g->streaming`）。128 GB 机器上 V4.1 必走流式，
     于是 decode 端最大杠杆（每 cycle 多吐 token）在这台机器上**结构性缺位**。
     omlx 的生产形态是 expert offload + engram offload + MTP 同开——这是要移植的核心能力。
   - **`ds41_measured_config` 在 `g->streaming` 下整体返回 false**（[ds4.c:41782-41795](ds4.c)），
     连带 Metal 的 V4.1 精确融合族（attention epilogue / qb_bf16 / gather reuse / verify rows，
     [ds4_metal.m:10545-10550](ds4_metal.m) 的 `g_ssd_streaming_mode` fail-closed）与
     **Engram step 异步 overlap、decode pipeline 免等待 commit** 全部关闭。
     这些是调度/次序改动（数值不变），关它们的理由只是"未在流式域实测"——
     **GLM53 调优门控放开（OMLX 文档附三）就是同形态先例，可照抄流程**。
   - **Engram 2×24 行随机读在流式 decode 里是同步等待**（overlap 需 `measured_m3_ultra`，
     [ds4.c:41857-41865](ds4.c)），而流式 decode 每 token 本就有 expert miss 读要重叠，
     这两类 I/O 互相不知道对方存在。
3. **prefill 侧的大机制 ds4 都有，缺的是流式域的阈值标定与实测**：CED 触发门槛
   （`total_count ≥ 8192` 且 wide）高于 omlx 实测有收益的 4096 域；encoder 常驻门槛
   `remaining ≥ 16384`；宽步 8192 要 `ctx ≥ 16384`。三者都该在流式域重扫边界。
4. **数字口径纪律（沿用 perf_lessons）**：omlx 的 CED +16.6%、宽步 +10.5% 都是
   **驻留 MLX 基线相对值，不得跨框架跨驻留态引用**；同模型驻留/流式差 5–12×。
   每个候选先交本机流式域的五桶/roofline 行，A/B 区间不重叠才宣称。

> **落地进度（滚动）**：P0 全清（§6.1–6.4）；**P1-A 调度门放开已实施 + 位一致 3/3 + steady +2.5~4%**
> （§6.5 P1-A，M5 域默认 ON，回滚 `DS4_METAL_DISABLE_V41_DECODE_PIPELINE_PORTS=1`）；
> **P3-A 阈值矩阵完成——prefill 由"sweep 行数 × CED"完全解释（2K 92 / 4K 117 / 8K 235 / 16K 426 t/s，
> CED 在 8K/16K 各 +26%/+51%）**（§6.5 P3-A）；**P3-A2 阈值重扫完成——P3-A"阈值不动"被实测推翻半条：
> 4096 行本来就 wide 可达，8192 字面量门是唯一阻塞；CED 阈值 env 化并默认下移 4096（steady ABBA +9.3%
> 簇间不重叠；隔离位一致 2/2；敏感档 fixture nll 双向漂移 ≤2.5% 零文本翻转；回滚
> `DS4_V41_DECODER_SUFFIX_MIN_ROWS=8192`）**（§6.5 P3-A2）；优化窗口收窄为 <4096 续推左端；
> **P3-B 核查后裁掉**（Metal indexer 已 clamp，零改动）；**P2-1 蓝图已解析本地 V4-Flash DSpark
> support schema**（81 张量 / mtp.0-2 + markov_head，V4.1 版为同 schema 参数替换，转换器脚手架
> `speed-bench/build_dspark_support_gguf.py` 已备未跑）；
> **P1-A2 已实施**：ATTN_EPILOGUE+QB_BF16 流式域默认 ON（位一致 2/2、四臂 A/B wall-clock
> 中性、−55 dispatch/step；GATHER_REUSE 审计后保持关闭——跨层 scratch 生命周期不可证）；
> **P2-0 判据跑穿**：verify 并集实测重叠极高（6→22.24/层）且 LRU 双账 x1.00——
> **expert 账本不否决 MTP，阻塞改判为"V4.1 support 资产 + 真实 accept 率"**。
> **P2 解除暂缓、流式 DSpark 施工推进到"管线全通、验证段最后一段"**（§7）：JigSawPT 资产
> 7.43 GiB 下载校验 ✅、转换器出 ds4 合同件 78/78 绑定零错 ✅、三道流式闸 env opt-in ✅、
> draft cycles/seed/capture 在流式域实跑 ✅、输出无损逐字节验证 ✅；剩余卡点已实锤定位：
> `ds41_graph_step_batch_spec` verify 行批的主干 MoE 走直读 view（流式域无此 view），需接
> 流式专家缓存 gather（对齐 prefill 行批已通路径），内核级一天。
> 剩余待办：**P2 verify 批 gather 重构（decode 提速兑现点）**、P1-B（#1035/#1034）、
> P1-C mailbox、以及 <4096 续推左端的攒批方案（需真实负载占比）。

---

## 1. 现状盘点

### 1.1 ds4 V4.1 流式 prefill 的三级路径（代码考古）

`ds41_prefill_count`（[ds4.c:42040-42098](ds4.c)）+ `ds41_graph_prefill_sweep`
（[ds4.c:42317](ds4.c)）+ `ds41_encoder_acquire`（[ds4.c:42109](ds4.c)）构成 antirez 所述
"三路径自动选择"的实现（对应 V41 调研 §1.1）：

| 路径 | 触发 | 机制 | 关键开关 |
|---|---|---|---|
| token-major（类 decode） | `remaining < minimum`；expert 缓存 ≥ 半量且续写时 minimum=1024 | 走 decode 图逐 token，吃热缓存 | `DS4_METAL_DISABLE_V41_LAYER_PREFILL` |
| layer-major 宽 sweep | `remaining ≥ 4096` 且 `carry_cap` 足够 | 每层全量映射一次扫完，expert 每层读一次；seed 回 decode cache | `DS4_METAL_DISABLE_V41_WIDE_PREFILL` |
| encoder 常驻 | `streaming && remaining ≥ 16384` | wire-lock 层 0–19 全部 routed expert（借 expert cache 预算，不加第二份），decoder 半可 defer | `DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY` |

CED 等价物 `decoder_suffix`（wide 且 total ≥ 8192）：层 20–39 只算收缩尾部
（每层少 127 行，层 39 只剩最后 128 行），global KV 由层 20 从 encoder-final hidden 投影
（[ds4.c:42411-42419](ds4.c)）——与 omlx CED（#3607；decoder 半只送 window=128 尾行、
midpoint CSA2 L20 产 global KV，`~/omlx/omlx/patches/deepseek_v41/config.py:103-105`）
**语义一致**。omlx Studio 驻留域实测 pp4096 +7.6%、pp8192 +16.6% vs 宽步基线；
ds4 侧流式域**从未锚定复测**。

Engram：表永远只在磁盘（uncached fd，无 mmap/Metal 视图，[ds4_engram.h:47](ds4_engram.h)）。
prefill 有专用预取线程（`ds41_engram_prefetch`，total ≥ 1024 时启用，
[ds4.c:42356-42360, 42375](ds4.c)）；decode 有 begin/end 异步 step 读，但 **overlap 分支门在
`measured_m3_ultra`**（[ds4.c:41845-41865](ds4.c)）——本机流式域走同步 `ds4_engram_read_batch`
（有界并发 pread，等待 ≈1 次随机读）。

### 1.2 ds4 V4.1 流式 decode 的现状

- 走 `ds41_graph_step`（[ds4.c:41806](ds4.c)）+ expert cache gather；
  hit/miss 统计已有（[ds4_metal.m:4588-4631](ds4_metal.m)），**P0 直接用它拿命中率**。
- decode queue（#1041，逐层不 wait、每 token 一次 CB wait）tp1 下不分流式，**已生效**
  （[ds4.c:41830-41838](ds4.c)）；但其上的 pipeline 变体（免等待 + Engram 提前起读）门在
  `measured_m3_ultra`，流式没有。
- Metal 侧 V4.1 精确融合族在 `g_ssd_streaming_mode` 全部 fail-closed
  （[ds4_metal.m:10545](ds4_metal.m)）。这些核是"输出保持的速度变体"（GLM53 附三同款语义），
  门只管性能。**候选放开项**：`V41_ATTN_EPILOGUE`、`V41_QB_BF16`、`V41_GATHER_REUSE`、
  `V41_VERIFY_HEAD_ROWS/ATTN_OUT_ROWS`（verify 相关两项只对 MTP 有意义，归 P2）。
- 实测口径（V41 调研 §1）：Q2 流式 decode warm **14.8–16.4 t/s**，冷专家首触衰减到 6.5 t/s；
  同机天花板参照（V4 Flash Q2 全驻，非同款模型）39–45 t/s，差 ≈2.5×。

### 1.3 MTP/DSpark：本机当前 = 无

两道门（都在 [ds4.c](ds4.c)）：

```text
engine open（72519-72522）:  V4.1 且 (dspark || mtp_path) ⇒ 必须 Metal && !ssd_streaming && tp0
spec cycle（86116）:         drafting = !g->streaming && ...   // streaming 直接退回单 token
```

旁证：`docs/SPECULATIVE_DECODING.md` 说 "On Metal, the main model can be resident or
SSD-streamed"——那是 **V4 Flash** 的 DSpark；V4.1 的门是后来收紧的，注释写明理由：
*"DSpark drafts for the resident, single-device Metal graph, whose verification equals
one-token decode"*——即 verify 多行在流式下的专家读放大**未做过**，不是已知会亏。

DSpark 几何（本机 Q2 metadata 实测）：block 5、Markov rank 256、draft 层 37–39、
draft 专家 128 top-3 → **verify ≤ 6 行，恒走 `v41_decode_batch` ≤8 行融合 decode**
（[ds4_metal.m:44838-44844](ds4_metal.m)，接受 IQ2XXS+Q2K / Q4K+Q4K / MXFP4 三族专家配方）。
**P0 必须先确认本机 Q2 的 `ffn_gate_exps/down_exps` 类型落在哪一族**（决定融合路径可用性）。

权重可用性：`download_model.sh` 无 V4.1 DSpark support 目标，本机 `gguf/` 只有 V4/Vision 的
support 文件。上游 `antirez/deepseek-v4.1-flash-gguf` 是否已发布配对 support GGUF 需查证；
若缺，路径是照 `DeepSeek-V4-Flash-DSpark-support-0731.gguf` 的 schema 从 HF draft 层自制
（`gguf-tools/`，元数据参数主文件全带，缺口只是张量文件）。

### 1.4 omlx 侧对照（本方案要移植的能力面）

| 能力 | omlx（证据在 ~/omlx-notes） | ds4 现状 |
|---|---|---|
| Engram 磁盘驻留 | `deepseek_v41_engram_ssd_offload` knob | ✅ 更强：默认盘读 + 预取/异步双形态 |
| CED prefill | #3607，驻留域 pp8192 +16.6% vs 宽步；三道硬闸（layout/verify/MTP-draft raise） | ✅ 有 `decoder_suffix`，流式域阈值未标定 |
| 宽步 8192 | 本地 main-retention，+10~15%，门 native+≥64GB | ✅ 有（ctx≥16k），另有 encoder 常驻 |
| **MTP + expert/engram offload 同开** | **生产形态**（Studio 422GB offload 配置常跑；verify ≤6 行短路径） | ❌ 硬互斥 |
| MTP 与 CED 共存 | ✅（draft 段不走 CED，verify 照常） | 移植时照此设计 |
| MTP 经济性 | GLM53 对照接受率 79.7–84.6%、2.0–2.3 tok/cycle；V4-Flash 在 M5 Max 仅 66%（#913） | V4.1 DSpark 在本机**从未测过** |

### 1.5 上游 PR 未挖矿（引用 V41 调研 §2，勿在此展开）

decode：#1042（再 +20% 融合）、#1035（Engram 并行读 +7.4%）、#1034（流式 decode +11%）、
#1060/#1061（长上下文 indexer）；argonaut 单项：mailbox router 回读 +6.6%、多盘副本 +19%、
evict 预扫描 +1.5%、BF16 舍入折叠 +1.5%。prefill：#758（indexed attn −11.3%）。
本 fork 已带 #1041/#1090/#1147 家族（`0743c7f`/`9bc0bd7`/`2ce4e40`）。

---

## 2. 瓶颈模型（P0 已实测回填，2026-09-30）

**decode 侧内存拓扑（实测锚定）**：本机 expert cache 80.51 GiB = **8,685 槽（56.6% 槽位覆盖）**；
warm 续推 sweep 后 per-layer 命中率 **0.834**（stats2 run，8K 域；每层常驻中位 ~230/384 槽）。
每 token DRAM 域读取 ≈ static 9.37 GiB + cached experts ≈2.22 GiB（40×6×9.49 MiB 的命中部分，
miss 多数走 page cache，`miss_willneed/evict_dontneed=0`，miss 仅 pread 13.25 GiB/整轮）
→ ≈11.6 GiB/token；24 t/s ⇒ **实测有效带宽 ~278 GB/s ≈ 50% of ~546 GB/s 峰值**。
**裁决：warm decode 不是 SSD-bound，是 DRAM 域 + 调度空隙混合**（一半峰值带宽被
逐层调度/小核/Engram 同步等待吃掉）。因此：
- **P1（重叠/融合族放开）直接抢回调度空隙 → 有明确的上行空间（→ 35+ t/s 方向）；**
- **P2（MTP）的收益要修正**（见 §3 P2）：cache 命中场景下权重读也在 DRAM 域，verify 批行的
  专家流量**不是乘 6 而是并集去重（~4.7×）**，与接受率 2.2 tok/cycle 基本打平——
  MTP 的净收益主要来自 static 权重/Engram/注意力的摊薄与每 cycle 一次 prefill-cache seed，
  **不能沿用"DRAM 型默认打平"的一句话结论**，P2-0 离线重放必须做且要含 DRAM 字节账。
- 冷启动 prefill（--ssd-streaming-cold，续推臂 60–72 t/s vs warm 84–97）说明 prefill 侧才吃 NVMe。

四桶采集方式（P1/P2 A/B 复用）：

| 桶 | 采集方式 |
|---|---|
| expert cache 命中率/miss 字节 | `DS4_METAL_MEMORY_REPORT=1` + `DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1`（v41 图释放前 dump，已加测量行 [ds4.c:40307-40312](ds4.c)） |
| decode 逐阶段 µs | `DS4_METAL_CB_TIMES`（server/short run stderr） |
| prefill 阶段 | wide1 型单发大 prefill + `--csv`（已建方法） |
| 交互态 | gaptest（已建：`/tmp/v41bench/gaptest.py`，M5 Max 无 power-wake 税） |

---

## 3. 方案（阶段化，每项带落点/闸门/回滚）

### P0 · 测量与裁决（先行，1–2 天，产出四张表）

1. **基线矩阵**：V4.1-Q2 `--ssd-streaming`，`ds4-bench`（promessi_sposi，2K→32K→64K，
   gen 128）冷/热两臂；带 gap（2 s/6 s）交互吞吐一臂。每臂锚定日志行：
   `V4.1 prefill loading ... encoder experts`、`ds4: V4.1 prefill layer=... encoder_resident=`、
   expert cache stats、effective cache bytes。**cold/warm、是否锁频、量化类型必须随数字标注**
   （冷 prefill 可崩 9×，V41 调研 §1.1）。
2. **专家类型确认**：`--inspect` 读 `blk.*.ffn_gate_exps/down_exps` 类型 → 决定
   `v41_decode_batch` 族可用性（§1.3）。
3. 五桶 + roofline 表（§2 四桶 + `DS4_METAL_ENCODER_TIMELINE` 逐核占比 @2048/@16k prefill）。
4. **DSpark support 可得性查证**：上游 HF 仓库 file listing；缺则写自制 GGUF 的工期评估。

**闸门**：交表即完成；表的判决结果决定 P1/P2 内部排序（按 §2 规则）。

### P1 · decode 快赢（bit-exact/输出保持类，沿 GLM53 附三先例）

**P1-A · 把 `measured_config` 拆分出"调度类"与"核类"两组门**（✅ 调度类已完成 2026-09-30，
实测 +2.5~4% steady、位一致 3/3，见 §6.5。真相修正：该门还含"设备名 == Apple M3 Ultra"与
"全层同质 Q4_K/MXFP4"两条，本机 Q2 在 resident 下同样从未生效；decode flush 本来就不受
measured 管辖。剩余的融合族逐 kernel 放开拆给 P1-A2。）
- 现状把三类东西塞进一个布尔：①Engram step 异步 overlap、②decode pipeline（免等待 commit）
  ——**次序变化、数值逐位不变**（Engram 注释自证 "The values are the same either way"）；
  ③Metal 融合族——输出保持变体。
- 落点：[ds4.c:41845/41857](ds4.c) 把 `measured_m3_ultra` 门拆成
  `streaming_ok = queue_layers && !getenv("DS4_DISABLE_V41_ENGRAM_STEP_OVERLAP")` 形态的新判据
  （Engram 表/缓冲与 `engram_rows_late` 的双输入条件已具备）；Metal 融合族则在
  [ds4_metal.m:10545](ds4_metal.m) 的 `ds4_gpu_dsv41_exact_admitted` 加一条
  `|| (g_ssd_streaming_mode && ds4_gpu_mpp_available() && !getenv(...))` 白名单，逐 kernel 打开。
- 验证：`--dump-first-logits` 逐字节对拍 + logprob 向量 + decode A/B（ABBA，流式 warm 域）。
  预期：流式 decode 每 token 少 1 次 Engram 随机读等待 + 40 次调度空隙收窄；
  融合族每项 1–3% 级。**每项独立开关，独立 A/B，不打包宣称。**

**P1-B · 上游 bit-exact 单项评估**：#1035（Engram 并行读）、#1034（流式 decode 重构）
优先试——直接命中流式域（#1034 作者口径 M2 Ultra Q2 流式 +11%）；随后 #1042。
cherry-pick → 本机流式域 A/B → 只留有非重叠区间的。

**P1-C · router 回读 mailbox 化（argonaut 单项，自报 +6.6% steady）**：
ds4 已有 shared-event 基础设施（`ds4_gpu_signal_selected_readback_ready`，
如 [ds4.c:26901](ds4.c)）。把"读回 router top-6 ids 再决定映射哪些 expert"改为
GPU mailbox + shared expert 先编码 + miss 读与 GPU 并行 + 超时兜底。这是新机制（非 cherry-pick），
放在 P1-A/B 之后，因它改动 commit 时序，需要与 P1-A 的 pipeline 变体合测。

**P1-D · Engram×expert I/O 协同**（本机特有，小项）：decode 的 Engram 读与 expert miss 读
目前互不知晓；`ds4_engram_read_step_begin` 已允许提前发射——在 streaming 路径把 begin 提到
上一层 encode 之前（而非 token 开头），并给 expert page-in 线程与 Engram pread 各设
QoS 预算（避免 4K 随机读互相插队）。P0 的 timeline 占比 >3% 才做。

### P2 · MTP/DSpark × 流式（最大杠杆，omlx 独有形态；P0/P1 之后主攻）

**为什么值得**：流式 decode 的固定成本（常驻权重 A、Engram E、attention、调度）与
cycle 行数近似无关——verify 6 行只让 **M** 变；接受率 ≥0.6 时每 token 摊薄显著。
粗算（用 P0 实测回填）：verify 6 行 top-6×6=36 槽/层，行级并集去重后 ~25–30 槽；
按 2.2 tok/cycle，每接受 token 专家槽 ≈ 12（对比 serial 的 6）——**M 上限约 2×，
而 A/E/attn 全摊到 1/2.2**。净收益符号取决于 miss 率与并集去重率，P2-0 用离线重放先算。

- **P2-0（纸面+离线，先于一切代码）**：写一个离线脚本（tests/ 或 gguf-tools/）——
  用 expert profile 采集（`g_expert_profile` 已有）抓 N 个真实 token 的 40 层 top-6 集，
  模拟 2/4/6 行 verify 的并集大小与给定 cache 覆盖率下的 miss 字节曲线。
  **P0 后修正口径（§2）**：warm 域 expert 读主要在 DRAM（hit 0.834），曲线必须同时记
  **DRAM 字节与 SSD pread 字节两本账**；**判据（修正）：每接受 token 总字节 ≤1.3× serial
  且 SSD pread ≤2.2× serial 才立项**；否则降级为"draft-only 常驻
  半驻留"方案（encoder 半常驻 + DSpark 只在驻留段开）并重新评估。**
- **P2-1（权重）**：按 P0-4 结果获取/自制 V4.1 DSpark support GGUF；`--mtp-model` 装载路径
  复用 V4 Flash DSpark 的既有代码（`e->dspark_weights`），确认 IQ2XXS+Q2K 族在
  `v41_decode_batch` 覆盖内（§1.3）。
- **P2-2（门放开）**：引擎门（[ds4.c:72519](ds4.c)）与 `drafting`（[ds4.c:86116](ds4.c)）
  增加 streaming 白名单：`Metal && tp1 && 无 imatrix/quality/images`。逐段验证顺序：
  ①verify 批行在流式 gather 下正确性（`v41_decode_batch` 路径的 expert slot 预算按并集数放大）；
  ②draft 3 层的驻留（support 权重小，常驻不入流式 cache）；③rewind ring 与
  `raw_log_from` 约束（[ds4.c:42346](ds4.c)）在多行接受下的推进；④KV checkpoint 语义
  （多行接受的 checkpoint 截断已有 resident 实现，流式只需走同一路径）。
- **P2-3（流式专属调度）**：draft→verify 的**专家预取桥**——draft 完成后、verify 编码前，
  把 verify 行 router ids（draft 已算出）交给 `metal_graph_stream_prepare_start` 做 miss
  page-in（prefill 已有同款 prepare 机制，[ds4.c:42400-42406](ds4.c)）；
  接受后 `seed` decode cache。这条是 omlx 没有、ds4 独有基础设施的增量收益点。
- **P2-4（经济性验收）**：三 prompt 族（代码/推理/长写作）× 温度 {0, 1}（exact 与
  opportunistic 各一臂）× warm/冷；**engage 判据：tok/s 提升且区间不重叠、接受率 ≥65%、
  冷缓存场景不劣化**。回滚：CLI 不传 `--dspark` 即回到现状（默认不变，保守落地）。
- **已知风险**：#913/#1002 记录 DSpark 在 M5 Max（V4 Flash，驻留）零收益——那是
  66% 接受率 + verify2=1.28× 的组合；V4.1 DSpark 是 block-diffusion 草稿（≤6 行恒短路径），
  形态不同但**必须用接受率实测裁决**，P2 每一小步都是独立 gate。若接受率在本模型族始终 <60%，
  P2 止损，转 P4 的量化杠杆。

### P3 · prefill（流式域阈值标定 + 机制增量）

- **P3-A · CED/宽步/encoder 常驻三阈值重扫**（纯参数实验，先于任何代码）：
  `DS4_METAL_DISABLE_V41_DECODER_SUFFIX`/`_WIDE_PREFILL`/`_WIDE_CHUNK`/`_8K_CHUNK`/`_ENCODER_RESIDENCY` 五开关 ×
  pp {2K,4K,8K,16K,32K,64K} 全矩阵，产出本机"prompt 长度 → 最优路径"曲线。
  若 decoder_suffix 在 4096–8191 段也有收益（omlx 驻留域 4096 已 +7.6%），
  把 [ds4.c:42335](ds4.c) 的 `total_count >= 8192u` 降到实测拐点；宽步 cap 的
  `ctx < 8192 → 2048`、`ctx < 16384 → 4096` 两级门（`ds41_prefill_limit`，
  [ds4.c:40200-40203](ds4.c)）同理。
- **P3-B · indexer 候选宽度 clamp**（omlx S2 副产品线索，端到端 ~1–3% 估）：
  上下文 <16k 时候选 2048 块×8 未用满，clamp 到实际池块数省 4 个候选层 ~50% 打分。
  omlx 后续撤回过该线索（其原生核已 clamp，`packed_index_topk` L758）——
  **ds4 Metal 侧要先确认 `dsv41_indexer` 是否同样已 clamp，确认没有浪费就不做**（防抄错作业）。
- **P3-C · DeepSeek layer-major split 路径的 async flush**（OMLX §S5 窄缺口）：
  `end_commands`（commit+wait）换 `flush_commands` + 周期 drain，编码占比 >2% 才做。
  注意 readback shared-event 与 transient 生命周期绑在等待上（OMLX §S5 落点警告）。
- **P3-D · MoE packed-RHS 索引化**（#4029 模式，+1–2%）与 **#758 indexed prefill 注意力**
  （M5 Max 实测 −11.3% attn 段）——按 P0 timeline 里 MoE/attn 占比排序。
- **P3-E · Engram prefill 预取深化**：现在 `total_count ≥ 1024` 才 overlap，
  且每表一次全量预取（`carry_cap×24×256` float 的缓冲）；评估分段流水线（当前
  `DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE` 已有雏形）在 16K+ sweep 下与 expert sweep
  读写并发时的 NVMe 队列竞争——P0 timeline 里 Engram 预取占比 <5% 则不动。

### P4 · 重杠杆（立项独立评估，不与上面混跑）

- **三值/Bonsai 量化主权重**（上游 #1085 提案）：流式 decode 按字节数外推 ×2–2.5，
  是**唯一能把 Q2 流式 decode 推到 30+ t/s 的单机路径**；纯提案无实测，
  先做 gguf-tools 侧的 1-epoch PTQ 可行性 spike（质量 fixture 闸门现成）。
- **多盘副本读**（argonaut `--ssd-streaming-replica` 思路，双 TB5 NVMe +19%）：
  ds4 侧落点 = expert page-in 线程池按盘加权 + Engram `pread` 按 fd 副本分流；
  需要用户有第二块盘（Thunderbolt 5），属可选部署形态。
- **#1089 活会话 rewind**：agent 负载的有效吞吐（免 re-prefill），非峰值 t/s。

---

## 4. 验收与纪律（全阶段通用）

1. **bit-exact 优先**：任何"输出保持"候选用 `--dump-first-logits` cmp +
   `ds4_test --logprob-vectors` + frontier logits 对拍；改变文本的候选（MTP、任何数值改写）
   走 quality fixture（nll/first_match/lcp band）+ greedy 对齐说明。
2. **A/B 纪律**：同 build 同机、ABBA ≥2 轮、区间不重叠才宣称；每臂记录
   quant 类型 / effective cache / encoder_resident / 接受率 / cold-warm 标签
   （同模型不同配置差 5–12× 的老坑）。
3. **空机 + 每 kernel 线独立窗口**：128 GB 上加载 341 GiB 模型时空机跑；
   不同时开两条数值线（OMLX §7 教训）。
4. 每项一个 kill switch env，命名跟随现有 `DS4_METAL_DISABLE_V41_*` 传统；
   默认值变更（如 CED 阈值下移）单独 commit，可单独 revert。
5. 结果写回本文档"实测回填区"（§6），不扩散到 README。

## 5. 风险登记

| 风险 | 缓解 |
|---|---|
| MTP verify 多行放大 miss，负收益 | P2-0 离线重放先行 + engage 硬判据 + 独立 gate 止损 |
| verify 行 × streaming gather 无现成路径（现在 ≤8 行融合批只在驻留/TP 验证过） | P2-2 分四段正确性阶梯，每段位一致对拍 |
| 门放开后流式域与 M3 Ultra 调优核互相污染归因 | 每 kernel 独立白名单位 + 逐 kernel A/B |
| V4.1 DSpark support 权重缺发布 | P0-4 先查证；自制路径（metadata 齐、schema 有 V4 蓝本） |
| 宽步/verify/encoder 常驻三者的 transient 内存叠加撞 memory guard | P3-A 起每臂记录 guard 日志（`DS4_*` throttle 零事件为门槛）；#1122 wired 封顶已在 fork |
| 冷专家首触（6.5–12 t/s）被误读为优化无效 | 所有表 cold/warm 双列；hotlist（`ds4_streaming_hotlist.inc`）命中单独锚定 |

## 6. 实测回填区（执行时填写）

### 6.1 P0-2 · 量化类型（✅ 2026-09-30，`./ds4 --inspect`）

| 事实 | 值 |
|---|---|
| 路由专家配方 | **IQ2_XXS gate/up（80 张量 87.01 GiB）+ Q2_K down（40 张量 55.37 GiB）** → 落 `v41_decode_batch` 第一族（DSpark verify 融合路径可用） |
| 专家体量 | 142.38 GiB → **3.56 GiB/层，≈9.5 MiB/槽**（6 槽/token/层） |
| 常驻非路由权重 | q8_0 7.07 + f16 2.01 + f32 0.30 ≈ **9.4 GiB**（decode DRAM 地板 ≈9.4 GB/token → ~19 ms @~500 GB/s，驻留上限 ~50 t/s 级） |
| Engram | i8 2 张量 188.83 GiB（层 1/14） |
| 文件 | 340.60 GiB，748.49B logical params |

### 6.2 P0-4 · DSpark support 可得性（✅ 2026-09-30，经 hf-mirror，直连 HF 不可达）

- `antirez/deepseek-v4.1-flash-gguf` 仓库文件：**仅 Q2 / Q4(part1+2) / Vision**，
  **无 `DeepSeek-V4.1-Flash-DSpark-support*.gguf`**。
- `deepseek-ai` 组织有 `DeepSeek-V4-Flash-DSpark`、`DeepSeek-V4-Pro-DSpark` 独立权重仓库，
  **无 V4.1-Flash-DSpark 独立仓库**；但 `DeepSeek-V4.1-Flash/resolve/main/config.json`
  的 text_config 内嵌全部 `dspark_*` 超参（block 5 / markov 256 / target 37-39 / 128 专家 top-3），
  且远程 omlx Studio 的 `DeepSeek-V4.1-Flash-oQ4e-mtp` 检查点 `n_mtp_layers=3` 证实
  **draft 权重在官方 V4.1-Flash checkpoint 内**（V4.1 与 V4 不同：DSpark 不再单独发权重仓库）。
- **P2-1 结论**：GGUF 侧 support 文件需**自制**——从官方 checkpoint 抽 draft 层张量，
  按 `deepseek41-dspark` GGUF schema 打包（蓝本 `DeepSeek-V4-Flash-DSpark-support-0731.gguf`，
  schema 已被 `ds4.c` v41_support 校验路径接受）；或等上游发布（挂关注）。
  注意镜像可能滞后上游，动手前再查一次原站。

### 6.3 P0-1 基线矩阵（✅ 2026-09-30，本机 M5 Max 128GB，`--power 100`，promessi_sposi，greedy gen 128）

| 臂 | 配置 | prefill 续推 (2K append) | prefill 单发大 prompt | decode steady | 备注 |
|---|---|---|---|---|---|
| warm×2 | `--ssd-streaming`，auto preload，ctx→32K | **84–97 t/s**，全程平（两轮复现） | — | **23–24.5 t/s** 跨 2K→32K 无衰减 | 宽步 8192 生效（ctx≥16K）；cache 87.63 GiB |
| wide（大 prompt） | 同上，单次 16K / 49K 全量 | — | **567.8 / 572.7 t/s** | 21.5–23.6 t/s | wide sweep + `decoder_suffix`(CED) 生效；`carry_cap` 在场时 encoder 常驻**不启用**（设计如此）；decode 略低于续推臂 = sweep 后 cache 重建成本 |
| cold | `--ssd-streaming-cold`（跳过热门预载） | **60–72 t/s**（比 warm 低 ~25%） | — | 23–24.7 t/s | cold 惩罚集中在 prefill；decode 由 hotlist+hotness 快速追平 |
| 交互 gap | server + 48 tok/轮，gap 2s/6s 各 4 轮 | ttft 341–363 ms（预热后） | — | **25.2–26.2 t/s** | **M5 Max 无 power-wake 税复现**（与文档"空档后首 token 更慢"不符——那是 studio 低占空比场景） |

关键路径日志锚定（warm 臂）：`expert budget before prefill reserve: 9453 (87.63 GiB)`；
`effective 87.63 GiB = 7.12 GiB prefill headroom + 80.51 GiB dynamic cache (8685 experts, 9.49 MiB each)`；
`prefill_cap=8192`（ctx 32897）。`kv_heads=1 head_dim=512 swa=128`。

### 6.4 P0-3 命中率与 roofline（✅ 2026-09-30）

- warm 短程（2K→8K）整体 **hit_rate=0.834**（hits 133k / misses 26.5k，wraps 79k，evictions 17k）；
  `miss_willneed=0.00 GiB, evict_dontneed=0.00 GiB, miss_pread=13.25 GiB / pread_ms=818`
  → miss 大多由 OS page cache 兜住，**真正 NVMe 触碰量很小**；decode 的墙在 DRAM/调度侧（§2 裁决）。
- per-layer：l0 命中率最低（0.824，pread 1.80 GiB——首层热度最散），l3+ 0.83–0.85。
- 测量行补记：ds4.c v41 图释放路径原本无任何 memory-report 调用点，本次加了
  `DS4_METAL_MEMORY_REPORT` 门控的一行 dump（[ds4.c:40307-40312](ds4.c)），
  P1/P2 全部 A/B 用它锚定命中率——**这是唯一动过的引擎代码，env 不开零影响**。

### 6.5 P1/P2/P3 各项 A/B、核查与蓝图

#### P3-A ✅ 阈值矩阵（2026-10-01，M5 Max，零改动）

**P3-A 阈值矩阵** | 曲线 | **✅ 完成** |

流式 warm cache 下"单次进入 sweep 的 token 数 → prefill t/s"实测矩阵（`--gen-tokens 0` 纯预填，steady 行均值）：

| 单次 append 行数 | 路径 | t/s | CED 关闭后 t/s | CED 贡献 |
|---|---|---|---|---|
| 2048 | tail 2048 行 sweep | **92.0** | —（不进 wide） | n/a |
| 4096 | wide sweep | **116.9** | —（4096<8192 无 CED） | n/a |
| 8192 | wide + CED | **234.8** | 185.6 | **+26.5%** |
| 16384 | wide + CED | **426.4** | 281.6 | **+51.4%** |

三点结论（写入 §2  roofline 的 prefill 侧注脚）：
1. **一次大 prefill 一次性 sync ≈ 这个矩阵的右端**（P0 的 570–573 t/s 即 16K/49K append 档）；
   **每轮 2K 的 agent 续推 ≈ 左端 92–117**。同一模型同一机器 prefill 差 5–6×，全部由
   "每 sweep 行数 × CED 是否生效"解释，不是 bug。
2. **阈值不动**（防抄错作业验证）：wide 的 `remaining>=4096` 与 sweep 几何耦合
   （count=carry_cap 对齐到 2048 的倍数，encoder_chunk=4096 时 4096 行不可能 wide）；
   CED 的 `total>=8192` 受 `wide &&` 前置约束，4096 档下 CED 在代码上不可达。
   omlx CED@4096 的 +7.6% 是 resident 数字，不能外推到这里的分支几何。
   **【P3-A2 实测更正】**"4096 档代码上不可达"半条**错了**：`ds41_encoder_chunk_cap(4096)=2048`，
   4096 行 sweep 本来 wide=true（本矩阵 4096 行"路径"列自己写了 wide sweep）；挡 CED 的只有
   `total>=8192` 字面量，与几何无关，且该字面量从未在流式域实测——正是 §2 第 3 条要求重扫的对象。
   下移实测见 P3-A2。"omlx 数字不能外推"半条仍成立：P3-A2 的 +9.3% 是本机流式域实测。
3. **优化窗口在左端**：agent 每轮 <4096 增量吃不到 CED。候选（未立项）：多轮增量在
   server 端攒批到 ≥8192 再 sync（延迟换吞吐）；或把 decode_suffix 的"尾部窗口省算"
   思想移植到 2048 行 tail sweep。**均需先有真实 agent 负载占比数据再动**。

#### P3-A2 ✅ CED 阈值重扫：4096 档实测成立，默认下移（2026-10-01，M5 Max，流式域）

**改动**：`ds41_decoder_suffix_min_rows()`（[ds4.c:42381](ds4.c#L42381)）——门限 env 化
`DS4_V41_DECODER_SUFFIX_MIN_ROWS`（下限 2542 保 layer-20 warm 窗 `first-127` 不下溢，非法值回退
旧门 8192），**默认 8192→4096**；回滚 `DS4_V41_DECODER_SUFFIX_MIN_ROWS=8192` 逐字节还原旧门，
`DS4_METAL_DISABLE_V41_DECODER_SUFFIX=1` 全禁 CED 仍有效。影响：agent 续推 remaining∈[4096,8191)
（sweep 4096/6144 行）从"无 CED"变"有 CED"；**<4096 仍不进 wide，左端窗口不动**（结论 3 维持）。

**速度**（ABBA 序 A1→B1→B2→A2，`--ssd-streaming --power 100 --ctx-start 4096 --step-incr 4096
--ctx-max 32768 --gen-tokens 0`，promessi warm；steady=12K–32K 窗行均值；A=旧门8192，B=新门4096）：

| 臂 | Run1 | Run2 | steady 均值 |
|---|---|---|---|
| A（CED@4096 关） | 77.46 | 77.98 | **77.7** |
| B（CED@4096 开） | 82.58 | 87.37 | **85.0** |

**+9.3%，簇间不重叠**（A_max 77.98 < B_min 82.58），逐档 append 增益 +5.0%~+21.9% 全正（最大在
8K frontier：A=[88.9, 87.0] → B=[106.2, 108.2]）。口径诚实：本日 A 臂绝对值（~78）低于同日凌晨
P3-A 矩阵 4096 档（116.9）——机器页缓存/专家热态漂移未归因，**跨会话绝对值不可比，只认会话内 ABBA**。

**验收闸门**（教训：闸门顺序改成"先证明旋钮拨动了，再做昂贵验证"）：
1. **Engagement（分钟级）**：B 臂速度阶梯差值本身即证明 CED@4096 生效，然后才投入正确性/质量验证。
2. **隔离位一致（本特性类的正确性闸门，与 8192 出厂同一把尺）**：`test_deepseek41_graph
   --decoder-suffix` 经 `DS4_TEST_SUFFIX_COUNT` 扩到 4096 行档；batch-off 固定算术下 shrink-tail
   裁剪**位精确**——初始 + 续推 prefill 的 state span/logits + decode step 全等（2/2 PASS），
   且隔离下 suffix 本身 29.4→43.7 / 29.1→42.7 t/s（+48%）。
3. **批开数值类（对照已发布行为）**：批开位一致从来不是 suffix 的契约——**已发布的默认 CED@8192**
   对显式关闭本来就有 step0 max|Δlogit|≈0.14（greedy 48/48 不翻，p9k 探针，双臂实测）；4096 新档
   4640-token 合成探针 max|Δlogit|≈0.37、greedy 第 5 步起换字。同一机制（shrink tail 改变 batch
   GEMM 行分区的舍入路径），但 4096–8191 档 prompt 的文本级漂移是本次新暴露的，按纪律走质量档：

**敏感档质量闸门**（router fixture 抽 band 内 3 例 + band 外控制 2 例，重建后的 score_official 双臂
一次加载对拍；X=旧门8192，Y=新默认）：

| case | prompt tok | avg_nll X | avg_nll Y | Δ | first_match/lcp |
|---|---|---|---|---|---|
| case_102 控制（双臂无 CED） | 2550 | 0.020007 | 0.020007 | 0（位同） | 同 |
| case_103 band 内 | 4137 | 0.261979 | 0.268462 | **+2.5%** | 0/0 → 0/0 |
| case_104 band 内 | 5187 | 0.160883 | 0.160188 | −0.4% | 1/1 → 1/1 |
| case_105 band 内 | 6710 | 0.132571 | 0.132070 | −0.4% | 1/14 → 1/14 |
| case_106 控制（双臂有 CED） | 8233 | 0.264598 | 0.264598 | 0（位同） | 同 |

带内漂移双向、幅度 ≤2.5%，**first_match/greedy_lcp 零翻转**；带外两控制例逐字节同——与已发布
8192 行为同类，通过。

**结论**：默认落 4096。P3-A 结论 2 相应更正（见其上勘误块）。<4096 左端（2048 行 tail sweep 的
carry 移植 / server 端攒批）维持原状，等真实 agent 负载占比数据。

**假闸门教训 ×3（方法论，防复用）**：
1. `env VAR=$x` 传空字符串仍算"已设"（C 侧 `!getenv()` 判 false）→ 早期 CLI 五臂 dump 对拍实际
   五臂全 CED-off，"5/5 位一致"无意义。**对拍脚本对不拨的旋钮必须 unset 而不是传空；昂贵验证前
   先跑 10 秒预检（二进制 mtime 差、env 列表无空串变量、目标 case 尺寸直方图）**。
2. `score_official` 静态链接 ds4.o——**改引擎后必须重建 harness**，否则跑旧代码：112×2 臂全量
   （~100 分钟）跑在门改动前的旧二进制上作废，改为 5 例敏感档重跑（~20 分钟）才有效。
3. fixture 先算**敏感区间**再选 case：112 例里 109 例 prompt<4K，对 8192→4096 阈值改动零敏感。

#### P3-B ✅ indexer clamp 核查（"确认没有浪费就不做"分支命中）
`kernel_dsv41_indexer_scores_packed`（[metal/dsv41.metal:288](metal/dsv41.metal#L288)）
对 `row0 >= visible` 提前 return 并写 −INF（visible=(pos+token+1)/ratio），与 omlx 已 clamp
的 `packed_index_topk` 同构——**无浪费可省，零改动**。

#### P2-1 蓝图扩充（为自建 V4.1 DSpark support GGUF 探路）

本地 `gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf`（5.6 GiB）解析成功，
schema 全集 = **81 张量**：`mtp.{0,1,2}.*` 每 stage 一套 draft 层
（attn_kv/q_a/q_b/output_a/output_b/norms/hc_attn_*/hc_ffn_* + ffn gate/up/down_exps
**256 draft 专家** + shexp + router）＋ 单件 `markov_head.markov_w1/w2`（256×129280）
＋ `confidence_head.proj` ＋ `hc_head_*` ＋ `main_proj/main_norm`；
metadata 仅 12 个 KV：`general.architecture=deepseek4-dspark`、
`dspark.block_size=5`、`markov_rank=256`、`noise_token_id`、`target_layer_ids=[40,41,42]`、
`stage_count=3`、`n_layers=3`。对照 V4.1 主 GGUF 已含
`dspark_block_size/markov_rank/target_layer_ids/n_routed_experts/num_experts_per_tok/noise_token_id`
元数据键（即 P0 结论"draft 参数全在 config"）。
**V4.1 版自建 = 同 schema，参数替换**（hidden 4096→5120、draft 专家 256→128、
targets [40,41,42]→[37,38,39]、词表按 V4.1），权重张量需从官方 checkpoint 的 draft
safetensors 转换（hf-mirror 可达）。此项独立于 P2-0，工作量为一次转换器脚本。

#### P1-A ✅ 调度门放开（2026-09-30，实施 + 验收完成）

**实施中发现的真相（比原设想更收）**：`measured_m3_ultra` 是三重门——
`g->streaming`、**设备名精确匹配 "Apple M3 Ultra"**、**全层同质 Q4_K/MXFP4 专家**
（[ds4.c:41786-41798](ds4.c)）。本机 Q2（IQ2XXS+Q2K 混配）连 resident 也从未吃到
pipeline/engram-overlap；而 decode flush（"commit 不等待"，#1041 家族的
`DS4_METAL_DISABLE_V41_DECODE_FLUSH` 门）**本来就不在 measured 域内**，本机一直生效。

**改动**（全部在 [ds4.c](ds4.c)，共 +28 行）：
- [ds4.c:41820-41829](ds4.c#L41820) 新增 `ds41_decode_pipeline_admitted`：
  单设备 + 非 imatrix/image/quality 时，M3-Ultra-measured **或** M5 系 Apple Silicon
  均放行调度类路径；回滚 env `DS4_METAL_DISABLE_V41_DECODE_PIPELINE_PORTS=1`。
- [ds4.c:41869-41874](ds4.c#L41869) `pipeline_layers` 与 [ds4.c:41882](ds4.c#L41882)
  engram step readers 改用新判据。位一致依据：dispatch 次序与数值不变，
  仅"layer-13 不再 drain / 尾表独立输入 / logits 头提前编码 / 表读与 layer 0 重叠"。

**验收**（同构建 AB 双臂 + 跨二进制位一致）：
- **位一致 3/3**：短 prompt 全量 logits、greedy 48 步 top-20、8K prompt greedy 32 步，
  新旧二进制逐字节一致；`DS4_METAL_DISABLE_V41_DECODE_PIPELINE_PORTS=1` 亦一致（回滚门真实）。
- **速度（server SSE steady，clean 轮次）**：新 **27.3/27.5/27.6/27.6 t/s** vs
  旧 **26.1–26.8 t/s** ⇒ **+2.5~4%**（簇间不重叠）。
- **host profile 归因**（64 步均值，ms/token）：hash+engram **0.68→0.003**（表读异步化）、
  logits 读回 **1.45→0.01**（头提前编码）、layers **~40→36.2**（layer-13 drain 消失 + GPU 提前起跑），
  合计 **−4~6 ms/token**；encodes/copies 每步计数两臂相同（1527.9/89.0）——调度改动不是伪装的数值改动。
- 标准测试套说明：本机无 V4 Flash 主权重（logprob vectors 属 v4 架构），v41 调度改动以
  上述 4 组位一致工件作为等价闸门。
- 副作用声明：resident 混配 Q2 文件在 M5 机上也一并进入 pipeline 域（同一 host 路径，
  位一致逻辑同此）；M3 之外的 resident 同质文件不受影响（measured 分支原样）。

**裁决**：默认 ON（M5 域）。原 P1-A 条目里的"Metal 融合族逐 kernel 放开"**不包在本条**，
那需要逐 kernel 审流式权重绑定（gather_reuse 等依赖 resident 映射），另立 P1-A2。

```text
[ ] P1-A2 融合族（ATTN_EPILOGUE/QB_BF16/GATHER_REUSE）流式域逐 kernel 审计+A/B
[ ] P1-B #1035 / #1034 试合
[ ] P1-C mailbox
[x] P2-0 并集/miss 离线曲线（DRAM+SSD 双账）→ §6.5 P2-0（判据跑穿：资产缺口，非账本）
[x] P3-A 阈值矩阵曲线 → §6.5 P3-A（阈值不动）
...
```

#### P1-A2 ✅ 激活类融合核流式域放开（2026-10-01，实施 + 验收完成）

**审计结论（三族三种命运）**：
- `ATTN_EPILOGUE`（把 heads 的 BF16 界+RoPE 融进注意力归约尾）与 `QB_BF16`
  （q_b 投影走融合 `ds41_matmul`）是**逐层 activation 纯函数**，读的是
  **static-locked 稠密/注意力权重**（流式下 9.37 GiB 全程常驻，与 resident 域同值）
  ——流式域放开数值安全，走新门 `ds4_gpu_dsv41_activation_exact_admitted`
  （[ds4_metal.m:10556-10577](ds4_metal.m#L10556)，M5 域默认 ON；master 回滚
  `DS4_METAL_DISABLE_V41_STREAMING_ACTIVATION_FUSIONS=1` + 各自单核 env）。
- `GATHER_REUSE` **审计后保持关闭**：它读的是 `selected_kv` 跨层 scratch，
  生命周期必须跨越 pipeline flush——在流式 decode 管线（il==0/4 flush）下属
  **位一致采样无法证明无竞争**的一类耦合；省 ~100 dispatch/token 不值这个险，
  想要时先做跨层 scratch 生命周期专项分析。
- `VERIFY_*_ROWS` 属 spec（drafting）路径门（`spec &&`），单 token decode 不涉及，不动。

**验收（四臂 A/B，同 build 四独立 server 会话，n=160 steady）**：
OFF 27.36/26.76/27.12；ON#1 26.87/26.89/24.58(离群)；**ON#2 27.46/27.25/27.46**；
仅 qb 27.17/27.32/27.23；仅 epilogue 27.57/27.06/27.02。encodes/step 1527→**1472**
（epilogue 生效），ON#2 的 layers 窗口 36.3–36.4ms ≤ OFF 36.6–37.1ms。
位一致 2/2（短 greedy-48 top-20 + 8K-prompt greedy-32，与 P1-A 前 golden 逐字节同）。

**裁决（诚实口径）**：本机流式 decode **wall-clock 中性**（decode 是带宽/流水
深度墙，不是 5% 派发墙），**非实质性提速**；保留默认 ON 的理由是严格更少的 GPU
命令编码（−55 dispatch/step，功耗/宿主争用友好、给未来批量路径留量）且四臂无一
回归。首次 ON 臂的 24.58 离群轮经复跑+layers 窗口归因判为热漂（非内核）。

#### P2-0 ✅ 双账曲线（2026-10-01，判据跑穿——MTP 立项的"生死判"有了诚实答案）

方法：`DS4_V41_TRACE_EXPERTS`（新增，env 门控探针，[ds4.c:41954-41975](ds4.c#L41954)，
须在 `DS4_METAL_DISABLE_V41_DECODE_QUEUE=1` 逐层 drain 下采集）抓 8K-prompt +
3824 token 解码的每 token×40 层 top-6 专家序列（152,960 条）；
`speed-bench/mtp_ledger_replay.py` 离线重放 GPU-cache LRU + page-cache 双账。

- **校准**：纯 LRU 在 217 slot/层给出 0.985 命中，实测 0.834 对应 ~80 slot/层
  等效压力（seed/hotness/page 粒度记账差异）——在 80 与 217 两档容量下各跑一遍。
- **并集熵实测**（决定一切的数字）：每层连续 token 专家集重叠度极高——
  window 1→6 的 distinct 专家数 **6.00→22.24**（无重叠上界 36），
  即 **6 行 verify 的专家字节只有单 token 的 3.71×**；摊到每接受 token：
  accept=a 时 `22.24/(a+1)/6` → a=1 **x1.85**、a=2 **x1.24**、a=3 **x0.93**、
  a=5 **x0.62**。**判据 1.3× 在 accept≥2 时过**（markov 型 draft 典型接受 2-3）。
- **双账 LRU 重放**：所有 accept 档 DRAM/SSD 比值 **x1.00**——顺序流里"6-token 并集"
  的新增 miss ≈ 新进窗口 token 自身的 miss（窗内旧 token 全在 LRU 白 hit），
  **verify 并集结构上不增加稳态 miss 字节**。

**边界（不可自欺的三件事，accepted-only trace 原理上测不到）**：
① 被拒行的专家在 off-trace 分布（悲观上界 6/(a+1)×，a≥4 才稳过）；
② draft 层 37-39 自身的前向与 128 个 draft 专家的驻留（trace 未覆盖）；
③ 引擎 block-spec 图的宿主侧开销。**结论：expert 账本不否决 MTP**（推翻 §2 里
"x1.3 且 pread x4.11 会击穿"的悲观推式——并集重叠实测极高所致）；
**阻塞点从"分析未跑"改判为"资产 + 真实接受率"**：需要先有 V4.1 support GGUF
（P2-1，转换器脚手架已备）再实测 accept 分布，才谈得上施工。P2 整体维持"暂缓"，
但这次的"暂缓"是跑穿判据后的决定，不是回避。


---

## 7. P2 解除暂缓：流式 DSpark 管线施工记录（2026-10-02，用户指令开工）

用户指令："开始流式 draft 管线施工，如果别的模型有类似实现，参考着做。"
本节记录从"无资产"到"管线首跑"的全部落地与**唯一剩余卡点**的定位。

### 7.1 资产（P2-1 完成 ✅）
- 第三方 draft：`JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF`（llama.cpp `dflash` 命名空间），
  7.43 GiB，SHA256 对上官方 SHA256SUMS；`dsv41_semantics=1`、目标层 [37,38,39]、
  3 stage、128×3 draft 专家（FFN 2304，真小于主干 18432）——资产即 V4.1 合同定制。
- 转换器 `speed-bench/jigsaw_to_ds4_dspark.py`：改名（`blk.N.*`→`mtp.N.*` 等 6 个重命名）+
  bf16→f16 数值转换（DENSE 拒 bf16）**+ confidence head bf16→f32**（`ds41_draft_block`
  CPU 点积按 float 读，`type==F32` 否则 NULL→draft 恒 0 提议，2026-10-02 实测定罪）+
  重算 tensor offset（f32 变宽，32B 对齐保持）+ 元数据按 blueprint 重建（arch
  `deepseek41-dspark`，family gate ds4.c:3292）。产物 `gguf/DeepSeek-V4.1-Flash-DSpark-ds4.gguf`
  引擎绑定 `tensors=78 missing=0 invalid=0 metadata_errors=0`。
- 副产品教训：**主模型 embed 是 f16、draft 表读核是 `get_rows_f16`，markov 转 f16 正确**；
  V4.1 激活是 "bf16 值装 f32 槽"（bf16-linear），CPU 端按 float 读是 by design。

### 7.2 引擎接线（三道闸，全部 env opt-in `DS4_EXPERIMENTAL_DSPARK_STREAMING`，默认关）
1. `ds4.c:72567` "DSpark needs resident single-device Metal" → 流式放行（附注释）。
2. `ds4.c:73068` "--ssd-streaming is not compatible with --mtp-model **yet**"（作者留的占位）
   → 仅 dspark+env 放行，legacy MTP 在流式域继续拦。
3. `ds4.c:86171` `drafting = !g->streaming && ...` → env 放行。
**拓扑澄清（推翻 10-01 晚的初判）**：V4.1 session 走 `ds41_graph_alloc`（第四套图），
12-KV 时代赖以推理的 75147 `metal_graph_configure_dspark_capture` 属 V4 `metal_graph` 系，
V4.1 原生实现在 ds41 域内：`ds41_draft`/`ds41_graph_verify`/`ds41_spec_commit`/
`ds41_session_ds41_spec_cycle`（86157）全链已写好，只是被闸 3 拦。参考实现即自家 ds41。

### 7.3 流式首跑实测（M5 Max，128G）
- 装配成功：draft 权重经 support model 全区 map view（73627 注册对 mtp_model 生效），
  与专家缓存零干涉；增量内存仅 ~+3 GiB mmap（footprint 45.2 vs 基线 41.2，128G 安全域）。
- **cycles 真转起来了**（DS4_DSPARK_STATS：cycles=48，propose 12ms/轮，drafted=5/轮稳定产出），
  target-hidden capture 在流式单步路径（ds41_graph_step 41937）被喂通。
- **无损验证**：4 种 dspark 流式配置 × 48 token greedy 输出与基线**逐字节一致**
  （count=0 全部回退普通步；输出只经 verify argmax 等价门 → 施工失败不污染正确性）。
- 修复保留：`ds41_draft_block` host 读 argmax 结果前补 `ds4_gpu_synchronize()`
  （end_commands 只 commit 不等待；流式队列比驻留域深，作者域测不到这个 race）。
- **调试方法论事故（记录防再犯）**：Metal shared buffer 在 begin/end_commands 窗口内
  CPU `contents()` 读的是旧镜像——本轮全部 "NaN" debug 打印均为**假信号**（打印点在命令
  队列提交前）。-ffast-math 还会把 `x==x` 优化成恒真，NaN 计数要用 `vs*x!=x` 逃逸。

### 7.4 唯一剩余卡点（定位完成，实锤一行错误）
强制过 conf 门让 verify 真跑后：
`ds4: Metal model range 2.04..3.13/3.14..4.23/4.24..5.62 GiB is not covered by mapped
model views → V4.1 session batch failed at layer 1 (6 rows) → DSpark verification failed`
即 `ds41_graph_step_batch_spec`（verify 行批）对主干 MoE 专家走 `routed_moe_batch_tensor`
**直读 mmap view** 路径——流式域 map spans 按设计只登记常驻静态权重，专家在流式缓存里。
**剩余工程 = 把 verify 批的 MoE/路由段接到流式专家缓存 gather（对齐
`ds41_graph_encode_layers` 的 gather 路径，即 prefill 行在流式域已走通的那条）**；
draft 侧（stage 前向/seed/标量 matmul）不碰主干权重，无需改动。
量级：`ds41_graph_step_batch_spec` 内 MoE 编码路径重构 + bit-exact/accept 验收，内核级一天。
可参考对象：同文件 prefill 行批（流式已通）；qwen4 `ds4_session_qwen4_spec_cycle`（同构
spec 循环，其流式域在 V-Flash/非流式场景）。

### 7.5 状态
- 代码改动：ds4.c 3 处 env 闸 + 1 处 sync 修复（全部默认关/无损）；转换器 f32 conf 修复。
- 增益状态：**尚未产生 decode 加速**（卡点在 verify 批）；已清除全部"未知"，
  从"能不能做"变为"剩一段已知代码"。基线复跑确认无回归（12.85 t/s 同 warm 域）。
- **后续（2026-10-01）**：verify 批 gather 已接线、draft 恒零真凶结案（支持 GGUF 12B
  对齐错位），accept 61.7% 功能闭环——但 verify 行成本使吞吐暂低于基线，见 §8。

## 8. 流式 DSpark 全链路打通（2026-10-01，draft 恒零根因结案 + 验收）

### 8.1 verify 批 gather 接线（§7.4 遗留工程，已完成）
`ds41_streaming_verify_moe_gather_admitted`（ds4.c:42745 区）+ `ds41_moe_batch(...,
gather_streaming_experts)` 形参：verify 行的主干 MoE/路由段改走流式专家缓存 gather
（与单步 decode 同一绑定），受自身前置条件闸（V41 Q2 路由配方 / 单层缓存容量 /
行窗口），不满足则 drafting 关闭退回普通步。draft 侧（seed/stage/标量 matmul）不碰
主干权重，未改动。

### 8.2 draft 恒零真凶 = 支持模型 GGUF 的 12 字节对齐错位（结案）
症状：seed/proposal 全链恒零（main_x/window/logits 全 0），accept 恒 0。
排查最终锁定 **`speed-bench/jigsaw_to_ds4_dspark.py` 的 `data_start_in = fin.tell()`
未对齐**：源文件头尾 5,254,324 非 32 对齐，GGUF 数据区应始于 align_up(...,32)=5,254,336
——**每个张量 payload 整体前移 12B**，q8_0 的 fp16 scale 字段读到前一张量 bf16 尾巴
（`0000 613f…` 形态），大量 d=0 → matmul 精确输出 0。源文件本身 q8_0 完全合法
（block 对齐采样 neg_d=0 / zero_d=0 / max|w|=0.28）。
- 修复：converter 读源 `general.alignment` 并 `align_up`（2 处编辑），重新生成
  `gguf/DeepSeek-V4.1-Flash-DSpark-ds4.gguf`（旧坏件挪至 /tmp/dspark_old_broken.gguf）。
- **被推翻的本轮旧结论（防再犯）**："ssd-streaming 下 aux NoCopy map 读零、必须
  materialize 成 private buffer" 是假说被坏文件喂出来的——对齐修复后 A/B：删掉
  `ds4_gpu_materialize_aux_map` 调用，accept 率 61.70% 分毫不差。materialize 全套
  （ds4.c 调用 / ds4_metal.m 实现 / ds4_gpu.h 声明）已删，ds4_gpu.h 回到 HEAD。
- 同理被怀疑过的：kernel_mul_mv_q8_0_f32、NSG、NoCopy 页驻留、MANAGED——全部无罪。
- 教训：**上游资产先做 block 对齐级字节验证（q8 scale 符号/零率/magnitude），再谈
  kernel/运行时**；mmap 侧一切"诡异漂移"优先怀疑喂进去的数据。

### 8.3 验收（`--dspark-confidence` 扫阈，M5 Max 128G，`-c 8192`，temp 0，同 prompt）
| 配置 | accept_rate | avg_accept/cycle | generation t/s |
|---|---|---|---|
| dspark conf=0 | **61.70%**（proposed 47/accepted 29，errors=0，replay_fallbacks=0） | 2.636 | 9.97 |
| dspark conf=0.5 | 73.58% | 1.857 | 10.41 |
| dspark conf=0.8（n=150） | 91.53% | 0.562 | 14.18 |
| 基线（无 dspark，同 warm 域） | — | — | **17.5–17.7** |

draft 数值健康：conf 0.85–1.0、提案为真实 token id、draft_len_hist 5:8/上限恒触。
**功能闭环达成**（恒零 → 61.7% accept、errors=0、greedy 输出与基线一致性由 verify
argmax 等价门保底）。

### 8.4 诚实边界与新卡点：streaming batch verify 行成本
verify=4.2s/11 cycles ≈ 382ms/轮 ≈ **106ms/行，是单步 decode（57ms）的 ~1.9×**：
batch verify 无 batching 红利，反付每层「行×top-k 专家并集」分页 + 每轮 host 同步，
avg_accept 2.6 的收益被吃光——**当前口径任何阈值都低于基线**。
下一步（新迭代项）：verify gather 的行间专家共享 / 一次分页多行复用 / 分块 verify
限流；在行成本降到 ≤1×单步前，dspark 流式只建议功能验证使用，生产仍走普通 decode。
（此前观察到的 "dspark 更慢" 假象其实是恒零 draft 下 verify 只跑 1 行；本轮才第一次
测到真实 verify 成本。）

### 8.5 代码清点（工作树，提交前）
- `ds4.c`：+gather admitted/moe_batch 形参、libm sigmoid 修复（-ffast-math exp 域界）、
  `ds41_draft_block` host 读前 `ds4_gpu_synchronize()`、streaming drafting env 闸
  （`DS4_EXPERIMENTAL_DSPARK_STREAMING`，默认关）、M5 decode pipeline/trace 探针等
  ——全部调试脚手架（SEED_DEBUG 206 行 / STATS 107 行 / MATMUL_DEBUG×4 / FORCE_VERIFY
  / dspark meta·pread·fill probes / materialize）已清除，净 diff 189 insertions。
- `ds4_metal.m`：激活类融合核流式域放开（`ds4_gpu_dsv41_activation_exact_admitted`，
  P1-A2 延伸）。`ds4_gpu.h`：回到 HEAD。
- `tests/test_deepseek41_attention.c:67`：旧契约「streaming 恒禁用 epilogue」与激活
  移植冲突，改为 `used==ds4_gpu_dsv41_qb_bf16_admitted()`（同一 bit-exact admission）。
- 测试闸门：make test 编译+eval 92+50/agent/web/metal-kernels/fusions/topk/attention
  全绿；--all 其余失败 = 本机缺 ds4flash.gguf 测试模型，非回归。

### 8.6 既有 20+ t/s 加速优化存活核查（用户问询）
在案（全部已提交 HEAD，未提交 diff 无删改）：a7ee2f4 队列化 commit 不等、
e2e3563 HC glue 融合（19→6 dispatch/层）、ef45eec decode MoE glue 融合、
0319311 M5 单发 router select、8c5f178 decode 形 indexer、4ec5fc4 Q4_K group-6
专家表、030b342 queue/fuse 与 fork decode graph 调和。
本机复跑（`--power 100`，promessi_sposi 6KB clip，`-c 16384` warm×2）：
prefill **110 t/s**（§6.3 锚 84–97，不降反升）、decode **19.3–19.7 t/s**
（空 prompt 同口径仅 15.2，证实 prompt 侧专家命中差；§6.3 的 23–24.5 是 2K 续推
32K-ctx 稳态口径，与本轮短轮口径不可直比）。结论：**优化全部在码在跑，无丢失、
无回归**。

---

## 9. 流式 DSpark 复盘：不是"没生效"，是**生效了但必然更慢**（2026-10-02 复测）

§7–§8 留下的问题是"管线通了、accept 61.7%、但比基线慢"。本节把它**测透**，并推翻 §8.4
对成本结构的判断。结论先行：

> **dspark 在流式域确实接线成功、输出逐字节正确；它慢的原因是所有流式专家快路径都按
> `n_tokens==1` 或 `n_tokens>=32` 准入，而 verify 只有 2–6 行，于是整条 MoE 掉回慢路径。**

### 9.1 复现（同 prompt / temp 0 / `-c 8192` / M5 Max 128G / `DeepSeek-V4.1-Flash-Q2.gguf`）

CLI（`--ssd-streaming`，`DS4_EXPERIMENTAL_DSPARK_STREAMING=1`）：

| 配置 | generation | cycles | proposed | avg_accept | accept | verify ms |
|---|---|---|---|---|---|---|
| 基线（无 dspark） | **15.70 t/s** | — | — | — | — | — |
| dspark conf=0.0 | 10.02 t/s | 44 | 211 | 1.909 | 39.8% | 12393 |
| dspark conf=0.2 | 10.21 t/s | 44 | 206 | 1.909 | 40.8% | 12146 |
| dspark conf=0.4 | 12.02 t/s | 49 | 132 | 1.612 | 59.9% | 10202 |
| dspark conf=0.6 | 12.30 t/s | 55 | 87 | 1.327 | 83.9% | 9844 |

`ds4-server`（同模型/参数，`:8199`，128 completion tokens）：

| 配置 | 端到端 | completion tokens |
|---|---|---|
| 无 dspark | **13.70 s** | 128（输出与 dspark **逐字节相同**） |
| `--dspark --dspark-confidence 0.4` | 17.22 s（+26%） | 128 |

**推翻了 §8.4 的"没生效"读法**：server 侧 `cycles=46 proposed=142 accepted=82
avg_accept=1.783 errors=0`，dspark 确实在跑；输出逐字节一致说明 verify argmax 等价门成立。
真正的现象是**它在流式域必然亏**，而不是没接线。

顺带一条硬约束：V4.1 **驻留模式在 128G 上装不下**（`needs 157.71 GiB before the expert
cache`）。所以 ssd-streaming 对本机不是可选项而是前提，dspark 只能活在流式域里。

### 9.2 成本模型（对 4 个 conf 点做最小二乘，这是本节的核心）

以 `rows = proposed/cycles + 1` 对 `verify_ms/cycles` 拟合：

```
verify_ms/cycle ≈ 32.3 ms × rows + 93.1 ms(固定)
基线单步 decode = 63.7 ms/token
```

- **行间摊销是真的**：32.3 ms/行 只有单步的 **0.51×**——批处理该拿的红利拿到了。
- **每轮固定开销 ≈ 93 ms**（≈1.5 个单步 decode）。即便只验 1 行也要付。
- **盈亏平衡需要 avg_accept ≈ 2.96**（即 ≥3）；实测 avg_accept 只有 1.33–1.91，
  而 dspark block=5 把上限钉在 5 → **任何阈值都在盈亏线以下**。
- propose 侧还要 **8–9 ms/cycle**（conf0.4：433.7 ms / 49）。即使 verify 免费，
  净收益也要先付这笔。

所以 §8.4 记的"行×top-k 专家并集分页"是**次要项**（对应 32.3 ms 那个斜率），
主项是那个 93 ms 固定项——两者差 3 倍，修错方向会白干。

### 9.3 根因：流式专家快路径对行数的准入，把 2–6 行 verify 全部挡在门外

`ds4_gpu_routed_moe_batch_tensor`（`ds4_metal.m:44509`，verify 行批与 prefill 行批共用）
里所有流式地址表/槽位分支都带行数门：

| 分支 | 位置 | 行数门 | verify(2–6 行) |
|---|---|---|---|
| `use_q4_selected_slots` | `ds4_metal.m:42080/42092` | **`n_tokens == 1`** | 关门 |
| `use_iq2_selected_slots` | `ds4_metal.m:42538` | **`n_tokens == 1`** | 关门 |
| `use_mxfp4_selected_slots` | `ds4_metal.m:42554` | **`n_tokens == 1`** | 关门 |
| `use_iq2_stream_addr_table` | `ds4_metal.m:42566` | **`n_tokens == 1`** | 关门 |
| `use_v41_stream_gather_addr` | `ds4_metal.m:44772` | **`n_tokens >= 32`** | 关门 |
| `use_iq2_batch_selected_addr` | `ds4_metal.m:44803` | 继承 `enabled()`（min=2） | 仅 conf 高时窄开 |

于是 2–6 行 verify 落到 `use_tiny_pair_mv`（`ds4_metal.m:44939`，`n_tokens<=8` 命中）
的**逐 token mul_mv 家族**——它按设计就要保 V4.1 的标量归约序，
**慢是标量顺序换来的，不是 bug**。prefill 行批（1024–2048 行）走的是
`use_v41_stream_gather_addr` 那条 mm-id gather，已经在流式域跑通并 bit-exact
（HANDOFF §12.7）——**同一条已验证的 gather，verify 够不到**。

这解释了固定项为什么这么大：每层每行都要付"没批红利"的价，40 层累积成 ~93 ms。

### 9.4 候选方案与诚实裁决

| # | 方案 | 收益上限 | 代价 / 风险 | 裁决 |
|---|---|---|---|---|
| A | 把 selected-slots 地址表的 `n_tokens==1` 放宽到 `<=8`（复用 `v41_decode_batch` 已有的 ≤8 标量契约） | 消除大部分 93 ms 固定项 | 需逐位对拍；`n_tokens==1` 是刻意的省门，放宽可能改变 dispatch 计数 | **首选**，改动最小、目标明确 |
| B | 把 `use_v41_stream_gather_addr` 的 `>=32` 下探到 2–8 行 | 同 A，且复用已 bit-exact 的核 | gather 是 mm-id 打包核，2–6 行 tile 填充率极低，可能**更慢**；§4.4 已有"小 prefill 只值 1.4×"的前科 | 次选，需实测 |
| C | verify 分块限流 / 行间专家共享 | 压 32.3 ms 斜率 | §8.4 已列为下一步；但斜率不是主项 | 降级 |
| D | 提高 avg_accept（换更长 draft） | 唯一能把 2.96 拉到 1 以下的杠杆 | block=5 是模型几何，改不了 | **不可行** |
| E | V4.1 DSpark 只在有驻留容量的机器开 | 恢复 §8.3 的 14.18 t/s 口径 | 本机 128G 装不下驻留 V4.1 | 仅适用于 ≥256G 机器 |

### 9.5 结论（给用户的一句话）

流式 dspark **接线是成功的**（server 与 CLI 双侧实测 cycles>0、errors=0、输出逐字节一致）；
慢的原因是**它每轮要付 ~93 ms 的固定 verify 成本**，而流式专家快路径只对
"1 行"或"≥32 行"开放，2–6 行的 verify 整条 MoE 掉回逐 token 慢核。
在把行数门放宽（A）之前，**V4.1 + ssd-streaming 生产应继续走普通 decode**。
