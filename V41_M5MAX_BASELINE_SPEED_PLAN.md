# V4.1 Flash × M5 Max 128 GB：流式 decode 基线提速方案（对应"建议 2"）

> 目标机器：MacBook M5 Max 128 GB；模型 `gguf/DeepSeek-V4.1-Flash-Q2.gguf`（151.77 GiB 主权重 + 188.83 GiB FP8 Engram，必须 `--ssd-streaming`）。
> 本方案全部为 **bit-exact 基线提速**，与 DSpark/投机解码无关，且刻意在 DSpark 关闭状态下执行。
> 前置结论（上一轮分析）：verify 每行边际成本 ≈ 完整单步成本，流式下投机上限 ≤1×；所以钱要花在"每一行都要付的那个 63 ms"上。
>
> **冷接读者请先读 §13（移交快照），再读 §12.6（阶段 3 归因裁决）——它们推翻/改写了 §6、§7 的原方案。**

---

## 0. 目标与成功判据

- **冻结基线**：开工前按 §3 协议冻结一条本机基线。**实测（rep1，2026-10-03，见 §12）：steady decode 22.0–24.4 t/s（2K→32K，对上下文几乎零衰减）**——远高于调研文档记录的 15–16 t/s（那是上游 bd66c40 水位；fork 的流水线/融合波已经吃掉了大头）。所有"预期百分比"以这条新基线为分母重估。
- **成功判据**：
  1. 每一项独立合入前后 **逐位零 DRIFT**（HANDOFF §7.2 同款闸门）；
  2. 每项有 ABBA 实测 + speed-bench CSV 存档（命令/机器/量化/失败项全记录，对齐 argonaut ladder 惯例）；
  3. 全部做完后 steady decode（2K/8K/32K 三档中位）相对冻结基线 **≥ +12%**（无硬件相），加 NVMe 副本相 **≥ +25%**。达不到就如实记录负结果，不硬凑。
- **不做清单**：不重开 #1041/#1035/路由融合（已在仓库，见 §1）；不碰 DSpark 路径；不引入任何改输出数的"近似"优化。

## 1. 现状盘点：真实差距比调研文档写的更小

逐项核对 `V41_M5MAX_SPEED_RESEARCH.md` §2 的"未取"清单，多数**其实已经在 fork 里**（文档那几行早于 #1090 移植与 P0 系列工作，需要更新，见 §8）：

| 上游/argonaut 项 | 状态 | 证据（file:line） |
|---|---|---|
| #1041 单盒 decode 命令排队 | **已取，默认开** | `ds4.c:41866-41868`（注释即写 "Upstream #1041"） |
| #1090 decode 流水线（Engram 表独立输入、layer13 不再 drain、logits head 提前编码、Engram 读与 layer 0 重叠、il∈{0,4} flush），M5 全量准入 | **已取，M5 默认开** | `ds4.c:41826-41836`（`ds41_decode_pipeline_admitted`），flush `ds4.c:41952`，early logits `ds4.c:41947` |
| #1035 Engram decode 并行读 | **已覆盖**（fork 自己的 step-reader/batch readers 即 dispatch_apply 并行） | `ds4_engram.c:197`、`ds4_engram.c:380` |
| #1042 路由单发融合（matvec+mega-gate，M5 门控） | **已取** | `ds4_metal.m:50170-50192`、`metal/dsv41.metal:721+` |
| #1042 HC glue 19→6 / shared-expert 12→3 融合族 | **已取**（`kernel_dsv41_hc_expand4_bf16`、`kernel_dsv41_shared_gate_up_swiglu_q8_0`、`kernel_dsv41_hc_collapse_norm4` 等均在树内） | `metal/dsv41.metal`、`metal/dsv4_hc.metal` |
| #1042 **残余 attention-glue 切片** | **确认未合入**：`kernel_dsv41_bf16_rope`、`kernel_dsv41_mul_mv_f16_f32_4_bf16`、`kernel_dsv41_qkv_norm_kv_tail`（HEAD 0 命中；源提交 f61a83d 从不是 main 的祖先，git merge-base 已验证）。#1042 其余部分由我们的 `9309a29`/`e2e3563`/`ef45eec`/`0319311`/`a226fd4` 覆盖 | 归因已定：attention-glue 整块没要，非 NAX 替代 → §5 |
| #1073 | decode 排队/flush/Engram 预取已被上面三者覆盖；shared-expert 并发段 PR 里就 gate `!g->streaming`；DSpark 段驻留专用 | 不再单独取 |
| argo #1151-1 mailbox（router ids 写 GPU mailbox，CPU 不等回读、miss 读与 shared expert 编码重叠） | **部分等价物已在**：`g_selected_readback_event`（MTLSharedEvent，`ds4_metal.m:379`）、selected-id 缓冲/热表（`ds4_metal.m:1146`、`1980-2047`）。剩余差距未量化 → §6 | 先测再做 |
| argo #1151-3 post-MoE flush + eviction prescan（+1.5%） | 缺 | 依赖 §6 的"等待窗口"，随 §6 做 |
| argo #1151-4 路由 fastpath + 小融合 | 与 #1042 路由融合同一件事，**不要双跑** | — |
| argo #1151-4 BF16 rounding 折叠 ×3 | ≈ 上表缺失的 3 内核（约 +1.5%，同源思想） | §5 |
| argo #1151-5 读路径（9 常驻线程、split reads、down-proj 重叠、异步 Engram、32-token hotness decay） | **部分已在**：常驻 pread 线程池（≤18 线程）`ds4_metal.m:13738-13847`；异步 Engram 已在；down-proj 重叠 / hotness decay 缺 → §7 差距分析后再定 | — |
| argo #1151-2 多盘 byte-identical 副本（18.66→21.05→22.25） | 缺（需第二块盘，硬件决策） | §6.5 / 待用户确认 |
| argo "锁 4,600 experts"（+3%，机器特异内存策略） | 缺一个扫描：我们有 `--ssd-streaming-cache-experts` 旋钮，没对本机扫过 | §4 |

**结论：可动的新肉 = ①cache 容量扫描 ②3 个残余融合内核 ③mailbox 剩余差距 + flush/prescan ④读路径残余差 ⑤（可选）NVMe 副本。**

## 2. 全局纪律（每项通用）

1. **执行窗口**：所有 bench/parity 都是模型加载类操作——动手前按全局铁律检查内存余量、确认无其他大模型驻留；一次执行窗口把该相所有腿跑完。
2. **DSpark 关闭**：执行期不设 `DS4_EXPERIMENTAL_DSPARK_STREAMING`、不带 `--dspark/--mtp DSpark 支持件`，保证 A/B 臂干净。
3. **逐位闸门先于速度**：每项合入顺序 = build → `make test` → `./ds4_test --logprob-vectors` → `tests/test_deepseek41_dspark` 的 streaming 腿（`check_short_prefill` streaming，`tests/test_deepseek41_dspark.c:169`；`check_verify_parity` streaming，`:50`）→（§6/§7 项加跑 `tests/test_metal_ssd_experts`）→ 才允许进速度门。任何 DRIFT 即否决，机制保留为 opt-in env（对齐 P0b 先例）。
4. **速度门**：ABBA 两对（A=基线臂，B=改动臂，交替），同 session  interleaved，取中位；报告 steady（去首步）。CSV + 一页 md 存 `speed-bench/`（含命令、env、量化、臂序、失败项）。
5. **每项一个 commit + env 关闭开关**；开关命名对齐现有风格（`DS4_METAL_DISABLE_V41_*`）。
6. **不叠加同族融合**：路由已融合（#1042 已取），argo 的 router fastpath 不再取。

## 3. 阶段 0：基线冻结与归因拆解（零改动，半天执行窗口）

产出三样东西：基线 CSV、时间归因表、一个可复用的 ABBA runner（仓库里目前没有 V4.1 流式 A/B 脚本，本次建一个，如 `speed-bench/v41_m5max_streaming_ab.sh`，后续每项直接调用）。

**协议**（对齐 argo ladder 与 HANDOFF §3 实测条件）：
- 模型：`gguf/DeepSeek-V4.1-Flash-Q2.gguf`；`--metal --ssd-streaming --warm-weights -c 393216`（与生产命令一致；cache 用 auto，另见 §4）；greedy；prompt `speed-bench/promessi_sposi.txt`。
- 三档上下文：prefix 2K / 8K / 32K，各生成 ≥512 token，标注 warm（cache 已热）/ cold；每档 ≥4 次重复取中位。
- 归因拆解：同一腿开 `DS4_METAL_V41_DECODE_HOST_PROFILE=1`（`ds4.c:41847`，现成的 host 计时）记录每 token 的 GPU 执行 vs host 等待；再跑一条 `metal_decode_schedule_bench --ssd-streaming` 腿记 command buffer 数。**这张表直接决定 §6 值不值得做**：如果 host 等待里"router 回读等待"占比已经很小（fork 的 selected-id overlap 已吃掉），mailbox 项降级或砍掉。

**成功判据**：基线数字 + 归因表 + runner 脚本入库；无任何代码行为改动。

## 4. 阶段 1：零代码旋钮——expert cache 容量扫描（**已执行，2026-10-03：结论=auto 保持不动**）

依据：argo 实测"锁 4,600（vs auto 3,688）= +3%"，纯内存策略；我们有现成旋钮。

**实测结果**（steady 中位，speed-bench/v41_m5max_ab/）：

| 臂 | 2K | 4K | 8K | 16K | 32K | 判定 |
|---|---:|---:|---:|---:|---:|---|
| auto（6959 专家 / 64.5 GiB） | 22.82 | 23.66 | 22.31 | 23.33 | 22.02 | 基线 |
| 8000 专家（74.2 GiB） | 22.99 | 24.15 | 22.00 | 22.83 | 22.18 | ≈auto，噪声内 |
| 4000 专家（37.1 GiB） | 19.58 | 23.04 | 18.45 | 20.76 | **18.29** | 长上下文 -15% |

- 结论：**加大 cache 无收益（argo 的"锁 +3%"不迁移到 Q2），auto 已在拐点右侧平台**；缩小显著变差 → auto 保持默认，本相关闭。
- **连带发现（已证伪，存档防跨会话误读）**：cache≈40 GiB 的 8K/32K 恰好 18.3–18.5 t/s，与用户生产 ~18 数字撞车，曾怀疑"有效 cache 被压低"。server 三臂探针（speed-bench/v41_m5max_server_probe.sh：原命令/+--warm-weights/---vision）实测**稳态全部 23–26 t/s**，参数无罪；18 = 长任务平均口径（冷请求含 prefill：同 256 token 冷 40.5s=6.3 t/s vs 热 9.8s=26 t/s；agent 多轮 append-sweep 仅 63–120 t/s，摊薄总平均）。结论：**参数保持现状，18 与 decode 引擎无关**；唯一残留嫌疑=长任务散热降频（未测）。
- 附带诊断缺口：`DS4_METAL_STREAMING_EXPERT_LAYER_STATS` 的打印点不在 ds41 decode 路径触发（计数器 `note_token` 在增，输出无人调用）——值得一个 3 行诊断 patch 给 V4.1 补上 hit_rate 打印，供 mailbox 相使用。

## 5. 阶段 2：三个残余融合内核（1 天代码 + 1 个执行窗口）

即 #1042/argo "BF16 rounding 折进产生它的内核" 一族：`kernel_dsv41_bf16_rope`、`kernel_dsv41_mul_mv_f16_f32_4_bf16`、`kernel_dsv41_qkv_norm_kv_tail`。

1. **归因（已执行，2026-10-03）**：三内核唯一来源 = #1042 head 提交 f61a83d "fuse the decode attention glue; shared down before the routed experts"，该提交从未进过 main；fork 的 0319311/e2e3563/ef45eec/a226fd4 只覆盖了 #1042 的 router/HC/MoE/expand4 部分。**attention-glue 切片整块未要**，非替代关系。（f61a83d 的 "shared down 提前到 routed 专家之前" 同时就是 argo 读路径的 down-proj 重叠思想——移植此项可顺带吃掉 §7 的一个子项。）
2. **移植**：从 /tmp/pr1042.diff 对位取 attention-glue hunk（`metal/dsv41.metal`、`ds4_metal.m` 对应 dispatch、`ds4_deepseek41_gpu.h` 接口），保留各自 rounding 边界；host 侧接入点按 fork 现状改写（树形已分叉，手工 rebase，不整文件覆盖；注意与 e2e3563/ef45eec 已改写的 glue 调用点冲突）。上游随附 `tests/test_deepseek41_metal --attn-fuse` 断言一并取。
3. **开关**：一个 env 关全族回原路径。
4. 门：§2.3 全套 + 上游 `tests/test_deepseek41_metal` 新增断言一并取。
- 预期：+1~3%（上游 M5 Max 整族 +20% 已取走，剩的是零头）。

## 6. 阶段 3：mailbox 剩余差距 + post-MoE flush + eviction prescan（本方案主菜，2–3 天代码）

**先测后做**（§3 归因表是门槛）：若 host wait 中"等 selected-ids 回读→才开始 miss 读"的窗口仍 ≥ 单步的 15%，做；否则降级为只记录。

- 现状 vs argo 的差距假设：fork 已有 `g_selected_readback_event` 与流式 selected-id 缓冲，可能已覆盖"不等 GPU 命令缓冲结束就拿 ids"；argo 多做的是——**拿到 ids 后立刻查 expert cache 并在"shared expert 还在 GPU 上跑"期间发起 miss pread**，以及 CPU 完全不阻塞（poll + 50 ms 阻塞回读兜底）。
- 改法（尽量薄）：
  1. 在现有 event 路径上，把 `ds4_gpu_stream_expert_pread_*`（`ds4_metal.m:13738+`）的触发点从"回读完成后"提前到"mailbox 序列号就绪后"；
  2. **仅 `n_tokens == 1` 生效**——多行（DSpark verify/sweep）保持现路径，避免与 `ds41_streaming_verify_moe_gather_admitted` 交叉；
  3. eviction prescan：CPU 自旋等待期间给驱逐候选排序，用时逐个复核（保守：复核不过就按原路径重读，不改驱逐正确性）；
  4. post-MoE flush：仅当该层确实等了 expert 读时才提前 commit。
- 门：§2.3 全套 + `tests/test_metal_ssd_experts`；额外跑一条"cache 极小（如 1GB）"腿压 miss 路径。
- 预期：+4~8%（argo 同硅片 Q4 实测 +6.6%+1.5%；Q2 单专家更小、miss 更便宜，收益略缩）。
- 回滚：`DS4_METAL_DISABLE_V41_ROUTER_MAILBOX=1`。

## 7. 阶段 4：读路径残余差距（先 diff 后移植，1–2 个执行窗口）

> **⚠ 2026-10-03 改向注记（详见 §12.6）**：归因实测证明 miss 读已被线程池+超读完全隐藏（`missing_wait_avg=0.004 ms`，免回读臂实测 **-2.7%**）。本节从"主菜"**降级为附项**。阶段 4 新主战场 = **V4.1 Q2 decode GPU kernel 工作 vs GLM NAX/#1090 族变体的逐项 diff**（GPU 执行占 cb 窗口 49.7%；GLM NAX 域 M5 实测 +21~23%）。开工步骤见 §13。

argo #1151-5 是"大部分单盘收益"所在，但 fork 已有一大块（线程池、异步 Engram）。逐项 diff `argonautlabsai/ds4-argodrive`（钉在 bd66c40，与我们分叉远，**按单项 diff，不 cherry-pick commit**）：

| 子项 | fork 现状 | 动作 |
|---|---|---|
| 常驻读线程 + split reads | 有池（≤18），split 未见 | diff 其切分参数，若实测有肉只移植参数层 |
| 常驻 expert 的 down-proj 与 miss 读重叠 | 未见 | 可做：命中专家的 down 投影在 miss pread 期间派发；中等改动，单独 ABBA |
| 32-token hotness decay 驱逐 | **已有等价物（已查实）**：715ed7a "Age Metal expert caches from the first routed layer"，`DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS=16` 的减半衰 + `note_route_hotness`，V4.1 routed_moe_one（ds4_metal.m:43099）每 token 喂热 | **裁决跳过**（只剩参数微调空间，非移植项） |
| 异步 Engram 行读 | 已有（step readers） | 跳过 |

- 每项独立 commit + 独立 ABBA；预期合计 +2~6%，但**以 diff 结果为准，可能砍到只剩一项**。

## 8. 阶段 5（可选硬件相）：byte-identical 副本盘加权读

argo #1151-2：`--ssd-streaming-replica PATH`（+ 10:6:6 权重），同机 18.66→21.05（TB5 单副本）。
- 前提：**需要你确认有没有第二块盘**（TB5 NVMe 或另一内置卷；Q2 全文件 ~341 GiB，副本盘要放得下）。
- 改法照抄 argo 语义：无 replica flag 时零行为变化；副本只服务 miss 读，权重按盘实测带宽配。
- 值得单独评估的 Q2 变体：Engram 表（189 GiB、每 token 必读、行随机）放副本盘、主权重留内置盘——比整文件副本少占空间，读分布可能更好；做一个"engram-only 副本"臂一起测。
- 预期：+8~15%（取决于盘）。无盘则整相跳过。

## 9. 阶段 6：文档回写（随手做）

- `V41_M5MAX_SPEED_RESEARCH.md`：§2.1 表更正（#1041/#1042/#1035 已在，#1073 结论补"streaming 下上游 DSpark 被 `g->streaming` 硬门控"证据行）；§4 基线表补本次冻结基线与最终数字（附条件：Q2/`-c 393216`/warm-weights/档位/日期）。
- 每项的"有意偏差/否决理由"按 HANDOFF 惯例记档（例如：若 mailbox 归因后收益不足 → 记"fork 已有 selected-id overlap，残余窗口 X ms，不值得第二套机制"）。

## 10. 执行顺序与收益台账

```
阶段0 冻结+归因 ──┬─→ 阶段1 cache 扫描（并行，零风险）
                  ├─→ 阶段2 三内核（独立）
                  ├─→ 阶段3 mailbox+flush/prescan（等归因表结论）
                  ├─→ 阶段4 读路径 diff（等归因表结论）
                  └─→ 阶段5 副本盘（等你确认硬件）
```

| 相 | 预期（诚实区间） | 风险 | 状态 |
|---|---:|---|---|
| 1 cache 扫描 | +0~4% | 极低 | **已关闭：保持 auto（§12.1）** |
| 2 三内核 | +0~2%（实测） | 低（逐位门兜底） | **已移植实测：≈中性（§12.2–12.4），保留合并选项** |
| 3 mailbox+flush | — | — | **已裁决砍掉（§12.6：读已隐藏、免回读实测 -2.7%）** |
| 4 GPU kernel 工作（改向）+ 读残余 | 未知（新主战场，§12.6） | 中 | 待开工 |
| 5 副本盘 | +8~15% | — | **已放弃（2026-10-03：无第二块 NVMe）** |
| **合计（无硬件）** | **+12~22%** | | **⚠ 阶段 2 实测归零、阶段 3 砍掉后，达标全押阶段 4（§12.6 改向后的 GPU kernel 主战场）——先按 §13 开工步骤出逐项 diff 结论再定预期** |

## 11. 风险与交互红线

1. **DSpark 交叉**：§6/§7 全部限定 `n_tokens==1`；驱逐策略改动要过 streaming verify parity（verify 的多行 gather 依赖 expert cache 不被提前逐出——P4 注记"draft 不碰 cache"同理适用于这里，驱逐 prescan 只排序不改变正确性）。
2. **NAX 冲突**：§5 触碰 `ds4_metal.m` 融合 dispatch 区，你们 GLM53/NAX 调优（9bc0bd7）改过同区域——手工 rebase，取 hunk 不取文件。
3. **内存**：阶段 1 大 cache 臂必须盯 swap/pressure；阶段 3/4 的常驻线程若加数量，同步核 128 GB 预算。
4. **别把 sweep（prefill）改动混进来**：P0 系列是 prefill sweep 域；本方案是 decode 域。归因表分开两张，别串线（§17 的"路径串线"教训）。
5. 全部执行窗口的前提：**本机没有其他大模型驻留**（当前不满足，先排队）。

## 12. 执行日志（实测记录，按时间序）

### 12.1 基线与 cache 扫描（已定）
- Fork 3b2abc4 工作树、Q2 `--ssd-streaming` auto cache（6959 experts/64.51 GiB）、warm、512 greedy：
  稳态 decode 中位数（n=4）：2K 22.82 / 4K 23.66 / 8K 22.31 / 16K 23.33 / 32K 22.02 t/s（spread ≤7%）。
  文档旧值 15–16 t/s（上游 bd66c40）作废，以本表为锚。
- cache8000（74.2 GiB）≈ auto（噪声内）；cache4000（37.1 GiB）18.29–19.58（长 ctx -15%）。
  **阶段 1 关闭：保持 auto。Argo "锁 cache +3%" 不迁移到 Q2 streaming 域。**
- 服务端探针（用户命令原样、3×256 greedy）：req1 40.5s（冷 prefill）→ req2/3 23.3/26.1 t/s。
  "18 t/s" = 长任务平均口径（prefill+append sweep 稀释），与 decode 引擎无关；--warm-weights/--vision 差异为噪声。
  遗留疑点：长程热节流（用户侧口径未回）。

### 12.2 阶段 2：#1042 attention-glue 切片（f61a83d）移植实录
工作树 `/Users/xieyongliang/ds4-attn-glue`，分支 p42-attn-glue（基 3b2abc4），9 处冲突全解，编译一次通过。移植中的三处实质修正：

1. **kernel 去重**：PR 在 dsv41.metal 重定义了 fork 已有的 `kernel_dsv41_mul_mv_q8_0_f32_bf16`（fork 版在 dense.metal，语义相同）→ 删 PR 副本。
2. **域准入（本案核心发现）**：PR 的 `ds4_gpu_dsv41_matvec_bf16`（Q8+F16 存储层 BF16 matvec）没有 fork 的 `ds4_gpu_dsv41_exact_admitted` 域闸——**fork 策略：读权重的 exact 融合禁入 quality/streaming/tp≠1/非 M3-Ultra 实测配置；纯激活融合有 M5 streaming_port 豁免**。裸移植在 streaming 域折叠未认证权重读路径 → logprob 24 红（main 仅 2 红）。修复：q8/f16 两支都换成 exact_admitted + classic-walk 名称卫哨；f16 另加 nr0==4 卫哨（内核 NR0=4 写死，plain dispatch 默认 nr0=2 会覆盖错位）。**后果：M5 streaming 域 matvec 折叠恒关（与 fork 现状一致），本片在 M5 的增益全部来自激活类融合（qkv_norm_kv_tail/bf16_rope/logits collapse/early-down）。**
3. **fork 特性回并 PR 新函数**：`ds41_attention` 重加 raw_log_write（融合内核 FP8 行同步回写 g->kv，等价成立）、gather_reuse、attention epilogue arm/used（与 fused 尾互斥）；logits encode 内折叠 PR 的 hc_collapse_norm（sinkhorn=0 模式）。

### 12.3 阶段 2 正确性闸门（M5 Max 实测）
| 闸门 | 结果 | 说明 |
|---|---|---|
| make 全量 | 绿 | 一次通过（去重后） |
| test_deepseek41_metal --attn-fuse | **绿** | matvec/qkv tail/rope/collapse byte-identical |
| --hc-fuse | 绿 | 10→3 dispatch，输出逐位一致 |
| --moe-fuse（probs memcmp :526） | 红 = **main 同点同象**（净 main 亦红）| M5 既有红，非移植回归 |
| ds4_test --logprob-vectors（streaming） | **红集==main**（仅 short_code_completion step0，2 条，fork 既有漂移） | 域准入修复后一致 |
| dspark --short-prefill-ssd | 绿 | 41+7/8 行 max diff 0 |
| dspark --verify-parity-ssd | 红 = **与 main 逐位一致**（6 行 differs，数值相同） | fork 在 M5 streaming 的既有状态 |
| test_deepseek41_cuda | 红 = main 同（macOS 不适用） | — |

（ABBA 台架见 12.4，占位）

### 12.4 阶段 2 ABBA 台架（glueA=ATTN_FUSE off / glueB=默认，n=2 腿，gen_steady_t/s）
| 前沿 | auto(n=4 冻结) | glueA | glueB | B vs auto | B vs A |
|---|---:|---:|---:|---:|---:|
| 2K  | 22.82 | 22.91 | 22.57 | -1.1% | -1.5% |
| 4K  | 23.66 | 23.96 | 23.80 | -0.5% | -0.7% |
| 8K  | 22.31 | 21.62 | 22.29 | -0.1% | +3.1% |
| 16K | 23.33 | 22.59 | 23.07 | -1.1% | +2.1% |
| 32K | 22.02 | 21.36 | 21.93 | -0.4% | +2.7% |
| 均值 | 22.83 | 22.49 | 22.73 | -0.4% | +1.1% |

读数：≥8K 三个前沿 4/4 对比较 B>A（+1.3~+4.4%，方向一致，真实约 +2%）；2K/4K 3/4 对 B<A（≈-1%）。
净效应 **glue 在 M5 streaming decode ≈ 中性偏正（+0~2%，长 ctx 更好），远低于 +12% 门槛**。

**归因**：M5 Q2 streaming decode 是 **IO/缓存命中主导**，不是 dispatch 主导——省下的每层 ~4-6 次 dispatch（各 ~10µs 级）
被异步编码管线藏在 SSD 读专家权重的等待后面；且 #1042 在 M3 上的大头（matvec 折叠，省 bf16 量化往返）
被 fork 域准入正确地挡在 streaming 域外。epilogue 折叠 fork 早已拥有。**结论：本片保留价值在位等价与上游对齐，
不是本机性能来源；+12% 目标只能从 §6/§7/§8（IO 域）出。**

### 12.5 用户决定（2026-10-03）
- **阶段 5（副本盘）放弃**：无第二块 NVMe。§8 作废。
- **"18 t/s" 结案**：插电、多轮对话实际场景 → 即 §12.1 结论（平均口径被每轮 prefill/append sweep 稀释；引擎稳态 decode 无异常）。热节流疑点排除（插电）。
- **阶段 3 开工**：IO 域归因优先。

### 12.6 阶段 3 归因实测（2026-10-03，pristine 3b2abc4，两条诊断腿 + cb 级分解）

**稳态 cache 画像**（MEMORY_REPORT+LAYER_STATS 腿，5 前沿×512 greedy）：
- 每层 slots=384，实住 ~170，**hit_rate 0.967–0.982**（均值 0.974）；miss ≈6.2 专家/token（9.49 MiB/专家），全 run miss 读 ~305 GiB。
- miss 读**完全被隐藏**：`missing_wait_avg=0.004 ms`，readahead 线程池已跑 121.3 GiB 超前行；`load_pread_avg=1.79 ms` 全部并行。
- cache8000 平坦 + cache4000 长 ctx -15% 联证：**auto 恰在"读可被计算隐藏"的拐点平台上**。

**selected-ids 回读真相**（TIMING_SUMMARY 腿）：`selected_calls=102400`（每 token 每层一次）
`read_avg=0.883 ms`（几乎全是 sync）——名义上占 41×0.88≈36 ms/step。
**但这是"GPU 时间换出口的地方"，不是可删的泡**：验证器臂
（`DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR=1`，该分支不做 end/wait/begin）实测
read_avg→**0.000** 而速度 **-2.7%（n=2 稳定）**——消除同步不省时间，还倒贴验证工作。

**cb 级 GPU 时间分解**（`DS4_METAL_CB_TIMES`，诊断树 p3-io-diag 放开 400 条打印上限，稳态窗口 n=14272 cb）：
| 分量 | 均值 | 占比 |
|---|---:|---:|
| GPU 执行 | 365 µs | **49.7%** |
| queue-wait（GPU 忙别的=重叠在工作） | 162 µs | 22.1% |
| GPU 空泡（gap-to-next-cb） | 178 µs | 24.3% |
| driver 提交 | 28 µs | 3.9% |

**裁决**：
1. **§6 mailbox 砍掉（"仅记录"）**：miss 读已被隐藏（wait 4µs），回读窗口被 GPU 执行填满，免回读的现成机制实测为负。argo +6.6% 的前提（CPU 阻塞 miss 读、盘慢于算）在本 fork V4.1 Q2 上不成立。post-MoE flush/prescan 随之砍（依赖同一窗口）。
2. 空泡只有 ~24% 且由 bind/begin/encode 细项组成，host 侧可回收上限 ≈5–8%，且需要不依赖 id 的编码路线（SPLIT 机器在但 missing≥3 触发稀少，0.1%）。
3. **真上限在 50% 的 GPU kernel 时间**：Q2 decode 每 token GPU 执行≈22ms，若 kernel 效率对齐 GLM NAX 族的 +21~23%（#1090 家族在 M5 的已验证水位），才有 ≥12% 量级肉。**阶段 4 主战场改为：V4.1 Q2 decode 的 MoE/attention GPU kernel 与 GLM NAX 变体的逐项 diff**（原"读路径残余"降级：读已隐藏）。

## 13. 移交快照（2026-10-03 晚，供下一个 session 冷启动）

**一句话现状**：基线 22.0–23.7 t/s（auto cache）已冻结；阶段 1/2/3/5 全部关账（保持/中性/砍掉/放弃，见 §10 台账）；**+12% 目标全押阶段 4 = V4.1 Q2 decode GPU kernel 工作 vs GLM NAX 族变体的 diff**（§12.6 裁决）。

### 13.1 阶段 4 开工三步（先做 a，一次静态 diff 就能定方向）
> **⚠ 2026-10-04 (a) 已完成并改写结论，见 §14**：(a) 的产出是逐 kernel 卡片表（§14.1/§14.3），两大预设被推翻——quality 门是误判（`--quality` 默认 off，生产进程 mpp_available==true）；NAX/tensor-API 不覆盖 decode 单 token 形状（四族 matmul2d 全要 ≥16/≥32/≥512 行 B）。**decode 热路径上没有"被门挡住的现成变体"**，肉在自种 kernel 调优：C=logits 508 MiB matvec tiling（主攻）> D=attention mmp 适配 > E=topk fast 准入 > B=Q8 形状扫；A（MoE routed SIMD→NAX）不可行关闭、F 低优先挂起。开工顺序改走 §14.4。
1. **(a) NAX/tensor-API 准入盘点（半天，纯静态）**：GLM 族在 M5 的 +21~23% 来自 NAX/#1090 家族调优（参考 commit 9bc0bd7 区域）。查 V4.1 Q2 decode 热路径 kernel——MoE `kernel_dsv41_mul_mv_*` / `g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline` / `g_moe_mul_mv_addr_q2_k_sum6_pipeline` 与 attention 管线——有没有 NAX（Metal4 tensor API）变体；准入函数链：`ds4_gpu_mpp_available()`（ds4_metal.m:2920，`g_metal4_tensor_api_enabled && !g_quality_mode`——**模型进程 quality=ON 时恒 false，是首要嫌疑**）、`ds4_gpu_dsv41_exact_admitted`（:10545，weight-reading 融合禁 streaming 域）、`ds4_gpu_dsv41_activation_exact_admitted`（:10566，M5 streaming_port 豁免先例）。产出：每张 kernel 一张卡——有变体被门挡住（改准入，小）/ 无变体（要移植，中）/ 不适用。
2. **(b) 逐卡预期与排序**：按 cb 分解里 GPU 执行 49.7% 的构成（可用 `DS4_METAL_CB_TIMES=1 DS4_METAL_CB_TIMES_MAX=400000` 在 ds4-io-diag 树复测，或加逐 kernel timing）给每张卡排收益上限，从高到低做。
3. **(c) 每项独立 worktree（从 3b2abc4 新开）+ §2.3 全套闸门 + ABBA ≥2 对**。附项（§7 降级表）只有 split-reads 参数与 down-proj overlap 两项未查实，热已排除；在主战场出结果前不动。

### 13.2 资产与路径
| 资产 | 位置 | 状态 |
|---|---|---|
| 主树 | `/Users/xieyongliang/ds4` | **WIP 勿动**：`DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS` 相关未提交改动；本 plan/research 文档改动也在工作区（tracked，未 commit） |
| 阶段 2 移植 | `/Users/xieyongliang/ds4-attn-glue`（分支 p42-attn-glue，自 3b2abc4） | **全部未提交**，等用户 review 合并；勿在此堆阶段 4 改动 |
| 诊断树 | `/Users/xieyongliang/ds4-io-diag`（p3-io-diag，自 3b2abc4） | 仅 7 行诊断补丁（`DS4_METAL_CB_TIMES_MAX` env 放开打印上限，零行为改动），可留可弃，ds4-bench 已建 |
| 对照树 | `/tmp/ds4-main` | pristine 3b2abc4 + ds4_test/ds4-bench 已建 + runner 已拷；**重启机器后 /tmp 会清空，需重建** |
| 台架 | `speed-bench/v41_m5max_streaming_ab.sh`（各树各一份） | `leg TAG REP [ENV=VAL...]` / `summary TAG CSV_DIR`；CSV 在 `speed-bench/v41_m5max_ab/`（auto-1..4=冻结基线，glueA/B、val-1..2、diag/diagt/cbs-1 在案） |
| 模型/提示词 | `gguf/DeepSeek-V4.1-Flash-Q2.gguf` / `speed-bench/promessi_sposi.txt` | runner 用 MODEL/PROMPT/CSV_DIR env 可覆盖（其他树指绝对路径） |

### 13.3 诊断 env 速查（全部零行为改动）
- `DS4_METAL_MEMORY_REPORT=1` + `DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1`：每层 hit/miss/evict/miss_pread/pread_ms（report 打印点 ds4.c:40304 在 V4.1 路径被 MEMORY_REPORT 门控——两个都开才有输出）。
- `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`：`selected_calls/read_avg/sync_avg`、`load_pread_avg`、`missing_wait_avg`、readahead 量——**阶段 3 裁决的主力仪表**。
- `DS4_METAL_CB_TIMES=1`（+诊断树的 `DS4_METAL_CB_TIMES_MAX`）：逐 cb driver/queue-wait/gpu/gap 四分量。
- `DS4_METAL_V41_DECODE_HOST_PROFILE=1`：host 侧分段（hash/layers/logits）。

### 13.4 本机既有红项（=main 位一致则非回归，别追）
`test_deepseek41_metal` moe-fuse probs memcmp（:526）、logprob short_code_completion step0（2 断言）、dspark `--verify-parity-ssd` 6 行、`test_deepseek41_cuda`（macOS 不适用）。单元测试跑不到模型进程的域准入（quality/streaming 标志不同）——**融合准入类改动必须过模型级闸门（logprob-vectors streaming 腿 + dspark short-prefill-ssd）**，阶段 2 翻车复盘结论。

### 13.5 铁律回顾（本任务特有）
单模型加载、动手前查 swap/无 LLM 驻留、后台 job 跑加载腿并盯内存；主树 WIP 不碰；worktree 间禁 `git checkout HEAD --`；结论诚实记录负结果。

## 14. 阶段 4(a) NAX/tensor-API 准入盘点（2026-10-04，纯静态，主树只读）

> 执行方式：全主会话静态 diff，零代码改动、零模型加载。所有 file:line 以工作树 HEAD 3b2abc4 + WIP 为准（ds4_metal.m 53083 行）。
> **先纠偏**：§13.1(a) 里"模型进程 quality=ON → mpp_available 恒 false，首要嫌疑"**不成立**——`ds4_gpu_set_quality(e->quality)`（ds4.c:73482/73525）只随 `--quality` CLI 开关（ds4_cli.c:2057）置位，默认 off；生产解码进程 `ds4_gpu_mpp_available()==true`（M5 + Metal4 probe 通过，`ds4_metal.m:3091`）。quality 是标量对照腿的开关，不是 V4.1 的隐藏门。

### 14.1 V4.1 Q2 流式 decode 逐层 GPU kernel 清单（实测路径，n_tokens==1）

| # | 环节 | kernel（host_name） | 量级/token/层 | NAX 变体 | 在 M5 streaming 的实际门 | 判卡 |
|---|---|---|---|---|---|---|
| 1 | router（384→6） | `kernel_dsv41_router_select` | F32 权重 0.75 MiB + 单发 select | 无 | fork 已有 M5 单发派发（fused_matvec，ds4_metal.m:50170） | 不适用 |
| 2 | routed gate/up+SwiGLU | `kernel_mul_mv_slots6_iq2_xxs_pair_swiglu_f32`（moe.metal:3924） | 读 ~6.3 MiB | 无 | `use_iq2_selected_slots`（:42528，expert cache 域） | 卡 A |
| 3 | routed down+sum6 | `kernel_mul_mv_slots6_q2_K_sum6_f32`（moe.metal:6071；v41 静态形状模板 :6068） | 读 ~3.2 MiB | 无（同上 impl 家族） | 同上 | 卡 A |
| 4 | shared expert | `kernel_dsv41_shared_gate_up_swiglu_q8_0` + `kernel_mul_mv_q8_0_f32` | Q8_0 ~5.3+5.3+2.7 MiB | 无（Q8_0 走 simd dot，非 matmul2d 布局） | 激活融合已开（streaming_port） | 卡 B |
| 5 | logits head | `kernel_mul_mv_q8_0_f32`（dense.metal:210） | **~508 MiB**（vocab 129280×5120×(5120/32)×34B/5120） | 无 | plain matvec，`nr0=2/nsg=4`，M5 侧仅有 dispatch 微调先例（`ds4_gpu_plain_mv_single_row`，:5626，≤1024 行才用） | 卡 C |
| 6 | attention QKV/共享投影 | `kernel_mul_mv_q8_0_f32` 等 | 合计 ~10 MiB | 无 | 同 5 | 卡 B/C |
| 7 | HC mix（4×5120） | `kernel_dsv41_hc_rms_norm_mix_f16` | F16 160 MiB | 无 | fork 融合已开 | 不适用 |
| 8 | attention decode | `kernel_flash_attn_ext_vec_f16_dk512_dv512`（split-K vec，非 simdgroup_matrix 版） | KV 读 ≤512×512×4B≈1 MiB(comp)+raw | 无 vec 域变体；全仓库唯一的 MPP flash（`kernel_flash_attn_ext_mpp_f16_dk512`，flash_attn.metal:1260）**prefill 专用**（门 :38085 `n_tokens>=16`） | — | 卡 D |
| 9 | indexer top-k | `kernel_dsv41_topk_*` 链 | 小 | fast 变体在 | 卡 E |
| 10 | compute-copy | blit | ~90 次/token 小拷贝 | — | `kernel_ds4_copy_words` 融合被 exact 门挡（:9641） | 卡 F（低优先） |

### 14.2 关键机制事实（本次静态查实）

1. **NAX 只长在 matmul（多行 B）上，decode 单 token 没有可挂的 NAX 形状**：全仓库 TensorOps（matmul2d）kernel 共四族——`kernel_mul_mm_mpp_direct_rhs`（dense，B≥n64）、`kernel_mul_mm_id_mpp[_packed]`（MoE，`kernel_mul_mm_id_mpp_packed` moe.metal:9112，**必须 packed RHS≥32 行**；门 `use_packed_mpp` :44914 硬性 `n_tokens>=512`）、`kernel_flash_attn_ext_mpp_*`（门 `n_tokens>=16`，:38085）、`kernel_dsv41_indexer_scores_packed`（ 门 `n_tokens>=16`，:19829）。GLM53 的 decode +21%（#1090 族）也**不是** decode 侧 matmul2d——是 M3-Ultra 调优族（fused router/shared、mixed 输入、KDA two-head 等）放开到 NAX 域（OMLX_WAVE_PORT_ANALYSIS.md 附三）；其 decode 主体仍是 `kernel_mul_mv_q8_0_f32` 同款 matvec。
2. **`kernel_dsv41_indexer_pack/_scores_packed`（dsv41.metal:252/277）是 V4.1 域唯一的 tensor-API kernel，prefill 专用**（`n_tokens>=16` 门），decode 不经过它。
3. **MPP MoE packed 门已对 V4.1 streaming 显式放开**（`use_packed_mpp` :44915 `(!g_ssd_streaming_mode || force_resident || use_v41_stream_gather_addr)`，配合 `DS4_METAL_DISABLE_V41_STREAM_GATHER_SPEC_ROWS` WIP）——但那在 `n_tokens>=512` 域，即 **prefill sweep**，不属于 decode 相。红线：勿把这条的收益记到 decode 头上。
4. decode MoE 真实路径 = `use_iq2_selected_slots`（:42528：`g_ssd_streaming_mode && gate IQ2_XXS && down Q2_K && n_expert==6 && n_tokens==1 && fuse_pair_swiglu && direct_down_sum`，kill=DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS）→ 卡 2/3 的 kernel，全部 SIMD 标量点积。**expert-cache 只读 tile 无法走 matmul2d**（packed RHS 需按专家物化 ≥32 行激活；单 token 无行可排）。
5. `ds4_gpu_dsv41_exact_admitted`（:10545）清单复核：`compute-copy`(:9641)、topk fast(:19524)、router_top6(:34631，M5 走 fork 自己的 router_select 单发，此门不触发)、hc_norm(:52990)、matvec_bf16(:53029，**另有 `ds4_gpu_mpp_available()` 硬 return 0**——上游 f61a83d 移植时已按域准入改写)、shared_fusion(:53074，fork 已有激活类替代)。**结论：M5 streaming decode 热路径上不存在"被 exact 门挡住的现成 NAX/调优变体"**——`matmul_q8_0_bf16`（f61a83d 的 matvec 折叠）即使开门也被 mpp_available 二次挡死；§12.4 ABBA 已证此类 dispatch 级融合在本机 ≈中性。
6. **§12.6 空泡 24% 的再归因**：`ds41_decode_pipeline_admitted`（ds4.c:41826，M5 准入、flush il∈{0,4}）+ logits early-encode（:41952）已把 host 编码藏进 GPU 窗口；剩余空泡主要为 **~41 次 logits 头 508 MiB matvec 的尾延迟**（每层排队串行，最后一个 kernel 独占 step 尾部，读 508 MiB ≈ 0.9 ms @ ~560 GB/s，其发射-完成边沿就是逐 cb 的 gap 形态）。与 §13.1(b) 的 cb/gap 复测计划吻合。

### 14.3 逐卡结论与排序（按预期收益上限）

| 卡 | 内容 | 类型 | 预期上限 | 风险 | 建议动作 |
|---|---|---|---:|---|---|
| **C** | logits head 508 MiB matvec：U 形循环 tiling（destination staging）+ M5 形状调参（nsg/nr0/ILDG 扫描）；或 verify 行共享已备（shared-rows 门 :20230，仅 verify 域） | 新 kernel 调优 | **大（第一大单项）** | 中（纯激活读，bit-exact 可对拍；rows>1 域已有 exact 先例） | **先做**：静态写 tiling 方案 → worktree 移植 → §2.3 全套 + ABBA |
| **D** | attention decode vec→mmp（`kernel_flash_attn_ext_mmp_f16_dk512` 适配 decode 行数/门改造） | 蓝本同仓库 | 中（Q2 主导权重下被 MoE 稀释；raw+comp 双段访存要重排） | 中高（数值漂移面） | 出估算（KV 字节占比）后再决定开不开工 |
| **E** | indexer topk fast 在 decode 域准入（0.014→0.0014 ms/层，quantile 路径，max/second-max 标度筛 4096 候选；kill=DS4_METAL_DISABLE_V41_TOPK_FAST 已在树） | 准入改造 | 小（~0.5% decode 端） | 低（bit-exact 对拍即 gate） | 排 C 后，若做需 logprob-vectors + dspark streaming 闸 |
| **B** | Q8_0 dense matvec 形状调参（nsg/nr0 粗扫，共享/logits 之外的 ~10 MiB 段） | 参数 | 小 | 低 | 附带做（diag 树上先扫再定） |
| **F** | compute-copy 融合（blit→kernel，省 2×90 编码器边界） | 准入 | 小（§12.4 同族已证≈中性） | 低 | 缓（除非 cb 分解显示 encoder 边界占比意外） |
| **A** | MoE routed pair/sum6 SIMD 优化 | 新 kernel | **不可达 NAX**（无 B 行） | 高（位精确难保） | **不做**（除非 C/D 之后仍有缺口且另有蓝本） |

### 14.4 阶段 4 修订后的开工顺序
> **⚠ 2026-10-03 (b) 实测已完成并再次改写排序，见 §14.6：decode 墙钟是 host 往返主导（GPU 忙仅 30%），卡 C 上限塌缩到 ~3%，新增卡 H（GLM 式 early-load + 免逐层 readback）为主攻。**

1. **(b) 量化腿**（前置）：在 io-diag 树跑 `DS4_METAL_CB_TIMES=1 DS4_METAL_CB_TIMES_MAX=400000`（512 greedy，2K/32K 各一条）+ `DS4_METAL_ENCODER_TIMELINE`（若 decode 域可用）拿逐 kernel GPU µs，验证 14.2-6 的 logits 尾延迟假设并定 C 卡真实上限（kv/weights 双基线）。→ **已完成（§14.6）**
2. **C 卡移植**：独立 worktree（自 3b2abc4），仅动 `metal/dense.metal` 新模板实例 + `ds4_metal.m` dispatch 分支（默认关、env 开），§2.3 全套闸门 + ABBA ≥2 对。→ **降级为可选（上限 ~3%）**
3. **E 卡**（可选，与 C 不同 worktree，不并行跑 bench）：topk fast decode 域准入 + 闸门同上。
4. **D 卡**：先纸面估算（attention KV 字节/token vs 14.3 C 的收益密度），>2× 差距才立项。
5. **A 卡关闭、F 卡挂起**（负结果入 §10 台账）。
6. **H 卡（新增，主攻）**：见 §14.6-4。

### 14.5 与既有文档的关系
- 本节推翻 §13.1(a) 的两个预设：quality 门嫌疑（14.0 纠偏）、"mailbox→NAX 准入"路径（NAX 根本不覆盖 decode matvec 形状，14.2-1）。
- 与 `SPARSE_MLA_NAX_DSV4_PLAN.md` 不冲突：那是 prefill indexed attention（已落地，M5+ 默认开）；本阶段是 decode 域，其"范围外"注记（decode 三腿已专门优化）与 14.1 表一致。
- `OMLX_WAVE_PORT_ANALYSIS.md` 附三的 GLM53 调优家族是"调优类 variant 在 NAX 域放开"的先例，但它放开的核**在 V4.1 侧均已有 fork 等价物**（router 单发/HC 融合/shared 融合），无现成肉可捡——阶段 4 的肉必须自己种（C/D/E）。

## 14.6 阶段 4(b) 实测裁决（2026-10-03，io-diag 树，主树只读未动）

> 工具：`DS4_METAL_CB_TIMES=1 DS4_METAL_CB_TIMES_MAX=400000`（io-diag 的放开补丁）+ `DS4_METAL_V41_DECODE_HOST_PROFILE=1` + `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` + `/usr/bin/sample`（对生成段主线程 1ms 采样，干净腿复核）。五腿 bench 全部稳态复现基线 21.9–23.8 t/s；swap 全程恒 2931 MB，零内存事故。

### 14.6.1 每 token 墙钟分解（稳态 p50，26 个 CB/token）

| 分量 | 值 | 占墙钟 | 来源 |
|---|---:|---:|---|
| **墙钟** | **44.0 ms** | 100% | marker 间距（26-CB 分组）+ CSV 22.4 t/s 互证 |
| GPU 忙 | 13.33 ms | 30% | 20 层 CB×477µs(9.51) + 5 io CB×~620µs(3.04) + logits CB 1.36 |
| CB 间 GPU 空隙 | 0.02 ms | ~0 | gap-from-prev-gpu-end p50=1µs |
| 队列等待（编码器饿 GPU） | 6.07 ms | 14% | qwait=GPUStart−kernelEnd，p50=246µs/CB |
| host 编码 | 4.0–4.9 ms | ~10% | encode/CB p50=100µs（25 CB），encodes/step=1465 |
| **CB 完成链之外的 host 串行段** | **~19 ms** | **~43%** | 44.0 − Σ(段内 delta 25.0) − logits 1.4 |
| hash+engram / logits 读回 / step 间 | 0.005/0.016/0.021 ms | ~0 | host profile 四桶 |

**结论一：V4.1 Q2 流式 decode 在 M5 Max 上是 host 延迟主导，不是 GPU 主导。** §13 的"GPU exec 49.7%"高估（实测忙时 30%，含饿 GPU 等待 44%）。

### 14.6.2 ~19 ms 隐藏段的归因（`sample` 生成段主线程，干净腿 1719 样本）

- `ds41_graph_step` 占 91%（1566）；其中 **`ds41_moe` → `ds4_gpu_routed_moe_one_tensor` 占 1525（89%）**。
- 二八开：**:43042 readback 边界（`selected_timing` 路径 `ds4_gpu_end_commands`→`finish_command_buffer`→`wait_command_buffer`→cvwait）1359 样本 = 79%**；:43281 专家加载（`load_selected_missing` 102 + `readahead_range` fcntl 63）≈10%。
- 机制：V4.1 流式每层 MoE 都要把 router 选中的 6 个专家 id **读回 CPU**（`ds4_gpu_tensor_read(selected)`，:43042）驱动 expert-cache 装载，为此**每层结束并等待 CB**（:43032 `ds4_gpu_end_commands`）→ 41 次/token 主机-GPU 往返。GLM53 已用 `begin_selected_load_tensor`（router 处一次性读 id + 异步 pread 池 + CB 保持打开）绕开同款问题。
- 树里已有但默认关的 GPU 侧替代：`DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR`（:14820，GPU 验证命中、免 id 回读）。**A/B 实测：2K/4K steady 23.19/23.19 vs 基线 23.48/23.2 —— 无收益**（validator 仍走 host 同步边界；单开它不够）。
- 附带证伪：247 MiB/token 专家预读是 miss 路径（hit 98.1%），非主瓶颈；`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` 的探针自身会把该 readback 变成强制路径，probe-observer 效应，后续测量必须避开该 env。

### 14.6.3 卡 C（logits 508 MiB matvec）改判

实测 1.36 ms/token = 墙钟 3.1%（GPU 时段 10%）。§14.3 的"第一大单项"**不成立**——即便 kernel 提速 50% 也只有 +1.5% 墙钟。原"508 MiB 尾延迟主导空泡"假设错误：24% 空泡的真身是 14.6.2 的 host 往返段。C 卡降级为可选（与 E/B 同批做，不再单独立项）。

### 14.6.4 修订后的卡排序（墙钟杠杆口径）

| 卡 | 内容 | 杠杆上限 | 证据 | 状态 |
|---|---|---:|---|---|
| **H（新）** | 消 41 次/token 的 MoE host 往返：移植 GLM53 `begin_selected_load_tensor` 模式（router 一次性读 id + 异步 pread + 层间 CB 保持打开），或扩 validator 为"GPU 验证 + 地址表直填 + 零回读" | ~~+25~35%~~ 实测 0 | 15 节 ABBA | **负结果，关闭** |
| B/E/F | 不变（小杠杆） | 各 <1.5% | §14.3 | 排后 |
| C | logits matvec tiling | ~3%（非 14.3 的"大"） | 14.6.3 | 降级可选 |
| D | attention mmp | GPU 时段内 ≤1 MiB 级，墙钟 <1% | 14.6.1 | **关闭** |
| A | routed SIMD→NAX | GPU 忙的 9.5ms 里占比可观，但墙钟上限 = 9.5/44，且位精确难 | 14.6.1 | 维持关闭 |

### 14.6.5 H 卡实现路径草案（下轮开工）

1. 参考 GLM53：`ds4.c:59136`（`glm_graph_forward_token` 流式腿）在 router 后立即 `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor`（读 id 一次 + pread 池异步装）+ `glm_stream_selected_prefetch` 复用；V4.1 对应位 = `ds41_graph_layer` 内 router 之后、`ds4_gpu_routed_moe_one_tensor` 之前。
2. 关键不动点：**数学零改动**——只是把"每层 readback id"换成"step 开始一次性读 41 层 id（或逐层但不清 CB）"；expert-cache 装载语义不变；`use_iq2_selected_slots` 域不变。
3. 闸门：`DS4_METAL_DISABLE_V41_MOE_EARLY_LOAD` 默认关=新路径开？**否——先 env 正开默认关**（`DS4_METAL_ENABLE_V41_MOE_EARLY_LOAD`），bit-exact 过 logprob-vectors + dspark 短 prefill/verify 后再翻默认。
4. 风险：`use_iq2_selected_slots` 依赖"ids 已知才能绑 slot buffer"——early-load 只是把等待从关键路径挪到 pread 池，slot 绑定仍需 per-layer 提交前完成（与 GLM 的 `prepare_load_buffers`/reusable batch 相同）。若某层 miss 未装完，回退当前同步路径（per-layer 分支，不损正确性）。
5. 验收：clean ABBA ≥2 对（io-diag 上先跑通再移植主树 worktree）；`DS4_METAL_V41_DECODE_HOST_PROFILE` 的 layers 桶应从 ~45ms 降到 ≤30ms；GPU 忙份额可降到 ~40-50%。

### 14.6.6 本节文件与数据
- 腿 CSV：`speed-bench/v41_m5max_ab/`（io-diag 树）：`b-kernel-1`(512tok×5 frontier)、`b-hostprof-full-1`、`b-pread-1`、`b-sample-0/2`、`b-clean-1`、`b-validator-1`。
- 日志：`/tmp/ds4_v41_b_run.log`、`/tmp/ds4_v41_b_full.log`、`/tmp/ds4_v41_b_pread.log`、`/tmp/ds4_v41_b_clean.log`、`/tmp/ds4_v41_b_sample2.txt`、`/tmp/ds4_v41_b_clean_sample.txt`。
- 采样命令（复现）：bench 后台起，`sleep 24` 后 `sample <pid> 20 1 -file …`（必须 `pgrep -x ds4-bench`，勿采包装 shell）。

## 15. 卡 H 实施与实测裁决（2026-10-04，ds4-cardh worktree，分支 p4-cardh-early-load）

### 15.1 实现（commit 5573d62，基线 3b2abc4，净差 = ds4.c +63 行，数学零改动）

`ds41_moe_partial` 内，env `DS4_METAL_ENABLE_V41_MOE_EARLY_LOAD` 门控（默认关），条件 = 单卡 + streaming + 非 quality + IQ2_K/IQ2_XXS/Q2_K + 6 专家：
1. router select 后在主线程打开的 batch CB 上 `ds4_gpu_signal_selected_readback_ready`（encodeSignalEvent）。
2. `metal_graph_selected_async_load_start_tensor` 起 worker：等事件 → 读 6 个 id → `begin_selected_load` 预暂存 pread（worker 注册为 service 线程，不得等待在途 cache 条目）。
3. 主线程照常编共享专家 → `ds4_gpu_flush_commands()` 异步提交 → `..._finish()` 收 id；worker 暂存失败但 ids_ok 时主线程重试同一暂存 + `ds4_gpu_routed_moe_set_selected_override`；routed kernel 的 override 分支取代逐层回读链。

**实现教训（已入 commit message）**：`ds4_gpu_flush_commands`（ds4_metal.m:9814）提交后会**立刻轮换出新 batch CB**（:9827），`ds4_gpu_begin_commands` 在 CB 已存在时返回 0——收尾必须写成 `if (!ds4_gpu_commands_active() && !ds4_gpu_begin_commands()) return false;`，无条件 begin 会让每层静默失败（症状："V4.1 layer 0 failed at position N"）。另外：改 ds4.c 后必须重编**全部**消费二进制（ds4/ds4-bench/ds4_test/tests/*，`make` 默认目标不含 ds4-bench 的测试目标），否则闸门测的是旧包。

### 15.2 正确性闸门（最终无诊断打印二进制复验，全部通过）

| 闸门 | 结果 |
|---|---|
| CLI 贪心 A/B（temp 0，env ON vs OFF，同 prompt 同二进制） | **逐字节一致**（48-tok 腿 + 收尾 4-tok 冒烟均一致） |
| `ds4_test --logprob-vectors`（streaming） | ON/OFF 同为 3/5；2 个失败 **env OFF 同样复现 = 存量 fixture/配置错位**，与卡 H 无关 |
| dspark `--verify-parity-ssd` | ON vs OFF 差异行 = 0（6 行与官方参考的偏差为存量流式行为，OFF 相同） |
| dspark `--short-prefill-ssd` | max logit diff 0 |
| `test_metal_ssd_experts` | 4/4 PASS |
| bench env ON 冒烟（8 tok，pipeline 默认开） | 320/320 层 signal=1、ok=1，零失败 |

### 15.3 ABBA 计时裁决：**负结果（0~−1.3%）**

`ds4-bench --ssd-streaming --gen-tokens 512 --step-mul 1`，ABBA 各腿成对，steady t/s：

| 腿 | env | 2K | 32K |
|---|---|---:|---:|
| a1 | OFF | 24.33 | 22.64 |
| a2 | ON | 23.21 | 22.78 |
| a3 | ON | 23.80 | 22.17 |
| a4 | OFF | 23.32 | 22.44 |
| 均值 | OFF **23.83** / ON 23.51 | OFF **22.54** / ON 22.48 |

组间差 −1.3% / −0.3%，小于组内噪声（OFF 对本身差 4%）。唯一可辨信号：2K 首 token 64.4/64.9→58.5/56.5 ms（−10%，单样本）。目标 +25% 完全未兑现。

### 15.4 为什么 14.6 的归因错了

14.6.1 把 ~19 ms/token 记为"MoE readback 的 GPU 等待串行"，异步化后应当消失。实测不动，说明这段墙钟**不是可被重叠消掉的等待**：
- decode queue/pipeline 模式下 GPU 只有 ~30% 忙，layer N 的 end→wait 通常**近零返回**（router kernel 早在前层编码期间就跑完了）；
- 本补丁把等待从主线程挪到 worker，但主线程在 `finish()` 处**照样阻塞**，还额外付出：每层一次 flush 提交边界、worker 线程 rendezvous、事件等待；
- 真正吃掉 44 ms 的是**主线程自身的簿记总量**（41 层 × 每层数十次 encode/dispatch/CB/表管理调用的 CPU 执行时间），它与 GPU 快慢无关，重叠手段对它无效。
结论：**墙钟主犯是主线程 CPU 指令量，不是任何等待段**。后续若想再攻墙钟，方向只剩：减少每层主线程 encode 调用条数（如更深的层融合/把整层 encode 移进 Metal graph capture 一次回放），或降 GPU 侧（A/C 卡的老路）。卡 H 及其"异步化 19ms"假设一并关闭。

### 15.5 数据与产物
- worktree `/Users/xieyongliang/ds4-cardh`（分支 p4-cardh-early-load @5573d62，env 默认关、不合入主干）；腿 CSV：`.cardh-ab/ab2_{a1..a4}_{2k,32k}.csv`，脚本 `.cardh-ab/run_abba2.sh`，被取代的旧腿在 `.cardh-ab/stale1/`。
- 正确性产物：`.cardh-ab/gen_off.txt`/`gen_on.txt`（48-tok 字节一致）、`parity_*.log`、`prefill_ssd.log`、`smoke.csv`。

## 16. 卡 I 立项：整层 encode 进 Metal graph capture（砍主线程指令量）（2026-10-04 立项；I-1 已实施并裁决，见 §17）

### 16.1 立项依据（一句话）

§15.4 关账结论：墙钟 44 ms = 主线程自身的 CPU 指令量（~750–900 dispatch/token + ~90 blit + 26 CB 轮转 + 数千参数/簿记调用；GPU 忙仅 13.3 ms），任何"重叠/异步"药方对纯指令量无效，唯一对口药方是**每层只录一次、每 token 回放**（`ds41_decode_island` 的既有模式推向单卡流式）。上限：砍掉主线程 ~20 ms 编码段后墙钟逼近 max(GPU 13.3 ms, 回放开销)，即 44→~20 ms/token 量级（+100% 级），远大于 B/E/F 残值。

### 16.2 静态普查（源码查实，估算口径已标）

- ds4_metal.m：304×dispatchThreadgroups + 17×dispatchThreads + 2×indirect，无集中 launch helper；`ds41_moe_partial` 16 个 GPU 入口；`ds4_gpu_routed_moe_one_tensor` 体 2733 行（大量 per-call peek/staging/CB 切分簿记）。
- 每 token ≈ 1000+ 次 Metal API 调用；30 ms 主线程 ÷ 1000+ ≈ 25–30 µs/次，与单 dispatch CPU 成本吻合（自洽验证）。

### 16.3 可捕获性分类

| 环节 | 判定 | 依据 |
|---|---|---|
| 40 层 attention/matmul/HC/top-k/routed kernel 本体 | **可捕获**（~95% dispatch 量） | 形状/指针逐 token 静态 |
| token id / position / engram 行 | 可捕获（内容更新，指针固定 buffer：`g->rows`、ids buffer） | 以 kernel-arg 标量传参的少数 kernel 需改为读 param buffer（llama.cpp CUDA-graphs 的 staging-buffer 抄法） |
| logits 回读 + host argmax | 留图外 | 每 token 唯一一次同步，pipeline 现状即如此 |
| **router→6 ids→槽位绑定/装载** | **不可捕获 = 全工程核心障碍** | CPU 每层现读现定；须 GPU 化：`use_stream_expert_addr_table` 现成内核路径 + service 线程消费 GPU ids（卡 H worker/inflight 语义直接复用） |
| expert cache miss 的 pread 上传 | 图外旁路，天然兼容 | 本来就是 service 线程独立 CB + 事件对齐 |

### 16.4 外部调研结论（llama.cpp / mlx，2026-10-04）

- ggml-metal **已实现 MTLGraph capture**（`ggml_backend_metal_capture_next_compute`，PR #20398/commit c363256，`GGML_METAL_CAPTURE_COMPUTE`），但**默认关、定位实验**——同类引擎都停在动态入参/缓冲生命周期这道坎上；其 CUDA Graphs 线的解法（形状 keyed 图缓存 + 动态标量写固定 staging buffer、图内读 buffer）可直接抄，实测 graph 执行段 −40%、整体 +14%。
- llama.cpp issue #7456 公开立项"每 token CPU 活动优化"，处方（跨 token 复用计算图）与本卡同构，未走完——我们无先例可整搬，但风险点清单是现成的。
- llama.cpp 无我们的逐层 CPU 路由决策问题（专家 mmap 兜底），ids→槽位绑定 GPU 化只能自研；其 MoE 磁盘流式 RFC #23324 的负结果清单（大 slab 反而慢、内存压力崩盘、零拷贝 slot 正确）与 §15 无冲突。

### 16.5 三段路线（各自独立闸门、独立收益，允许中途止损）

1. **I-1 路由决策 GPU 化**（先不动 capture）：`g->selected` ids buffer 直喂 addr-table 内核路径，service 线程按 GPU ids 自主装载，主线程 per-layer 只留 flush 语义；env `DS4_METAL_ENABLE_V41_MOE_GPU_BINDING` 默认关。收益假设小但可单独测（消每层 readback + staging 决策段，23.8→? t/s），且是 I-2/I-3 的前置。**→ 已实施：bit-exact 全过、ABBA +3.2%/+2.7%，实施细节与两条硬知识见 §17。**
2. **I-2 capture 岛扩到单层**（单卡流式）：把 `ds41_decode_island` 机制的准入从 TP2 非流式推向 `tp_world==1 && streaming`（I-1 后才有意义），keyed 于 ctx 形状；每层 encode 从数百调用降到 1 次回放 + 少量动态更新。
3. **I-3 全 token 一图**：40 层 + logits 头单图，动态输入全走 param buffer；flush/pipeline 语义退化为一次 commit。

- 闸门沿用：bit-exact（CLI 贪心 A/B 逐字节 + logprob ON==OFF + dspark 两测 ON==OFF + test_metal_ssd_experts）→ clean ABBA ≥2 对（2K/32K × 512 tok）。
- 已知雷（全部来自 §15 实测）：flush 会轮换 CB（capture/begin 前先 `ds4_gpu_commands_active()`）；改码后 ds4/ds4-bench/ds4_test/tests/* 必须全量重编；bit-exact 对照永远用同一次构建的 OFF 腿。
- 基线：main @ 当前 HEAD 新 worktree；分支命名建议 `p4-cardI-graph-capture`。

### 16.6 调研出处

- llama.cpp commit c363256（metal graph capture env）：github.com/ggml-org/llama.cpp/commit/c363256839fdffae184279ad8e5f0f3775c48b77
- NVIDIA blog（CUDA Graphs in llama.cpp，动态标量 staging 方案与收益数）：developer.nvidia.com/blog/optimizing-llama.cpp-ai-inference-with-cuda-graphs
- llama.cpp issue #7456（每 token CPU 活动剖析）；Discussion #23324（MoE 磁盘流式 RFC+负结果）；mlx-lm issue #1438（Apple Silicon 专家流式，读在 decode 关键路径）

## 17. 卡 I / I-1 实施与实测裁决（2026-10-04，ds4-cardI worktree，分支 p4-cardI-graph-capture）

### 17.1 实现（commit 376687e，基线 d3a6ab6，净差 = ds4_metal.m +595/−67，数学零改动）

env `DS4_METAL_ENABLE_V41_MOE_GPU_BINDING` 门控（默认关；要求 streaming + 单卡 + `n_tokens==1` + IQ2 selected slots + 非 full-addr 形态，任一不满足即逐层退回原路径）。`ds4.c`/`ds4_gpu.h`/`metal/*.metal` 一行未动。

1. **路由决策搬到 GPU 侧**：主线程不再回读 6 个 id。routed kernel 走现成 addr-table 入口，ids 直接取 router 自己写的 GPU buffer，权重地址从每层 uint64 表里取（该表本就由 install/evict 维护自洽：装入写槽、驱逐清零、在途槽不可驱逐）。
2. **service 线程自主装填**：新增 `ds4_gpu_v41_moe_gpu_binding_*` 模块（128 深 FIFO ring + 一条线程）。batch CB 上 `encodeSignalEvent(ids_event)` 表示 router 已写完 ids → 线程等事件、读 ids、跑原先主线程跑的簿记（hotness/hotlist/peek/缺失 pread/inflight 标记/回填 6 槽/prune）→ `ready_event` 释放 → CB 上 `encodeWaitForEvent(ready_event)` 之后的 dispatch 才会跑。表**构造性完整**，主线程全程不知道 ids。
3. **地址表门只加在 `..._addr_table_requested()`**，不加在 `..._kernel_requested()`：否则 fallback 层可能在内核准入为真但表未填（V4.1 Q2_K down 形态）时静默读到空槽。绑定模式下本地强制 `use_stream_expert_addr_table = true`。
4. 4 个 encode helper 加尾参 `bool ids_only`：本线程看不见 entries，跳过 entries 校验/inflight 标记/useResource，改挂 slab 标记（见 17.4）。三处 `stream_split_ready`/`compact_addr`/`hit_validator` 加 `!v41_gpu_binding`，OFF 腿逐位不变。
5. 失败语义照抄 TP 门：**先 latch 再无条件释放**（GPU 不得挂起），此后各层走同步路径并打印一次原因（`binding off from layer N: ...`），run 作废而非静默降级。

### 17.2 正确性闸门（最终无诊断打印二进制，全部通过）

| 闸门 | 结果 |
|---|---|
| CLI 贪心 A/B（temp 0，48 tok，同一次构建 ON/OFF，默认 env=masked 族开） | **逐字节一致**（OFF/ON 同 sha `0c4eab7b…`，201 B） |
| ON 腿自复现（masked 族关，3 次独立 run） | 3/3 与 OFF 参考同 sha（此前是 3 次 3 样） |
| `ds4_test --logprob-vectors`（`DS4_TEST_SSD_STREAMING=1` + `DS4_TEST_MODEL`） | ON/OFF 全日志 **0 差异行**；两边同为 2 个存量 `short_code_completion` 断言 |
| dspark `--verify-parity-ssd`（6000 B prompt） | ON/OFF 同为 6 行存量偏差（`prefix 1021 rows 6`），差值 = 0 |
| dspark `--short-prefill-ssd` | ON/OFF 全部 `max logit difference from decode 0` |
| `test_metal_ssd_experts` 四模式 | 全 PASS，0 FAIL |
| ABBA ON 腿诊断 | 4 腿均零 `binding failed/off from layer` ⇒ 512 token 全程真在绑定 |

### 17.3 ABBA 计时裁决：**+3.2% / +2.7%（正向，达标线之上，远未达 +25%）**

`ds4-bench --ssd-streaming --gen-tokens 512 --step-mul 1`，ABBA = OFF/ON/ON/OFF，steady gen t/s：

| 腿 | env | 2K | 32K | 2K TTFT |
|---|---|---:|---:|---:|
| a1 | OFF | 22.34 | 21.62 | 136.63 |
| a2 | ON | 23.20 | 22.37 | 130.45 |
| a3 | ON | 23.13 | 22.42 | 129.91 |
| a4 | OFF | 22.54 | 22.01 | 129.52 |
| 均值 | — | OFF **22.44** / ON **23.17**（+3.2%） | OFF **21.82** / ON **22.40**（+2.7%） | OFF 133.1 / ON 130.2 |

ON 两腿间距仅 0.3%，OFF 两腿 0.9%；**四个比较点 ON 全部高于 OFF 且两组不重叠**（2K：min ON 23.13 > max OFF 22.54）。这不是噪声。对照 §15 的 OFF 基线 23.83/22.54：本轮 OFF 腿整体低 1.4/0.7 t/s（同机同二进制，属批次漂移），故只看组内 ABBA 差。

**收益量级解释（与 §15.4 自洽）**：+0.73~0.58 t/s ≈ 每 token 省 1.1~1.3 ms ≈ 每层 28~33 µs，正好是一次"CB 收尾 + 回读 + 决策 + staging"的主线程指令量。I-1 只砍掉了这 40 个回读边界，没动每层数百次 encode/dispatch/参数调用——那部分正是 I-2/I-3 的对象。

### 17.4 两条必须入档的硬知识（第一版就是靠这两条翻车的）

1. **inflight 标记不能用"当前 batch 序列"，必须用 arm 时抓下来的序列。** 编码线程比 service 线程快一到数个 CB：service 线程 serve 到第 N 层时，环境里的 `g_stream_expert_cache_batch_seq` 指的是**更晚的 CB**，而刚 drain 完的窗口里它等于**已完成的 owned_seq 甚至 0**。前者过保守（无谓），后者等于"这 6 个槽随时可驱逐"——正在飞的 dispatch 还在读它 ⇒ 驱逐/slab 复用与 dispatch 抢同一块内存。症状极具误导性：同一命令三次 run 出三代不同文本、且**一条错误日志都没有**；只有把 protect_seq 显式随请求传递并在为 0 时直接拒绝绑定（干净退回同步路径，不 latch）才收敛。
2. **经地址表取权重的 dispatch 必须让 encoder `useResource` 标到那些 buffer，否则 Metal 不保证内核真看得见它们。** 同步路径一直在标（entries 交给 helper 时逐个 `useResource`），addr-table 路径若只交地址表、不交槽 buffer，读到的就是不保证内容——同样静默、同样跨 run 漂移。绑定模式编码时不知道是哪 6 个槽（ids 还没回来到 GPU 之外），故改为标**整个 slab 池**：本机 streaming 缓存只有 **19 个 slab**，且实测 1880 次 serve 的 6 个槽**全部** slab-backed（零非 slab 条目），代价可接受；这条 marking 一上，绑定路径立刻变成 3/3 逐位一致。
   - 反面结论留给 I-2：**任何 capture 形态都必须把这 6 个 buffer 的资源标记带进图里**，否则同样静默出错。
3. 顺手补一处存量隐患：`take_reusable_batch` 的 "batch reuse" 分支（ds4_metal.m:16065 附近）是全仓四处 `wait_inflight` 中唯一没有 `on_service_thread()` 守卫的；service 线程一旦走到就会等待"可能正在等自己释放"的 CB。I-1 的 `load_selected_missing` 恰好能到达此处（本机实测未触发）。已按另三处的写法补齐（commit 434dd55）：service 线程无 victim 时直接失败上报，绝不自等。补齐后重跑 gate1（逐字节一致）/gate2（0 差异）/gate4（四模式全过）。

### 17.5 数据与产物、I-2 判定

- worktree `/Users/xieyongliang/ds4-cardi`（分支 `p4-cardI-graph-capture`，代码 commit **376687e** + 守卫补丁 **434dd55**，env 默认关；台账行 = main `0a5831c`）。脚本与腿全部在 worktree 内 `.cardi-ab/`（未入库）：闸门 `run_gates2.sh`/`run_gates3_abba.sh`/`run_recheck.sh`，ABBA `ab2_{a1..a4}_{2k,32k}.csv`，正确性 `g1_*/g2b_*/g3_*/g4*`，复现性 `r2_*/r3_*`。
- **I-2 判定：继续（go）**。理由：① 前置障碍（GPU 侧路由决策 + service 线程装填 + 事件门控）已落地且**逐位无损**，capture 岛扩到"单卡 + 流式"最大的动态性来源已被固定成"图外一条线程 + 两个事件"；② I-1 自身 +3% 独立成立，即使 I-2 失败也不亏；③ 17.4.2 给出 I-2 的硬性准入条件（图内必须带权重 buffer 标记，或改为把 6 个地址写进可捕获的间接 buffer）。
- I-2 起手必读：`ds41_decode_island` + `ds4_gpu_decode_graph_begin/end`（ds4.c ~42388，现准入 `tp_world==2 && !streaming`）、§16.4 的动态标量 staging 抄法、以及 §15 教训（flush 会轮换 CB，capture/begin 前先 `ds4_gpu_commands_active()`）。

### 17.6 上游 PR（同日）：[antirez/ds4#1178](https://github.com/antirez/ds4/pull/1178)

按 #1177（PF-6）的做法另起 PR 分支：从 **upstream `origin/main` `0aaea5a`** 切 `cardi-upstream`，只放 I-1 两个 commit（`5805843` 主改动 + `4e48bb5` 服务线程守卫），**零 cherry-pick 依赖**——流式专家缓存与槽/inflight/驱逐规则（`9ba160a` "Implement SSD streaming"、`005afed`）、地址表内核、守卫要用的 `note_service_thread`（`519c4d8`）全部已在主仓；cherry-pick 只有一处插入点冲突（HEAD 侧是本地专有的 `ds4_gpu_m5_*_shape_args_match` 两个形状函数，按上游侧丢弃即可，上游无人引用），PR 树只 `ds4_metal.m` +599/−68。上线前把代码注释里的内部编号（卡 I/§16.5）清了，commit message 与 PR 正文按 #1177 的写法带两组实测。

PR 基线（= 上游 main + 这两个 commit）单独跑了一遍闸门与 ABBA（同一二进制两腿，V4.1 Flash Q2，`--ssd-streaming`，512 tok）：

| 上下文 | OFF 两腿 | OFF 均值 | ON 两腿 | ON 均值 | delta |
|---|---|---:|---|---:|---:|
| 2K | 18.23 / 17.99 | 18.11 | 18.63 / 18.32 | 18.48 | **+2.0%** |
| 32K | 17.13 / 16.93 | 17.03 | 17.45 / 17.62 | 17.54 | **+3.0%** |

两上下文的 ON 两腿都高于 OFF 两腿；TTFT 在噪声内不变。正确性：贪心 A/B 逐字节一致（同 sha，两边 `0c…`→上游基线 `176a045d…`）、`ds4_test --logprob-vectors`（streaming）ON/OFF 全日志 0 差异（两边同为存量 2 个 `short_code_completion` 断言）、`test_metal_ssd_experts` 四模式全过。绝对值比 fork 树低（18.1 vs 22.4 t/s）是因为上游 main 没有 fork 的 decode 融合线（#1041/#1042/#1043/#1090/#1120 + 本地流式工作），PR 正文里两组数字分开写清、并点名 fork 树所带的那些 cherry-pick。
