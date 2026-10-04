# 迭代开发计划：V4.1 Flash（SSD 流式）× GLM-5.3 Flash（驻留）——2026 Q4

> 依据：[V41_SSD_STREAMING_ROADMAP.md](V41_SSD_STREAMING_ROADMAP.md)（R 系列）、[GLM53_SPEEDUP_ROADMAP.md](GLM53_SPEEDUP_ROADMAP.md)（G 系列）、
> [V41_M5MAX_BASELINE_SPEED_PLAN.md](V41_M5MAX_BASELINE_SPEED_PLAN.md) §16–17（卡 I 立项）、[LOCAL_INVENTORY.md](LOCAL_INVENTORY.md)。
> 优先级裁决（已定）：**潜力主战场 = V4.1 流式；但第一优先工作是两边共用的 Metal graph 地基，按 V4.1 立项、GLM 驻留域首验收。**
> 本文是执行文档：里程碑 M0–M6，每个任务带落点/预算/闸门/判定线/回滚/产出物。**本文档本身零模型加载**；
> 所有"加载类"步骤已集中标注，统一走 §2 的执行窗口规程（内存预检 + 停产品 + 一次构建一次跑完）。

---

## 1. 总 KPI（冻结基线 → 各里程碑出口线）

基线冻结于 2026-10 各锚点文档；**每个里程碑出口必须在同一张表上落一行新数**，未达标要写原因，不允许换口径。

| KPI | 冻结基线 | M0 复核（10-05 窗） | M1 出口 | M2 出口 | M3 出口 | M5 出口 | 物理顶/备注 |
|---|---|---|---|---|---|---|---|
| V4.1 decode steady 2K / 32K | 22.44 / 21.82 t/s（ABBA OFF 均值） | **23.35 / 22.16**（OFF×2，成立） | ≥23.5（I-1 默认化） | **≥27.4 / 27.3（+25%）** | 维持不回退 | ≥30（若 I-3） | DRAM roofline ≈45 |
| V4.1 首 token（12K ctx） | 127–133 ms | 未测（不在 M0-4 腿单） | 不回退 | ≤120 | — | — | |
| V4.1 831 行 append（冷） | 6.1 s | **6.24 s**（成立；热后 4.5–4.6 s） | 不回退 | 不回退 | **≤5.0 s** | — | 命中率主导 |
| V4.1 100–500 行段（生产冷域） | 100 行 ≈30.2 t/s | 未测（归属 R2 线） | — | — | **≥+15%** | — | R2a/R2c |
| V4.1 [2,32) 行 append | ≈21–24 t/s（token-major） | **23.5–23.8**（step4/8；零批量增益复现，SPEC_ROWS 双臂 logits 逐字节等） | — | — | **≥45 t/s**（E1） | — | 先例 223 行 81.6 |
| GLM decode @12k | 32.0 t/s（10-03 复测） | **33.30**（G1/G2 中位，成立）；短 ctx 2K=34.75 vs 旧法 38–39 存疑不判负（见 notes） | **≥35（+10%）** | — | — | 36–38 + MTP 另档 | 带宽顶 ≈39–40 |
| GLM prefill @41k | 480–486 t/s | 未测（窗上界 12k，PF 域避开） | — | — | — | +4~8%（G2 系列后） | GPU 96% busy |
| GLM MTP 净收益 @17k | **−15%**（默认关） | **−19.8%** @1.3k（eff 25.2 vs plain 31.4，accept 60.6%，G4a 首账） | — | — | — | **≥0（翻正判定）** | 需 ≥80% 接受率 & cycle <1.35× |

> 逐位契约是高于一切速度指标的门槛：**任何指标改善若伴随 bit-exact 门破防且未经书面裁定，视为该里程碑未完成。**

---

## 2. 执行窗口规程（所有加载类步骤的统一前置，逐条打勾）

1. **预检（进窗前）**：`pgrep -fl 'ds4|ds4-server|ds4-bench|mlxCamlRunner|python.*omlx'` 无残留（有 `ds4-bench --child` 即拒绝开跑，SR-2 教训）；
   `sysctl vm.swapusage` < 3 GB 且稳定；`vm_stat` 可用+purgeable ≥ 目标模型驻留峰值的 1.2×（V4.1 流式 ≈70 GiB cache+wired 口径，GLM ≈100 GiB）。
2. **产品让位**：测量窗口必须停 `ds4-server`（生产 8055）与一切 LLM runtime；单实例 flock 生效。
3. **一次构建、一次跑完**：同一个改动臂的所有加载类测试（功能门 → bit-exact → ABBA → profile）**合并进同一窗口、同一构建产物**；
   禁止"跑 5 分钟→失败→改码→重跑"循环（先一次性 preflight：`test -r/-x`、脚本 `bash -n`、fixture 存在性）。
4. **ABBA 协议**：A/B 交替 ≥2 对（差别预算 <±15% 时 4 对起步，PF-2 教训）；臂 = 同一二进制 + 每腿前缀 env（不 export/unset）；
   产物 CSV 入 `speed-bench/<里程碑>/<任务id>/`，含命令、时间戳、量化、cold/warm、失败项。
5. **bit-exact 口径**：永远与同一次构建的 OFF 腿对照；bit-exact 改动不做 TTFT 措辞承诺（#1090 教训）。
6. **中止线**：swap > 6 GB、出现 memory pressure warning、或任一 ABBA 腿绝对值掉出历史区间 ±15%（共享机干扰判据）→ 中止并记录。
7. 已知计时坑照抄：功率状态冷腿补偿、页缓存爬坡（每个新缓存机制先 `--warm-weights`/预热再测）、GLM `--mtp 0` 解析陷阱（一律裸 `--mtp`）。

**执行窗口预算总表**（估算，供排期）：M0 窗 ×1（短）；M1 窗 ×3–4；M2 窗 ×4–5；M3 窗 ×3；M4 窗 ×3–4；M5 窗 ×2；M6 窗 ×1–2。
窗口外（纯编码/静态审计/离线分析）不受此限。

---

## 3. 里程碑与任务分解

任务 id 沿用路线编号；`[W]` = 需要执行窗口，`[O]` = 窗口外可做。
每个任务的 **DoD**（完成定义）统一为：代码提交（env 默认关或有裁定）+ bit-exact 门结果入档 + ABBA CSV 入档 +
LOCAL_INVENTORY 对应行更新 + KPI 表回填。**缺一不算完成。**

### M0 · 还账与账本冻结（1 短窗 + 杂活；预计 2–3 天）

| id | 任务 | 落点/做法 | 预算 | 判定/产出 |
|---|---|---|---|---|
| M0-1 [O] | `tests/test_glm53_router_shared.c` 小补丁提交 | 主树未提交 5 行（附三已证本机转绿） | 10 min | 入 main |
| M0-2 [O→W] | 主树 WIP 三项关账：`STREAM_GATHER_SPEC_ROWS`、`GLM53_ROUTED_MPP_PACKED`（已判负臂定去留：按 PF-2 裁决**删臂**）、`--mtp-timing`（=G4a，**留，补完**） | 补完 mtp-timing 结算打印的 GLM/plain/verify 三段口径；其余两 env 各跑一臂 A/B 或按既有裁决直接删 | 半天 + 并入 M0-4 窗 | 主树 `git status` 干净；每项一行台账 |
| M0-3 [O] | p42-attn-glue 定夺 | 位等价性复核（与现默认位的 diff 语义），**裁定=合入取位、弃 worktree**（预期终态，因为实测 ≈中性） | 半天 | 合入 commit 或删除记录，台账 §D 更新 |
| M0-4 [W] | 基线复核窗 | 一次窗口跑齐：V4.1 ABBA（2K/32K×512）、GLM ds4-bench（短+12k）、V4.1 831/1524 行 append dump+profile、GLM MTP-timing 冒烟（裸 --mtp） | 一窗（~3 h） | KPI 表"复核列"落数；`speed-bench/M0/` 全量归档；发现漂移先归因再开 M1 |
| M0-5 [O] | 上游协同 | 检查 PR #1177/#1178 review 意见；`git fetch origin` + `git cherry -v` 核对吸收状态，刷新 LOCAL_INVENTORY SHA | 1 h | 台账更新 |
| M0-6 [O] | 工具入库 | `speed-bench/v41_m5max_streaming_ab.sh` 等台架脚本 + `v41_m5max_ab/` 数据入 main（台账 E 节遗留） | 1 h | commit |

**M0 出口门**：主树干净；KPI 表有新鲜复核列；无未登记在途项。

### M1 · Metal graph 地基 + GLM 驻留域首验收（主线一期；预计 2–3 周）

> 目标：把 `ds4_gpu_decode_graphs_supported()/begin()/end()/abort()` 的 **Metal 实现**（MTLGraph capture + keyed 图缓存）
> 做出来并用 GLM FFN-tail（现成接线，ds4.c:58571+）验收。**这一步不做任何路由/槽位动态性**——那是 M2 的事。

| id | 任务 | 落点/做法 | 预算 | 闸门与判定 |
|---|---|---|---|---|
| M1-1 [O] | Metal 后端地基 | `ds4_metal.m` 新增：`MTLCommandBuffer` capture scope（`captureScopeWithCommandBuffer:`）、key = `ds4_decode_graph_key`（il/island/variant + 关键 buffer 身份指针）的图缓存（LRU，容量常数起步 64）、`begin` 命中即"安排 replay 返回 1"、miss 返回 0 并开 capture、`end` 提交 `recompile`、`abort` 作废；动态标量走 per-key `param buffer`（固定地址、图内读）；总开关 env `DS4_METAL_DECODE_GRAPHS`（默认开与否由 M1-3 定，先默认关） | 3–4 天编码 | **不加载模型的单元验收**：`tests/test_metal_graph_capture.c`（新）——dummy kernels 的 capture/replay/参数更新/abort 重录/容量淘汰全过；三铁律自查（flush 轮换 CB：capture 前先 `ds4_gpu_commands_active()`） |
| M1-2 [O] | GLM 侧核对 | 现有 island（attention→FFN 分段、key 字段含 hc 指针身份）在 keyed 语义下的正确性推演；补 engage/disengage 日志锚（`decode graph <key>: captured/replayed n=…`） | 0.5 天 | 代码评审 + 日志锚就位 |
| M1-3 [W] | GLM FFN-tail 首验收 | 开 env 跑全套：GLM53 100-case@4096、长 fixture×2、frontier logits identical、100-ABBA N=6 采样数对拍 | 一窗 | **bit-exact 全绿 + replay 命中计数>0**；有任何 DRIFT→按 §17.4 三雷排查（首嫌 param 生命周期/`useResource`） |
| M1-4 [W] | GLM FFN-tail 速度门 | ABBA ×2（短 ctx + 12k，`ds4-bench`）；同时采 `DS4_METAL_*TIMELINE` 看空隙是否收窄 | 并入 M1-3 窗 | 中性以上（±2% 内）即收下地基；收益计 M1-5 |
| M1-5 [O→W] | GLM 扩整层岛 | 把 KDA/attention/router/LM-head 段逐段包岛（keyed），从 FFN-tail 扩到每层 2–3 岛（piecewise 形态，vLLM 同构）；每加一岛单独 ABBA | 编码 2–3 天 + 窗 ×2（分两腿） | **GLM decode ≥35 t/s @12k（+10%）** 为 M1 出口；<+5% 时先归因（空隙残余 vs host 编码），不许直接续加岛 |
| M1-6 [W] | 审计旁刀（搭车） | G2f 之"≥2 GiB offset 单测"在 GLM 大 buffer 上实测执行 | 并入任一窗 | 全绿或按 S11 先例修复入账 |

**M1 出口门**：GLM decode +≥10%、bit-exact 全绿、图缓存有淘汰/失效路径的测试覆盖。
**止损线**：若 MTLGraph 与 `MTLSharedEvent`/异步 flush 的相互作用无法稳定（OOM 类或 capture 失败率>0），退到"非 MTLGraph 的大 CB 合并编码"形态并如实降级记录（收益预期改 ±5%），M2 决策点重开。

### M2 · V4.1 流式 decode 岛（卡 I-2 本体；预计 2–3 周，M1 后）

| id | 任务 | 落点/做法 | 预算 | 闸门与判定 |
|---|---|---|---|---|
| M2-1 [O] | I-1 合入主线 | `p4-cardI-graph-capture`（376687e+434dd55）rebase 并默认翻 ON（fork 侧；不等上游 #1178），处理与 M1 地基的树合并 | 0.5 天 | 既有 ABBA +3.2/+2.7 数字引用；翻 ON 后 CLI 贪心逐字节复验 |
| M2-2 [O] | 流式准入改造 | `ds41_graph_decode_layer` 门（ds4.c:41561 一带）：`tp_world==2 && !streaming` → 增 `tp_world==1 && streaming && v41_gpu_binding`；每层 3 岛（before-attn / attn-out / after-attn+MoE），ids 与 addr 表留 GPU，service 线程 `ids_event/ready_event` 桥**留在岛外**（岛=纯计算段）；Engram step-reader、miss-upload 天然图外不动 | 3–4 天编码 | 编译期不变量评审；失败语义照 I-1：**先 latch 再无条件释放**，run 作废不静默降级 |
| M2-3 [O] | 动态标量入 param buffer | token id/position/engram 行号等 kernel-arg 标量改为图内读 staging（llama.cpp CUDA-graphs 抄法，BASELINE §17.4-1） | 1–2 天 | 逐 token 更新路径单测（先不加载模型，测试 harness 直接改 buffer 校验读回） |
| M2-4 [W] | bit-exact 全集 | 一次窗：CLI 贪心 ON/OFF 逐字节；`DS4_TEST_SSD_STREAMING=1 ds4_test --logprob-vectors`；`--verify-parity-ssd` + `--short-prefill-ssd`（6000B prompt）；`test_metal_ssd_experts` 4/4；slug/223 行 dump | 一窗 | **DRIFT=0**；任一破防→按准入三雷排查，两次未愈即停线归因（不许连改连跑） |
| M2-5 [W] | 速度门 | ABBA ≥2 对 ×{2K,32K}；顺带 `--power 100`；埋 `gpu_busy` 对照（预期 30%→55%+） | 并入 M2-4 窗 | **steady ≥+25%**（KPI 表 M2 列）；+10~25% 之间 = 达标部分收账并写残余归因；<+10% = 停，出新瓶颈画像后再裁 |
| M2-6 [O→W] | argmax 进 GPU | logits 头 argmax 内核 + 只回读 (id, logprob[, top-k])；sampling 语义逐位不变（贪心路径先行，采样温度路径若归约序敏感则只做贪心） | 1 天 + 搭 M2-5 窗 | 与 host argmax 逐字节一致；此后每 token 最后一次大回读消失 |
| M2-7 [W] | I-2 收账归因 | §14.6 式再归因：wall/gpu_busy/gap/missing_wait 四件套 + 新 host profile；产出"墙钟新构成"一页纸 | 并入窗 | **这是 R3（DSpark 复议）与 R6（deferral）的开工令**，无此页不开 R3/R6 |
| M2-8 [O?] | I-3 全 token 一图 | 仅当 M2-5 ≥+25% 且 M2-7 显示逐层岛边界仍有可测空隙；flush/commit 语义退化为一次 | 另立执行文档 | 目标 33 t/s；未达标即止于 piecewise |

**M2 出口门**：V4.1 steady ≥27.4 t/s 或有带归因的诚实中评；M2-7 报告入库；KPI 表回填。

### M3 · V4.1 prefill 命中率包 + 小项（与 M1/M2 穿插；预计 2 周）

| id | 任务 | 落点/做法 | 预算 | 闸门与判定 |
|---|---|---|---|---|
| M3-1 [W] | R2a-1 离线标定 | 用录制 session 回放器（e4488c1 工具链）+ 生产日志统计每层跨轮共现 top-K（K=cache 容量的 60/75/85% 三档）；产出 `ds4_streaming_hotlist_v2.inc` 生成脚本（`gguf-tools/` 或 `tools/`） | 编码 1–2 天 [O] + 回放一轮 [W]（工具运行=加载类） | 标定报告：每层候选集对次轮 gather 的纸面命中增益（预期 ≥25pp） |
| M3-2 [O] | R2a-2 引擎保护位 + 空闲预热 | cache 槽加 `protected` 优先级（LRU 先逐非保护，**总容量不变**）；service 线程空闲窗按 hotlist 预热（复用既有 pread 池）；env `DS4_V41_HOTLIST_PROTECT=0/1`（默认关）+ `DS4_V41_IDLE_PREWARM=0/1` | 2 天 | 单测：淘汰次序真值表（窗口外可测，纯簿记） |
| M3-3 [W] | R2a-3 验收 | bit-exact：`--moe-bind-parity` 27/27 + 223/1524 dump IDENTICAL + 小 cache 压力腿（P0c 配方）；速度：**多轮 append 模拟器**（固定内容 6 轮工具型流量，冷启动起）ABBA，采每轮 prefill 墙钟 + 每层 gather hit/miss 表 | 一窗 | **[100,512) 段 ≥+15%**；命中增益与纸面差 >10pp 先查老化参数 |
| M3-4 [W] | R2c 容量复扫 | auto/8000/9000/10000 × {831,1524} + decode 2K/32K 不回退约束；盯 swap（≥10000 臂预检不过不跑） | 一窗（长） | 一次定夺写死进文档；改默认要 KPI 表两域同时不劣 |
| M3-5 [O→W] | E1 低行批量 | 前置 E0.1：microtest 扩 `--moe-bind-parity` 到 rows 2–31（addr vs whole-map，layers {0,20,39}）[W]；过则 admission 32→2（ds4_metal.m gather 准入 + CPU float-ptr parity 同步）[O]；契约核对表（P0c §14.3 列表重跑）[W] | 编码 1 天 + 窗 ×1 | 新红灯（[2,32) 契约变化）→ 按 §13.1 纪律**先停等人裁**，默认落地 = scalar 家族路径（rows>8 若走 `*_mpp` 族则数值上移一档，写明） |
| M3-6 [O→W] | P2-CED suffix-aware spans | `decoder_suffix` 生效时 layer 20–39 的 pagein spans 用收缩行数决定 gather miss 集（ds4.c sweep prepare × `metal_graph_stream_prefill_layer_pagein_start`）；env `DS4_V41_CED_SPARSIFY` 默认关 | 1 天 + 搭 M3-4 窗 | ≥4096 wide prefill 墙钟 −15~25% 且逐位不变（spans 不改变被算字节的语义——论证入档） |
| M3-7 [O] | E0.3 契约卫生 | streaming 版 `check_short_prefill` 红灯入 100-case 契约表登记 | 1 h | 入档 |

**M3 出口门**：831 行冷 append ≤5.0 s 或带解释的最近值；100–500 段 +≥15%；E1/CED 各自 DoD 齐。

### M4 · GLM kernel 波（纯穿插，各任务独立可单飞；预计与 M2/M3 并行推进）

| id | 任务 | 预算 | 判定线（出处：GLM 路线图 G2/G3） |
|---|---|---|---|
| M4-1 | **G2a KDA blocked 递推**（供体 omlx #3984 形态；`metal/glm53_kda.metal` recurrence 重写 + `_prefill_prepare/_output` conv4+SiLU+L2 融合小项二期） | 编码 1–1.5 天 + 窗 ×1 | 独有闸门：conv/recurrent 双态跨 chunk 一致 + `prefill(T)==prefill(T−k)+decode(k)` 对拍；**prefill +2.5~4%（全 ctx）**；不达标只收无回归子集 |
| M4-2 | G2c dense `direct_rhs` tile 扫描（n128→n256/2m，`mul_mm_q8_0/q4_K` 族） | 窗 ×1（扫参型） | 定夺入档；预期 −2~+5% |
| M4-3 | **G2f M5 tensor-API ≥2 GiB 寻址审计**（枚举 direct_rhs/mpp/packed/mla/indexer 内核的 slice 地址构造，对照 upstream #28748 模式；每族一条 ≥2 GiB offset 单测） | 1 天 [O] + 单测执行搭任一窗 [W] | 全清白也入档；发现问题→S11 先例修复+回归向量 |
| M4-4 | G3a q8_0 matvec 特化一次（`K=4096 rows=1`，#1120 sum6 2-simdgroup 思路；**限时一个窗口，+30% 目标、<20% 即停**） | 编码 1 天 + 窗 ×1 | decode ≤+1.3% 后按">40% roofline 才开刀"纪律封刀 |
| M4-5 | G3b KDA decode 融合核（conv+状态+门+norm 进 decode 单核；vLLM K3 同款；顺带产出 G4b 的种子融合件） | 1–1.5 天 + 窗 ×1 | decode +2~3%；KDA 状态一致性测试（已有）必绿 |
| M4-6 | G2b MoE 主件 tile 形 vs 真实分布扫描（30/64/128 档实例 + 判别日志锚） | 窗 ×1 | 限时定夺，负即收（PF-2 门槛照用） |
| M4-7 | PF-5 profile 轮（sweep 循环空洞率，30–60 min） | 搭任一 GLM 窗 | <1% 永久搁置 PF-5 |

**M4 无独立出口门**——各任务独立收账；两项以上负结果不必强凑。

### M5 · 投机复议（严格依赖 M1/M2 完成；预计 1–2 周）

| id | 任务 | 预算 | 判定 |
|---|---|---|---|
| M5-1 | **G4b GLM 起草头砍半**：GPU argmax + 单 int 回读；draft 绑专属 fused-FFN/router 内核；第 46 层 selected/ids/bias 常驻静态数组（驻留 only 门控）；种子融合接 M4-5 产出 | 编码 2–3 天 | 起草 cycle 16.2→~8–9 ms（用 G4a `--mtp-timing` 账本直接读） |
| M5-2 | G4c 经济复测（G1 已在 + G4b 完成后）：`--mtp-timing` + 金样 manifest + 工作流量影子；**翻正线 = 接受率 ≥80% 且 cycle <1.35×** | 窗 ×1 | 翻正→默认策略申请（工作流量 ctx<~8k 条件默认开 + 接受率采样）；再关账→DC-1 补"适用域"字样继续封存。两种结果都写台账 |
| M5-3 | G4d ngram 自起草原型（`ngram-mod` 同型，跨请求 hash 池；只付 verify 行） | 编码 2–3 天 [O] + 影子测 1 窗 | 高重复流量 ≥+8% 才转正；负即弃 |
| M5-4 | R3 V4.1 DSpark 桌面复议（用 M2-7 新归因重推盈亏平衡；不重开管线除非平衡点 ≥ accept 上限） | 半天 | 预期维持封存，补"适用域"注记 |
| M5-5 | （条件，M5-2 翻正才开）DC-1.1 链式起草编码 + DSpark 资产验证立项 | 另立 | — |

### M6 · 条件里程碑（不进本季度承诺；触发条件写死）

- **G5 KV Q8 门 + ctx 档位表**：触发 = 有人申请 ≥96k 生产域。窗 ×1；产物 = "FP32 ≤~96k / Q8 ≤~220k"档位表入文档，PF-3/PF-4 立项申请必须引用。
- **R5 expert-contiguous 布局微测工具**：触发 = prefill 冷域仍有 >10% 未解释缺口。先做离线转换+pread 微测（编码 [O]、执行 [W]），**≥+8% 才立项改 loader**。
- **PF-3 indexer NAX / PF-4 wide 8192**：触发 = G5 完成且出现 ≥64k 域。
- **R6 deferral / down-proj overlap**：触发 = M2-7 归因显示 expert-ready 事件成串空泡。
- **PF-2 tile 形补扫、I-3 全图**：各自上游门的延伸，随主线走。

---

## 4. 依赖图与排期形态

```
M0(还账/冻结) ─┬─→ M1(地基+GLM首验收) ─→ M2(V4.1流式岛) ─→ M2-7(归因) ─┬─→ M5(投机复议)
               │            ↑                     └──→ (I-3 条件)      └─→ R6(条件)
               ├─→ M3(prefill包，全程可穿插，仅 M3-3/4/5/6 占窗)
               └─→ M4(GLM kernel 波，全程穿插)
M6 = 条件门控，任何里程碑后按需触发
```

- **人手形态**：主会话串行执行 + 每刻最多一个后台分析/编码子任务（provider 允许时）；所有判定在主会话。
- **窗口经济**：加载类工作全部合并进窗——例如 M3-1 回放标定和 M3-3 验收共用一个 GLM/V4.1 驻留状态；
  M4-1/4-5 共用一个 GLM 窗；M2 全部门 + M2-6 + M2-7 共用一个 V4.1 窗（~4–5 h，中途按 §2-6 中止线守护）。
- **上游节奏**：#1177/#1178 的 review 意见每次窗口前检查一次；被吸收则按 LOCAL_INVENTORY 纪律删行重核依赖。

## 5. 风险登记与预案

| 风险 | 命中里程碑 | 预案 |
|---|---|---|
| MTLGraph 与 async flush/MTLSharedEvent 相互作用不稳 | M1/M2 | 已写止损线（M1 尾）：退大-CB 合并编码并降级预期；piecwise 是保底交付 |
| 图缓存 keyed 于指针/形状，ctx 或 session 切换导致重录风暴 | M1/M2 | 图缓存命中率日志锚 + LRU 上限 + key 含 ctx 桶；重录率 >1% 判不健康 |
| E1 [2,32) 契约红灯（batch 家族数值上移） | M3-5 | §13.1 纪律：新红灯先停等人裁；默认落 scalar 家族路径 |
| hotlist v2 在生产流量漂移（标定过拟合录制集） | M3-3 | 保护位只是 LRU 优先级不改数学；速度不达标只回滚 env，不伤正确性 |
| MTP 翻正后质量敏感（gold zone 80% 边缘） | M5-2 | 默认策略限定条件（短 ctx + 采样率达标才开），保留一键关；账本常驻 |
| 执行窗口被生产需求挤占 | 全部 | 顺序弹性已给：M3/M4 不依赖主线可先行；M2 编码可全部提前完成等窗 |
| 共享机测量噪声（±30%） | 全部窗 | 停产品 + 成对设计 + ≥2σ 判读；测量窗一律申请独占 |

## 6. 台账与产出物规约（每任务收尾清单）

1. commit：env 开关名进 `ds4_help.c`（若是用户可见旋钮）；cherry-pick 相关按 AGENTS 惯例带 trailer。
2. LOCAL_INVENTORY：B 节新行（本地实现）或 D 节裁决行（负结果**必须**入，防复踩）。
3. `speed-bench/<M>/<id>/`：ABBA CSV + profile/日志 + `notes.md`（命令原文、臂定义、失败项）。
4. KPI 表（§1）回填 + 路线图（V41/GLM53）对应小节加"→ 已实施/已判负 @commit"回链。
5. fork 同步：每个里程碑出口把 main 推 fork（AGENTS 铁律：不许滞后整批）。

## 7. 验收日历（建议节奏，窗口按机器实况排）

| 周 | 主线 | 穿插 | 窗 |
|---|---|---|---|
| W1 | M0 全清 | — | W×1 |
| W2 | M1-1/1-2（地基编码） | M3-1 标定脚本、M4-3 审计 [O] | — |
| W3 | M1-3/1-4（GLM 首验收） | M3-2（保护位编码） | W×1–2 |
| W4 | M1-5（整层岛）+ 出口门 | M4-1 KDA blocked | W×2 |
| W5 | M2-1/2-2/2-3（流式岛编码） | M3-5 E0.1 微测 | W×1 |
| W6 | M2-4/2-5/2-6（全套门+速度） | — | W×1（大） |
| W7 | M2-7 归因 + M2 出口评审 | M3-3（R2a 验收） | W×1–2 |
| W8 | （I-3 决策）M5-1 起草砍半编码 | M4-4/4-5、M3-4/3-6 | W×1 |
| W9 | M5-2/5-3（经济复测、ngram 影子） | M3 收尾出口评审 | W×1–2 |
| W10+ | 里程碑复盘 → M6 按需 | 上游 PR 吸收清账 | 按需 |

> 与既定铁律的最后一次重申：本计划任何一格真正执行前，都先过 §2 预检；当前机器状态（有其它 LLM 驻留/内存紧张）时，
> 只推进 [O] 列。**计划的完成定义里没有"预期达标"四个字——只有 KPI 表上的实测数与台账里的诚实裁决。**
