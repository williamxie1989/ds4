# 本地仓库 vs 上游 antirez/ds4：提交与在途工作清单

> 生成日期：2026-10-03（2026-10-04 全仓代码对账更新）。基准：`main` = `4c95017`，比 `origin/main`（上游）领先 **58 个提交**（34 cherry-pick + 1 merge + 22 本地实现 + 1 台账提交）；`fork/main` 已同步至 `4c95017`——注意其中含 HANDOFF/PLAN/本台账等私有文档，fork 可见性需自查。
>
> 依据：cherry-pick trailer、patch-id 对拍、`pr-*`/`up/pr/*` 本地分支、合并提交 `0743c7f` 正文、以及仓库内 9/25 之后的移交/规划文档。
>
> **本文件是 fork 私有台账**：只跟踪在本地/fork 的 main 上，按 PR 纪律（文末）不会进入任何提给上游的 PR。维护规则见 [AGENTS.md](AGENTS.md) 的 "Upstream picks & LOCAL_INVENTORY.md maintenance" 一节。

---

## A. Cherry-pick 清单（按上游 PR 分组）

### A1. 经合并提交 `0743c7f` 并入（09-30 "Merge upstream perf PRs for GLM-5.3-Flash and DeepSeek"，M5 Max 已验证；GLM 侧 19 片）

| 上游 PR | 主题 | 落到 main 的提交 | 采纳度 |
|---|---|---|---|
| #1090 (trueimage) | exact GLM Flash kernels + V4.1 流水线 decode + server 精确前缀 | `e37c7ad`, `9309a29`, `dbee467` | **3/6**（未取 MXFP4 保护、gguf-tools 审计、docs 三片；两片冲突手工解） |
| #1093 (Michael Kuebbeler) | GLM live checkpoint 跨 tool 轮保活（分析见 #1092） | `423a37f` | 1/1 |
| #1120 (Emilian Bold) | M5 resident MoE/Q2 sum6/KDA Q8 decode 形状特化 | `2db1152`, `fa58254`, `17d8717`, `7192fe4`, `23e722c` | 5/5 |
| #1122 (Adam Margulies) | Metal GLM 内存 guard 按 engine open 时 wired 内存封顶 | `818752e` | 1/1 |
| #1126 (Adam Margulies) | keep-alive 线程启动前先解析 keep-alive pipeline | `646a9bf` | 1/1 |
| #1135 (Sebastian Christiansen) | Metal visual attention 压缩 key 因果性修复 | `7bbad7a` | 1/1 |
| #1147 (Emilian Bold) | validated selection prefixes 找回 Metal GLM prefill 速度 | `7fcabea` | 1/1 |
| #1149 (Will) | Qwen3.8 QSA prefill attention 上 tensor units | `245acef`, `d22546a` | 2/2 |
| #1150 (Will) | hyper-connection 残差写延迟进下一个 stream norm | `9744a34` | 1/1 |
| #1127 (Emilian Bold) | callback prefill layer overlap | **实测否决、未入树**：GLM 短 prompt prefill −19%（分支 `pr-1127` 保留） | 0/1 |

### A2. 单独 cherry-pick 到 main 第一父链

| 上游 PR | 主题 | 提交 | 落点 |
|---|---|---|---|
| #1006 (Simone Scarduzio) | tool-call 静默期 SSE keepalive 注释（防客户端 300s 超时重灌） | `428e191` | merge 之前的基座 |
| #1000 (ayuan1129) | rewind 泄漏 stop-boundary draft token；terminal rewind 后补文本续写 | `8963eae`, `e66f811` | merge 后（09-30 序列） |
| #1003 (Emilian Bold) | DSpark 快照复用做即时 rewind + 防 stale handle | `9c4cf2c`, `bca9985` | 同上（分支 `fix/spec-rewind-rebuild`） |
| #1041 (Adrian Galilea) | 单盒 decode 命令排队、逐层 commit 不等待 | `a7ee2f4` | **1/2**（未取 `fdbf7f2` "queue through the logits tail"） |
| #1042 (Adrian Galilea) | decode MoE/HC/路由 glue 融合族（19→6 dispatch/层）+ 回退开关 | `e2e3563`, `ef45eec`, `0319311`, `1f12bb2`, `a226fd4`, `f8d0e93` | **6/7**（未取 `f61a83d` attention glue——后在 p42-attn-glue worktree 单独实测，见 D） |
| #1043 (Adrian Galilea) | V4.1 Flash routed 形状默认 Q4_K group-6 专家表 | `4ec5fc4` | V4.1 批次 |
| #1061 (RTSAI Coder) | decode 形状的 V4.1 indexer 内核（长上下文） | `8c5f178` | V4.1 批次 |
| #1089 (RTSAI Coder) | V4.1 live session 直接 rewind 不重建 | `ce97350` | V4.1 批次 |
| #985 (Rick Ratmansky) | 归一化 tool replay 跨轮保住 KV cache | `b8e4474` | 10-02 前缀复用批次 |
| #727 (LEFBE) | transient metadata block 剥离后保住 live KV（issue #364，#378 的返工） | `bfbf557`, `ba80b4b` | 同上 |

## B. 本地实现提交（24 个，含从自己 feat 分支重新落地）

| 提交 | 主题 | 性质 |
|---|---|---|
| `0743c7f` | merge：18 个上游 perf PR 集成 + M5 Max 验证记录 | 集成 |
| `201bdc6` | `DS4_METAL_FORCE_GLM53_TUNING=1` 供 A/B | 本地（#1122 邻接） |
| `96ac715` `ede6d45` `9bb058e` | glm-guard：wired baseline 每进程采一次 + refuse 前重采 | 本地，**修复 #1122** |
| `ff5b4e0` | GLM tool-call 参数按 request schema 恢复类型 | 本地自研；**与未取的上游 #1152 同一问题域**（patch-id 不同，合并 #1152 前需人工对拍） |
| `1e09244` `003e007` | GLM-5.3 sparse MLA prefill 上 NAX tensor units + 回退开关接入 | 自 feat 分支重放（`feat/glm53-sparse-mla-tensor-units`） |
| `9bc0bd7` | GLM53 tuning 家族放开到 NAX（M4/M5）域 | 自 feat 分支重放（`feat/glm53-tuning-gate-nax-domain`） |
| `2ce4e40` | DSV4 sparse indexed attention prefill 上 NAX（M5 默认开） | 本地；**是上游 #758（M5 Max indexed prefill）的自研替代** |
| `030b342` | 调和 #1041/#1042 cherry-pick 与 fork decode graph | 本地修复（修 `a226fd4` 条件断链） |
| `975238a` `977a8eb` | P0c：streaming gather 镜像进 mm-id 内核族、bit-exact、默认开 + P0d 地板 256→32 | 本地（HANDOFF §14） |
| `20e6f7d` `cb6cdd9` `3f74a1b` `b5be850` `c9c0cdb` `2dcc1c8` | P0f/P0f2/P0g：gather 窗口 1024→2048（负：不再扩）、1002 生产验证、P0e 改序、P0G 任务书 + prepare ring 1→N（速度为负，opt-in 休眠） | 本地 |
| `6fb153e` | tokenizer：CONTROL-token 文本重放为单一 special id（保 exact-prefix KV 复用） | 本地（#727/#985 主线延伸） |
| `e4488c1` `3b2abc4` | 录制 session 前缀复用 A/B 回放器 + compaction prunes 修正 | 本地（同上主线） |
| `d09ca3b` | GLM-5.3 prefill router 走调优 BF16 tensor-unit matmul（M5 +7.3%） | 本地（依赖 `9bc0bd7` 的 NAX 门放开） |
| `4c95017` | 本台账 + AGENTS.md 维护规则入库 | 台账 |
| `db90539` | Metal decode-graph 骨架（M1-1/M1-2）：keyed LRU/poison/param buffer/日志锚全量入库，执行层占位（本机无回放基底），env 默认关，含 `tests/test_metal_graph_capture` 零模型单元验收 | 本地（ITERATION_PLAN M1，处置见 D） |
| `5916126` | M3-1（R2a-1）标定工具链：引擎侧 `DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT[_MS]` 周期快照（默认关、纯观察器）、回放器 opt-in 分轮快照捕获、`tools/hotlist_v2_from_turns.py`（跨轮共现 top-K → `ds4_streaming_hotlist_v2.inc`，K=60/75/85% 三档）、`tools/hotlist_v2_calibrate.py`（保护位 LRU 仿真纸面命中增益，leave-one-out）、`speed-bench/M3/m3-1/run_calibration.sh` 窗口 harness | 本地（ITERATION_PLAN M3-1，[W] 回放腿待窗，见 D） |

## C. 本地工作 → 依赖的 cherry-pick PR（与 main 的差异所在）

| 本地工作 | 直接依赖（PR） | 证据 |
|---|---|---|
| `030b342` | **#1041, #1042** | 正文点名：queue_layers 谓词用 #1041 原版、恢复 `ef45eec` 形状修 `a226fd4` |
| P0 全链（`975238a`…`2dcc1c8`） | **#1041, #1042**（经 `030b342`）+ **#1090** | 舞台是 `ds41_decode_pipeline_admitted` 流水线域（#1090 引入）；#1041 排队域；绑定谓词在 `030b342` 之上 |
| glm-guard 三修 | **#1122**（间接 #1126） | wired-memory guard 即 #1122 引入，修其基线采样 |
| `6fb153e` + `e4488c1`/`3b2abc4` | **#727, #985**（间接 #1093/#1090 server 片） | 同一条"跨轮 exact-prefix KV 复用"主线；回放器即验证这些修复的 A/B 工具 |
| `d09ca3b`、`9bc0bd7`、`1e09244`、`2ce4e40`（NAX 族） | 无硬代码依赖 cherry-pick；验证基线建立在 `0743c7f` 之上 | `2ce4e40` 移植 `1e09244` 的模板；`d09ca3b` 需 `9bc0bd7` 的 M5 门放开；#1149/#1150 是同域借鉴但非同函数 |
| `ff5b4e0` | 无（与 #1152 功能重叠、代码独立） | stash 重放，注明"与上游 1152 不同 patch" |
| #1000/#1003 rewind 片 | **#1089, #1041**（同域） | 同 `ds41_graph` rewind 路径 |
| p42-attn-glue（worktree，未提交） | **#1042** 剩余片 `f61a83d` | 见 V41_M5MAX_BASELINE §12.2 |

## D. 9/25 之后文档 → 已放弃/不再需要的在途工作

### 已明确关账/放弃（别再开工）

| 工作项 | 文档证据 | 裁决 |
|---|---|---|
| V4.1 streaming + DSpark/MTP（含 128G 机器开 MTP） | `V41_STREAMING_ENGRAM_SPEED_PLAN.md` §9：盈亏平衡需 avg_accept≈2.96，实测 1.33–1.91，verify 固定 93 ms/轮 | **放弃**：流式 V4.1 生产走普通 decode；候选 A 已提交 `c7c2b90`（env `DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS` 默认关），作为 E1 [2,32) 低行批量的准入种子保留，翻默认前须过 E0.1；M0-4 窗 A/B 已关账（速度中性、[4,8) 行 logits 逐字节等） |
| GLM top-8 packed 门臂（M0-2 关账） | 原主树 WIP `DS4_METAL_ENABLE_GLM53_ROUTED_MPP_PACKED` | **删臂**（PF-2 packed 已判负；几何 pin 的 packed 对 GLM 顶-8 在 ~57 rows/expert 摊不平 pack 流量的假设未获翻案证据）。臂从未入 git，M0-2 关账后不再携带，勿再从工作区记忆重建 |
| GLM-5.3 开 MTP 提速 | `GLM53_SPEEDUP_NEXT_PLAN.md` DC-1：17k ctx 实测 effective 27.5 vs plain 32.3 t/s（−15%），且越长越亏 | **保持默认关闭**；链式起草/draft head 减半作为前置 P0，但"开 MTP"本身已裁决 |
| P1 Metal row-batch prefill | `HANDOFF-V41-STREAMING-SMALL-PREFILL.md` §4：只值 1.4×，comparator 判 DRIFT | **否决** |
| P0f2：gather 窗口 >2048 | HANDOFF §14.7：2605 行两臂完全平手 | **负结果，帽维持 2048，"别再往上调"** |
| P0g：prepare ring 深度 1→N | HANDOFF §17：24 runs A/B 无差/更差，假设证伪 | **速度为负**，环以 opt-in 休眠（`2dcc1c8` 已入 main，默认 1） |
| 基线方案 阶段1 cache 扩容 | `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1：cache8000≈auto | **关闭，保持 auto** |
| 基线方案 阶段3 mailbox+flush/prescan | §12.6：读已隐藏、免回读实测 −2.7% | **砍掉** |
| 基线方案 阶段5 副本盘 | §10：2026-10-03 无第二块 NVMe | **放弃** |
| 阶段4 卡 A（MoE routed SIMD→NAX） | §14：NAX 要求 B≥16/32/512 行，单 token 形状不适用 | **不可行关闭**；卡 F 低优先挂起；卡 C 降为可选（上限 ~3%） |
| E2 单一数值家族（根治跨路契约） | HANDOFF §15.5 "1002 改判"：**E2 封存** | 仅当 KV checkpoint 混用/金标回归真咬到时重开 |
| P0e/E1/E0.1/E0.2（低行数批量、标量 parity 微测） | HANDOFF §15.5：E1 并入 P0g 顺路项、收益薄；P0g 落地后 §17.6 把靶子改回 **E1+缓存保温** | E1 **降级未死**：100 行档的真正靶子；E0.3 契约卫生挂任意后续 gate 轮 |
| 合并 #1127 prefill layer overlap | `0743c7f` 正文：GLM 短 prompt prefill −19% | **实测否决**，分支保留存档 |
| `V41_M5MAX_SPEED_RESEARCH.md` 的"未取清单" | BASELINE §1 逐项核对：多数已内置于 fork（#1041/#1090/#1042/#1035 等） | **文档已过期**，决策以 BASELINE 为准 |
| GLM "明确不做"清单 | GLM53 计划 §1：HC pre 融合、causal 稠密前缀、qk_rope=64、短尾<32、**Q 残差双趟（已实测回退）** | 禁止再开题；MLX/omlx 百分比一律不作预期 |
| omlx NAX kernel 源码直接搬运 | `OMLX_WAVE_PORT_ANALYSIS.md` §6：布局+量化+框架三重不兼容 | 不可行（思想可借鉴，代码不可搬） |

### 进行中（仍有效，勿当废弃处理）

| 工作 | 载体 | 状态 |
|---|---|---|
| 主工作区未提交改动（M0-2 关账 @2026-10-05） | `--mtp-timing` GLM 三段结算 = **G4a 留、补完并入** `f5c9902`；indexed prefill / DSpark draft trace env = `959a596`；`SPEC_ROWS` env → 见 D 节关账行；`GLM53_ROUTED_MPP_PACKED` → 见 D 节删臂行 | 主树 `git status` 干净；SPEC_ROWS A/B 已随 M0-4 窗关账（中性+logits 等位，维持默认关） |
| p42-attn-glue（M0-3 定夺 @2026-10-05） | 分支 `p42-attn-glue` @`8392281`（WIP archive），worktree 已弃 | **弃臂存档，不合入**：位等价复核实测（M5 Max，带 `DS4_GPU_TEST_V41_FUSIONS` 强开）matvec_bf16 对 Q8_0 5120→1280 逐字节不等、MoE 融合臂 probs 契约同红；M3-Ultra-only admission 即为本险设，位契约优先于 ≈中性。证据 `speed-bench/M0/m0-3/`，M3 Ultra 上位验证通过前勿重试 |
| p3-io-diag | worktree `~/ds4-io-diag` | 7 行诊断补丁，可留可弃 |
| PF-6 上游 PR 移植 | 分支 `pf6-upstream`（基于 origin/main `0aaea5a`，提交 `71268a1`，已推 fork）；worktree `/tmp/pf6-pr-base`（重启即失，重建见杂项） | PR 分支自测 41k +5.2%（364.1→382.8）、锚点确认（f32 router 0 调用、cvt=42、上游 wrapper 无门控故 M3/M5 默认生效）；**已开 [antirez/ds4#1177](https://github.com/antirez/ds4/pull/1177)（OPEN）**，质量数字引用 `d09ca3b` 实测并注明测量树 |

### 本地保留但**未取用**的上游 PR（评审存档，非依赖）

- `pr-710`（live-prefix rewind 上 Flash/Metal；与本地 `2ce4e40` 域不同）、`pr-856`（thinking 通道空白符 cache miss）、`pr-1058`（tool 会话 KV checkpoint 键改 client-visible transcript + 重启恢复冒烟）
- `up/pr/` 旧 ref：**758**（M5 indexed prefill——被 `2ce4e40` 替代）、**1034/1035**（Dango233 SSD decode/Engram——Engram 并行读已被 fork 自身覆盖）、**1060**（indexer radix-select）、**1073**（Tarjei 54-commit 大 PR）、**1152**（GLM 参数类型——`ff5b4e0` 已自研覆盖）、**1153/1154/1155**（image-prefix snapshot / Qwen PLE prefetch / detached backend cache-miss）
- 这些 PR 后续若在上游合入，本地 `ff5b4e0`、`2ce4e40`、gather 系列都可能与之冲突，对拍时需按本清单 A2/B 逐项核对。

### 杂项处置（2026-10 定案）

- **M0-4 基线复核窗（2026-10-05，停产品独占，用户批准）**：14/14 腿绿；KPI 表全部复核成立（ITERATION_PLAN §1 新增 M0 复核列 + `speed-bench/M0/m0-4/notes.md`）。SPEC_ROWS A/B 关账：decode 与 [4,8) 行 append 均中性，step4/13 + step8/10 frontier logits 逐字节等（`m0-4/parity/*.sha256`）→ 维持默认关作 E1 种子。GLM MTP 首账 effective −19.8%（G4a 结算行）。方法学两教训（腿间门用组合可用量、swap 解析）已入 notes。
- **M0-5 上游复核（2026-10-05）**：`git fetch origin` 后 origin/main 仍 `0aaea5a`，`git cherry` 68 片全部未吸收；#1177/#1178 均 OPEN、零 review 零评论；本文件引用的全部 SHA（含 `cardi-upstream 5805843/4e48bb5`、`pf6-upstream 71268a1`）仍有效，无需刷新。窗口前照例复查。
- **M1-1/M1-2 落账 + macOS 26.5.1 Metal 捕获基底关账（2026-10-05，无窗口、零模型加载）**：M1-1 原方案（MTLGraph `captureScopeWithCommandBuffer:`）经三重取证在 macOS 26.5.1 (25F80)/Xcode SDK 26.5 上不存在（SDK 全目录零 graph 头、活跃 `AGXG17XFamilyCommandBuffer` 全继承链零 capture/replay selector、dyld 共享缓存 selector 字符串零命中、Apple 文档 404）；p42-attn-glue 分支亦无历史 capture 代码。用户裁决改道 MTLIndirectCommandBuffer，实现完成时以 7 变体裸 Metal 探针（A–G：concurrent/legacy、shared/private、threads/threadgroups、serial/concurrent encoder、inherit/绑定 组合）实测驱动黑洞：**任何带绑定的 CPU 录制（`indirectComputeCommandAtIndex` + `setComputePipelineState:`）直接 segfault 于 `AGXG17XFamilyIndirectComputeCommand setComputePipelineState:`（lldb 栈确认）**，唯一不崩的 inherit 形态继承 pipeline/buffer 不生效、dispatch 静默蒸发。**裁决（用户）：M1 捕获路线搁置**，骨架入库（`db90539`，default off，`tests/test_metal_graph_capture` 全绿），M1-3/1-4/1-5 窗口与 GLM ≥35 t/s 出口线作废重定（ITERATION_PLAN §1/§3 已注）。复活路径 = ds4_metal.m 三处 "M2-3 SUBSTRATE HOOK"（begin 建录制缓冲 / proxy recordDispatch 落命令 / execute_entry 执行），**动手前重跑 `git stash` 无关的探针矩阵**：/tmp 探针文件重启即失，重建要点=本条描述 + 崩溃点符号，每次 macOS 升级后先探后用。V4.1 侧 decode graphs（V4.1 islands 共用 supported()）同步搁置——env 关闭下 islands 恒 eager，无行为变化。**别把它当 bug 修**：`ds4_dg_execute_entry` 返回 NO 是裁决形态不是遗漏。
- **M3-1 标定工具链落账（2026-10-05/06，无窗口、零模型加载）**：[O] 编码半程完成（`5916126`）——引擎快照 dump（env 默认关；无 env 时记录路径零新增分支/线程）、回放器分轮快照捕获（opt-in，默认行为逐行不变）、生成/标定双脚本（各 6/6 组 selftest 绿 + 合成件端到端跑通）、窗口 harness（§2 预检 + MAX_STEPS 切片 + swap 看门狗 + 独立 kv-disk-dir 不碰生产 `~/.ds4/server-kv`）。候选 session 已在本机 DSH session 库筛出 208 条 ≥8 轮 agent 型（最大 1179 步，标定窗建议 `MAX_STEPS=40..80` 切片）。**[W] 回放腿待用户开窗**：`SESSION=<jsonl[.zst]> MAX_STEPS=60 speed-bench/M3/m3-1/run_calibration.sh`；纸面增益闸门 ≥25pp。
- **本机新增既有红项（2026-10-06 stash 对照定案，= HEAD `f08c897` 非回归，别追）**：`tests/test_deepseek41_fusions`（matvec_bf16 与 Q8_0 位不等，与 M0-3 p42 裁决同源的平台性红）与 `tests/test_deepseek41_engram_admission`（step-reader 准入断言 `line 56: calls[i]==(i==reader)`）在**未含本会话改动的 HEAD 重建件**上复现同 rc=1、同输出；`./ds4_test` 全量在默认模型缺失时于 `qwen4-prefill-checkpoints` 断言退出（`ds4flash.gguf` 链接目标 5 月起缺失的环境资产红，历史窗均显式 `DS4_TEST_MODEL` 运行）。零模型验收口径 = `ds4_test --server` + 非模型测试二进制 + `test_metal_graph_capture` + `test_metal_ssd_experts`，全部绿。
- 一次性噪声已入 `.git/info/exclude`（纯本地、不随任何 PR 携带）：`pic.jpeg`、`start.md`/`start.txt`（server 启动备忘）、`tests/*.o.tmp`、`tests/test_metal_rewind`（构建残留）、`tests/test_metal_graph_capture`（构建残留，2026-10-06 补）、`.codegraph/`、`.cursor/`
- 本文件与 `AGENTS.md` 已纳入 git 跟踪。**PR 纪律**：PR 分支一律从上游 `origin/main` 切、只 cherry-pick 目标修复提交；不 merge 本地 main、不 `git add -A`——docs commit 不在 PR 范围即不会携带
- `/tmp/ds4-main`、`/tmp/pf6-pr-base` 对照 worktree：重启即失，需按 BASELINE §13.2 重建流程

## E. 全仓代码对账（2026-10-04）：A/B/D 之外的代码

对 main、全部分支、4 个 worktree、stash、未跟踪文件逐一枚举后，**main 的 58 个提交与 A/B 全部对上账**（上游作者 34 片、本地 23 片、merge 1）；`pr-*`/`up/pr/*` 均为 PR 原文存档（D 已列），`feat/glm53-*` 两分支仅多 3 个提交 = `1e09244`/`003e007`/`ff5b4e0` 的原件（无独立代码），`fix/*` 指向 main 上的提交，`/tmp` 两对照 worktree 无独有代码。真正的"账外代码"只剩以下三项（stash 三项已裁决丢弃）：

| 项 | 内容与规模 | 状态/建议 |
|---|---|---|
| `stash@{0}`/`{1}`/`{2}` | 0/1 = `9bc0bd7`、`ff5b4e0` 的已应用残留；2 = `ds4.c` +4 `vocab_size` 缺失回退 129280（unsloth GGUF 适配，未入 main） | **2026-10-04 用户裁决全部丢弃**，`git stash clear`；patch 备份在 `.git/LOCAL_BACKUPS/stashes-2026-10-04.patch`，日后要捡回 vocab 回退从这里取 |
| card H 实现（`~/ds4-cardh`，**`5573d62` @ `p4-cardh-early-load`**，基线 3b2abc4，ds4.c +63 行） | `DS4_METAL_ENABLE_V41_MOE_EARLY_LOAD` 门控（默认关）：router 后 GPU 事件 + worker 线程读 6 id/预暂存 pread，与共享专家编码重叠，`set_selected_override` 免逐层回读；`.cardh-ab/` 产物未入库 | **2026-10-04 关账：bit-exact 全过（CLI 贪心 A/B 逐字节一致、logprob ON==OFF、dspark 两测 ON==OFF、专家缓存 4/4），但速度负结果**：ABBA 2×(2K,32K)×512tok steady OFF 23.83/22.54 vs ON 23.51/22.48（−1.3%/−0.3%，组内噪声 4%）。此前"layer 0 failed"是旧二进制未重编 + flush 轮换 CB 后无条件 begin 的 bug，均已修；parity 0.36 漂移证实为存量流式行为（env OFF 相同）。14.6 的"19ms=可重叠 GPU 等待"假设证伪（GPU 30% 忙，等待近零返回；真主犯是主线程 CPU 簿记量）。裁决见 BASELINE §15；**不合入、勿重试此形态** |
| card I / I-1 实现（`~/ds4-cardi`，**`376687e` @ `p4-cardI-graph-capture`**，基线 d3a6ab6，ds4_metal.m +595/−67，ds4.c/头文件/内核零改动） | `DS4_METAL_ENABLE_V41_MOE_GPU_BINDING` 门控（默认关，单卡+streaming+单 token）：routed MoE 走现成 addr-table 入口吃 GPU 侧 `g->selected` ids，新 service 线程等 CB 的 ids 事件后自主读 ids、装载缺失、标 inflight、回填 6 槽并 prune，CB 等 ready 事件再 dispatch——主线程每层不再回读；`.cardi-ab/` 产物未入库 | **2026-10-04 I-1 收账：bit-exact 全过 + 正向小收益**。CLI 贪心 A/B 逐字节一致（默认 env，含 masked 族）、logprob ON/OFF 全日志 0 差异、dspark 两测 ON==OFF、专家缓存四模式全过；ABBA 2×(2K,32K)×512tok OFF 22.44/21.82 → ON 23.17/22.40（**+3.2% / +2.7%**，四比较点全不重叠）。两条入档硬知识：①service 线程标 inflight 必须用 arm 时抓的 CB 序列而非环境序列（否则 drain 窗口=无保护→同命令三代输出且零日志）；②经地址表取权重的 dispatch 必须 `useResource` 标到槽 buffer（绑定模式改为标 19 个 slab，标上即 3/3 复现）——**I-2 的 capture 形态必须带上这个标记**。另补存量隐患：`take_reusable_batch` 的 batch-reuse 分支缺 `on_service_thread()` 守卫（四处 wait_inflight 中唯一没有的）。PR 基线（origin/main `0aaea5a` + `5805843`/`4e48bb5`）单独实测：gate1 逐字节一致、logprob ON==OFF、专家缓存四模式过，ABBA OFF 18.11/17.03 → ON 18.48/17.54（**+2.0%/+3.0%**，PR 树没带 fork 的 decode 融合线所以绝对值低）——已推 fork `cardi-upstream` 并开 **[antirez/ds4#1178](https://github.com/antirez/ds4/pull/1178)**（+599/−68 单文件，零 cherry-pick 依赖：流式缓存 `9ba160a`/地址表内核/服务线程标记 `519c4d8` 全在上游）。裁决见 BASELINE §17；**在途，I-2 判定 go，env 保持默认关** |
| 未跟踪工具脚本 ~1.2k 行（M0-6 关账 @2026-10-05，`3571118`/`40daf5e`） | `gguf-tools/deepseek41_dspark_convert.py`(349)、`speed-bench/{build_dspark_support_gguf.py(228), jigsaw_to_ds4_dspark.py(211), mtp_ledger_replay.py(160)}` = DSpark/MTP 资产管道；`speed-bench/{v41_m5max_streaming_ab.sh(66), v41_m5max_server_probe.sh(50)}` + `v41_m5max_ab/` 实测数据 = 基线方案台架 | 全部已入库 main：DSpark 四件套随 MTP 线裁决**封存备查**（≥256G 机重开时要用）；v41 台架 + CSV 为本尊（`ds4-io-diag` 等处仍是拷贝，清理时别认错） |

另：`CLAUDE.md` 旧孪生文档已于 M0-6（`40daf5e`）改为指针，AGENTS.md 为单一事实源；`tests/test_glm53_router_shared.c` 的测试旗标小补丁已随 M0-1 入 main（`6d8fdee`）。

