# 本地仓库 vs 上游 antirez/ds4：提交与在途工作清单

> 生成日期：2026-10-03。基准：`main` = `d09ca3b`，比 `origin/main`（上游）领先 **57 个提交**（34 个 cherry-pick + 1 个 merge + 22 个本地实现）；`fork/main`（williamxie1989/ds4）落后本地 main 35 个提交——最后 35 个（含 09-30 之后的全部 NAX 移植、P0 系列、prefix-replay 工具链）尚未推到 fork。
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

## B. 本地实现提交（22 个，含从自己 feat 分支重新落地）

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
| V4.1 streaming + DSpark/MTP（含 128G 机器开 MTP） | `V41_STREAMING_ENGRAM_SPEED_PLAN.md` §9：盈亏平衡需 avg_accept≈2.96，实测 1.33–1.91，verify 固定 93 ms/轮 | **放弃**：流式 V4.1 生产走普通 decode；候选 A 仅以 env `DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS`（工作区未提交）实验性存在 |
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
| 主工作区未提交改动 | `ds4.c`/`ds4_metal.m`：`DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS`（MTP A 候选实验）+ `DS4_METAL_ENABLE_GLM53_ROUTED_MPP_PACKED`（GLM top-8 packed 门，待 prefill A/B）+ `--mtp-timing` 统计 + indexed prefill trace env | BASELINE §13.2 明示 **WIP 勿动**，A/B 未定 |
| p4-cardh-early-load（卡 H：decode early-load） | worktree `~/ds4-cardh` + `.cardh-ab/` | BASELINE §14.6 新增**主攻** |
| p42-attn-glue | worktree `~/ds4-attn-glue`，全部未提交 | 实测 ≈中性，**等用户 review 决定合/弃** |
| p3-io-diag | worktree `~/ds4-io-diag` | 7 行诊断补丁，可留可弃 |
| `tests/test_glm53_router_shared.c` 未提交补丁 | 测试补 `ds4_gpu_test_set_flags` | 小修，可随手提交 |

### 本地保留但**未取用**的上游 PR（评审存档，非依赖）

- `pr-710`（live-prefix rewind 上 Flash/Metal；与本地 `2ce4e40` 域不同）、`pr-856`（thinking 通道空白符 cache miss）、`pr-1058`（tool 会话 KV checkpoint 键改 client-visible transcript + 重启恢复冒烟）
- `up/pr/` 旧 ref：**758**（M5 indexed prefill——被 `2ce4e40` 替代）、**1034/1035**（Dango233 SSD decode/Engram——Engram 并行读已被 fork 自身覆盖）、**1060**（indexer radix-select）、**1073**（Tarjei 54-commit 大 PR）、**1152**（GLM 参数类型——`ff5b4e0` 已自研覆盖）、**1153/1154/1155**（image-prefix snapshot / Qwen PLE prefetch / detached backend cache-miss）
- 这些 PR 后续若在上游合入，本地 `ff5b4e0`、`2ce4e40`、gather 系列都可能与之冲突，对拍时需按本清单 A2/B 逐项核对。

### 杂项处置（2026-10 定案）

- 一次性噪声已入 `.git/info/exclude`（纯本地、不随任何 PR 携带）：`pic.jpeg`、`start.md`/`start.txt`（server 启动备忘）、`tests/*.o.tmp`、`tests/test_metal_rewind`（构建残留）、`.codegraph/`、`.cursor/`
- 本文件与 `AGENTS.md` 已纳入 git 跟踪。**PR 纪律**：PR 分支一律从上游 `origin/main` 切、只 cherry-pick 目标修复提交；不 merge 本地 main、不 `git add -A`——docs commit 不在 PR 范围即不会携带
- `/tmp/ds4-main`、`/tmp/pf6-pr-base` 对照 worktree：重启即失，需按 BASELINE §13.2 重建流程
