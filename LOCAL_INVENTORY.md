# 本地仓库 vs 上游 antirez/ds4：提交与在途工作清单

> 生成日期：2026-10-03（2026-10-04 全仓代码对账更新；**2026-10-08 rebase 至上游 `fc80bd6` 后刷新**；2026-10-09 pick #1192 后刷新）。基准：`main` = `c4e5e97`，比 `origin/main`（上游）领先 **116 个提交**（A 段 cherry-pick 31 + 其余本地实现/文档/台账 85；merge 提交 0）；`fork/main` 停在 `3cfd0c0`（旧链），**待同步**——注意其中含 HANDOFF/PLAN/本台账等私有文档，fork 可见性需自查。
>
> **2026-10-08 rebase 事件**：`git pull --rebase`（等价 `git rebase origin/main`）把上游 9 笔直接提交（`962ac19`→`fc80bd6`，10-05~10-07）吸收为祖先；117 步重放后本地 114 笔。按“上游与本地同目标以上游为准”丢弃 3 笔：`9c4cf2c`/`bca9985`（#1003，被上游 `fc80bd6` 取代）、`4db95ee`（B2，上游机制无 vision 排除）。merge 提交 `0743c7f` 被 rebase 丢弃（内容由重放各片保留）。冲突处置与核验见 §F。
>
> 依据：cherry-pick trailer、patch-id 对拍（**注意**：patch-id 含上下文行，基座不同即“伪不一致”——本轮 13 个带 trailer 的 pick 全部不一致，不能据此断言被改写）、`pr-*`/`up/pr/*` 本地分支、以及仓库内 9/25 之后的移交/规划文档。`0743c7f` 正文自本轮起不再作为依据（提交已不存在）。
>
> **本文件是 fork 私有台账**：只跟踪在本地/fork 的 main 上，按 PR 纪律（文末）不会进入任何提给上游的 PR。维护规则见 [AGENTS.md](AGENTS.md) 的 "Upstream picks & LOCAL_INVENTORY.md maintenance" 一节。

---

## A. Cherry-pick 清单（按上游 PR 分组）

### A1. 09-30 "Merge upstream perf PRs for GLM-5.3-Flash and DeepSeek" 批次（原经 merge `0743c7f` 并入；2026-10-08 rebase 丢弃该 merge 后各片以独立提交重放，M5 Max 已验证；GLM 侧 19 片）

| 上游 PR | 主题 | 落到 main 的提交 | 采纳度 |
|---|---|---|---|
| #1090 (trueimage) | exact GLM Flash kernels + V4.1 流水线 decode + server 精确前缀 | `3ee98ab`, `4a84e65`, `1fe1ec5` | **3/6**（未取 MXFP4 保护、gguf-tools 审计、docs 三片；两片冲突手工解） |
| #1093 (Michael Kuebbeler) | GLM live checkpoint 跨 tool 轮保活（分析见 #1092） | `196bc4f` | 1/1 |
| #1120 (Emilian Bold) | M5 resident MoE/Q2 sum6/KDA Q8 decode 形状特化 | `52c482a`, `292ae70`, `6cfc129`, `bb98ec3`, `4847e99` | 5/5 |
| #1122 (Adam Margulies) | Metal GLM 内存 guard 按 engine open 时 wired 内存封顶 | `dddd374` | 1/1 |
| #1126 (Adam Margulies) | keep-alive 线程启动前先解析 keep-alive pipeline | `331f8af` | 1/1 |
| #1135 (Sebastian Christiansen) | Metal visual attention 压缩 key 因果性修复 | `0b793da` | 1/1 |
| #1147 (Emilian Bold) | validated selection prefixes 找回 Metal GLM prefill 速度 | `cfbd15c` | 1/1 |
| #1149 (Will) | Qwen3.8 QSA prefill attention 上 tensor units | `06cf4ac`, `76d3e35` | 2/2 |
| #1150 (Will) | hyper-connection 残差写延迟进下一个 stream norm | `3a99293` | 1/1 |
| #1127 (Emilian Bold) | callback prefill layer overlap | **实测否决、未入树**：GLM 短 prompt prefill −19%（分支 `pr-1127` 保留） | 0/1 |

### A2. 单独 cherry-pick 到 main 第一父链

| 上游 PR | 主题 | 提交 | 落点 |
|---|---|---|---|
| #1006 (Simone Scarduzio) | tool-call 静默期 SSE keepalive 注释（防客户端 300s 超时重灌） | `380487e` | merge 之前的基座 |
| #1000 (ayuan1129) | rewind 泄漏 stop-boundary draft token；terminal rewind 后补文本续写 | `2498e0f`, `b5c4ce2` | merge 后（09-30 序列）；**2026-10-08 与上游 `fc80bd6` 同址冲突**——保留 #1000 的早停/无条件回滚/零图短路语义，采用上游新 rewind 结构（见 §F） |
| ~~#1003 (Emilian Bold)~~ | DSpark 快照复用做即时 rewind + 防 stale handle | **已退役**（2026-10-08 rebase 丢弃 `9c4cf2c`/`bca9985`）：上游 `fc80bd6`（10-07）在同机座（`spec_frontier_*`）上实现同目标，并补上本地登记的残留缺口（边界 logits）+ TP 协调；PR 原文存档仍在分支 `fix/spec-rewind-rebuild` |
| #1041 (Adrian Galilea) | 单盒 decode 命令排队、逐层 commit 不等待 | `656623a` | **1/2**（未取 `fdbf7f2` "queue through the logits tail"） |
| #1042 (Adrian Galilea) | decode MoE/HC/路由 glue 融合族（19→6 dispatch/层）+ 回退开关 | `520fcc2`, `10b59ae`, `177ce3b`, `79a46b2`, `6d382a5`, `20f3b3d` | **6/7**（未取 `f61a83d` attention glue——后在 p42-attn-glue worktree 单独实测，见 D） |
| #1043 (Adrian Galilea) | V4.1 Flash routed 形状默认 Q4_K group-6 专家表 | `c99c247` | V4.1 批次 |
| #1061 (RTSAI Coder) | decode 形状的 V4.1 indexer 内核（长上下文） | `115d973` | V4.1 批次 |
| #1089 (RTSAI Coder) | V4.1 live session 直接 rewind 不重建 | `423c5c2` | V4.1 批次 |
| #985 (Rick Ratmansky) | 归一化 tool replay 跨轮保住 KV cache | `504f298` | 10-02 前缀复用批次 |
| #727 (LEFBE) | transient metadata block 剥离后保住 live KV（issue #364，#378 的返工） | `4a9efa8`, `97e8a0c` | 同上 |
| #1192 (Emilian Bold) | 非因果 top-k argsort 接 simd_shuffle_xor 寄存器排序网络（plain 内核此前只走 threadgroup 网络；作者 M5 Max 实测整调用 −20~23%，输出 bit-identical） | `c4e5e97` | 2026-10-09 批次；PR **OPEN**（base `0aaea5a`，对上游 mergeable clean）、`cherry-pick -x` + `Upstream-PR:` trailer；机制与已默认上产的 `..._causal_shuffle` 同模板零新增。收益落点：GLM indexed prefill 多行 topk（`ds4.c:57077`）、V4.1/GLM decode fast-path 回退排序（`ds4_metal.m:21474` 的 plain sort）、10000-experts router topk（`ds4_metal.m:36821`）。内核验收全绿：`test_deepseek41_topk`（600 poisoned）、`test_glm53_topk_fast`（720）、`test_glm53_router_shared`、`test_glm53_q8_inputs`；`test_deepseek41_metal` 内多行非因果 topk 与 causal ties 断言通过（该套件另有存量 attn_flags 红，见 D 既有红项，非本 pick 引入）。**本机 ABBA 复测（同二进制 + `DS4_METAL_ARGSORT_SOURCE` 换源，零模型，2026-10-09，腿 log `/tmp/ab_{A1,B1,A2,B2}.txt`，取各形状两轮最好）**：prefill 形状 256 行——8192×256 983→746 µs（**−24.1%**）、16384×256 1861→1377（**−26.0%**）、32768×256 3685→2831（**−23.2%**），与作者 −20~23% 吻合；decode 形状 1 行——indexer-8k −1.3%、indexer-32k +1.8%、cand-4k −1.5%、router-384 −2.2%，**全部埋在 ~150 µs 固定 commit/wait 地板里 = 噪声带内，decode 无感**。⇒ 收益集中在多行非因果 topk（GLM indexed prefill 主路）；DSv4.1 causal prefill 早已走 causal_shuffle 无新增。原文存档分支 `pr-1192` |

## B. 本地实现提交（29 个，含从自己 feat 分支重新落地；2026-10-08 rebase 后 SHA 已刷新）

| 提交 | 主题 | 性质 |
|---|---|---|
| ~~`0743c7f`~~ | merge：18 个上游 perf PR 集成 + M5 Max 验证记录 | **2026-10-08 rebase 丢弃 merge 提交**：内容由 A1 各片重放保留；当时的手工冲突解在本轮以冲突形式重现并逐处重解（见 §F） |
| `aaf15df` | `DS4_METAL_FORCE_GLM53_TUNING=1` 供 A/B | 本地（#1122 邻接） |
| `e57990c` `f2c5d98` `c27b5de` | glm-guard：wired baseline 每进程采一次 + refuse 前重采 | 本地，**修复 #1122** |
| `e067c8f` | GLM tool-call 参数按 request schema 恢复类型 | 本地自研；**与未取的上游 #1152 同一问题域**（patch-id 不同，合并 #1152 前需人工对拍） |
| `2fe096e` `62bbc0b` | GLM-5.3 sparse MLA prefill 上 NAX tensor units + 回退开关接入 | 自 feat 分支重放（`feat/glm53-sparse-mla-tensor-units`） |
| `22193b5` | GLM53 tuning 家族放开到 NAX（M4/M5）域 | 自 feat 分支重放（`feat/glm53-tuning-gate-nax-domain`） |
| `64b6461` | DSV4 sparse indexed attention prefill 上 NAX（M5 默认开） | 本地；**是上游 #758（M5 Max indexed prefill）的自研替代** |
| `6cde6b7` | 调和 #1041/#1042 cherry-pick 与 fork decode graph | 本地修复（修 `6d382a5` 条件断链） |
| `f75afe7` `f3151ef` | P0c：streaming gather 镜像进 mm-id 内核族、bit-exact、默认开 + P0d 地板 256→32 | 本地（HANDOFF §14） |
| `cbbe344` `5d36e85` `0697898` `6016540` `519638a` `705a5cb` | P0f/P0f2/P0g：gather 窗口 1024→2048（负：不再扩）、1002 生产验证、P0e 改序、P0G 任务书 + prepare ring 1→N（速度为负，opt-in 休眠） | 本地 |
| `ed3aafb` | tokenizer：CONTROL-token 文本重放为单一 special id（保 exact-prefix KV 复用） | 本地（#727/#985 主线延伸） |
| `d34a178` `67fe336` | 录制 session 前缀复用 A/B 回放器 + compaction prunes 修正 | 本地（同上主线） |
| `05cf0f3` | GLM-5.3 prefill router 走调优 BF16 tensor-unit matmul（M5 +7.3%） | 本地（依赖 `22193b5` 的 NAX 门放开） |
| `88f2eed` | 本台账 + AGENTS.md 维护规则入库 | 台账 |
| `fde5b63` | Metal decode-graph 骨架（M1-1/M1-2）：keyed LRU/poison/param buffer/日志锚全量入库，执行层占位（本机无回放基底），env 默认关，含 `tests/test_metal_graph_capture` 零模型单元验收 | 本地（ITERATION_PLAN M1，处置见 D） |
| `ccc04b1` | M3-1（R2a-1）标定工具链：引擎侧 `DS4_MOE_RECORD_SELECTED_HOTLIST_SNAPSHOT[_MS]` 周期快照（默认关、纯观察器）、回放器 opt-in 分轮快照捕获、`tools/hotlist_v2_from_turns.py`（跨轮共现 top-K → `ds4_streaming_hotlist_v2.inc`，K=60/75/85% 三档）、`tools/hotlist_v2_calibrate.py`（保护位 LRU 仿真纸面命中增益，leave-one-out）、`speed-bench/M3/m3-1/run_calibration.sh` 窗口 harness | 本地（ITERATION_PLAN M3-1，[W] 回放腿待窗，见 D） |
| `c9505e0` | M4-3（G2f）NAX ≥2 GiB 寻址审计可执行化：`ds4_gpu_test_nax_2gib_units()` 五族单元（direct_rhs/mpp/packed/mla/indexer，真实内核、同一 backing buffer 内 2 GiB+64 KiB 摆双份相同字节、位相等断言）+ `tests/test_metal_nax_tensor_units` C 驱动（内存地板预检）+ Makefile 目标 | 本地（ITERATION_PLAN M4-3，审计结论与窗口复跑口径见 D） |
| `1c44caa` + `0e96c41` | 解码侧 draftless n-gram 投机：可行性（`tools/ngram_accept_sim.py`/`ngram_accept_policy.py`/`NGRAM_DRAFTING_STUDY.md`，零模型加载）→ 窗口实测成本斜率（`tools/ngram_cost_model.py` + `tests/test_deepseek41_dspark.c --verify-scan[-ssd]`）——**实测 a≈0.68（一行 verify ≈0.68 decode 步），最优策略 1.000×，结构性上界 1.387×；判死，不实现** | 本地（**2026-10-06 关账：实测不可行**，见 D；study 原 1.36–1.70× 已就地更正） |
| `005a452` | M3-7/M3-5 E0：`--short-prefill-ssd-rows` 登记红灯臂 + bind-parity 行集扩 {9,16,31}；实测 [9,31] 批量==步进逐位等、≥32 红灯（1.9–2.8 logits）；**bind 轴 [9,31] addr-vs-whole-map DRIFT ≤7.6e-06 新红灯（工具 rc=1）**，E1 按 M3-5 红线停线待裁 | 本地（HANDOFF §15.2b；**裁决 2026-10-05=iii 弃 E1**，红灯长期登记） |
| `168df92` `3833a39` `2148647` `7385e6f` | V4.1 decode 瓶颈线（全部 test-only + 文档，零引擎改动）：`--verify-scan-ssd` 改用 `proc_pid_rusage` 记进程自身读盘、`--stage-timing` 支持 streaming 域并修正 `force_resident` 硬编码、`--weight-inventory` 按显式张量名分组、`--routed-split-ssd`（预热地址表后单独计 gather 内核，并把 paging／per-layer 排空／往返分臂） | 本地（测量与文档；结论见 D 与 `V41_DECODE_BOTTLENECK.md` §8） |
| `b203362` | V4.1 decode 未命中路径归因插桩：streaming-expert 计时系统加 5 组 env-gated 计数器（residency_scan、prune_layer/prune_global 的时间+驱逐+扫描量、pread pool submit+sync fallback、load wait 拆 overlap/block），`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY` 关时零行为变更、零模型验收全绿；三腿归因读数见 D 与 §10 | 本地（测量插桩；结论见 D 与 `V41_DECODE_BOTTLENECK.md` §10，2026-10-06） |

### B2. 2026-10-08 退役记录（视觉会话快照资格修复）

| 提交 | 主题 | 状态 |
|---|---|---|
| ~~`4db95ee`~~ | 视觉会话纳入 DSpark 回滚快照资格（去掉 `!ds4_session_has_vision_state(s)`）。症状=图片会话 tool 轮后整段 re-prefill（`live kv cache miss ... common=0 vision=mismatch reason=token-mismatch`）；根因链=DSML tool 结束点落在投机块中间 → server stop-boundary rewind → 图片会话被排除在快照外（`dspark_rollback=0`）→ `checkpoint_valid=false`（len 保留）→ 下一请求 `common=0`+`vision=mismatch` → 全量重灌 | **已退役（2026-10-08 rebase 丢弃）**：上游 `fc80bd6` 的 `dspark_rewind_*` 机制对 vision 会话**不设排除**（守卫只拒非 leader 的 TP 侧），B2 的 bug 在结构上不存在。其回归测试 `tests/test_server_vision_tool_boundary.py`（211 行）与随片日志（引擎侧 `dspark_rollback=...` 日志来自 #1003 片，server 侧 3 行来自本片）一并退役。**日后复验该症状**：按上游机制重写等价回归（断言 `ds4_session_rewind_speculative` 命中），并跑真实模型窗口 |

## C. 本地工作 → 依赖的 cherry-pick PR（与 main 的差异所在）

| 本地工作 | 直接依赖（PR） | 证据 |
|---|---|---|
| `6cde6b7` | **#1041, #1042** | 正文点名：queue_layers 谓词用 #1041 原版、恢复 `10b59ae` 形状修 `6d382a5` |
| P0 全链（`f75afe7`…`705a5cb`） | **#1041, #1042**（经 `6cde6b7`）+ **#1090** | 舞台是 `ds41_decode_pipeline_admitted` 流水线域（#1090 引入）；#1041 排队域；绑定谓词在 `6cde6b7` 之上 |
| glm-guard 三修 | **#1122**（间接 #1126） | wired-memory guard 即 #1122 引入，修其基线采样 |
| `ed3aafb` + `d34a178`/`67fe336` | **#727, #985**（间接 #1093/#1090 server 片） | 同一条"跨轮 exact-prefix KV 复用"主线；回放器即验证这些修复的 A/B 工具 |
| `05cf0f3`、`22193b5`、`2fe096e`、`64b6461`（NAX 族） | 无硬代码依赖 cherry-pick；验证基线建立在 A1 各片之上（原 merge `0743c7f` 已于 2026-10-08 rebase 丢弃） | `64b6461` 移植 `2fe096e` 的模板；`05cf0f3` 需 `22193b5` 的 M5 门放开；#1149/#1150 是同域借鉴但非同函数 |
| `e067c8f` | 无（与 #1152 功能重叠、代码独立） | stash 重放，注明"与上游 1152 不同 patch" |
| #1000 rewind 片（`2498e0f`/`b5c4ce2`） | **#1089, #1041**（同域）+ **上游 `fc80bd6`**（同址） | 同 `ds41_graph` rewind 路径；`fc80bd6` 重写同一 `decode_again` 块，语义已手工合并 |
| ~~#1003 rewind 片~~ | — | **已退役**（上游 `fc80bd6` 取代）；原依赖 #1089/#1041 随片消失 |
| ~~B2 视觉快照资格（`4db95ee`）~~ | #1000, #1003 | **已退役**（上游机制无 vision 排除）；回归测试随之退役，见 B2 |
| p42-attn-glue（worktree，未提交） | **#1042** 剩余片 `f61a83d` | 见 V41_M5MAX_BASELINE §12.2 |

## D. 9/25 之后文档 → 已放弃/不再需要的在途工作

### 已明确关账/放弃（别再开工）

| 工作项 | 文档证据 | 裁决 |
|---|---|---|
| V4.1 streaming + DSpark/MTP（含 128G 机器开 MTP） | `V41_STREAMING_ENGRAM_SPEED_PLAN.md` §9：盈亏平衡需 avg_accept≈2.96，实测 1.33–1.91，verify 固定 93 ms/轮 | **放弃**：流式 V4.1 生产走普通 decode；候选 A 已提交 `b9617e5`（env `DS4_METAL_V41_STREAM_GATHER_SPEC_ROWS` 默认关），作为 E1 [2,32) 低行批量的准入种子保留，翻默认前须过 E0.1；M0-4 窗 A/B 已关账（速度中性、[4,8) 行 logits 逐字节等） |
| **投机解码整类**（draftless n-gram 实测；机理同时覆盖 MTP/DSpark 的失败） | `NGRAM_DRAFTING_STUDY.md` Outcome + `tests/test_deepseek41_dspark.c --verify-scan[-ssd]`：一行 verify 成本 **a≈0.68** decode 步（四配置一致：缓存 958→8078 experts 不动、8 行同 token 亦不动）；逐匹配长度 cap 寻优后 **1.000×**；**oracle 上界 1.387×** | **整类关闭**：V4.1-Q2 streaming 上任何投机方案（n-gram / MTP / DSpark / 未来变体）上界 1.387×，不值得；**重开任一变体前必须先重测 a**（`--verify-scan-ssd`） |
| GLM top-8 packed 门臂（M0-2 关账） | 原主树 WIP `DS4_METAL_ENABLE_GLM53_ROUTED_MPP_PACKED` | **删臂**（PF-2 packed 已判负；几何 pin 的 packed 对 GLM 顶-8 在 ~57 rows/expert 摊不平 pack 流量的假设未获翻案证据）。臂从未入 git，M0-2 关账后不再携带，勿再从工作区记忆重建 |
| GLM-5.3 开 MTP 提速 | `GLM53_SPEEDUP_NEXT_PLAN.md` DC-1：17k ctx 实测 effective 27.5 vs plain 32.3 t/s（−15%），且越长越亏 | **保持默认关闭**；链式起草/draft head 减半作为前置 P0，但"开 MTP"本身已裁决 |
| P1 Metal row-batch prefill | `HANDOFF-V41-STREAMING-SMALL-PREFILL.md` §4：只值 1.4×，comparator 判 DRIFT | **否决** |
| P0f2：gather 窗口 >2048 | HANDOFF §14.7：2605 行两臂完全平手 | **负结果，帽维持 2048，"别再往上调"** |
| P0g：prepare ring 深度 1→N | HANDOFF §17：24 runs A/B 无差/更差，假设证伪 | **速度为负**，环以 opt-in 休眠（`705a5cb` 已入 main，默认 1） |
| 基线方案 阶段1 cache 扩容 | `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1：cache8000≈auto | **关闭，保持 auto** |
| 基线方案 阶段3 mailbox+flush/prescan | §12.6：读已隐藏、免回读实测 −2.7% | **砍掉** |
| 基线方案 阶段5 副本盘 | §10：2026-10-03 无第二块 NVMe | **放弃** |
| 阶段4 卡 A（MoE routed SIMD→NAX） | §14：NAX 要求 B≥16/32/512 行，单 token 形状不适用 | **不可行关闭**；卡 F 低优先挂起；卡 C 降为可选（上限 ~3%） |
| E2 单一数值家族（根治跨路契约） | HANDOFF §15.5 "1002 改判"：**E2 封存** | 仅当 KV checkpoint 混用/金标回归真咬到时重开 |
| P0e/E1/E0.1/E0.2（低行数批量、标量 parity 微测） | HANDOFF §15.5：E1 并入 P0g 顺路项、收益薄；P0g 落地后 §17.6 把靶子改回 **E1+缓存保温** | E1 **降级未死**：100 行档的真正靶子；E0.3 契约卫生挂任意后续 gate 轮 |
| 合并 #1127 prefill layer overlap | `0743c7f` 正文：GLM 短 prompt prefill −19%（该 merge 已于 2026-10-08 rebase 丢弃，正文取自 `backup/pre-rebase-20261008`） | **实测否决**，分支保留存档 |
| `V41_M5MAX_SPEED_RESEARCH.md` 的"未取清单" | BASELINE §1 逐项核对：多数已内置于 fork（#1041/#1090/#1042/#1035 等） | **文档已过期**，决策以 BASELINE 为准 |
| GLM "明确不做"清单 | GLM53 计划 §1：HC pre 融合、causal 稠密前缀、qk_rope=64、短尾<32、**Q 残差双趟（已实测回退）** | 禁止再开题；MLX/omlx 百分比一律不作预期 |
| omlx NAX kernel 源码直接搬运 | `OMLX_WAVE_PORT_ANALYSIS.md` §6：布局+量化+框架三重不兼容 | 不可行（思想可借鉴，代码不可搬） |

### 进行中（仍有效，勿当废弃处理）

| 工作 | 载体 | 状态 |
|---|---|---|
| 主工作区未提交改动（M0-2 关账 @2026-10-05） | `--mtp-timing` GLM 三段结算 = **G4a 留、补完并入** `c252713`；indexed prefill / DSpark draft trace env = `c44b87c`；`SPEC_ROWS` env → 见 D 节关账行；`GLM53_ROUTED_MPP_PACKED` → 见 D 节删臂行 | 主树 `git status` 干净；SPEC_ROWS A/B 已随 M0-4 窗关账（中性+logits 等位，维持默认关） |
| p42-attn-glue（M0-3 定夺 @2026-10-05） | 分支 `p42-attn-glue` @`8392281`（WIP archive），worktree 已弃 | **弃臂存档，不合入**：位等价复核实测（M5 Max，带 `DS4_GPU_TEST_V41_FUSIONS` 强开）matvec_bf16 对 Q8_0 5120→1280 逐字节不等、MoE 融合臂 probs 契约同红；M3-Ultra-only admission 即为本险设，位契约优先于 ≈中性。证据 `speed-bench/M0/m0-3/`，M3 Ultra 上位验证通过前勿重试 |
| p3-io-diag | worktree `~/ds4-io-diag` | 7 行诊断补丁，可留可弃 |
| PF-6 上游 PR 移植 | 分支 `pf6-upstream`（基于 origin/main `0aaea5a`，提交 `71268a1`，已推 fork）；worktree `/tmp/pf6-pr-base`（重启即失，重建见杂项） | PR 分支自测 41k +5.2%（364.1→382.8）、锚点确认（f32 router 0 调用、cvt=42、上游 wrapper 无门控故 M3/M5 默认生效）；**已开 [antirez/ds4#1177](https://github.com/antirez/ds4/pull/1177)（OPEN）**，质量数字引用 `05cf0f3` 实测并注明测量树 |

### 本地保留但**未取用**的上游 PR（评审存档，非依赖）

- `pr-710`（live-prefix rewind 上 Flash/Metal；与本地 `64b6461` 域不同）、`pr-856`（thinking 通道空白符 cache miss）、`pr-1058`（tool 会话 KV checkpoint 键改 client-visible transcript + 重启恢复冒烟）
- `up/pr/` 旧 ref：**758**（M5 indexed prefill——被 `64b6461` 替代）、**1034/1035**（Dango233 SSD decode/Engram——Engram 并行读已被 fork 自身覆盖）、**1060**（indexer radix-select）、**1073**（Tarjei 54-commit 大 PR）、**1152**（GLM 参数类型——`e067c8f` 已自研覆盖）、**1153/1154/1155**（image-prefix snapshot / Qwen PLE prefetch / detached backend cache-miss）
- 这些 PR 后续若在上游合入，本地 `e067c8f`、`64b6461`、gather 系列都可能与之冲突，对拍时需按本清单 A2/B 逐项核对。

### 杂项处置（2026-10 定案）

- **M0-4 基线复核窗（2026-10-05，停产品独占，用户批准）**：14/14 腿绿；KPI 表全部复核成立（ITERATION_PLAN §1 新增 M0 复核列 + `speed-bench/M0/m0-4/notes.md`）。SPEC_ROWS A/B 关账：decode 与 [4,8) 行 append 均中性，step4/13 + step8/10 frontier logits 逐字节等（`m0-4/parity/*.sha256`）→ 维持默认关作 E1 种子。GLM MTP 首账 effective −19.8%（G4a 结算行）。方法学两教训（腿间门用组合可用量、swap 解析）已入 notes。
- **M0-5 上游复核（2026-10-05）**：`git fetch origin` 后 origin/main 仍 `0aaea5a`，`git cherry` 68 片全部未吸收；#1177/#1178 均 OPEN、零 review 零评论；本文件引用的全部 SHA（含 `cardi-upstream 5805843/4e48bb5`、`pf6-upstream 71268a1`）仍有效，无需刷新。窗口前照例复查。
- **M1-1/M1-2 落账 + macOS 26.5.1 Metal 捕获基底关账（2026-10-05，无窗口、零模型加载）**：M1-1 原方案（MTLGraph `captureScopeWithCommandBuffer:`）经三重取证在 macOS 26.5.1 (25F80)/Xcode SDK 26.5 上不存在（SDK 全目录零 graph 头、活跃 `AGXG17XFamilyCommandBuffer` 全继承链零 capture/replay selector、dyld 共享缓存 selector 字符串零命中、Apple 文档 404）；p42-attn-glue 分支亦无历史 capture 代码。用户裁决改道 MTLIndirectCommandBuffer，实现完成时以 7 变体裸 Metal 探针（A–G：concurrent/legacy、shared/private、threads/threadgroups、serial/concurrent encoder、inherit/绑定 组合）实测驱动黑洞：**任何带绑定的 CPU 录制（`indirectComputeCommandAtIndex` + `setComputePipelineState:`）直接 segfault 于 `AGXG17XFamilyIndirectComputeCommand setComputePipelineState:`（lldb 栈确认）**，唯一不崩的 inherit 形态继承 pipeline/buffer 不生效、dispatch 静默蒸发。**裁决（用户）：M1 捕获路线搁置**，骨架入库（`fde5b63`，default off，`tests/test_metal_graph_capture` 全绿），M1-3/1-4/1-5 窗口与 GLM ≥35 t/s 出口线作废重定（ITERATION_PLAN §1/§3 已注）。复活路径 = ds4_metal.m 三处 "M2-3 SUBSTRATE HOOK"（begin 建录制缓冲 / proxy recordDispatch 落命令 / execute_entry 执行），**动手前重跑 `git stash` 无关的探针矩阵**：/tmp 探针文件重启即失，重建要点=本条描述 + 崩溃点符号，每次 macOS 升级后先探后用。V4.1 侧 decode graphs（V4.1 islands 共用 supported()）同步搁置——env 关闭下 islands 恒 eager，无行为变化。**别把它当 bug 修**：`ds4_dg_execute_entry` 返回 NO 是裁决形态不是遗漏。
- **M3-1 标定工具链落账（2026-10-05/06，无窗口、零模型加载）**：[O] 编码半程完成（`ccc04b1`）——引擎快照 dump（env 默认关；无 env 时记录路径零新增分支/线程）、回放器分轮快照捕获（opt-in，默认行为逐行不变）、生成/标定双脚本（各 6/6 组 selftest 绿 + 合成件端到端跑通）、窗口 harness（§2 预检 + MAX_STEPS 切片 + swap 看门狗 + 独立 kv-disk-dir 不碰生产 `~/.ds4/server-kv`）。候选 session 已在本机 DSH session 库筛出 208 条 ≥8 轮 agent 型（最大 1179 步，标定窗建议 `MAX_STEPS=40..80` 切片）。**[W] 回放腿已跑（2026-10-04 窗，`run-20261004-163825` / `-164540`，均入 exclude 本地留存）——红卡待裁决**：60 步切片（omlx session-76700db1，688 步取头 60）两轮回放 60/60 绿、快照 60、prefix 复用 96–99%。纸面闸门 ≥25pp **不可达**：6959/1500/3000/800/300 experts 五档扫描，protected-K(60/75/85%) 增益 −1.1..+0.3pp（中预算为负、极紧预算勉强正），baseline 恒 0.64–0.90。根因：单会话语料内 expert 需求平稳（全 60 轮并集仅 ~1.5–1.7k 对）、LRU 近最优，跨轮频率保护没有可救的驱逐——25pp 门隐含的是**多会话轮换 + 小缓存**工况，单会话回放族结构上给不出。**裁决（用户，2026-10-04）：按 (a) 拒绝 ship，M3-1/R2a 就此结案**（存档未采纳选项：(b) 换语料复测、(c) 小缓存机型限定启用）。快照 dump/回放捕获/生成标定/harness 全部保留入库、默认关、零生产影响；若日后出现多会话轮换小缓存工况的证据，(b) 的复测路径已备好。**联动：M3-2（保护位+空闲预热）/M3-3（验收）因失去 ship 需求与标定源一并搁置**（M3-2 的淘汰次序单测与 M3-3 的小 cache 压力腿配方随时可复活）。另记：harness `CACHE_EXPERTS` 只进纸面仿真器、不进 server（server 自动预算），注释已澄清。
- **M3-4 容量复扫（2026-10-04 夜窗，`speed-bench/M3/m3-4/run-20261004-171205/`，runner `0f7dba4`）**：auto(7822)/8000/9000 三臂 + 10000 预检门。9000（83.8 GiB）全域单调 decode steady +6.0..+7.8%（五 frontier 齐涨），append1524 +0.8%；8000 形状不稳（2K/4K +9.5/+8.6 但 8K+ 平/负，单腿噪声嫌疑）；**10000 在 128G 机不可选**（92.7 GiB 预算+工作集>物理，预检设计性不过——≥192G 机选项）。方法学两教训入 notes（purge-lag 3 分钟再闸、9000→8000 降序臂序）。**默认位未动**：按 M3-4 DoD 改默认需 KPI 两域不劣 + 用户裁决；候选=128G 机 auto 预算 7822→9000。**后续（2026-10-05）：默认位不动，手动档转正使用——`--ssd-streaming-cache-experts 9000`（server/CLI/ds4-bench 三处同旗，支持 N 或 NGB 形式，引擎保留 to-fit 缩减保护）；重载机器叠加场景自行让出 11 GiB 头部空间，GLM 同型窗慎用**。本轮为前向单腿、无 ABBA 复测，引用时注意。GLM 域容量互感未测（同 flag 影响 GLM streaming 窗内存画像）。
- **解码侧 draftless n-gram 投机：可行性 → 窗口实测 → 关账（`1c44caa` 研究 + `0e96c41` 实测，2026-10-05/06，见 `NGRAM_DRAFTING_STUDY.md` Outcome）**：**[O] 离线（零模型加载）**用 14 个最大会话（1.48M token / 36.0 万生成 token，`model=deepseek-v4-flash` 本模型产出）仿真：tokenizer 从 GGUF 元数据重建（vocab 129280 + merges，预切分镜像 `ds4.c::bpe_tokenize_text`），投机循环按“严格早于查询的最远 k-gram 匹配→草稿→最长前缀接受→前进 1+acc”重放（含防泄漏哨兵 0.362）。**[W] 窗口实测成本斜率 a（4 配置，`--verify-scan-ssd`，V4.1 Q2 / SSD-streaming / ctx 4096 / prefix 1024，n∈{2..8}，8 = `DS4_TP_BATCH_MAX_ROWS` 硬顶）**：cache 958 experts → D=61.28 ms、V(8)/D=5.779、a=0.683；**cache 6028 → D=42.95 ms（= 23.3 t/s，生产同档）**、V(8)/D=5.767、a=0.681；cache 8078 → D=41.68 ms、V(8)/D=5.928、a=0.704；6028 且 8 行同 token（专家并集下界）→ V(8)/D=5.663、a=0.666。即 **V(n) ≈ D·(1+0.68(n−1))**，且三条排除：①**非缓存伪影**（缓存 8.4× 变化只动 V/D 0.2%，而 D 动 47%）；②**非专家并集增长**（同 token×8 同斜率）；③**不可摊薄**（2 行 verify = 2.36 decode 步，比两步 decode 还差；n=8 时单行仍 0.72 步）。**结论**：对实测成本曲线做逐匹配长度 cap 寻优，最优策略只在 0.8% 的步上投机 1–6 草稿，**端到端 1.0001–1.0020×**；**结构性上界 1.387×**（oracle 起草者每步 7 草稿全对 = 8/(1+0.68·7)）——这同时解释了 MTP/DSpark 为何失败，故**投机解码在本引擎按“整类”关闭，别再开单个变体**。**方法论更正两处（`tools/ngram_cost_model.py` 为更正后模型；`ngram_accept_policy.py` 已标 SUPERSEDED，勿再引用其速度数字）**：①原“selective policy”表把条件 token 除以条件成本，丢掉了 ~90% 不投机的步——9.8% 步投机的策略在任何 verify 成本下上界 1.19×，故 1.70×/1.36× 是“投机步内”而非端到端；②成本模型按“接受行”而非“提交行”计费，与它自己 docstring 声明的 c(n)=1+a(n−1) 不符。另记两点：原索引把 14 会话汇成一张表（跨会话匹配生产拿不到），**逐会话 1.514 token/步**（非 1.653）；**16-gram 命中平均接受 7.8 token**，远高于原 k=4 索引所能见。**未采纳的复活路径**：草稿源模块（滚动哈希 n-gram 索引 + 门控，env 默认关）**未实现**——a≈0.68 下它是死代码；若日后某配置实测 a≤0.17（+10% 门槛）再落地，接入点测绘见 study（`ds4.c:86717` 加“无支持模型时的 n-gram 草稿”分支、`ds4.c:86593` 为单轮 verify 模板、`ds4_tp_spec_cycle` 为“显式喂草稿”的 API 形状），届时先重跑 `--verify-scan` 确认 a。**并推翻本线的立论前提**：原假设“decode 卡在每步固定 host 往返”被否——若如此，8 行 verify 应远低于 8 步、2 行应仅略高于 1 步；实测 2 行 = 2.36 步、8 行 = 5.77 步，**没有可摊薄的固定成本**。GPU ~30% 忙而单步 40 ms 是**延迟**特征（SM 等权重流量而空转），不是批量能收回的余量。故 decode 提速的靶子是**逐行权重流量**（专家缓存容量 / 装载调度），不是 host 侧工作；此结论不涉及普通 decode 路径自身的 host 开销（M0-4/E1 已测中性）。**附带实测（读数注意）**：`D` 的组内前后两次标定最大差 17%（6028 档 46.4 ms vs 39.5 ms），故 `D` 只按稳态读数用——958 experts 61.1 ms、6028 39.5 ms、8078 39.8 ms，即**缓存杠杆在 ~6000 experts 以下值 ~35%，6000 以上本轮分辨不出**（容量杠杆的定量结论仍以 M3-4 为准，勿引用本行）。`--verify-scan` 的内存预算口径：6028 experts 计划 74.47 GiB、8078 计划 93.47 GiB（9000 档生产同配需 ≈90.6 GiB 预算，本轮未跑）。
- **M4-3（G2f）NAX ≥2 GiB 寻址审计 + 五族可执行单元落账（2026-10-06，无窗口、零模型加载，`c9505e0`）**：上游 llama.cpp #28748 钉住失败模式——M5 tensor-API 的 tensor slice 字节偏移是 int32，`(slice - base) ≥ 2 GiB` 即回绕。**静态审计结论（分族，均 CLEAN）**：`ds4_metal_args_mul_mm[_id]` 全部 `nb*` 字段 uint64；direct_rhs 的 slice 全为 threadgroup 内 (0,0) 瓦片、设备侧指针 rebase `(i12/r2)*nb02` 走 uint64；mpp/packed 的 `offset0=(uint64)(im-tp_base)*nb02`，packed(EXPERT_ADDRESSES) 走 uint64 GPU 地址表（生产 `[buffer gpuAddress]` + `useResource` 声明）；mla/indexer 的行 rebase 全部 `(uint64)` 显式提升；`metal/dsv41.metal:311` keys slice 偏移被 keys 张量跨度封顶（393k ctx × 512 B ≈ 193 MiB）。**watch 项**：direct_rhs 的 `dst + im*N*M` 为 int 乘法，回绕需输出缓冲本身 ≥2 GiB/批——现生产不可能，留作日后输出批扩型时的检查点。**静态结论现已可执行**：五族单元各跑真实内核两遍（或单派发双批次），相同字节摆 `2 GiB + 64 KiB` 两侧、位相等断言，M5 Max 上 5/5 PASS；direct_rhs 另用全 1 操作数钉住边界批的绝对值（out==K）。测试侧两个踩坑记入产品知识：`f16_f32` 的 RHS 域是 **float32**（模板 T1=float，喂 half 得 1.0009 假值）；**地址表寻址绕过 setBuffer 绑定，编码器必须 `useResource` 声明 backing buffer，否则读回全零且验证层零告警**（生产 dispatch 本已声明，单元同步钉住此要求）。驱动带内存地板（默认合成可用 ≥24 GiB 才允许多 GiB 分配；窗口内加载模型后跑用 `DS4_NAX_2GIB_MIN_AVAIL_GIB=0` 交还给 §2 预检管辖）。**[W] 窗口复跑完成（2026-10-04 窗，V4.1 Q2 录制服务常驻 + 回放负载下，`DS4_NAX_2GIB_MIN_AVAIL_GIB=0`）：5/5 PASS——M1-6 计划行（G2f ≥2 GiB 单元搭窗实测执行）同步按"全绿"出口结案。**
- **V4.1 decode 瓶颈定位（2026-10-06 窗口，`168df92`，见 `V41_DECODE_BOTTLENECK.md`）**：四配置实测 + 三个新仪器（`--verify-scan-ssd` 现用 `proc_pid_rusage` 记**进程自身**读盘字节、`--stage-timing` 支持 streaming 域、`--weight-inventory` 按显式张量名分组）。**结论一：盘完全不参与**——稳态每 token 读盘 **0.8 MB**，对 9.78 GiB 权重流量是 **0.02%**，专家缓存实际命中 ≈100%（958→8078 experts 只动冷相位、不动稳态）。**⇒ 加内存 / 加缓存 / SSD 调优 / 预取 / 减磁盘占用全部作废**（card H 的负结果至此有了机理）。**结论二：每 token 读 9.78 GiB / 15.816 G 权重**——routed experts 2.224 GiB（8.494 G，Q2_K）、attention projections 2.024 GiB + output 2.988 GiB（5.065 G，**Q8**）、shared expert 1.401 GiB（Q8）、head 0.655 GiB；早先“~3.5 GB/token”低估 3×，因为漏了 Q8 的注意力权重比 Q2 专家还多。**结论三（关键）：所有段都在 400–500 G 权重/s**（attention projections 440、attention output 503、shared expert 424、routed experts ~414 由减法得），而**字节率相差 4.6×**（dense Q8 535 GB/s vs Q2 gather 116 GB/s）——带宽受限的引擎做不到这一点。**⇒ decode 受“每秒处理的权重数”限制而非字节数，因此“降比特重量化”这条也死**（Q2 每权重已与 Q8 同速）。**唯一剩下的杠杆**：batch=1 被锁在矩阵-向量内核（约 1 权重/lane-op），prefill 用的 tensor units 需要 **B≥16/32/512 行**；而 `DS4_TP_BATCH_MAX_ROWS=8`——**所有测量（含 a=0.68）都在 ≤8 行、即慢路径内完成，8 行以上的批量化曲线从未被测过**，这是唯一还可能出阶跃的候选，需用户裁决（动 `ds4_tp.h` + `ds41_spec` 缓冲 + 位契约面；E0.3 那条“8 行以上契约不存在”是 **append** 路径的红灯，verify 路径的逐位性须重新建立而非假定）。**另两条待查线索**：①`ds4_gpu_dsv41_exact_admitted()` 在 `g_ssd_streaming_mode` 下恒 false，且 tuned 路径还要求 `ds4_gpu_device_is_m3_ultra()`——**M5 Max + streaming 结构性拿不到那些路径**；②GGUF 文件 340.6 GB，但张量合计仅 155.87 GiB（167 GB），**2.03× 无法解释**（不影响 decode，但值得弄清）。**已知局限**：`--stage-timing` 的 routed experts 段在 streaming 域直接 abort（“not covered by mapped model views”），故 ~414 G/s 是减法值；直接测该段是信任此数字的前提。
- **V4.1 decode 第二窗：批量化悬崖不存在（2026-10-06，`3833a39`，续上条）**：上条留的“唯一候选”= **a=0.68 可能只是小批量伪影**（此前所有测量都在 `DS4_TP_BATCH_MAX_ROWS=8` 以内，而 tensor units 要 B≥16/32/512 行）。本轮把上限临时提到 32、`--verify-scan` 扩到 12/16/24/32 行实测：**每行成本纹丝不动**——V/D = 2.19 / 2.74 / 3.46 / 4.84 / 5.94（8 行）/ 9.05（12）/ 11.43（16）/ 16.85（24）/ 22.39（32），大端拟合 `V(n)=0.451+0.686n`（≈18 ms 固定批开销 + 27.6 ms/行，单步 40.3 ms），**0.69 的比值从 8 行平到 32 行**。⇒ **投机解码的死因比原先更强：它不是小批量伪影，加大 verify 也救不回来；唯一候选关闭**。上限已**还原为 8**（全局抬高会同时放宽 9..32 行的批路径 = E0.3 登记红灯的地界，而这个实验没换来任何东西）；`--verify-scan` 改为**跳过**超上限的行而非钳制，避免把上限当成测量结果报出来。**同轮新增的关键对照**：`q_b` 投影（Q8、1.68 G 权重、40 层）单行 3.25 ms、边际行 2.40 ms = **700 GB/s**——**这台机器在 dense matvec 上至少能跑 700 GB/s**，而 routed expert 路径的 116 GB/s 差 6×，却占 decode 步的一半。**该段为何仍测不了（已写清机理）**：streaming 域下 routed MoE 不由逐层 encode 派发，而是由 `ds41_graph_step` 的**主机循环**逐层回读 selected ids 并 page 专家，故任何 encode 侧调用（`routed_moe_one_tensor` / `ds41_moe_partial` / `ds41_moe`+`after_moe`）都只报 ~0.3 ms（活儿排在别处）；raw `one_tensor` 是 resident 路径、绑 whole-map 视图，streaming 不映射故 abort。**⇒ 下一个真问题是：把“gather 内核慢”与“喂它的主机 paging 循环慢”分开**（card I 只拿 +2.7~3.2%，暗示回读不是主项，但本轮没有任何测量能分开这两者），这需要换仪器而不是再修这个。
- **续查（`2148647`）：routed 派发不在本层同一个 command buffer 里——两者按构造不可分**：`--stage-timing` 第 4 段此前 abort 的真因是 harness 把 `ds4_gpu_routed_moe_one_tensor` 最后一个参数写死成 `true`；该参数是 `force_resident`，真实调用在 `ds41_moe_partial` 里传 `!g->streaming`（`ds4.c:41095`）。改对之后 abort 消失，但该段报 **0.33 ms/token（40 层全算）= 8 µs/层**——2.224 GiB 专家权重不可能在这个时间里搬完，**⇒ encode 路径根本不是 routed 成本所在：gather 是在另一个 command buffer 里派发的，等主机为它准备好状态之后**。这同时说明：任何“按顺序单独调用 encode helper”的隔离尝试测到的都是一张未填充的地址表，所以第 4 段永远测不出真值。交错点已定位：`metal_graph_encode_decode_layer_phase`（`ds4.c:24104`，逐层 decode 编码器）在 `ds4.c:26252` / `ds4.c:26765` 调 `metal_graph_decode_set_hash_selected_override`（`ds4.c:23101`，paging）与 `metal_graph_decode_selected_readahead_override`（`ds4.c:23240`，readahead）。**⇒ “gather 内核慢”与“主机 paging 慢”在当前代码里按构造不可分**（paging 是派发的前置条件，两者在 CB 级别交错），要分开必须**插桩那个循环**，而不是隔离内核。本轮 routed 数字仍是减法，但**对照的六段全部走真实 decode 路径测出**（attention projections 4.63 + core 3.26 + output 5.98 + shared 3.37 + hc/norms 1.76 + head 1.20 = 20.20 ms / 40.32 ms），比上一轮的四段减法扎实。
- **V4.1 decode 第三窗：routed 那 ~20 ms 既不是 gather 内核也不是主机 paging 循环，而是逐层排空（2026-10-06，`7385e6f`，续上两条，见 `V41_DECODE_BOTTLENECK.md` §8）**：上一轮判"两者按构造不可分"——对真实路径成立，对内核不成立：只要**先**把地址表填好就能单独计时。新仪器 `--routed-split-ssd` 跑 72 步真实 decode（顺带填满专家缓存），读进程自己写的 per-(layer,expert) 命中快照，取**本次 decode 真正选中的** 6 个专家逐层绑好，再计时派发（4 次取最好，单位 ms/token）：**A gather only 单 CB = 6.76（353 GB/s）**；B 每层 flush = 7.05；**C A + 每层 `begin_selected_load` = 6.75（−0.01）**；E 只做每层排空、不派发 = 0.87；**D gather + 每层排空（= 引擎实际形态）= 13.62（175 GB/s）**。routed 每 token 搬 2.22 GiB（6/384 专家，9.49 MiB/个：gate 2.90 + up 2.90 + down 3.69），故 **A = 8.494 G 权重 / 6.61 ms = 1285 G 权重/s——全引擎最高的权重速率，是 dense Q8 `q_b`（700 G/s）的 1.8×**。此前"116 GB/s、比 dense 慢 6×"是**按字节**读一个**按权重**受限的引擎得出的，单独量出来 Q2 gather 才是快的那个。**⇒ C 回答了本轮的问题：主机 paging 循环代价 −0.01 ms，免费。** 回读拷贝同样免费（64 步 `read_avg 0.823` 中 `copy_avg 0.000`，全部是 `sync_avg`，即 `ds4_gpu_end_commands()`）。**真正的成本是那次排空**：`ds4_gpu_routed_moe_one_tensor` 需要把 6 个专家 id 拿到主机侧才能选缓存槽，于是逐层 commit+wait+重开缓冲——**每 token 40 次**——D−A = **6.86 ms**，几乎正好等于内核自身的 GPU 时间（排空把它从重叠变成完全暴露）。`DS4_METAL_DISABLE_V41_DECODE_QUEUE` 与 `ds41_graph_step` 的逐层 `flush_commands` 正是为避免这个往返而存在，被派发内部的排空抵消。主机侧佐证：`DS4_METAL_V41_DECODE_HOST_PROFILE=1` 报 **43.401 / 43.01 ms 全在层循环里**，每步 1969.3 次 compute encode + 139.1 次 blit。**routed 桶（≈20.13 ms）的分解**：内核 6.6 + 逐层排空/丢失重叠 6.9 + 专家缓存未命中 6.0（386 次装载/64 token，每次 ≈1.0 ms = `load_prepare` 0.385 + `load_pread` 0.611 + readahead；`bind_avg` 0.149 ms × 2560 次调用，命中 15% 未命中时涨到 ≈1.6 ms）+ router/胶水 ≈0.6。**注意未命中项是主机时间且是逻辑 pread**：`miss_pread` 57 MiB/token，而进程自身 `ri_diskio_bytesread` 仍是 0.8 MB/token——页缓存吸收掉了，§1 的"盘不参与"依旧成立，但**通往盘的那条路径本身要钱**。**裁决建议（未改任何默认）**：①**别动 gather 内核**——它是全引擎每权重最快的，且只占桶的 1/3；②**别动 paging 决策**——arm C 证明其为零；③真正的两个靶子是**逐层回读排空（6.9 ms）**与**未命中路径（6.0 ms）**；前者树里已有两份实现把回读移出关键路径（通用 decode 路径的 `metal_graph_selected_async_load_*` worker，`ds4.c:23532`，V4.1 步未接；card I 的 GPU 绑定 service 线程 = PR #1178，实测 +2.7~3.2%），但**两者都保留排空本身、只把主机移出关键路径**——这与 card I 的小收益一致，也说明 6.9 ms 是这类修法的天花板。④`DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` 看着像"免回读"的答案（`selected_ids_available=false`、不排空），实际不是：它绑**整层**专家张量（3.64 GiB/层）走 model map，是 resident 形态，正是 streaming 映射不了的。本窗 D = 43.01 ms（23.25 t/s），同配置前两窗为 40.2–40.3 ms；各臂只在同一轮内互相比较。内存口径：82 GiB / 8078 experts，计划 93.47 GiB，实测峰值 resident 75.1 GiB，swap 全程未涨。
- **V4.1 decode 第四窗：排空在真实步里的净损失 = 6.3-6.5 ms（消融实测，非减法）；未命中项 6.1-6.8 ms 同轮坐实（2026-10-06，`c1d7192`，续第三窗，见 `V41_DECODE_BOTTLENECK.md` §9）**：本轮用**一次性消融探针**（stale-ids 回放，`DS4_METAL_V41_PROBE_STALE_IDS`，输出必然错误、默认关、**用完已删、从未提交，勿再引入**）给第三窗的两项嫌疑定价：探针把首 token 每层真实 ids 冻结成陈旧集，经与 override 相同的 `ds4_gpu_stream_selected_ids_prepare` 路径让主机与内核读同一份 ids；`=2` 保留每层排空（真实回读落 scratch 后丢弃）、`=1` 跳过 `end_commands`/`begin_commands` 对（CB 跨层保持打开）。两腿负载完全同构（每步 1969.3 compute encode + 139.1 blit、bind_avg 0.002、窗口 load_calls=0、2560/2560 全 resident）——**唯一变量是每 token 40 次排空**。读数（82 GiB/8000 experts，64-token 窗口 delta，同轮三腿）：live **layers 44.962 / D 44.74**；PROBE=2 **38.900 / 37.94**；PROBE=1 **32.417 / 31.63**。⇒ **排空净损失 Δ=6.48 ms（layers）/ 6.31 ms（D）**：每层主机在 `ds4_gpu_end_commands` 等 0.848 ms（33.9 ms/token），其中 6.5 ms 是完全暴露的串行化（GPU 空转等主机重开缓冲再编码）——第三窗 D−A=6.86 的孤立值**没有被空 CB 几何放大，真实步输掉同样多**。**判定表第一行命中（Δ=6.48 ≥ 4 ms）：排空确为净损失，且 card I 没把它拿回来**——card I 的 +2.7~3.2%（≈1.3 ms）只有这 6.5 ms 的 ~20%：它把主机移出关键路径，但 CB 边界处的 GPU 气泡还在。⇒ **T3 具备立项条件，但默认不做，动手前必须用户裁决**（形状 (b) 动 E0.3 红灯绑定面）。**T0 坐实**：live 腿窗口 `load_calls=386`（**6.03/token**，2174 all_resident + 386 mixed，cache_missing_avg 0.17），冻结 ids 后窗口命中 100%、`bind_avg` 0.150→0.002、整条 prune/pread 机器静默 ⇒ **bind 与 6.0 ms 未命中项是同一笔账**（bind×40=5.95 ≈ load_calls×~1.0 ms），live−stale = 6.1-6.8 ms 即未命中成本。**⇒ T2 立项（prune 计时探针仍在计划内，但已知道上限就是这 6 ms，且 `load_prepare` 未归因残差 ~0.29 ms/次是其子项而非整项）；T3 不满足"未命中≈0"的关闭条件、但也不能按 6.86 立项——它面对的是"已有实现只拿回 1/5"的新事实，动 E0.3 红灯绑定面之前需用户裁决（默认：不动）。** 本窗 live D=44.74（上窗 43.01，同配置轮间差在 7% 带内；只同轮比较）。内存：82 GiB/8000 experts，窗口内 resident 峰值 84.4 GiB（cleanup 口径 75.0），swap 全程未涨。探针腿 log：`/tmp/t1_{base,drain,nodrain}.log`。
- **V4.1 decode 第五窗：未命中路径账本闭合——3.55 ms/token 是排空后串行 pread 的墙钟；readahead 净 −0.07 保留；两个 prune 结构性无罪（2026-10-06，`4d1b792` + 插桩 `b203362`，续第四窗，见 `V41_DECODE_BOTTLENECK.md` §10）**：先入裁决：**T3 经用户裁决于 2026-10-06 关闭**（6.5 ms 排空空损接受；形状 (b) 需动 E0.3 绑定面，不值）。T2.0 给 streaming-expert 计时系统加 5 组 env-gated 计数器（residency 扫描 / prune_layer / prune_global 时间+驱逐+扫描量 / pread 池提交+同步回退 / load wait 拆 overlap·block；默认关、零行为变更）。同配方三腿（82 GiB/8000 experts、64-token 窗口 delta、swap 全程未涨）：**default** per-load 0.944 ms = prepare 0.353（readahead 1287×0.084≈**0.28**——上轮"未归因 0.29"的全部——+ buffer 0.07 + 其余）+ **串行 pread 0.588**（9.66 MiB、3 task、16.4 GB/s）+ install 0.003；**readahead 关**→ prepare 0.078、pread 0.933，per-load 净劣化 0.07 ⇒ **readahead 自偿，保留开**；**pread threads=1** → 0.695，3 倍并行只值 +0.107（每线程 16-19 GB/s 与线程数无关）⇒ pread 受拷贝路径限速、非盘延迟，加线程/切块无效。**prune layer/global 各 2565 次调用、平均 0.000 ms、驱逐 0、扫描 0**——两嫌疑当场判死：`effective_cap(layer, 384, 6) = 384` 且每层缓存按 expert 索引，prune_layer 驱逐循环结构上无法执行；稳态 `entry_count == budget`，prune_global 每次走廉价返回，驱逐走 buffer 复用路径、prune 轮不上。双记账吻合：`bind_avg 0.142 × 2560 = 5.68 ms/token` = 同一笔 5.7（load 侧）记入 bind，与第四窗 live−frozen 6.1 一致。**结论：缓存管理侧全部开关合计挪不动 ≤0.11 ms/load；这 6 ms 在位契约+bind 面不动的边界内无可调项——3.55 ms 是 ids 排空后才拿到的 pread 串行墙钟（重叠需 split masked gather = 改累积形状（E0.3 家族）且只影响 15% mixed 层，或回到已关闭的 T3 形状）。唯一诚实的下一步不是代码：10000 experts 生产规格未测——若其热集闭合，生产上未命中项为 0，T2 整体搁置。** 腿 log：`/tmp/t2_{attr,no_ra,thr1}.log`。待用户裁决（10000-experts 腿 / split 实验（红线旁，需 A/B 位等）/ 关闭 T2）。
- **V4.1 decode 第六窗：10000-experts 生产规格实测——128 GB 机器锁得住（9861 条、91.48 GiB、零失败、swap 全程未涨），D 43.7→40.8，未命中减半但热集仍存活（2026-10-06，`b45352c`，纯测量零代码，见 `V41_DECODE_BOTTLENECK.md` §11）**：用户选 1，跑 `EXPERTS=10000 GIB=100` 同款 64-token 窗口腿（log `/tmp/t2_10k.log`）。预算侧实况：请求 10000，harness 先把 100 GiB 压到 99 GiB（graph 工作集压力闸）⇒ 实得 **9861 条 @9.49 MiB = 91.41 GiB**，mlock 1555 ms 全锁成功 **failed=0**。窗口对照（同轮同配方）：loads/token **6.03→3.13**、全 resident 层 **85%→92.2%**（2360/2560）、窗口缺失专家 435→207、未命中项 **5.7→3.85 ms/token**、bind_avg 0.142→0.098、**D=40.82（8k 家族 43.7-45.5 之下 ~7-8%，本轮实测 t/s 24.50）**。代价藏在单次成本里：pread 0.588→0.847 ms（92.7 GiB 缓存 mlock 后挤压页缓存，读落到真盘的比例升高），但次数减半净赚 ~1.85 ms/token。**裁决建议：128 GB 机器生产按 10000 experts 跑，白拿这 7-8%；热集在本长 prompt 上仍 >9861（3.1 次/token 轮转未归零），剩余 ~3.9 ms 未命中 + ~6.5 ms 排空只剩两条已关闭形状（T3、split masked gather）可达 ⇒ 建议就此关闭 T2。** **用户裁决 2026-10-06：T2 关闭（`9822961`）**——生产按 10000 experts 跑；计划四任务全部下落：T0 坐实未命中路径、T1 定价排空 6.5 ms、T3 关、T2 收割 -7~8% 后关账。再大装不下本机。仪器注：该腿 host profile 的 layers 82.666 是 before/after 双臂混窗重复计数（D 与 bind 记账自洽），读 D 不读它。
- **本机新增既有红项（2026-10-06 stash 对照定案，= HEAD `cfe5932` 非回归，别追）**：`tests/test_deepseek41_fusions`（matvec_bf16 与 Q8_0 位不等，与 M0-3 p42 裁决同源的平台性红）与 `tests/test_deepseek41_engram_admission`（step-reader 准入断言 `line 56: calls[i]==(i==reader)`）在**未含本会话改动的 HEAD 重建件**上复现同 rc=1、同输出；`./ds4_test` 全量在默认模型缺失时于 `qwen4-prefill-checkpoints` 断言退出（`ds4flash.gguf` 链接目标 5 月起缺失的环境资产红，历史窗均显式 `DS4_TEST_MODEL` 运行）。零模型验收口径 = `ds4_test --server` + 非模型测试二进制 + `test_metal_graph_capture` + `test_metal_ssd_experts` + `test_metal_nax_tensor_units`（c9505e0 起），全部绿。
- **既有红项补登记（2026-10-09 pick #1192 对照定案，非回归，别追）**：`tests/test_deepseek41_metal` 在 `line 1432: attn_flags == 0` 断言失败——`check_tp_attention` 的 `{rows=1, selected=0}` 用例期望 NAX tile 不派发（n<32），实际 engagement 位非零；在 pick 前树 `50f950b` 的独立重建件上复现**同行同断言**，与 top-k/argsort 无关（该套件此前各 topk/attention 数值断言全过）。属 NAX tile 门控/标志残留问题，域邻近本地 `64b6461`（M5 默认开），未动。
- 一次性噪声已入 `.git/info/exclude`（纯本地、不随任何 PR 携带）：`pic.jpeg`、`start.md`/`start.txt`（server 启动备忘）、`tests/*.o.tmp`、`tests/test_metal_rewind`（构建残留）、`tests/test_metal_graph_capture`、`tests/test_metal_nax_tensor_units`（构建残留，2026-10-06 补）、`.codegraph/`、`.cursor/`
- 本文件与 `AGENTS.md` 已纳入 git 跟踪。**PR 纪律**：PR 分支一律从上游 `origin/main` 切、只 cherry-pick 目标修复提交；不 merge 本地 main、不 `git add -A`——docs commit 不在 PR 范围即不会携带
- `/tmp/ds4-main`、`/tmp/pf6-pr-base` 对照 worktree：重启即失，需按 BASELINE §13.2 重建流程

## E. 全仓代码对账（2026-10-04）：A/B/D 之外的代码

对 main、全部分支、4 个 worktree、stash、未跟踪文件逐一枚举后，**main 的 58 个提交与 A/B 全部对上账**（上游作者 34 片、本地 23 片、merge 1）；`pr-*`/`up/pr/*` 均为 PR 原文存档（D 已列），`feat/glm53-*` 两分支仅多 3 个提交 = `2fe096e`/`62bbc0b`/`e067c8f` 的原件（无独立代码），`fix/*` 指向 main 上的提交，`/tmp` 两对照 worktree 无独有代码。真正的"账外代码"只剩以下三项（stash 三项已裁决丢弃）：

| 项 | 内容与规模 | 状态/建议 |
|---|---|---|
| `stash@{0}`/`{1}`/`{2}` | 0/1 = `22193b5`、`e067c8f` 的已应用残留；2 = `ds4.c` +4 `vocab_size` 缺失回退 129280（unsloth GGUF 适配，未入 main） | **2026-10-04 用户裁决全部丢弃**，`git stash clear`；patch 备份在 `.git/LOCAL_BACKUPS/stashes-2026-10-04.patch`，日后要捡回 vocab 回退从这里取 |
| card H 实现（`~/ds4-cardh`，**`5573d62` @ `p4-cardh-early-load`**，基线 67fe336，ds4.c +63 行） | `DS4_METAL_ENABLE_V41_MOE_EARLY_LOAD` 门控（默认关）：router 后 GPU 事件 + worker 线程读 6 id/预暂存 pread，与共享专家编码重叠，`set_selected_override` 免逐层回读；`.cardh-ab/` 产物未入库 | **2026-10-04 关账：bit-exact 全过（CLI 贪心 A/B 逐字节一致、logprob ON==OFF、dspark 两测 ON==OFF、专家缓存 4/4），但速度负结果**：ABBA 2×(2K,32K)×512tok steady OFF 23.83/22.54 vs ON 23.51/22.48（−1.3%/−0.3%，组内噪声 4%）。此前"layer 0 failed"是旧二进制未重编 + flush 轮换 CB 后无条件 begin 的 bug，均已修；parity 0.36 漂移证实为存量流式行为（env OFF 相同）。14.6 的"19ms=可重叠 GPU 等待"假设证伪（GPU 30% 忙，等待近零返回；真主犯是主线程 CPU 簿记量）。裁决见 BASELINE §15；**不合入、勿重试此形态** |
| card I / I-1 实现（`~/ds4-cardi`，**`376687e` @ `p4-cardI-graph-capture`**，基线 29a4474，ds4_metal.m +595/−67，ds4.c/头文件/内核零改动） | `DS4_METAL_ENABLE_V41_MOE_GPU_BINDING` 门控（默认关，单卡+streaming+单 token）：routed MoE 走现成 addr-table 入口吃 GPU 侧 `g->selected` ids，新 service 线程等 CB 的 ids 事件后自主读 ids、装载缺失、标 inflight、回填 6 槽并 prune，CB 等 ready 事件再 dispatch——主线程每层不再回读；`.cardi-ab/` 产物未入库 | **2026-10-04 I-1 收账：bit-exact 全过 + 正向小收益**。CLI 贪心 A/B 逐字节一致（默认 env，含 masked 族）、logprob ON/OFF 全日志 0 差异、dspark 两测 ON==OFF、专家缓存四模式全过；ABBA 2×(2K,32K)×512tok OFF 22.44/21.82 → ON 23.17/22.40（**+3.2% / +2.7%**，四比较点全不重叠）。两条入档硬知识：①service 线程标 inflight 必须用 arm 时抓的 CB 序列而非环境序列（否则 drain 窗口=无保护→同命令三代输出且零日志）；②经地址表取权重的 dispatch 必须 `useResource` 标到槽 buffer（绑定模式改为标 19 个 slab，标上即 3/3 复现）——**I-2 的 capture 形态必须带上这个标记**。另补存量隐患：`take_reusable_batch` 的 batch-reuse 分支缺 `on_service_thread()` 守卫（四处 wait_inflight 中唯一没有的）。PR 基线（origin/main `0aaea5a` + `5805843`/`4e48bb5`）单独实测：gate1 逐字节一致、logprob ON==OFF、专家缓存四模式过，ABBA OFF 18.11/17.03 → ON 18.48/17.54（**+2.0%/+3.0%**，PR 树没带 fork 的 decode 融合线所以绝对值低）——已推 fork `cardi-upstream` 并开 **[antirez/ds4#1178](https://github.com/antirez/ds4/pull/1178)**（+599/−68 单文件，零 cherry-pick 依赖：流式缓存 `9ba160a`/地址表内核/服务线程标记 `519c4d8` 全在上游）。裁决见 BASELINE §17；**在途，I-2 判定 go，env 保持默认关** |
| 未跟踪工具脚本 ~1.2k 行（M0-6 关账 @2026-10-05，`ecfcb3b`/`8606206`） | `gguf-tools/deepseek41_dspark_convert.py`(349)、`speed-bench/{build_dspark_support_gguf.py(228), jigsaw_to_ds4_dspark.py(211), mtp_ledger_replay.py(160)}` = DSpark/MTP 资产管道；`speed-bench/{v41_m5max_streaming_ab.sh(66), v41_m5max_server_probe.sh(50)}` + `v41_m5max_ab/` 实测数据 = 基线方案台架 | 全部已入库 main：DSpark 四件套随 MTP 线裁决**封存备查**（≥256G 机重开时要用）；v41 台架 + CSV 为本尊（`ds4-io-diag` 等处仍是拷贝，清理时别认错） |

另：`CLAUDE.md` 旧孪生文档已于 M0-6（`8606206`）改为指针，AGENTS.md 为单一事实源；`tests/test_glm53_router_shared.c` 的测试旗标小补丁已随 M0-1 入 main（`2663de6`）。

## F. 2026-10-08 rebase 记录（吸收上游 9 笔 + 丢弃 3 笔）

**上游 9 笔**（10-05~10-07，antirez 直接提交，非 PR 合并；本轮起为 main 祖先）：`962ac19`（V4.1 CUDA attention + Spark RDMA + `ds4_engram.c` FP8→BF16 LUT）、`77570b4`（双机 CUDA TP；**同时改写 Metal 侧 DSpark 图路径**）、`43a91aa`（Spark 投机解码分发；重写 `metal_graph_encode_layer_ffn_batch` 等）、`63059b6`（原生 MXFP4 DSpark 专家；`tp_dspark_split_eligible` 放开）、`b3afc8e`（双 Spark 文档与发布检查）、`f0962e3`（传输失败后停止 TP 推理）、`b14bf62`（DSpark/ROCm 回归测试）、`d6b516a`（API 首选打分与备选顺序解耦）、`fc80bd6`（DSpark 响应边界不再全量重建）。**TP 协议台阶 14→15→20→21**。

**丢弃 3 笔**（同目标以上游为准）：`9c4cf2c`/`bca9985`（#1003 → `fc80bd6`）、`4db95ee`（B2 → `fc80bd6` 无 vision 排除）。

**冲突处置（117 步中 6 处冲突 + 3 处 skip）**：

| 步 | 提交（新 SHA ← 原 SHA） | 文件 | 处置 |
|---|---|---|---|
| 3 | `52c482a` ← `2db1152`（#1120） | `Makefile` | `clean:` 目标取并集（双方各加一行） |
| 16 | `4a84e65` ← `9309a29`（#1090） | `Makefile` | `clean:` 列表取并集（上游 `test_cuda_tokentile` 并入 glm 行） |
| 26 | `64b6461` ← `2ce4e40`（本地 NAX indexed prefill） | `tests/test_deepseek41_metal.c` | 并集：上游 `cases[]{rows,selected}` 结构 + 本地 NAX 开关 / `max_nax` 报告 |
| 27 | `2498e0f` ← `8963eae`（#1000 片1） | `ds4_server.c` | 保留 #1000 的早停 + 无条件回滚块，采用上游 `server_generation_rewind(resample,last_token)` 新结构 |
| 28 | `b5c4ce2` ← `e66f811`（#1000 片2） | `tests/test_session_state.c` | 并集：上游 Spark/split_reserve 测试（225 行，补 `}`）+ 本地文本续写测试（42 行） |
| 38 | `423c5c2` ← `ce97350`（#1089） | `ds4.c` | 上游 fast-path 行 + 本地 odd-target 块；丢弃 `dspark_rollback` 旧上下文，保留 `ds41_graph_rewind` |
| 29/30/115 | — | — | skip：`9c4cf2c`、`bca9985`、`4db95ee` |

**忠实性核验（本轮实测）**：
- 祖先：`git merge-base main origin/main` = `fc80bd6`；`git rev-list --count origin/main --not main` = **0**（上游全吸收）；本地领先 114。
- 排除 9 笔触碰的 73 文件后，`git diff backup/pre-rebase-20261008 main` **为空** ⇒ 其余本地文件逐字未变（无附带损失）。
- 编译：`make`（Metal 默认目标）**0 error / 0 warning**，产出 ds4 / ds4-server / ds4_test / ds4-bench；`make tests/test_session_state tests/test_deepseek41_metal` 均干净。
- 零模型单测：`./tests/test_session_state` **全绿**（含本轮合并的 #1000 文本续写测试）。
- **未做（待窗口）**：`tests/test_deepseek41_metal` 运行（本机有生产 server 驻留 ~84 GB、swap 7793/9216 MB，GPU 单测留到窗口）；`make cpu`；任何模型加载类回归；**上游 `77570b4`/`43a91aa` 改写的 Metal DSpark 语义需 M5 Max 窗口复验**（上游自认 "requires physical Metal and ROCm release QA"）。
- 回退点：`backup/pre-rebase-20261008` = 旧 tip `84f65f5`。
- `fork/main` 仍停 `3cfd0c0`，**待同步**（本轮未推）。
- **约定缺口（既有状态，非本轮引入）**：A 段 30 个 pick 中仅 13 个带 `(cherry picked from commit ...)` trailer，**0 个**带 `Upstream-PR:` 行——与 AGENTS.md 的提交约定不符；后续新 pick 按约定补，历史 pick 不回溯改（改了会破坏 patch-id 对拍基线）。
