# 上游 antirez/ds4 变更复核：2026-10-05 ~ 10-07（9 笔）

> 复核日期：2026-10-08，主工作区 `/Users/xieyongliang/ds4`，DSH 主会话（未派子代理）。
> 基准：本地 `main` = `84f65f5`；上游 `origin/main` = `fc80bd6`；共同祖先（merge-base）= `0aaea5a`（9-20）。
> 触发问题：GitHub 上从 "Commits on Sep 18, 2026" 起上游有大量提交；且本仓库有"PR 晚很多天才合并、日期看着偏旧"的惯例，需确认本地与上游的真实差集。

---

## 0. 结论摘要

1. **差集确实是 9 笔**，且是**按祖先关系**算出来的（与日期无关）：`git rev-list --count origin/main --not main` = 9；`git cherry` 显示 9 笔全无 patch-id 等价物（真未吸收）。
2. **Sep 18–20 那 11 笔早已全部吸收**（本地 main 的祖先链里就有它们），包括作者日期 9-14 / 9-17 的"晚合并"件。
3. **日期不可信，且不只是"旧"**：Oct 5 那 6 笔的作者时间戳与提交时间戳**完全相同**（`2026-10-05T14:17:41+02:00`），作者名是小写 `antirez`；后 3 笔用 `Salvatore Sanfilippo`。这一栈的日期被重写为落库时刻，原始 PR 作者日期被抹掉。→ 判断先后只能靠祖先关系 / patch-id。
4. **反直觉：这 9 笔虽以"双 Spark CUDA TP"为名，但大量改动落在 Metal 侧与共享代码**（77570b4 的 ds4.c 842 行里以 `metal_graph_*` / `ds4_session_*` 为主；43a91aa 重写 `metal_graph_encode_layer_ffn_batch` +154/−114、新增测试钩子 `ds4_test_engine_placement` +233）。不能按"CUDA-only、macOS 无关"整批跳过。
5. **TP 线协议台阶 14 → 15 → 20 → 21**：本地是 14；77570b4 推到 15，43a91aa 推到 20，fc80bd6 推到 21。版本不符即握手拒绝，TP 相关笔不能部分取、两侧二进制必须同版本重建。
6. **头号撞车：`fc80bd6` 与本地 `#1003` pick（`9c4cf2c` / `bca9985`）+ B2（`4db95ee`）**——同一套 `spec_frontier_*` 机座上的两份平行实现；上游那份补掉了本地已登记的残留缺口（`pos == start` 无 logits），并支持 TP 协调；上游那份**没有 vision 排除**，B2 的 bug 在结构上不存在。
7. **账本分叉**：`LOCAL_INVENTORY.md` 工作区版与 HEAD 版各有对方没有的内容（见 §8），吸收任何一笔前需先定真态。

---

## 1. "本地与上游只差 9 个 commit"的验证

### 1.1 差集（祖先关系，日期无关）

| 检查 | 命令 | 结果 |
|---|---|---|
| 上游是否被强推 | `git fetch origin --prune` | `0aaea5a..fc80bd6 main -> origin/main`（**快进**，无重写） |
| 差集计数 | `git rev-list --count origin/main --not main` | **9** |
| 共同祖先 | `git merge-base main origin/main` | `0aaea5a`（9-20） |
| 基座是否真在本地 | `git merge-base --is-ancestor 0aaea5a main` | **真** |
| patch-id 等价性 | `git cherry -v main origin/main` | 9 行**全为 `+`**（无等价物，真未吸收） |

9 笔（旧→新，均为 antirez 直接提交，非 PR 合并）：

```
962ac19  Improve V4.1 CUDA attention and Spark RDMA exchanges
77570b4  Add two-host CUDA tensor parallelism for DeepSeek V4 Flash
43a91aa  Distribute and accelerate Flash speculative decoding on Spark
63059b6  Optimize Spark attention and support native MXFP4 DSpark experts
b3afc8e  Document dual-Spark inference and its release checks
f0962e3  Stop tensor-parallel inference after a transport failure
b14bf62  Repair DSpark and ROCm regression tests
d6b516a  Score API top choices independently of alternative order
fc80bd6  Avoid full context rebuilds at DSpark response boundaries
```

### 1.2 Sep 18–20 那批：逐笔确认已在本地

`git merge-base --is-ancestor <c> main` 全部为真：

`0719a0b`（作者 9-14，提交 9-19）、`6c00e2d`、`d9bc45d`、`aafc65b`、`077a257`、`5994f06`、`4067320`、`25334fe`、`06f7135`、`6687c6f`；更早的 `ecf7a93`（作者 9-18）、`63bbba4`（作者 9-15，提交 9-18）亦在其中。

### 1.3 日期不可信的两类证据

**类型 A：作者日期滞后于落库日期（PR 晚合并的常态）**

| commit | 作者 | 提交 | 滞后 |
|---|---|---|---|
| `0719a0b` Improve Metal router softplus accuracy | 09-14 22:21 | 09-19 23:00 | 5 天 |
| `63bbba4` Advertise the loaded GLM model id | 09-15 11:36 | 09-18 15:54 | 3 天 |
| `aafc65b` `077a257` `5994f06` `4067320` `25334fe` | 09-18 | 09-19 05:27 | 1 天 |

**类型 B：日期被重写为落库时刻（Oct 5 那一栈）**

6 笔（`962ac19` → `f0962e3`）作者时间戳 = 提交时间戳 = `2026-10-05T14:17:41+02:00`，且作者名是小写 `antirez`；`b14bf62` 起改用 `Salvatore Sanfilippo`。**这一栈的原始作者日期不可见**——所以"按 GitHub 上的日期分组"会把它们与真实作者时间混在一起。

### 1.4 文件级完整性（没有"本地丢了上游文件"）

对 9 笔触碰的 73 个文件按"基座 `0aaea5a` 有没有 / 本地有没有"分档：

| 档 | 含义 | 数量 |
|---|---|---|
| A | 基座有 + 本地有 → **冲突面** | 43 |
| B | 基座有 + 本地无 → 本地丢了上游文件 | **0** |
| C | 基座无 + 本地无 → 这 9 笔新增 | 30 |

B 档为 0 说明本地没有丢过上游文件（`ds4_gpu_copy.h` 属 C 档：它由 `43a91aa` **新建**，不是在改一个本地缺失的旧文件）。

**残留风险（本复核未覆盖）**：祖先关系只能证明上游提交"没被漏掉"，不能证明上游**内容**没有被后来的本地提交改回去（revert）。本地相对基座的删除量为 `ds4.c −293` / `ds4_server.c −113` / `Makefile −8` / `tests/test_deepseek41_metal.c −5`——这些正是本地工作**改写**上游代码的位置，也即 rebase 冲突点。

---

## 2. 九笔逐条

| commit | 主题 | 规模（文件/±行） | 平台相关 | 本地冲突面 | 一句话内容 |
|---|---|---|---|---|---|
| `962ac19` | V4.1 CUDA attention + Spark RDMA | 13 / +557−78 | CUDA 为主；**ds4_engram.c 跨平台** | ds4.c、ds4_engram.c、ds4_tp.c、Makefile、.gitignore | 批量 causal attention 选择、FP8→BF16 Engram 转换 LUT、融合共享专家激活、RDMA 大消息 + 注册缓冲；TP slab 优先走 CUDA 共享 slab（Metal 侧由 `!defined(__APPLE__)` 排除） |
| `77570b4` | 双机 CUDA 张量并行 | 27 / +2444−595 | **大量 Metal 侧 + 共享** | ds4.c、ds4_tp.c/h、ds4_cuda.cu、cuda/mmq/*、Makefile、tests | 跨两台 CUDA peer 驻留 Q2/MXFP4；切分 attention/共享专家/短投机；**跨后端对齐 DSpark 目标位置、confidence 输入、滚动历史**（延伸 Donato Capitella 的 `394865c5`） |
| `43a91aa` | Spark 上的投机解码分发与加速 | 35 / +9637−596 | **大量 Metal 侧 + 共享** | ds4.c、ds4.h、ds4_gpu.h、ds4_tp.c/h、ds4_cuda.cu、Makefile | 起草/验证词表头跨 peer 切分、有序 RDMA 命令邮箱、接受前缀历史跨 restore/skip 保留、MXFP4 专家均衡、Q8 形状特化；修 active-stream 次序、成对投影行覆盖、indexer top-k 阈值 |
| `63059b6` | Spark attention + 原生 MXFP4 DSpark 专家 | 17 / +1818−101 | CUDA 为主；**ds4.c 判定跨平台** | ds4.c、ds4_gpu.h、ds4_help.c、gguf-tools/*、ds4_cuda.cu | F16 投影拷贝合并、Q8 词表特化、indexer 打分分块、top-k 压缩排序；**放开 MXFP4 drafter 专家**（不改默认支持格式）；修 top-k 压缩分歧 |
| `b3afc8e` | 双 Spark 文档与发布检查 | 6 / +277−49 | **纯文档** | 无（6 文件本地均未改；仅 docs/SPECULATIVE_DECODING.md 有本地 +13 的 #1003 痕迹） | 记录 Q2/MXFP4 网络 TP、DSpark 默认值、双 peer 配置、实测性能口径；原生 drafter 转换；扩展 QA：物理 RDMA、内存归属、active-stream、图检查、官方续写打分、上下文边界、取消与回退 |
| `f0962e3` | 传输失败后停止 TP 推理 | 5 / +141−10 | 共享（ds4_tp.c） | 无（ds4_tp.c 本地未改） | 链路失败后拒绝命令与各档门；失败的命令发送与 verify 窗口的 setup/completion 失败按致命处理，不再在未同步队列上重放；**恢复必须重连**；修"单侧 head-row 失败后把 peer drain 流量当模型部分和消费" |
| `b14bf62` | 修 DSpark / ROCm 回归测试 | 3 / +269−20 | 构建/测试 | Makefile（本地 +188−8） | HIP MMVQ 测试单独编译 + 复用既有 ROCm fold stub；EOS contract fixture 按 TP proposals / split verifier heads 更新，保留历史负控 |
| `d6b516a` | API 首选打分与备选顺序解耦 | 2 / +105−6 | **跨平台、零冲突** | 无（2 文件本地未改） | `api_top1_update`：取**报告的最大 logprob**（不再是 `alts[0]`）、并列最大值只计一次、不可映射的最大值直接跳过而不顶替低排名 token；覆盖次序/并列/损坏 token 字节 |
| `fc80bd6` | DSpark 响应边界不再全量重建 | 10 / +1122−14 | **跨平台（引擎+server+TP）** | ds4.c、ds4.h、ds4_server.c、ds4_tp.c/h、Makefile、tests | 保留最近 verifier frontier，响应结束在已接受块内时恢复有界前缀；**保存 frontier logits**使 `pos == start` 可直接采样；TP restore 走状态-only ACK 命令后再重放；不可用/过期快照回落到旧 rebuild 契约 |

**新增文件（30 个，全在 C 档）**：`ds4_gpu_copy.h`（`43a91aa`）、`tests/test_dspark_rewind.py`（597 行，`fc80bd6`）、`tests/test_server_speculative_stop.py`（247 行，`fc80bd6`），以及 27 个 CUDA 专项测试（`test_cuda_*`、`test_mxfp4_staged_cuda.cu` 等）。

---

## 3. 冲突面（本地相对 `0aaea5a` 已改的文件）

43 个 A 档文件中，本地偏离量较大的：

| 文件 | 本地相对基座 | 被哪些笔触碰 |
|---|---|---|
| `ds4.c` | **+3001 / −293** | 962ac19、77570b4、43a91aa、63059b6、fc80bd6 |
| `ds4_server.c` | **+1835 / −113** | fc80bd6（`server_generation_rewind` + speculative-boundary 站点） |
| `ds4_gpu.h` | +243 / −0 | 43a91aa、63059b6 |
| `tests/test_deepseek41_metal.c` | +467 / −5 | 962ac19、63059b6 |
| `Makefile` | +188 / −8 | 962ac19、77570b4、63059b6、b14bf62、fc80bd6 |
| `tests/test_session_state.c` | +173 / −0 | 43a91aa（+232）、fc80bd6 |
| `ds4_engram.c` | +99 / −0 | 962ac19（FP8 LUT） |
| `.gitignore` | +23 / −0 | 962ac19、77570b4、43a91aa、63059b6 |
| `docs/SPECULATIVE_DECODING.md` | +13 / −0 | b3afc8e（且本地 +13 即 #1003 的文档片） |
| `ds4.h` | +3 / −0 | 43a91aa、fc80bd6（本地 3 行是测试钩子 `ds4_test_use_deepseek41_shape`，与上游 hunk 不同区，冲突小） |

**本地未改（干净）**：`ds4_tp.c`、`ds4_tp.h`、`ds4_gpu_tp.h`、`ds4_cuda.cu`、`ds4_deepseek41_cuda.cuh`、`ds4_help.c`、`cuda/mmq/*`、`gguf-tools/*`、`docs/DGX_SPARK.md`、`docs/DISTRIBUTED.md`、`QA_BEFORE_RELEASES.md`、`tests/test_tp_*.c`、`tests/test_metal_tp_*.c`、`tests/test_dspark_eos_contract.py`、`tests/test_dspark_window.py`、`tests/test_quality_api.c`、各 `tests/test_cuda_*.c(u)`。

> 注：`ds4_tp.c/h` 本地未改，意味着 `f0962e3`（纯 ds4_tp.c）在**文本层面**几乎无冲突——但它的行为依赖同栈的协议版本与帧定义（见 §4）。

---

## 4. 依赖链与"能不能单挑"

### 4.1 TP 协议版本台阶（决定性）

| 提交 | `DS4_TP_PROTOCOL_VERSION` |
|---|---|
| `0aaea5a`（本地基座） | **14** |
| `962ac19` | 14 |
| `77570b4` | **15** |
| `43a91aa` | **20** |
| `63059b6` … `d6b516a` | 20 |
| `fc80bd6` | **21** |

握手处有严格相等检查（`theirs.version != DS4_TP_PROTOCOL_VERSION` → 中止）。因此：

- 任何一笔改动协议版本后，**两侧二进制必须同版本重建**；本地 Metal TP 测试（`test_metal_tp_spec`、`test_metal_tp_cancel`、`test_tp_link`、`test_tp_commands`、`test_tp_rdma`、`test_tp_tcp`）都在这条契约上。
- `77570b4` 把 wire 布局从 `n_layer × BATCH_MAX_ROWS` 改成 `n_layer × BATCH_GATES_PER_LAYER(2) × BATCH_MAX_ROWS`，并新增 `ds4_tp_batch_block_begin_gates()`（声明 + 实现 + 使用都在本笔内，`ds4_tp.c:2219` 为 wrapper）→ 该笔自闭合，但会把协议推到 15。

### 4.2 `fc80bd6` 单挑会失败（上下文依赖）

`fc80bd6` 的 `ds4.h` / `ds4_tp.h` hunk 的上下文由 `43a91aa` 引入，本地树**缺失**：

| 上下文符号 | 本地状态 | 引入者 |
|---|---|---|
| `ds4_session_tp_dspark_draft()` 声明 | 缺失 | `43a91aa` |
| `DS4_TP_FRAME_DSPARK_DRAFT = 23` 枚举 | 缺失 | `43a91aa` |
| `ds4_session_tp_spec_cycle(..., uint32_t flags, ...)` 签名 | 本地是旧签名（无 `flags`） | `43a91aa` |

→ 要么先落 `43a91aa`，要么手工改写上下文/签名（新增行本身不引用 `flags`，手工适配可行）。

### 4.3 其余配套关系

- `b14bf62` 是 `43a91aa`/`63059b6` 的**测试配套**（EOS fixture 按 TP proposals / split verifier heads 更新）——单取它会与本地旧行为不一致。
- `f0962e3` 使用基座已有的 `ds4_tp_failed` / `ds4_tp_mark_failed` / `ds4_tp_wait_command_status`（本地已有，非新造 API）——可独立取，但 `fc80bd6` 的失败处理依赖它引入的 `ds4_tp_failed` 语义（"失败即致命、恢复需重连"）。
- `fc80bd6` 自身闭包完整：声明（ds4.h/ds4_tp.h）+ 实现（ds4.c/ds4_tp.c）+ 调用（ds4_tp.c worker、ds4_server.c）+ 新测试（`test_dspark_rewind.py` 597 行、`test_server_speculative_stop.py` 247 行）。

---

## 5. 头号撞车：`fc80bd6` vs 本地 `#1003` + B2

两套实现建在**同一套机座**上（基座 `0aaea5a` 已有 `spec_frontier_snapshot/restore/commit_prefix/commit_prefix1`，`DS4_SPEC_PREFIX_SLOTS`）。

| 维度 | 本地（#1003 pick + B2） | 上游（`fc80bd6`） |
|---|---|---|
| 命名 | `dspark_rollback` / `ds4_session_dspark_rollback_eligible()` | `dspark_rewind_*` / `ds4_session_rewind_speculative()` / `ds4_session_restore_speculative_prefix()` |
| 来源 | `9c4cf2c`（Emilian Bold，9-09）、`bca9985`（9-17，hardening）、`4db95ee`（本地，10-08 凌晨） | `fc80bd6`（antirez，10-07） |
| 资格 | `backend==METAL && !tp.active && !ssd_streaming && !distributed && !glm && !qwen4`（**窄**） | 无 vision/streaming/distributed 排除；仅 `tp.active && !tp_leader` 时拒绝（**宽**） |
| 边界 logits | **不保存** → 消费点要求 `pos > dspark_rollback_start`（严格）；`pos == start` 即掉 checkpoint（B2 登记的残留缺口，约占 rewind 4%） | **保存** `dspark_rewind_logits`（`DS4_N_VOCAB` float）；`prefix == 0` 时直接采样 |
| 回放 | 逐 token `metal_graph_eval_token_raw_swa`，无上界钳制 | 有界：`base` 钳到 prefix slots，重放 ≤ 1 个 verifier block |
| TP | 不支持（资格里直接排除 TP） | `ds4_tp_send_spec_restore` 状态-only 命令 + ACK；失败双侧 invalidate；协议 21 |
| 失败语义 | 掉 checkpoint（`dspark_rollback=0` 日志） | 回落到旧 rebuild 契约，且**先 invalidate 两侧** |
| vision 会话 | B2 本地补丁去掉排除 → 图片会话可命中 | 结构上不排除 → B2 的 bug 不存在 |
| 本地测试/日志 | `tests/test_session_state.c` +173、`tests/test_server_vision_tool_boundary.py`、`dspark_rollback=0` 日志形态 | `tests/test_dspark_rewind.py`（597）、`tests/test_server_speculative_stop.py`（247）；无 `dspark_rollback` 日志字段 |

**冲突落点**（文本层面必然撞）：`ds4_session_rewind`（两边都在同一函数插分支）、`ds4_session_eval_dspark_speculative_argmax` / `..._stochastic`、`ds4_session_tp_spec_cycle`、`tests/test_session_state.c`（本地 +173 vs 上游 +232）、`docs/SPECULATIVE_DECODING.md`（本地 +13）、`ds4_server.c` 的 `server_generation_rewind` 与 speculative-boundary 站点（B2 的日志就在这一带）。

**三个选项**：

| 选项 | 动作 | 代价 |
|---|---|---|
| (a) 吸收上游 | 落 `fc80bd6`，退役 `9c4cf2c`/`bca9985`/`4db95ee`（或改写为薄适配） | 失去本地命名与日志形态；B2 的回归测试需重写（断言的是 `dspark_rollback=0`）；换来 logits 边界、TP 协调、有界回放 |
| (b) 保持本地 | 跳过 `fc80bd6` | 保留 `pos == start` 残留缺口、无 TP 协调、无回放钳制 |
| (c) 移植 | 把上游的 logits 保存 + TP ACK + 钳制移植进本地命名 | 工作量最大，但保住 B2 测试/日志与本地硬化（`bca9985` 的共享资格助手 + 边界算术单测） |

---

## 6. 平台相关性（反直觉要点）

以 `ds4.c` 的改动按函数分布为证（`+` 为插入行）：

**`77570b4`（标题是"双机 CUDA TP"，但落点以 Metal 侧为主）**

```
metal_graph_encode_layer_attention_batch     +93/-45
metal_graph_dspark_ring_maintain             +83/-7
ds4_session_prepare_dspark_draft_impl        +54/-60
metal_graph_encode_layer_ffn_batch           +53/-9
metal_graph_verify_suffix_tops_impl          +32/-6
ds4_session_eval_speculative_argmax_impl     +17/-21
metal_graph_seed_dspark_initial_cache_from_prefill  +17/-17
metal_graph_eval_dspark_stage0_batch         +17/-3
ds4_engine_open_internal / ds4_engine_tp_bind  +16/-6 / +10/-6
metal_graph_upload_prompt_embeddings_hc / metal_graph_prefill_chunked_range  +6/-0
```

提交正文自认：**"cross-backend history changes still require physical Metal and ROCm release QA"**——即上游明确要求物理 Metal 验收，而那正是本仓库的平台。

**`43a91aa`（同样是 Metal 侧大户）**

```
metal_graph_visual_tokens_for_batch              +270
ds4_test_engine_placement（测试钩子）             +233
metal_graph_encode_layer_ffn_batch               +154/-114
ds4_session_prepare_legacy_mtp_draft             +129
metal_graph_dspark_verify_selected_profile_enabled +98
ds4_sessions_eval_batch_with_prefill             +82
metal_graph_encode_layer_attention_batch         +80/-54
ds4_session_tp_spec_cycle                        +79/-7
metal_graph_hc_rms_scale_project                 +40
```

**真正的 CUDA-only 部分**：`ds4_cuda.cu`、`cuda/mmq/*`、`ds4_deepseek41_cuda.cuh`、`ds4_gpu_tp.h` 的实现，以及 27 个新 CUDA 测试。

**其余跨平台/共享要点**：
- `962ac19` 的 `ds4_engram.c`：FP8→BF16 转换改为 256×256 LUT（`pthread_once`），声称"preserve exact FP8 Engram conversion"、保留参考的 F32 scaling 与 BF16 rounding → 值得独立验证位等价（本地 `ds4_engram.c` 有 +99 偏离）。
- `63059b6` 的 `ds4.c`：`tp_dspark_split_eligible` 放开原生 MXFP4 drafter 专家（跨平台判定）；`gguf-tools/deepseek4-quantize.c` 保留 MXFP4；`ds4_help.c` 文案更新（`--dspark-confidence` 默认值分机型/格式）。
- `d6b516a`：质量打分器，纯跨平台、零冲突。
- `b3afc8e`：纯文档（双机 `dgxspark1`/`dgxspark2` 直连 200 Gb/s RoCE 做网络 TP；发布前 QA 清单扩到物理 RDMA/内存归属/官方续写打分等）。

---

## 7. 建议（按可操作性）

| 笔 | 建议 | 理由 |
|---|---|---|
| `d6b516a` | **可直接取** | 2 文件本地未改，纯跨平台修复；`api_top1_update` 修"取 `alts[0]` 当首选"的打分错误 |
| `b3afc8e` | **可直接取** | 纯文档，6 文件本地未改（唯一擦边是 `docs/SPECULATIVE_DECODING.md` 的本地 +13，与 §5 决策联动） |
| `b14bf62` | 可考虑，需配套 | Makefile hunk 在 `test-mmvq-rocm` 区（本地 Makefile 有 +188 偏离）；EOS fixture 更新绑定 `43a91aa`/`63059b6` 行为 |
| `962ac19` 的 `ds4_engram.c` 片 | 可考虑 | 跨平台且声称位等价；但与本地 +99 偏离对拍后才能落 |
| `fc80bd6` | **需决策**（见 §5） | 与本地 #1003 + B2 正面撞车；单挑还会在头文件上下文失败（§4.2） |
| `77570b4` / `43a91aa` / `63059b6` | **建议缓取或不取** | 主体是双机 CUDA TP 主线，但重写 Metal DSpark 图路径、把协议推到 15/20；本机无 DGX Spark 可验，取进来等于把"需要物理 Metal/ROCm QA"的语义改动拉进本地 |
| 若确实要取 TP 线 | **必须整栈原子落地** | `962ac19 → 77570b4 → 43a91aa → 63059b6 → b3afc8e → f0962e3`，并同步 Metal 对端、重建两侧二进制（协议 14→20/21） |

---

## 8. 附带发现：`LOCAL_INVENTORY.md` 已分叉（工作区 vs HEAD）

未提交改动 `+24/−16`，且两版**各有对方没有的内容**——不是单纯"未提交增量"：

| 内容 | HEAD（`84f65f5`） | 工作区 |
|---|---|---|
| B 段计数 | 29 个 | 24 个 |
| draftless n-gram 投机 | `eb5d66f`+`497f754`，**实测 a≈0.68、最优 1.000×、判死不实现** | `(this)`，**"待窗口测成本斜率 a"，估 1.36–1.70×** |
| V4.1 decode 瓶颈线四行 | `2386ce9`/`9014239`/`5d9cc3b`/`5885841` 在册 | **缺失** |
| 归因插桩行 | `c306365` 在册 | **缺失** |
| B2（10-08 凌晨） | 无 | 有（含证据与残留缺口登记） |

→ 吸收任何一笔之前，先确认哪一版是真态（工作区版像是"旧账本 + 新 B2"，HEAD 版像"新账本 − B2"）。按 AGENTS.md 的台账纪律，`LOCAL_INVENTORY.md` 必须与真态一致，否则后续 rebase 对账会认错。

另：账本 §D 提到的上游"同症状线索"（`#1007`、`#1194`、`#962/#977/#1153`）都是**未合并** PR；本地 `up/pr/*` ref 最新只到 `1155`（9-30），而上游 PR 编号已到 `1194` → 若要评估这些线索，需先补 fetch PR refs。

---

## 附录：本次复核用到的命令

```sh
git fetch origin --prune
git rev-list --count origin/main --not main                 # 9
git merge-base main origin/main                             # 0aaea5a
git merge-base --is-ancestor 0aaea5a main                   # 真
git cherry -v main origin/main                              # 9 行全 +
git log --pretty='%h A:%ad C:%cd %s' --date=iso-strict -n 22 origin/main
for c in 962ac19 77570b4 43a91aa 63059b6 b3afc8e f0962e3 b14bf62 d6b516a fc80bd6; do
  git show -U0 --pretty=format:'' "$c" -- ds4.c |
    awk '/^@@/{f=$0;sub(/^@@[^@]*@@ */,"",f);cur=f;next}/^\+/{a[cur]++}/^-/{d[cur]++}
         END{for(k in a) printf "%-70s +%d/-%d\n",k,a[k],(k in d)?d[k]:0}' | sort -t+ -k2 -rn
  git show "$c:ds4_tp.c" | grep -m1 DS4_TP_PROTOCOL_VERSION
done
```
