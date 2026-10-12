# REQ-002: GLM-5.3 live KV 中途分歧回退（消灭压缩/裁剪触发的全量 re-prefill）

关联: 无（新需求）；证据日志 `ds4_10092333.log`、DSH 会话 `session-692df918`
状态: 已确认（默认 8 槽）→ 内核 + 探针 + 单测已落地；真机红绿待排期

## 目标

让 ds4-server 在 **GLM-5.3** 上支持"live KV 回退到公共前缀、只重算尾巴"，
从而消掉由客户端历史改写（DSH compaction / tool-result prune）触发的整段
re-prefill。

用户: 本机开发者。现场配置: `./ds4-server -m gguf/GLM-5.3-Flash-Q2.gguf
--metal -c 256000 --port 8055 --vision gguf/GLM-5.3-Flash-Vision-Encoder.gguf`，
客户端 DSH 走 `/v1/chat/completions`。

## 根因（证据链，先于方案）

**线程 C（新）: 中途分歧 → 没有任何回退层可用 → 从 0 重灌。**

1. 现场（`ds4_10092333.log`，整段 126 次 prompt start，只有 3 次 `ctx=0..`）:

   | 时刻 | 行 | 事件 |
   |---|---|---|
   | 22:41:46 | 1 | 冷启动 `ctx=0..32559` |
   | 23:20:27 | 2062 | `live kv cache miss live=151035 prompt=146174 common=38326 vision=match reason=token-mismatch` → `ctx=0..146174`（324.0 s） |
   | 23:28:34 | 2240 | `live kv cache miss live=153838 prompt=114680 common=114307 vision=match reason=token-mismatch` → `ctx=0..114680`（≈240 s） |

2. 时间点与 DSH 会话记录逐条对齐（`session.v4.jsonl`）:
   - `23:20:27 compaction/prune` ×3（shadowedSeqs 29/69/305，共 8824 token）→ 同一秒 miss，
     `common=38326` = seq 29 在 prompt 里的位置。
   - `23:28:34 compaction/prune`(seq 643) + `compaction/start` → 同一秒 miss。
     `common=114307`、`prompt=114680`，差的 373 token 就是压缩指令。

3. DSH 侧的设计本来就是"前缀复用"（`packages/compaction/compaction-basic/src/summarizer.ts`
   注释）: 摘要调用把 region 会话原样重放、把压缩指令放在**最后一条 user 消息**，
   目的就是让这次调用成为上一次路由请求的**前缀 + 尾巴**。实测确实成立:
   `common=114307` = region 长度，尾巴 373 token = 指令。**客户端没有做错。**

4. 服务端为什么没接住（两处，缺一不可）:

   - **探针不给回退**。`ds4_server.c:12208` 把第 5 个实参写成
     `common == req->prompt.len ? common : 0`；中途分歧时传 0，
     `live_prefix_rewind_target()` 首行 `common <= 0` 直接返回 -1。
     末尾那段"分歧回退"兜底（`ds4_server.c:12232`）又被 `ds41_rewind` 门住，
     GLM 永远进不去。→ 探针返回 `REUSE_NONE`，所以 miss 日志的 reason 是
     `token-mismatch` 而不是 `live-prefix-rewind`（两份日志里
     `live-prefix-rewind` 出现 **0** 次，与推断自洽）。

   - **GLM-5.3 内核根本不能回退**。`ds4.c:89431`
     `state_ok = !s->glm_graph.glm53 || ds4_session_glm_mtp_rewind(s, pos)`；
     而 `ds4_session_glm_mtp_rewind()`（`ds4.c:77535`）只认
     `checkpoint.len == rollback_pos + 2` 且 `pos ∈ {rollback_pos, rollback_pos+1}`
     —— 即**最多回退 2 个 token 的 MTP 投机窗口**。
     所以 GLM-5.3 上任何 >2 的 rewind 都会 `checkpoint_valid = false`，
     `ds4_session_common_prefix()` 随即返回 0，`rewind_valid` 判定失败，
     调用方落回冷 prefill。GLM-5.2（`!glm53`）反而可以（`state_ok = true`）。

   根因一句话: **上游的 GLM rewind（7694112）在 GLM-5.3 上是死代码**
   （被 KDA 层挡住），而 V4.1 的中途分歧回退（423c5c2）没给 GLM。
   代码里已有注脚: `ds4_server.c:13687`「For GLM the compressed DSA KV cannot be
   rolled back by truncation (ds4_session_glm_mtp_rewind only handles the 2-token
   MTP cycle …)」—— 当时用"可见键层"绕开了 tool 轮，但绕不开历史改写。

5. 为什么现有四层复用都救不了（逐层排除）:

   | 层 | 为什么不行 |
   |---|---|
   | visible 键层（`REUSE_THINKING_VISIBLE`） | 要求新 prompt 文本以旧键为前缀；prune/压缩把历史**中间**改了，不是前缀关系 |
   | token 精确层 | 采样序列 ≠ 规范分词（handoff §4.2），本来就不承担复用 |
   | canonicalize | 只能延长 |
   | 磁盘层 | 未开 `--kv-disk-dir`；且只存请求前沿，114307 不是前沿 |

## 方案

给 GLM-5.3 补上"可回退"能力：**KDA/索引器状态的分层快照环 + 回退重放**。

原理: GLM-5.3 里 45 层中 34 层是 KDA（线性注意力），状态是**递推摘要**
（`layer_kda_conv_state` / `layer_kda_recurrent_state`），不能靠截断行数回滚；
其余 11 层是位置索引的 KV/DSA 缓存，截断天然安全。所以唯一缺的是 KDA 状态。

设计:

1. **快照环**（新增，`ds4_glm_gpu_graph`）: `slots` 份 spec-state 快照
   （复用已有的 `glm53_graph_spec_state_tensors()` 枚举，与 MTP 回滚同一套张量，
   即 `mtp_state_backup` 的多槽版本）+ 每槽记录 `pos`。
2. **写入点**: GLM prefill 主循环（`ds4.c:79150` 分支）每完成一个 chunk、
   `ds4_session_glm_note_dense_cache()` 之后，若 `pos` 跨过 `stride` 边界就存一份。
   只在前缀阶段存；decode 阶段不动（分歧点都在历史里）。
3. **回退**（`ds4_session_glm_rewind(s, pos)`）: 取 ≤ `pos` 的最新快照 `p` →
   恢复 spec state → `glm_dense_cache_len = p` → 用 prefill 批路径重放
   `[p, pos)` → `checkpoint.len = pos`、`checkpoint_valid = true`。
   接入 `ds4_session_rewind()`（替换 `ds4.c:89431` 那行）。
4. **探针**（`ds4_server.c`）: 让 GLM 也走"分歧回退"层 —— 把
   `common` 原样传进 `live_prefix_rewind_target()`，并允许 `align == 1`
   （GLM 没有 V4.1 的偶数位约束）。

代价与收益（ctx=256000）:

| slots | stride | 内存 | 最大重放 | 覆盖回溯 |
|---|---|---|---|---|
| 8（默认） | 8192 | 1.16 GiB | ~8.2k token（≈15 s） | 65k |
| 16 | 8192 | 2.33 GiB | ~8.2k token | 131k |
| 0 | — | 0 | 功能关闭 | — |

按现场数字: 23:28:34 那次 114680 token（240 s）→ 快照 106496 + 重放 7811 + 尾巴 373
≈ 15 s；23:20:27 那次回溯 112709 > 覆盖，仍走冷 prefill（不劣化）。

## 成功标准

1. `./ds4_test --server` 新增单测: 构造"GLM 形状 + 中途分歧 prompt"，
   `slot_probe_reuse_locked()` 必须返回 `REUSE_MEMORY_REWIND`（当前返回 `REUSE_NONE`）。
   该用例不需要加载模型。
2. 真机: 用 `tools/prefix_reuse_replay.py --apply-prunes` 重放录制会话，
   剪枝步之后的 cache 命中率恢复到 ≥0.95（修复前该步为 0）。
3. 真机: 记录一次"快照回退"日志（新日志行，含 `from= to= replayed=`），
   且该请求的 `prefill` 耗时 < 30 s（对照修复前 240 s）。
4. 回退正确性: 回退后重放的 logits 与"从 0 冷 prefill 到同一 prompt"逐 token 一致
   （`--dump-logprobs` 对比，temp=0）。这是本次改动的核心不变量。
5. 默认关闭时行为与修复前完全一致（`DS4_GLM_REWIND_SLOTS=0`）。

## 已知约束

- **内存铁律**: 本机 128 GB 上常驻一份 GLM-5.3（RSS ≈ 90 GB）。真机验证前必须查
  free+purgeable / swap / 有无 LLM 进程；单实例锁意味着验证需要停掉用户现有
  8055 server，**必须先获得用户同意**。
- 快照只在前缀阶段产生；decode 阶段产生的新位置没有快照（由 visible 键层兜底）。
- 不改任何已有 API/响应体；只新增环境变量（契约"只新增"）。
- TP（`tp_world == 2`）路径: 本版**不做 TP 快照/恢复**——恢复侧
  `ds4_session_rewind_snapshot()` 没有 `ds4_session_rewind()` 里的
  `ds4_tp_send_rewind()` 广播，leader 单独恢复会让 worker 停在旧位置之上。
  要真支持 TP，需要给快照恢复补一条带槽位号的 TP 广播，让两个 rank 都
  `glm53_graph_copy_spec_state()` 到本地环。

## 假设与默认选择（每条带不成立的代价）

1. 默认 `slots = 8`（≈1.16 GiB）。若用户认为太贵，改 4 或 0，代价 = 覆盖回溯减半
   或功能关闭，改一个默认值。
2. 环境变量命名 `DS4_GLM_REWIND_SLOTS` / `DS4_GLM_REWIND_STRIDE`，沿用 V4.1
   `DS4_V41_RAW_LOG_ROWS` 的"默认开、0 关闭"风格。若上游偏好别的名字，
   代价 = 改名 + 文档同步，分钟级。
3. 只对 `glm53` 生效；GLM-5.2 / V4.1 / Qwen 路径不动（避免回归面）。
   若后续要统一，代价 = 另开需求。
4. 快照写入只挂在 prefill 主循环；不挂 decode。若不成立（发现 decode 期分歧
   也常见），代价 = 需要给 decode 加节流快照，额外一档工作量。
5. 回退用 prefill 批路径重放，不用逐 token decode 路径。若不成立，
   代价 = 重放速度从 ~550 t/s 掉到 ~30 t/s，方案失去意义。

## 实施锚点（行号为 2026-10-10 工作区快照）

| 位置 | 用途 |
|---|---|
| `ds4.c:47077` `ds4_glm_gpu_graph` | 加快照环张量 + `slots/stride/head/pos[]` |
| `ds4.c:53977` `glm53_graph_copy_spec_state` | 抽出"按槽拷贝"版本，MTP 备份复用槽 -1 |
| `ds4.c:54030` 分配处 | 环分配（沿用 `glm53_graph_spec_state_bytes()`） |
| `ds4.c:79150` GLM prefill 主循环 | 快照写入点（`note_dense_cache` 之后） |
| `ds4.c:77535` `ds4_session_glm_mtp_rewind` | 作为参考；新增 `ds4_session_glm_rewind` |
| `ds4.c:89431` `ds4_session_rewind` GLM 分支 | 接入新回退 |
| `ds4.c:64021` `session_glm_*_state_bytes` | 内存核算（`ds4_session_memory_bytes`） |
| `ds4_server.c:12057` `live_prefix_rewind_target` | 允许 `align == 1` 的分歧回退 |
| `ds4_server.c:12208` 探针调用 | 传真实 `common`，GLM 走分歧层 |
| `ds4_server.c:12232` 分歧兜底 | 门条件从 `ds41_rewind` 放宽到"可回退引擎" |
| `ds4_server.c:17289` 探针单测表 | 新增 GLM 形状用例 |
| `tests/test_session_state.c` | 回退语义单测 |

## 任务分解

1. 内核: 快照环 + 写入点 + `ds4_session_glm_rewind` + 接入 `ds4_session_rewind`
   → 验证: `make` 编译通过；`make cpu`（`DS4_NO_GPU`）不破。
2. 服务端: 探针分歧回退层 + 单测 → 验证: `./ds4_test --server` 全绿（不加载模型）。
3. 真机红绿: `DS4_GLM_REWIND_SLOTS=0` 复现 miss，默认值复现命中
   → 验证: `tools/prefix_reuse_replay.py --apply-prunes` 命中率 ≥0.95。
4. 正确性: 回退路径 vs 冷 prefill 的 logits 逐 token 一致
   → 验证: `--dump-logprobs` 对比脚本。
5. 文档: `docs/SERVER.md` + `LOCAL_INVENTORY.md` 记录。

## 范围外

- 修 DSH 的压缩/裁剪策略（客户端侧已是最优前缀形态）。
- V4.1 的同类缺口（`ds41` 的 raw ring 只有 8192 行，回溯超过即重建）——
  同族问题，另开需求。
- 磁盘 KV 快照、跨会话复用。

## 实施结果（as-built，2026-10-10）

已落地，`./ds4_test --server` 红绿成立（关掉新层 6 条断言失败，打开全绿）。
相对上面"方案"有 4 处偏差，都是往更小、更安全的方向收：

1. **回退重放不手写**。快照恢复后只把 `checkpoint.len` 截到快照位并保持
   `checkpoint_valid`，尾巴交给既有的 `ds4_session_sync_internal()` 前缀续跑路径
   重算（`start = checkpoint.len` → `resumed_checkpoint = true`，KDA 状态**不会**
   被 reset）。因此没有复制任何 prefill 分块/索引 prefill/TP/progress 逻辑，
   重放路径与"普通续轮"完全同一条，风险面最小。
2. **不挂在 `ds4_session_rewind()` 上**，而是新增
   `ds4_session_rewind_snapshot()`，只由服务端复用探针的续跑路径调用。
   原因：`ds4_session_rewind()` 同时是投机 stop-boundary 回退，其调用方可能
   立刻接着采样，需要新鲜 logits；快照恢复刻意不刷新 logits。
   生成期路径行为与修复前逐字一致。
3. **探针改为向内核要"可达位"**（`ds4_session_rewind_floor()`），而不是先提
   `prompt_len - 1` 再等回退失败。这样 GLM 的严格截断场景也顺带修好了
   （原先必然失败：`ds4_session_glm_mtp_rewind` 只认 2 token 窗口），
   且不会打出"rewound … requires rebuild"这种自相矛盾的日志。
4. **快照环按会话持有**（`glm_graph` 在 `ds4_session` 内），因此 `--parallel N`
   下每个 slot 一份；内存核算已计入 GLM memory guard 的 scratch 估算，
   但 guard 本身按单会话评估（既有口径，未改）。

新增开关：

| 变量 | 默认 | 含义 |
|---|---|---|
| `DS4_GLM_REWIND_SLOTS` | 8 | 快照槽数；0 = 关闭（行为与修复前一致） |
| `DS4_GLM_REWIND_STRIDE` | 8192 | 两个快照之间的 token 距离 |

失败降级：环分配失败 / 快照拷贝失败 / 目标位没有快照 / checkpoint 带图 →
一律回落到原来的冷 prefill，不重试、不污染。

新增/改动清单：

| 文件 | 内容 |
|---|---|
| `ds4.c` | 环字段 + 惰性分配；`glm53_graph_copy_spec_state_into()`（原函数改为薄封装）；`ds4_session_glm_rewind_note()`（prefill 分块末按 stride 存）；`ds4_session_rewind_floor()`；`ds4_session_rewind_snapshot()`；内存估算 `glm53_graph_rewind_ring_bytes()`；测试钩子 `ds4_test_use_glm53_shape()` / `ds4_session_set_test_rewind_floor()` |
| `ds4.h` | 三个新声明 + 两个测试钩子 |
| `ds4_server.c` | `glm_live_rewind_target()`；探针新增 GLM 分歧回退层（排在所有免重算层之后）；`REUSE_MEMORY_REWIND` 分支先试快照；单测 `test_slot_probe_glm_rewind_tiers()` |

### 已知遗留

- ~~`make cpu` 在本机（macOS）链接失败~~ **2026-10-10 验收时推翻，两个洞都已修**：
  1. **本次改动自己的洞**（`-fsyntax-only` 抓不到，只到编译层、没到链接层）：
     `ds4_session_rewind_floor()` / `ds4_session_rewind_snapshot()` 声明在 `ds4.h`
     无条件可见，定义却整体落在 `ds4.c` 的 `#ifndef DS4_NO_GPU` 区（`77297..78135`）
     → CPU 构建的 `ds4_server_cpu.o` 链接时缺符号。已按本仓既有写法（见
     `ds4.c:62663` 同类 stub）在区外补 `#ifdef DS4_NO_GPU` 的"无可达位"实现。
  2. 既有洞：Darwin `cpu` 目标用 `$(LDLIBS)`，不含 `ds4_image.o`（#1197 WebP）
     需要的 ImageIO/CoreGraphics。已加 `IMAGE_FRAMEWORKS` 变量并挂到该目标。
  现在 `make cpu` 全绿（0 error 0 warning）。
- 快照只在前缀阶段产生；decode 阶段跨过的位置没有快照（由 visible 键层兜底）。
- **TP 判定纠正（提 PR 时）**：原写"TP 走逐 token prefill 分支，不经过快照写入
  点"，**不成立**——GLM-5.3 的 prefill 就在 `ds4_session_sync_internal()` 的
  共享分块循环里，该函数内没有任何 TP 分支（`ds4_session_glm_tp_spec_cycle()`
  只管 decode），且 `glm_graph_ready` 对 TP 会话同样为真 → 两个 rank 都会真的
  写环。叠加恢复侧缺 `ds4_tp_send_rewind()`，这是会让 TP KV 分裂的真洞。现在
  `ds4_session_glm_rewind_note()` 与 `ds4_session_rewind_snapshot()` 都按
  `s->engine->tp.active` 直接返回：TP 不写环、不恢复，floor 恒为 -1，探针走原路
  （不劣化，也无收益）。上游 PR 同此。

## 验收补记（2026-10-10，独立复现）

**已过（可复跑）**

- 成功标准 1：`./ds4_test --server` 全绿。红绿**独立复现**——把探针 GLM 层改成
  `false && glm_rewind` 后在 `/tmp` 另编一个 `ds4_test`，恰好 6 条断言失败
  （`ds4_server.c:17439/17440` × 3 组用例），与上面 as-built 声称一致。
- `tests/test_session_state_gpu`：环槽位偏移恢复 + 越界拒绝 PASS。
- `-Wall -Wextra`：`ds4.c`、`ds4_server.c` 各 0 warning。
- 内存实测核对：34 个 KDA 层 × 4.28 MiB + 11 层索引尾 = **145.6 MiB/槽**，
  8 槽 = **1.138 GiB/会话**（本文 1.16 GiB 成立）；且经
  `session_mem.total_bytes → glm_reserved_graph_bytes → glm_session_graph_bytes`
  **逐会话**计入 guard，`--parallel N` 不会漏算（上文 as-built 第 4 条担心的
  "单会话口径"实测不存在）。
- `make test` 编译层已修：`test_glm_rewind_floor()` 直接摸 `s->glm_graph*`，
  而 CPU（`-DDS4_NO_GPU`）变体没有这些成员 → 该目标编译中断（HEAD 版可编，
  确因本次改动）。已按本文件既有约定加 `#ifndef DS4_NO_GPU` 门，并把调用移进
  已有的 GLM 测试组。

**已改的行为（一处）**

- 标准 5 原写"默认关闭时行为与修复前**完全一致**"，严格说不成立：GLM 从
  `live_prefix_rewind_target()` 早层摘掉后，**严格截断**若落在
  `ds4_session_glm_mtp_rewind()` 的 2-token 投机窗口内（`checkpoint.len ==
  rollback_pos + 2`），修复前能复用、`slots=0` 或环为空时会变成冷灌。
  已修：`glm_live_rewind_target()` 在严格截断且环无可达位时仍报
  `prompt_len - 1`，由 `REUSE_MEMORY_REWIND` 分支既有的
  `ds4_session_rewind()` 兜底。新增回归行（`truncation + 空环` →
  `REUSE_MEMORY_REWIND/7`）**先红 2 条后绿**。

**真机已收（2026-10-10，M5 Max，`GLM-5.3-Flash-Q2` + `-c 256000`，默认 8×8192 环；证据 `~/ds4-evidence/glm-live-rewind/`）**

- 标准 4 **通过**：同一份 30,586-token prompt、`temperature=0`、48 token 贪心，
  空会话冷灌（trace `cache_source: none`）与"快照回退 + 重放 3963 token"
  （`cache_source: memory-rewind`）输出**逐字节相同**。冷灌 56.8 s → 回退
  10.6 s（5.4×）。核心不变量成立。
- 标准 3 **通过（措辞待对齐）**：`rewound live prefix from 30634 to 26624;
  3963 prompt tokens will be reevaluated`；回退落点 `26624 = 2048 + 3×8192`
  正落在快照网格上 → "最多浪费一个 stride"实测成立。本文承诺的
  `from= to= replayed=` 独立日志行仍未实现，但现日志已含 from/to/重放数三个量，
  建议以现日志为准改掉这条承诺而不是加日志。
- 标准 2 **部分通过，并暴露默认容量的真实边界（本次最有价值的发现）**：
  用 `tools/prefix_reuse_replay.py --replay --apply-prunes` 把录制会话
  `session-692df918`（210 请求步、4 次剪枝）全量重放到刚重启的干净 server：
  - 总复用率 **0.9802**（2,998 万 prompt token 只重灌 59.4 万）；**59 次快照
    回退**，累计少读 32.4 万 token ≈ 省 602 s 冷灌。
  - 仍有 **22 步复用率 <0.95**，其中 step 123（prompt 149,173、cached 0、
    328.9 s）是**全量重建**：该步剪掉的是**较早**的消息，公共前缀塌到比环里
    最老快照还老 → `ds4_session_rewind_floor()` 返回 -1 → 只能冷灌。
    **默认 8×8192 只覆盖最近 65k**，比这更深的分歧按设计不覆盖。
  - 所以剪枝**深度**决定成败，不是这一层没生效。要覆盖 ~95k 深度：slots 提到
    12（+0.6 GiB/会话），或 stride 放宽到 16384（**内存不变**，代价是每次最多多
    读 16k ≈ 30 s，仍远小于 90k token 的重建）。这条要写进 PR 描述和
    `docs/SERVER.md` 的调参说明。
  - 另有 ~15 步卡在 0.94–0.95：主要是回放器口径（转录里的 assistant 文本与工具
    调用是当次真实生成，回放用 `max_tokens=1` 重新生成，尾部对不齐 → 公共前缀
    提前结束）叠加 8192 粒度，不是新层失效。
  - **别拿 `--baseline` 的 9/206 直接对比**：基线是 DSH 真实压缩策略下的 prompt
    尺寸（峰值 ~146k），回放器不复现同样的压缩（峰值 225k）。同口径 A/B 需要
    `DS4_GLM_REWIND_SLOTS=0` 再跑同一套 210 步。

**提 PR 前的收口（现已全部收完，PR 已开）**

- **prefill 开销（CONTRIBUTING 的速度关注点）已量：无可测回归。** 同一台机器、
  两个 fixture（30,574 / 35,708 token）在 `SLOTS=0` 与默认 8×8192 下各做一次冷
  灌，且**两个配置里 fixture 的先后顺序对调**以抵掉热漂移：
  `SLOTS=0` 558.6 / 508.9 tok·s⁻¹，默认 569.9 / 541.7 tok·s⁻¹。同 fixture 的差值
  被"座位"效应（第二次 prefill 因持续负载降频约 10%）盖住，去掉该效应后环开启
  一侧反而略高 → 结论只能写成**低于噪声地板（~3%）**，不能写成具体百分比。
  物理上也应如此：一次 30k prefill 只写 ~4 个快照 ≈ 0.55 GiB 设备拷贝，摊到 55 s
  ≈ 10 MB/s，相对流式专家缓存本身的带宽可忽略。
  注：启动日志里的 "host reclaimable after session allocation" 分辨不出两种配置
  （环是首次写快照时才落地的惰性分配），A/B 的配置差异由环境变量与代码路径确定，
  而"环真的在写"由回放里 59 次回退证明。
- **速度闸门最终口径（ds4-bench，已写进 PR #1205 的 Speed 节，取代上一条的
  "低于噪声地板 ~3%"）**：`~/ds4-base`(`fc80bd6`) 与 `~/ds4-pr-rewind` 两个
  `ds4-bench` 做 **ABBA 短扫掠**（`--ctx-start 8192 --ctx-max 24576 --step-incr 4096
  --gen-tokens 64`，4 轮 = 20 个配对样本，每对样本相隔约 2 分钟；奇数轮 base→pr、
  偶数轮 pr→base，让残余单调热漂移在配对里相消）：`prefill_tps` 均值 **−0.08%**
  （95%CI −0.71%..+0.56%，中位 −0.04%），`gen_tps` 均值 **−0.19%**（CI −0.64%..+0.26%，
  中位 −0.29%），单对散布 90 分位 2.5% / 2.0%，逐轮均值正负交替
  （+0.90 / +0.48 / −0.39 / −1.30）→ **无可测回归，且区间包住零**。
  **踩坑（重要，别再来一次）**：先按"全长扫掠 2048→65536、两轮相隔 40 分钟"测，得出
  prefill −15%、gen −6.8% 的**假回归**；其形状是 ctx=2048（时钟受限）掉 24%、ctx=63k
  （带宽受限）只掉 1.5%，且 gen 路径本次一行未改、两侧分配决策逐项相同（只有 buffers
  3862→5027 MiB 是环本体），同样的对照隔 2 分钟就变成 −0.04%。**在这台被动散热的机器上，
  相隔几十分钟的单次 before/after 不能当证据**；ds4-bench 的 A/B 必须 ABBA 交错配对。
  另一个坑：护栏里 `pgrep -f "ds4-bench -m"` 会匹配到自身脚本中含该字样的命令行，
  要等进程应使用 `pgrep -x ds4-bench`。
- **同口径 `SLOTS=0` 对照回放：已收**（同 `session.v4.jsonl`、同回放器，只换环境
  变量；跑到 step 49 主动掐表，环关掉后每步都成冷灌，整轮要两小时以上）：前 49 步
  （不含首轮）**从零重建 13 次 vs 环开 0 次**，墙钟 **2,054 s vs 258 s（7.97×）**，
  重灌 **935,092 vs 105,686 token（-89%）**；同一步 49（91,617 token）环关 208.3 s
  vs 环开 6.1 s（复用 0.9742）。→ 标准 5 真机证据到手；"那 23 步 <0.95 不是环带来的
  回归"同时结清（同一位置环关掉是整篇重建）。
- **`DS4_GLM_REWIND_STRIDE=16384` 对照：已收，结论与调参猜测相反**。126 步同切片：
  从零重建 1→0，但聚合复用 0.9659→**0.9609**、墙钟 1,125→**1,319 s（+17%）**、重灌
  452,134→**519,104 token（+15%）**；step 123 本身 328.9 s 重建 → 303.4 s
  "回退到最老槽 18,432 再重灌 13 万 token"，只省 25 s。**根因**：那步剪得太狠，
  公共前缀只剩 ~20-35k/149,173（77-86% 是新内容），本来就没有长前缀可复用，
  任何环配置都救不了；把 stride 调宽只是把"重建"换成"很贵的回退"，同时让每次
  普通回退多读最多一个 stride（step 126：2.8 s → 12.6 s）。**默认值不动；要深度就
  加 slots**。已同步改：`docs/SERVER.md` 调参段（两份树）、本条台账。
- 标准 3 的 `from= to= replayed=` 措辞与实现对齐（改文档还是补日志，二选一）。
- `docs/SERVER.md`「Live KV rewind」已补两个环境变量；`LOCAL_INVENTORY.md`
  进行中表已登记本需求（含 PR 拆分注意）。

**PR 已提交（2026-10-10）**

- 上游 **[antirez/ds4#1205](https://github.com/antirez/ds4/pull/1205)**：分支
  `pr-glm-live-rewind`（fork 上同名分支）自 `origin/main fc80bd6` 切出，两提交
  `db1e50c`（引擎）+ `636bba7`（服务端 + SERVER.md），5 文件 +552/−15，
  OPEN / MERGEABLE。正文按上游 house style 走 Summary / Why / Behavior / Memory /
  Correctness evidence / Speed / Known limits / Tests / Related 九节。
- **不变量在本分支自己的构建上复验通过**（17:36，`636bba7`）：冷灌 86.6 s vs 快照回退
  14.4 s（6.0×），48 token 贪心续写 sha1 `217778c256573586` 逐字节相同；中段分歧步
  A（30,587 token）同槽回退到 26,624、输出如设计不同；PR 正文的不变量表与日志行均已
  换成本构建实测。**教训固化：模型-backed 证据只对产生它的构建成立**——正文里回放
  规模数字（13→0、30.0M 等）出自 fork 构建，已显式标注归属。
- 撞车处理：上游 open PR **#1005** 只泛化探针能力门、没有 GLM-5.3 恢复机制，已按
  "独立开 PR + 正文点名 + 去 #1005 留言"的决策在 #1005 留了带实测数字的评论
  （`gh pr comment` 已发），说明它的 `common` 目标在 GLM-5.3 上会被
  `state_ok = !glm53 || ds4_session_glm_mtp_rewind()` 拒掉，本 PR 提供它假设存在的恢复机制。
  **当晚收口（10-10）**：作者**当众撤回**其正文"能力门把 GLM-5.3 锁在门外"的前提
  （`is_glm_dsa` 按 family 测、旧门早已放行 5.3，其谓词实际只新增 Qwen4），并在正文
  增设引用本 PR 的专节；我第一条留言复述了该错前提，已公开认账更正
  （#issuecomment-6096007177）。分工定调：`ds4_engine_can_rewind()` 答"能不能"、本 PR
  的 `ds4_session_rewind_floor()` 答"落在哪"，不合并成一个谓词；合并时
  `slot_probe_reuse_locked()` 一处调用点做 ~3 行适配，我方跟 rebase。
- 未进 PR 的本树内容（保持本地）：`ds4_help.c` 的 `--image-cache-mb`、vision
  `SERVER_IMAGE_CACHE_BYTES`、menubar `/v1/status` 文档、Makefile 的 ImageIO
  框架修法（上游 `CPU_CORE_OBJS` 不含 `ds4_image.o`，那边 `make cpu` 本来就绿）、
  `docs/specs/` 本身。
- 提 PR 后又补的正确性修正（已在 PR 内）：TP 会话不写不恢复快照，见「已知约束」的
  TP 条目——原来的"TP 不经过快照写入点"是**错的**，真洞是 `rewind_snapshot()` 缺
  `ds4_tp_send_rewind()` 广播。
