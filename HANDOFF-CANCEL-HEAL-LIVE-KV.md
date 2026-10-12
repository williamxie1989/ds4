# HANDOFF：decode-cancel 后恢复 live KV 绑定（方案 1 实施手册）

> 基线：`main` = `2576049`（2026-10-12）。所有行号以该基线为准，动手前先 `git log -1` 确认；若已前进，用符号名重新定位而非行号。
> 本手册自包含：新会话不需要重做调研，只需按 §4 实施、§6 自测、§8 记账。

---

## 1. 问题与验收标准

**现象**：coding agent 对接 ds4-server，会话进行中途用户暂停（客户端 abort），再发一句话 → 全量 re-prefill（实测 130644 token ≈ 97s）。omlx 等框架同场景只增量 prefill 新消息。

**根因链**（已定位，证据见 §2）：
1. 每轮 visible 命中后，增量 suffix 被**独立分词**追加进 checkpoint（`build_live_prompt_suffix`），checkpoint 在每个轮次拼接点与 canonical 全量分词不同构；首个拼接点 = 首轮工具调用 frontier（实测 34101）。
2. 正常轮次全部命中 thinking-visible 层，token 级漂移一直潜伏。
3. 用户暂停 → 客户端断连 → worker 取消早退（`ds4_server.c:15174`）：**清空 thinking_live 且不回滚 frontier**（半句采样 token 留在 KV，live=130708），静默 return（无日志）。
4. 下一请求 visible/text 层全部落空 → token 层比较 → `common` 停在首个拼接点 34101。
5. rewind 目标 34100（按 KV token 对向下对齐）超出 raw-log 环容量（默认 8192 行，`ds4.c:41165`）→ `ds41_graph_rewind` 失败 → `checkpoint_valid=false` → "requires rebuild" → 从 0 全量重建。

**验收标准（DoD）**：
- 回归测试先红后绿（§6）。
- 生产窗口场景：暂停→补一句话，日志必须出现 `thinking live continuation match=visible-prefix`，**不得**出现 `live kv cache miss` / `requires rebuild`；`prompt start` 的 ctx 跨度只含新消息（几百 token 级），无长 prefill。
- `make` 0 error / 0 warning；`./ds4_test --all` 全绿。
- 不破坏 §3 的任何一条设计不变量。

---

## 2. 证据链（引用，勿重做）

日志：`ds4_10121948.log`（工作区根）。关键行：
- `19:49:11` `rewound live prefix from 217 to 8; 33742 prompt tokens will be reevaluated` —— 主对话冷 prefill，token 0..33750 为 canonical 全量分词。
- `19:50:22` `tool-call live checkpoint remembered ctx=8..33750:33742 live=34101 visible=33742` —— 首轮工具调用 frontier=34101。
- `19:50:22` `thinking live continuation match=visible-prefix cached=34101 prompt=36690` —— 第二轮起走 visible 命中，suffix 独立分词 → **34101 起 checkpoint 非 canonical**（首个拼接点）。
- `20:10:15` `chat ctx=130642..130692:50 gen=2100 TOOLS DSML_START decoding` —— 第 N 轮（prompt 止于 128592）生成中。
- `20:10:16`（会话转录）`assistant/message interrupted` + `turn/end aborted/user` —— 用户暂停，HTTP 断连。
- `20:10:43` `live prefix rewind from 130708 to 34100 requires rebuild` + `live kv cache miss live=130708 prompt=130644 common=34101 vision=match reason=token-mismatch` + `chat ctx=0..130644:130644 TOOLS prompt start` —— **全量重建**。
- 注意 20:10:15→20:10:43 之间无任何 finish/disconnect 日志行——取消早退是静默的（上游 #785 正在修这个日志缺口）。

对照：上游 #333 同族日志 `common=67527/live=67580`（差 53，分叉在尾部）。我们 common=34101（差 96k，分叉在首个增量拼接点）——我们的形态更深一层，引用时区分。

## 3. 设计意图约束（不得破坏）

1. **位契约**：采样 token 逐字保留在 checkpoint，重放文本不得重新分词覆盖旧采样 span（#985/#1169/ed3aafb 主线）。方案 1 不触碰采样 span——只回滚"客户端已放弃的半句"。
2. **fail-closed rewind**：rewind 必须精确落位，拒绝即重建（#1005 正文 antirez 确认有意）。方案 1 不得加"容错对齐"绕过。
3. **visible-key 一致性**：probe 命中要求 `thinking_live.live_tokens == live_pos`（`ds4_server.c:12218`）。heal 后必须维持该不变量，否则假命中比 miss 更糟。
4. **取消语义"宁脏不续"**：不得保留描述旧 frontier 的旧 key。方案 1 是**回滚 frontier + 用本请求文本重 arm**，不是保留旧 key。
5. **内存经济学**：raw-log 环 8192 行（≈0.6GB/会话）不得为 rewind 深度扩大（#1122 guard、预算表）。

## 4. 方案 1 规格

**语义**：decode-cancel = "本轮从未发生"。把 frontier 回滚到本轮 prompt 末尾（`old_pos`），并用取消请求自己的 canonical 渲染 `j->req.prompt_text` 重新 arm thinking-visible 绑定。下一请求（历史 + 新 user 消息）直接 visible 命中，只增量 prefill 新消息——对齐 omlx 行为。

**改动点**：`ds4_server.c` `generate_job_inner` 的 decode-cancel 早退（基线 15174）：

```c
if (job_cancelled(j)) {
    /* NEW: heal——回滚到本轮 prompt 末尾并重绑 visible frontier */
    if (!server_heal_live_on_cancel(s, slot, j, old_pos))
        request_live_state_clear(s, slot);   /* fallback = 现行为 */
    /* ...原有 free + return 不变... */
}
```

新函数 `server_heal_live_on_cancel(server *s, server_slot *slot, const job *j, int prompt_frontier)`（放在 `request_live_state_clear` 附近，便于单测）：
1. 前置检查：`j->req.kind == REQ_CHAT`、`j->req.prompt_text` 非空、非 `g_stop_requested`。不满足 → return false。
2. `pthread_mutex_lock(&s->inference_mu); ds4_session_rewind(slot->session, prompt_frontier);` 复验 `ds4_session_pos(...) == prompt_frontier && ds4_session_checkpoint_valid(...)`；失败 → unlock → return false。（注意 `ds4_session_rewind` 内部 `ds4_session_rewind_speculative` 可能早退不截断——复验必须做，不能省。）
3. 持 `tool_mu`：清 `responses_live` / `anthropic_live`（它们的 key 不再描述回滚后 frontier）；然后 `thinking_live_remember(s, slot, j->req.prompt_text, &j->req)` 重 arm。**注意锁序**：`thinking_live_remember` 内部自锁 `tool_mu`，第 2 步的 `inference_mu` 必须先释放；别嵌套。
4. return true。

**作用域事实**（已核实）：`old_pos` 在 `generate_job_inner` 顶部定义（基线 14265），15174 处可见；`j->req.prompt_text` 在取消时仍存活。

**rewind 跨度**：本轮已生成 token 数。环容量 8192−128 ≈ 8064；超出则 `ds4_session_rewind` 失败 → 自动 fallback 现行为，不会更糟。**不要**为"救长 turn"去调大环或绕过 fail-closed。

## 5. 已知坑（逐条规避）

1. **evict-store 落短快照（#1005 正文点名）**：已核实 `kv_cache_store_current`（基线 11795）只在 `thinking_live.live_tokens == tokens->len`（key-frontier 一致）时以 visible key 落盘——heal 后 key 与 frontier 一致，安全。若实现顺序写反（先 rewind 后 clear 再 store），会退化成 #1005 的 regression。顺序：rewind → 复验 → re-arm → 才允许任何后续 store。
2. **#785（OPEN）在同函数**：它给取消各早退出口补日志。落地时一并加一行 `server_log(DS4_LOG_DEFAULT, "chat ... cancelled at %d, healed to %d")`，措辞对齐 #785，避免下个 rebase 冲突时两边日志打架。
3. **prefill-cancel 早退（14612/14646/14656）不动**——那是 #376 的 resume 领地。
4. **shutdown（`g_stop_requested`）不 heal**。
5. **响应协议分流**：Responses/Anthropic 绑定的 slot 上取消，只清对应 live 状态、不 arm thinking key（它们的 key 形状不同，混 arm 会造成假命中）。
6. **协议标记字面量**：全程不得在对话/提交信息/日志格式串里字面写出 think/tool 协议标记序列；需要字面量时用字符码拼接落盘。

## 6. 自测清单（先红后绿）

**零模型单测**（新增，加入 `ds4_server.c` 测试块，命名对齐 `test_slot_probe_*` 系列，注册在基线 25334 附近）：
- `test_slot_probe_cancel_heal`：
  1. `ds4_session_new_test_checkpoint` 造 prompt-frontier checkpoint；按基线 17654 风格 arm `thinking_live`。
  2. frontier 推入若干"部分生成"token；`job_mark_cancelled`。
  3. 调 `server_heal_live_on_cancel`；断言：`ds4_session_pos == old_pos`、`thinking_live.valid`、`thinking_live.live_tokens == ds4_session_pos(...)`（INV-CANCEL-1）、probe 对 `prompt_text + 新消息` 命中 `REUSE_THINKING_VISIBLE`。
  4. 负例：rewind 不可达（`checkpoint_valid=false` 或 pos 复验失败）→ 断言 fallback 全清（`thinking_live.valid == false`），行为不劣于现状。
- 现有 `test_slot_probe_live_state_tiers` / `test_slot_probe_v41_rewind_tiers` 必须保持绿（不回归）。

**命令**：
```sh
make                 # 0 error / 0 warning
./ds4_test --all   # 全绿
```

**生产窗口验证（可选，内存铁律）**：动手前查可用物理内存、swap、有无 LLM 进程驻留；有余量才起 server。复现脚本化：两轮工具调用 → 中途 abort → 补一句话 → 核对日志（见 §1 DoD）。

## 7. 上游对齐

- 无在途 PR 直接修本缺口：#333（issue，承认症状）、#376（prefill abort）、#785（取消检测+日志）、#1194（Responses-only heal，仍付一次全量）、#1005（probe target 放宽，OPEN）、#1205（本 fork，rewind_floor/snapshot ring，OPEN）。
- 上游留言：在 #333 下登记 decode-cancel 形态（脱敏日志证据 + `common` 停在首个增量拼接点的观察），引用 #1194 的 "heal" 词汇并标注差异（本方案零 re-prefill）。
- 提交信息：本地 commit 引用 `#333 #785 #1005 #1205`；不碰 `pr-*`/`up/pr/*` refs。

## 8. 台账义务（AGENTS.md）

- `LOCAL_INVENTORY.md` C 段：登记本 local-work 与 #785/#1005/#1194 的依赖/冲突关系（同函数改动点）。
- 若后续 cherry-pick #785：commit 带 `(cherry picked from ...)` + `Upstream-PR: #785` trailer，并在 C 段更新本方案的冲突处置记录。
- `fork/main` 不得落后一个工作批次。

## 9. 快速开始（新会话前 5 步）

1. `git log -1` 确认基线 ≥ `2576049`；`grep -n "job_cancelled(j)) {" ds4_server.c` 重钉 15174 取消点。
2. 读 §3、§4，写 §6 的失败测试（预期红）。
3. 实现 `server_heal_live_on_cancel` + 接入点；跑 `make && ./ds4_test --all`。
4. 补日志行（§5.2）；更新 LOCAL_INVENTORY.md（§8）。
5. 窗口可用时做 §6 生产验证；在 #333 登记（§7）。

## 10. 符号索引（基线 2576049）

| 符号 | 位置 | 说明 |
|---|---|---|
| decode-cancel 早退 | [ds4_server.c:15174](ds4_server.c#L15174) | 唯一接入点 |
| `old_pos` 定义 | [ds4_server.c:14265](ds4_server.c#L14265) | heal 的 rewind 目标 |
| `thinking_live_remember` | [ds4_server.c:10947](ds4_server.c#L10947) | 重 arm，内部自锁 tool_mu |
| `request_live_state_clear` | [ds4_server.c:11015](ds4_server.c#L11015) | fallback 现行为 |
| `build_live_prompt_suffix` | [ds4_server.c:11969](ds4_server.c#L11969) | 拼接点毒源（根因，本方案不修，见根因候选 issue） |
| `slot_probe_reuse_locked` | [ds4_server.c:12124](ds4_server.c#L12124) | visible 命中条件 12218 |
| `kv_cache_store_current` | [ds4_server.c:11795](ds4_server.c#L11795) | key-frontier 一致才落盘（§5.1） |
| `ds4_session_rewind` | [ds4.c:89643](ds4.c#L89643) | 含 speculative 早退陷阱（§4.2） |
| `ds41_graph_rewind` | [ds4.c:41691](ds4.c#L41691) | 环回补 ≤128 行 |
| `ds41_raw_log_rows` | [ds4.c:41165](ds4.c#L41165) | 环容量 8192 |
| 测试注册表 | [ds4_server.c:25334](ds4_server.c#L25334) | 新测试注册处 |

---

## 11. 实施记录（2026-10-12，本会话落地）

**一处与 §4/§10 的实质分歧（重要，勿"好心改回"）**：手册把 rewind 目标写成 `old_pos`
（14265 处捕获），但 `old_pos` 是 **job 开始时**的 frontier = 上一轮结束，**不含本轮 user
message**。回滚到那里再用整段 `prompt_text` arm，会让 key 的文本覆盖比 KV 多出一条 user
message——下一轮 visible 命中后 suffix 独立分词会**静默跳过该 user message**（probe 只比
字节前缀，`build_live_prompt_suffix` 不复核，假命中比 miss 更糟）。手册三处本意指本轮
prompt 末尾：参数名 `prompt_frontier`、§6"造 prompt-frontier checkpoint"、§4 rewind 跨度
="本轮已生成 token 数"。实施按本意：在 decode 循环入口（`server_generation_enter` 前）捕获
`prompt_frontier = ds4_session_pos(...)`，heal 回滚到它并用 `j->req.prompt_text` arm——
prompt 保留、半句丢弃，"本轮从未发生"语义完整。

**实现**：`server_heal_live_on_cancel(s, slot, j, prompt_frontier)`（`request_live_state_clear`
之后）：守卫 REQ_CHAT+API_OPENAI+prompt_text 非空+非 `g_stop_requested`+frontier≥0 →
`inference_mu` 下 `ds4_session_rewind`+**复验 pos 且 checkpoint_valid**（§4.2 陷阱）→ 释放
`inference_mu` → 清 responses/anthropic live → `thinking_live_remember(prompt_text)`，
key_len=0 视为失败。调用点（原 15174）heal 失败 fallback 现行为；成功打
`chat cancelled at %d, healed to %d`（§5.2 措辞）。§5.1 顺序天然满足（函数内无 store）。

**测试 hook（ds4.c/ds4.h，仅测试路径置位，生产零分支）**：`ds4_session_append_test_tokens`
（无引擎推进 frontier）、`ds4_session_set_test_rewind_keeps_valid`（模拟图回滚保住
checkpoint——无图测试 checkpoint 的 rewind 默认 fail-closed 失效，probe 的 valid 闸门会挡住
正向用例）。

**测试**：`test_slot_probe_cancel_heal`（注册于 `test_slot_probe_live_state_tiers` 后）：
正向（heal 落位+INV-CANCEL-1+probe 命中 visible+alien 请求仍 miss）、fail-closed 负例
（rewind 不可达→fallback 全清，不劣于现状）、守卫负例（completions/Responses/Anthropic/
空 prompt/`g_stop_requested` 一律不 heal 不回滚）。先红后绿。

**门控结果**：`make` 0 error/0 warning；`./ds4_test --server`（含新测试）全绿；无权重可跑的
其余组（think-tool-recovery、metal-kernels、streaming-decode-prefill-checkpoints 系、
dspark-verify-depth）全绿。**模型权重层未跑**：根目录 `ds4flash.gguf` 符号链接指向的
IQ2XXS 权重已不在 `gguf/`（环境既有问题，与本次改动无关）。

**生产窗口验证（2026-10-12 23:52 窗，V4.1-Q2 + ssd-streaming 小 ctx 实例，隔离 KV 目录）**：
两轮工具调用→第二轮 decode 中途断连→补一句话。日志（证据
`ds4_10122353_heal_verify.log`）：`chat cancelled at 334, healed to 331` →
`thinking live continuation match=visible-prefix cached=331 prompt=362` →
`chat ctx=331..362:31 TOOLS prompt start`，无 `live kv cache miss`/`requires rebuild`。
增量 31 token，DoD 全项成立。注意：验证窗口 server 需 `--power 100`（V4.1 引擎校验强制，
`--power 60` 会拒绝启动）。

**下一客户端形态依赖**：heal 命中要求下一请求把中断轮 assistant 消息**重放回历史**
（DSH 有 tools 时 preserve reasoning 的渲染形态满足：key 尾部 assistant 起始头+think 开标记
是其字节前缀）。完全丢弃中断轮历史的客户端会 miss——回落现行为，不劣化。

### 11.1 完整回归补账（2026-10-13 窗，用户批准测试窗口）

权重事实：`ds4flash.gguf` 断链的 IQ2XXS 专属权重已迁离本机；**本机制式 = 台账
`de65ed9` 行登记的 91G vision-exp 权重**（`DS4_TEST_MODEL=gguf/DeepSeek-V4-Flash-Vision-Exp-…
gguf`，常驻）。结果：

- **零权重全量（make test 配方 29 项，含 ds4-eval×2 / ds4_agent_test / test_session_state /
  tp×3 / engram / host-memory / sampling / glm53_kda / ds41-exact×5 / metal_graph_capture /
  prompt_prefix / vision_image / image_decode / gpu_args×2 / layer_pack / web-recovery[python
  驱动形态] / q4k·mxfp4-dot）**：全绿，除 `test_deepseek41_fusions`（matvec bf16
  draw=0 word=0 old=c6ae0000 new=c6ae7b36）与 `test_deepseek41_engram_admission`
  （line 56 `calls[i]==(i==reader)`）——两红经 `2576049` 基线 worktree 重编重跑**签名逐字
  一致，存量红**（新登记，未见台账先前记录，处置待定）。
- **DeepSeek 族模型层（vision-exp 常驻，台账配方）**：session-snapshot / session-rewind /
  session-rewind-resample / tool-call-quality / think-tool-recovery / long-context /
  control-token-roundtrip / metal-short-prefill / metal-tensor-equivalence /
  **logprob-vectors(官方向量)** / dspark-verify-depth(vexp-support 配套，单跑) 全绿；
  `local-golden-vectors` 红 = 台账已登记的 fixture-量化不匹配存量红（top1 逐位等、仅尾部
  overlap 不过阈值），非回归。
- **GLM 族（GLM-5.3-Flash-Q2，`DS4_TEST_SSD_STREAMING=1` 缓存限 24GiB）**：session×3 /
  tool-call-quality / control-token-roundtrip / glm53-continued-prefill / server 全绿。
  两红定性：`--logprob-vectors`（glm-openrouter 向量）step0 top token 即缺 = 权重↔向量
  版本不匹配的未登记组合（行为组全绿佐证非回归；要跑需按 `fetch_openrouter_glm_vectors.py`
  对本地量化重采向量）；`--metal-ssd-streaming-cache-pressure` = 该组自身要求 16GiB 缓存
  与更大 session ctx，24GiB 限档下 `ds4_session_create` 准入不过 = 运行条件红。
- **qwen4-prefill-checkpoints / qwen4-restore-reuse**：本机无 Qwen3.8 权重，结构性不可跑。
- **生产窗口 E2E（§11 正文）**与 `--server`（含 cancel-heal 测试）各两轮全绿。
- 内存纪律：全程无 LLM 并发驻留，swap 增量 0（6.3G 均为用户声明的历史残留）。

操作备忘（防下个会话踩坑）：macOS 无 `timeout` 命令，回归脚本别用它（rc=127 假失败）；
`tests/test_web_recovery` 必须走 `python3 tests/test_web_recovery.py ./tests/test_web_recovery`；
测试 server 用 V4.1 权重时 `--power` 必须 100（引擎校验）。
