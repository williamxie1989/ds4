# ds4 KV 全量 re-prefill 交接文档（2026-10-08 会话产物；线程 A/B 均已修复）

> 给下一个 session：先读 §0。线程 A、B 都已落地并各有回归测试；§7 是"不要重走"清单，§8 是工作区状态。

---

## 0. 一句话现状

两条**互相独立**的"本可命中的 KV 被整段丢弃 → 从 0 全量 prefill"线程，**都已修复、已提交、已真机回归**：

| 线程 | 症状 | 状态 | 提交 |
|---|---|---|---|
| **A. 图片会话 tool 轮后全量重灌** | `live kv cache miss ... common=0 vision=mismatch` | 已修 + 真机回归 | `4db95ee`（+ 上游 PR emilianbold/ds4#1） |
| **B. stop 轮后可见键失配 → 全量重灌** | `live kv cache miss live=163524 prompt=163635 common=135132 vision=mismatch reason=token-mismatch` | 已修 + 真机红绿回归 | `744924d`（修复）+ `84f65f5`（回归） |

两条都不是客户端 bug，也不是"重放字节 ≠ 采样字节"的字面差异：

- **A**：图片会话被 `ds4_session_dspark_rollback_eligible()` 排除在 DSpark 回滚快照之外 → stop-boundary rewind 无法恢复 → `checkpoint_valid=false` → `common=0`。
- **B**：`remember_transient_checkpoint()` 把可见键**存成"剥过瞬态块"的形态**，而探针拿**原文**去比 → 逐字节比较在第一个被剥掉的块处断掉 → 可见键层永不命中；此时 token 层也命中不了（见 §4.2）→ 全量重灌。

---

## 1. 背景

两者都是"KV 明明完好，却被整段丢弃"：

- **A**：`checkpoint_valid=false`（但 `checkpoint.len` 保留）→ `ds4_session_common_prefix()` 直接返回 0；两个 vision 检查也因 `!checkpoint_valid` 返回 false → 探针 `REUSE_NONE`。
- **B**：live checkpoint 完好（`live=163524`），但**四层复用机制全部失效**：① 可见键层被"存剥过 / 比原文"的不对称比较挡住；② token 精确层因为历史里有 tool 轮而永远对不上（§4.2）；③ canonicalize 只能延长，不适用；④ 磁盘层对带图请求被设计性关闭。

---

## 2. 环境

- 机器：M5 Max 128G。模型：`gguf/DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf` + `--vision gguf/DeepSeek-V4-Flash-Vision-Encoder.gguf` + `--dspark --mtp-model gguf/DeepSeek-V4-Flash-Vision-Exp-DSpark-support.gguf`。
- 用户原始启动：`./ds4-server -m <上面模型> --vision <encoder> --metal -c 393216 --port 8055 --warm-weights --dspark --mtp-model <support>`
- **没有 `--kv-disk-dir`**（磁盘 KV 关闭）。
- 复现用变体（B 的回归就是这么跑的，够用且省内存）：**不带 `--vision`、不带 `--dspark`、`-c 8192`**。
- **铁律**（来自用户全局 AGENTS.md，必须遵守）：
  1. 任何加载模型/发真实请求的操作，动手前先查内存（free+purgeable、swap、有无 LLM 进程驻留），确认有余量；不确定就先问。
  2. **provider 为 ds4 / omlx-local 时禁止派发任何子代理**（`subagent` / `subagent_fork` / `workflow`），全部在主会话内做。
  3. 不要在正文/思考里字面写出"尖括号+竖线"协议标记（会触发 DSH sanitizer 截断），需要指代时用描述性名称。
  4. worktree 操作前先 `pwd` + `git status`；永远不要在 worktree 里跑 `git checkout main -- <file>`。
  5. ds4 有单实例锁：旧进程没退干净时新进程会拒绝启动（`another ds4 process is already running`）。

---

## 3. 线程 A：已完成（证据链，供 PR 追问时引用）

### 3.1 根因链
1. DSML 工具调用结束点通常落在投机块中间 → server 检测到 tool 结束即停止解码，但引擎已提交整块 → 执行 stop-boundary rewind（`ds4_server.c:14798` 有日志；代码来自上游 #1000，本地 pick `8963eae`/`e66f811`）。
2. DeepSeek 压缩 KV 无法靠截断行数回滚，唯一快路径是 DSpark 回滚快照（上游 #1003，本地 pick `9c4cf2c`/`bca9985`）。而资格函数里有 `!ds4_session_has_vision_state(s)` → 图片会话永远不 eligible。
3. → `state_ok=false` → `checkpoint_valid=false` → `common=0` → 全量重灌。

### 3.2 数字证据
- 现场：step 39/41 是 tool 轮（cacheRead 132023 / 133019），step 40/42 **全量重灌**（cacheWrite 132915 / 134541）。
- server 侧逐字同形：`live kv cache miss live=132783 prompt=132915 common=0 vision=mismatch reason=token-mismatch`（00:56:50；01:03:42 / 01:09:23 同形）。
- 复现：修复前 11 次 stop-boundary rewind → 6 次掉 checkpoint；修复后 23 次 rewind → 0 掉线，续轮 4/4 命中。
- **完整日志（§4.7）复核：44 次 stop-boundary rewind、`rewind dropped` 0 次** —— 修复在生产里成立。

### 3.3 交付物
- 本地 commit **`4db95ee`**（`main`）：`ds4.c`（去掉 vision 排除 + 回滚失败日志）、`ds4_server.c`（stop-boundary rewind 日志）、`tests/test_session_state.c`（`test_dspark_rollback_eligibility()`）、`tests/test_server_vision_tool_boundary.py`。
- **上游 PR：https://github.com/emilianbold/ds4/pull/1**（base `emilianbold:opportunistic-rollback` ← head `williamxie1989:fix/dspark-rewind-vision`，1 commit `606687d`，2 文件 +62/−4）。
  - 分支在本机 worktree `/private/tmp/ds4-pr-vision`（**别删**）；也已推到用户 fork。
  - PR 正文口径：回应 #1003 评审里 aaa2015 的第 3 点（`pos >= last_vision_token_end` 天然成立）。
  - **若作者回复**：
    - "想保持 #1003 原样、这个另开 PR 对 main" → 同一分支 rebase 到 `origin/main` 后对 antirez 开 PR。
    - "请把它并进 #1003" → 让作者 cherry-pick `606687d`。
    - 要求显式断言 `pos >= last_vision_token_end` → 在 `ds4_session_rewind()` 的 consume 分支加条件（`s->checkpoint_images[i].token_start + token_count`）。

### 3.4 线程 A 的残留缺口（已登记未修，别当新 bug）
`kept == 1`（块内首 token 即 stop token）时，rewind 目标恰等于快照位置，而快照不存 logits（consume 守卫 `pos > s->dspark_rollback_start`，`ds4.c:87852`），仍会掉 checkpoint。日志形态 `dspark_rollback=1 handle=A..0`，实测约占 rewind 的 4%。修法即 aaa2015 第 1 点：`spec_frontier_snapshot()`（`ds4.c:64569`）时把 `s->logits` 存进 `s->dspark_rollback_logits`，条件放宽到 `pos >= start`。**需要 logit 级 A/B 再落地。**

---

## 4. 线程 B：已定位 / 已修 / 已真机红绿回归

### 4.1 现场（用户完整日志 `ds4_10080700.log`，2026-10-08 07:16:56）
```
07:15:30 multimodal live kv hit images=6 cached=162702 prompt=162773 identity=fingerprint-match
07:15:30 thinking live continuation match=visible-prefix cached=162702 prompt=162773
07:15:30 chat ctx=162702..162773:71 TOOLS prompt start
   ... gen=751 finish=stop ...
07:15:58 transient live checkpoint remembered ctx=162702..162773:71 live=163524 visible=559833 stripped=486172
07:16:56 live kv cache miss live=163524 prompt=163635 common=135132 vision=mismatch reason=token-mismatch
07:16:56 chat ctx=0..163635:163635 TOOLS prompt start
```

### 4.2 为什么 token 精确层救不了（**这是本次新结论，推翻上一版"重放字节 ≠ 采样字节"的表述**）
探针里可见键层排在 token 层之前；一旦可见键层不命中，就落到 token 层。而 **token 层在"历史里有 tool 轮"之后永远对不上**：

- `ds4_session_common_prefix()` 比的是 **live 的采样 token 序列** vs **重渲染文本的规范分词**。模型是逐 token 采样，同一段字节可以有不同的合法切分，两者不必相等。
- 真机复现（`-c 8192`，tool 轮 + stop 轮 + 追问轮，`--trace`）：
  ```
  live_tokens_before: 478   live_prompt_common: 444   memory_miss_reason: token-mismatch
  ```
  `common` 冻在 **tool 轮结束时的 frontier**（444/461），live 已经往前走（478/494）——即"tool 轮之后的重放文本，分词结果与采样序列不同"。
- 完整日志复核：**整段 7.8 小时里每一次非冷启动续轮都是 `thinking live continuation match=visible-prefix`**（可见键层），token 精确层一次都没中过。**可见键层是这条会话唯一在工作的复用层**，它一失配就是全量重灌。
- 补充（上一版的算术修正）：`135132 − 135081 = 51`，而 `[reasoning 36][end-think 1][content 15]` 规范分词是 **52** 个 token，所以首个失配点落在正文最后一个字符附近，并非干净地落在工具信封首字节。**这个 1 token 的偏移不影响结论**（token 层本来就不承担这条会话的复用），不必再追。

### 4.3 真正的根因：可见键"存剥过的、比原文的"不对称
- `remember_tool_call_checkpoint()`（tool 轮）→ `build_tool_call_visible_text()`：存 **原文**（`prompt_text + suffix`），不剥。
- `remember_transient_checkpoint()`（无 tool 的 stop 轮）→ 存 **剥过瞬态块**的形态；若一个块都没剥到，则 `thinking_live_clear()` **直接清键**。
- 探针 `slot_probe_reuse_locked()` 一律拿 **原文** `ptext` 去比。

于是：
1. **07:15:30 命中证明客户端是"逐字回放"的** —— 它命中的是 07:15:25 存下的**未剥**键（`visible=556706`），而该键里含 `<system-reminder>` 块；若客户端会把块剥掉，这次命中就不可能发生。
2. 07:15:58 是 stop 轮 → 存下的是**剥过**的键（`visible=559833 → stripped=486172`，剥掉 73661 字节）。
3. 07:16:56 的请求里块还在（服务端渲染是 messages 的纯函数，块在早期 user 消息里，只增不减）→ **剥过的键不可能是原文的前缀** → 可见键层不命中；token 层又不中用 → 全量重灌。

一句话：**`remember_transient_checkpoint()` 用一把只对"会剥块的客户端"有效的钥匙，覆盖掉了本来能命中的原文钥匙。**

### 4.4 修复（commit `744924d`，只碰 `ds4_server.c`）
1. **两侧都归一到"剥过瞬态块"的空间**：
   - `thinking_live_remember()` 存键前统一 `strip_transient_blocks()`（返回长度，便于日志）。
   - 探针先 `strip_transient_blocks(ptext)` 再建可见键、再比。逐字回放的客户端与重建历史的客户端因此**都能绑定同一个 frontier**。
2. **`stripped_prefix_raw_offset()`**：键长是在"剥过"空间量的，而 live 后缀要从**原文**里切，所以按同一套 span 走一遍把键长映回原文偏移。**这一步是防污染的关键**——没有它，`suffix_off` 会切进块中间，把上下文写坏。
3. **`remember_transient_checkpoint()` 一律记键**（不再"没剥到就清键"）：token 层不可靠，没有键就等于放弃。
4. 抽了 `transient_block_end()`，`strip_transient_blocks()` 与偏移映射共用同一套 tag 匹配，保证两个空间严格同步。

### 4.5 真机红绿回归（commit `84f65f5`，`tests/test_server_transient_stop_boundary.py`）
场景形状 = 现场形状：**tool 轮 → stop 轮 → 追问轮**；`--log` 断言 + usage 计数。两种客户端形态各跑一遍（逐字回放带块 / 完全不带块）。

| | 修复前 | 修复后 |
|---|---|---|
| block-followup | `cached=0`，`live kv cache miss live=484 prompt=504 common=447 vision=match reason=token-mismatch` | 复用 478 / 478 |
| plain-followup | `cached=0`（同形） | 复用 459 / 459 |

修复后 trace 仍显示 `memory_miss_reason: token-mismatch`（token 层照旧不行），但 `thinking live continuation match=visible-prefix cached=478` 接住了——**正是修复要保证的行为**。

**生产配置复核**（`--vision` + `--warm-weights` + `--dspark --mtp-model` + `-c 393216`，RSS 91GB；用户 2026-10-08 现场配置）：

| | 真实整段重灌 | 判定点 |
|---|---|---|
| 修复前 `/tmp/ds4-server.prefix` | **2 次**（text / image 各一），日志 `live kv cache miss live=1563 prompt=2690 common=1526 vision=match reason=token-mismatch` | 5 个点里只有 2 个走可见键层，其余靠"刚重灌过、token 层暂时又能用"侥幸复用 |
| 修复后 | **0 次**（`ctx=0..` 只剩两次对话各自的冷启动） | **7/7 全部复用满 frontier**，每次剥掉 4–17KB 提示块 |

注意旧二进制的"间歇性"：一次整段重灌会把 live 重新规范分词，于是**下一轮 token 层又能用了**，再下一轮才又断——所以生产里看起来像偶发，实际是"每次纯文字结束轮之后必然踩"。

跑法：
```bash
cd /Users/xieyongliang/ds4
./ds4-server -m gguf/DeepSeek-V4-Flash-Vision-Exp-...gguf --metal -c 8192 \
  --port 8055 --trace /tmp/ds4_fixed_trace.log > /tmp/ds4_fixed.log 2>&1 &
sleep 25; curl -s -m 5 -o /dev/null -w "http=%{http_code}\n" http://127.0.0.1:8055/v1/models
python3 tests/test_server_transient_stop_boundary.py \
  --url http://127.0.0.1:8055 --log /tmp/ds4_fixed.log
```
单元层：`./ds4_test --server`（`test_stripped_prefix_raw_offset`、
`test_thinking_visible_binds_verbatim_transient_replay`——后者在旧比较下 4 条断言全红）。

### 4.6 用户日志证据（`/Users/xieyongliang/ds4/ds4_10080700.log`，4569 行）
范围 `1008 00:00:16` → `1008 07:47:58`（+ 09:14:19 shutdown）。统计：

| 签名 | 次数 | 结论 |
|---|---|---|
| `stop-boundary rewind` | 44 | rewind 路径频繁走到 |
| `rewind dropped` | **0** | **线程 A 的修复在生产里生效** |
| `live kv cache miss` | 5 | 4 次线程 A（`common=0`/`common=1`）+ **1 次线程 B（`common=135132`）** |
| `raw_tool_text=1` / `=0` | 96 / **0** | 生成侧"抓取模型采样原文"正常，§4.5 的"抓取侧"分支被排除 |
| `canonicaliz*` | **0** | 没有任何一次需要 canonicalize，与上一条自洽 |
| `thinking live continuation match=visible-prefix` | 每一次非冷启动续轮 | token 精确层从未生效，可见键层是唯一在工作的层 |
| `transient live checkpoint remembered` | 2（07:15:58 / 07:47:58） | 每次都是 stop 轮；07:15:58 那次直接导致 07:16:56 的全量重灌 |

关键消息：`seq=236` / `call_2dec5bd5b19a16face46de34329d88cc`（bash + heredoc），其生成时刻 = `1008 07:01:01 tool calls ctx=0..135081:135081 n=1 raw_tool_text=1`（那次本身是线程 A 的全量重灌）。

补充：rewind 后的重同步语义（对排查很重要）：
```
07:05:40 stop-boundary rewind ctx=141374..141955:581 finish=tool_calls kept=4 accepted=5 pos=144934
07:05:40 chat ctx=144934..144977:43 TOOLS prompt start
```
即 live 被截断到 `pos`，下一请求从 `pos` 起用**客户端渲染**重新同步。

---

## 5. 复现配方

### 5.1 起 server（务必等旧进程退干净，否则新 server 会拒绝启动）
```bash
cd /Users/xieyongliang/ds4
pkill -TERM -f "ds4-server -m gguf"; for i in $(seq 1 30); do sleep 1; \
  [ "$(pgrep -f 'ds4-server -m gguf' | wc -l | tr -d ' ')" = "0" ] && break; done
nohup ./ds4-server -m gguf/DeepSeek-V4-Flash-Vision-Exp-...gguf \
  --metal -c 8192 --port 8055 --trace /tmp/ds4_trace.log > /tmp/ds4_diag.log 2>&1 &
sleep 25; curl -s -m 5 -o /dev/null -w "http=%{http_code}\n" http://127.0.0.1:8055/v1/models
```
线程 A 的真机回归需要 `--vision` + `--dspark --mtp-model`；线程 B 的不需要（省 ~7GB）。

### 5.2 线程 A 回归（真机，图片+tools 多轮）
```bash
python3 tests/test_server_vision_tool_boundary.py --url http://127.0.0.1:8055 --log /tmp/ds4_diag.log --turns 8
```

### 5.3 线程 B 回归（真机，tool 轮 + stop 轮 + 追问轮）
```bash
python3 tests/test_server_transient_stop_boundary.py --url http://127.0.0.1:8055 --log /tmp/ds4_diag.log
```

### 5.4 只做分词实验（**不加载权重**，RSS ≈ 26MB）
```bash
./ds4 -m <model>.gguf --dump-tokens --raw --prompt-file <文本文件>
```
`--raw` 走 `ds4_dump_text_tokenization()` → `tokenize_rendered_chat_vocab()`，与 server 建 prompt 用的是同一套分词。构造含协议标记的文本时，**用脚本从 `ds4_server.c` 的 `#define DS4_DSML` 里取字面量**，不要把标记写进对话流。

---

## 6. 已排除 / 不要重走（重要）

1. **不要放开 `canonicalize_tool_checkpoint` 的 multimodal 闸门**——没有"原地改写"这回事（`ds4_session_rewrite_requires_rebuild`，`ds4.c:78804`），唯一兜底是磁盘重建，而磁盘 payload 无图片身份 → 会污染上下文。
2. **不要放开磁盘缓存的 `!multimodal` 闸门**（同上理由）。真要做需要 vision-aware payload（记录图片身份并在恢复前校验）——上游 **#1007** 那个量级。
3. **不要动 `tool_memory_attach_to_messages()` 的"只按 id 匹配"语义**。其容量上限 100000 条 / 512MB（`DS4_TOOL_MEMORY_DEFAULT_MAX_IDS`），本会话 96 次调用远未触及淘汰，**"id 被淘汰"不是线程 B 的成因**。
4. **不要试图"让 token 精确层重新工作"**：采样序列与规范分词本来就不必相等（§4.2）。可见键层就是为这件事存在的设计，修的方向是让它更可靠，不是绕开它。
5. **`suffix_off` 必须落在原文空间**：可见键在"剥过"空间量长度，直接拿它当 `prompt_text` 偏移会切进块中间、污染上下文。任何改这条路径的补丁都要保住 `stripped_prefix_raw_offset()`。
6. **不要用 `timeout` 命令**（macOS 没有），用工具级超时。
7. `LOCAL_INVENTORY.md` 里**有用户未完成的编辑**（第 64 行两条记录粘连），别把它一起提交（见 §8）。

---

## 7. 关键坐标

| 位置 | 说明 |
|---|---|
| `ds4.c:63583` | `ds4_session_dspark_rollback_eligible`——线程 A 的修复点（已改） |
| `ds4.c:64569` | `spec_frontier_snapshot`（残留缺口要动的地方） |
| `ds4.c:82957` | `dspark_rollback_replay_fits` |
| `ds4.c:87853` | rewind consume 守卫 `pos > s->dspark_rollback_start`（线程 A 残留缺口的根） |
| `ds4.c:87886/87893` | `rewind dropped the live checkpoint` 日志 |
| `ds4.c:78804 / 78820` | `ds4_session_rewrite_requires_rebuild` / `ds4_session_rewrite_from_common` |
| `ds4.c:77322 / 77361` | `ds4_session_vision_prefix_matches` / `..._fingerprint_prefix_matches` |
| `ds4_server.c:10747` | **线程 B 修复点 1**：`thinking_live_remember` 存键前统一剥瞬态块，返回键长 |
| `ds4_server.c:11986` 起（探针 `REUSE_THINKING_VISIBLE` 分支，命中在 12008） | **线程 B 修复点 2**：入参也剥 + `stripped_prefix_raw_offset()` 映回原文 |
| `ds4_server.c:10220/10221` | 上面两个函数的前置声明（`strip_transient_blocks` / `stripped_prefix_raw_offset`） |
| `ds4_server.c:13230 / 13250 / 13291` | `transient_block_end` / `strip_transient_blocks` / `stripped_prefix_raw_offset`（共用同一套 tag 匹配） |
| `ds4_server.c:13532` | **线程 B 修复点 3**：`remember_transient_checkpoint` 一律记键，不再"没剥到就清键" |
| `ds4_server.c:13564 / 13602 / 13738 / 15210` | `canonicalize_tool_checkpoint` / multimodal skip / `should_canonicalize_...` / 调用点 |
| `ds4_server.c:10918 / 10944 / 15143` | `tool_memory_remember` / `tool_memory_attach_to_messages` / 生成侧调用点 |
| `ds4_server.c:6220 / 6612 / 6813` | DSML 解析器设置 `raw_tool_text` 的三处 |
| `ds4_server.c:14183` | miss 日志（`live= prompt= common= vision= reason=`） |
| `ds4_server.c:14879` | stop-boundary rewind 日志 |
| `tests/test_session_state.c` | 线程 A 资格单测（`test_dspark_rollback_eligibility`） |
| `tests/test_server_vision_tool_boundary.py` | 线程 A 真机回归 |
| `tests/test_server_transient_stop_boundary.py` | **线程 B 真机回归** |

---

## 8. 工作区代码状态（防误提交）

- `main` = **`84f65f5`**（`744924d` 修复 + `84f65f5` 回归），在 `4db95ee` 之上。
- **未提交**：`LOCAL_INVENTORY.md`（= 用户未完成的编辑 + 前一会话加的 B2 段，**没有一起提交**）；未跟踪：`KV_REPREFILL_HANDOFF.md`、`PR1178_HANDOFF.md`、`PR1178_SESSION2_RESULTS.md`、`ds4_10080700.log`。
- worktree：`/private/tmp/ds4-pr-vision` → 分支 `fix/dspark-rewind-vision` = `606687d`，**已推 fork，勿删**。
- 纪律：不要 `git add -A`；只提交自己碰过的文件。

---

## 9. 数字快照

- 线程 A 现场：prompt 132915 / 134541 两次全量重灌；`common=0 vision=mismatch`；live 132783。
- 线程 B 现场：`live=163524 prompt=163635 common=135132`；`visible=559833 stripped=486172`（剥掉 73661 字节）；`transient live checkpoint remembered ctx=162702..162773:71`。
- 线程 B 真机复现（修复前）：`live=484 prompt=504 common=447 vision=match reason=token-mismatch`，`cached=0`；修复后同形状复用 478/478、459/459。
- 分词事实（vocab-only 实测）：`reasoning`(现场 seq=236 那条) = 36 token；`content` = 15 token；`reasoning + end-think + content` = 52 token。
- 图片：6 张（`vision embedding cache hit` ×6），均为 1280x800 PNG；`deepseek4-vision.image.max_tokens = 384`。
- 上游状态：**#1000** `aaa2015:fix/speculative-stop-rewind` @ `c7d4b67`（open）；**#1003** `emilianbold:opportunistic-rollback` @ `4e9574b`（open）；**#1007** 同族多模态复用问题（open）；我们提的 **emilianbold/ds4#1**（open）。

---

## 10. 用户待办 / 下一步可做

1. ~~保存 server log~~ **已提供完整版**：`/Users/xieyongliang/ds4/ds4_10080700.log`（00:00:16 → 07:47:58）。结论见 §4.6。
2. **线程 B 修复的上游去向待定**：这条改动独立于线程 A，可考虑单独对上游开 PR（先确认作者对 `emilianbold:opportunistic-rollback` 之外的口径）。
3. 线程 A 的残留缺口（§3.4，`kept == 1` 那一类）仍未修。
4. 若要在本机复跑真机回归：先按 §2 铁律 1 查内存（本次实测：停掉 omlx-server 后 free ≈ 93GB、无 LLM 进程驻留，`-c 8192` 跑得很稳，4 个请求约 4 秒）。

---

## 11. 本会话产出的临时文件（/tmp 可能已被清）

- `/tmp/dsh3/s.v4.jsonl`：解开后的用户会话记录（523 行）；`/tmp/dsh3/mk.py`：从 `ds4_server.c` 取标记字面量、生成候选文本的脚本；`/tmp/dsh3/cand/`、`/tmp/dsh3/tok_*.txt`：分词实验产物。
- `/tmp/ds4_fixed.log`、`/tmp/ds4_fixed_trace.log`：线程 B 修复后真机回归的日志与 trace。
- `/tmp/ds4_prefix.log`、`/tmp/ds4_prefix_trace.log`：同一回归在**修复前**二进制上的红态日志。
- `/tmp/ds4-server.prefix`：修复前二进制（= `git show HEAD~1:ds4_server.c` 编出来的），做红/绿对比用。
- `/tmp/ds4_server_fixed.c` / `/tmp/ds4_server_prefix.c`：切换构建时留的源码副本。
- 以上都可重建；重建方法见 §5。
