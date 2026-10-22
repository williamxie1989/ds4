# HANDOFF：sf-ds4-1flash 调研 → 可借鉴项验证/移植（2026-10-21）

> 承接 `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md`。
> 本次只做调研：**未改代码、未加载模型、未跑新 A/B**。
> 当前 HEAD `89e068c`；工作区有未提交的 DSpark/ledger 改动，**不要覆盖**。

---

## 0. 现状与目标

| 项 | 状态 |
|---|---|
| fork 调研 | **完成**，记录见 `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md` |
| fork 宣称 | 在其口径下基本属实；**不是**我们当前树的提升空间 |
| 我们树已有 | #1041（1/2）、#1042（6/7）、#1090（3/7）、#1120（5/5）、#1192 已 pick |
| 剩余粗比 | **decode +18~30%、prefill +20~84%**（auto 64.5 GiB vs fork 82 GB，非严格 A/B） |
| 首 token | 2K/4K 我们更快（77/75 ms vs 185/144 ms）；8K/32K fork 更快（152/194 ms vs 我们 368/279 ms） |
| 本轮目标 | 验证并选择性移植 P0–P4；**不是**追 README 的 +78% |
| 明确不借 | `--boost`/fanboost、dual-drive、整树 prune、盲目 82 GB cache |

**勘误**：上一轮口头说“explicit prefill API/测试已在树内”是误把 fork 的文件当成本地。
逐文件核对后：**本地没有** `ds4_gpu_stream_prefill_bind_layer` / `streaming_prefill_bytes` /
`ds41_prefill_expert` / `ds41_prefill_expert_pread`；这套是 fork 的 `60/100` 新增，需整块移植。
本地已有的是 18-thread `pread` pool、slab 分配器、mlock relief。

---

## 1. 关键事实（先读，别重复调研）

### 1.1 fork 的数字

- prefill **+45.0%~+99.4%**、steady decode **+61.7%~+78.3%**、
  generation（含首 token）**+109.3%~+129.2%**、first token after prefill **2.565 s → 0.185 s**。
- baseline 是 ds4 **`0aaea5a`（2026-09-20）**；性能表对应 fork **`fa51b38`**（sync 前的 `fc80bd6`）。
- 工况：M5 Max 128 GB / V4.1 Flash Q2 / `--ssd-streaming --ssd-streaming-cache-experts 82GB` /
  I Promessi Sposi / greedy 128 tokens。
- “first token”是 **prefill 完成后的第一个 decode token**，不是总 TTFT；2048 总 TTFT 仍约 11.7 s + 0.185 s。
- “generation”含首 token；`steady decode` 才是每 token 速度。
- `--boost` 列是 fanboost（全速风扇），不是软件；主表不含 boost。

### 1.2 为什么快

- baseline decode token ≈ 56 ms，**78% 在 expert path**，其中最大头是 **selected-id readback 的 sync ≈ 38 ms**。
- wide prefill sweep 后 decode cache 冷、page cache 被 evict，首 token 2.1~2.4 s；fork 用 `60/100/268/270` 修。
- prefill 大头：`60` one-sweep tail / decoder suffix / index batches / 单 chunk explicit buffers；
  `100` wide sweep explicit buffers + RAM-before-disk；`268` 页对齐 uncached reads。
- decode 大头：`30` #1041 queue（我们已有）、`131/132/210` readback、`70` #1073 glue（我们大部分已有）。
- 证据链：parity oracle 10 prompts token-identical；每 step token gate；bitwise step 过 `--bitwise`；
  A/B 是 A B B A、median + bootstrap CI；page-cache/thermal 不对称已明说并控制。
- **没有发现计时作弊**；但只有一台机器/一个模型/一个量化/一个 cache 档/一条 prompt。

### 1.3 我们树已经有什么

| fork 改动 | 我们树 |
|---|---|
| `30` #1041 queue | **已有，默认开**（`LOCAL_INVENTORY.md` L37；`V41_M5MAX_BASELINE_SPEED_PLAN.md` §1） |
| `70` #1073 glue | **大部分被 #1042/#1090 覆盖**（#1042 6/7、#1090 3/7；#1073 不单独取） |
| `300` #1192 | **已 pick**（`LOCAL_INVENTORY.md` L44） |
| `40` 18-thread pool/slab/mlock relief | **已有类似物**；#1125 四项修复不在 ledger |
| `60` S1a one-sweep tail | **已有类似逻辑**：`ds41_prefill_count` medium sweep（`ds4.c:43118`） |
| `60` S1b decoder suffix | **有，默认 4096**；没有 fork 的 short_sweep/128 warm；DSpark parity 约束（`ds4.c:43424`） |
| `60` S2 index batch | **有较旧版**（`ds4.c:42482`）；缺 candidates_batch/select-all/ratio-1 publish |
| `60/100` explicit expert buffers | **没有**（见勘误） |
| `131/132/210` readback | **没有**（`g_wait_poll_ns`/`g_readback_box` 不存在） |
| `268` 页对齐 reads | **没有**（依赖 explicit prefill driver） |
| `270` cache lock at open | **没有**（有 lazy mlock + relief） |

---

## 2. 下一步（按优先级）

### P0：先借方法论

- 移植 fork 的 `speed-bench/ab_bench.py`、`readme_bench.py`、`tests/test_ab_bench.py`；
  不碰引擎。
- 冻结我们 HEAD 基线：同 GGUF/flags/prompt、auto 64.5 GiB、512/128 greedy；先跑 A/A。
- 没有这套，后面的百分比会被路径漂移/热漂移污染。

### P1：小改动

- `270` cache lock at engine open（fork `5b6482a`）：engine open 时分配并 mlock 所有 cache slot；
  fork 首 token 2048 209→190.5 ms、8192 157.5→140.1 ms；prefill 5000 +1.8%；decode 2048 +2.4%（CI -0.6..+3.5）。
- `268` aligned uncached reads（fork `220dd7d`）：`F_NOCACHE` 按 16 KiB 页对齐；
  fork first token 8192 1230→150 ms、ttft 10000 +5.7%、append+1500 +12.5%。
  **条件**：先有 explicit buffer driver，否则我们 mmap page-in 路径不适用。

### P2：decode readback

- `131/132/210`（fork `f8dd83e`/`67b75b8`/`37a5d1e`）：bounded status poll、router 单独 commit、
  selected-id mailbox；fork +2.8~3.1% / +1.6~2.2% / +5.5~6.2% decode。
- 我们 `ds4_gpu_routed_moe_one_tensor` 路径与 fork 不同；**不能整包复制**；
  先对齐 `ds41_moe_partial`/readback 结构，选一条 opt-in 路径 + 回退开关。
- `#1125` 四项修复单独评估；不要当纯速度项（fork decode 8192 -0.37%）。

### P3：prefill 最大项

- `60/100` explicit expert buffers + RAM-before-disk（fork `0d88af9`/`896785c`）：
  wide sweep 显式读进两套完整 expert layer buffer；下一层读与当前层 compute 重叠；
  decode cache ready experts 先 blit 到 slot。
- fork：ttft 3500 +23.3%、5000 +8.5%、7500 +19.0%、10000 +23.9%；append+1500 +34.8%；
  首 token 3500 1373→258 ms、7500 1372→297 ms、10000 164→125 ms。
- 我们树没有这套 API/driver；port 约 300–400 行；先只做 wide sweep（>=2048 行），
  避免和 `ds41_streaming_sweep_moe_gather_admitted` 的 small append gather 重复。
- 必须过 bitwise、DSpark streaming、长上下文、SSD streaming + resident、Metal default。

### P4：小-中

- `60` S2 index batch：ratio-1 publish、batched candidate blocks、select-all short rows；
  fork ttft 7500 +2.3%/+1.69%/+4.3%；约 100 行。
- `60` S1b decoder suffix 2541/short sweep：fork ttft 3500 +2.6%、7500 +30.2%；
  我们默认 4096，且 DSpark 128-row parity 约束；**先重验 DSpark streaming short-prefill/verify**，不能直接改。
- `70` #1073 缺口：q_a/kv、RoPE/quantize、attention gather；单项约 0.4~0.8% decode；
  先对照 #1042/#1090，只补缺口。

---

## 3. 约束与闸门

- **模型级测试前先停掉 8055 `ds4-server`**（当前约 85 GB RSS），确认 free+inactive 内存大于峰值；
  不要并发跑两个 huge model 进程。
- 每项：token-identical → `--bitwise` → `./ds4_test --logprob-vectors` →
  DSpark streaming `--short-prefill-ssd` 与 verify parity → `--long-context` →
  `tests/test_metal_ssd_experts` / `test-metal-command-memory` →
  SSD streaming + resident 两态 → Metal default。
- 负结果按 `LOCAL_INVENTORY.md` §D 登记；别重复。
- 不借：`--boost`/fanboost、`140` dual-drive、整树 prune、盲目 82 GB cache。
- cache 权衡：fork 自己的 86/88/90 GB 数据首 token 明显变差；我们 auto 64.5 GiB 在 2K 首 token 77 ms，
  比 fork 185 ms 更快。按负载调，不是越大越好。

---

## 4. 下个 session 交接提示词（可直接粘贴）

```text
你是 ds4 仓库（/Users/xieyongliang/ds4）的 coding agent。上一个 session 完成了对
Chida82/sf-ds4-1flash 的调研，记录在 SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md，
交接在 HANDOFF-SF-DS4-1FLASH-20261021.md。当前 HEAD 89e068c，工作区有未提交的
DSpark/ledger 改动，不要覆盖；本次只做调研，未改代码。

目标：验证并选择性移植 fork 的可借鉴项，不是追 README 的 +78%。
关键事实：fork 的 +45~99% prefill / +62~78% decode 是相对 ds4 0aaea5a；我们树已有
#1041/#1042/#1090/#1120/#1192，按 speed-bench/v41_m5max_ab/auto-1/2.csv 粗比只剩
decode +18~30%、prefill +20~84%；首 token 2K/4K 我们更快、8K/32K fork 更快；
fork 的 82 GB cache 与 explicit buffers 是主因之一。

先做 P0：借 fork 的 speed-bench/ab_bench.py、readme_bench.py、tests/test_ab_bench.py
与 perf-record 纪律，冻结我们 HEAD 基线（同 GGUF/flags/prompt、auto 64.5 GiB、
512/128 greedy），跑 A/A。然后 P1→P4：
P1 270 cache lock（fork 5b6482a）、268 aligned uncached reads（fork 220dd7d；依赖 explicit
buffer driver）；P2 131/132/210 readback（f8dd83e/67b75b8/37a5d1e）；P3 60/100 explicit
expert buffers（0d88af9/896785c，wide sweep 优先）；P4 60 S2 index batch、60 S1b decoder
suffix（先过 DSpark 128-row parity）、#1125、#1073 缺口。

不要借：--boost/fanboost、140 dual-drive、整树 prune、盲目 82 GB cache。
任何模型级测试前先停掉 8055 ds4-server（当前约 85 GB RSS），确认内存余量；
不要并发跑两个 huge model。闸门：token-identical → --bitwise → --logprob-vectors →
DSpark streaming short-prefill/verify → long-context → SSD streaming + resident → Metal default；
负结果按 LOCAL_INVENTORY.md §D 登记。先读 SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md
与 HANDOFF-SF-DS4-1FLASH-20261021.md，再动手。
```

---

## 5. 参考

- `SF_DS4_1FLASH_SPEED_BORROW_ASSESSMENT.md`（本调研全文，含表格、代码坐标、fork 提交清单）
- `V41_M5MAX_BASELINE_SPEED_PLAN.md` §12.1（我们树 22.0–24.4 t/s 基线；§1 #1041/#1090/#1035/#1073 盘点）
- `LOCAL_INVENTORY.md` #1041（L37）、#1042（L38）、#1090（L19）、#1120（L21）、#1192（L44）、#1073（L236）
- `speed-bench/v41_m5max_ab/auto-1.csv`、`auto-2.csv`
- fork：`README.md`、`docs/PERFORMANCE.md`、`speed-bench/perf-record.md`、`docs/upstream-prs.md`
- fork 关键提交：`5410123`、`b044217`、`0d88af9`、`6da9533`、`896785c`、`f8dd83e`、`67b75b8`、
  `37a5d1e`、`220dd7d`、`5b6482a`、`42bc593`
- 当前生产 server：`./ds4-server -m gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf -c 393216 --port 8055
  --ssd-streaming --power 100 --warm-weights --vision ... --dspark --mtp-model ...`
