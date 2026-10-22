# ds4 的做法 → omlx 可借鉴性评估（反向）

> 评估日 2026-10-18。只做方案/代码层评估：**不加载模型、不做新 A/B**。
> ds4 侧证据：本仓库 `V41_SSD_STREAMING_ROADMAP.md`、`V41_STREAMING_ENGRAM_SPEED_PLAN.md`、
> `speed-bench/M3/`、`ds4.c`；omlx 侧证据：`~/omlx-notes/`、`~/omlx` 源码
> （`omlx/patches/deepseek_v41/moe_offload.py`、`language.py`、`scheduler.py`）。
> 当前 ds4 的旧 checkpoint 已删除、新 checkpoint 未下载；omlx 侧也没有为此新开窗口。
> 本文所有数字都是既有文档/日志的引用，**不代表当前二进制复测**。

---

## 0. 一页结论

ds4 对 omlx 的可借鉴点**不在 kernel，而在缓存组织与长 prefill 驻留**。按价值排序：

1. **CED 感知的每层容量配额（encoder/decoder 借用）**——唯一候选，但 **2026-10-18
   proxy 复核后判负**。omlx 现在每层固定 146 槽；CED 下 decoder 半只算 128 行尾，
   看似有余量。但现有 93-token decode 窗口的 decoder distinct 中位数已是 181，
   128 行尾只会更大，146 槽没有余量；借槽会先伤 decoder。**预期 0，不改
   `ExpertOffloadPlan`**；真 prefill trace 未抓（模型重启后未加载，env 未设），
   hook 片段在笔记（未入树），脚本已备。这是 ds4 `ds41_encoder_acquire` 的“部分版”，
   但内存/工作集算术不成立。
2. **通用的每层可变容量（跨层借用）**——同一套机制的一般化。omlx 每层固定 146，
   ds4 是全局 7822 槽、per-layer cached 162–232；层间 hotness 有偏斜。静态配额
   预期 **3–8pp 命中 / 2–5% 墙钟**，但需要 per-layer capacity 的 plan 改造，工作量中高。
3. **预填种子的 recency 加权**——ds4 的 `ds41_prefill_seed` 给最近 32 token 一个
   `count × topk` 的大权重；omlx 的 `seed_hot_set`/`refresh_pins` 只用全史计数。
   小改动，预期 **1–5%**，低风险。
4. **确定性 prefill 几何**——ds4 的 `ds41_prefill_count` 一次算出 chunk 数，配
   `carry_cap`/prefill reserve；omlx 的 `_prefill_step_size_for_progress` 会因
   contention/memory guard 逐 chunk 缩，导致同一 prompt 2 vs 5 chunk、TTFT 2×。
   直接速度收益 ~0，但能把 A/B 噪声和尾延迟压下来。
5. **bit-exact / 多 frontier logits 门**——ds4 的方法论；omlx 的 CED/offload 变更
   可以借用“多 frontier logits + 固定 prompt”的验收形状。MLX 未必逐位可复现，
   用紧容差/argmax 等价替代。

**不借鉴/已等价**：LFU 默认（omlx IT-16 已测：warm +4.3%、cold 更差）、staging/
read-ahead（omlx 已有/已证负）、CED/wide/MTP（omlx 更靠前或形态不同）、容量扩容
（128GB 内存墙）、单会话串行/tool-call 精确重放（omlx 架构不同）。

---

## 1. 口径差异

| 维度 | ds4 | omlx |
|---|---|---|
| 引擎 | C + Metal/CUDA | Python + MLX |
| 专家缓存 | **全局共享池**（生产 7822/15360） | **每层固定 slot 池**（146/384） |
| 每层容量 | `effective_cap` 返回 384，全局 prune 管总量 | `ExpertOffloadPlan.capacity` 全层同一个值 |
| 层间偏斜 | per-layer cached 162–232、evictions 39–175 | 每层固定 146，热层无法借冷层的槽 |
| prefill 调度 | `ds41_prefill_count` 一次定 path/rows；medium/wide/encoder/CED | `_prefill_step_size_for_progress` 逐 chunk 可缩 |
| 长 prefill 驻留 | `ds41_encoder_acquire`：借 cache 预算 wire-lock encoder 0–19 | CED：decoder 半只算尾行；**无 encoder 驻留** |
| 数值契约 | 逐位/多 frontier logits/ bind parity | 紧容差/单测为主 |

同模型不同引擎、不同量化，**ds4 的 0.996 命中率不能直接当 omlx 的预期**；
本文只借机制，不借数字。

---

## 2. ds4 的结构性设计（为什么值得看）

### 2.1 全局专家缓存 + 每层统计

`ds4_metal.m:15217` 的 `effective_cap` 注释写得很直白：

> The residency policy is global: every layer can use any expert slot it routes to,
> and global pruning decides which existing entry is least valuable. A per-layer cap
> made cache size depend on model depth rather than the actual byte budget.

生产 M3-3 的 per-layer 统计（cache 7822）：

| 指标 | 范围 | 含义 |
|---|---|---|
| cached | 162–232 | 热层比冷层多用 ~70 槽 |
| evictions | 39–175 | 热层驱逐 ~4.5× |
| hit_rate | 0.994–0.997 | 全局 LFU + 播种后的稳态 |

omlx 的每层固定 146 槽正好是 ds4 注释里批评的“per-layer cap”：热层不够、冷层闲置。
omlx 的 `moe-hotness` 端点已经暴露 `distinct_experts_per_layer` 和
`layers_above_capacity`，可以直接量化这个偏斜。

### 2.2 `ds41_prefill_seed`：每层、按当前请求、带 recency

`ds4.c:42849`：

- 每层 sweep 完，统计该层当前请求的 selected ids 频率；
- 最近 32 token 的引用额外加 `count × DS4_N_EXPERT_USED` 的大权重；
- 取 top `configured_count / DS4_N_LAYER`（生产 ~195）个，seed 进全局 cache；
- seed 的 priority 写进 route hotness/use_count，后续 LFU 在 16-token 衰减前会保它们。

omlx 的 `seed_hot_set`（`moe_offload.py:1188`）是同类机制，但：

- 默认 `top_k=0`，关闭；实测 K=4 −1.9% s/tok、K=8 持平；
- 排序用 `prefill_hotness`（miss 引用），没有 recency 权重；
- 每层独立；seed 发生在该层最后一个 block，使用该层本次 prefill 的 miss 计数，
  但没有 recency 权重。

**可借的点**：recency 加权。ds4 的“最近 32 token 权重大”是对“下一个 chunk 更像
刚读过的 token”的朴素先验；omlx 的长 prompt 多 chunk 场景同样适用。

### 2.3 `ds41_encoder_acquire`：长 prefill 的 encoder 驻留

`ds4.c:42984`：

- 条件：`streaming && remaining >= 16384`，且没有 wide/carry；
- 把 expert cache budget 设为 0（借走 decode cache），wire-lock encoder 层 0–19 的
  routed expert 权重（约 20×384×9.49 MiB ≈ 71 GiB）；
- decoder 半可以 defer；sweep 完恢复 cache budget；
- 效果：encoder 层只读一次，不被后续 sweep 重新换入。

omlx 的 CED 只做了“decoder 半只算尾行”，**没有 encoder 半驻留**；encoder 层在
多 chunk prefill 里仍会重复读 expert。omlx 的每层 146 槽装不下 384 个 encoder
expert，所以第二次 chunk 仍会 miss 大部分。

### 2.4 `ds41_prefill_count`：一次定几何

`ds4.c:42904`：

- `remaining < minimum` → token-major（minimum 32/256/1024，取决于 cache）；
- `< 2048` 且 gather 准入 → gather sweep；
- `> 2048 && < 8192` 且 `carry_cap` → medium sweep（一次 layer sweep）；
- `>= 4096` → wide sweep；
- `>= 16384` 且无 carry → encoder 常驻；
- `>= 4096` → CED。

这些判断在进入 sweep 前一次完成，sweep 内部不再因瞬时 contention 改 step。
omlx 的 `_prefill_step_size_for_progress` 每 chunk 都可以被
`_contended_prefill_cap`/memory guard 缩小；同一个 8.4K prompt 因此出现 2 vs 5 chunk、
TTFT 29.35 vs 60.86 s（2.07×）。

### 2.5 bit-exact / 多 frontier logits 门

ds4 的验收形状：`--moe-bind-parity`、`--dump-frontier-logits-dir`、logprob vectors、
golden manifest；任何 cache/调度改动都必须过“同一 prompt 多 frontier 的 logits 逐字节”。
omlx 的 CED 验收用了 greedy diff（math 相同、summary 1-token tie-flip），比 ds4 弱一档；
但 MLX 的跨设备逐位复现未必可行，可以退成“固定 prompt + 多 frontier + 紧容差/argmax”。

---

## 3. 可借鉴项逐条评估

### 3.1 CED 感知的每层容量配额（首选候选；2026-10-18 proxy 后判负）

**问题**：omlx 每层固定 146 槽。CED 下：

- encoder 层 0–19：仍处理全部行，working set 170–314；
- decoder 层 20–39：只处理 `ced_tail = window_size`（128）行尾，看似 working set 小。

**2026-10-18 22:30 复核（decode 窗口 proxy）**：现有 `it16_warm_trace.json` 8 个
episode、每层 93 个 decode 调用；decoder 层 20–39 的 distinct 为
**min 130 / median 181 / max 246**，160 个层-样本里只有 18 个 <146。CED 的
decoder 只跑 128 行尾，128 行的 distinct 只会 ≥ 93 行窗口；因此 decoder 层
大多数没有余量，借槽会先伤 decoder。**结论：proxy 不支持 CED 配额；不改
`ExpertOffloadPlan`。** 真 prefill trace 未抓（模型重启后未加载，env 未设）；
hook 片段在笔记（未入树），脚本已备，日后要复核可直接加回。

**如果仍要复核**（不推荐先做）：

**做法**（按代价从低到高）：

1. **静态 CED 配额**：prefill 时给 encoder 层 `capacity_hi`、decoder 层
   `capacity_lo`，满足 `20×hi + 20×lo = 40×146`。例如 `hi=196, lo=96`。
   需要 `ExpertOffloadPlan` 支持 per-layer capacity（当前 `plan.capacity` 是标量）。
2. **按层 working set 标定**：用 `moe-hotness` 的 `distinct_experts_per_layer`
   或生产 trace，给每层一个静态配额；CED 只是其中一个特例。
3. **动态借用**：prefill 前按 CED 开关和当前 prompt 长度重算配额；decode 恢复
   均匀 146。动态版收益更大，但要处理 plan/数组重建。

**预期**：**0（proxy 不支持；真 prefill 未测但方向一致）**。零额外内存的前提是
decoder 层有余量，而 93-token 窗口已经证明大多数 decoder 层没有；**这是 ds4
`ds41_encoder_acquire` 的“部分版”，但内存/工作集算术不成立**。ds4 借走整个
cache 并 wire-lock 全部 encoder expert；omlx 只借 decoder 层的余量，但余量不存在。

**风险**：encoder 层容量变大后，块边界/`stream_capacity`/pin 窗口都要按层算；
`_admission` 的 `resident_bytes_at(capacity)` 要改成 per-layer 求和；decode 恢复
配额时可能触发一次重排。建议先做**静态配额 + 纸面回放**，不要一上来做动态。

### 3.2 通用每层可变容量（跨层借用）

**问题**：即使没有 CED，omlx 的层间 working set 也有偏斜（ds4 的 cached 162–232
是直接证据；omlx 的 `moe-hotness` 可验证）。固定 146 让热层不够、冷层闲置。

**做法**：

- 先做**静态配额**：按离线 trace 的每层 distinct/hit 曲线，解一个“总槽数固定、
  最大化总命中”的小优化；配额粒度 8–16 槽，避免 plan 碎片。
- 再做**动态借用**：prefill/decode 分模式，或按最近 N 次调用的层 hotness 调整。
- 不建议直接做“全局池”：omlx 的每层 expert 是独立的 MLX module/数组，真正的
  全局 slot 池要跨层共享 buffer 和地址表，改动远大于 per-layer capacity。

**预期**：静态配额 **3–8pp 命中 / 2–5% 墙钟**；动态版可能更高，但风险/工作量也高。
**先决条件**：`moe-hotness` 数据显示确实有“部分层 above capacity、部分层 below”
的偏斜；如果所有层都 above，配额无用，只有加容量。

### 3.3 预填种子的 recency 加权（小改动）

ds4 的 `ds41_prefill_seed` 对最近 32 token 加 `count × topk` 权重；omlx 的
`seed_hot_set` 只按全史 miss 计数排序。建议：

- 在 `_ExpertSlots` 里加一个 `prefill_recent_hotness` Counter，每个 block 更新；
- `seed_hot_set` 的排序分数改为 `prefill_hotness + α × recent`（α 可 env）；
- 或者直接把它接到 `refresh_pins` 的分数里（`prefill_hotness + prefill_hit_hotness
  + decode_hotness + α × recent`）。

**预期**：**1–5%**，低风险；用现有 `OMLX_DSV41_EXPERT_SEED_TOP_K`/pin A/B 框架
即可测。注意 omlx 的 seed 默认关闭、K=4/8 收益小；recency 可能让 K=16/32 重新有值，
但不要期待大杠杆。

另外两个 ds4 细节值得一起 A/B：

- **seed 排序用全部引用，不只 miss。** omlx 的 `seed_hot_set` 用 `prefill_hotness`
  （miss 引用）；ds4 的 `frequency[]` 统计所有 selected ids。一个一直命中、最后被
  tail 驱逐的热专家，在 miss-only 排序里可能排不进去。建议对比
  `prefill_hotness` vs `prefill_hotness + prefill_hit_hotness`。
- **seed 目标 K 补扫。** ds4 每层 seed `capacity/layers`（生产 ~195）；omlx 只扫过
  K=4/8。可补 K=32/64/128/146 的 **seed-only（不 protect）**，看是否比 pin 128
  更稳（pin 的失败在 protect 压小动态池，不在 seed 本身）。

### 3.4 确定性 prefill 几何（测量/尾延迟）

omlx 已经在 2026-10-17 加了 `[prefill-geometry]` 和每请求 `chunks=`（`4abf902a`），
这是第一步。还差：

- **计划一次、执行一次**：请求开始时算出 `planned_steps` 列表（含 wide/CED/guard
  预留），执行中除非硬内存拒绝，不逐 chunk 改 step；
- **planned vs actual 告警**：两者不一致时打 warning，并把该 run 标成不可比；
- **prefill reserve**：像 ds4 一样预留固定 headroom（ds4 的 `prefill expert reserve`
  7.12 GiB），而不是让 guard 在 chunk 边界临时缩。

**预期**：直接速度 ~0；但能消掉 omlx 的 2× TTFT 方差和尾延迟，A/B 可信度提升。
这是“测量基建”，不是“提速项”。

### 3.5 bit-exact / 多 frontier logits 门（质量基建）

ds4 的门：同一 prompt 在 2048/10240/18432/26624/32768 等 frontier 上全量 logits
逐字节；omlx 的 CED 只用 greedy diff。建议 omlx 对 CED/offload/attention 改动加：

- 固定 3–5 个 prompt（代码/中文/数学），每个 3–5 个 frontier；
- 记录 top-k logits / argmax / NLL，紧容差（如 1e-4）或 argmax 等价；
- 不追求逐位（MLX/Metal 跨设备不保证），但要求“同一设备同一 build 可复现”。

**预期**：0 直接速度；减少“速度赢了但数值悄悄漂”的风险。omlx 已有
`test_deepseek_v41_attention_rounding` 等，可复用。

---

## 4. 不借鉴 / 已等价

| ds4 做法 | omlx 现状 | 判定 |
|---|---|---|
| 全局 LFU + 16-token 热衰减 | IT-16 已测 LFU：warm +4.3%、prefill +3.3–4.8pp、cold 更差；默认 LRU | **不换默认**；LFU 是 opt-in，pin 是当前 workaround |
| expert cache staging 零拷贝 | omlx staging 池已有；ds4 是直接 pread 进 slot | 已等价 |
| prefill read-ahead | omlx 已有 block 间 read-ahead；ds4 P0g 层间预读负 | 已等价/不适用 |
| CED / wide 8192 | omlx 已有 CED + wide 8192 | 已领先 |
| MTP + offload | omlx 已有；ds4 因 `a≈0.68` 关闭 | 形态不同，不借 |
| 容量扩容 | omlx IT-16：146→161 +7.8% decode，但 +7.3GB；128GB 墙 | 内存墙，非代码 |
| 单会话串行 / disk KV checkpoint | omlx 多会话 + 分层 KV | 架构不同 |
| tool-call 精确重放 | omlx 已有 copy/prefix 复用（copy 82–83%） | 已等价或收益小 |
| bit-exact 逐字节 | MLX 跨设备不保证 | 退成紧容差/多 frontier |
| `ds41_prefill_seed` 全量 seed | omlx `seed_hot_set` 默认 0，K=4/8 收益小 | 可试 recency，不照搬 K |

---

## 5. 建议执行顺序（先零模型，后上机）

1. **纸面量化层间偏斜（零模型）**：用 omlx 的 `moe-hotness` / `moe-decode-trace`
   历史数据，或生产 trace，统计每层 `distinct_experts`、hit/miss、evictions。
   判断是否存在“部分层 above capacity、部分层 below”。如果所有层都 above，
   可变容量无用，直接关账。
2. **CED 配额纸面回放（零模型）**：用 `it16_eviction_replay.py` 的框架，模拟
   `hi=196, lo=96`（或其他组合）在 CED 长 prompt 上的命中/字节/驱逐；目标函数
   `misses + λ×load_calls`。**纸面 ≥3pp 命中或 ≥2% 墙钟才动 plan 改造。**
3. **recency seed A/B（需模型）**：在 `seed_hot_set` 加 recent Counter，扫
   `α ∈ {0.5, 1, 2}`、`K ∈ {16, 32, 64}`；用现有 pin/seed 框架，ABBA ≥4 轮。
4. **确定性几何（零模型 + 上机各半）**：先加 `planned_steps`/`actual_steps` 日志
   和告警；再考虑 prefill reserve。不要先改调度。
5. **bit-exact/多 frontier 门（零模型）**：加固定 prompt 的 logits/argmax 基线，
   给 CED/offload 变更用。

**不建议先做**：通用全局池（改动太大）、完整 encoder 驻留（内存算不过来）、
LFU 默认（已测）、容量扩容（内存墙）。

---

## 6. 预期收益台账（诚实区间）

| 项 | 预期 | 依据/前提 |
|---|---|---|
| CED 感知每层配额 | **0（proxy 不支持）** | decoder 93-token distinct median 181 >146；128 行尾更大；不改 plan |
| 通用每层可变容量 | **3–8pp 命中 / 2–5% 墙钟** | 层间偏斜；静态配额；高工作量 |
| recency 加权 seed | **1–5%** | ds4 的 last-32 权重；omlx K=4/8 收益小，需重扫 K |
| 确定性几何 | 0 直接；消 2× 方差/尾延迟 | omlx 已有 geometry 日志；计划一次执行一次 |
| 多 frontier logits 门 | 0 直接；质量基建 | ds4 方法论；MLX 用紧容差 |
| LFU 默认/更快衰减 | 0–4%，已测 | IT-16 warm +4.3%、cold 更差 |
| 完整 encoder 驻留 | 0（大概率装不下） | 20×384×12.37 MiB ≈ 95GB；128GB 不可行；Studio oQ4e ≈146GB 也不够 |
| 全局池 | 5–10% 理论，但改动过大 | 不建议先做 |
| 容量扩容 | +7.8% decode，+7.3GB | 内存墙，非代码 |

**一句话**：ds4 给 omlx 的最大启发是“**缓存容量应该按层/按阶段借，而不是每层一刀切**”；
但 **CED 感知的 encoder/decoder 配额已被 decode 窗口 proxy 判负**（decoder 没有余量）。
剩下的低成本项是 **recency 加权 seed**；其余要么已等价，要么是内存/架构墙。

---

## 7. 参考

### ds4

- `V41_SSD_STREAMING_ROADMAP.md` §1.1/§1.3/§2.2/§3
- `V41_STREAMING_ENGRAM_SPEED_PLAN.md` §0/§6.5/§9
- `speed-bench/M3/m3-3/run-20261012-010604/`（per-layer 统计、M3-3 A/B）
- `ds4.c:42849`（`ds41_prefill_seed`）、`ds4.c:42904`（`ds41_prefill_count`）、
  `ds4.c:42984`（`ds41_encoder_acquire`）、`ds4_metal.m:15217`（全局 cache 策略）、
  `ds4_metal.m:1083`（16-token 热衰减）

### omlx

- `omlx/patches/deepseek_v41/moe_offload.py:559`（`ExpertOffloadPlan.capacity`）、
  `:979`（`_ExpertSlots`）、`:1156`（`refresh_pins`）、`:1188`（`seed_hot_set`）、
  `:1800`（prefill block loop）
- `omlx/patches/deepseek_v41/config.py:106`（`ced_prefill`）、
  `language.py:789`（CED tail）、`language.py:974`（`ced_tail` 计算）
- `omlx/scheduler.py:6547`（`_prefill_step_size_for_progress`）、
  `:3379`（`_dsv41_wide_prefill_step`）
- `~/omlx-notes/dsv41_offload_state_and_levers_2026-10-17.md` §9.1–9.14
- `~/omlx-notes/it16_eviction_policy_findings.md`（LFU/capacity）
