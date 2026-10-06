# V4.1 routed 路径：下一步执行计划（2026-10-06 交接）

> 给新 session 的自包含交接。**不需要读上一轮对话**，本文件带齐上下文、路径、行号、
> 命令、验收与停线条件。按 §5 的顺序做；每个任务做完就停下来报数，不要连做。
>
> 工作区：`/Users/xieyongliang/ds4`（下称 `$R`）。起始 HEAD = `fork/main` = `95861cc`。

---

## 0. 一句话

上一轮把 V4.1 decode 的 routed 桶（每 token ~20 ms，占整步一半）拆开了：
**gather 内核不慢（353 GB/s = 1285 G 权重/s，全引擎最高），主机 paging 决策免费（−0.01 ms）**；
桶内是 **内核 6.6 ms + 逐层排空 6.9 ms + 专家缓存未命中 6.0 ms**。
本轮要做的只有两件事：**用最便宜的办法给"排空"定价（T1）**，以及**确认并优化未命中路径（T0/T2）**。
T3（排空的结构性修复）是条件任务，要先看 T1 的数再决定，且必须先拿用户裁决。

---

## 1. 已确立的事实（别重做）

配置：V4.1 Q2 / SSD-streaming / ctx 4096 / prefix 1024 / 缓存目标 82 GiB → 8078 experts，
计划内存 93.47 GiB，实测峰值 resident 75.1 GiB。

| 事实 | 数 |
|---|---|
| 一个 decode 步 D | **40.2–43.0 ms**（同配置两轮，轮间差 7%，只在同一轮内比较） |
| 每 token 权重流量 | **9.78 GiB / 15.816 G 权重触摸** |
| 稳态物理读盘 | **0.8 MB/token = 0.02%**（`proc_pid_rusage` 的 `ri_diskio_bytesread`） |
| 六段隔离测量（走真实 decode 路径） | attention projections 4.62 + core 3.27 + output 5.98 + shared 3.37 + hc/norms 1.75 + head 1.20 = **20.19 ms** |
| routed 桶（= D − 六段） | **≈20–22 ms** |
| 桶内分解 | 内核 6.6 + 逐层排空 6.9 + 未命中 6.0 + router/胶水 ~0.6 |

### routed 分臂实测（`--routed-split-ssd`，4 次取最好，ms/token）

| 臂 | 说明 | ms/token | GB/s |
|---|---|---|---|
| A | gather only，单个 command buffer | **6.76** | **353** |
| B | gather only，每层 flush（commit 不 wait） | 7.05 | 339 |
| C | A + 每层 `begin_selected_load` | **6.75** | **354** |
| E | 只做每层排空、不派发 | 0.87 | — |
| D | gather + 每层 `end_commands`/`begin_commands`（引擎实际形态） | **13.62** | 175 |

- routed 每 token 搬 2.22 GiB（6/384 专家，9.49 MiB/个：gate 2.90 + up 2.90 + down 3.69）。
- **A = 8.494 G 权重 / 6.61 ms = 1285 G 权重/s**，是 dense Q8 `q_b`（row-1 517、边际行 700 G/s）的 1.8–2.5×。
- **C − A = −0.01 ms ⇒ 主机 paging 决策免费。** 回读拷贝也免费（`copy_avg 0.000`）。
- 此前"routed 116 GB/s、比 dense 慢 6×"是**按字节**读一个**按权重**受限的引擎，作废。

### 真实路径的主机侧计数（`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`，64 步窗口）

| 计数 | 值 | 说明 |
|---|---|---|
| `sync_avg` | 0.823 ms × 40 层 = **33.1 ms/token** | 全在 `ds4_gpu_end_commands()` 里 |
| `bind_avg` | 0.149 ms × 40 层 = **5.95 ms/token** | 槽绑定 + prune + addr 准备 |
| `load_calls` | **386 / 64 token = 6.03/token** | `cache_all_resident 2174`、`cache_mixed 386`、`cache_all_missing 0` |
| `load_prepare_avg` / `load_pread_avg` | 0.385 / 0.611 ms | 每次装载 ≈1.0 ms |
| `readahead_total` | 133 ms / 64 = 2.1 ms/token | |
| `DS4_METAL_V41_DECODE_HOST_PROFILE` | **layers 43.401 / D 43.01 ms** | 整步都在层循环里；1969.3 compute encode + 139.1 blit/步 |

### 机理（源码级，已核对）

- V4.1 decode 步 = `ds41_graph_step`（`ds4.c:41834`）→ 逐层 `ds41_graph_layer`（`ds4.c:41521`）
  → `ds41_moe` → `ds41_moe_partial`（`ds4.c:41029`）→ **`ds4_gpu_routed_moe_one_tensor`**
  （`ds4_metal.m:43399`），`force_resident = !g->streaming` = **false**（`ds4.c:41101`）。
- **V4.1 步不走** `metal_graph_encode_decode_layer_phase`（`ds4.c:24104`）里那套
  `metal_graph_selected_async_load_*` 异步回读；那套只服务通用 decode 路径。
- `one_tensor` 的 selected-id 分支链在 `ds4_metal.m:44602` 起。**没有任何 override 被 arm**，
  所以每层落进最后的 `else`（`ds4_metal.m:44654`）：
  1. `ds4_gpu_end_commands()`（`44661`）← **逐层排空，每 token 40 次**
  2. `ds4_gpu_tensor_read(selected, …)`（`44671`）← 回读 6 个 id
  3. `ds4_gpu_begin_commands()`（`44685`）
- 排空的存在使 `ds41_graph_step` 的 `queue_layers` / 逐层 `flush_commands`（`ds4.c:41954/41987`，
  注释明说"40 CPU↔GPU round trips per token become one"）失效。
- `DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` 看着像"免回读"的答案，
  **实际不是**：`ds4_gpu_stream_full_expert_addr_table_prepare`（`ds4_metal.m:16845`）用
  `ds4_gpu_wrap_model_exact_range_owned` 绑**整层** 384 个专家（3.64 GiB/层），是 resident 形态。

---

## 2. 已关闭的路线（别重开，别"好心修坏"）

- **投机解码整类**（n-gram / MTP / DSpark / 任何一步猜多 token）：oracle 上界 1.387×，实测 1.000×。
- **降比特重量化**：引擎是"每秒权重数"受限，Q2 每权重已与 Q8 同速。
- **加缓存 / 加内存 / SSD 调优 / 预取 / 减磁盘占用**：盘只占 0.02%。
- **抬高 `DS4_TP_BATCH_MAX_ROWS`**：8→32 无悬崖，已实测；上限已还原为 8。
- **修 `--stage-timing` 第 4 段**：按构造测不到（地址表未填充）。
- **gather 内核微优化**：它是全引擎每权重最快的，只占桶的 1/3。
- **`DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE` 当 streaming 解法**：见 §1 末。

---

## 3. 铁律（违反即事故）

1. **位契约高于一切**；改默认需用户裁决。新 env 一律默认关。
2. 红卡（闸门失败 / 新红灯）→ **停线写选项交用户裁决**，不许自行修绿。
3. **禁止启动生产 ds4-server（8055）**。用户可能在跑。
4. **禁止派发子代理**（provider 为 `ds4` / `omlx-local` 时一律禁止 `subagent` / `subagent_fork` / `workflow`）。
5. 动了 `ds4.c` / `ds4_gpu.h` / `ds4_metal.m` ⇒ **全量重建** `ds4 ds4-server ds4-bench ds4-eval
   `ds4-agent ds4_test` + 重跑零模型验收集（见 §5 通用验收）。
6. A/B 一律 `--dump-logprobs` 逐 step 比对（**禁止**用文本 / argmax 一致当验收）。
7. 任何加载模型 / 长上下文的命令：先查三件套（可用内存 / swap / 有无 LLM 进程驻留），
   后台跑 + 盯内存，swap 涨就 kill，**绝不允许 OOM**。
8. 输出里**不要出现尖括号+竖线形式的协议标记字面量**。不要用 `sandbox_permissions`。
9. 台账纪律：先提交代码，再更新 `LOCAL_INVENTORY.md`（B 行引用 SHA + D 段写结论），然后 push fork main。
10. **`LOCAL_INVENTORY.md` 工作区版本属于另一条工作线，别提交、别还原、别覆盖。**
    要更新台账就用"从 HEAD 重建 + `git hash-object -w` / `git update-index --cacheinfo` 直写索引"
    的方式（`a239d38`、`95861cc` 是这么做的，可 `git show` 参考）。目标 md5 保持
    `c607115b20005a4fbfa34f3ebd672e87`。同理 `PR1178_*.md` 是未跟踪文件，别提交。

---

## 4. 仪器与配方（原样抄）

### 4.1 harness

`tests/test_deepseek41_dspark.c`（`#include "../ds4.c"`，可访问引擎内部）。

```sh
make tests/test_deepseek41_dspark
```

| 模式 | 用途 |
|---|---|
| `--routed-split-ssd PROMPT` | **本轮主力**：跑 72 步真实 decode 报 D，再跑 §1 的 A/B/C/D/E 五臂 |
| `--verify-scan-ssd PROMPT` | n 行 verify 墙钟 vs 1 行 decode、V/D、a、进程自身读盘 MB/token |
| `--stage-timing PROMPT` | 七段分别计时（第 4 段无意义，见 §2）；配 `DS4_TEST_SSD=1` 走 streaming |
| `--weight-inventory` | 按显式张量名分组报 bytes 与权重数 |

### 4.2 关键 env

| env | 作用 |
|---|---|
| `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` | 引擎关闭时打印流式专家缓存全量计数；配合 `ds4_gpu_print_memory_report` 可打"窗口增量" |
| `DS4_METAL_V41_DECODE_HOST_PROFILE=1` | 每 64 步打印 `hash+engram / layers / logits / between steps` 主机耗时 + 每步 encode/blit 数 |
| `DS4_METAL_SELECTED_PROFILE=1` + `DS4_METAL_SELECTED_PROFILE_LAYER=N` | 逐层打印 `path=` / `mode=` / `ids=`（`readback` vs `override`）/ `read=` / `bind=` / cache 计数 |
| `DS4_TEST_STREAMING_CACHE_GIB` / `DS4_TEST_STREAMING_CACHE_EXPERTS` | 测试进程的缓存预算 |

### 4.3 标准窗口命令（本轮主力）

```sh
cd /Users/xieyongliang/ds4
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt
```

单次约 6 分钟、峰值 resident ≈75 GiB、计划 93.47 GiB。**后台跑 + 盯内存。**

### 4.4 通用验收（任何改了引擎文件的一轮都要做）

```sh
make                       # 全量重建，必须零错误
./ds4_test --server        # 零模型验收，必须 "ds4 tests: ok"
for t in test_metal_graph_capture test_metal_ssd_experts test_metal_nax_tensor_units; do
  ./tests/$t; echo "$t rc=$?"; done
```

已知**既有红项（非本轮引入，别追）**：`tests/test_deepseek41_fusions`（matvec_bf16 与 Q8_0 位不等）、
`tests/test_deepseek41_engram_admission`（step-reader 准入断言）；`./ds4_test` 全量在默认模型缺失时
于 `qwen4-prefill-checkpoints` 退出（环境资产红）。

---

## 5. 任务

### T0 — 确认稳态未命中率（**随 T1 免费得到，不单独开窗**）

**为什么**：桶内的 6.0 ms 未命中项完全取决于"6 次装载/token"这个数。它是本机 82 GiB / 8078 experts
配置下测到的，但**生产命令用的是 `--ssd-streaming-cache-experts 10000`**，且
`PR1178_HANDOFF.md` 记录过"长 prompt 纯热态 OFF/ON 逐字节相等"，暗示某些工况下可能 0 miss。
**如果稳态未命中 ≈0，T2 整条不存在。**

**怎么做**：`--routed-split-ssd` 在 `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1` 时会在
decode 窗口前后各打一次 `ds4_gpu_print_memory_report`，直接读 `delta` 行的
`load_calls`、`cache_all_resident`、`cache_mixed`、`cache_all_missing`，除以窗口 token 数（64）。

**判定**：
- `load_calls / 64 ≥ 3` ⇒ 未命中项真实存在，T2 立项。
- `< 1` ⇒ **T2 关闭**，在台账写清"稳态无未命中，6.0 ms 项不成立"，只做 T1/T3。
- 1–3 ⇒ 先按 T2.0 加计时再定。

**顺带**：用同样的数复核 §1 的 `bind_avg 0.149`。若 `bind_avg × 40` 与 `load_calls × 1.0 ms` 对不上，
说明 `bind` 里还有未计入的项，T2.0 的探针要覆盖到。

---

### T1 — 排空探针：给"逐层排空"定价（**先做这个，最便宜、最决定性**）

**为什么**：§1 的 D − A = 6.86 ms 是**孤立测量**（空 CB 里只放一次 gather）。真实步里排空窗口内
本来就有必须等的 GPU 工作，所以 6.86 ms 里有多少是**净损失**没测过。
而唯一一次结构正确的实现尝试（card I / PR #1178，GPU 事件门控 service 线程）只拿到 **+2.7~3.2%
≈1.3 ms**。两者差 5×，必须解释。

**目标**：在**真实 decode** 里做一次"去掉逐层排空"的消融，直接读出 D 的变化。
**这是一个纯计时探针：输出必然错误（用陈旧 ids），默认关，永不提交、永不入 PR、用完即删。**

#### T1.1 实现（改 `ds4_metal.m` 一个文件，约 40 行）

插入点：`ds4_metal.m:44654` 的 `else {` 分支（`g_routed_moe_selected_override_n = 0;` 那一行之后）。

新增文件级静态（放在 `ds4_gpu_routed_moe_one_tensor` 之前的文件作用域）：

```c
/* PROBE ONLY -- never committed. Prices the per-layer readback drain by
 * replaying a stale per-layer id set, so the ids are always resident and the
 * only variable is whether the drain happens. Output is wrong by construction. */
static int32_t g_probe_stale_ids[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER]
                                [DS4_METAL_MAX_ROUTED_EXPERT_USED];
static uint8_t g_probe_stale_have[DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER];
```

分支内改成（保持原有正常路径逐行不变，只是包一层）：

```c
} else {
    const char *probe_env = getenv("DS4_METAL_V41_PROBE_STALE_IDS");
    const int probe = (probe_env && probe_env[0]) ? atoi(probe_env) : 0;
    const bool probe_replay =
        probe != 0 && g_probe_stale_have[layer_index] &&
        layer_index < DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER &&
        n_expert <= DS4_METAL_MAX_ROUTED_EXPERT_USED;
    if (probe_replay) {
        memcpy(selected_ids, g_probe_stale_ids[layer_index],
               (size_t)n_expert * sizeof(selected_ids[0]));
        selected_id_source = "probe-stale";
        /* 关键：置 true 让内核也读到陈旧 ids。否则内核会直接从 g->selected
         * 读**真实** ids，而地址表是按陈旧 ids 填的 —— 输出同样是错的，但
         * 掩码/槽位语义不自洽。置 true 后走 ds4_gpu_stream_selected_ids_prepare
         * 把 ids 拷进 per-layer 的 g_stream_selected_id_buffers，与 override
         * 路径（ds4_metal.m:44617）行为一致。 */
        selected_exec_ids_from_host = true;
    }
    g_routed_moe_selected_override_n = 0;
    if (g_batch_cb != nil) {
        if (probe_replay && probe == 1) {
            /* skip the drain entirely: CB stays open, ids came from the probe */
        } else {
            /* ---- 原有代码，逐行不动：end_commands / tensor_read / begin_commands ---- */
        }
    } else {
        /* 原有 else 分支，逐行不动 */
    }
    if (probe != 0 && !probe_replay) {
        memcpy(g_probe_stale_ids[layer_index], selected_ids,
               (size_t)n_expert * sizeof(selected_ids[0]));
        g_probe_stale_have[layer_index] = 1;
    }
}
```

语义：
- **不设**：完全正常。
- **`=2`**：用陈旧 ids，但**保留** `end_commands`/`begin_commands`（对照组）。
- **`=1`**：用陈旧 ids，**跳过**这一对（实验组）。

注意：`probe == 1` 时不能调 `begin_commands()`（从没 `end` 过）；`g_batch_cb == nil` 时退化成正常路径。
`selected_id_source` 设成 `probe-stale` 便于在 `DS4_METAL_SELECTED_PROFILE` 里确认探针生效。

#### T1.2 编译与验收

改了 `ds4_metal.m` ⇒ 走 §4.4 全量重建 + 零模型验收。

#### T1.3 两条腿（每条一个窗口，各约 6 分钟）

```sh
# 对照组：陈旧 ids + 保留排空
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
DS4_METAL_V41_PROBE_STALE_IDS=2 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt 2>&1 | tee /tmp/t1_drain.log

# 实验组：陈旧 ids + 跳过排空
DS4_TEST_STREAMING_CACHE_GIB=82 DS4_TEST_STREAMING_CACHE_EXPERTS=8000 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 DS4_METAL_V41_DECODE_HOST_PROFILE=1 \
DS4_METAL_V41_PROBE_STALE_IDS=1 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --routed-split-ssd tests/long_context_security_prompt.txt 2>&1 | tee /tmp/t1_nodrain.log
```

**读数**：`DS4_METAL_V41_DECODE_HOST_PROFILE` 的 `layers` 字段（≈D，基线 43.401）。
两组都**必须**先确认：
- `DS4_METAL_SELECTED_PROFILE=1 DS4_METAL_SELECTED_PROFILE_LAYER=0` 打出 `ids=probe-stale`；
- 两组的 `load_calls` 数量级相同（陈旧 ids 固定了 240 个专家，未命中会**下降**，
  这正是设 `=2` 对照组的原因——两组都下降，差值是干净的）。

#### T1.4 判定表

| Δlayers = (`=2`) − (`=1`) | 结论 | 下一步 |
|---|---|---|
| **≥ 4 ms** | 排空确实是净损失，且 card I 的实现没把它拿回来 | T3 立项（先拿用户裁决） |
| 1.5–4 ms | 部分是净损失；card I 已拿掉一部分 | 只做 T2；T3 记为"低性价比，存档" |
| **≤ 1.5 ms** | **card I 的 +3% 已经是这个方向的全部** | **T3 永久关闭**，在台账写死机理 |

**做完 T1 无论结果如何：删掉探针代码，`git checkout -- ds4_metal.m`，全量重建回干净件。**

---

### T2 — 未命中路径（**条件：T0 显示 ≥3 misses/token**）

**为什么**：6.03 次装载/token × ≈1.0 ms = 5.95 ms/token（≈14%）。
注意这是**主机时间**、**逻辑 pread**：`miss_pread` 57 MiB/token，而进程自身物理读盘仍 0.8 MB/token，
页缓存吸收掉了。§1 的"盘不参与"不变，但**通往盘的那条路径本身要钱**。

**已知的子项**（来自 `..._TIMING_SUMMARY`）：

| 子项 | 值 | 备注 |
|---|---|---|
| `load_prepare_avg` | 0.385 ms | **其中只有 `prepare_buffer` 0.061 + `prepare_task` 0.000 有计时** ⇒ **≈0.29 ms/次未归因** |
| `load_pread_avg` | 0.611 ms | 9.49 MiB → 15.5 GB/s；`ds4_gpu_stream_expert_pread_pool_*` 已存在（默认上限 9 线程） |
| `load_install_avg` | 0.002 ms | 可忽略 |
| `reuse_scan_avg` | 0.062 ms（每次扫描 30720 条目） | 386 次 |
| `prepare_batch_reuse_avg` | 0.086 ms | |

#### T2.0 先把 0.29 ms 归因（**必做，不要跳过**）

在 `ds4_gpu_stream_expert_cache_begin_selected_load`（`ds4_metal.m:18511`）与
`..._load_selected_missing_with_source`（`ds4_metal.m:18807`）里补三个子计时，挂到现有
`ds4_gpu_stream_expert_timing_note_*` 体系（照 `note_prepare_buffer` 的样子加）：
1. 6 个专家的 residency 检查循环（`entry_matches` 那段）；
2. **`ds4_gpu_stream_expert_cache_prune_layer`（`ds4_metal.m:17331`）** —— 它是
   `while (layer_count > cap)`，**每次驱逐扫最多 `n_total_expert`(384) 条**，且**目前完全没有计时**；
3. **`ds4_gpu_stream_expert_cache_prune_global`（`ds4_metal.m:17984`）** —— 预算满时提前返回，
   但**一旦超过预算，每个 victim 扫 40×512 = 20480 条**。同样没有计时。

同时确认 `ds4_gpu_stream_expert_pread_pool_enabled()` 与
`ds4_gpu_stream_expert_pread_thread_count(18)` 在实际 miss 路径上是否真的走了线程池
（若退回 `ds4_gpu_stream_expert_pread_tasks` 单线程，这一项就有 6× 空间）。
`DS4_METAL_STREAMING_EXPERT_PREAD_THREADS` 可扫 1/3/9/18 做剂量-响应。

**预算**：1 次重建 + 1 个窗口。产出一张"6.0 ms 的构成表"。

#### T2.1 按 T2.0 的结果优化

按可能性排序的候选（**只做被 T2.0 证实的那个**）：
1. `prune_layer` / `prune_global` 的 O(N) 扫描 → 维护增量候选堆，或把 `cap` 抬到不必每层驱逐。
2. pread 线程数 / 池是否生效 → 剂量-响应选最优。
3. `prepare_load_buffers` 的 slab 分配路径。

**验收**：位契约不能动。优化只碰**缓存管理**，不碰 gather 的数学与绑定语义 ⇒
A/B 用 `--dump-logprobs` 逐 step 比对**必须逐字节相等**（`temp 0`，短 prompt 冷启动 +
长文热态两条腿）。性能用同一窗口的 `--routed-split-ssd` D 与 `layers` 字段 ABBA 对比。

**停线条件**：任何一步出现 logit 不一致 ⇒ 立即停线，回滚，写选项交用户裁决（铁律 2）。

---

### T3 — 排空的结构性修复（**条件：T1 显示 Δlayers ≥ 4 ms；且必须先拿用户裁决**）

**先说清楚为什么这不是小事**：
routed 的 6 个 id 是 GPU 算出来的，**主机必须先知道它们才能选缓存槽**。所以
"排空"不是 bug，是这个设计的必然。要去掉它只有三条路：

| 路 | 内容 | 代价 |
|---|---|---|
| (a) 把回读搬到 service 线程 + GPU 事件门控 | card I / PR #1178 已做过 | **实测只 +2.7~3.2%**，且该实现**位不一致、根因未钉死**（见 §6） |
| (b) 按层维护 384 槽指向**缓存 buffer** 的地址表 + 一个读 ids 算 mask 的小内核 | 新内核 + 缓存元数据重构 | 高；动 masked/split gather 绑定面 = **E0.3 登记红灯**（bind 轴 DRIFT ≤7.6e-06） |
| (c) 让路由可预判（预取下一 token 的 ids） | 需要预测器 | 未验证，风险高 |

**若用户批准做 (b)，动手前必须先：**
1. 读 `V41_DECODE_BOTTLENECK.md` §8 与 `LOCAL_INVENTORY.md` 的 E0.3 行；
2. 写出**位契约论证**：新表与旧槽绑定必须产出逐字节相同的 gather 输入；
3. 定验收：`--dump-logprobs` 逐 step 位等（短 prompt 冷启动 + 长文热态 + `MAX_LAYER` 式消融）；
4. 拿到用户对"改默认"的裁决（铁律 1）。

**默认预期：不做。** 见 §7 的性价比论证。

---

## 6. 明确不在本计划内：PR #1178 / card I

`~/ds4-cardi`、`~/ds4-pr-cardi`、`PR1178_HANDOFF.md`、`PR1178_SESSION2_RESULTS.md` 属于**另一条工作线**，
有自己的交接文档与用户裁决流程。**本计划不要碰它们。**

给新 session 的必要背景（避免误判"card I 已经验过，落地就行"）：
- card I（`DS4_METAL_ENABLE_V41_MOE_GPU_BINDING`，默认关）声称开/关 bit-identical、decode +2~3%。
- **第三方 Flor1an-B 报告 ON/OFF 非位一致，本机复现为真**（temp=0 各自确定，但 step 0 起 top-1 logit 不同；
  长文全热态逐字节相等 ⇒ 分叉只在**冷 miss 窗口**）。
- 第二轮排查 19 条腿，排除了 ids / 装载字节 / 调度竞态 / 内核族算术 / `useResource` 标记 / 缓存布局 /
  `didModifyRange` / `pread` 落盘方式；**根因未钉死**，收敛到两个嫌疑，作者自评 **(A) 分歧根本不在 routed MoE 里
  的概率约 70%**。
- 当前建议动作是 `PR1178_HANDOFF.md` §8（回复评论 + 关闭 PR #1178），**不是落地**。

**唯一与本计划相关的一点**：card I 的 +2.7~3.2% 是"把主机移出回读关键路径"的**实测上界**，
T1.4 判定表直接用它。它的位不一致不影响这个计时结论（计时腿本身不需要正确）。

---

## 7. 性价比：先读这段再决定做多少

- 六段隔离测量 + 桶内分解已经覆盖整步的 **97%** ⇒ **没有隐藏的大块**，余量只在 routed 桶内部。
- 桶内：6.6 内核（**不该动**）+ 6.9 排空（T1 定价，card I 实测上界 1.3 ms）+ 6.0 未命中（T2）。
- ⇒ **现实余量约 3~4 ms（7~9%）；乐观 8 ms（19%）；理论上限 13 ms（30%）。**
- **T3 的 (b) 是唯一能拿满 6.9 ms 的路，但它是新内核 + 缓存元数据重构，动的是登记红灯的绑定面。**
- **机器一变大，这个问题自动消失**：排空只因为 streaming 存在；≥192 GB 机器上模型常驻，
  没有缓存、没有回读。为一个随硬件贬值的问题投几周，回报递减。

**推荐执行力度**：T0 + T1 一定做（一个探针 + 两个窗口）。
T2 看 T0 的数。T3 **默认不做**，除非 T1 给出 ≥4 ms 且用户明确要榨这台机器的上限。

---

## 8. 内存与安全（本机踩过两次 swap 顶爆）

- 跑前查三件套：`vm_stat`（**页大小 16384，不是 4096**）算 free+inactive、
  `sysctl vm.swapusage`、`pgrep -fl 'ds4|llama|mlx|ollama|vllm'`。
- 本计划单腿口径：计划 93.47 GiB、实测峰值 resident ≈75 GiB。
  **free+inactive < 100 GiB 就不要开腿。**
- 后台跑（`run_in_background`）+ 定期盯 `vm.swapusage`；**swap 增长超过 ~1 GiB 立即 kill**。
- 一批一报备腿数，不要在"再加一条探针"的节奏里超支。
- 模型腿必须独占：ds4 会锁 `/tmp/ds4.lock`（`ds4.c` 的 `ds4_acquire_instance_lock`），
  用户的 ds4-server 会占锁 ⇒ **跑之前确认用户已停 server**。

---

## 9. 交付与台账

每完成一个任务：

1. **先提交代码**（探针代码**不提交**——T1 做完就 `git checkout -- ds4_metal.m`）。
2. 把结论写进 `V41_DECODE_BOTTLENECK.md`（新增一节，风格照 §8：先给表，再给机理，再给后果）。
3. 更新 `LOCAL_INVENTORY.md`：**B 行**引用 SHA，**D 段**写结论。**按铁律 10 的 index-only 方式**，
   不要碰工作区那份。
4. `git push fork HEAD:main`（fork = `williamxie1989/ds4`）。
5. 报给用户时给**裁决词**：做了什么 / 数是多少 / 建议下一步 / 需要用户裁决什么。

---

## 10. 一页速查

```
起点        HEAD = fork/main = 95861cc
主力仪器    ./tests/test_deepseek41_dspark MODEL --routed-split-ssd PROMPT
关键 env    DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1
            DS4_METAL_V41_DECODE_HOST_PROFILE=1
            DS4_METAL_SELECTED_PROFILE=1 DS4_METAL_SELECTED_PROFILE_LAYER=0
基线数      D = 43.01 ms / layers = 43.401 / 六段 20.19 / 桶 ≈21.5
            arm A 6.76(353 GB/s) C 6.75 E 0.87 D 13.62
            load_calls 6.03/token @ ~1.0 ms, bind_avg 0.149 ms, sync_avg 0.823 ms

T0  读 load_calls/64                 → ≥3 则 T2 立项，<1 则 T2 关闭
T1  探针 =1 vs =2，比 layers         → Δ≥4ms 则 T3 立项；≤1.5ms 则 T3 永久关闭
T2  T2.0 归因 0.29ms → 再优化        条件：T0 ≥3
T3  GPU 侧地址表                     条件：T1 Δ≥4ms + 用户裁决；默认不做
X   PR #1178 / card I                不在本计划内，见 §6
```
