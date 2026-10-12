# 多图长会话 vision 全量重编码分析与改进方案（2026-10-10）

> 症状：会话里传过图片后，**每一轮增量 prompt 都会把历史里全部图片重新过一遍 vision tower**，TTFT 每次多花 1 分钟以上，尽管 KV 前缀复用本身是命中的（`multimodal live kv hit ... identity=fingerprint-match`）。
> 结论：这不是"为避免全量 re-prefill 而必须付的代价"。KV 身份校验用的是指纹（解码像素的 SHA-256），**不需要像素塔的输出**；塔的输出只有在 prefill 需要评估图片 token 时才被消费。把"编码"推迟到复用决策之后、只对越过 KV frontier 的图片运行，即可把稳态 TTFT 里的 vision 时间降到 0。

---

## 1. 症状与证据

用户日志（1010 会话，8 图长会话，ctx≈110K）：

```
00:33:00 chat ctx=110720..110742:22 gen=75 TOOLS ... finish=tool_calls 2.682s
00:33:00 vision embedding encoded; cache=130774960   ← 下一请求，图1 miss
00:33:00 vision embedding encoded; cache=128649776   ← 图2 miss（put 时逐出，净减 2MB）
00:33:01 vision embedding encoded; cache=130774926   ← 图3 miss
00:33:01 vision embedding encoded; cache=49499746    ← 图4，cache 骤降 81MB（整清或大图逐出）
00:34:06 vision embedding encoded; cache=129500804    ← 00:33:01→00:34:06 的 65s 即大图塔耗时
00:34:07 vision embedding encoded; cache=129498992
00:34:07 vision embedding encoded; cache=129500580
00:34:08 vision embedding encoded; cache=130775182
00:34:08 multimodal live kv hit images=8 cached=110817 prompt=110908 identity=fingerprint-match
00:34:08 chat ctx=110817..110908:91 TOOLS prompt start
00:34:09 chat ctx=110817..110908:91 prefill chunk 91/91 ... 0.737s
```

要点：

- 每轮请求 8 张图 **全部 miss、全部重新 encode**（8 行 `encoded`，0 行 `cache hit`）。
- KV 复用完全命中：`cached=110817 / prompt=110908`，本轮真正 prefill 只有 91 个 token（0.737s）。**8 张图的塔输出没有任何一个字节被消费**——纯浪费。
- `cache=` 长期钉在 ~130MB（预算 128MiB=134.2MB）→ 每次 put 都在逐出旧条目，cache 处于容量临界点抖动。
- 对照早期日志（ds4_10080700.log，小图 5.6MB/张）：`encoded=12, hit=347`，命中率 97%。**同一机制，工作集小到装得下时一切正常**；图大、图多时崩塌。

## 2. 根因（三条，相互独立、相互放大）

### 根因 1（架构，主因）：parse 期 eager 编码，先烧塔再做复用决策

请求解析路径（客户端线程）无条件把**全部历史图片**过塔：

- [ds4_server.c:3651](ds4_server.c#L3651) `request_tokenize_multimodal_prompt` → 对每张图调 [server_encode_image](ds4_server.c#L10476)（cache miss 则 `ds4_engine_vision_encode_memory` 跑塔）。
- 调用点在请求解析处（[ds4_server.c:4286](ds4_server.c#L4286)、[4513](ds4_server.c#L4513)、[5543](ds4_server.c#L5543)），**早于** worker 里的复用决策。
- 复用决策在 [generate_job_inner](ds4_server.c#L14197)：`slot_probe_reuse` 算出 `cached`，然后 [ds4_server.c:14603](ds4_server.c#L14603) 才把 spans 交给 `server_session_sync_multimodal`。

而复用决策**根本不需要塔的输出**：

- KV 身份 = `token_start + token_count + fingerprint` 三元组比对（[ds4.c:78702](ds4.c#L78702) `ds4_session_vision_prefix_matches`），`fingerprint` 是**解码像素的 SHA-256**（[ds4_image.c:230-236](ds4_image.c#L230)），在 `ds4_image_decode` 时顺手算好，毫秒级。
- `token_count` 来自 CPU patchify 的 grid 元数据（[ds4.c:75869](ds4.c#L75869)），也在塔之前。
- prefill 消费 embedding 是按 token 位置查表取行（[ds4.c:30378-30391](ds4.c#L30378) `metal_graph_visual_embedding_row`，且只查正在评估的 token），**frontier 以下的 span 永远不会被读**。

所以"eager 编码"对避免 re-prefill 没有任何贡献，纯粹是把"tokenize 需要图片占位 token"实现成了"tokenize 需要图片 embedding"。

### 根因 2：embedding cache 容量与长会话工作集错配 → LRU 抖动

- [ds4_server.c:10316-10317](ds4_server.c#L10316)：`SERVER_IMAGE_CACHE_ENTRIES 32`、预算 `128MiB`。
- 大截图一条 embedding ≈ 47MB（≈4800 token × 2560 dim × fp32）→ 容量只装 **2~3 张大图**。
- 8 图会话工作集 ~130MB+ > 预算 → 每轮 put 逐出下一轮要用的图 → **每轮 100% miss**（日志中 cache 钉在预算线、每 put 必逐出，即此现象）。

### 根因 3：内存压力 WARN 即整清

- [ds4_server.c:10347-10349](ds4_server.c#L10347) `server_memory_release_images`：压力 ≥ WARN 就 `server_image_cache_clear`（整清），且它在 [generate_job_inner 开头](ds4_server.c#L14204) 每请求执行。
- macOS 的 WARN（`kern.memorystatus_vm_pressure_level==2`，[ds4_host_memory.h:49](ds4_host_memory.h#L49)）在跑 110K ctx + 大权重的机器上是**常态**，不是紧急状态。日志里 cache 从 130MB 骤降到 49MB 即整清+重建的形态。
- 整清把"还能用"的 embedding 全部扔掉，下一轮全量重烧塔，与根因 2 相互放大。

次要低效（不致命但顺手可修）：cache key 是**完整编码字节 memcmp**（32 条目 × 数 MB）；embedding 以 fp32 存储（塔输出本源自 bf16）；命中时 malloc+memcpy 整块。

## 3. 成熟项目怎么做（vLLM / llama.cpp / SGLang，2026-10-10 并行调研）

### 3.1 vLLM — 三层缓存 + "KV 命中即跳 encoder"

- **Processor 缓存（CPU 层）**：`--mm-processor-cache-gb`（默认 4GiB，0=禁用）缓存 HF processor 输出；`--mm-processor-cache-type lru|shm`。[docs](https://docs.vllm.ai/en/stable/cli/bench/mm_processor)
- **MultiModalHasher**（[hasher.py](https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/hasher.py)）：对媒体做长度前缀 framing 序列化后 blake3/sha256 摘要；PIL 图优先 EXIF ImageID，否则 hash mode+像素数组；hash 覆盖媒体数据及预处理 kwargs。[optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization)
- **EncoderCacheManager（GPU 层）**（[encoder_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/encoder_cache_manager.py)）：encoder 输出按 mm_hash 缓存 + **引用计数**，容量不足只逐出无运行请求引用的 `freeable` 条目；请求结束 `free()` 通知 worker 释放显存。跨请求同图命中"hit, no encoder run"。
- **APC×多模态**：block hash = tokens+parent hash+extra keys，图片 mm_hash 经 `_gen_mm_extra_hash_keys` 混入 block key（[prefix caching docs](https://docs.vllm.ai/en/stable/design/prefix_caching)）。**KV 前缀命中覆盖图片 token 时 encoder 整个跳过**——scheduler.py 注释"num_computed_tokens>0 则 encoder inputs 已算过可跳过"，`check_and_update_cache` 命中即 continue。span 内仍有未算完的 mm 数据时只补算未覆盖部分，已算图片从 encoder cache 取。
- **性能证据**：shm IPC 缓存实测缓存请求 prefill +69.9%、TTFT −40.5%（[2025-11 blog](https://vllm.ai/blog/2025-11-13-shm-ipc-cache)）；跨请求 embedding 复用由 RFC [#21113](https://github.com/vllm-project/vllm/issues/21113) 推动；历史 V0 chunked-prefill×mm shape bug 见 [#18414](https://github.com/vllm-project/vllm/issues/18414)。
- **调度**：同一步内 encoder 先于 decoder，跨步流水（请求 N 的 encoder 可与 N−1 的 decode 并行）；EPD 编码分离见 [EPD blog](https://blog.vllm.ai/blog/vllm-epd)。

### 3.2 llama.cpp / mtmd — 靠 KV 复用跳 encode，无 embedding 级缓存

- **无内容寻址 encoder 缓存**：主线 libmtmd 没有按图片 hash 的 embedding 缓存；`~/.cache/llama.cpp` 只是 `-hf` 模型下载缓存。mtmd.h:183 有可选 bitmap ID（"useful for KV cache tracking"），server 未做 embedding 缓存（open PR 增加 `image_embedding` 输入，未合并）。
- **KV 复用**：`--cache-prompt`（默认开）+ `--cache-reuse N`（KV shifting 最小 chunk）按 token 前缀（LCP）复用 KV，**命中覆盖图片位置时不再跑 encoder**——与 P0 同一原则。
- **`--cache-ram`**（PR #16391，默认 8192MiB）：host 内存 KV state prompt cache，`--cache-idle-slots` 把空闲 slot 存进去；checkpoint 结构不含 bitmaps → 命中 KV 即跳 encode，但历史图片不留存，换会话需重编码。图片 prompt 撑爆 RAM 见 [#22629](https://github.com/ggml-org/llama.cpp/issues/22629)；**磁盘 embedding 层至今只是 feature request**（#20697）。

### 3.3 SGLang — 图片 hash 编进 radix key

- **RadixAttention×多模态**：论文原文"compute the hash of the input images and use it as the key in the radix tree"（[arXiv 2312.07104](https://arxiv.org/pdf/2312.07104)）。现行实现把多模态占位 token 替换为 hash 派生 pad_value（`MM_PAD_SHIFT_VALUE + mm_hash % 2^30`，[mm_utils.py](https://github.com/sglang-project/sglang/blob/main/python/sglang/srt/managers/mm_utils.py)），radix 命中覆盖图片 token → 跳过 ViT。
- **Embedding 缓存**：`MultiModalStaticCache`（[multimodal_cache.py](https://github.com/sglang-project/sglang/blob/main/python/sglang/srt/mem_cache/multimodal_cache.py)）GPU LRU 按 mm_hash 存 embedding，命中跳过 ViT；hash 用 sha256 前 8 字节（CPU）或 triton `gpu_tensor_hash`；CPU 侧 preprocess cache 带 single-flight。
- **调度**：非 EPD 模式 encoder 内联 prefill 无独立 overlap；EPD 分离 [#8223](https://github.com/sglang-project/sglang/issues/8223)→PR #12263、Global Encoder Cache [#15118](https://github.com/sglang-project/sglang/issues/15118)、细粒度 mm cache RFC [#24945](https://github.com/sglang-project/sglang/issues/24945)。

### 3.4 共性对照

| 机制 | vLLM | SGLang | llama.cpp | **ds4 现状** | **ds4 方案** |
|---|---|---|---|---|---|
| 内容 hash 做 key | blake3 framed | sha256/triton→pad_value | 仅 bitmap ID（弱） | ✅ 像素 SHA-256（已有！） | P1 换 fingerprint key |
| KV 命中跳过 encoder | ✅ scheduler skip | ✅ radix 命中跳 ViT | ✅ LCP 命中跳过 | ❌ **eager 全烧**（根因1） | **P0** |
| encoder 输出独立缓存 | ✅ GPU refcount LRU | ✅ GPU LRU | ❌ 无 | ⚠️ 128MB 字节 memcmp LRU（根因2） | P1 |
| 逐出保护 | refcount freeable | LRU | — | ❌ WARN 整清（根因3） | P1 钉 checkpoint 集 |
| 磁盘 embedding 层 | shm IPC | 路线中 | ❌ 仅 feature request | ❌ | P2（可选，三家皆无） |
| encoder/prefill overlap | ✅ 跨步流水 | EPD 路线 | ❌ | ❌ | P3 不做 |

**结论**：三家共同范式 = 内容 hash 做 key + **encoder 只对 KV 前缀未覆盖的图片运行** + embedding 分层缓存。ds4 已经有正确的身份机制（fingerprint 进 KV 校验），唯独把 encode 放在了复用决策**之前**——这是三家无一采用的位置。P0 就是把 encode 挪到别人都在的位置。

## 4. 改进方案

### P0 — 惰性编码：只对越过 KV frontier 的图片烧塔（核心修复）

流程重排（与 vLLM/SGLang 同构）：

1. **parse 期**：解码图片 → fingerprint + 尺寸；CPU patchify 拿 grid/token_count；tokenize 占位 span。**不跑塔、不分配展开块**。
2. **worker 期**：`slot_probe_reuse` 照旧（它只依赖 token 与 fingerprint，已满足）。
3. **复用决策后**：对 `token_start + token_count > cached` 的 span（按 rebase 后的 `prompt_for_sync` 坐标），从 embedding cache 按 fingerprint 取；miss 才跑塔并回填。frontier 以下的 span 保持 `data=NULL`。
4. sync/prefill 不变；`metal_graph_visual_embedding_row` 已有 `data` 判空。

配套改动点：

- [ds4.c:79001](ds4.c#L79001) `ds4_session_sync_multimodal` 的校验放宽为"frontier 以上 span 必须有 data"（frontier 由 sync 内部已知，或 server 传入 `cached`）。
- 展开块（`ds4_prompt_append_deepseek4_vision` 的 permute+memcpy）同样只对 frontier 以上的 span 做；frontier 以下的 span 只留 token 布局。
- 请求对象里 span 改为可携带"未物化"状态（引用 `server_image_input` + fingerprint）。

效果（用户场景）：稳态轮次（KV 全命中，如日志 cached=110817）vision encode 次数 8→**0**，TTFT 去掉整个 65s+；只有新图那一轮烧新图的塔。

### P1 — embedding cache 升级（配合 P0，覆盖"frontier 推进"轮）

1. **key 换 fingerprint**：32B hash 比较代替全字节 memcmp（hash 已在 decode 时算好）。
2. **fp16/bf16 存储**：容量 ×2（塔输出本源自 bf16；消费点转回 fp32 的行级转换是 memcpy 量级）。
3. **预算可配 + 默认调大**：如 `--image-cache-mb`，默认 512~1024MB；按机器内存自适应。（**2026-10-10 作废**：本机常驻内存约束，实落 256MiB，见 §7）
4. **逐出策略**：LRU 但**钉住 live checkpoint 的 fingerprint 集合**（`checkpoint_images[].fingerprint` 就是现成的工作集清单）；WARN 不再整清，只有 CRITICAL 才清——把 [ds4_server.c:10348](ds4_server.c#L10348) 的 `>= WARN` 改为 `== CRITICAL`，并改逐出而非整清。

### P2 — 磁盘持久化 embedding 缓存（可选，三家皆无现成实现）

- 按 fingerprint 内容寻址落盘（`<cache-dir>/<fp-hex>.f16`），重启/换 slot/KV 冷启动时读盘代替烧塔（47MB NVMe 读 ~10-20ms vs 塔 ~8-16s）。
- 定位：llama.cpp 的磁盘 embedding 层至今只是 feature request（#20697），vLLM 只有 shm IPC（进程间非持久）——做成即领先三家，但非必选项。
- 顺带补上多模态会话没有磁盘 KV 续热（[ds4_server.c:14402](ds4_server.c#L14402) `!multimodal` 关闭）的缺口：KV 必须重灌时，至少 embedding 不用重烧。

### P3 — encoder 与 prefill overlap（暂不做）

vLLM 的 encoder-scheduler overlap 在 P0/P1 之后收益很小（稳态已无 encode），不建议现在引入复杂度。

## 5. 验证方案

1. **回归（不加载模型，进程内）**：仿 [ds4_server.c:24637](ds4_server.c#L24637) 现有 image cache 测试，新增：
   - 工作集 > 预算时，钉住的 fingerprint 不被逐出（红→绿）。
   - frontier 以下 span `data=NULL` 时 `ds4_session_sync_multimodal` 校验通过、frontier 以上缺 data 报错。
2. **真机（遵守内存铁律：先查 free/swap/驻留进程）**：8 张大图长会话，开关对比：
   - 指标：每轮 `vision embedding encoded` 次数（期望：仅新图轮 >0）、TTFT、`multimodal live kv hit` 不变。
   - 断言 fingerprint 身份校验不因惰性化而放松：中途换图像素 → 必须 miss（仿 [ds4_server.c:17565](ds4_server.c#L17565) "pixel mismatch binds nowhere" 回归）。
3. **正确性**：`./ds4_test --server` 全绿；logprob 向量对比（同图同 prompt 输出与 eager 路径逐位一致）。

## 6. 风险与边界

- **指纹≠像素永不分叉**：fingerprint 是解码像素哈希，与 #976 存档的"webp 有损解码器像素差"教训一致——惰性化不改变 fingerprint 计算位置（仍在 decode 期），身份语义不变。
- **rewind/rebase 路径**：frontier 判定必须用 rebase 后的 span 坐标（[ds4.c:78758](ds4.c#L78758) `ds4_session_rebase_vision_state` 已处理 token_start），实现时以 sync 收到的最终坐标为准。
- **TP 路径**：[ds4_tp.c:2998](ds4_tp.c#L2998) 已按 fingerprint 走 wire，embedding 物化在 leader 侧，惰性化需确认 worker 侧 span 同步语义不变。

## 7. 决策与实施记录（2026-10-10，本批落地）

**约束修正**：本机 GLM-5.3 Flash + DSv4 Flash 常驻 110–118 GB、空闲仅 ~14 GB、不开 SSD KV ⇒ P1 原文"默认 512~1024MB"作废，改为**有界小预算**；P2 磁盘缓存因用户不开 SSD KV 彻底搁置。

**上游 cherry-pick 检查（用户指令：有则优先）**：穷举 open PR（vision/encoder/re-encode/image cache/frontier 多组检索）+ 本地 `up/pr/*` ref——**无任何未合并 PR 实现"KV 覆盖即跳塔"**。最接近的 **#968（CLOSED 未合并）** 即现树 `server_image_cache` 机制的出处（bytes 全等键+memcmp+LRU），它本身不做 lazy encode，pick 无增量；其 256MiB 预算选择成为本批先例。#961/#977/#962 依赖 SSD 路径（本机不适用），#1022 为有损裁图，#1153 是文本前缀快照（不触碰塔）。评估详情已入 [LOCAL_INVENTORY.md](LOCAL_INVENTORY.md) D 节。

**本批实施（server-only，未动 engine/TP，与 GLM rewind 未提交改动同树共存）**：
1. 默认预算 128→**256 MiB**（用户 8 图工作集 ~130MB fp32，可整驻；命中路径 memcmp+memcpy 合计 <100ms，TTFT 65s→~1s 量级）。
2. 新增 **`--image-cache-mb N`**（0 关闭；运行期 `server_image_cache.max_bytes`，put 的 budget 参数由调用点传入，签名未动）。
3. `server_memory_release_images`：**WARN 不再整清，仅 CRITICAL 清空**（WARN 在本机为常态，清空即触发全历史重烧；预算本身即内存契约）。
4. 存储保持 **fp32**（不改 fp16——保持"命中=首编逐位一致"的位精确回放语义，仓库位契约文化优先）。
5. 测试：`test_server_memory_pressure` 更新为新契约（WARN 保留/CRITICAL 清空）、新增 `test_image_cache_mb_option` 与零预算禁用用例；`make ds4-server ds4_test` `-Wall -Wextra` 零警告，`./ds4_test --server` 全绿（进程内，未加载模型）。

**遗留终态（P0 lazy encode，未实施）**：engine 侧把 `ds4_prompt_append_vision` 拆为 layout/materialize 两段、`ds4_session_sync_multimodal` 校验放宽为"仅 frontier 以上 span 需 data"、TP wire leader 物化守卫（`tp_active` 时维持 eager）。改动面 ~400-600 行跨 ds4.c/ds4.h/ds4_server.c，稳态视觉内存趋零；当前 server-only 方案已把用户可感 TTFT 问题解决，P0 留作后续对齐 vLLM/SGLang 的架构终态。
