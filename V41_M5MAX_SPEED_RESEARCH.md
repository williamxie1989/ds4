# DeepSeek V4.1 Flash × 单台 MacBook M5 Max（128 GB）速度调研

> 调研日期：2026-09-30。范围：**只看 DeepSeek V4.1 Flash + 单台 M5 Max 128 GB**（Metal）。
> 数据来源：本仓库文档、上游 [antirez/ds4](https://github.com/antirez/ds4) 的 issue/PR、第三方验证库 [evanwtf/local-llm](https://github.com/evanwtf/local-llm)、X/HN 公开实测。
> 所有引用数字均标注来源与条件；未合并 PR 的效果以作者自报口径为准，并注明验证状态。

---

## 0. 前提

V4.1 Flash 权重体积（[ds4#1023](https://github.com/antirez/ds4/issues/1023)、[ds4#1085](https://github.com/antirez/ds4/issues/1085)）：

| 量化 | 文件体积 | 构成 |
|---|---:|---|
| Q2 | 341 GiB | 151.77 GiB 主权重 + 188.83 GiB FP8 Engram 表（Engram 只从磁盘读，不常驻） |
| Q4 | 518.6 GB | ~294 GB 主权重 + 189 GB Engram 表 |

**结论：128 GB M5 Max 上 V4.1 Flash 必然走 `--ssd-streaming`**，速度分两个口径：流式 prefill 与流式 decode。参照上限是同机 **V4 Flash Q2 全驻留**（不同模型，仅作天花板参考）：prefill 790 t/s @2K、decode 39–45 t/s（[docs/PERFORMANCE.md](docs/PERFORMANCE.md)、[speed-bench/m5_max.csv](speed-bench/m5_max.csv)、[#1124](https://github.com/antirez/ds4/pull/1124)）。

---

## 1. V4.1 Flash 在 M5 Max 上的实测速度

### 1.1 Q2（341 GiB）SSD 流式

| 来源 | Prefill | Decode | 条件与备注 |
|---|---|---|---|
| **antirez 本人**（X 帖，[evanwtf#321](https://github.com/evanwtf/local-llm/issues/321) 存档核实 2026-09-11/12） | **~800 t/s**（大 prefill，loader 8 s 切到 encoder 半常驻后） | **~16 t/s 稳定** | 三种 prefill 路径自动选择：小 prefill 走 expert-cache（类 decode）、较大走 layer-major + hidden next-layer load、超大走 full-resident encoder。full-resident decoder 会冲掉 experts cache，但流式能在数秒内恢复 |
| **gilbert-barajas**（[ds4#1023](https://github.com/antirez/ds4/issues/1023) 评论 2026-09-13） | 冷 **47–72 t/s**（2K/8K/16K/32K）；warm **112–130 t/s** | **14.8–16.4 t/s**（2K→32K 平坦） | MacBook Pro M5 Max 128 GB，macOS 27.0，ds4 `bd66c40`，`--metal --ssd-streaming --power 100`，promessi_sposi 扫 2K→32K。冷专家首触：prefill 崩到 ~12 t/s、decode 衰减到 6.5 t/s——冷热必须分开标注 |
| **TheDavidTai**（X 帖，evanwtf 核实 2026-09-17，[链接](https://x.com/TheDavidTai/status/2100263416711589982)） | — | **~10 t/s**（Q4） | 128 GB M5 MacBook |

**800 t/s 的验证状态**：目前只有 antirez 本人数据 + YouTube 演示佐证（"loaded only the half of the model that reads prompts on a 128 GB M5 Max … prompt reading reached 800 tokens per second"）。第三方 evanwtf/local-llm 已做预注册复现，但因 340.6 GiB 超过其 256 GB 下载上限被 operator 搁置并关闭（[evanwtf#321](https://github.com/evanwtf/local-llm/issues/321)，2026-09-18 parked / 2026-09-30 closed）。HN 评论指出常见口径是 [400 t/s](https://news.ycombinator.com/item?id=48142458)——短 prompt 走单 token 路径看不到真实上限。独立第三方 M5 Max 实测落在 **冷 47–72 / warm 112–130**。

### 1.2 Q4（518 GB）SSD 流式 — argonautlabs fork（[ds4#1151](https://github.com/antirez/ds4/issues/1151)，2026-09-29，全部逐位一致）

fork：[argonautlabsai/ds4-argodrive](https://github.com/argonautlabsai/ds4-argodrive)，钉在 `bd66c40`；GPU 锁频 1620 MHz，零 swap 增长。

| 配置 | Decode tok/s | Prefill tok/s |
|---|---:|---:|
| 上游 bd66c40，内置 SSD | 10.18 | 17.96 |
| fork，仅内置 SSD | **18.66（+83%）** | 30.98 |
| fork + 1 块 Thunderbolt 5 NVMe 副本 | 21.05 | 36.88 |
| fork + 2 块 NVMe 副本 | **22.25（+119%）** | **42.41（+136%）** |

单项拆解（均为 M5 Max 实测、bit-exact）：

| 候选 | 效果 | 说明 |
|---|---|---|
| Router ids 走 GPU mailbox，shared expert 先编码再 commit | **+6.6%** steady decode | CPU 不等待读回，miss 读与 GPU 并行；50 ms 兜底回退（实测 8000 次轮询零回退） |
| 缺失专家从多盘字节相同副本加权读取 | 18.66→22.25 | 拟新增 `--ssd-streaming-replica PATH` |
| post-MoE flush + 驱逐预扫描 | **+1.5%** @pp512/tg512 | 等待 mailbox 时并行排驱逐候选 |
| V4.1 单 token router 快路径 + 4 个小融合 | **+2.2~2.8%** | 同一 bitonic 选择网络与平票规则 |
| 3 次 BF16 舍入折进上游核 + pipeline-state 缓存 | **~+1.5%** | |
| 读路径重构（9 常驻读线程、split reads、down-projection 与 miss 读重叠、异步 Engram、32-token hotness 衰减） | 单盘增益大头 | 未对上游单列，rebase 时补测 |

### 1.3 速度全景（M5 Max，V4.1）

- decode 现状：Q2 流式 **~15–16 t/s**（antirez 与 gilbert-barajas 交叉印证；**fork 3b2abc4 本机 M5 Max 实测已达 22.0–23.7 t/s 稳态（2K–32K，auto cache 64.5 GiB，见基线方案 §12.1）**）；Q4 流式上游 ~10、优化 fork **18.7–22.3**。
- prefill 现状：Q2 大 prefill 作者口径 ~800、独立口径 400–700（冷 47–72 / warm 112–130）；Q4 流式 18→42。
- 与同机 V4 Flash Q2 驻留（decode 39–45）差约 **2.5×**，这个差距就是提速 PR 的目标空间。

---

## 2. 能提升 V4.1 + M5 Max 的近期 PR（含未合并）

### 2.1 decode 侧（大头）

| PR | 状态 | 自报效果 | 对 M5 Max 的意义 / 验证状态 |
|---|---|---|---|
| [#1090](https://github.com/antirez/ds4/pull/1090) trueimage | OPEN | **V4.1 decode +57~59%**（M3 Ultra 驻留 Q4：18.15→28.93 t/s @512，+51.5% @65K）；prefill +17.3%→+2.8%；逐位一致（到 130,880 位置） | 上游 fast path 门控 M3 Ultra；**本 fork 已 cherry-pick**（`0743c7f`），同族 GLM 调优已在 NAX 域（M4/M5）默认放开并实测 decode +21~23%。官方 M5 Max 复测（[evanwtf#604](https://github.com/evanwtf/local-llm/issues/604)）尚未跑 |
| [#1041](https://github.com/antirez/ds4/pull/1041) adriangalilea | OPEN | 单机 decode 按层排队、逐层 commit 不等待：**+37%**（M3 Ultra 驻留 Q4 16.68→23.09），逐位一致 | 消掉每 token 40 次等待造成的 9.6 ms 调度空隙（排队后 0.75 ms）；机制与机器无关，M5 大概率兑现。回滚 `DS4_METAL_DISABLE_V41_DECODE_QUEUE` |
| [#1042](https://github.com/antirez/ds4/pull/1042) adriangalilea | OPEN | 在 #1041 之上再 **+20%**：融合 HC 胶水（19→6 dispatch）、router（11→2，M5 上 1）、shared expert（12→3）等 | 对齐 DeepSeek 生产解码的 Mega-mHC/Mega-Gate 形态，保留全部舍入点，逐位一致 |
| [#1035](https://github.com/antirez/ds4/pull/1035) Dango233 | OPEN | Engram 并行读 **+7.4%**（M3 Ultra 驻留，ABBA，逐位一致）；与 #1041 叠加共 **+48%** vs 上游默认（24.86 t/s） | macOS 通用，作者建议 macOS 默认开 |
| [#1034](https://github.com/antirez/ds4/pull/1034) Dango233 | OPEN | SSD 流式 decode **+11%**（M2 Ultra Q2） | 流式路径直接相关；驻留场景收益更大（+28~29%，见 #1034/#1041 评论） |
| [#1043](https://github.com/antirez/ds4/pull/1043) adriangalilea | OPEN | Q4_K group-6 专家表默认 **+2.3%** decode（M3 Ultra），greedy 一致 | 体积/带宽双收益 |
| [#1151→拆 PR 候选](https://github.com/antirez/ds4/issues/1151) argonautlabs | OPEN（issue） | 见 §1.2 单项表 | **全部在 M5 Max + V4.1 Q4 上实测**，目前 M5 Max 专属最强 decode 数据 |
| [#1067](https://github.com/antirez/ds4/pull/1067) | OPEN | V4.1 decode 每 token 单次 CB wait | 硬门控 `pre_m5_apply_silicon()`，**M5 上不生效** |
| [#1060](https://github.com/antirez/ds4/pull/1060) / [#1061](https://github.com/antirez/ds4/pull/1061) gaineyllc | OPEN | V4.1 indexer radix-select top-k / decode 形状专用核 | 长上下文 indexer 优化 |
| [#1089](https://github.com/antirez/ds4/pull/1089) gaineyllc | OPEN | 活会话 rewind 而非重建 | 多轮 agent 免 re-prefill（有效吞吐，不是峰值 t/s） |

### 2.2 prefill 侧

| PR | 状态 | 效果 | 验证状态 |
|---|---|---|---|
| [#758](https://github.com/antirez/ds4/pull/758) rinaldofesta | OPEN | **M5 Max 实测**：indexed prefill 注意力阶段 **−11.3%**（88.6→78.6 ms/层，21/21 层全胜）；8192-token prefill **+2.9~5.7%**（两次平衡均值 +2.94%）；top-512 融合 −5.8%；logits 逐字节一致 | dsv4 共享稀疏注意力路径；回滚开关齐备 |
| 本地 `2ce4e40`（上游未合并） | **本 fork 已落地** | DSV4 稀疏 indexed attention 上 tensor units（NAX）：内核 **≈2×**（GPU 占比 18.1%→9.4%）；41K 长 prefill 按 GLM 同病理外推 **+13~17%**；端到端在 ±13% 噪声带内 | M5 Max 实测；质量闸门全过（short-100 逐位一致、long-9 greedy_lcp 9/9、frontier top-1 6/6）。详见 [SPARSE_MLA_NAX_DSV4_PLAN.md](SPARSE_MLA_NAX_DSV4_PLAN.md) §8/§9 |
| [#1073](https://github.com/antirez/ds4/pull/1073) kernelpool | OPEN | V4.1 Metal 整体提速 + DSpark：M3 Ultra 驻留 Q2 decode 19.8→**36.2 t/s（≈+80%）**、prefill 339→389；jjakemaness 第三方复跑 256 GB 驻留 37.1 t/s、DSpark 40.8–47.4 | 驻留场景——**128 GB M5 Max 用不上**；本 fork 标记"需手工 rebase"未取 |

### 2.3 更大杠杆（提案 / 未验证）

- [#1085](https://github.com/antirez/ds4/issues/1085)：**三值 PTQ（Bonsai-2 风格，~1.72 bpw）** V4.1 主权重。128 GB 流式 decode 按每 token 字节数外推 **×2~2.5**（4.5–5.5 → ~11–14 t/s 基线口径）；256 GB 机器可全驻留。**纯提案，无实测**。
- **DSpark/MTP**：M5 Max 上此前实测为零收益/净亏（[#913](https://github.com/antirez/ds4/issues/913)：接受率 66%、verify2 1.28×；[#1002](https://github.com/antirez/ds4/issues/1002)：净亏）；#1073 声称代码类 prompt 到 256k 接受率好（M3 Ultra）。这是 decode 端最大未兑现变数。

### 2.4 本 fork 已生效的相关改动（上游未合并）

- `0743c7f`：cherry-pick #1147/#1120/#1093/#1135/#1126/#1122/#1149/#1150/#1090（**剔除 #1127**，GLM 短 prompt −19% 实测复现）。
- `9bc0bd7`：GLM53 调优门控放开到 NAX 域（M4/M5）默认生效——M5 Max 实测 decode **+20%**、prefill **+5~12%**（≤8K +11.8%）；`DS4_METAL_DISABLE_GLM53_TUNING` 总回滚。#1090 的 V4.1 pipelined decode 同批进入。
- `2ce4e40`：见 §2.2。

---

## 3. 结论与预期

1. **现状（M5 Max 128 GB，V4.1 Flash）**：Q2 流式 decode **~15–16 t/s**（上游口径；fork 3b2abc4 本机实测稳态 22.0–23.7 t/s，见基线方案 §12.1）、prefill 作者口径 ~800（独立口径 400–700，冷 47–72 / warm 112–130）；Q4 流式上游 ~10 t/s，argonaut fork 单盘 **18.7**、双 NVMe 副本 **22.3**。
2. **合并后合理预期**：#1041（+37%）→ #1042（再 +20%）→ #1035（叠加 +48%）与 #1090（+57%）这条 bit-exact 流水线家族已在 M3/M2 Ultra 验证、机制与机器无关；叠加 argonaut 的 mailbox/融合单项（#1151）与 prefill 侧 #758/本地 NAX（+3~17%）——**Q2 流式 decode 有望 ~16 → 25–30 t/s，Q4 单盘逼近 20+ t/s**。
3. **风险/待办**：
   - #1090 等 fast path 上游门控在 M3 Ultra，M5 Max 官方复测（evanwtf#604）未跑；本 fork 已在 NAX 域放开同族门控并有实测。
   - "800 t/s prefill" 尚无第三方在 M5 Max 上复现（evanwtf 因下载上限搁置）。
   - DSpark 在 M5 Max 上仍是零收益，decode 端最大杠杆未兑现。
   - 冷/热、锁频与否、cache 预算对数字影响巨大（冷 prefill 可崩 9×），引用任何数字必须带条件。

## 4. 复现命令速查

```sh
# 标准驻留基线口径（V4 Flash，做同机天花板参照）
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128

# V4.1 Flash Q2 SSD 流式（gilbert-barajas 口径）
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming --power 100 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 --gen-tokens 128

# 大 prefill（触达 antirez 所述大 prefill 路径）
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --metal --ssd-streaming \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 32768 --ctx-max 65536 --step-incr 32768 --gen-tokens 128
```

---

## 附：来源清单

| 来源 | 内容 |
|---|---|
| [ds4#1023](https://github.com/antirez/ds4/issues/1023) | V4.1 权重发布；gilbert-barajas M5 Max Q2 流式冷热实测 |
| [ds4#1151](https://github.com/antirez/ds4/issues/1151) | argonautlabs fork：V4.1 Q4 流式 M5 Max 阶梯 + 单项拆解 |
| [ds4#1085](https://github.com/antirez/ds4/issues/1085) | 三值量化提案（流式 decode ×2~2.5 外推） |
| [ds4#1090](https://github.com/antirez/ds4/pull/1090) / [#1041](https://github.com/antirez/ds4/pull/1041) / [#1042](https://github.com/antirez/ds4/pull/1042) / [#1035](https://github.com/antirez/ds4/pull/1035) / [#1034](https://github.com/antirez/ds4/pull/1034) / [#1043](https://github.com/antirez/ds4/pull/1043) / [#1067](https://github.com/antirez/ds4/pull/1067) / [#1060](https://github.com/antirez/ds4/pull/1060) / [#1061](https://github.com/antirez/ds4/pull/1061) / [#1089](https://github.com/antirez/ds4/pull/1089) / [#1073](https://github.com/antirez/ds4/pull/1073) / [#758](https://github.com/antirez/ds4/pull/758) | 未合并提速 PR（见 §2） |
| [ds4#913](https://github.com/antirez/ds4/issues/913) / [#1002](https://github.com/antirez/ds4/issues/1002) | DSpark 在 M5 Max 零收益/净亏 |
| [evanwtf/local-llm#321](https://github.com/evanwtf/local-llm/issues/321) | antirez 800 t/s / 16 t/s X 帖存档核实；第三方复现预注册与搁置记录 |
| [evanwtf/local-llm#604](https://github.com/evanwtf/local-llm/issues/604) | #1090 V4.1 decode +57% 的 M5 Max 复测立项（未跑） |
| [x.com/antirez/status/2098422997719650788](https://x.com/antirez/status/2098422997719650788) 等 | antirez 本人 X 帖（经 evanwtf verify_posts.py 核实存在） |
| [HN 48142458](https://news.ycombinator.com/item?id=48142458) | "400 t/s 是常见口径" 讨论 |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) / [speed-bench/m5_max.csv](speed-bench/m5_max.csv) / [docs/SSD_STREAMING.md](docs/SSD_STREAMING.md) | 同机驻留/流式基线 |
| [SPARSE_MLA_NAX_DSV4_PLAN.md](SPARSE_MLA_NAX_DSV4_PLAN.md) / [OMLX_WAVE_PORT_ANALYSIS.md](OMLX_WAVE_PORT_ANALYSIS.md) | 本 fork NAX 移植计划与上游 PR 清点（§8/§9 为 M5 Max 实测） |
