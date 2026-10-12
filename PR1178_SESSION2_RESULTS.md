# PR #1178 第二轮排查结果（2026-10-04 晚 22:50–23:50 会话）

> 承接 `PR1178_HANDOFF.md` 与 `.cardi-pr/PR1178_DMR_HYPOTHESIS.md`。
> 本轮跑了 **19 条模型腿**（机器全程稳定：avail ~110 GiB，swap 恒定 ~7.0 GiB，无 fuse 触发）。
> 复现口径：`-m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming --ctx 8192 --nothink --temp 0`
> 意大利语 34-token prompt，`-n 48`，`--dump-logprobs`；最小复现 `-p "Ciao" -n 4`（16 次 forward，13 s/腿）。
> 基线 md5：**OFF = `be411474f174292326db2f6f5c2ee31c`**（step0 top1 = 23.7921906），
> **ON = `1a68a0eab0e82bf01100194bd0899cf0`**（step0 top1 = 26.1374130）。

---

## 1. 结论一：`didModifyRange` publish barrier 无效（并行会话的修复方案已死）

`t1.sh LEGBATCH=all CE=60` 五腿 A/B：

| 腿 | md5 |
|---|---|
| off | `be411474` |
| onnb（无 barrier） | `32ab430c` |
| onnb2（无 barrier） | `32ab430c` |
| onfx（有 barrier） | `32ab430c` |
| onfx2（有 barrier） | `32ab430c` |

**4 条 ON 腿逐字节相同 ⇒ barrier 开关零影响。**

另：并行会话自己的 harness（`.cardi-pr/harness/harness`，全区 XOR 校验 / 1500 轮 / 5 模式）
我重跑了一遍——**全模式 0 mismatch，包括 `fixed-fresh` 无 barrier**。
我自己那版 harness（4 GiB slab + mlock + 在飞改地址表 + alloc churn，300 样本/变体）同样 0 mismatch。

⇒ Metal shared 存储的"提交后 CPU 写、同 CB 读"在这台机器上**没有**可观测问题。
`.cardi-pr/PR1178_DMR_HYPOTHESIS.md` §1 的根因假设与 §2 的修复落点都应作废。

## 2. 结论二：裸地址读专家权重**必须**被 Metal 标记（首次直接证明）

`DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS=1`（每个专家走独立 `ds4_stream_expert_combined` 缓冲）：

| 腿 | md5 | step0 top1 | 输出 |
|---|---|---|---|
| **OFF** + noslab | `be411474` **= OFF** | 23.7921906 | 正常 |
| **ON** + noslab | `bc0362e6` | **10.4660988** | 退化成 `"  e  a  a  a a a…"` |

两条腿唯一的差别是：OFF 用 slots6 内核**显式 bind** 这 6 个 buffer；ON 走裸地址 + `use_slabs` 标记。
⇒ `useResource` 标记是承重的，缺了它裸地址读会读到坏数据。这是本 PR 机制里第一个被直接证明的硬约束。

## 3. 结论三：OFF 对所有旋钮免疫，ON 处处不同

| 旋钮 | OFF md5 | ON step0 top1 |
|---|---|---|
| 默认 | `be411474` | 26.1374130 |
| 强制 addr 表 (`ENABLE_STREAMING_EXPERT_ADDR_TABLE=1`) | `be411474` | — |
| 禁用 slab | `be411474` | **崩坏 10.47** |
| slab=512MB（88 个 slab） | `be411474` | 26.2865410 |
| slab=32GB（2 个 slab） | — | 26.1966209 |
| CE=60（1 个 slab） | `be411474` | 26.1966209 |
| 禁用 masked 内核 | — | `1a68a0ea`（**= 默认 ON**） |

缓存报告（`DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`）：
`evictions=0 buffer_reuses=0 mlock failures=0`，默认缓存 11 个 slab、SLAB=512 时 88 个、SLAB=32768 时 2 个。
（`buffer_allocs` 同时计入 slab 与非 slab 分配；11 = 纯 slab，无非 slab 回退。）

## 4. 结论四：`use_slabs` 覆盖缺口是真的，但**不是主因**

机制（源码确定）：encode 线程编码整个 token 的全部 dispatch 时才调 `use_slabs`，
而 slab 是 serve 线程在 CB **执行期间**才创建的 → 执行期新建的 slab 在本 token 内无标记。

剂量-响应实验（slab 数 88 → 2，即"未标记首读"事件数变化 ~44 倍）：

```
88 slabs → +2.494    11 slabs → +2.345    2 slabs → +2.405    1 slab → +2.405
```

**只动了 0.089（占总偏移 2.4 的 3.7%）⇒ 覆盖缺口只能解释很小一部分。**

## 5. 结论五：`pread()` vs 普通 CPU store 无关

给 `ds4_gpu_stream_expert_pread_into` 加 `DS4_V41_DIAG_STAGED_PREAD`（pread 到 malloc 缓冲再 memcpy 落盘，
末尾 `__sync_synchronize()`）——这是引擎与 harness 之间最后一个未测差异：

```
off_staged = be411474 (= OFF)      on_staged = 1a68a0ea (= ON)      on_staged2 = 1a68a0ea
```

**零影响。** 该诊断补丁已从工作区还原，`ds4_metal.m` 恢复为并行会话的 barrier 版（未动其代码）。

## 6. 本轮排除清单（累计）

- ❌ `didModifyRange` / Metal 共享内存契约违例（5 腿 A/B + 两个 harness）
- ❌ masked vs plain 内核（禁用后与默认 ON 逐字节相同）
- ❌ addr 内核族 vs slots6 内核族（OFF 强制 addr 表仍 `be411474`）
- ❌ split vs 非 split（源码证明 down 只跑一次、`active_mask=0x3f`，顺序稳定）
- ❌ slab 复用 / 驱逐（`evictions=0 buffer_reuses=0`）
- ❌ slab 数量 / `use_slabs` 覆盖缺口（剂量-响应只值 0.089）
- ❌ `pread()` 落盘方式（staged pread 零影响）
- ❌ 在飞 CPU 写可见性（两个独立 harness，0 mismatch）
- ❌ arm 的 CB/事件结构（热态逐字节相同）
- ❌ M5 parallel full-FFN（`ds4.c` 里 `!g->ssd_streaming` 直接排除，streaming 模式下永不开）
- ❌ arm 与 dispatch 之间存在 CB 边界（已逐行核对 41900–43200，无 flush）

## 7. 剩余嫌疑（收敛到两个）

**(A) 分歧不在 routed MoE 里。** 理由：专家字节（mmap 比对）、地址表（GPU 侧读取）、
内核算术、dispatch 几何、标记覆盖、写可见性全部核对通过；且 OFF 对缓存/ slab 布局完全免疫而
ON 敏感，说明 ON 读到了"与布局相关"的东西——但覆盖缺口只值 0.089。若 (A) 成立，
分歧在层的其余部分（attention / KV / HC mix / shared expert），由"绑定 + miss"间接触发。

**(B) routed MoE 里还有一个量没测过：dispatch 时刻 GPU 实际读到的字节。**
上面所有"字节正确"的证据都是 CPU 侧或扰动探针；唯一没做的是**内核内、不进 CB 时序**的校验。

## 8. 下一步与成本

| 方案 | 内容 | 预估 |
|---|---|---|
| **B1** | 内核内校验器：serve 把 6 专家各段首 8 字节指纹写进地址表 buffer 的备用区（`ensure_addr_buffers` 多分配 4 KB），pair/down 内核读同一地址比对，不一致 `atomic_add` 计数，退出时打印 | **45–60 min**（含改 2 文件 + 编译 + 3 腿） |
| **A1** | 若 B1 证明字节正确 → 转向 (A)：把 L0 的 routed dispatch 强制回同步路径、保留 serve，看分歧是否消失 | ~30 min |
| **关账** | 直接执行 `PR1178_HANDOFF.md` §8（回复评论 + 关闭 PR） | 0 |

**倾向**：(A) 概率约 70%。若 (A) 成立，PR 的核心论断（"六个专家常驻后逐层位一致"）在
**与专家数据路径无关**的地方不成立，修复落点不明确，关账性价比明显上升。

## 9. 产物清单

- `/tmp/pr1178_bar_{off,onnb,onnb2,onfx,onfx2}.json` — barrier A/B
- `/tmp/bar2_{noslab,nomask}.json`、`/tmp/bar3_{offaddr,offnoslab}.json`
- `/tmp/bar4_rep_{off,on}.json` + `.err`（含缓存报告）
- `/tmp/bar5_{off_s512,on_s512,on_s32768}.json` — 剂量-响应
- `/tmp/bar6_{off_staged,on_staged,on_staged2}.json` — staged pread
- `/tmp/bar7_tiny_{off,on}.json` — 最小复现（Ciao, n=4）
- 我版 harness：`/tmp/pr1178_vis/vis_test.m`（可重编）

---

# 结案（2026-10-05 02:0x）：根因已钉死 + 已修复 + 已推上游

## 根因

**`use_slabs()` 是"编码时刻"的 slab 快照，而 slab 是 serve 线程在 command buffer 执行期间惰性创建的。**
落在"快照之后新建的 slab"里的专家，dispatch 时**没有任何 Metal resource 声明**，GPU 读到的是那些页
当时的内容。PR 的 commit message 其实已经点到要害（"a dispatch that reaches its weights through the
address table binds no buffers ... marks the cache slabs instead (19 in this configuration, **with every
published address inside one**)"）——最后那句假设就是洞。

## 定位过程（决定性实验）

在 arm 时记录 `slab_count`、在 serve 时再记录一次，直接量"有多少 dispatch 读到了编码之后才建的 slab"：

```
V41DIAG gap seq=1 layer=0 token=1 slabs 0 -> 1
V41DIAG dispatches reading a slab created after encode=1 max_delta=1
```

**整轮运行恰好一个**，就是第一个 dispatch。它污染首 token 的 KV，之后整段生成漂移 —— 这解释了
step 0 即分叉、两态各自确定、热态无差、以及 slab 数 88→2 的"剂量-响应"几乎平坦（每种配置都污染首 token）。

对照实验（决定性的正反两向）：
- 强制在第一个 dispatch 之前把整个 slab 池建出来（21 slabs / 8793 slots）→ **ON == OFF 逐字节**。
- 只把池在 token 边界扩（上一 token 的 CB 已完成）→ **ON == OFF 逐字节**。

## 修复

- `2c76fd3` metal: reserve expert slab headroom before a gated dispatch is encoded
  - 抽 `ds4_gpu_stream_expert_cache_new_slab()`；
  - token 边界（arm 里按 `decode_tokens` epoch 守卫）预留整 token 最坏用量
    `MAX_LAYER × MAX_SELECTED` 个 slot；
  - allocator 改为挑"仍有空位的最新 slab"（此前老 slab 全满，旧逻辑只看最新一个 ⇒ 行为等价）。
- `4fce595` metal: never hand a gated dispatch an expert outside the slab pool
  - 非 slab 的 `ds4_stream_expert_combined` 缓冲同样不在 `use_slabs()` 覆盖内；改为让该次装载失败，
    由既有的 latch 机制回退同步路径。

已推 `fork/cardi-upstream`，#1178 head = `4fce595`，已回复 Flor1an-B：
https://github.com/antirez/ds4/pull/1178#issuecomment-5982750031

## 验收矩阵（OFF vs ON 逐字节，全 JSON 比对）

| 腿 | md5 |
|---|---|
| 默认缓存 8793 专家 / 34-token / n=48 | `be411474f174292326db2f6f5c2ee31c` |
| 缓存 3000 / 800 / 60 | `be411474…`（同） |
| "Ciao" n=4 | `a2ef0f30628355a5c5660387edaa4760` |
| 34-token n=128 | `92b0484ae8c432c70626493c2d96fc8c` |
| `--ctx 32768` n=48（含 forced-addr 对照） | `fb12a14797ba791e80dfaaf6c49cc999` |
| 6 KB prompt n=32 | `4bb984994a4e41eb3b39cf3f048f3e36` |
| 长 prompt n=100 | `7c7495e6bbc7156e98d168d5813a46df` |
| `ds4_test --logprob-vectors`（`DS4_TEST_SSD_STREAMING=1`） | stderr `f08a2e195234758b08f41e345036d293`（两腿同） |

## 性能（重要，已在上游 PR 正文与评论中撤回 "+2~3%"）

| 配置 | OFF | ON |
|---|---|---|
| 默认缓存 / 34-token prompt / 200 贪心步（ABBA） | 13.67、12.38 t/s | 9.16、10.70 t/s |
| 长 prompt / 100 步（同字节输出） | 22.91 s wall | 24.02 s wall |
| slab target 32 GB（3 slabs 而非 11） | 13.13 t/s | 8.44 t/s |

⇒ **绑定路径在本机是负收益**，且与 slab 标记无关（slab 数 11→3 无改善）。修复本身在噪声内中性
（修前 ON 9.39 t/s）。PR 应作为"正确性修复 + 性能待查"看待。

## 其它

- 并行会话的 `didModifyRange` barrier 修复：**实验证伪**（4/4 ON 腿逐字节相同），已从工作区移除，
  从未进入任何提交。`auto.sh` 因其判定依赖该符号而作废，说明见 `.cardi-pr/legs/README.md`。
- 本轮未采用"关账"路径，故不需要把 PR 提交/诊断代码移出 main；`main` @ `e2ef52c` 未含任何 PR 代码。


---

# 最终裁决（2026-10-05）：修复成立，但功能撤回，PR 已关闭

## 性能 ABBA 复测（决定性）

**热缓存 / 长上下文**（`ds4-bench`，2K→32K frontier，每 frontier 128 贪心 token，A B B A 各两遍）：

| ctx | OFF steady t/s ×2 | ON steady t/s ×2 | delta |
|---|---|---|---|
| 2048 | 18.56 / 17.98 | 18.69 / 18.35 | +1.4% |
| 10240 | 17.58 / 17.26 | 17.60 / 17.39 | +0.4% |
| 18432 | 16.63 / 16.00 | 16.34 / 15.87 | −1.3% |
| 26624 | 16.81 / 16.53 | 15.83 / 15.88 | **−4.9%** |
| 32768 | 18.81 / 18.47 | 18.60 / 18.75 | +0.2% |

decode 均值 **−0.8%**，全部落在 OFF 自身 run-to-run 波动（1.7~3.9%）内；prefill 均值 +1.1%（5/5 微正，同样在噪声内）。
⇒ 当初的 **+2.0%/+3.0% 是单对 ABBA 读到了自身噪声**（同一份数据里 OFF 自己相邻两遍就差 3.2%）。

**冷缓存**（34-token prompt、200 贪心步、3×3 交错）：

| | OFF | ON |
|---|---|---|
| decode t/s | 13.37 / 13.30 / 13.63 | 8.94 / 9.18 / 9.65 |
| prefill t/s | 7.09 / 6.73 / 6.71 | 4.81 / 4.71 / 4.75 |

两组零重叠：**decode 与 prefill 各 −31%**。

## 正确性验收（修复后，比短 prompt 强得多）

`ds4-bench --dump-frontier-logits-dir` 在 **2048 / 10240 / 18432 / 26624 / 32768 五个 frontier 的全量 logit
逐字节相同**。加上此前的缓存 60/800/3000/8793、Ciao n=4、n=128、`--ctx 32768`（含 forced-addr 对照）、
6 KB prompt、长 prompt n=100 —— 全部 OFF == ON 逐字节。

## 处置

- **PR #1178 已关闭**：收尾评论 https://github.com/antirez/ds4/pull/1178#issuecomment-5988494902
  （先发 issuecomment-5982750031 说明根因与修复，再发收尾评论说明为何不值得合并）。
- 理由：~740 行并发机制（新线程 + 双共享事件 + ring + inflight 标记 + 服务线程守卫）、opt-in 默认关、
  输出正确但实测更慢，且已产出**两个静默正确性 bug**（inflight 序列、slab 快照）。
- **分支保留为归档**：`fork/cardi-upstream` 与本地 `cardi-upstream`（head `4fce595`）；
  `main` 从未含这些提交，无需搬移。
- **fork 的 I-1 树已同步修复**：`~/ds4-cardi` @ `p4-cardI-graph-capture` 新增 `701c5db`。
  验收：默认缓存 + Ciao 逐字节一致；`ds4-bench` 五个 frontier 全量 logit 逐字节一致；0 次装载失败。
  已知残留：`--ssd-streaming-cache-experts 60` 这类极小缓存下 serve 线程 reuse 扫描找不到 victim →
  装载回退、该层出错。**该现象在移植前就存在**（当时静默回退到未标记的独立缓冲 `578b30d7` ≠ OFF
  `56f09bc6`），移植只是把它变成显式失败（29 次 "could not load"）。
