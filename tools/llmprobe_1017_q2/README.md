# llmprobe 1017 Q2 十题复跑

这个目录只做一件事：**在你已经跑着的 ds4-server 上，复跑 `ddalcu/llmprobe` 的“前 10 个 reasoning questions”**，并把结果汇总成“对 / 错 / 超 token / 引擎错误”。

它**不加载模型、不启动第二个 ds4-server、不重新量化**。模型已经在 `127.0.0.1:8055` 上跑着，脚本只发 OpenAI 请求。

对应评论是 `antirez/ds4` Issue #1224 里 `lin16303` 的那条：

> `I tested the first 10 reasoning questions with ddalcu/llmprobe and got 7 wrong, while GLM-5.3-Flash-Sushi-2.4bpw got all of them right. I feel there might be some issue with this quantization.`

## 1. 前置条件

- 模型已经在跑，且加载的是 **1017 Q2 + 1017 Vision**：
  - `gguf/DeepSeek-V4.1-Flash-1017-Q2.gguf`
  - `gguf/DeepSeek-V4.1-Flash-1017-Vision.gguf`
- 默认端点：`http://127.0.0.1:8055/v1`
- 默认模型 id：`deepseek-v4.1-flash`
- Node >= 22.19.0；脚本固定用 `llmprobe@0.6.17`（与评论时间附近的版本一致）。
- 本机内存已经比较紧（模型常驻），所以**不要**再启动另一个大模型进程。

## 2. 快速开始

```sh
cd /Users/xieyongliang/ds4

# 只检查端点/模型/内存，并打印将要执行的命令，不跑 eval
./tools/llmprobe_1017_q2/run.sh --dry-run

# 正式跑 10 题（默认 core 套件、reasoning medium、greedy）
./tools/llmprobe_1017_q2/run.sh
```

默认输出到 `/tmp/ds4-llmprobe-1017q2/`：

- `report.json`：`llmprobe --save` 的 JSON 报告（核心是 `reasoning.cases`）
- `run.log` / `run.err.log`：stdout/stderr
- `summary.txt`：`summarize.py` 的汇总

也可以单独汇总已有报告：

```sh
python3 tools/llmprobe_1017_q2/summarize.py /tmp/ds4-llmprobe-1017q2/report.json
```

## 3. 实际执行的 llmprobe 命令

脚本等价于：

```sh
npx --yes llmprobe@0.6.17 http://127.0.0.1:8055/v1 \
  --model deepseek-v4.1-flash \
  --eval-only \
  --eval-questions 10 \
  --eval-suite core \
  --reasoning medium \
  --concurrency 1 \
  --save /tmp/ds4-llmprobe-1017q2/report.json \
  --no-save \
  --no-color \
  --label "1017 Q2 HF file; 10q core eval; reasoning medium; greedy"
```

为什么是这些参数：

- `--eval-only`：只跑 reasoning eval（仍会做一次免费的 surface discovery）。
- `--eval-questions 10`：取 92 题 core eval 的**前 10 题**（GPQA Diamond / SuperGPQA / AIME2025 / COMPSEC 交错）。
- `--eval-suite core`：默认套件；`hard` / `code` 是另外的题集，不能混比。
- `--reasoning medium`：llmprobe 的默认 thinking effort；与评论者默认运行最接近。
- `--concurrency 1`：串行，降低 KV/内存压力；本机 free 很低。
- 默认 greedy（不传 `--sampling`）；`--sampling precise` 等会改变可比性。

## 4. 怎么判读结果

`summarize.py` 会打印：

```
suite=core total=10 passed=? failed=? stopped=? error=?
```

四个状态的含义：

| status | 含义 | 算“错题”吗 |
|---|---|---|
| `passed` | 答案正确 | 不算 |
| `failed` | 答案错误 | **算**（评论里的“7 wrong”通常指这个） |
| `stopped` | token 用尽/没有 `Answer:` 行 | **不算**；这是预算事实 |
| `error` | 请求/引擎错误 | **不算**；这是基础设施问题 |

所以先看 `failed` 是不是 7；不要把 `stopped` / `error` 混进“错题”。如果 `stopped` 或 `error` 很多，先修 `--eval-max-tokens`、超时或服务，再谈质量。

`scopeNote` 会明确写“只跑了前 N 题”；**10 题是 floor check，不是完整质量基准**，不能和 92 题全量直接比。

`bySource` 会显示这 10 题来自哪些来源；不同来源难度不同，7/10 的分布可能集中在某一类。

`reasoningEffortRejected=true` 表示引擎拒绝了 `reasoning_effort`，实际按默认跑；这时与显式 medium 不完全可比。`thinkingOff` 非空也说明 `off` 有回退/异常。

## 5. 复现那条评论的判断规则

- **`failed=7`、`passed=3`**：复现了那条 anecdote；但**仍然不能证明是量化缺陷**。需要基线对比（见下）。
- **`failed` 不是 7**：没有复现；可能是 llmprobe 版本/题集、sampling、reasoning effort、文件、可执行程序或服务差异。
- **`failed=0`**：这条评论在这份文件/这套设置下不成立；不能据此说“量化没问题”，只能说明这 10 题没复现。
- **`stopped`/`error` 很多**：先别解读质量；先修预算/超时/服务。

## 6. 基线对比（把“量化”与“模型/评测”分开）

`summarize.py` 支持 `--baseline`：

```sh
python3 tools/llmprobe_1017_q2/summarize.py \
  /tmp/ds4-llmprobe-1017q2/report.json \
  --baseline /path/to/baseline.json
```

或直接：

```sh
./tools/llmprobe_1017_q2/run.sh --baseline /path/to/baseline.json
```

基线必须用**同样 10 题、同样 llmprobe 版本、同样 reasoning/sampling 设置**生成。可用的基线：

1. **官方 1017 全精度**（Transformers/vLLM 的 OpenAI 兼容端点）跑同样 10 题——分离“模型能力/评测难度” vs “量化损伤”。
2. **更高精度的 ds4 1017 quant**（如果有）——看 Q2 是否比高精度明显差。
3. **同一 1017 revision 重新量化**（官方 `deepseek41_quantize.py` + 同一 imatrix）——分离“recipe 整体不适合” vs “这份文件有具体偏置”。
4. **旧 checkpoint df42c109 的 Q2**——看 7/10 是不是 1017 特有；不是 apples-to-apples 的模型对比，但能看趋势。

**不要**把 `GLM-5.3-Flash-Sushi-2.4bpw` 当 apples-to-apples 基线：它是不同模型（GLM）、不同格式（Sushi/EXL3）、不同引擎（Sushi），不是 ds4 的 IQ2_XXS/Q2_K 工具链。它全对 10 题只能说明“另一个模型/另一条工具链在这 10 题上更好”。

## 7. 安全与内存

- 模型已经在跑；脚本只发请求，**不会**启动第二个模型。
- 脚本会打印 `hw.memsize`、`vm_stat`、`vm.swapusage`；free 很低或 swap 已用 >1 GiB 时会警告。
- 跑 eval 时建议另开一个终端看 `vm_stat` / `sysctl vm.swapusage`；swap 快速增长就 Ctrl-C。
- 如果不想看内存检查：`--no-memory-check`。

## 8. 本地文件核验（这次已做）

- Q2：`365,713,686,528` bytes；SHA256 `f57e6c3e5e68c513249e1c472a2600bf59792ca78cacb7ac0531a6056376856e`
- Vision：`970,555,552` bytes；SHA256 `861f1097e8686111fa23dc4a6bf139fd0c4cc114db2f1b4090a267f7683c0dbf`
- 两个头部 `general.source.revision = 1bae0e1088993b9b82de08f59de5be5a34e42170`
- Q2 1046 个张量，`blk.39.ffn_down_exps.weight` 为 Q2_K；Vision 266 个张量
- SHA256 与 HF README checksum / LFS oid 一致

这说明文件完整、revision 正确、张量齐全；**这个测试测的是质量，不是完整性**。文件完整不等于量化质量一定好，也不等于量化一定有缺陷。

## 9. 版本/可比性备注

- 固定 `llmprobe@0.6.17`；评论者当时若用别的版本/题集，结果可能不同。
- `--eval-suite hard` / `code` 是不同 deck；`--sampling precise/balanced/creative` 与 greedy 不可直接比。
- `--reasoning off/low/high/default` 与默认 medium 不可直接比；报告会写 `reasoningEffort`/`rejected`。
- 10 题 run 的 `scopeNote` 会标“不完整”；要更强证据，跑完整 92 题或 ds4 的 `score_official`。
