# 卡 I 交接提示词（新会话直接粘贴使用）

接手 /Users/xieyongliang/ds4/V41_M5MAX_BASELINE_SPEED_PLAN.md 的**卡 I**（§16：整层 encode 进 Metal graph capture，砍主线程指令量）。开工前先通读文档 §14.1/§14.6/§15/§16，它们是本案的全部前置结论，不要重新推导。

## 你要做的事（三段独立闸门，允许中途止损）
- I-1 路由决策 GPU 化：streaming decode 下让 routed kernel 走 use_stream_expert_addr_table 地址表路径直吃 GPU 上的 g->selected ids buffer，service 线程自主装载槽位；主线程删掉 per-layer readback+staging 决策。新 env `DS4_METAL_ENABLE_V41_MOE_GPU_BINDING` 正开默认关，零数学改动。worktree 从 main 当前 HEAD 新建（建议分支 p4-cardI-graph-capture），先只做 I-1 并单独 ABBA（收益假设小，它是 I-2 的前置）。
- I-2 capture 岛扩到单层：把 ds4.c 的 ds41_decode_island + ds4_gpu_decode_graph_begin/end 机制（现门控 tp_world==2 && !streaming，见 ds4.c ~41633）推向单卡+streaming，per-layer 编码压成 1 次回放；动态标量走固定 param buffer（llama.cpp CUDA-graphs staging 抄法，§16.4）。
- I-3 全 token 一图（I-2 达标才做）。

## 铁律（全部是上一张卡用真金白银买的教训）
1. 动手前查内存：sysctl vm.swapusage + pgrep ds4/ds4-server，swap 稳定 <3GB 且无驻留进程才允许加载模型；模型加载类命令一律后台 job。
2. 路径/前置先验证再跑：模型 gguf/DeepSeek-V4.1-Flash-Q2.gguf（worktree 里 symlink 到 /Users/xieyongliang/ds4/gguf），prompt 用 speed-bench/promessi_sposi.txt（不存在 bench/ 目录）；产物写 worktree 内 .cardi-ab/，别用 /tmp（会被清扫）。
3. 改了 ds4.c/ds4_metal.m 后必须全量重编：make -j8 && make ds4_test && make tests/test_deepseek41_dspark && make tests/test_metal_ssd_experts——漏一个就是在测旧包（卡 H 曾因此连败三轮）。
4. ds4_gpu_flush_commands 提交后会立刻轮换出新 batch CB（ds4_metal.m ~9827），begin_commands 在 CB 已存在时返回 0；任何 flush 后的收尾写成 if (!ds4_gpu_commands_active() && !ds4_gpu_begin_commands()) return false。
5. 测试节奏：先读码推断，再一次构造好一次跑通；禁止"跑 5 分钟→失败→看码→重跑"的循环。任何跑模型的重测之前，先用一次性 preflight（test -r/-x、pgrep、swap、脚本 bash -n）把失败可能清零。
6. 禁止派发子代理（provider 铁律）；一切主会话内完成。

## 闸门（顺序不许换）
bit-exact：CLI 贪心 A/B 逐字节（./ds4 --model ... --ssd-streaming --temp 0 -n 48 --prompt-file .cardi-ab/prompt_short.txt，ON vs OFF 同构建两腿 cmp）→ DS4_TEST_SSD_STREAMING=1 ./ds4_test --logprob-vectors（判定口径 = ON/OFF 差分为零；两个 short_code_completion 存量失败 env OFF 同样复现，非本案引入）→ DS4_TEST_SSD_STREAMING=1 ./tests/test_deepseek41_dspark gguf/... --verify-parity-ssd 与 --short-prefill-ssd（prompt 需 >1100 tok，用 6000 字节版）→ ./tests/test_metal_ssd_experts 4/4。
速度：clean ABBA ≥2 对，脚本模式抄 /Users/xieyongliang/ds4-cardh/.cardh-ab/run_abba2.sh（每次调用前缀 env，别用 export/unset；旧结果先归档避免读错），ds4-bench --ssd-streaming --gen-tokens 512 --ctx-start/--ctx-max 2048 与 32768 两档。

## 当前基线数（用于判定，勿用旧文档数）
OFF steady：2K 23.83 / 32K 22.54 t/s（ABBA 4 腿均值，2026-10-04，main 3b2abc4 同代码）。判定线：I-1 单腿收益 ≥3% 才值得进 I-2；I-2 目标 steady ≥ +25%。负结果就诚实记录并关账（卡 H 先例：§15）。

## 现场状态（2026-10-04 交接时）
- 卡 H 已关账：bit-exact 全过、速度负结果；证据与实现留在 /Users/xieyongliang/ds4-cardh（分支 p4-cardh-early-load @5573d62，env DS4_METAL_ENABLE_V41_MOE_EARLY_LOAD 默认关，**不合入**）。卡 H 的 worker/inflight/flush 教训全部适用 I-1，开工前读那个 commit 的 diff 和 message。
- 主树 /Users/xieyongliang/ds4 有未提交 WIP（§13.2 明示勿动）与未跟踪文档；LOCAL_INVENTORY.md card H 行已关账。
- io-diag worktree 有未提交 DS4_METAL_CB_TIMES_MAX 诊断补丁（留用，做 I-2 计时可能有用）；/private/tmp/ds4-main 是游离 worktree，别清理。
- 目标系统里那个卡 H goal 可以作废；本案若需立 goal 用 §16.5 的三段验收原话。
