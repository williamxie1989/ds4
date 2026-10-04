# M1 开工交接提示词（新会话直接粘贴使用）

当前状态锚点（照此校验后开工）：main HEAD = `74f3519`，领先 origin/main（`0aaea5a`，antirez）78 片，fork/main 已同步；`git status` 干净；`p42-attn-glue` 分支为已弃臂存档（勿动）。M0（还账与账本冻结）已全部关账：台账 = LOCAL_INVENTORY.md 杂项 2026-10 五条 + D 节；KPI 表 = ITERATION_PLAN_2026Q4.md §1 已带 M0 复核列（10-05 窗全绿：V4.1 2K/32K=23.35/22.16、831 冷 append 6.24 s、GLM@12k=33.30、GLM MTP 首账 −19.8%/accept 60.6%）。生产 ds4-server（8055）由用户手动停、短期不起——但**每次加载类测试前仍必须查内存**（组合可用量 free+inactive+spec+purgeable，swap used 取 `used` 后第二个 token）。

## 你要做的事：ITERATION_PLAN_2026Q4.md 的 **M1-1 [O] 与 M1-2 [O]**（编码期全在窗口外，不许加载模型）

先通读 ITERATION_PLAN_2026Q4.md §4 M1 全表 + §2 窗口规程 + §17 三雷，再动代码。

- **M1-1**：`ds4_metal.m` 实现 `ds4_gpu_decode_graphs_supported()/begin()/end()/abort()`（声明在 ds4_gpu.h，先读现有调用点与 CUDA 侧语义）：
  - MTLCommandBuffer capture scope（`captureScopeWithCommandBuffer:`），**capture/begin 前必须先 `ds4_gpu_commands_active()`——flush 会立刻轮换 CB，无条件 begin 会每层静默失败**（§17.4 第一雷）。
  - 图缓存：key = `ds4_decode_graph_key`（il/island/variant + 关键 buffer 身份指针），LRU 容量常数 64 起步；`begin` 命中→安排 replay 返回 1；miss→返回 0 并开 capture；`end` 提交 recompile；`abort` 作废。动态标量走 per-key 固定地址 param buffer（图内读），不做 per-token 重编码。
  - 总开关 env `DS4_METAL_DECODE_GRAPHS`，**默认关**（默认开与否留 M1-3 数据定）。
  - 验收 = 新建 `tests/test_metal_graph_capture.c`（挂 Makefile tests 目标，模式抄 tests/test_glm53_router_shared）：dummy kernels 的 capture/replay/参数更新/abort 重录/容量淘汰全过，**不加载任何模型**。
- **M1-2**：GLM 侧核对（ds4.c:58571+ 的 FFN-tail island 现成接线，含 hc 指针身份 key 字段）在 keyed 语义下的正确性推演写成代码注释/评审纪要；补 `decode graph <key>: captured/replayed n=…` 日志锚。
- 每项 DoD：代码 commit（默认关）+ 单元验收绿 + 台账 LOCAL_INVENTORY B/D 节各一行。改 ds4.c 后必须全量重编 ds4/ds4-server/ds4-bench/ds4_test/tests/*（第二雷）。bit-exact 对照永远用同一次构建的 OFF 腿（第三雷）。

## 铁律
1. 位契约高于一切速度指标；红牌先停，等人裁定，不许静默改默认位。
2. [O]=窗口外随时可做；[W]=必须先过 §2 预检 + 征询用户开窗（停产品、独占、腿间 25s、ABBA≥2 对、产物入 speed-bench/M1/<id>/；窗 runner 抄 speed-bench/M0/m0-4/run_window.sh，门用组合可用量，教训见 m0-4/notes.md 方法学节）。
3. 加载模型前必查三样（可用内存/swap/驻留进程），不确定先问用户，加载类一律后台 job 盯内存。
4. 子代理：provider 为 ds4/omlx-local 一律禁派；否则最多一个。
5. 里程碑出口：push fork main（fork=williamxie1989/ds4）。
6. worktree `ds4-cardi`（p4-cardI-graph-capture）、`ds4-cardh` 是 M2-1 卡 I 资产，本任务不碰。
7. 上游 PR #1177/#1178 仍 OPEN（10-05 复核），窗口前照例 `git fetch origin` + `git cherry` 复查吸收。

## 完成定义
M1-1 单元验收全绿 + M1-2 评审纪要与日志锚就位 + 台账行齐 → 汇报并申请 M1-3 窗口（GLM FFN-tail 首验收：100-case@4096、长 fixture、frontier logits identical、ABBA N=6）。不达标不许"预期达标"式收尾。
