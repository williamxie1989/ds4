# REQ-001: ds4-server 菜单栏实时 prefill/decode 状态

关联: 无（新需求）
状态: 草案 → 待确认
变更记录: docs/changes/CR-*（如有）

## 目标

ds4-server 运行时,在 macOS 菜单栏实时显示模型的 prefill 进度/速度与 decode 速度,
交互形态复用 `~/.omlx/menubar/omlx_menubar.py`(独立 Python sidecar 轮询 server
只读端点,不碰推理引擎代码)。

用户: 本机开发者。日常用法(现状实测): `./ds4-server -m gguf/GLM-5.3-Flash-Q2.gguf
--metal -c 256000 --port 8055 --vision ...`,默认端口 8000。

## 成功标准

1. `curl http://127.0.0.1:8055/v1/status` 返回 200 + 合法 JSON,字段稳定可解析。
2. 有请求时菜单栏状态实时切换:`idle` → `PP xx%`(蓝) → `GEN xx.x t/s`(绿) →
   `idle`;有排队时显示 `WAIT n`(橙);server 不在线显示 `offline`。
3. 菜单详情能看到:每个 slot 的 prefill 进度(已处理/总量、实时 t/s)、decode
   速度与已生成 tokens、会话累计(请求数、prefill/decode tokens、平均 tps)。
4. 不加载模型的单元测试覆盖:JSON 构造、状态优先级、waiting 计数。

## 已知约束

- 只新增只读 GET 端点,不改任何已有 API/响应体(契约"只新增" = T2)。
- stats 更新路径不得引入锁顺序反转:专用 `stats_mu`,更新端只拿这一把锁,
  快照端拿完 `stats_mu` 再单独短暂拿 `s->mu`(不嵌套)。
- sidecar 是纯 HTTP 客户端,不 import ds4 任何代码(与 omlx 同构)。
- 机器内存余量 18%(128GB 上有一份 GLM-5.3 常驻),验证阶段不加载第二份模型副本。

## 假设与默认选择(每条带不成立的代价)

1. 端点命名 `GET /v1/status`。若不成立,代价 = 全部文档/脚本同步改名,分钟级。
2. sidecar 脚本放仓库 `menubar/ds4_menubar.py`,运行时状态(pid)放
   `~/.ds4/menubar/ds4_menubar.pid`。若用户偏好 omlx 式"仓库外 bypass",
   代价 = 移动文件 + 改 README,分钟级。
3. decode 实时速度 = 最近一次上报窗口的 chunk tps,更新粒度沿用日志节流点
   (每 50 token + 请求结束),不逐 token 加锁。代价 = 速度显示最坏滞后
   ≤50 token(60 t/s 时约 0.8s),对 1s 轮询的菜单栏无感。
4. 会话累计(totals)只在请求成功走到 decode 后统计(以 `trace_finish` 为锚点,
   复用 `j->req.cache_read_tokens/cache_write_tokens`);prefill 失败的请求
   不计入平均速度。代价 = 失败请求不出现在累计里,对显示无影响。
5. 无鉴权(与现有 ds4-server 端点一致,本机环回使用)。代价 = 若未来对外暴露,
   需要另加鉴权,本次不做。
6. sidecar 端口发现用 `--port` 参数(默认 8000),不做配置文件/自动探测。
   代价 = 用户换端口时要手动传一次;README 写明 `--port 8055`。

## 新增契约: GET /v1/status

响应 200 `application/json`(CORS 跟随 `--cors`,与现有端点一致):

```json
{
  "model": "glm-5.3-flash",
  "slots": 1,
  "batched": false,
  "state": "generating",
  "waiting": 0,
  "prefilling": [
    {"slot": 0, "current": 512, "total": 4096, "tps": 812.0, "elapsed": 0.62}
  ],
  "decoding": [
    {"slot": 0, "tokens": 180, "tps": 61.2, "avg_tps": 59.8, "elapsed": 3.1}
  ],
  "totals": {
    "requests": 5,
    "prefill_tokens": 20480,
    "decode_tokens": 1024,
    "avg_prefill_tps": 810.2,
    "avg_decode_tps": 59.8
  }
}
```

- `model`: 引擎模型名(与 `/v1/models` 同源);engine 为空(单测)时省略。
- `prefilling[].current/total`: 沿用 prefill 进度回调的 display 语义
  (cache 命中的前缀不计入进度);`tps` = 最近窗口实时速度;`elapsed` = 自
  prefill 起点的秒数。
- `decoding[].tps` = 最近窗口 chunk 速度,`avg_tps` = 自 decode 起点平均。
- `state` 优先级(与 omlx 菜单栏一致): `prefilling` > `generating` >
  `queued`(waiting>0) > `idle`。
- `waiting` = 尚未派发到 slot 的排队 job 数(快照时短暂拿 `s->mu` 数队列)。

## 菜单栏显示映射(沿用 omlx)

| 菜单栏 | 颜色 | 含义 |
|---|---|---|
| `PP 45%` | 蓝 | 正在 prefill(详情含进度/速度/ETA) |
| `GEN 42.1 t/s` | 绿 | 正在 decode(速度/tokens/耗时) |
| `WAIT 3` | 橙 | 有请求排队 |
| `idle` / `offline` | 灰/暗 | 空闲 / 连不上 |

多 slot 并发时标题后缀 `+n`(与 omlx 相同);菜单逐 slot 列详情,
末尾加 totals 段(请求数、平均 prefill/decode tps)。

## 实施锚点(ds4_server.c,行号为 2025-10-09 工作区快照)

| 位置 | 用途 |
|---|---|
| `struct server`(10403) | 加 `stats_mu`、totals 字段 |
| `struct server_slot`(10263) | 加 per-slot prefill/decode 两个相位结构 |
| `server_progress_cb`(13036) | prefill 相位更新(已有 chunk_tps/avg_tps 计算) |
| `decode_again`/`decode_t0`(14524-14527) | prefill 相位清零 + decode 相位起点 |
| decode 日志节流点(14810, 15005) | decode 相位更新(50 token 粒度 + 结束) |
| `trace_finish` 调用点(15174) | totals 汇总(复用 cache_read/write_tokens) |
| `generate_job`(15413) | 请求结束后清零该 slot 相位(兜底所有提前返回路径) |
| `client_main` 路由(15958 区) | 新增 `GET /v1/status` 分派 → `send_status` |
| server 初始化(16621-16660) | `stats_mu` 初始化;`server_close_resources`(16176) 销毁 |
| 服务端单测注册表(24629) | 新增 `test_server_status_json` 等用例 |

## 任务分解(会话内推进)

1. server: stats 结构 + 更新点 + 快照/JSON + 路由 + 初始化/销毁 → 验证: `make` 编译通过。
2. 服务端单测(不加载模型): JSON 构造、状态优先级、waiting 计数 → 验证:
   `./ds4_test --server` 全绿(先确认内存余量,遵守模型加载铁律)。
3. sidecar `menubar/ds4_menubar.py` + `menubar/README.md`(安装/PyObjC/--port/
   --parent/--pid 冲突处理,照 omlx README 结构) → 验证: 用 mock HTTP server
   伪造 /v1/status 序列(PP→GEN→idle)做菜单栏视觉烟测(不加载模型)。
4. 文档: `docs/SERVER.md` API 表加 `/v1/status` 行 + 小节 → 验证: 对照实现读一遍。
5. 真机 E2E(需要用户重启现有 8055 server 或另择时机): 观察真实请求下
   PP/GEN 切换。验证: 菜单栏 + curl 人工确认。

## 范围外

- 鉴权/TLS、多 server 聚合显示、历史曲线持久化、omlx 侧任何改动。
