# ds4 菜单栏实时状态图标 — 安装说明

一个单文件菜单栏小工具:`ds4-server` 运行时在 macOS 菜单栏实时显示推理状态
(prefill 进度/速度、decode 速度),不用开浏览器盯日志。

| 菜单栏显示 | 含义 |
|---|---|
| `PP 45%`(蓝) | 正在 prefill(进度/速度/ETA,点菜单看详情) |
| `GEN 42.1 t/s`(绿) | 正在 decode(实时速度/已生成 tokens/耗时) |
| `WAIT 3`(橙) | 有请求排队 |
| `idle` / `offline` | 空闲 / 连不上 server |

数据来源是 ds4-server 的只读端点 `GET /v1/status`(1 秒轮询一次)。

## 前提

- macOS(Apple Silicon),已在跑 ds4-server(如 `./ds4-server -m gguf/... --port 8055`)
- 一个装了 PyObjC 的 Python 3.11+

## 第 1 步:装 PyObjC

```bash
python3.11 -m pip install pyobjc-framework-Cocoa
```

(装哪个 python 都行,只要那个环境有 PyObjC;本机 homebrew 的 python3.11 已装好。)

## 第 2 步:放下脚本

脚本随仓库分发:`menubar/ds4_menubar.py`。克隆 ds4 仓库即可,无需拷贝。

## 第 3 步:启动(手动方式,先验证)

```bash
cd ds4 仓库目录
python3.11 menubar/ds4_menubar.py --port 8055 &
```

菜单栏立刻出现 `idle`(server 没开时显示 `offline`)。发一个请求试试,
会实时变 `PP xx%` → `GEN xx.x t/s`。点图标可看逐槽详情、本次会话累计
(prefill/decode 平均 t/s、tokens 总量),以及"Open /v1/status"直接打开 JSON。

## 第 4 步(推荐):让图标随 `ds4-server` 自动起停

`ds4-server` 自带 `--menubar`:bind 成功后 fork/exec `menubar/ds4_menubar.py`,
退出时 SIGTERM 掉它,图标跟着 server 生命周期出现/消失。**带不带 `./` 前缀都行**:

```bash
./ds4-server -m gguf/GLM-5.3-Flash-Q2.gguf --metal -c 256000 --port 8055 --menubar
```

- sidecar 按**当前工作目录**找脚本(和 Metal kernel 同一套约定),在仓库目录里起即可;
  在别处起用 `--chdir /path/to/ds4`。
- 用的是 `PATH` 里的 `python3`;没有 PyObjC 时脚本秒退,server 日志里给出提示后照常服务。
  用 `--no-menubar` 显式关掉(后出现的 flag 生效)。
- 调试:`DS4_MENUBAR_DEBUG=1 ./ds4-server ... --menubar`,sidecar 的轮询日志不再吞掉。

### 旧办法:zsh 函数(别和 `--menubar` 同时用)

`~/.zshrc` 里那段函数靠"不带 `./` 前缀"才拦得到(`./ds4-server` 是路径调用,zsh 不会走
函数,直接跑二进制)。它只传 `--port` 给图标,**不需要**再加 `--menubar`;两边同时用会
重复拉起,第二个实例被 PID 文件挡掉,server 日志里出现"sidecar exited immediately"提示
(功能无碍,但白扯一个进程)。用了 `--menubar` 就可以把这段函数删掉。

## 常见问题

- **没有图标**:多半是第 1 步装错了 Python 环境;在对应环境里
  `pip install pyobjc-framework-Cocoa` 即可。
- **一直 `offline`**:server 没跑,或端口不对;用 `--port 8055` 显式指定。
- **出现两个图标**:`pkill -f ds4_menubar.py` 清掉旧的,删
  `~/.ds4/menubar/ds4_menubar.pid` 再启动。
- **调试**:`DS4_MENUBAR_DEBUG=1 python3.11 menubar/ds4_menubar.py --port 8055`
  能在终端看到每次轮询和标题变化日志。

## 原理(一句话)

独立 Python 进程(PyObjC)轮询 `GET /v1/status`(1 秒),把 prefill/decode
状态渲染成菜单栏彩色文字;不 import ds4 任何代码,升级 ds4 不受影响;
PID 文件 `~/.ds4/menubar/ds4_menubar.pid` 防重复实例;`--parent` 跟随
server 进程自动退出。
