# M0-4 基线复核窗（2026-10-05 13:52–14:21，M5 Max 128 GB，停产品独占）

用户批准立即开全量窗（生产 ds4-server 由用户手动停、短期不起）。全部 14 腿 rc=0；
窗口唯一中断是**自家用门 bug**（见"方法学记录"2），非被测系统问题。

## 命令
- 主 runner：`sh speed-bench/M0/m0-4/run_window.sh`（腿=V4.1 decode ABBA、831/1524 append、
  [4,8) 行 SPEC_ROWS A/B+logits dump、GLM decode G1）；补跑：`run_resume.sh`（g2、m1_smoke）。
- 汇总：`python3 speed-bench/M0/m0-4/summarize.py`。逐腿日志 `logs/`，CSV 在 `v41/`、`glm/`。

## 结果（对照冻结基线，均在 ±15% 中止线内 → 基线全部成立）

| KPI | 冻结 | 本窗复核 | 判定 |
|---|---|---|---|
| V4.1 decode steady 2K/32K（OFF×2 中位） | 22.44 / 21.82 | **23.35 / 22.16** | 成立（+4.1%/+1.6%，噪声带内） |
| V4.1 831 行 append 冷 | 6.1 s | **6.24 s**（首个 append 133 t/s；热后 4.5–4.6 s） | 成立 |
| V4.1 1524 行 append | 194–248 t/s | 187–230 t/s | 成立（下沿略低，噪声内） |
| V4.1 [2,32) 行 append（step4/8） | ≈21–24 t/s | step4 OFF 23.5、step8 OFF 23.8 t/s | 成立——**零批量增益实测复现**（E1 动机再证） |
| GLM decode @12k（G1/G2 中位） | 32.0 | **33.30** | 成立（+4%） |
| GLM decode 短 ctx（2K） | 38–39（旧法，自带高估 4–6% 注记） | 34.75 | 存疑不判负：折算后仍低 ~5%，语义=bench 512tok steady 列，M1-4 起以同建 ABBA 同腿为准 |
| GLM MTP 净收益 | −15% @17k | **effective 25.2 vs plain 31.4 = −19.8%**（1.3k ctx，accept 60.6%，1.48 tok/cycle，verify2 48.4 ms / head+draft 17.3 ms / plain 31.9 ms） | 成立（同负区间；G4a 结算行首次全量落账，M5 翻转目标的起点数据） |
| V4.1 首 token @12K / 100–500 行段 / GLM prefill @41k | — | 未测（不在本窗腿单） | 留各归属里程碑 |

## SPEC_ROWS A/B（M0-2 关账臂，搭本窗）
- decode 各 frontier：**中性**（−0.8%…0.0%，rows=1 不触分支，符合预期）。
- [4,8) 行 append：−0.7% / −2.0% = 中性（噪声内）——该臂不带来批量增益，E1 需真批量内核；
  臂保留默认关作 M3-5 准入种子。
- **位等价采样：step4 13/13、step8 10/10 frontier logits 逐字节相等**（`parity/*.sha256`
  清单，raw dumps 70 MB 已弃）。[2,32) 域在本机无红牌，M3-5 过 E0.1 时可直接引用本采样作先导证据。

## 方法学记录（买来的教训，后续窗口照用）
1. swap 解析 off-by-2：`used` 后第二个 token 才是值（`used = 2617M`），首版取到 `=` 号 fail-open；
   修成 case 校验 + 整数比较 fail-closed。
2. **腿间内存门必须用组合可用量**（free+inactive+spec+purgeable）：96.5 GiB GLM 腿退出后其
   文件页整驻 inactive（即时可回收、零 swap），只计 free 的严格门在 g1 后误触 abort（avail 实际 102 G）。
   真死信号是 swap 增长 + wired/compressor 压力；本窗 swap 全程 2609–2610 M 纹丝不动。
3. ds4-bench `--step-incr N --gen-tokens 0` 即 append-N-行 测量（prefill_tps 列）；
   带 logits dump 的腿 `--gen-tokens ≥1`。
4. MTP 冒烟 15 s 完成是正常的：warm mmap + 1.3k prefill + 256 tok decode。

## 失败项
- 无腿级失败（14/14 rc=0）。窗口中断 1 次 = 门 bug（上 2），已修并 `run_resume.sh` 续跑。
