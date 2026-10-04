# M0-3 p42-attn-glue 位等价性复核（2026-10-05，M5 Max）

**结论：弃臂存档，不合入。** 分支 `p42-attn-glue` @ `8392281`（WIP archive），worktree 已弃。

## 命令（臂 = worktree `~/ds4-attn-glue` @ 3b2abc4 + 全部未提交 WIP + 测试补 DS4_GPU_TEST_V41_FUSIONS 旗标）

```sh
make tests/test_deepseek41_metal
./tests/test_deepseek41_metal --moe-fuse    # 见 moe-fuse-flagged.log
./tests/test_deepseek41_metal --attn-fuse   # 见 attn-fuse-flagged.log
```

## 结果

- 无旗标时融合臂被 `ds4_gpu_dsv41_exact_admitted`（test flag 或 M3 Ultra+measured）拒载，
  `matvec_bf16` 返回 0 → `--moe-fuse` 在 `==1` 硬断言处红（moe-fuse-noflag.log:512）。
- 补上 `DS4_GPU_TEST_V41_FUSIONS` 强开融合内核后：
  - `--attn-fuse`：`matvec bf16 mismatch: weight 0 (type 8 Q8_0, 5120 -> 1280)`，
    即 store-side BF16 舍入 matvec 在 M5 Max 与 standalone 对**逐字节不一致**（attn-fuse-flagged.log:619）。
  - `--moe-fuse`：probs 契约 `memcmp(probs[0],probs[1],E*4)` 失败（moe-fuse-flagged.log:526）。

## 裁决语义

ITERATION_PLAN M0-3 预裁 = "合入取位、弃 worktree（预期终态，因为实测 ≈中性）"，
前提为位等价。复核证伪前提：这批 exact 内核在 M5 Max 位不等价（house 的
M3-Ultra-only admission 正是为此设，"weight-reading exact kernels stay out of
runs on unmeasured devices"）。§1 铁律"位契约高于一切"→ 走 DoD 的另一支：
删除记录。≈中性（既测）另有解释：本机生产永不分发这些内核（门不含 M5），
此前端到端测的是 rc=0 回退路径。若未来在 M3 Ultra 上做位验证通过，可从
存档 commit 重取；届时按 §13.1 纪律走窗口 bit-exact 门。

## 失败项

- 2/2 契约红（moe/attn-fuse，带旗标）。无速度臂（位门先破，不值窗）。
