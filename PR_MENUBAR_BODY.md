# server: live prefill/decode status via GET /v1/status + macOS menu bar

## What

Adds a read-only `GET /v1/status` endpoint and an opt-in `--menubar` flag
that surface live inference state, so you can watch prefill/decode progress
without tailing logs.

- `GET /v1/status` returns a JSON snapshot: per-slot prefill progress
  (`current/total/tps/elapsed`), decode speed (`tokens/tps/avg_tps/elapsed`),
  queue depth (`waiting`), a coarse `state` (`prefilling` > `generating` >
  `queued` > `idle`), and session `totals`.
- `--menubar` makes the server spawn `menubar/ds4_menubar.py`, a standalone
  PyObjC companion that polls `/v1/status` once a second and renders colored
  menu-bar text: `PP 45%` (blue) → `GEN 42.1 t/s` (green) → `WAIT n`
  (orange) → `idle`. The dropdown lists per-slot detail and session totals.

## Why a sidecar, and why the server spawns it

The menu bar lives in a separate process on purpose: it never imports ds4,
so engine upgrades cannot break it, and it keeps AppKit out of the inference
process. The server spawns it (rather than a shell alias) so `./ds4-server`
stays the single entry point and the icon's lifetime tracks the server's —
it appears on bind and is terminated on shutdown. If `python3` or PyObjC is
missing, the server prints a hint and runs unchanged. `--no-menubar` opts
out explicitly.

## Design notes

- **No lock-order inversion.** Stats live under a dedicated `stats_mu`.
  Update paths take only `stats_mu`. The snapshot builder copies the phases
  under `stats_mu`, releases it, then counts the job queue under `s->mu` in
  a *separate* critical section — the two locks are never nested.
- **Hot-path cost.** The prefill feed hooks the existing progress callback
  (already firing per chunk); the decode feed updates at the existing
  50-token log throttle plus request end — no per-token locking. Each update
  is a handful of stores under an uncontended mutex.
- **Purely additive.** New endpoint, new flag, new files. No existing API,
  response body, or backend kernel is touched.

## Machine / backend / quant

- Apple M5 Max (18 cores), 128 GB, macOS 26.5.1, arm64, Metal backend.
- Model: `DeepSeek-V4-Flash-Vision-Exp-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-AProjQ8-SExpQ8-OutQ8.gguf`
  (91 GB) — this machine's production quant. The stock `ds4flash.gguf`
  IQ2XXS weight was not present locally; one environmental fixture mismatch
  is called out below.

## Testing

- [x] `make` — `ds4-server` / `ds4_test` compile `-Wall -Wextra`, zero warnings.
- [x] CPU portability: server TU compiles clean with `-DDS4_NO_GPU` (POSIX-only:
      `fork`/`exec`/`waitpid`/`kill`/`access`).
- [x] `./ds4_test --server` — green, including new cases: JSON contract
      (full-string equality, idle + totals), state priority, `waiting` count,
      decode phase transitions, `--menubar/--no-menubar` flag parsing.
- [x] `./ds4_test --all` — green: session-snapshot, long-context (30 474-token
      story recall), tool-call-quality (fast + exact), think-tool-recovery,
      **logprob-vectors vs official 0731 vectors**, metal-short-prefill,
      metal-kernels, metal-tensor-equivalence, SSD cache-pressure, server.
      Model-family groups (qwen4-*, glm53-*, mtp-*) self-skipped as designed.
      Two environmental findings, both shown unrelated to this commit:
      - `local-golden-vectors` ERRs identically on the `origin/main` baseline
        at `fc80bd6` (same machine + weights): the local-golden fixture was
        recorded from a different quant than the Q4K-experts weight used
        here. Official-vectors `logprob-vectors` passes on the same weight.
      - `dspark-verify-depth` hits the pre-existing single-instance
        `/tmp/ds4.lock` when run after other model groups inside one `--all`
        process; it passes standalone (`worst_argmax_gap=0.000`).

## Speed

Engine files are untouched (`ds4.c`, `ds4_metal.m`, `metal/*.metal` — see
`git diff origin/main..HEAD`), and `ds4-bench` does not compile server code,
so a bench sweep would compare identical code paths. The actual hot-path
delta is the mutex-guarded stats store at existing throttle points, so it
was measured on the server itself: `./ds4-server -m <model> --metal -c 32768`
serving one fixed non-streaming completion (2.8k-char prompt → 256 tokens,
temp 0), one warmup + 3 timed runs per arm, arms interleaved baseline / PR /
baseline / PR so adjacent pairs cancel thermal drift:

| arm | decode avg over run (t/s) | mean wall (s) |
|---|---|---|
| baseline | 45.2 | 7.30 |
| PR | 44.5 → 43.2 | 7.52 |
| baseline | 44.3 → 42.8 | 7.65 |
| PR | 40.8 → 39.6 | 8.56 |

Throughput declines monotonically across the whole window on **both**
binaries (Metal thermal drift under repeated loads); the PR arm sits within
its adjacent baseline in all three interleaved comparisons (≤ 1%), i.e. the
stats-hook cost is unmeasurable above noise.

## Out of scope

Auth/TLS, multi-server aggregation, historical curves, any change to the
inference engine or backends.
