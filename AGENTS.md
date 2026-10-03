# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Build & Test

```sh
make              # Build ds4, ds4-server, ds4-bench (Metal on macOS, CUDA on Linux)
make cpu          # CPU-only build for reference/debug (DS4_NO_GPU)
make test         # Build and run unit/regression tests (./ds4_test --all)
./ds4_test --logprob-vectors   # Logprob vector comparison tests
./ds4_test --server             # Server integration tests
make clean        # Remove all binaries and .o files
```

macOS defaults to Metal backend. Linux defaults to CUDA. Set `CUDA_ARCH` for cross-building:
```sh
make CUDA_ARCH=sm_120
```

Download model weights before running:
```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM machines
./download_model.sh q4-imatrix   # >= 256 GB RAM machines
```

Set `DS4_TEST_MODEL` env var to point tests at a specific GGUF file.

## Architecture

**DwarfStar 4** is a single-model inference engine for DeepSeek V4 Flash. It is not a generic GGUF runner — the tensor layout, quantization mix, and metadata are specific to the published GGUFs for this project.

Three backends share the same `ds4_session` / KV cache / checkpoint format:
- **Metal** (macOS, default): `ds4_metal.m` + `metal/*.metal` kernels
- **CUDA** (Linux, default when built with NVCC): `ds4_cuda.cu`
- **CPU** (reference/debug): compiled with `-DDS4_NO_GPU` via `make cpu`

### Source layout

| File | Purpose |
|---|---|
| `ds4.c` | Core engine: GGUF loading, tokenizer, CPU reference kernels, Metal graph scheduling, session management, KV cache, disk-cache payload serialization |
| `ds4.h` | Public API header — narrow interface for CLI and server code |
| `ds4_cli.c` | Interactive CLI with linenoise REPL, multi-turn chat, `/think`/`/nothink` commands |
| `ds4_server.c` | HTTP server: OpenAI `/v1/chat/completions` + `/v1/completions`, Anthropic `/v1/messages`, SSE streaming, tool-call mapping, disk KV cache policy |
| `ds4_metal.m` | Objective-C Metal runtime: kernel compilation, buffer management, graph dispatch |
| `ds4_cuda.cu` | CUDA graph executor: kernel launches, cuBLAS integration, GPU memory management |
| `ds4_gpu.h` | Shared GPU interface header (used by both Metal and CUDA backends) |
| `ds4_bench.c` | Incremental prefill/generation throughput benchmark |
| `metal/*.metal` | 20 Metal compute kernels (dense, MoE, flash attention, RoPE, norm, etc.) |
| `ds4_cuda.cu` | CUDA kernels and graph executor (434KB) |
| `ds4_metal.m` | Objective-C Metal runtime and kernel dispatch (637KB) |
| `ds4.c` | Core engine: model loading, tokenizer, CPU path, Metal graph scheduling, sessions, disk-cache serialization (780KB) |
| `ds4_server.c` | HTTP server: OpenAI/Anthropic-compatible API, streaming, tool calls, disk KV cache |
| `ds4_cli.c` | Interactive CLI with linenoise REPL |
| `rax.c` / `rax.h` | Embedded radix tree library (used for tool-id exact-DSML replay map) |
| `linenoise.c` / `linenoise.h` | Embedded line editing library for the CLI REPL |
| `ds4_bench.c` | Incremental throughput benchmark at context frontiers |
| `ds4.h` | Public engine API — narrow header for CLI/server consumers |

### Metal kernels (`metal/`)
20 compute kernels: dense ops, MoE, flash attention, RoPE, KV cache, norms, softmax, quantization ops, etc.

### Tests (`tests/`)
- `ds4_test.c` — C test runner with server tests, graph correctness tests, logprob vector comparison
- `test-vectors/` — short and long-context continuation vectors from official DeepSeek V4 Flash API
- `long_context_security_prompt.txt` — long-context security test fixture

### Other directories
- `bench/` — benchmark prompt text (`promessi_sposi.txt`), chart SVG, thinking pool test script
- `dir-steering/` — activation steering tools and examples
- `gguf/` — downloaded model weight files

## Key Architecture

- **Engine vs Session**: `ds4_engine` is the loaded model (mmap-backed GGUF, immutable). `ds4_session` is one mutable inference timeline owning the KV cache and logits. Multiple sessions can be created from one engine.
- **Backend abstraction**: `ds4_gpu.h` declares backend-agnostic GPU graph types and functions. `ds4_metal.m` and `ds4_cuda.cu` implement them. `ds4.c` selects the backend at compile time. `DS4_NO_GPU` flag enables the CPU-only path.
- **KV cache is a disk citizen**: The compressed KV cache of DeepSeek V4 Flash is designed for disk persistence. The server writes checkpoints to disk so stateless agent clients can reuse prefixes across requests.
- **Server is single-session**: One live KV checkpoint in memory; concurrent requests serialize on the graph worker. Disk KV cache is the resume mechanism for switching between sessions.
- **Tool call exact replay**: DSML tool calls are stored by tool ID for byte-exact replay, avoiding KV cache rebuilds on subsequent turns.

## Build commands

```sh
make              # Build ds4, ds4-server, ds4-bench (default GPU backend)
make cpu          # Build CPU-only binaries (DS4_NO_GPU flag)
make test         # Build and run ./ds4_test --all
./ds4_test --logprob-vectors   # Run logprob vector regression tests
./ds4_test --server            # Run server integration tests
make clean        # Remove all binaries and .o files
```

Set `CUDA_ARCH` for cross-compilation: `make CUDA_ARCH=sm_120`.

## Debug commands

```sh
./ds4 --dump-tokens -p "text"          # Tokenize and exit before inference
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "text"
./ds4-server --trace /tmp/trace.txt ... # Full session trace
```

## Model setup

```sh
./download_model.sh q2-imatrix   # 96/128 GB RAM
./download_model.sh q4-imatrix   # >= 256 GB RAM
./download_model.sh mtp           # Optional MTP speculative decoding model
```

## Project structure

- `ds4.c` (~780KB) — Core engine: model loading, tokenizer, CPU reference, Metal graph scheduling, sessions, disk-cache payload serialization. The central file.
- `ds4.h` — Narrow public API header. Engine open/close, session create/sync/eval, tokenization, chat prompt encoding, KV cache save/load.
- `ds4_cli.c` — Interactive CLI with linenoise REPL, `/think`/`/nothink` commands, multi-turn chat.
- `ds4_server.c` — HTTP server with OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` APIs, streaming, tool-call handling, disk KV cache policy.
- `ds4_metal.m` — Objective-C Metal runtime: buffer management, graph dispatch, kernel wrapper functions.
- `ds4_cuda.cu` — CUDA graph executor: GPU memory management, kernel launches, cuBLAS integration.
- `ds4_gpu.h` — Shared GPU interface header (tensor definitions, kernel declarations).
- `metal/*.metal` — 20 Metal compute shader kernels (attention, MoE, dense, RoPE, norms, etc.)
- `ds4_bench.c` — Incremental throughput benchmark tool.
- `ds4_cli.c` — CLI/REPL frontend with linenoise integration.
- `tests/ds4_test.c` — Test runner: engine tests, logprob vector comparison, server integration tests.
- `tests/test-vectors/` — Official-API continuation vectors for regression detection.
- `dir-steering/` — Directional steering activation edit support.
- `bench/` — Benchmark text data and thinking pool test script.
- `download_model.sh` — Download GGUF weights from HuggingFace.

## Upstream picks & LOCAL_INVENTORY.md maintenance

This fork carries cherry-picks from upstream (antirez/ds4) PRs that upstream has
not merged, plus local work layered on top of them. `LOCAL_INVENTORY.md` at the
repo root is the ledger for that state (A = cherry-pick↔PR tables, B = local
commits, C = local-work → cherry-pick dependencies, D = active/abandoned work
verdicts). Keep it true:

- **Record on creation.** Whenever you cherry-pick from an upstream PR (or a
  branch that tracks one), or create a local commit that builds on / reconciles
  a cherry-pick, update `LOCAL_INVENTORY.md` in the same session: PR number,
  original commit, landed commit(s), adoption level (picked n/m), and — for
  dependent local commits — the B/C rows. An evaluated-but-rejected PR also gets
  an entry with the rejection evidence, so nobody re-tries it.
- **Commit convention.** Cherry-pick commits must carry both the standard
  `(cherry picked from commit <sha>)` trailer (`git cherry-pick -x`) and an
  `Upstream-PR: #NNNN` line, so the ledger can be reconciled mechanically.
- **Delete on absorption.** After `git pull --rebase` (or any rebase onto an
  updated `origin/main`) that absorbs a cherry-pick — upstream merged the PR and
  the pick was dropped during replay — **delete the corresponding entry** from
  `LOCAL_INVENTORY.md` (optionally leaving a one-line "merged upstream <sha>"
  note). Detect absorption with `git fetch origin` first, then `git cherry -v
  origin/main main` / `git patch-id`, not by memory.
- **Re-check after rebase.** Rebases rewrite local SHAs: refresh every SHA
  referenced in LOCAL_INVENTORY.md and re-validate the C-section dependencies.
  Never rewrite the `pr-*` / `up/pr/*` refs — they are the original patch-id
  sources used for matching.
- **Fork sync.** `fork/main` (williamxie1989/ds4) is the off-machine backup; do
  not let it lag `main` by a whole work batch.