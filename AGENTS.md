# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, ...). Single source of truth; `CLAUDE.md` is only a pointer. Read this first, then the linked docs.

## What this repo is

DwarfStar 4 is a small native inference engine for a few models — DeepSeek V4/V4.1 Flash (with vision), DeepSeek V4 PRO, GLM 5.2/5.3/5.3 Flash, Qwen3.8 Flash Next — **not** a generic GGUF runner. Tensor layouts, quantization mixes, and metadata are specific to this project's GGUFs. Fork of `antirez/ds4` with local cherry-picks; `LOCAL_INVENTORY.md` is the ledger.

## Direction & engineering rules

- Production path is whole-model Metal graph inference; keep it fast and correct. Fixes must not break SSD streaming, CUDA, ROCm, distributed/TP, or Metal default.
- Metal default loading stays mmap-backed; SSD streaming uses explicit buffers and hides loads; CPU is reference/debug only.
- Correctness before speed: no faster path with unexplained attention/KV/logit drift. Keep long agent sessions practical via live KV reuse and disk checkpoints.
- Keep code small, sharp, readable. No slop: fragile special-case patches, dead code, unnecessary abstraction.
- Comment important inference code beside the implementation; explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow; CLI/server/agent code must not know tensor internals. No C++; no permanent semantic variants behind flags (diagnostic switches are fine).
- Match existing style; surgical changes; don't refactor unrelated code.

## Safety

- Never run multiple huge model processes concurrently; the instance lock is intentional.
- Before loading a model or model-backed tests, check free RAM, swap, and other LLM processes; don't OOM the machine.
- Avoid large CPU inference on macOS (kernel VM failures with huge mappings); CPU is compile/reference only.
- Ask the user before distributed/TP or CUDA/ROCm testing; don't assume access.
- Keep traces and saved sessions private; they may contain real conversations.

## Build & test

```sh
make                 # macOS: Metal; Linux: CUDA. Builds ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent.
make cpu             # CPU-only reference/debug build (DS4_NO_GPU).
make test            # Build + run default suite (ds4_test, ds4_agent_test, ds4-eval, session/memory/model tests; needs right GGUF/backend).
./ds4_test           # Same as --all.
./ds4_test --server  # API/prompt/tool/KV server tests (model-free).
./ds4_test --logprob-vectors      # Official continuation vectors.
./ds4_test --local-golden-vectors # Long-prefill logit drift.
./ds4_test --long-context         # Long-context fact recall.
./ds4_test --tool-call-quality    # DSML tool-call behavior.
./ds4_test --metal-kernels        # Isolated Metal kernel numerics.
make clean           # Remove binaries and .o files.
```

- Platform builds: `make cuda-spark`, `make cuda-generic`, `make cuda CUDA_ARCH=sm_120`, `make strix-halo` / `make rocm`, `make test-rocm`, `make cuda-regression`.
- Model-free checks: `make ds4_test ds4_agent_test test-session-state`; `./ds4_agent_test`; `./ds4_test --server`; Metal tensor tests `make tests/test_session_state_gpu tests/test_glm53_kda tests/test_mxfp4_metal`.
- Model-specific: `make test-deepseek41-*`, `make check-mxfp4-half-lut`, `make dspark-verify-depth`, `make mtp-verify-depth`, `make metal-decode-schedule-bench`.
- Eval: `./ds4-eval --validate-cases`, `--self-test-extractors`, `--suite core|hard|all`, `--list-cases`, `--regrade-trace`.
- `DS4_TEST_MODEL` selects the GGUF (default `ds4flash.gguf`); `DS4_TEST_VECTOR_FILE`, `DS4_TEST_LOCAL_GOLDEN_FILE`, `DS4_TEST_LONG_PROMPT` override fixtures. Vectors must match the checkpoint (`tests/test-vectors/flash-0731` for 0731 GGUFs).
- Downloads: `./download_model.sh ds4f-q2` (~81 GB; 96/128 GB), `ds4f-q4` (~153 GB; 256+ GB), `ds41f-q2` (~341 GB disk / 152 GiB main; Metal only, SSD streaming on 128 GB), `glm53-q2`, `qwen38-q2`; optional `ds4f-dspark`. See `docs/MODELS.md`.
- Debug: `./ds4 --dump-tokens -p "text"`; `./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "text"`; `./ds4 --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt`; `./ds4-server --trace /tmp/trace.txt`. See `docs/TESTING.md`.

## Verification gates

- Before commit: `make` warning-free; relevant tests; `git diff --check`. Never skip or comment out tests to go green — fix the code first.
- Bugfix: regression test first (red → green).
- Inference changes: test Metal default + SSD streaming; distributed/TP and CUDA/ROCm only with user access. Record machine, backend, quant, commands, failures.
- Tokenizer/template/KV/kernel/quant/prompt changes: `--logprob-vectors` with matching vectors; API/prompt/cache/tool: `--server`; long-prefill: `--local-golden-vectors`; Metal command-buffer/copy: `make test-metal-command-memory`.
- Speed: `./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128`; accept a regression only when fixing a correctness bug. Release: `QA_BEFORE_RELEASES.md`; matrices in `docs/TESTING.md` and `CONTRIBUTING.md`.

## Architecture at a glance

- **Engine vs session**: `ds4_engine` is the immutable mmap-backed model; `ds4_session` is one mutable timeline owning KV and logits; multiple sessions per engine.
- **Backend abstraction**: `ds4_gpu.h` declares backend-agnostic graph/tensor types; `ds4_metal.m`, `ds4_cuda.cu`, `ds4_rocm.cu` implement them; `ds4.c` selects at compile time; `DS4_NO_GPU` selects CPU. All backends share the same `ds4_session`/KV/checkpoint format.
- **KV cache is a disk citizen**: compressed KV is designed for disk persistence; the server writes checkpoints for prefix reuse; disk KV is the resume mechanism for switching sessions. Disk is the additional persistence layer, not the only way to retain sessions.
- **Server sessions**: one resident session by default; `--batched-session N` preallocates independent KV slots and queues when busy; concurrent requests serialize on the graph worker.
- **Tool-call exact replay**: DSML tool calls are stored by tool ID for byte-exact replay, avoiding KV rebuilds.
- **Other paths**: SSD streaming (routed experts/Engram on disk with a cache budget; support is model/backend-specific), distributed TP/pipeline, native `ds4-agent`, vision encoders, DSpark/MTP. See `README.md` and `docs/`.

## Source map

| Area | Files | Purpose |
|---|---|---|
| Engine/API | `ds4.c`, `ds4.h` | Central file: GGUF loading, tokenizer, CPU reference kernels, Metal graph scheduling, sessions, KV cache, disk payload, streaming/distributed glue. `ds4.h` is the narrow public API (engine/session lifecycle, sync/eval, tokenization, chat prompt, KV save/load). |
| CLI/server | `ds4_cli.c`, `ds4_server.c` | CLI/linenoise REPL, multi-turn chat, `/think`/`/nothink`; HTTP server: OpenAI `/v1/chat/completions` + `/v1/completions`, Anthropic `/v1/messages`, SSE, tool-call mapping, images, disk KV policy. |
| Agent/bench | `ds4_agent.c`, `ds4_bench.c` | Native coding agent (direct inference, sessions, tools, web); incremental prefill/generation throughput benchmark at context frontiers. |
| Metal/CUDA | `ds4_metal.m`, `ds4_cuda.cu`, `metal/*.metal`, `cuda/mmq/` | Metal runtime (kernel compile, buffers, graph dispatch) + CUDA graph executor (launches, cuBLAS, GPU memory); `metal/*.metal` compute kernels (dense, MoE, flash-attn, RoPE, KV, norm, softmax, quant); vendored llama.cpp mmq tier. |
| ROCm | `ds4_rocm.cu`, `ds4_rocm_compat.cu`, `ds4_rocm_unavailable.cu`, `rocm/*.cuh` | ROCm/HIP backend for Strix Halo. |
| GPU contracts | `ds4_gpu.h`, `ds4_gpu_copy.h`, `ds4_gpu_tp.h`, `ds4_gpu_mgpu.h` | Backend-agnostic graph/tensor/copy/TP/multi-GPU contracts. |
| State/streaming | `ds4_ssd.c/.h`, `ds4_layer_pack.c/.h`, `ds4_engram.c/.h`, `ds4_kvstore.c/.h` | SSD cache planning/mlock; layer placement; V4.1 Engram tables; disk KV checkpoint format. |
| Distributed | `ds4_distributed.c/.h`, `ds4_tp.c/.h` | Pipeline/distributed runtime; TP transport/lockstep. |
| Image/web/eval | `ds4_image.c/.h`, `ds4_web.c/.h`, `ds4_eval.c/.h` + `ds4_eval_cases.c/.h` | Image decode; web tools; eval runner/cases. |
| Help/args/prefix | `ds4_help.c/.h`, `ds4_gpu_args.c/.h`, `ds4_prompt_prefix.c/.h`, `ds4_tool_text.h` | Help; GPU args; prefix preload; tool text. |
| Memory/misc | `ds4_host_memory.h`, `ds4_linux_memory.h`, `ds4_rocm_memory.h`, `rax.c/.h`, `linenoise.c/.h` | Host/OS memory helpers; radix tree for tool-id replay; line editing. |
| Model-specific GPU | `ds4_deepseek41_gpu.h`, `ds4_deepseek41_cuda.cuh`, `ds4_qwen4_cuda.cuh`, `ds4_glm53_vision_gpu.cuh`, `ds4_deepseek4_vision_gpu.cuh` | V4.1/Qwen/GLM/vision kernel contracts. |
| Tests | `tests/` | Test runners/fixtures/vectors; `ds4_test.c`, `ds4_agent_test.c`, `test-vectors/`, model-specific/Python scripts. |
| Tools | `gguf-tools/`, `speed-bench/`, `dir-steering/`, `tools/`, `menubar/` | GGUF/quant/imatrix/quality scoring; benchmarks/charts (`speed-bench/promessi_sposi.txt`); activation steering; analysis scripts; menu-bar live status. |
| Data/misc | `misc/`, `third_party/`, `licenses/`, `gguf/`, `docs/` | Notes; vendored codecs; licenses; weights; guides. |
| Build/docs | `Makefile`, `download_model.sh`, `README.md`, `AGENTS.md`, `AGENT.md`, `CLAUDE.md`, `LOCAL_INVENTORY.md`, `CONTRIBUTING.md`, `QA_BEFORE_RELEASES.md`, `EVAL_DATA.md` | Build/test targets; model downloader; guides; fork ledger; release gate. |

## Tests

- `tests/ds4_test.c` — main C runner: server, graph correctness, logprob/local-golden vectors, long-context, tool-call quality, Metal kernels.
- `tests/ds4_agent_test.c` — agent unit tests.
- `tests/test-vectors/` — official/local continuation vectors per checkpoint (`flash-0731`, `flash-pre-0731`); fixtures must match the GGUF. `long_context_security_prompt.txt` is the long-context security fixture.
- Metal/CUDA/ROCm/model-specific test binaries and Python quality/vision scripts live under `tests/`; see `docs/TESTING.md`.

## Upstream picks & LOCAL_INVENTORY.md maintenance

This fork carries upstream (`antirez/ds4`) cherry-picks plus local work; `LOCAL_INVENTORY.md` is the ledger (A picks, B local commits, C dependencies, D verdicts). Keep it true:

- Record on creation: PR number, original/landed SHAs, adoption level (picked n/m), B/C rows; rejected PRs get an entry with rejection evidence. Cherry-pick commits carry `(cherry picked from commit <sha>)` (`git cherry-pick -x`) and `Upstream-PR: #NNNN`.
- After rebase: detect absorption with `git fetch origin` + `git cherry -v origin/main main` / `git patch-id`; delete absorbed entries (optionally leaving a one-line "merged upstream <sha>" note); refresh every SHA and re-validate C dependencies. Never rewrite `pr-*` / `up/pr/*` refs (patch-id sources).
- PR branches start from `origin/main` and carry only the target cherry-pick; don't merge local `main` or `git add -A`; `LOCAL_INVENTORY.md` is fork-private.
- `fork/main` (`williamxie1989/ds4`) is the off-machine backup; don't let it lag `main` by a whole work batch.

## Where to read next

- `README.md` — quick start, hardware/models, everyday use.
- `docs/METAL.md`, `docs/CUDA_MULTI_GPU.md`, `docs/DGX_SPARK.md`, `docs/STRIX_HALO.md` — build/run per platform.
- `docs/MODELS.md` — downloads/memory/vision; `docs/SSD_STREAMING.md` — larger-than-RAM; `docs/SPECULATIVE_DECODING.md` — DSpark/MTP; `docs/QWEN38_FLASH_NEXT.md` — Qwen setup/MTP/vision.
- `docs/SERVER.md` — APIs/sessions/images/disk KV; `docs/CLIENTS.md` — Pi/OpenCode/Codex CLI/Claude Code; `docs/DISTRIBUTED.md` — TP/pipeline.
- `docs/TESTING.md` — focused checks/output investigation; `CONTRIBUTING.md` — regression tracks; `QA_BEFORE_RELEASES.md` — release gate; `docs/PERFORMANCE.md` — benchmarks.
- `AGENT.md` — upstream goals/quality/safety notes (layout/testing predate newer fork backends); `LOCAL_INVENTORY.md` — fork ledger.
- `dir-steering/README.md`, `gguf-tools/README.md`, `speed-bench/README.md`, `tests/test-vectors/README.md`.

## Current work

In-flight workstreams have plan/handoff docs at the repo root (e.g. `ITERATION_PLAN_2026Q4.md`, `M1_HANDOFF.md`, `V41_*`, `HANDOFF-*`). Read the relevant one before continuing; they record decisions/open items, not rules.
