## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

### Metal decode schedule A/B

Build the balanced, same-engine Metal decode comparison with:

```
make metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench \
  -m ds4flash.gguf \
  --include-selection
```

The harness prefills two sessions and alternates both variant order and
variant-to-session assignment. It aborts unless every full-vocabulary logit
row is bit-identical and, with `--include-selection`, both variants select the
same non-EOS token. Use `--candidate-env NAME` to measure a rollback control,
or `--help` to compare explicit split schedules.

Repeat `--candidate-env NAME` (up to eight distinct names) to compare several
decode changes enabled versus all of them disabled in one paired run; the
candidate sets every named variable to `1`. These are dispatch-time controls,
not options cached when the engine opens. An unused name provides an A/A control.
Repeat measurements with `--reverse-order` as well. This inverts the initial
variant order and session pairing, so periodic compressor work is not always
measured with the same variant first.

The M5 resident Q2 sum6 decode specialization fixes selected projection
dimensions and strides while preserving the original two-SIMDgroup
threadgroups and floating point accumulation order. Unmatched shapes, SSD
streaming, TP, quality mode and non-M5 devices retain the generic selection.
To compare against its rollback:

```
make test-metal-q2-decode metal-decode-schedule-bench
./speed-bench/metal_decode_schedule_bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt --prefix-tokens 2048 --ctx 4096 \
  --warmup 32 --tokens 768 --control-first 4 --control-second 0 \
  --candidate-env DS4_METAL_DISABLE_M5_Q2_SUM6_TUNING --include-selection
# Repeat with --reverse-order. Control is specialized; candidate is generic.
```

On M5 Max 128 GB, macOS 26.6.2 high power mode, Flash Q2 with Q8 attention,
shared and output weights, this comparison measured +0.17% and +0.22% decode
throughput in opposite orders. All logits and selected tokens matched exactly;
the gain is small, not a several-percent end-to-end improvement.

The model-free test compares all consumed mid/output values bitwise, with
poisoned outputs, guards, repeated/edge expert IDs, and tail/addend fallback
cases. Its `--reference-only` mode keeps rollback enabled and prints an output
checksum for cross-source comparisons.

The resident M5 decode shape templates now cover three six-route Q2_K down
projections (V4 Flash, V4.1 Flash and V4 PRO), the V4 Flash IQ2_XXS pack2
gate/up pair, and the eight-route GLM 5.3 Flash IQ2_XXS pair and per-expert
Q2_K down projection. Shape and tensor
quantization are checked separately. Each template retains a generic fallback;
SSD streaming, TP, batched decode and unmatched strides do not use these fixed
routes. Qwen's ten-route MoE has a separate implementation.

```sh
make test-metal-q2-decode
MTL_DEBUG_LAYER=1 ./tests/test_metal_q2_decode_exact --full-experts
MTL_DEBUG_LAYER=1 ./tests/test_metal_q2_decode_exact --profile v41
MTL_DEBUG_LAYER=1 ./tests/test_metal_q2_decode_exact --profile pro
MTL_DEBUG_LAYER=1 ./tests/test_metal_q2_decode_exact --profile glm53
```

The full-expert case compares the Flash IQ2 and Q2 changes independently. The
V4.1/PRO cases use synthetic eight-expert tensors with the real projection
dimensions; GLM uses eight selected IDs from 288 synthetic experts. Only V4
Flash and GLM 5.3 Flash have local model-backed throughput measurements. To
compare the new IQ2 choices with a same-engine logit-exact rollback, pass
`--candidate-env DS4_METAL_DISABLE_M5_IQ2_PACK2_SHAPE` for V4 Flash or
`--candidate-env DS4_METAL_DISABLE_M5_IQ2_GLM53_SHAPE` for GLM 5.3 Flash to
`metal_decode_schedule_bench`, then repeat with `--reverse-order`. The Q2
rollback above still applies to the three six-route Q2 shapes. The GLM
eight-route per-expert Q2 down kernel has its own independent rollback,
`DS4_METAL_DISABLE_M5_GLM53_Q2_DOWN_SHAPE`, because it preserves eight
separate down rows and the following sum8, rather than using sum6.

On M5 Max 128 GB in high-power mode, 768-step V4 Flash decode A/B measured
about 0.58-0.66% higher throughput with the shaped IQ2 pair in opposite
orders; 512-step GLM 5.3 Flash measured about 0.62-0.65%. Both comparisons
checked full-vocabulary logits and selected tokens exactly. Isolated Q2
dispatch timings improved for V4.1/PRO shapes, but without local model weights
there is no V4.1/PRO end-to-end claim. A synthetic isolated Flash IQ2 test
regressed despite its real-model improvement; cache and concurrent-shared-FFN
conditions matter, so do not infer throughput from isolated timings alone.
With GLM's IQ2 shape held enabled, 512-step GLM decode gained 0.39-0.47%
from its separate per-expert Q2 down shape; all compared outputs were exact.

The GLM 5.3 KDA-output dense Q8_0 shape is separate from the MoE enum: it
fixes K8192, output4096 and the 8704-byte row stride without changing Q8 dot
or reduction order. `DS4_METAL_DISABLE_M5_GLM53_KDA_Q8_SHAPE` restores the
generic dense kernel; mismatched shapes, TP, SSD and quality mode also use it.
`make test-metal-q8-decode-shape` checks bit-exact output and nearby fallbacks.
On M5 Max, the retested direct kernel saved 2.0-2.6% of GPU span for hot
weights and 1.2-1.3% for 16 rotating matrices. Its individually toggled GLM
model decode benefit was about 0.04-0.08%, smaller than the A/A bias in some
short runs. This is a reproducible isolated improvement with a small model
contribution, not a confidently measured standalone throughput gain.

Combined 512-step GLM and 768-step Flash resident Q2 decode tests on the M5
Max (128 GiB, high-power mode) use one engine, alternating session/order and
the same 4/0 split schedule for both variants. Control has all listed shapes
enabled; candidate rolls back all listed shapes. Every full-vocabulary logit
row and selected token is bit-exact. Prefill is shared and untimed.

| Model / rolled-back shapes | Normal control/candidate t/s | Reversed control/candidate t/s | Combined shaped benefit |
| --- | ---: | ---: | ---: |
| V4 Flash: IQ2 pack2 + Q2 sum6 | 43.3406 / 42.9706 | 41.0412 / 40.6088 | 0.86% / 1.06% |
| GLM 5.3 Flash: IQ2 pair + Q2 down + KDA Q8 | 31.3639 / 30.9077 | 31.2736 / 30.8494 | 1.48% / 1.38% |

Contemporaneous A/A controls showed apparent shaped-over-generic offsets of
+0.23%/-0.05% for Flash and +0.01%/-0.15% for GLM in the same order pair.
Results are per-model and cannot be combined across models or generalized to
other contexts. The GLM KDA Q8 result was reconstructed and retested after its
earlier variant was removed; these numbers refer to the current implementation.

To compare the default pre-M5 ratio-4 compressor pack/transpose fusion with the
legacy decode path, including token selection, use:

```
./speed-bench/metal_decode_schedule_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_COMPRESSOR_RATIO4_DECODE_PACK_FUSION \
  --include-selection \
  --tokens 1024
```

### Metal prefill variant A/B

Build the balanced prefill comparison. To compare the default resident pre-M5
MXFP4 pair tail-SIMDgroup cull against the original pair kernel, make the
rollback path the candidate:

```
make metal-prefill-variant-bench
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_PAIR_TAIL_SIMDGROUP_CULL
```

To isolate the default routed-down tail-SIMDgroup cull from the retained pair
default, use its down-specific rollback as the candidate:

```
./speed-bench/metal_prefill_variant_bench \
  --candidate-env DS4_METAL_DISABLE_PRE_M5_MXFP4_MOE_MM_ID_DOWN_TAIL_SIMDGROUP_CULL
```

The harness uses one Metal engine and fresh sessions for every run. It warms
both variants with at least 32 tokens, alternates control/candidate order in
ABBA and BAAB blocks, poisons host logit buffers before copying, and aborts
unless every final full-vocabulary logit row is bit-identical. Defaults are an
8192-token prefix, an automatically sized 8193-token context, and two repeats;
use `--help` to override them.

Numeric runtime tuning controls can use `--candidate-value TEXT` (default `1`), and
`--prefill-chunk N` selects the same chunk size for both variants (default
4096). For GLM, pass `--prefill-chunk 0` to let the model choose its own chunk
size; GLM rejects an explicit chunk size. `--interleave` requires a nonzero
chunk size, so use the default ABBA/BAAB mode for GLM. The control unsets the
named variable. Use only controls read at
dispatch time: this harness keeps one engine alive and cannot compare settings
cached during initialization.

`DS4_QWEN4_TIMING=2` reports each prefill chunk's position, token count, and
stage times, including the attention mixer separately from GDN/attention.
These measurements include synchronization overhead and are for attribution;
use uninstrumented alternating runs for throughput claims.
`DS4_QWEN4_MOE_PROFILE=1` further splits tiled MoE into routing, expert-list
construction, routed/shared gate-up, routed/shared down, and reduction,
reporting one line per layer. Its synchronization also makes it diagnostic only.

Qwen routed MoE quantization specialization is enabled by default on M3 Ultra.
`DS4_QWEN4_MOE_MM_SPECIALIZE=0` restores the generic kernels; `=1` opts in on
other devices. To compare the final default against rollback, pass
`--candidate-env DS4_QWEN4_MOE_MM_SPECIALIZE --candidate-value 0` to the harness.

### Concurrency: many requests at once

`session_concurrency_bench` measures the engine without HTTP overhead.

```sh
make session-concurrency-bench
./speed-bench/session_concurrency_bench -m ds4flash.gguf \
  --concurrency 1,2,4,8 --ctx 4096 --gen 128 --budget-gib 28 \
  --csv /tmp/concurrency.csv
```

Streams use different corpus offsets to avoid identical expert routing.
The report includes aggregate and per-stream throughput, step latency and
prefill time. Set `--budget-gib` to bound session allocations; monitor actual
memory and swap too. `--mixed` adds a concurrent continued prefill, `--spec`
enables Qwen MTP, and `--verify` fails on a greedy-token mismatch against
serial decoding. Use the official-continuation scorer to assess quality when
different reduction orders change a close token choice.

For an interleaved comparison against a diagnostic rollback:

```sh
./speed-bench/session_concurrency_bench --ctx 4096 --concurrency 8 \
  --candidate-env DS4_QWEN4_SESSION_BATCH --candidate-value 0 --repeat 3
```

The control unsets the variable; the candidate sets it in alternating ABBA
blocks. Use only controls read at dispatch time. Keep prompts, context, memory
pressure and thermal conditions comparable, and retain generated text when
measuring speculation. `concurrency_sweep.sh` runs each cell in a fresh process.

`serve_concurrency_bench.py` measures the HTTP server, including queueing,
prompt rendering, prefix reuse and streaming:

```sh
python3 speed-bench/serve_concurrency_bench.py \
  --concurrency 8 --prompt-tokens 4096 --max-tokens 128 --requests 32 \
  --ignore-eos --json /tmp/serve-8.json
```

Start the server separately with an appropriate model, context and
`--batched-session 8`. The report includes first-token and inter-token
latency, request latency and throughput. Prompts use fresh nonces;
`--shared-prefix` tests cache reuse instead.
