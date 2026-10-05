# Draftless n-gram speculative decoding — feasibility study (2026-10-05)

**Question.** Decode is the pain point (prefill is 180–400 t/s, decode 22–23.5 t/s).
GPU is only ~30% busy during decode, so the cost is per-step host round-trips, not
bandwidth. The one lever that amortises round-trips without touching the bit
contract is speculative decoding — and the MTP/DSpark route was already ruled out
(verify break-even needed `avg_accept ≈ 2.96`, measured 1.33–1.91). Draftless
n-gram lookup changes the economics because the draft costs nothing; what remains
is whether *our* workload offers enough repetition for a verify step to confirm
several tokens.

This study measures that on real transcripts. **No model was loaded** — the
tokenizer comes from GGUF metadata, the workload from recorded sessions.

## Method

- **Corpus**: 14 largest recorded DSH sessions (`~/.dsh/sessions/*/session-*/`),
  1,476,726 tokens total, of which **359,778 (24.4%) are model-generated**
  (`assistant/message` records); the rest is context (user turns, tool results)
  that the lookup may match against but never generates. 23.8% of generated
  tokens are tool-call JSON arguments.
- **Provenance**: the transcripts carry `model: deepseek-v4-flash`, i.e. our own
  engine produced them.
- **Tokenizer fidelity**: vocab (129,280) + merges (127,741) read straight from
  `gguf/DeepSeek-V4.1-Flash-Q2.gguf` metadata; byte-level BPE rebuilt with
  `tokenizers`, pre-split rules mirrored from `ds4.c::bpe_tokenize_text`
  (`tokenizer.ggml.pre = joyai-llm`).
- **Simulation**: greedy replay with an incremental most-recent-occurrence n-gram
  index (llama.cpp `ngram-simple` shape). At each generated position, look up the
  last k tokens among positions that end **strictly before** it, draft what
  followed the most recent match, accept the longest matching prefix, advance by
  `1 + accepted` (the +1 is the model's own token at the first mismatch).
  A leakage sentinel guards the off-by-one that would otherwise make the draft
  equal the real future (its value settles at ~0.33, the honest first-token hit
  rate, and drafts were eyeballed against real continuations).

## Results (k=4, draft cap 12 unless noted)

Survival curve `S(j) = P(acc ≥ j)` over 218,575 verify steps:

| j | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|
| S(j) | .203 | .101 | .065 | .047 | .036 | .030 | .021 | .013 |

**Acceptance depends sharply on match length** — this is the exploitable signal:

| longest match | steps | share | mean accepted |
|---|---|---|---|
| 4-gram | 21,457 | 9.8% | **2.86** |
| 3-gram | 24,284 | 11.1% | 0.82 |
| 2-gram | 88,124 | 40.3% | 0.68 |

Unconditional speculation, by draft cap:

| draft cap d | tokens/step | speedup a=.15 | a=.25 | a=.40 |
|---|---|---|---|---|
| 4 | 1.416 | 1.23x | 1.13x | 1.00x |
| 8 | 1.528 | 1.30x | 1.19x | 1.05x |
| 12 | 1.596 | **1.35x** | **1.23x** | 1.08x |

**Selective policy — speculate only when the longest match is ≥ m** (draft cap 8):

| m | steps speculated | tokens/step | speedup a=.25 | a=.40 |
|---|---|---|---|---|
| 2 | 61.2% | 1.862 | 1.27x | 1.07x |
| 3 | 20.9% | 2.312 | 1.47x | 1.20x |
| 4 | 9.8% | **2.969** | **1.70x** | **1.36x** |

By content type (draft cap 8): **tool-call JSON 2.123 tokens/step (1.48x / 1.25x)**,
prose 1.417 (1.13x / 1.00x). k ∈ {2,3,4,6} barely matters — the short-match
fallback dominates, so the policy should be driven by match length, not by k.

## Cost model

`c(n) = 1 + a·n` decode-step units for a verify that commits n rows; steps that
find no draft cost exactly 1 (they are unchanged). `a` is the marginal cost of one
extra verified row and is **not yet measured on this engine** — the three columns
above bracket it. For reference, the MTP line measured its own verify at ≈3
decode steps for 2–3 rows, which would put `a ≈ 0.7`; that shape is heavier than a
n-gram verify needs to be (no draft model, no second expert union), so the useful
range is likely a = 0.15–0.40.

## Verdict

- Unconditional n-gram drafting is **not** worth it (≈1.0–1.1x at plausible cost).
- **Match-gated drafting is**: speculate only when a 4-gram match exists →
  **1.36–1.70x**, i.e. 30–35 t/s decode on the V4.1 Q2 streaming path, with
  bit-exactness preserved by construction (verify + commit, same kernels).
- It composes with the capacity lever (+6–8% from 9000 experts) and needs no
  change to the numerical contract: rejected drafts simply roll back, and the
  committed stream is the greedy stream.
- Caveat: recorded sampling parameters are not in the transcripts. If those runs
  sampled with temperature > 0, the recorded continuation is a sample rather than
  the argmax, so these acceptance numbers are a **lower bound**.

## Decisive next measurement (needs a window)

Measure `a`: wall-clock of an n-row verify on the real engine
(`ds4_session_eval_speculative*` path or a small bench), n ∈ {2,4,8,12}, against a
1-row decode step. If `a ≤ 0.30`, build it: expected **+20–40% decode** on agent
traffic for roughly a day of work (draft source + match gate + wiring into the
existing verify path; default-off env).

## Reproduction

```sh
python3 tools/ngram_accept_sim.py    --sessions 14 --max-chars 350000 --tag big --sweep
python3 tools/ngram_accept_policy.py --cache /tmp/ds4_ngram/stream_big.json
```

Raw sweep: `/tmp/ds4_ngram/sweep_big.json`; report: `/tmp/ds4_ngram/policy.md`.
