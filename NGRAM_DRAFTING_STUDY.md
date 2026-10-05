# Draftless n-gram speculative decoding — feasibility study (2026-10-05)

## Outcome (2026-10-06): CLOSED — measured, not viable

The decisive measurement was made on a window. **The verify cost slope is
a ≈ 0.68, not ≤ 0.30**, so the payoff is **1.000×**: the line is dead, and so is
every other speculative scheme on this engine.

Measured on `gguf/DeepSeek-V4.1-Flash-Q2.gguf`, SSD-streaming, ctx 4096,
prefix 1024 (`tests/test_deepseek41_dspark.c --verify-scan-ssd`, four runs):

| expert cache | D, one decode step | V(8)/D | a(8) |
|---|---|---|---|
| 958 experts | 61.28 ms | 5.779 | 0.683 |
| 6028 experts | 42.95 ms (= **23.3 t/s**, production parity) | 5.767 | 0.681 |
| 8078 experts | 41.68 ms | 5.928 | 0.704 |
| 6028 experts, all 8 rows the *same* token | 43.02 ms | 5.663 | 0.666 |

`V(n) ≈ D·(1 + 0.68·(n−1))`. Three things make this conclusive:

- **It is not a caching artifact.** A 8.4× change in expert-cache size moves
  `V(n)/D` by 0.2%, while `D` itself moves 47%. The cost is per row, not per
  weight miss.
- **It is not expert-union growth.** Verifying the *same* token eight times —
  the minimum possible expert union — gives the same slope (0.666).
- **The row cost is irreducible.** A verify of 2 rows costs **2.36 decode
  steps**, i.e. *worse* than two plain decode steps. Batching only starts to
  amortise at n=8, and even there one row costs 0.72 of a step.

Consequences, both recorded in `LOCAL_INVENTORY.md` §D:

- **Payoff of the best possible n-gram policy: 1.0001×–1.0020×.** Optimising
  per-match-length caps against the *measured* cost curve (`tools/ngram_cost_model.py`)
  collapses the policy to speculating on 0.8% of steps with 1–6 drafts, for no
  gain. This holds with the study's own optimistic pooled index too.
- **Structural ceiling: 1.387×.** A perfect oracle drafter — seven correct
  drafts on *every* step, the most an 8-row verify can check
  (`DS4_TP_BATCH_MAX_ROWS`) — yields 8/(1+0.68·7) = 1.387×. No speculative
  scheme on this engine can beat that, which is why MTP/DSpark, this line, and
  any future variant all fail for the same reason. **Do not reopen speculative
  decoding for V4.1-Q2 streaming without first re-measuring `a`.**

Two corrections to the analysis below, found while making the decision
(`tools/ngram_cost_model.py` is the corrected model):

1. The "selective policy" table divided *conditional* tokens by *conditional*
   cost, dropping the ~90% of steps that never speculate. A policy that
   speculates on 9.8% of steps cannot exceed 1.19× at any verify cost. The
   1.70×/1.36× figures are per-speculated-step, not end-to-end.
2. The cost model charged for *accepted* rows (`rows = 1 + min(acc, d)`) rather
   than the rows actually submitted. Rejected rows cost the same as accepted
   ones. The model's own docstring already said `c(n) = 1 + a·(n−1)`; the code
   did not.

The offline method below still stands and its numbers reproduce; only the
end-to-end arithmetic and the verdict were wrong. Two measurement notes worth
keeping: the study's index pooled all 14 sessions into one lookup table
(cross-session matches a deployment cannot have — the per-session figure is
1.514 tokens/step, not 1.653), and longer matches are far richer than a k=4
index can see (a 16-gram match accepts 7.8 tokens on average, per-session).

---

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

> **Corrected 2026-10-06 — see Outcome above.** The model below is stated
> correctly, but the tables that follow were computed with `rows = 1 + min(acc, d)`
> (accepted rows) instead of the submitted rows, and the selective table divided
> conditional tokens by conditional cost. Both inflate the result. Use
> `tools/ngram_cost_model.py` for the corrected numbers.

`c(n) = 1 + a·n` decode-step units for a verify that commits n rows; steps that
find no draft cost exactly 1 (they are unchanged). `a` is the marginal cost of one
extra verified row and is **not yet measured on this engine** — the three columns
above bracket it. For reference, the MTP line measured its own verify at ≈3
decode steps for 2–3 rows, which would put `a ≈ 0.7`; that shape is heavier than a
n-gram verify needs to be (no draft model, no second expert union), so the useful
range is likely a = 0.15–0.40.

**Measured 2026-10-06: a ≈ 0.68.** The MTP line's ≈0.7 was not a DSpark artifact —
it is what a verified row costs on this engine, full stop.

## Verdict

> **Superseded 2026-10-06.** The bullets below were written before `a` was
> measured. With a ≈ 0.68 the second bullet is false: the payoff is 1.000×.

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

## Decisive measurement (made 2026-10-06, window)

Measure `a`: wall-clock of an n-row verify on the real engine, n ∈ {2,4,8,12},
against a 1-row decode step. **Done — `a ≈ 0.68`, so the a ≤ 0.30 gate fails and
the line is closed.** The instrument is
`tests/test_deepseek41_dspark.c --verify-scan[-ssd]`; it scans
n ∈ {2,3,4,5,6,7,8} (8 is `DS4_TP_BATCH_MAX_ROWS`, the hard row ceiling), reports
median/min/max over shifted token windows, and times the host argmax and commit
separately from the verify. Reproduce with:

```sh
make tests/test_deepseek41_dspark
DS4_TEST_STREAMING_CACHE_GIB=63 DS4_TEST_STREAMING_CACHE_EXPERTS=6000 \
  ./tests/test_deepseek41_dspark gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --verify-scan-ssd tests/long_context_security_prompt.txt
```

`DS4_TEST_VERIFY_WINDOW=same` verifies one token eight times (expert-union floor).
Pre-flight memory first: 6028 experts plans 74.47 GiB, 8078 plans 93.47 GiB, and
the 9000-expert production setting needs ≈90.6 GiB of budget on a 128 GiB host.

## Reproduction

```sh
# offline acceptance model (no model load)
python3 tools/ngram_accept_sim.py    --sessions 14 --max-chars 350000 --tag big --segments
python3 tools/ngram_accept_policy.py --cache /tmp/ds4_ngram/stream_big.json   # superseded
python3 tools/ngram_cost_model.py    --cache /tmp/ds4_ngram/seg_big.json      # corrected
```

Raw sweep: `/tmp/ds4_ngram/sweep_big.json`; original report:
`/tmp/ds4_ngram/policy.md` (carries the two errors above — do not cite it);
corrected report: `/tmp/ds4_ngram/cost_model.md`.

`--segments` writes `seg_<tag>.json` with session boundaries preserved, which is
what `ngram_cost_model.py` needs for the deployment-realistic per-session index.
Both caches are in `/tmp` and do not survive a reboot; rebuilding them is ~20 s
and reads only GGUF metadata (RSS ≈ 600 MB, no weights) — still do not run it
inside a model-load window.
