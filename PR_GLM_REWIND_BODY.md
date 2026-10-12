# PR body — GLM-5.3 live KV rewind (antirez/ds4)
# Branch: pr-glm-live-rewind (cut from origin/main fc80bd6), 2 commits.
# Placeholders marked with [[ ]] are filled from the last measurements.

## Summary

- Snapshot the GLM-5.3 recurrent KDA state and pooled indexer tail on a token
  stride during prefill, into a small per-session ring, and restore from it.
- Rewind the live KV to the newest snapshot at or below the shared prefix when a
  prompt diverges below the frontier, instead of rebuilding the whole prompt.
- Ask the core which position is actually restorable before naming a rewind
  target, so the probe never promises a rewind the engine will refuse and then
  rebuild anyway.
- Document the two knobs (`DS4_GLM_REWIND_SLOTS`, `DS4_GLM_REWIND_STRIDE`) and
  their memory cost in `docs/SERVER.md`.

## Why

GLM-5.3 carries 34 layers of recurrent linear-attention state plus a pooled
indexer tail. None of it can be truncated: to put a session back at an earlier
position you have to have saved what was live there. So a request whose prompt
shares a long token prefix with the live session and then differs -- a client
that compacted its history, pruned tool results, or edited a message -- discards
the whole context and prefills from token zero.

The existing GLM rewind tier could not help. It accepted only a *strict
truncation* of the checkpoint, and for that it named `prompt_len - 1`, a
position GLM-5.3 cannot restore unless it happens to fall inside the two-token
speculative window. The probe claimed `memory-rewind`, the rewind refused, and
the request rebuilt from zero regardless -- and a prompt that *diverged* below
the frontier had no tier at all. In one recorded agent session a 149k-token
request spent 329 s of prefill to answer in two.

The gate was never the blocker: `ds4_engine_is_glm_dsa()` tests the shape
*family*, and GLM-5.3's shape is `{.family = GLM_DSA, .variant = GLM53}`, so the
tier was already reachable there -- which is why this branch *carves* GLM-5.3 out
of that tier (`&& !ds4_engine_is_glm53()`) instead of admitting it, and gives it a
tier that asks the core where it can actually land.

## Behavior

The new tier sits last in the probe ladder, after every tier that needs no
recompute, so a visible-transcript or rendered-text prefix still wins. It fires
only for GLM-5.3, only when the shared prefix is non-empty and strictly below
the frontier, and only when the core reports a restorable position. The target
is always strictly below `prompt_len`, so the final prompt token is re-read and
logits are fresh -- the same contract as the other tiers that recompute.

When the ring holds nothing for a strict truncation, the tier names
`prompt_len - 1` and the ordinary rewind path keeps the two-token rollback it
already supported: pre-existing behavior, unchanged. The older GLM frontier
rewind keeps the truncation tier untouched. A session whose checkpoint holds
images is never rewound -- replaying those positions would feed image rows to
the text path.

## Memory

| knob | default | effect |
| --- | --- | --- |
| `DS4_GLM_REWIND_SLOTS` | 8 | snapshots kept; `0` turns the ring off |
| `DS4_GLM_REWIND_STRIDE` | 8192 | tokens between two snapshots |

One slot is 145.6 MiB at the GLM-5.3 shape, so the defaults cost about 1.1 GiB
per session and make the last ~65k positions rewritable: a divergence re-reads
at most one stride plus the new tail instead of the whole prompt. The ring is
allocated on the first snapshot and included in the session memory estimate, so
the existing host-memory guard accounts for it -- same box, same benchmark
session, the planned-memory line moves by exactly the ring:

```
before: buffers 3.04 GiB + ... = 93.64 GiB planned
after:  buffers 4.18 GiB + ... = 94.78 GiB planned     (+1.14 GiB = 8 x 145.6 MiB)
```

Swap stayed flat across a
210-step replay at 256k context.

## Correctness evidence

Machine: Apple M5 Max, 128 GB, Metal backend, `GLM-5.3-Flash-Q2`,
`-c 256000`.

The invariant and the log line below are from **this branch's own build**
(`636bba7`). The replay-scale numbers come from the local tree this branch was
extracted from -- same ring, same floor, same two guards; the probe tier is the
only part rewritten to upstream's shape, and that part is covered by the unit
tests under Tests.

**The invariant: a rewound session must produce what a cold prefill would.**
A 30,586-token prompt at `temperature 0`, 48 tokens generated:

| path | prefill | result |
| --- | --- | --- |
| cold prefill from zero | 86.6 s | sha1 `217778c256573586` |
| snapshot rewind to 26,624 (ratio 0.87) | 14.4 s | sha1 `217778c256573586` |

Byte-identical continuation, 6.0x faster. The intervening probe request -- same
prefix, one token longer, different tail -- rewound from the same slot to the
same position and continued differently, as a different prompt must.

**Replay of one recorded 210-step agent session** (a real DSH session,
`tools/prefix_reuse_replay.py --replay --apply-prunes`, same file, same
harness; only the environment differs). Over the 49 replayed requests after the
first cold turn:

| | defaults (8 x 8192) | `DS4_GLM_REWIND_SLOTS=0` |
| --- | --- | --- |
| requests rebuilt from zero | 0 | 13 |
| wall clock of those requests | 258 s | 2,054 s |
| prompt tokens re-prefilled | 105,686 | 935,092 |
| worst single request | 6.1 s | 91,617 tokens, 208.3 s |

That last row is one and the same request on both sides -- replayed request 49.
Without a snapshot it prefilled 91,617 tokens from zero in 208 s; with one it
reused 97.4% of its prefix and answered in 6.1 s.

Over the full 210 steps with defaults: 30,013,791 prompt tokens of which 625,684
were re-prefilled (aggregate reuse 0.9792), across 59 snapshot rewinds that
skipped 249,775 tokens -- about 454 s of cold prefill at the ~550 tok/s this box
sustains. Only two requests rebuilt from zero: the first turn, and the one deep
history edit outside the ring window (below).

What a rewind looks like -- the log line and the trace field, straight from this
branch's server:

```
ds4-server: rewound GLM live prefix from 30634 to 26624; final prompt token will be reevaluated

  cache_source: memory-rewind
```

## Speed

`ds4-bench`, same box, same model, same prompt file, before (`fc80bd6`) vs after.
The measurements are **paired**: short sweeps (`--ctx-start 8192 --ctx-max 24576
--step-incr 4096 --gen-tokens 64`) alternating the two builds in ABBA order, four
rounds, giving 20 matched (round, frontier) samples taken about two minutes
apart.

| metric | mean delta | 95% CI | median | per-pair scatter (90th pct) |
| --- | --- | --- | --- | --- |
| `prefill_tps` | -0.08% | [-0.71%, +0.56%] | -0.04% | 2.5% |
| `gen_tps` | -0.19% | [-0.64%, +0.26%] | -0.29% | 2.0% |

No measurable regression: both intervals straddle zero and sit well inside the
scatter of individual pairs. That is the expected shape -- the only work added
inside prefill is one 145.6 MiB copy every `DS4_GLM_REWIND_STRIDE` (8192) tokens,
~18 KB/token amortized, and the decode path is untouched.

Why paired rather than one before/after CSV: two full 2048..65536 sweeps taken 40
minutes apart (baseline first) showed -15% prefill and -7% generation for this
build, with the loss concentrated in small clock-bound contexts and gone by 63k
where the work is bandwidth-bound. Minutes apart, the same comparison is -0.04%.
Run order dominates an effect of this size on a passively cooled box, so a
single before/after pair is not evidence in either direction.

## Known limits

- The ring covers the last `SLOTS x STRIDE` positions (~65k by default). One
  request in the recorded session -- a prune that dropped almost everything,
  leaving a common prefix of only ~20-35k of 149,173 tokens -- is beyond any
  ring setting, and rebuilt (328.9 s). Measured alternative: raising
  `DS4_GLM_REWIND_STRIDE` to 16384 (same memory) did reach it, landing on the
  oldest slot and re-reading 87% of the prompt for a 25 s win (303.4 s), while
  every ordinary rewind paid up to a 16k re-read -- over 126 requests, 17% more
  wall clock and 15% more re-prefilled tokens. So the defaults stay, and depth
  is worth buying with `DS4_GLM_REWIND_SLOTS` rather than stride.
- Snapshots are taken while reading prompts, not while generating, so a
  divergence inside freshly generated tokens is still served only by the
  speculative window.
- Tensor-parallel sessions take no snapshots. `ds4_session_rewind()` keeps ranks
  in step by sending the rewind to the workers; a snapshot restore has no such
  message, and restoring on the leader alone would leave them holding the
  positions above it. A TP session therefore reports no reachable position and
  keeps its current path.

## Tests

- `make -j8` -- clean, 0 warnings
- `make cpu` -- clean, 0 warnings (the CPU reference build has no graph state,
  so the two entry points answer "nothing reachable" and callers keep their
  ordinary rebuild)
- `./ds4_test --server` -- `server: OK`, including the new
  `test_slot_probe_glm_rewind_tiers`: divergence, divergence above the newest
  snapshot, strict truncation, ring empty, exact extension still wins, and an
  image-bearing checkpoint refused
- `./tests/test_session_state` and `./tests/test_session_state_gpu` -- the GPU
  one also proves that a restore from a non-zero ring slot reproduces the saved
  KDA conv and recurrent state exactly, and that an offset past the ring is
  refused instead of clamped
- `git diff --check`
- model-backed: the invariant run and the two replays above

## Related

- #1005 generalizes the probe's capability gate (`ds4_engine_can_rewind()`) and
  relaxes the fork target to `common`. That is the right shape for the probe,
  and this tier can call it directly if that lands. It does not by itself make
  GLM-5.3 rewindable mid-divergence: `ds4_session_rewind()` for GLM-5.3 succeeds
  only inside the two-token speculative window, so a `common` target below the
  frontier is refused and the request rebuilds anyway -- which is the failure
  `glm_live_rewind_target()` avoids by asking the core first. This PR supplies
  the restore mechanism the gate assumes.
- #1093 and #1089 attack the same rebuild from the checkpoint side; this one
  keeps the live KV usable.
