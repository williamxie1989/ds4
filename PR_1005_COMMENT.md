<!-- Draft comment for antirez/ds4#1005. Post AFTER our PR exists; fill [[PR]]. -->

@aaa2015 heads-up from someone building the same thing from the other end — I
hit the wall your gate runs into, and I have measurements now.

`ds4_engine_can_rewind()` admits GLM-5.3 to the rewind tier and the fork case
relaxes the target to `common`. On GLM-5.3 that target is one the rewind refuses:

```c
if (s->checkpoint_valid && ds4_session_is_glm(s)) {
    state_ok = !s->glm_graph.glm53 || ds4_session_glm_mtp_rewind(s, pos);
```

so for a GLM-5.3 session `state_ok` needs the MTP rollback to take, which it only
does inside the two-token speculative window. Below the frontier there is nothing
to roll the recurrent KDA state back to, `state_ok` comes out false, the
checkpoint is invalidated, and the request rebuilds from token zero — the exact
outcome the tier exists to avoid. Your second comment already flags that the
capability argument was read off `ds4_session_rewind()` rather than measured, so
this is probably not news; it is just the part that decides whether the tier ever
fires on GLM-5.3, and I could not make it fire mid-divergence on a real run.

What makes it fire is a restore path. I landed one locally: snapshots of the KDA
conv/recurrent state plus the pooled indexer tail, taken every `N` prefill tokens
into a small per-session ring, and a `ds4_session_rewind_floor()` that answers
which position is actually restorable — so the probe names a target the rewind
will accept instead of one it will refuse. On a replayed 210-step agent session
(Apple M5 Max, Metal, `GLM-5.3-Flash-Q2`, 256k ctx):

- with the ring off, 13 of the first 50 requests prefilled from zero, worst one
  91,617 tokens / 208 s; with it, 0 of 50, and 59 snapshot rewinds over the whole
  run (29.9M prompt tokens, 594k re-prefilled)
- greedy continuation of a 30,586-token prompt is byte-identical between cold
  prefill and snapshot rewind: 56.8 s vs 10.6 s

Sending that as [[PR]] (mechanism, plus a probe tier that asks the core first).
Nothing in it conflicts with what you did: the tier can call
`ds4_engine_can_rewind()` in place of its GLM-5.3 test the moment your gate
lands, and your `common` relaxation for forks is orthogonal to where the target
has to sit. If you would rather fold the restore path into this branch instead,
say the word and I will re-target it at your branch.

One thing I found while checking the restore path against TP, worth knowing for
your change too: `ds4_session_rewind()` tells the workers via
`ds4_tp_send_rewind()`, so any snapshot restore that skips that broadcast leaves
the leader rewound and the workers holding the positions above it — GLM-5.3
prefill runs in the shared chunked loop in `ds4_session_sync_internal()`, which
has no TP branch, so TP sessions do take snapshots unless something declines on
their behalf. [[PR]] declines them.
