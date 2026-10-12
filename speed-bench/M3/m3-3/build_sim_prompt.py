#!/usr/bin/env python3
"""M3-3 multi-turn append simulator prompt builder (R2a speed gate).

Builds a fixed-content "6 rounds of tool-type agent traffic" prompt: the same
tool-call/answer block repeated ROUNDS times, so turn k's append reproduces
turn k-1's expert routing (cross-turn reuse) — the domain the protect bit
targets.  Written to speed-bench/M3/m3-3/sim_prompt.txt.

ds4-bench appends step-incr rows per frontier from a single fixed token
sequence, so "rounds" maps to frontiers: the simulator picks step-incr so the
prompt covers exactly ROUNDS appends inside [start, max].
"""
import argparse

BLOCK = (
    "Tool call: read_file(path=\"src/router/hotlist.c\") -> "
    "ok bytes=4178 lines=112 sha=f3a91c\n"
    "Tool result: static uint32_t hotlist_slots(const ds4_layer *l) "
    "{ return l->n_expert / 6; }\n"
    "Review note: slot math divides by top-k; guard n_expert==0 before use.\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--block-tokens', type=int, default=400,
                    help='approx tokens per round (repeat block to reach it)')
    ap.add_argument('--out', default='speed-bench/M3/m3-3/sim_prompt.txt')
    args = ap.parse_args()
    # ~3.5 chars/token on code-ish English text; repeat the block so one
    # round is ~block-tokens tokens.
    rep = max(1, round(args.block_tokens * 3.5 / len(BLOCK)))
    one_round = BLOCK * rep
    text = one_round * args.rounds
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(text)
    print('wrote %s: %d rounds x %d reps = %d chars' %
          (args.out, args.rounds, rep, len(text)))


if __name__ == '__main__':
    main()