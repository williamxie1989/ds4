#!/usr/bin/env python3
"""P2-0 offline MTP feasibility replay over a ds4 V4.1 decode expert trace.

Input: DS4_V41_TRACE_EXPERTS binary, records of (uint32 pos, uint32 layer,
int32 x DS4_N_EXPERT_USED selected ids), written only while
DS4_METAL_DISABLE_V41_DECODE_QUEUE=1 drains every layer.

Two ledgers mirror the streaming cache:
  GPU expert cache: per-layer LRU, `gpu_slots` slots per layer (9.49 MiB each).
  OS page cache:   whole-file LRU, `pc_slots` slots (misses not resident here
                   become SSD pread).
Reported per regime: GPU-cache misses (== DRAM bytes) and SSD pread bytes.

Serial regime: one token = 6 experts per layer.
MTP regime (block 5, verify 6 rows, early-stop impossible on the MoE union):
  every round unions the selected sets of the next up-to-6 decode tokens at a
  cost of 6-row union sets, and advances accept+1 tokens.  For each acceptance
  value the per-accepted-token ledgers are compared against serial.
Acceptance here is exogenous: the trace gives the exact selected sets for
accepted tokens; the union over 6 consecutive tokens is the same regardless of
how the draft split them into rounds, so replay is exact for the ledger math.

Usage: mtp_ledger_replay.py TRACE [--gpu-total 8685] [--pc-gib 90] [--warmup 1024]
"""
import struct
import sys
from collections import OrderedDict

N_LAYER = 40
N_EXPERT = 384
TOPK = 6
SLOT_MIB = 9.49

def load(path):
    rec = struct.Struct('<II6i')
    toks = {}
    with open(path, 'rb') as f:
        buf = f.read()
    step = rec.size
    for off in range(0, len(buf) - step + 1, step):
        pos, il, *ids = rec.unpack_from(buf, off)
        toks.setdefault(pos, [None] * N_LAYER)[il] = frozenset(ids)
    seq = [toks[p] for p in sorted(toks)]
    if seq and any(l is None for l in seq[0]):
        # A partial trailing token cannot be replayed layer-exactly.
        seq.pop()
    return seq

class Ledger:
    __slots__ = ('gpu', 'cap', 'pc', 'pc_cap', 'dram_slots', 'ssd_slots')
    def __init__(self, gpu_per_layer, pc_slots, shared_pc=None):
        self.gpu = [OrderedDict() for _ in range(N_LAYER)]
        self.cap = gpu_per_layer
        self.pc = shared_pc if shared_pc is not None else OrderedDict()
        self.pc_cap = pc_slots
        self.dram_slots = 0
        self.ssd_slots = 0

    def touch(self, layer, expert):
        g = self.gpu[layer]
        key = expert
        if key in g:
            g.move_to_end(key)
            return
        self.dram_slots += 1
        self.gpu[layer][key] = 1
        if len(g) > self.cap:
            del g[next(iter(g))]
        pc = self.pc
        gkey = (layer, expert)
        if gkey in pc:
            pc.move_to_end(gkey)
        else:
            self.ssd_slots += 1
            pc[gkey] = 1
            if len(pc) > self.pc_cap:
                del pc[next(iter(pc))]

def run(seq, gpu_per_layer, pc_slots, warmup):
    led = Ledger(gpu_per_layer, pc_slots)
    seen = 0
    hits = tot = 0
    for tok in seq:
        for il in range(N_LAYER):
            for e in tok[il]:
                g = led.gpu[il]
                pre = e in g
                led.touch(il, e)
                if seen >= warmup:
                    tot += 1
                    hits += pre
        seen += 1
    return led, (hits / tot if tot else 0)

def run_mtp(seq, gpu_per_layer, pc_slots, warmup, accept):
    # One round per (accept+1) advanced tokens costs the union of the next
    # 6 tokens' sets (verify 6 rows; the rejected row still paid its experts).
    led = Ledger(gpu_per_layer, pc_slots)
    advanced = 0
    i = len(seq)
    n = 0
    pos = 0
    while pos < len(seq):
        rows = seq[pos:pos + 6]
        # A short tail still pays a verify of its available rows.
        for il in range(N_LAYER):
            union = set()
            for tok in rows:
                union |= tok[il]
            for e in union:
                led.touch(il, e)
        adv = min(accept + 1, len(rows))
        advanced += adv
        pos += adv
        if pos + 6 > len(seq) and pos < len(seq) and accept + 1 >= len(seq) - pos:
            pos = len(seq)
    return led

def main():
    args = sys.argv[1:]
    path = args[0]
    opts = dict(a.split('=') for a in args[1:]) if len(args) > 1 else {}
    gpu_total = int(opts.get('gpu_total', 8685))
    gpu_per_layer = gpu_total // N_LAYER
    pc_slots = int(float(opts.get('pc_gib', 90)) * 1024 / SLOT_MIB)
    warmup = int(opts.get('warmup', 1024))
    seq = load(path)
    n_tok = len(seq)
    if n_tok <= warmup + 64:
        sys.exit(f'trace too short: {n_tok} tokens')
    # Shared page cache is NOT shared between regimes; each gets its own.
    ser, hit = run(seq, gpu_per_layer, pc_slots, warmup)
    span = n_tok - warmup
    s_dram, s_ssd = ser.dram_slots / span, ser.ssd_slots / span
    print(f'tokens={n_tok} warmup={warmup} gpu_slots/layer={gpu_per_layer} '
          f'pc_slots={pc_slots}')
    print(f'SERIAL    sim hit_rate={hit:.3f}  (measured anchor ~0.834)')
    print(f'SERIAL    DRAM {s_dram:.3f} slots/token '
          f'({s_dram*SLOT_MIB:6.2f} GiB)   SSD {s_ssd:.3f} slots/token '
          f'({s_ssd*SLOT_MIB:6.2f} GiB)')
    print()
    print('MTP block-5, verify 6 rows   DRAM/acc-token  ratio   SSD/acc-token  ratio   verdict')
    ok_any = False
    for a in range(0, 6):
        m = run_mtp(seq, gpu_per_layer, pc_slots, warmup, a)
        # per accepted token:
        acc_tokens = span
        dram = m.dram_slots / acc_tokens
        ssd = m.ssd_slots / acc_tokens
        rd, rs = dram / s_dram, ssd / s_ssd
        verdict = 'PASS' if rd <= 1.3 and rs <= 2.2 else 'fail'
        ok_any |= verdict == 'PASS'
        print(f'  accept={a}  DRAM {dram*SLOT_MIB:8.2f} GiB  x{rd:5.2f}   '
              f'SSD {ssd*SLOT_MIB:8.2f} GiB  x{rs:5.2f}   {verdict}')
    print()
    print('MTP ledger verdict:', 'feasible at >=1 acceptance level' if ok_any
          else 'INFEASIBLE under the 1.3x DRAM / 2.2x SSD criterion')

if __name__ == '__main__':
    main()
