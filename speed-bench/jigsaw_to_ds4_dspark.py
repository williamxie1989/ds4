#!/usr/bin/env python3
"""JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF -> ds4 dspark sidecar re-header.

Mechanical, contract-verified against ds4.c:
  names    blk.N.<suffix>          -> mtp.N.<suffix>          (bind: 8770-8796)
           fc.weight               -> mtp.0.main_proj.weight
           enc.output_norm.weight  -> mtp.0.main_norm.weight
           output_norm.weight      -> mtp.<last>.norm.weight
           markov_w{1,2}.weight    -> mtp.<last>.markov_head.markov_w{1,2}.weight
           conf_proj.weight        -> mtp.<last>.confidence_head.proj.weight
  dtypes   bf16 -> f16 numerically (DENSE kind rejects bf16; ds4.c:6106-6121)
           widths identical, offsets/sizes unchanged; data blob otherwise
           copied verbatim (q8_0 dense, f32 norms, mxfp4 routed experts
           are all in ds4's accepted sets).
  metadata rebuilt as the 13-KV sidecar contract; architecture MUST be
           deepseek41-dspark for the DEEPSEEK41 family
           (support_model_checkpoint_compatible, ds4.c:3292-3301).

Usage: jigsaw_to_ds4_dspark.py IN.gguf OUT.gguf
"""
import os
import struct
import sys

import numpy as np

BF16, F16 = 30, 1
OUT_KV = [
    ('general.architecture', 8, b'deepseek41-dspark'),
    ('general.name', 8, b'DeepSeek V4.1 Flash DSpark (re-headered JigSawPT draft)'),
    ('general.alignment', 4, 32),
    ('dspark.block_size', 4, 5),
    ('dspark.markov_rank', 4, 256),
    ('dspark.noise_token_id', 4, 128799),
    ('dspark.target_layer_ids', (9, 4, [37, 38, 39]), None),
    ('dspark.stage_count', 4, 3),
    ('dspark.n_layers', 4, 3),
    ('dspark.expert_count', 4, 128),
    ('dspark.expert_used_count', 4, 3),
    ('general.source.url', 8,
     b'https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF'),
    ('general.source.revision', 8,
     b'0fe521efb6e3aaab7af758d50aa5d7688a4bfa8a'),
]


def u64(f):
    return struct.unpack('<Q', f.read(8))[0]


def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    fin = open(src_path, 'rb')
    if fin.read(4) != b'GGUF' or struct.unpack('<I', fin.read(4))[0] != 3:
        sys.exit('expected GGUF v3')
    n_tensors, n_kv = u64(fin), u64(fin)

    def take(n):
        b = fin.read(n)
        assert len(b) == n
        return b

    def kvskip(t):
        scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                  10: 8, 11: 8, 12: 8}
        if t in scalar:
            take(scalar[t])
        elif t == 8:
            take(u64(fin))
        elif t == 9:
            et = struct.unpack('<I', take(4))[0]
            for _ in range(u64(fin)):
                kvskip(et)
        else:
            sys.exit('bad value type %d' % t)

    align_in = 32
    for _ in range(n_kv):
        key = take(u64(fin)).decode()
        typ = struct.unpack('<I', take(4))[0]
        if key == 'general.alignment' and typ == 4:
            align_in = struct.unpack('<I', take(4))[0] or 32
        else:
            kvskip(typ)

    rename = {
        'fc.weight': 'mtp.0.main_proj.weight',
        'enc.output_norm.weight': 'mtp.0.main_norm.weight',
        'output_norm.weight': 'mtp.2.norm.weight',
        'markov_w1.weight': 'mtp.2.markov_head.markov_w1.weight',
        'markov_w2.weight': 'mtp.2.markov_head.markov_w2.weight',
        'conf_proj.weight': 'mtp.2.confidence_head.proj.weight',
    }
    infos = []
    for _ in range(n_tensors):
        name = take(u64(fin)).decode()
        dims = [u64(fin) for _ in range(struct.unpack('<I', take(4))[0])]
        dtype = struct.unpack('<I', take(4))[0]
        off = u64(fin)
        if name.startswith('blk.'):
            name = 'mtp.' + name[4:]
        elif name in rename:
            name = rename[name]
        infos.append([name, dims, dtype, off])
    if len(infos) != 78 or any(not n.startswith('mtp.') for n, *_ in infos):
        sys.exit('unexpected tensor set after rename')
    # GGUF tensor data starts at the alignment-rounded offset past the
    # tensor directory, not at the raw header end.
    data_start_in = (fin.tell() + align_in - 1) // align_in * align_in

    def nbytes(dims, dtype):
        nelem = int(np.prod(dims))
        w = {0: 4, 1: 2, 30: 2}.get(dtype)
        if w:
            return nelem * w
        block, bpp = {8: (32, 34), 39: (32, 17)}[dtype]
        assert nelem % block == 0
        return nelem // block * bpp

    spans = sorted((off, off + nbytes(dims, dt), name, dt)
                   for name, dims, dt, off in infos)

    # ds41_draft_block() CPU dot-products confidence_head.proj as float32
    # (ds4.c: proj = type==F32 ? map+offset : NULL) — widen it to f32.
    # Everything else bf16 -> f16 keeps widths, but f32 grows, so the data
    # blob is rebuilt per-tensor with fresh sequential offsets.
    CONF = 'mtp.2.confidence_head.proj.weight'
    new_off = {}
    out_dt = {}
    cur = 0
    for o0, o1, name, dt in spans:
        new_off[o0] = cur
        out_dt[o0] = 0 if name == CONF else (1 if dt == BF16 else dt)
        delta = (o1 - o0) * 2 if (out_dt[o0] == 0 and dt == BF16) else (o1 - o0)
        cur = (cur + delta + 31) // 32 * 32   # keep 32B tensor alignment

    fout = open(dst_path + '.tmp', 'wb')
    fout.write(b'GGUF')
    fout.write(struct.pack('<I', 3))
    fout.write(struct.pack('<Q', len(infos)))
    fout.write(struct.pack('<Q', len(OUT_KV)))
    for key, typ, val in OUT_KV:
        kb = key.encode()
        fout.write(struct.pack('<Q', len(kb)) + kb)
        if isinstance(typ, tuple):
            arr_t, item_t, items = typ
            fout.write(struct.pack('<I', arr_t))
            fout.write(struct.pack('<I', item_t))
            fout.write(struct.pack('<Q', len(items)))
            for it in items:
                fout.write(struct.pack('<I', it))
        elif typ == 8:
            fout.write(struct.pack('<I', typ))
            fout.write(struct.pack('<Q', len(val)) + val)
        else:
            fout.write(struct.pack('<I', typ))
            fout.write(struct.pack('<I', val))
    for name, dims, dt, off in infos:
        nb = name.encode()
        fout.write(struct.pack('<Q', len(nb)) + nb)
        fout.write(struct.pack('<I', len(dims)))
        for d in dims:
            fout.write(struct.pack('<Q', d))
        fout.write(struct.pack('<I', out_dt[off]))
        fout.write(struct.pack('<Q', new_off[off]))
    header_end = fout.tell()
    data_start_out = (header_end + 31) // 32 * 32
    fout.write(b'\0' * (data_start_out - header_end))

    # Per-tensor copy: bf16 payloads stream through raw (f16-converted in a
    # second pass); the confidence head widens to f32 right here.
    for o0, o1, name, dt in spans:
        if fout.tell() % 32:
            fout.write(b'\0' * (32 - fout.tell() % 32))
        assert fout.tell() == data_start_out + new_off[o0]
        fin.seek(data_start_in + o0)
        n = o1 - o0
        if out_dt[o0] == 0 and dt == BF16:
            raw = np.frombuffer(fin.read(n), dtype=np.uint16)
            fout.write(((raw.astype(np.uint32) << 16).view(np.float32)).tobytes())
            print('widened bf16->f32:', name, raw.size, 'elems')
        elif out_dt[o0] == 0:
            fout.write(fin.read(n))
        else:
            copied = 0
            while copied < n:
                m = min(1 << 26, n - copied)
                fout.write(fin.read(m))
                copied += m
    fout.close()
    fin.close()

    # bf16 -> f16 in place on the output (widths identical).
    with open(dst_path + '.tmp', 'r+b') as fdst, open(src_path, 'rb') as fsrc:
        for off0, off1, name, dt in spans:
            if dt != BF16 or out_dt[off0] == 0:
                continue
            fsrc.seek(data_start_in + off0)
            raw = np.frombuffer(fsrc.read(off1 - off0), dtype=np.uint16)
            f32 = (raw.astype(np.uint32) << 16).view(np.float32)
            fdst.seek(data_start_out + new_off[off0])
            fdst.write(f32.astype(np.float16).tobytes())
            print('converted bf16->f16:', name, raw.size, 'elems')

    os.replace(dst_path + '.tmp', dst_path)
    print('wrote %s (%.2f GiB, data_start %d)' %
          (dst_path, os.path.getsize(dst_path) / 2**30, data_start_out))


if __name__ == '__main__':
    main()
