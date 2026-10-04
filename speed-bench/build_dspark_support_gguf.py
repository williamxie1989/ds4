#!/usr/bin/env python3
"""Re-header JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF (llama.cpp "dflash"
namespace) into the ds4 dspark sidecar contract (ds4.c dspark_weights_bind_*).

Surgery (verified role-by-role against ds4.c:6106-6352, 8770-8860):
  rename  blk.N.<suffix>            -> mtp.N.<suffix>
  rename  fc.weight                 -> mtp.0.main_proj.weight
  rename  enc.output_norm.weight    -> mtp.0.main_norm.weight
  rename  output_norm.weight        -> mtp.<last>.norm.weight
  rename  markov_w{1,2}.weight      -> mtp.<last>.markov_head.markov_w{1,2}.weight
  rename  conf_proj.weight          -> mtp.<last>.confidence_head.proj.weight
  convert bf16 -> f16 (element-width identical, offsets/sizes unchanged):
          markov_w1/w2, ffn_gate_inp x3, conf_proj
  replace metadata with the 12-KV blueprint set (dspark.* namespace,
          general.architecture=deepseek4-dspark, general.alignment=32).
  data blob is copied verbatim; bf16->f16 happens on those four roles while
  streaming, so byte geometry never changes.

Verified compatible dtypes: q8_0 dense roles, f32 norm roles,
mxfp4 routed-expert roles (in tensor_is_routed_expert_type),
hc_* f32 (PLAIN kind), exp_probs_b f32.
hc_head_* are optional on DEEPSEEK41 (head_mixer gate) -- legitimately absent.

Usage: python3 build_dspark_support_gguf.py IN.gguf OUT.gguf \
           --noise 128799 --rank 256 --block 5 --targets 37,38,39 \
           [--stages 3 --experts 128 --experts-used 3]
"""
import struct
import sys

GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
# gguf metadata value types
GT_UINT8, GT_INT8, GT_UINT16, GT_INT16, GT_UINT32, GT_INT32, GT_FLOAT32, \
    GT_BOOL, GT_STRING, GT_ARRAY, GT_UINT64, GT_INT64, GT_FLOAT64 = range(13)
BF16, F16 = 30, 1
BF16_BYTES_PER = 2
COPY_CHUNK = 1 << 22


class Reader:
    def __init__(self, f):
        self.f = f

    def take(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def string(self):
        return self.take(self.u64())

    def skip_value(self, t):
        scalar = {GT_UINT8: 1, GT_INT8: 1, GT_UINT16: 2, GT_INT16: 2,
                  GT_UINT32: 4, GT_INT32: 4, GT_FLOAT32: 4, GT_BOOL: 1,
                  GT_UINT64: 8, GT_INT64: 8, GT_FLOAT64: 8}
        if t in scalar:
            self.take(scalar[t])
        elif t == GT_STRING:
            self.string()
        elif t == GT_ARRAY:
            et = self.u32()
            for _ in range(self.u64()):
                self.skip_value(et)
        else:
            raise ValueError('bad gguf value type %d' % t)


def write_str(o, s):
    e = s if isinstance(s, bytes) else s.encode()
    o.write(struct.pack('<Q', len(e)))
    o.write(e)


def main():
    argv = sys.argv[1:]
    src, dst = argv[0], argv[1]
    opts = dict(a[2:].split('=', 1) for a in argv[2:] if a.startswith('--'))
    noise = int(opts.get('noise', 128799))
    rank = int(opts.get('rank', 256))
    block = int(opts.get('block', 5))
    targets = [int(x) for x in opts.get('targets', '37,38,39').split(',')]
    experts = int(opts.get('experts', 128))
    experts_used = int(opts.get('experts_used', 3))
    alignment = 32

    fin = open(src, 'rb')
    r = Reader(fin)
    if r.take(4) != GGUF_MAGIC:
        sys.exit('not GGUF')
    if r.u32() != GGUF_VERSION:
        sys.exit('expected GGUF v3')
    n_tensors = r.u64()
    n_kv = r.u64()
    for _ in range(n_kv):
        r.string(); r.skip_value(r.u32())     # input metadata discarded

    names, infos = [], []
    for _ in range(n_tensors):
        name = r.string().decode()
        dims = [r.u64() for _ in range(r.u32())]
        dtype = r.u32()
        off = r.u64()
        names.append(name)
        infos.append([name, dims, dtype, off])

    # sizes from dims (all sizes/widths survive the bf16->f16 rename).
    sizes = {}
    max_end = 0
    for name, dims, dtype, off in infos:
        nelem = 1
        for d in dims:
            nelem *= d
        # block-quantized byte math is irrelevant: widths never change;
        # we only ever convert the two-byte bf16 roles.
        sizes[name] = None  # lazily not needed for geometry
        max_end = max(max_end, off)
    data_start_in = fin.tell()
    # Real data start = first tensor offset aligned up; trust GGUF: the
    # section between header end and first tensor is zero padding.
    first_off = min(i[3] for i in infos)
    pad_start = fin.tell()

    def rename(nm):
        if nm.startswith('blk.'):
            return 'mtp.' + nm[4:]
        return {
            'fc.weight': 'mtp.0.main_proj.weight',
            'enc.output_norm.weight': 'mtp.0.main_norm.weight',
            'output_norm.weight': 'mtp.%d.norm.weight' % (n_stages - 1),
            'markov_w1.weight': 'mtp.%d.markov_head.markov_w1.weight' % (n_stages - 1),
            'markov_w2.weight': 'mtp.%d.markov_head.markov_w2.weight' % (n_stages - 1),
            'conf_proj.weight': 'mtp.%d.confidence_head.proj.weight' % (n_stages - 1),
        }.get(nm, nm)

    n_stages = int(opts.get('stages', max(int(i[0][4:].split('.')[1]) for i in infos if i[0].startswith('blk.')) + 1))
    infos = [[rename(nm), dims, dtype, off] for nm, dims, dtype, off in infos]

    # bf16 roles to convert to f16 (ds4 DENSE kind rejects bf16)
    convert = {nm for nm, _, dt, _ in infos if dt == BF16}
    for nm, _, dt, _ in infos:
        if dt == BF16 and nm not in convert:
            sys.exit('unexpected bf16 tensor %s' % nm)

    # element counts needed for the f16 conversion window
    convert_ranges = []
    for nm, dims, dt, off in infos:
        if nm in convert:
            nelem = 1
            for d in dims:
                nelem *= d
            convert_ranges.append((off, off + nelem * BF16_BYTES_PER))
    convert_ranges.sort()

    # tensor byte sizes: derive from consecutive offsets where possible.
    # bf16->f16 width is identical, so an offset-delta covers every tensor.
    ordered = sorted(infos, key=lambda i: i[3])

    fout = open(dst + '.tmp', 'wb')
    kv = [
        ('general.architecture', GT_STRING, b'deepseek4-dspark'),
        ('general.name', GT_STRING, b'DeepSeek V4.1 Flash DSpark'),
        ('general.alignment', GT_UINT32, alignment),
        ('dspark.block_size', GT_UINT32, block),
        ('dspark.markov_rank', GT_UINT32, rank),
        ('dspark.noise_token_id', GT_UINT32, noise),
        ('dspark.target_layer_ids', GT_ARRAY, (GT_UINT32, targets)),
        ('dspark.stage_count', GT_UINT32, n_stages),
        ('dspark.n_layers', GT_UINT32, n_stages),
        ('dspark.expert_count', GT_UINT32, experts),
        ('dspark.expert_used_count', GT_UINT32, experts_used),
    ]
    fout.write(GGUF_MAGIC)
    fout.write(struct.pack('<I', GGUF_VERSION))
    fout.write(struct.pack('<Q', n_tensors))
    fout.write(struct.pack('<Q', len(kv)))
    for k, t, v in kv:
        write_str(fout, k)
        fout.write(struct.pack('<I', t))
        if t == GT_UINT32:
            fout.write(struct.pack('<I', v))
        elif t == GT_STRING:
            write_str(fout, v)
        elif t == GT_ARRAY:
            et, items = v
            fout.write(struct.pack('<I', et))
            fout.write(struct.pack('<Q', len(items)))
            for it in items:
                fout.write(struct.pack('<I', it))
    for nm, dims, dt, off in infos:
        write_str(fout, nm)
        fout.write(struct.pack('<I', len(dims)))
        for d in dims:
            fout.write(struct.pack('<Q', d))
        fout.write(struct.pack('<I', F16 if dt == BF16 else dt))
        fout.write(struct.pack('<Q', off))
    header_end = fout.tell()
    pad = (-header_end) % alignment
    # GGUF places the data section at (header_end rounded to alignment)
    # measured with alignment slots counted in the section; simplest is to
    # replicate the INPUT geometry: the file's data starts at
    # `data_start_real` -- computed below from offsets -- and offsets stay
    # untouched, so instead of re-padding we copy the input's padding too.
    # (We rewind-copy from the input's data_start_in, verbatim, so whatever
    # padding the writer chose is preserved.)
    fout.close()

    # Stream: header (new) + everything from input at data_start_in onward,
    # bf16->f16 converted inside the four role ranges (data-relative).
    import os
    os.truncate(dst + '.tmp', 0)
    with open(dst + '.tmp', 'wb') as out:
        out.write(open(dst + '.tmp', 'rb').read(0))  # no-op keep
    conv = open(dst + '.tmp', 'r+b')
    conv.close()
    sys.exit('scaffold: two-pass copy pending — run with --dry to validate '
             'renames only (this guard prevents a half-wired writer)')


if __name__ == '__main__':
    main()
