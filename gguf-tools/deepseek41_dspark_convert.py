#!/usr/bin/env python3
"""Convert official DeepSeek-V4.1-Flash MTP shards (44-46) into a ds4 dspark
sidecar GGUF, mirroring what DeepSeek-V4-Flash shipped for the V4 draft.

Source geometry (verified against config.json + safetensors headers):
  mtp.{0,1,2}.ffn.experts.{e}.w{1,2,3}.weight  I8   (2304, 2560) / (5120, 1152)
      with .scale F8_E8M0 (2304, 160) / (5120, 72)
      -> I8 packs 2 logical elements per byte, one E8M0 scale per 16 logical
         elements. Logical widths are 5120 (in) and 2304 (out).
  mtp.{0,1,2}.ffn.shared_experts.w{1,2,3}.weight F8_E4M3 + 32x32 E8M0 scales
  mtp.{0,1,2}.attn.*            F8_E4M3 / BF16 / F32
  mtp.{0,1,2}.hc_*             F32
  mtp.{0,1,2}.ffn.gate.weight  BF16 (router)
  mtp.0.main_proj.weight       F8_E4M3 (5120, 15360) -> logical (15360, 5120)
  mtp.2.markov_head.embed.weight BF16 (129280, 256)
  mtp.2.confidence_head.proj.weight BF16 (1, 5376)

Output contract: the 78 tensor names dspark_bind_block() binds (ds4.c:8770-8796)
plus the 12-KV dspark metadata block, identical to the JigSawPT re-header that
ds4 already accepts (missing=0 invalid=0).

Recipes (--recipe):
  main   routed experts follow the MAIN MODEL: IQ2_XXS gate/up, Q2_K down.
         This is what DeepSeek shipped for the V4 dspark draft and what
         deepseek41_quantize.py uses for V4.1 backbone layers.
  mxfp4  routed experts as MXFP4, matching the third-party JigSawPT draft.

confidence_head.proj is always widened to F32: ds41_draft_block() dot-products
it as float32 (ds4.c), and the bf16->f32 widening in jigsaw_to_ds4_dspark.py
was the "draft confidence always zero" root cause (plan doc 8.2).

Usage:
  deepseek41_dspark_convert.py --source DIR --out FILE [--recipe main|mxfp4]
                                [--imatrix FILE] [--threads N] [--dry-run]
"""
import argparse
import concurrent.futures
import dataclasses
import json
import os
import re
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepseek41_metadata import GGUF_ALIGNMENT as _MAIN_ALIGNMENT
import glm53_quantize as _q  # noqa: E402
from deepseek41_quantize import NativeQuantizer  # noqa: E402
from glm53_quantize import (  # noqa: E402
    QTYPE_BF16, QTYPE_F16, QTYPE_F32, QTYPE_IQ2_XXS, QTYPE_I8,
    QTYPE_NAMES, QTYPE_Q2_K, QTYPE_Q8_0, SourceDB, TensorPlan, align,
    kv_string, qtype_nbytes, tensor_header,
)

# The dspark sidecar declares general.alignment=32, so offsets must be padded
# to 32. deepseek41_metadata.GGUF_ALIGNMENT is the 16 KiB the V4.1 backbone
# uses; reusing it here makes ds4 read every tensor at the wrong offset.
GGUF_ALIGNMENT = 32

# MXFP4 exists in the C quantizer (DS4Q_TYPE_MXFP4 = 39) but the Python wrapper
# never exported a constant for it.
QTYPE_MXFP4 = 39
QTYPE_NAMES[QTYPE_MXFP4] = "MXFP4"
_q.QTYPE_LAYOUT[QTYPE_MXFP4] = (32, 17)

STAGES = 3
EXPERTS = 128
EXPERTS_USED = 3
BLOCK_SIZE = 5
MARKOV_RANK = 256
NOISE_TOKEN_ID = 128799
TARGET_LAYERS = (37, 38, 39)
HIDDEN = 5120
INTER = 2304
Q_LORA = 1280
O_LORA = 1024
O_GROUPS = 8
HEADS = 64
HEAD_DIM = 512
HC_DIM = 24
VOCAB = 129280

RECIPES = {
    "main": {"gate": QTYPE_IQ2_XXS, "up": QTYPE_IQ2_XXS, "down": QTYPE_Q2_K},
    "mxfp4": {"gate": QTYPE_MXFP4, "up": QTYPE_MXFP4, "down": QTYPE_MXFP4},
}

OUT_KV = [
    ('general.architecture', 8, b'deepseek41-dspark'),
    ('general.name', 8, b'DeepSeek V4.1 Flash DSpark (official mtp shards)'),
    ('general.alignment', 4, 32),
    ('dspark.block_size', 4, BLOCK_SIZE),
    ('dspark.markov_rank', 4, MARKOV_RANK),
    ('dspark.noise_token_id', 4, NOISE_TOKEN_ID),
    ('dspark.target_layer_ids', 9, (4, list(TARGET_LAYERS))),
    ('dspark.stage_count', 4, STAGES),
    ('dspark.n_layers', 4, STAGES),
    ('dspark.expert_count', 4, EXPERTS),
    ('dspark.expert_used_count', 4, EXPERTS_USED),
    ('general.source.url', 8,
     b'https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash'),
]


def kv_records(recipe):
    """Serialize OUT_KV into GGUF metadata records."""
    out = []
    for key, typ, val in OUT_KV:
        kb = key.encode()
        rec = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", typ)
        if typ == 8:
            out.append(rec + struct.pack("<Q", len(val)) + val)
        elif typ == 9:
            item_type, items = val
            out.append(rec + struct.pack("<I", item_type) + struct.pack("<Q", len(items))
                       + b"".join(struct.pack("<I", i) for i in items))
        else:
            out.append(rec + struct.pack("<I", val))
    out.append(kv_string("deepseek41.dspark.recipe", recipe))
    return out

# official source name -> (ds4 tensor name, qtype) for the non-expert tensors.
# Verified against dspark_bind_block() (ds4.c:8770-8796). The hc_* tensors
# carry no .weight suffix upstream; hc_*_fn arrives transposed as (24, 20480).
# The attention tensors use ds4's flat names, so the attn.<x> prefix is dropped.
PLAIN = [
    # head mixing (source has no .weight suffix)
    ('hc_attn_fn', 'hc_attn_fn.weight', QTYPE_F32),
    ('hc_attn_scale', 'hc_attn_scale.weight', QTYPE_F32),
    ('hc_attn_base', 'hc_attn_base.weight', QTYPE_F32),
    ('hc_ffn_fn', 'hc_ffn_fn.weight', QTYPE_F32),
    ('hc_ffn_scale', 'hc_ffn_scale.weight', QTYPE_F32),
    ('hc_ffn_base', 'hc_ffn_base.weight', QTYPE_F32),
    # norms
    ('attn_norm.weight', 'attn_norm.weight', QTYPE_F32),
    ('ffn_norm.weight', 'ffn_norm.weight', QTYPE_F32),
    # attention
    ('attn.attn_sink', 'attn_sinks.weight', QTYPE_F32),
    ('attn.kv_norm.weight', 'attn_kv_a_norm.weight', QTYPE_F32),
    ('attn.q_norm.weight', 'attn_q_a_norm.weight', QTYPE_F32),
    ('attn.wq_a.weight', 'attn_q_a.weight', QTYPE_Q8_0),
    ('attn.wq_b.weight', 'attn_q_b.weight', QTYPE_Q8_0),
    ('attn.wkv.weight', 'attn_kv.weight', QTYPE_Q8_0),
    ('attn.wo_a.weight', 'attn_output_a.weight', QTYPE_Q8_0),
    ('attn.wo_b.weight', 'attn_output_b.weight', QTYPE_Q8_0),
    # router
    ('ffn.gate.weight', 'ffn_gate_inp.weight', QTYPE_F16),
    ('ffn.gate.bias', 'exp_probs_b.bias', QTYPE_F32),
    # ds4 binds only one bias slot; bias_vl has no home and is dropped.
]

# source name -> output shape, for entries whose ds4 shape is not the source
# shape. GGUF labels HF (out, in) matrices with reversed (in, out) dims while
# the flat bytes stay in HF row-major order (see _orient). Shapes are taken
# from the JigSawPT sidecar ds4 already accepts (missing=0 invalid=0).
SHAPES = {
    'hc_attn_fn': (HIDDEN * 4, HC_DIM),
    'hc_ffn_fn': (HIDDEN * 4, HC_DIM),
    'attn.wq_a.weight': (HIDDEN, Q_LORA),
    'attn.wq_b.weight': (Q_LORA, HEADS * HEAD_DIM),
    'attn.wkv.weight': (HIDDEN, HEAD_DIM),
    'attn.wo_a.weight': (HEADS * HEAD_DIM // O_GROUPS, O_GROUPS * O_LORA),
    'attn.wo_b.weight': (O_GROUPS * O_LORA, HIDDEN),
    'ffn.gate.weight': (HIDDEN, EXPERTS),
}

SHARED = [('w1', 'gate'), ('w3', 'up'), ('w2', 'down')]


def validate(tensors):
    """Reject any source tensor whose fp8 geometry is not the expected one.

    Two layouts coexist in the official checkpoint: routed experts are I8 with
    one E8M0 scale per 16 logical elements (I8 packs 2 elements per byte), while
    shared/attention weights are F8_E4M3 with 32x32 block scales.
    """
    for name, info in tensors.items():
        if not name.endswith(".weight"):
            continue
        dtype, shape = info["dtype"], info["shape"]
        if dtype not in ("I8", "F8_E4M3"):
            continue
        if dtype == "I8":
            if len(shape) != 2:
                raise ValueError(f"{name}: expert weights must be 2-d I8, got {shape}")
            # I8 packs 2 logical elements per byte, one E8M0 scale per 16 of
            # those bytes -- same convention as deepseek41_quantize.py.
            expected = [shape[0], shape[1] // 16]
        else:
            expected = [(d + 31) // 32 for d in shape]
        scale = tensors.get(name.removesuffix(".weight") + ".scale")
        if not scale or scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected}, got "
                             f"{scale['shape'] if scale else None}")


def build_plan(db, recipe):
    q = RECIPES[recipe]
    plan = []

    def add(name, shape, qtype, role, source, experts=0, part=None, layer=None):
        plan.append(TensorPlan(name, tuple(shape), qtype, role, source=source,
                               expert_layer=layer, expert_part=part,
                               expert_count=experts))

    for stage in range(STAGES):
        s = f"mtp.{stage}"
        dst = f"mtp.{stage}"
        for suffix, out, qtype in PLAIN:
            src = f"{s}.{suffix}"
            shape = SHAPES.get(suffix, tuple(db.info(src)["shape"]))
            add(f"{dst}.{out}", shape, qtype, "dense", src)

        for w, part in SHARED:
            src = f"{s}.ffn.shared_experts.{w}.weight"
            shape = (INTER, HIDDEN) if w == "w2" else (HIDDEN, INTER)
            add(f"{dst}.ffn_{part}_shexp.weight", shape, QTYPE_Q8_0, "shared", src)

        for w, part in SHARED:
            # HF stores (out, in) as I8; ds4 wants the 3D stack reversed to
            # (in, out, expert) so the transposed read is contiguous per expert.
            src = f"{s}.ffn.experts.{{expert}}.{w}.weight"
            shape = (INTER, HIDDEN, EXPERTS) if w == "w2" else (HIDDEN, INTER, EXPERTS)
            add(f"{dst}.ffn_{part}_exps.weight", shape, q[part], "experts", src,
                experts=EXPERTS, part=part, layer=stage)

        if stage == 0:
            add(f"{dst}.main_proj.weight", (HIDDEN * 3, HIDDEN), QTYPE_Q8_0,
                "dense", f"{s}.main_proj.weight")
            add(f"{dst}.main_norm.weight", (HIDDEN,), QTYPE_F32, "norm",
                f"{s}.main_norm.weight")
        if stage == STAGES - 1:
            add(f"{dst}.norm.weight", (HIDDEN,), QTYPE_F32, "norm",
                f"{s}.norm.weight")
            add(f"{dst}.markov_head.markov_w1.weight", (MARKOV_RANK, VOCAB),
                QTYPE_F16, "dense", f"{s}.markov_head.embed.weight")
            add(f"{dst}.markov_head.markov_w2.weight", (MARKOV_RANK, VOCAB),
                QTYPE_F16, "dense", f"{s}.markov_head.head.weight")
            # ds4 dot-products this one as float32.
            add(f"{dst}.confidence_head.proj.weight", (5376, 1), QTYPE_F32,
                "dense", f"{s}.confidence_head.proj.weight")

    return plan


def _orient(values, item, np):
    """GGUF matrices keep the HF row-major (out, in) flat bytes and reverse the
    DIMS LABEL to (in, out) -- the same convention deepseek41_quantize.py uses
    for the backbone and the JigSawPT sidecar ds4 already accepts. Quantizer
    blocks run along the fast HF axis. Never transpose the data itself:
    transposing the flat bytes on top of the reversed dims double-flips every
    matrix, which is why the first official-main build proposed tokens the
    target model never accepted."""
    want = item.shape[:2] if item.expert_count else item.shape
    if values.ndim == 1:
        return values
    if tuple(values.shape) == tuple(reversed(want)):
        return values
    raise ValueError(f"{item.name}: dequantized {values.shape} is not the "
                     f"reverse of the planned dims {want}")


def write_gguf(args, plan, records, db):
    quantizer = NativeQuantizer(args.quants_library)
    np = quantizer.np
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(i) for i in plan)
    data_start = (len(header) + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT * GGUF_ALIGNMENT
    header += bytes(data_start - len(header))
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")

    with open(args.out, "xb") as fp, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        fp.write(header)
        fp.seek(data_start)
        for index, item in enumerate(plan):
            started = time.monotonic()
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")

            if item.expert_count:
                def convert(expert):
                    values = quantizer.to_f32(db, item.source.format(expert=expert))
                    return quantizer.encode(_orient(values, item, np), item.qtype)
                for start in range(0, item.expert_count, args.threads):
                    futures = [pool.submit(convert, e) for e in
                               range(start, min(start + args.threads, item.expert_count))]
                    for future in futures:
                        data = future.result()
                        if len(data) != item.nbytes // item.expert_count:
                            raise ValueError(f"wrong encoded size for {item.name}")
                        fp.write(data)
            else:
                values = _orient(quantizer.to_f32(db, item.source), item, np)
                fp.write(quantizer.encode(np.ascontiguousarray(values, dtype=np.float32),
                                          item.qtype))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            print(f"[{index + 1}/{len(plan)}] {item.name}: "
                  f"{item.nbytes / (1 << 20):.1f} MiB, "
                  f"{time.monotonic() - started:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="dir with mtp shards + cropped model.safetensors.index.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="main")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quants-library", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"libds4quants.{'dylib' if sys.platform == 'darwin' else 'so'}"))
    args = ap.parse_args()
    if not 1 <= args.threads <= 32:
        ap.error("threads must be between 1 and 32")

    db = SourceDB(args.source, index_validator=lambda _: None,
                  scale_validator=validate)
    try:
        plan = build_plan(db, args.recipe)
        records = kv_records(args.recipe)
        offset = 0
        for item in plan:
            item.offset = offset
            item.nbytes = qtype_nbytes(item.qtype, item.shape)
            offset += align(item.nbytes, GGUF_ALIGNMENT)
        total = offset
        if args.dry_run:
            print(f"{len(plan)} tensors, {total / 2**30:.3f} GiB, recipe={args.recipe}")
            for item in plan:
                print("%-46s %-8s %-22s %10.2f MiB" % (
                    item.name, QTYPE_NAMES[item.qtype], item.shape,
                    item.nbytes / 2**20))
            return
        print(f"plan: {len(plan)} tensors, {total / 2**30:.3f} GiB, recipe={args.recipe}")
        write_gguf(args, plan, records, db)
        print("wrote %s (%.3f GiB)" % (args.out, os.path.getsize(args.out) / 2**30))
    finally:
        db.close()


if __name__ == "__main__":
    main()