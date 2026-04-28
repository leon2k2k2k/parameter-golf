"""Repack a brotli-serialized .int6.ptz into the per-group lrzip+brotli format.

Used to fix 060A's over-cap brotli artifact in-place without retraining.
The quantized weights inside the .ptz are byte-identical; this just swaps
the outer compression scheme. Pergroup typically saves ~250 KB on this
model class (lrzip-zpaq beats brotli on int8 weight blobs).

The pergroup helpers from #1855's train_gpt.py are inlined verbatim here
to avoid importing the full training module at runtime.

Usage: python repack_pergroup.py <input.int6.ptz> <output.int6.ptz> <num_layers>
"""
import sys
import io
import os
import struct
import tempfile
import subprocess
import collections

import brotli
import torch


_GROUP_ORDER = [
    "_tok_emb.weight.q",
    "attn.c_k.weight.q", "attn.c_q.weight.q",
    "attn.c_v.weight.q", "attn.proj.weight.q",
    "mlp.fc.weight.q", "mlp.proj.weight.q",
]
_SIMSORT_KEYS = {"_tok_emb.weight.q", "attn.c_q.weight.q", "mlp.fc.weight.q"}
_PACK_MAGIC = b"PGRP"


def _similarity_sort_l1(matrix):
    import numpy as _np
    n = matrix.shape[0]
    used = _np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True
    cur = matrix[0].astype(_np.float32)
    for _ in range(n - 1):
        dists = _np.sum(_np.abs(matrix[~used].astype(_np.float32) - cur), axis=1)
        unused = _np.where(~used)[0]
        best = unused[_np.argmin(dists)]
        order.append(best)
        used[best] = True
        cur = matrix[best].astype(_np.float32)
    return _np.array(order, dtype=_np.uint16)


def _lrzip_compress(data, tmpdir, label):
    inp = os.path.join(tmpdir, f"{label}.bin")
    out = f"{inp}.lrz"
    with open(inp, "wb") as f:
        f.write(data)
    subprocess.run(["lrzip", "-z", "-L", "9", "-o", out, inp], capture_output=True, check=True)
    with open(out, "rb") as f:
        result = f.read()
    os.remove(inp); os.remove(out)
    return result


def _pack_streams(streams):
    n = len(streams)
    hdr = _PACK_MAGIC + struct.pack("<I", n)
    for s in streams:
        hdr += struct.pack("<I", len(s))
    return hdr + b"".join(streams)


def _serialize_pergroup(quant_result, quant_meta, num_layers, tmpdir):
    import numpy as _np
    groups = collections.defaultdict(list)
    remainder = {}
    for name, t in sorted(quant_result.items()):
        if t.dtype != torch.int8:
            remainder[name] = t
            continue
        parts = name.split(".")
        routed = False
        if parts[0] == "blocks" and parts[1].isdigit():
            key = ".".join(parts[2:])
            if key in _GROUP_ORDER:
                groups[key].append((int(parts[1]), t))
                routed = True
        else:
            group_key = "_" + name
            if group_key in _GROUP_ORDER:
                groups[group_key] = [(0, t)]
                routed = True
        if not routed:
            remainder[name] = t

    streams = []
    all_perms = b""
    shape_manifest = {}

    for group_key in _GROUP_ORDER:
        if group_key not in groups:
            streams.append(b"")
            continue
        tensors = sorted(groups[group_key], key=lambda x: x[0])
        blob = b""
        grp_shapes = []
        for idx, t in tensors:
            arr = t.numpy()
            orig_shape = arr.shape
            if arr.ndim == 2:
                if group_key in _SIMSORT_KEYS:
                    order = _similarity_sort_l1(arr)
                    all_perms += order.tobytes()
                    arr = arr[order]
                arr = _np.ascontiguousarray(arr.T)
            blob += arr.tobytes()
            grp_shapes.append(orig_shape)
        shape_manifest[group_key] = grp_shapes
        compressed = _lrzip_compress(blob, tmpdir, group_key.replace(".", "_"))
        streams.append(compressed)

    remainder_buf = io.BytesIO()
    torch.save({"r": remainder, "m": quant_meta, "s": shape_manifest}, remainder_buf)
    streams.append(brotli.compress(remainder_buf.getvalue(), quality=11, lgwin=24))
    streams.append(brotli.compress(all_perms, quality=11) if all_perms else b"")

    return _pack_streams(streams)


def main():
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <input.int6.ptz> <output.int6.ptz> <num_layers>")
        sys.exit(1)
    in_path, out_path, num_layers = sys.argv[1], sys.argv[2], int(sys.argv[3])

    with open(in_path, "rb") as f:
        blob = f.read()
    print(f"[repack] read {len(blob)} bytes from {in_path}")

    raw = brotli.decompress(blob)
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    quant_result = state["w"]
    quant_meta = state["m"]
    print(f"[repack] decompressed: {len(quant_result)} quant tensors, "
          f"{len(quant_meta)} meta entries")

    tmpdir = tempfile.mkdtemp(prefix="pgrp_repack_")
    new_blob = _serialize_pergroup(quant_result, quant_meta, num_layers, tmpdir)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    with open(out_path, "wb") as f:
        f.write(new_blob)

    in_size = len(blob)
    out_size = len(new_blob)
    delta = out_size - in_size
    print(f"[repack] OK")
    print(f"[repack]   input  (brotli):    {in_size:>10} bytes")
    print(f"[repack]   output (pergroup):  {out_size:>10} bytes")
    print(f"[repack]   delta:              {delta:+10} bytes ({delta/1024:+.1f} KB)")
    print(f"[repack]   16MB cap headroom:  {16_000_000 - out_size:>10} bytes")


if __name__ == "__main__":
    main()
