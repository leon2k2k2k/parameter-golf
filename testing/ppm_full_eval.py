"""Full per-byte PPM evaluation.

Calls the modified C scorer's `ppm_score_dump` function which writes per-byte
arrays for mix_nll, ppm_nll, nn_nll, gate confidence, and actual byte. Then
runs authoritative analyses on the dumped data:

  - Quartile / decile breakdown (NN bits → PPM rescue rate)
  - Per-category contribution (URL/PROSE/etc.)
  - Top-N catastrophic positions post-PPM
  - Gate-fire rate vs NN difficulty
  - Per-doc-segment PPM gain (split by BOS positions)

Inputs: cached .npz from inspect_with_ppm.py + tokenizer + ppm_scorer.c
Output: dumped per-byte arrays (.npz) + aggregated markdown report.
"""
import argparse, ctypes, math, os, struct, subprocess, tempfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import sentencepiece as spm

PPM_C_SRC = Path(__file__).parent / "ppm_scorer.c"
BOS_ID = 1
HEADER = 1024

def build_token_bytes_lut(sp, vocab_size):
    sz = max(int(sp.vocab_size()), vocab_size)
    bytestrs = [b""] * sz
    has_space = np.zeros(sz, dtype=np.uint8)
    is_boundary = np.zeros(sz, dtype=np.uint8)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            is_boundary[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if sp.is_byte(tid):
            bytestrs[tid] = bytes([int(piece[3:-1], 16)])
            continue
        if piece.startswith("▁"):
            has_space[tid] = 1
            piece = piece[1:]
        bytestrs[tid] = piece.encode("utf-8")
    flat = b"".join(bytestrs)
    lens = np.array([len(b) for b in bytestrs], dtype=np.int32)
    offs = np.zeros(sz, dtype=np.int32)
    offs[1:] = np.cumsum(lens[:-1])
    return np.frombuffer(flat, dtype=np.uint8).copy(), offs, lens, has_space, is_boundary

def compile_lib():
    so = Path(tempfile.gettempdir()) / "ppm_scorer_with_dump.so"
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
           "-o", str(so), str(PPM_C_SRC), "-lm"]
    print(f"[compile] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    lib.ppm_score_dump.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
        # dump arrays
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ppm_score_dump.restype = ctypes.c_int
    return lib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dump", required=True, help="output .npz with per-byte arrays")
    ap.add_argument("--out_md", required=True, help="output markdown report")
    ap.add_argument("--ppm_order", type=int, default=4)
    ap.add_argument("--lambda_hi", type=float, default=0.9)
    ap.add_argument("--lambda_lo", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--log_cache_size", type=int, default=1048576)
    args = ap.parse_args()

    print(f"[load] tokenizer {args.tokenizer}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = sp.vocab_size()

    print(f"[load] cached NLLs from {args.npz}")
    d = np.load(args.npz, allow_pickle=False)
    tokens_full = d["tokens"]
    nll_nats_full = d["nll_nats"]
    n_tok = len(nll_nats_full)
    print(f"[load] {n_tok} tokens scored")

    # Build call args (matches inspect_with_ppm.py)
    print(f"[setup] building byte LUT")
    flat, offs, lens, has_space, is_boundary = build_token_bytes_lut(sp, vocab)
    target_ids = tokens_full[1:n_tok+1].astype(np.int64)
    prev_ids = tokens_full[0:n_tok].astype(np.int64)
    nll_nats = nll_nats_full.astype(np.float64)

    # Estimate max bytes (conservative): sum of (lens[t] + 1) for each target token
    max_bytes = int((lens[target_ids].astype(np.int64) + 1).sum()) + 1024
    print(f"[setup] allocating dump arrays for up to {max_bytes:,} bytes")

    dump_mix = np.zeros(max_bytes, dtype=np.float32)
    dump_ppm = np.zeros(max_bytes, dtype=np.float32)
    dump_nn = np.zeros(max_bytes, dtype=np.float32)
    dump_conf = np.zeros(max_bytes, dtype=np.float32)
    dump_gate_hi = np.zeros(max_bytes, dtype=np.uint8)
    dump_byte = np.zeros(max_bytes, dtype=np.uint8)
    dump_n = np.zeros(1, dtype=np.uint64)
    out_agg = np.zeros(6, dtype=np.float64)

    lib = compile_lib()
    print(f"[run] ppm_score_dump (single-threaded — expect ~16 min on full val)")
    t0 = time.time()
    rc = lib.ppm_score_dump(
        target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        nll_nats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(n_tok),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        has_space.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        is_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(vocab),
        ctypes.c_int(args.ppm_order),
        ctypes.c_double(args.lambda_hi),
        ctypes.c_double(args.lambda_lo),
        ctypes.c_double(args.threshold),
        ctypes.c_uint32(args.log_cache_size),
        out_agg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dump_mix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dump_ppm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dump_nn.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dump_conf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dump_gate_hi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dump_byte.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dump_n.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
    )
    elapsed = time.time() - t0
    print(f"[run] returned rc={rc} in {elapsed:.1f}s")
    if rc != 0:
        print(f"ERROR: rc={rc}"); return

    n_bytes = int(dump_n[0])
    mix = dump_mix[:n_bytes]
    ppm = dump_ppm[:n_bytes]
    nn = dump_nn[:n_bytes]
    conf = dump_conf[:n_bytes]
    gate_hi = dump_gate_hi[:n_bytes]
    by = dump_byte[:n_bytes]

    print(f"\n=== Aggregate (from C scorer) ===")
    print(f"  mix_bpb     = {out_agg[0]:.5f}")
    print(f"  ppm_only    = {out_agg[1]:.5f}")
    print(f"  nn_byte_bpb = {out_agg[2]:.5f}")
    print(f"  bytes       = {n_bytes:,}")
    print(f"  gate_high   = {out_agg[5]:.5f}")
    print(f"  total NN bits  = {nn.sum()/math.log(2):.0f}")
    print(f"  total mix bits = {mix.sum()/math.log(2):.0f}")
    print(f"  PPM saved      = {(nn.sum()-mix.sum())/math.log(2):.0f} bits")

    # Save dump
    print(f"\n[save] {args.out_dump}")
    np.savez_compressed(args.out_dump,
        mix_nats=mix, ppm_nats=ppm, nn_nats=nn,
        conf=conf, gate_hi=gate_hi, byte=by,
        agg_out=out_agg)

    # ---------------- analyses ----------------
    LOG2 = math.log(2.0)
    nn_bits = nn / LOG2
    mix_bits = mix / LOG2
    ppm_bits = ppm / LOG2
    delta = nn_bits - mix_bits  # positive = PPM helped

    out_lines = []
    out_lines.append(f"# PPM full-val per-byte evaluation\n")
    out_lines.append(f"**N bytes scored**: {n_bytes:,}")
    out_lines.append(f"**Aggregate**: NN={out_agg[2]:.5f} bpb, PPM-only={out_agg[1]:.5f} bpb, mix={out_agg[0]:.5f} bpb. PPM gain = **{out_agg[0] - out_agg[2]:+.5f} bpb**")
    out_lines.append(f"**Gate-fire rate**: {out_agg[5]:.4f} ({100*out_agg[5]:.1f}%)")
    out_lines.append(f"")

    # ===== Quartile breakdown =====
    out_lines.append(f"## Quartile breakdown of NN difficulty\n")
    out_lines.append(f"Sort all {n_bytes:,} bytes by NN bits descending. Split into quartiles. For each quartile, see what fraction of NN's loss PPM rescues.\n")
    sorted_idx = np.argsort(-nn_bits)
    q_size = n_bytes // 4
    out_lines.append(f"| Quartile | bytes | NN total bits | Mix total bits | PPM saved | rescue rate | gate-fire % | mean NN bits/byte | mean mix bits/byte |")
    out_lines.append(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for q in range(4):
        lo, hi = q*q_size, (q+1)*q_size if q<3 else n_bytes
        idx = sorted_idx[lo:hi]
        nn_sum = float(nn_bits[idx].sum())
        mix_sum = float(mix_bits[idx].sum())
        gate_pct = 100*float(gate_hi[idx].mean())
        nn_per = float(nn_bits[idx].mean())
        mix_per = float(mix_bits[idx].mean())
        rescue = 100*(nn_sum - mix_sum)/max(nn_sum, 1e-9)
        label = ["Q1 (worst NN — top 25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (best NN — bottom 25%)"][q]
        out_lines.append(f"| {label} | {len(idx):,} | {nn_sum:,.0f} | {mix_sum:,.0f} | {nn_sum-mix_sum:+,.0f} | **{rescue:+.1f}%** | {gate_pct:.1f}% | {nn_per:.2f} | {mix_per:.2f} |")
    out_lines.append("")

    # ===== Decile breakdown =====
    out_lines.append(f"## Decile breakdown\n")
    out_lines.append(f"| Decile | bytes | NN bits | Mix bits | PPM saved | rescue rate | gate-fire % |")
    out_lines.append(f"|---|---:|---:|---:|---:|---:|---:|")
    decile = n_bytes // 10
    for d in range(10):
        lo, hi = d*decile, (d+1)*decile if d<9 else n_bytes
        idx = sorted_idx[lo:hi]
        nn_sum = float(nn_bits[idx].sum())
        mix_sum = float(mix_bits[idx].sum())
        gate_pct = 100*float(gate_hi[idx].mean())
        rescue = 100*(nn_sum - mix_sum)/max(nn_sum, 1e-9)
        out_lines.append(f"| D{d+1} ({d*10}-{(d+1)*10}%) | {len(idx):,} | {nn_sum:,.0f} | {mix_sum:,.0f} | {nn_sum-mix_sum:+,.0f} | **{rescue:+.1f}%** | {gate_pct:.1f}% |")
    out_lines.append("")

    # ===== Wins and losses =====
    out_lines.append(f"## Big wins and losses\n")
    n_wins = int((delta > 1.0).sum())
    n_losses = int((delta < -1.0).sum())
    n_huge_wins = int((delta > 5.0).sum())
    n_huge_losses = int((delta < -3.0).sum())
    out_lines.append(f"- Saved >1 bit/byte: **{n_wins:,}** positions ({100*n_wins/n_bytes:.2f}%)")
    out_lines.append(f"- Saved >5 bits/byte: **{n_huge_wins:,}** positions ({100*n_huge_wins/n_bytes:.3f}%)")
    out_lines.append(f"- Lost >1 bit/byte: **{n_losses:,}** positions ({100*n_losses/n_bytes:.2f}%)")
    out_lines.append(f"- Lost >3 bits/byte: **{n_huge_losses:,}** positions ({100*n_huge_losses/n_bytes:.3f}%)")
    win_bits = float(delta[delta > 0].sum())
    loss_bits = float(-delta[delta < 0].sum())
    out_lines.append(f"- Total bits saved by helping positions: {win_bits:,.0f}")
    out_lines.append(f"- Total bits lost by hurting positions: {loss_bits:,.0f}")
    out_lines.append(f"- **Net savings**: {win_bits - loss_bits:+,.0f} bits = {100*(win_bits-loss_bits)/(nn_bits.sum()):+.2f}% of NN val_bpb")
    out_lines.append("")

    # ===== Gate analysis =====
    out_lines.append(f"## Where PPM hijacks (HIGH regime, NN was right)\n")
    nn_correct_byby = (nn_bits < 1.0)  # NN gave high prob to actual byte
    hijacks = (gate_hi == 1) & nn_correct_byby & (delta < -1.0)
    n_hijacks = int(hijacks.sum())
    hijack_cost = float(-delta[hijacks].sum())
    out_lines.append(f"Gate fired HIGH AND NN bits < 1.0 (NN was confident-right) AND mix > NN by >1 bit:")
    out_lines.append(f"- **{n_hijacks:,}** hijack positions ({100*n_hijacks/n_bytes:.2f}% of bytes)")
    out_lines.append(f"- Total cost from hijacks: **{hijack_cost:,.0f}** bits ({100*hijack_cost/win_bits:.1f}% of total wins)")
    out_lines.append("")

    # ===== Where post-PPM loss concentrates =====
    out_lines.append(f"## Where post-PPM loss concentrates (sorted by mix bits desc)\n")
    sorted_mix = np.argsort(-mix_bits)
    cum = mix_bits[sorted_mix].cumsum() / mix_bits.sum()
    out_lines.append(f"| Worst N% of positions (post-PPM) | bits | % of total mix |")
    out_lines.append(f"|---|---:|---:|")
    for pct in [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(pct * n_bytes))
        bits_in_top = float(mix_bits[sorted_mix[:k]].sum())
        out_lines.append(f"| top {pct*100:.1f}% | {bits_in_top:,.0f} | {100*bits_in_top/mix_bits.sum():.1f}% |")
    out_lines.append("")

    # ===== Per-byte category split =====
    # Map each byte to a coarse category
    out_lines.append(f"## By byte category\n")
    is_alpha = (((by >= 0x41) & (by <= 0x5A)) | ((by >= 0x61) & (by <= 0x7A)))
    is_digit = (by >= 0x30) & (by <= 0x39)
    is_space = (by == 0x20)
    is_punct = ((by == 0x2E) | (by == 0x2C) | (by == 0x21) | (by == 0x3F) | (by == 0x3B) | (by == 0x3A))
    is_quote = ((by == 0x22) | (by == 0x27))
    is_other = ~(is_alpha | is_digit | is_space | is_punct | is_quote)
    cats = [("alpha", is_alpha), ("digit", is_digit), ("space", is_space), ("punct", is_punct), ("quote", is_quote), ("other", is_other)]
    out_lines.append(f"| category | bytes | % | NN bits/byte | mix bits/byte | PPM saved bits/byte | gate-fire % |")
    out_lines.append(f"|---|---:|---:|---:|---:|---:|---:|")
    for name, mask in cats:
        cnt = int(mask.sum())
        if cnt == 0: continue
        nn_per = float(nn_bits[mask].mean())
        mix_per = float(mix_bits[mask].mean())
        saved = nn_per - mix_per
        gate_pct = 100*float(gate_hi[mask].mean())
        out_lines.append(f"| {name} | {cnt:,} | {100*cnt/n_bytes:.1f}% | {nn_per:.3f} | {mix_per:.3f} | {saved:+.3f} | {gate_pct:.1f}% |")
    out_lines.append("")

    Path(args.out_md).write_text("\n".join(out_lines))
    print(f"[done] wrote {args.out_md}")
    print(f"\n=== KEY FINDINGS ===")
    print(f"PPM gain (full val, authoritative): {out_agg[0]-out_agg[2]:+.5f} bpb")
    print(f"Gate fire rate: {100*out_agg[5]:.2f}% of bytes")
    print(f"Q1 rescue rate (worst NN bytes): see report")

if __name__ == "__main__":
    main()
