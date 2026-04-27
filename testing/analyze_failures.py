"""Post-hoc analyses on the cached .npz from inspect_with_ppm.py.

Doesn't need a model or pod — operates entirely on the cached per-token NLLs.
"""
import argparse
import math
from pathlib import Path
from collections import Counter

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = sp.vocab_size()

    print(f"[load] {args.npz}")
    d = np.load(args.npz, allow_pickle=False)
    tokens = d["tokens"]
    nll_nats = d["nll_nats"]      # (n-1,)
    top_ids = d["top_ids"]        # (n-1, 5)
    top_logp = d["top_logp"]      # (n-1, 5)
    n_pred = nll_nats.shape[0]

    NLL_bits = nll_nats.astype(np.float64) / math.log(2.0)
    print(f"[load] {n_pred} positions, mean NLL = {NLL_bits.mean():.4f} bits")

    # Top-1 prediction confidence per position
    top1_logp = top_logp[:, 0].astype(np.float64)
    top1_p = np.exp(top1_logp)
    top1_id = top_ids[:, 0]

    # Was top-1 correct?
    actual_id = tokens[1:n_pred+1]
    top1_correct = (top1_id == actual_id)
    top1_acc = top1_correct.mean()
    print(f"[acc] top-1 accuracy = {top1_acc*100:.2f}%")

    out_lines = []
    out_lines.append(f"# Post-hoc analyses on 047B val NLLs")
    out_lines.append(f"")
    out_lines.append(f"Source: `{args.npz}`. {n_pred} scored positions.")
    out_lines.append(f"")

    # ===== A) Confidence vs correctness =====
    out_lines.append(f"## (A) Top-1 confidence calibration")
    out_lines.append(f"")
    out_lines.append(f"Does the model's softmax probability match its accuracy? Group positions by top-1 confidence, see how often top-1 is correct.")
    out_lines.append(f"")
    out_lines.append(f"| Top-1 confidence bucket | count | % positions | top-1 accuracy | mean NLL (bits) |")
    out_lines.append(f"|---|---:|---:|---:|---:|")
    conf_buckets = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.001)]
    for lo, hi in conf_buckets:
        mask = (top1_p >= lo) & (top1_p < hi)
        if mask.sum() == 0:
            continue
        c = int(mask.sum())
        pct = 100*c/n_pred
        acc = top1_correct[mask].mean()*100
        mean_nll = NLL_bits[mask].mean()
        out_lines.append(f"| {lo:.2f}–{hi:.2f} | {c:,} | {pct:.1f}% | {acc:.1f}% | {mean_nll:.2f} |")
    out_lines.append(f"")

    # ===== B) Overconfident-wrong positions =====
    out_lines.append(f"## (B) Overconfident-wrong: model said `p_top1 > 0.9` but was wrong")
    out_lines.append(f"")
    overconfident_wrong = (top1_p > 0.9) & (~top1_correct)
    n_oc = int(overconfident_wrong.sum())
    total_oc_bits = float(NLL_bits[overconfident_wrong].sum())
    pct_oc_bits = 100 * total_oc_bits / NLL_bits.sum()
    out_lines.append(f"- **{n_oc:,} positions** ({100*n_oc/n_pred:.2f}% of total)")
    out_lines.append(f"- Mean NLL on these: **{NLL_bits[overconfident_wrong].mean():.2f} bits/token**")
    out_lines.append(f"- Total bits in these positions: {total_oc_bits:,.0f}")
    out_lines.append(f"- **{pct_oc_bits:.1f}% of total NN val_bpb comes from overconfident-wrong positions** ← surprising failure mode")
    out_lines.append(f"")

    # Top 10 overconfident-wrong by NLL
    oc_indices = np.where(overconfident_wrong)[0]
    oc_sorted = oc_indices[np.argsort(-NLL_bits[oc_indices])][:15]
    out_lines.append(f"### Top-15 overconfident-wrong (NLL = bits 'wasted' on a wrong sure thing)")
    out_lines.append(f"")
    out_lines.append(f"| pos | NLL | top-1 prediction (p) | actual | left context (last 50 chars) |")
    out_lines.append(f"|---|---:|---|---|---|")
    pieces = [sp.id_to_piece(int(t)) if 0 <= int(t) < vocab else "<oor>" for t in tokens]
    def ctx_str(pos, k=50):
        end = pos + 1
        start = max(0, end - 25)
        s = "".join(pieces[i].replace("▁", " ") for i in range(start, end))
        return s.replace("\n", "↵")[-k:]
    for pos in oc_sorted:
        pos = int(pos)
        actual = pieces[pos+1]
        top1 = sp.id_to_piece(int(top1_id[pos])) if int(top1_id[pos]) < vocab else "<oor>"
        out_lines.append(f"| {pos} | {NLL_bits[pos]:.2f} | `{top1!r}` ({top1_p[pos]:.3f}) | `{actual!r}` | `{ctx_str(pos)}` |")
    out_lines.append(f"")

    # ===== C) Confident-and-right (the easy bytes) =====
    confident_right = (top1_p > 0.9) & top1_correct
    n_cr = int(confident_right.sum())
    bits_cr = float(NLL_bits[confident_right].sum())
    out_lines.append(f"## (C) Confident-and-right (the 'free' bytes)")
    out_lines.append(f"")
    out_lines.append(f"- **{n_cr:,} positions** ({100*n_cr/n_pred:.2f}% of total) — model knew exactly what was coming")
    out_lines.append(f"- Mean NLL on these: **{NLL_bits[confident_right].mean():.4f} bits/token** (effectively free)")
    out_lines.append(f"- Total bits: {bits_cr:,.0f} ({100*bits_cr/NLL_bits.sum():.1f}% of total)")
    out_lines.append(f"")

    # ===== D) Overall confidence-correctness joint =====
    out_lines.append(f"## (D) Confidence × correctness joint distribution")
    out_lines.append(f"")
    out_lines.append(f"Cross-tabulating where the bits/positions actually live.")
    out_lines.append(f"")
    out_lines.append(f"| | confident wrong | confident right | unsure wrong | unsure right |")
    out_lines.append(f"|---|---:|---:|---:|---:|")
    cw = (top1_p > 0.9) & ~top1_correct
    cr = (top1_p > 0.9) & top1_correct
    uw = (top1_p <= 0.9) & ~top1_correct
    ur = (top1_p <= 0.9) & top1_correct
    masks = [cw, cr, uw, ur]
    labels = ["confident wrong", "confident right", "unsure wrong", "unsure right"]
    out_lines.append(f"| count | {int(cw.sum()):,} | {int(cr.sum()):,} | {int(uw.sum()):,} | {int(ur.sum()):,} |")
    out_lines.append(f"| % positions | {100*cw.mean():.1f}% | {100*cr.mean():.1f}% | {100*uw.mean():.1f}% | {100*ur.mean():.1f}% |")
    out_lines.append(f"| mean NLL (bits) | {NLL_bits[cw].mean():.2f} | {NLL_bits[cr].mean():.4f} | {NLL_bits[uw].mean():.2f} | {NLL_bits[ur].mean():.2f} |")
    out_lines.append(f"| total bits | {NLL_bits[cw].sum():,.0f} | {NLL_bits[cr].sum():,.0f} | {NLL_bits[uw].sum():,.0f} | {NLL_bits[ur].sum():,.0f} |")
    bits_total = float(NLL_bits.sum())
    out_lines.append(f"| % of val_bpb | {100*NLL_bits[cw].sum()/bits_total:.1f}% | {100*NLL_bits[cr].sum()/bits_total:.1f}% | {100*NLL_bits[uw].sum()/bits_total:.1f}% | {100*NLL_bits[ur].sum()/bits_total:.1f}% |")
    out_lines.append(f"")

    # ===== E) Doc-boundary first-byte analysis =====
    out_lines.append(f"## (E) Document boundaries (first byte after BOS)")
    out_lines.append(f"")
    BOS = 1
    bos_positions = np.where(tokens == BOS)[0]
    # Position right after BOS in our predictions array: bos_pos -> predicts token at bos_pos+1
    # The prediction at output index bos_pos is for tokens[bos_pos+1]; needs context from input at bos_pos which IS BOS.
    # In our array, NLL_bits[i] = surprise at predicting tokens[i+1] given tokens[<=i].
    bos_first_byte_idx = bos_positions[(bos_positions >= 0) & (bos_positions < n_pred)]
    if len(bos_first_byte_idx) > 0:
        bb_nll = NLL_bits[bos_first_byte_idx]
        out_lines.append(f"- **{len(bos_first_byte_idx):,} positions** are the first byte of a new document")
        out_lines.append(f"- Mean NLL: **{bb_nll.mean():.2f} bits** vs overall mean {NLL_bits.mean():.2f} bits")
        out_lines.append(f"- {100*len(bos_first_byte_idx)/n_pred:.2f}% of positions, contribute {100*bb_nll.sum()/bits_total:.2f}% of val_bpb")
        out_lines.append(f"")

    # ===== F) Where does the loss really come from? =====
    out_lines.append(f"## (F) Cumulative loss contribution (sorted positions)")
    out_lines.append(f"")
    sorted_nll = np.sort(NLL_bits)[::-1]
    cum = sorted_nll.cumsum() / NLL_bits.sum()
    pcts = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    out_lines.append(f"What fraction of total bits comes from the worst N% of positions?")
    out_lines.append(f"")
    out_lines.append(f"| Worst N% of positions | bits accounted for |")
    out_lines.append(f"|---|---:|")
    for p in pcts:
        idx = int(p * n_pred)
        out_lines.append(f"| top {p*100:.0f}% | {cum[idx-1]*100:.1f}% of val_bpb |")
    out_lines.append(f"")

    Path(args.out).write_text("\n".join(out_lines))
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
