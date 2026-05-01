"""Check canonical bytes (LUT-derived) vs sidecar file bytes for full val."""

import os, sys, math, importlib.util
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/ngram_analysis")
from proper_ppm_mixer_rigorous import load_train_gpt

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True)
    ap.add_argument("--n_tokens", type=int, default=47_000_000)
    args = ap.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.Load(args.tokenizer)
    V = 8192
    has_ls = np.zeros(V, dtype=np.bool_); is_bdy = np.zeros(V, dtype=np.bool_)
    nb_lut = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_bdy[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_ls[tid] = True; piece = piece[1:]
            nb_lut[tid] = len(piece.encode("utf-8"))
        except: pass

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1
    targets = arr[1:].copy()
    prevs = arr[:-1].copy()

    # Canonical bytes:
    # - byte-content token: nb + (1 if include_space else 0)
    # - control token: 0
    is_byte_content = ~is_bdy[targets]
    nb_per = nb_lut[targets]
    has_ls_per = has_ls[targets]
    prev_is_bdy = (prevs < 0) | is_bdy[prevs]
    include_space = has_ls_per & ~prev_is_bdy
    canonical_per = nb_per * is_byte_content + include_space.astype(np.int32)
    canonical_total = int(canonical_per.sum())

    # Also: pieces-only canonical (excludes include_space prepend)
    canonical_pieces_only = int((nb_per * is_byte_content).sum())

    # Sidecar file
    sidecar = np.fromfile(args.val_bytes_bin, dtype=np.uint16)
    OFFSET = 906
    sb_aligned = sidecar[OFFSET+1:OFFSET+1+N].astype(np.int64)
    sidecar_total = int(sb_aligned.sum())

    n_include_space = int(include_space.sum())
    n_byte_content = int(is_byte_content.sum())
    n_multibyte = int(((nb_per >= 2) & is_byte_content).sum() | ((nb_per >= 1) & include_space).sum())

    print(f"N positions:                    {N:,}")
    print(f"Byte-content tokens:            {n_byte_content:,}")
    print(f"Include-space tokens:           {n_include_space:,}")
    print(f"  fraction include-space:       {100*n_include_space/N:.1f}%")
    print(f"")
    print(f"Sidecar file bytes:             {sidecar_total:,}")
    print(f"Canonical (pieces only, no LS): {canonical_pieces_only:,}")
    print(f"Canonical (with LS prepend):    {canonical_total:,}")
    print(f"")
    print(f"Diff canonical(LS) − sidecar:   {canonical_total - sidecar_total:,}  ({(canonical_total-sidecar_total)/sidecar_total*100:.2f}%)")
    print(f"Diff sidecar − canonical(no LS): {sidecar_total - canonical_pieces_only:,}")
    print(f"")
    print(f"Per-position averages:")
    print(f"  sidecar_bytes/pos:         {sidecar_total/N:.4f}")
    print(f"  canonical_pieces_only/pos: {canonical_pieces_only/N:.4f}")
    print(f"  canonical_with_LS/pos:     {canonical_total/N:.4f}")

if __name__ == "__main__":
    main()
