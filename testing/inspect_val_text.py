"""Quick: decode the first ~600 bytes of val data and print with token boundaries.

Goal: let us SEE what our model is trying to predict, before adding NN inference.
Highlights surface structure (URLs, numerals, code, formatting) where PPM would
shine vs semantic structure where NN dominates.

Run on pod (CPU, no GPU needed):
    python3 /workspace/parameter-golf/tmp_exec/inspect_val_text.py
"""
import sys
import struct
import sentencepiece as spm
from pathlib import Path

# Paths on volume (NE-1)
DATA_DIR = Path("/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved")
VAL_TOK = DATA_DIR / "fineweb_val_000000.bin"
VAL_BYTES = DATA_DIR / "fineweb_val_bytes_000000.bin"
TOKENIZER = "/workspace/pg-050B-lora-787f672/records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"

BOS_ID = 1
N_TOKENS = 200  # how many tokens to inspect

def main():
    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER)
    print(f"vocab size: {sp.vocab_size()}")
    print()

    # Read first N tokens from val
    # Format: header (1024 bytes per nanogpt convention or compact)
    raw = VAL_TOK.read_bytes()
    # Try uint16
    n_uint16 = len(raw) // 2
    tokens = list(struct.unpack(f"<{n_uint16}H", raw))[:N_TOKENS]
    print(f"loaded {n_uint16} val tokens, showing first {N_TOKENS}")
    print()

    # Show token-by-token
    print("=" * 78)
    print("Per-token decode (token_id | piece | byte string)")
    print("=" * 78)
    for i, tid in enumerate(tokens):
        if tid >= sp.vocab_size():
            print(f"  [{i:3d}] {tid:5d}  <out-of-range>")
            continue
        piece = sp.id_to_piece(tid)
        # Mark control / boundary
        flags = []
        if sp.is_control(tid): flags.append("CTRL")
        if sp.is_unknown(tid): flags.append("UNK")
        if sp.is_byte(tid): flags.append("BYTE")
        if tid == BOS_ID: flags.append("BOS")
        flag_str = f" [{','.join(flags)}]" if flags else ""

        # Decode to bytes
        if piece.startswith("▁"):  # SP whitespace marker
            raw_bytes = " " + piece[1:]
        else:
            raw_bytes = piece
        raw_str = repr(raw_bytes)
        print(f"  [{i:3d}] {tid:5d}  {piece!r:30s} -> {raw_str}{flag_str}")

    # Reassemble as text
    print()
    print("=" * 78)
    print("Reassembled text (first ~600 bytes):")
    print("=" * 78)
    text_pieces = []
    for tid in tokens:
        if tid == BOS_ID:
            text_pieces.append("|BOS|")
            continue
        if tid >= sp.vocab_size() or sp.is_control(tid):
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            text_pieces.append(" " + piece[1:])
        else:
            text_pieces.append(piece)
    text = "".join(text_pieces)
    print(text[:1200])

    # Quick surface-structure scan: which tokens look like URLs/numbers/code?
    print()
    print("=" * 78)
    print("Surface-structure candidates (PPM would shine here):")
    print("=" * 78)
    surface_count = 0
    for i, tid in enumerate(tokens):
        if tid >= sp.vocab_size() or sp.is_control(tid) or tid == BOS_ID:
            continue
        piece = sp.id_to_piece(tid)
        decoded = piece[1:] if piece.startswith("▁") else piece
        # Heuristics
        is_url = any(s in decoded for s in ["http", "www.", ".com", ".org", ".net", "://"])
        is_num = decoded and decoded.replace(".", "").replace(",", "").isdigit()
        is_code = any(c in decoded for c in ["{", "}", "[", "]", "()", ";"])
        is_short_repeat = len(decoded) <= 3 and not decoded.isalpha()
        if is_url or is_num or is_code or is_short_repeat:
            surface_count += 1
            kind = "URL" if is_url else "NUM" if is_num else "CODE" if is_code else "SHORT"
            print(f"  [{i:3d}] {kind:6s} {decoded!r}")

    print()
    print(f"surface-structure tokens: {surface_count} / {N_TOKENS} ({100*surface_count/N_TOKENS:.1f}%)")

if __name__ == "__main__":
    main()
