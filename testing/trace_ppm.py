"""Trace PPM-D scoring on a slice of val byte-by-byte.

Picks illustrative examples of PPM big-wins, neutral, and big-losses.
Slow Python PPM (~30s per 100K bytes), but sufficient for finding examples.
"""
import argparse, math, struct
from collections import defaultdict
from pathlib import Path
import numpy as np
import sentencepiece as spm

BOS_ID = 1
HEADER = 1024

class PPMD:
    """Order-4 PPM-D byte scorer with simple escape-D and full backoff."""
    def __init__(self, order=4):
        self.order = order
        self.tables = [defaultdict(lambda: defaultdict(int)) for _ in range(order + 1)]
        self.totals = [defaultdict(int) for _ in range(order + 1)]
        self.uniques = [defaultdict(int) for _ in range(order + 1)]
        self.history = []

    def score_byte(self, b):
        """Return (p_byte, top_byte, top_p, max_prob_for_gate)."""
        # Try contexts of decreasing length
        ctx = bytes(self.history[-self.order:]) if self.history else b""
        log_p = None
        max_p = 0.0
        top_byte = -1
        top_p = 0.0
        escape_log = 0.0
        for k in range(min(self.order, len(ctx)), -1, -1):
            sub_ctx = ctx[-k:] if k > 0 else b""
            tbl = self.tables[k][sub_ctx]
            tot = self.totals[k][sub_ctx]
            uni = self.uniques[k][sub_ctx]
            if not tbl:
                continue
            den = tot + uni  # PPM-D-style denominator (escape gets uni mass)
            if den <= 0:
                continue
            # Capture max prob at deepest matching context (for gate)
            if max_p == 0.0:
                for bb, c in tbl.items():
                    p = c / den
                    if p > max_p:
                        max_p = p
                        top_byte = bb
                        top_p = p
            count = tbl.get(b, 0)
            if count > 0:
                # Found at this order
                log_p = escape_log + math.log(count / den)
                break
            else:
                # Escape: probability uni/den, back off
                if uni > 0:
                    escape_log += math.log(uni / den)
        if log_p is None:
            # Fall back to uniform 1/256
            log_p = escape_log + math.log(1/256)
        return log_p, top_byte, top_p, max_p

    def update(self, b):
        for k in range(self.order + 1):
            sub_ctx = bytes(self.history[-k:]) if k > 0 else b""
            tbl = self.tables[k][sub_ctx]
            if b not in tbl:
                self.uniques[k][sub_ctx] += 1
            tbl[b] += 1
            self.totals[k][sub_ctx] += 1
        self.history.append(b)
        if len(self.history) > self.order:
            self.history.pop(0)

def reconstruct_byte_stream(tokens, vocab, sp):
    """Reconstruct the actual byte stream by decoding tokens through SP."""
    pieces = []
    for tid in tokens:
        tid = int(tid)
        if tid == BOS_ID or tid >= vocab:
            pieces.append("")
            continue
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            pieces.append("")
            continue
        if sp.is_byte(tid):
            piece = sp.id_to_piece(tid)
            pieces.append(chr(int(piece[3:-1], 16)) if len(piece) >= 6 else piece)
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            pieces.append(" " + piece[1:])
        else:
            pieces.append(piece)
    return pieces

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val_tok", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin")
    ap.add_argument("--n_tokens", type=int, default=50000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = sp.vocab_size()

    print(f"[load] npz {args.npz}")
    d = np.load(args.npz, allow_pickle=False)
    tokens = d["tokens"][:args.n_tokens].astype(np.int32)
    nll_nats = d["nll_nats"][:args.n_tokens-1].astype(np.float32)
    top_ids = d["top_ids"][:args.n_tokens-1]      # (n-1, 5)
    top_logp = d["top_logp"][:args.n_tokens-1]    # (n-1, 5)

    pieces = reconstruct_byte_stream(tokens, vocab, sp)
    # Convert pieces to flat byte stream + per-byte token-NLL contribution
    byte_stream = bytearray()
    byte_to_token = []  # for each byte, which token index produced it
    byte_in_token_idx = []  # position within the token's bytes
    for i, p in enumerate(pieces):
        encoded = p.encode("utf-8")
        for j, b in enumerate(encoded):
            byte_stream.append(b)
            byte_to_token.append(i)
            byte_in_token_idx.append(j)

    print(f"[stream] {len(byte_stream)} bytes from {len(pieces)} tokens")

    # Score byte-by-byte with PPM
    ppm = PPMD(order=4)
    wins = []   # (pos, info)
    losses = []
    neutral = []
    LOG2 = math.log(2.0)
    PPM_LAMBDA_HI = 0.9
    PPM_LAMBDA_LO = 0.05
    PPM_THR = 0.9

    for byte_pos, b in enumerate(byte_stream):
        # Score
        log_p_ppm, top_byte, top_p, max_prob = ppm.score_byte(b)
        # Compute NN per-byte logp from token-level NLL spread uniformly
        tok_idx = byte_to_token[byte_pos]
        if tok_idx == 0 or tok_idx > len(nll_nats):
            ppm.update(b)
            continue
        # nll_nats[tok_idx - 1] = NLL of token at position tok_idx (predicted from token_idx-1)
        # Spread uniformly over the token's bytes
        token_n_bytes = sum(1 for x in byte_to_token if x == tok_idx)
        if token_n_bytes == 0:
            ppm.update(b)
            continue
        nn_logp_byte = -float(nll_nats[tok_idx - 1]) / token_n_bytes
        # Mix logic
        if max_prob >= PPM_THR:
            lam = PPM_LAMBDA_LO  # NN weight 5%, PPM 95%
            regime = "HIGH"
        else:
            lam = PPM_LAMBDA_HI  # NN weight 90%, PPM 10%
            regime = "LOW"
        # Mix in probability space
        p_nn = math.exp(nn_logp_byte)
        p_ppm = math.exp(log_p_ppm)
        p_mix = lam * p_nn + (1 - lam) * p_ppm
        nll_nn_bits = -nn_logp_byte / LOG2
        nll_ppm_bits = -log_p_ppm / LOG2
        nll_mix_bits = -math.log(max(p_mix, 1e-300)) / LOG2
        delta = nll_nn_bits - nll_mix_bits  # positive = PPM helped

        # Get context (last 40 bytes)
        ctx_start = max(0, byte_pos - 40)
        ctx = bytes(byte_stream[ctx_start:byte_pos]).decode("utf-8", errors="replace")
        actual_chr = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"

        # NN top-1 token at this position
        nn_top_id = int(top_ids[tok_idx - 1][0])
        nn_top_p = math.exp(float(top_logp[tok_idx - 1][0]))
        nn_top_piece = sp.id_to_piece(nn_top_id) if 0 <= nn_top_id < vocab else "<oor>"

        info = {
            "pos": byte_pos,
            "ctx": ctx,
            "actual": actual_chr,
            "nll_nn": nll_nn_bits,
            "nll_ppm": nll_ppm_bits,
            "nll_mix": nll_mix_bits,
            "delta": delta,
            "regime": regime,
            "max_prob": max_prob,
            "top_byte_pred": chr(top_byte) if top_byte >= 32 and top_byte < 127 else (f"\\x{top_byte:02x}" if top_byte >= 0 else "?"),
            "top_p": top_p,
            "nn_top_piece": nn_top_piece,
            "nn_top_p": nn_top_p,
            "actual_token_piece": pieces[tok_idx],
        }
        # Categorize
        if regime == "HIGH" and delta > 1.0:
            wins.append(info)
        elif regime == "HIGH" and delta < -2.0:
            losses.append(info)
        elif regime == "HIGH":
            neutral.append(info)

        ppm.update(b)

    print(f"[done] processed {len(byte_stream)} bytes")
    print(f"  HIGH-regime wins (Δ>1 bit):  {len(wins)}")
    print(f"  HIGH-regime losses (Δ<-2):    {len(losses)}")
    print(f"  HIGH-regime neutral:          {len(neutral)}")

    out = []
    out.append(f"# PPM byte-trace on first ~{len(byte_stream)} bytes of val\n")
    out.append(f"## Setup")
    out.append(f"- Order=4, λ_hi=0.9 (NN dominant when PPM unsure), λ_lo=0.05 (PPM dominant when confident)")
    out.append(f"- Gate threshold: PPM max-prob ≥ 0.9 → HIGH regime\n")

    out.append(f"## Big PPM wins — PPM correct, NN spread or wrong\n")
    wins.sort(key=lambda x: -x["delta"])
    out.append(f"| pos | context (last 40 chars) | NN top-1 token (prob) | PPM said (prob) | actual byte | NN bits | PPM bits | mix bits | **saved** |")
    out.append(f"|---|---|---|---|---|---:|---:|---:|---:|")
    for info in wins[:25]:
        ctx = info["ctx"].replace("\n", "↵").replace("|", "\\|")
        nn_piece_disp = info['nn_top_piece'].replace("▁", "_").replace("\n", "↵")
        out.append(f"| {info['pos']} | `{ctx}` | `{nn_piece_disp}` ({info['nn_top_p']:.2f}) | `{info['top_byte_pred']}` ({info['top_p']:.2f}) | `{info['actual']}` | {info['nll_nn']:.2f} | {info['nll_ppm']:.2f} | {info['nll_mix']:.2f} | **{info['delta']:+.2f}** |")

    out.append(f"\n## PPM losses (HIGH regime, hurt by >2 bits)\n")
    losses.sort(key=lambda x: x["delta"])
    out.append(f"| pos | NN bits | PPM bits | mix bits | lost | PPM thought (p) | actual | ← context |")
    out.append(f"|---|---:|---:|---:|---:|---|---|---|")
    for info in losses[:15]:
        ctx = info["ctx"].replace("\n", "↵").replace("|", "\\|")
        out.append(f"| {info['pos']} | {info['nll_nn']:.2f} | {info['nll_ppm']:.2f} | {info['nll_mix']:.2f} | **{info['delta']:+.2f}** | `{info['top_byte_pred']}` ({info['top_p']:.2f}) | `{info['actual']}` | `{ctx}` |")

    out.append(f"\n## Aggregate per regime\n")
    cls = {"HIGH wins": wins, "HIGH neutral": neutral, "HIGH losses": losses}
    out.append(f"| Regime | count | mean NN bits | mean mix bits | mean Δ |")
    out.append(f"|---|---:|---:|---:|---:|")
    for label, lst in cls.items():
        if not lst: continue
        n = len(lst)
        mean_nn = sum(x["nll_nn"] for x in lst) / n
        mean_mix = sum(x["nll_mix"] for x in lst) / n
        mean_d = sum(x["delta"] for x in lst) / n
        out.append(f"| {label} | {n} | {mean_nn:.2f} | {mean_mix:.2f} | {mean_d:+.2f} |")

    Path(args.out).write_text("\n".join(out))
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
