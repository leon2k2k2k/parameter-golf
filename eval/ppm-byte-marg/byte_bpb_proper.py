"""Byte-based val_bpb scoring for an NN with PROPER marginalization at byte_0.

Computes per-token NLL decomposition:
    NLL_token = NLL_byte_0 + NLL_remainder    (chain rule, conserved)

where NLL_byte_0 is computed from the proper byte-level marginalization (one
function of history alone, no leak), and NLL_remainder is the chain-rule
residual.

Also reports "naive" token-level val_bpb (= total NLL / total bytes / log(2))
for sanity check — this should equal the model's known pre-quant val_bpb.

Output: clean BPB number + per-byte stats (avg byte_0 NLL, avg remainder NLL).
"""

import os, sys, time, math, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_train_gpt(path):
    spec = importlib.util.spec_from_file_location("train_gpt_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True)
    ap.add_argument("--n_tokens", type=int, default=2_000_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    # Required env vars must be set BEFORE Hyperparameters() is constructed
    os.environ.setdefault("SMEAR_GATE_ENABLED", "1")
    os.environ.setdefault("SPARSE_ATTN_GATE_ENABLED", "1")
    os.environ.setdefault("LQER_ENABLED", "1")
    os.environ.setdefault("LQER_ASYM_ENABLED", "1")
    os.environ.setdefault("ASYM_LOGIT_RESCALE", "1")
    os.environ.setdefault("CASEOPS_ENABLED", "1")

    device = torch.device("cuda")
    print(f"[load] train_gpt {args.train_gpt}", flush=True)
    tg = load_train_gpt(args.train_gpt)
    tg.BOS_ID = 1   # set the global

    print(f"[load] checkpoint {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state.state_dict()

    h = tg.Hyperparameters()
    h.distributed = False; h.world_size = 1; h.rank = 0; h.local_rank = 0
    h.eval_seq_len = args.seq_len
    print(f"[h] smear_gate={h.smear_gate_enabled} sparse={h.sparse_attn_gate_enabled} lqer={h.lqer_enabled} asym_logit={getattr(h,'asym_logit_rescale','?')}", flush=True)

    m = tg.GPT(h).to(device)
    miss, unex = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unex)}", flush=True)
    if miss:
        print(f"  miss[:5]: {miss[:5]}")
    if unex:
        print(f"  unex[:5]: {unex[:5]}")
    m.eval()
    m.looping_active = True
    print(f"[setup] looping_active=True, encoder_indices={m.encoder_indices}", flush=True)

    # Build tokenizer LUTs
    print(f"[lut] building from {args.tokenizer}", flush=True)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    V = 8192
    first_byte_id = np.full(V, -1, dtype=np.int16)
    has_leading_space = np.zeros(V, dtype=np.bool_)
    is_boundary = np.zeros(V, dtype=np.bool_)
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_boundary[tid] = True
                continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_leading_space[tid] = True
                piece = piece[1:]
            b = piece.encode("utf-8")
            if b:
                first_byte_id[tid] = b[0]
        except Exception:
            pass
    print(f"[lut] V={V} leading_space={int(has_leading_space.sum())} boundary={int(is_boundary.sum())}", flush=True)

    # Two byte-level marginal masks (for the C1-clean construction)
    first_byte_mask = np.zeros((V, 256), dtype=np.float32)
    non_ls_first_byte_mask = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if first_byte_id[tid] >= 0:
            first_byte_mask[tid, int(first_byte_id[tid])] = 1.0
            if not has_leading_space[tid]:
                non_ls_first_byte_mask[tid, int(first_byte_id[tid])] = 1.0
    first_byte_mask_t = torch.from_numpy(first_byte_mask).to(device)
    non_ls_first_byte_mask_t = torch.from_numpy(non_ls_first_byte_mask).to(device)
    leading_space_mask_t = torch.from_numpy(has_leading_space.astype(np.float32)).to(device)
    fb_t = torch.from_numpy(first_byte_id.astype(np.int64)).to(device)
    has_ls_t = torch.from_numpy(has_leading_space).to(device)
    is_bdy_t = torch.from_numpy(is_boundary).to(device)
    SPACE_BYTE = 0x20

    # Load val tokens and byte sidecar
    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    sidecar = np.fromfile(args.val_bytes_bin, dtype=np.uint16)
    OFFSET = 906   # skip header/padding
    arr = raw[OFFSET:OFFSET + args.n_tokens].astype(np.int64)
    sb_aligned = sidecar[OFFSET + 1:OFFSET + len(arr)].astype(np.float64)   # aligned to targets
    N = len(arr) - 1
    print(f"[val] {N} target positions, {sb_aligned.sum():.0f} bytes via sidecar", flush=True)

    # Forward pass — proper marginalization at byte_0 + chain-rule remainder
    # Accumulate: NLL_byte_0 (proper), NLL_remainder, NLL_token (sanity)
    sum_nll_b0 = 0.0
    sum_nll_rem = 0.0
    sum_nll_token = 0.0
    sum_bytes_b0 = 0.0    # 1 byte per token (the b_0)
    sum_bytes_rem = 0.0   # remaining bytes per token
    sum_bytes_total = 0.0 # via sidecar (ground truth)
    n_tok = 0

    seq_len = args.seq_len
    chunk = seq_len * args.batch_seqs
    BOS_ID = 1
    pos = 0
    t_fwd = 0.0
    t0 = time.time()
    print(f"[fwd] N={N} seq_len={seq_len} batch_seqs={args.batch_seqs} chunk={chunk}", flush=True)
    with torch.no_grad():
        while pos < N:
            end = min(pos + chunk, N)
            ids = torch.from_numpy(arr[pos:end+1]).to(device)
            x = ids[:-1]; y = ids[1:]
            bos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu, ms = tg._build_cu_seqlens(bos, x.numel(), x.device, seq_len, 64)
            t1 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = m.forward_logits(x[None], cu_seqlens=cu, max_seqlen=ms)
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t1

            logits = logits.reshape(-1, V).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            nll_token = -log_probs.gather(1, y[:, None]).squeeze(1)   # (n,)

            # Three matmuls for proper byte_0 marginal (case-split on prev_boundary)
            byte_dist = probs @ first_byte_mask_t                     # (n, 256) — full
            non_ls_byte_dist = probs @ non_ls_first_byte_mask_t        # (n, 256) — exclude LS
            p_leading_space = probs @ leading_space_mask_t             # (n,) — Σ over LS

            # Determine realized byte_0 + corresponding proper P:
            #   prev_is_boundary: byte_0 = first_byte_id[y]; P from full byte_dist
            #   prev_not_boundary AND has_leading_space[y]: byte_0 = SPACE; P = leading_space_sum
            #   prev_not_boundary AND NOT has_leading_space[y]: byte_0 = first_byte_id[y]; P from non_ls_byte_dist
            prev_bdy = is_bdy_t[x]
            tgt_has_ls = has_ls_t[y]
            tgt_fb = fb_t[y].clamp(min=0)

            # P from full byte_dist (used when prev_is_boundary)
            p_full = byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            # P from non-LS byte_dist (used when prev_not_boundary AND not leading-space target)
            p_non_ls = non_ls_byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            # P at SPACE (used when prev_not_boundary AND leading-space target)
            p_space = p_leading_space.clamp(min=1e-30)

            # Select the right P per position
            p_b0 = torch.where(
                prev_bdy,
                p_full,
                torch.where(tgt_has_ls, p_space, p_non_ls)
            )
            nll_b0 = -torch.log(p_b0)
            nll_rem = nll_token - nll_b0

            # Per-token byte counts: from sidecar (one byte = byte_0; rest = remainder)
            sb_local = torch.from_numpy(sb_aligned[pos:end]).to(device)
            valid = sb_local > 0   # tokens that contribute bytes (skip operators)

            sum_nll_b0 += float(nll_b0[valid].sum())
            sum_nll_rem += float(nll_rem[valid].sum())
            sum_nll_token += float(nll_token[valid].sum())
            n_tok_valid = int(valid.sum())
            n_tok += n_tok_valid
            sum_bytes_b0 += n_tok_valid                              # 1 byte_0 per valid token
            rem_bytes = (sb_local[valid] - 1).clamp(min=0).sum()     # remaining bytes
            sum_bytes_rem += float(rem_bytes)
            sum_bytes_total += float(sb_local[valid].sum())
            pos = end

    elapsed = time.time() - t0
    LN2 = math.log(2.0)

    print(f"\n=== TIMING ===")
    print(f"Wall:    {elapsed:.2f}s")
    print(f"Forward: {t_fwd:.2f}s ({100*t_fwd/elapsed:.1f}%)")
    print(f"Throughput: {N/t_fwd/1e6:.2f}M tok/s")

    print(f"\n=== TOTALS ===")
    print(f"valid tokens: {n_tok:,} (of {N:,})")
    print(f"total bytes (sidecar): {sum_bytes_total:.0f}")
    print(f"  byte_0 bytes:    {sum_bytes_b0:.0f}")
    print(f"  remainder bytes: {sum_bytes_rem:.0f}")
    print(f"  bytes/token: {sum_bytes_total/max(n_tok,1):.3f}")

    bpb_token = sum_nll_token / sum_bytes_total / LN2
    bpb_b0 = sum_nll_b0 / sum_bytes_b0 / LN2 if sum_bytes_b0 > 0 else 0
    bpb_rem = sum_nll_rem / sum_bytes_rem / LN2 if sum_bytes_rem > 0 else 0

    print(f"\n=== val_bpb ===")
    print(f"Token-level (Σ nll_token / Σ bytes / log2):")
    print(f"  val_bpb = {bpb_token:.6f}")
    print(f"\nByte-level decomposition (proper marginalization):")
    print(f"  byte_0 BPB:    {bpb_b0:.6f}  ({sum_nll_b0/n_tok/LN2:.4f} bits/token)")
    print(f"  remainder BPB: {bpb_rem:.6f}")
    print(f"  combined BPB:  {(sum_nll_b0+sum_nll_rem)/sum_bytes_total/LN2:.6f}  (= token-level by chain rule)")
    print(f"\n  (Note: byte_0 and remainder BPB use different denominators; combined matches token-level.)")


if __name__ == "__main__":
    main()
