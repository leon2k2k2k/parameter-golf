"""Three-way comparison at byte_0 ONLY (no PPM mixing yet, just NN's per-token b_0 claim).

A = flat-everywhere:    nll_b0 = nll_token / n_bytes_in_stream
B = #2039 hybrid:       if include_space: A; else: -log(byte_dist[first_byte_id[y]])
C = no-if (marginal):   -log(byte_dist[first_byte_id[y]]) for ALL positions

Sum Σ nll_b0 over scored token positions, average over (one byte_0 per token).
"""

import sys, os, time, math, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_train_gpt(path):
    spec = importlib.util.spec_from_file_location("train_gpt", path)
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
    ap.add_argument("--n_tokens", type=int, default=200_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[load] train_gpt {args.train_gpt}", flush=True)
    tg = load_train_gpt(args.train_gpt)
    print(f"[load] checkpoint {args.ckpt}", flush=True)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state.state_dict()

    h = tg.Hyperparameters()
    h.distributed = False
    h.world_size = 1
    h.rank = 0
    h.local_rank = 0
    h.eval_seq_len = args.seq_len
    model = tg.GPT(h).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

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

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)

    per_tok_bytes = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        if is_boundary[tid]:
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            piece = piece[1:]
        per_tok_bytes[tid] = len(piece.encode("utf-8"))
    per_tok_bytes_t = torch.from_numpy(per_tok_bytes).to(device)

    chunk = args.seq_len * args.batch_seqs
    N = len(arr) - 1
    BOS_ID = 1

    sum_A = 0.0  # flat-everywhere
    sum_B = 0.0  # #2039 hybrid
    sum_C = 0.0  # no-if marginal everywhere
    sum_D = 0.0  # leading-space marginal for include, byte_dist for non-include
    sum_E = 0.0  # truly proper: prev-conditioned only, target-independent distribution
    n_inc = 0
    n_total = 0

    t_fwd = 0.0
    t_marg = 0.0

    pos = 0
    t_start = time.time()
    print(f"[fwd] {N} positions, chunk={chunk}", flush=True)
    with torch.no_grad():
        while pos < N:
            end = min(pos + chunk, N)
            input_ids = torch.from_numpy(arr[pos:end+1]).to(device)
            x = input_ids[:-1]
            y = input_ids[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = tg._build_cu_seqlens(bos_pos, x.numel(), x.device, args.seq_len, 64)
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            torch.cuda.synchronize()
            t_fwd += time.perf_counter() - t0

            logits = logits.reshape(-1, V).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            nll_token = -log_probs.gather(1, y[:, None]).squeeze(1)

            t1 = time.perf_counter()
            byte_dist = probs @ first_byte_mask_t                       # (B*T, 256) — full marginal
            non_ls_byte_dist = probs @ non_ls_first_byte_mask_t          # (B*T, 256) — non-LS only
            p_leading_space = probs @ leading_space_mask_t               # (B*T,) — Σ over LS tokens
            torch.cuda.synchronize()
            t_marg += time.perf_counter() - t1

            tgt_fb = fb_t[y].clamp(min=0)
            p_byte_dist = byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            nll_C = -torch.log(p_byte_dist)

            n_bytes_canon = per_tok_bytes_t[y]
            inc = has_ls_t[y] & (~is_bdy_t[x])
            n_bytes_stream = n_bytes_canon + inc.long()
            valid = (n_bytes_canon > 0) & (y != 0) & (x != 0)

            nll_A = nll_token / n_bytes_stream.clamp(min=1).float()
            nll_B = torch.where(inc, nll_A, nll_C)
            # D: include → P(byte_0=SPACE) = leading_space marginal; non-include → byte_dist[first_byte_id]
            nll_D_inc = -torch.log(p_leading_space.clamp(min=1e-30))
            nll_D = torch.where(inc, nll_D_inc, nll_C)

            # E: TRULY PROPER — conditional only on prev (C1-clean):
            #   if prev_is_boundary: distribution = byte_dist (full mask), realized byte = first_byte_id[y]
            #   if prev_not_boundary AND has_leading_space[y]: realized byte = SPACE, P = leading_space_sum
            #   if prev_not_boundary AND NOT has_leading_space[y]: realized byte = first_byte_id[y],
            #                                                       P = non_ls_byte_dist[that byte]
            prev_is_bdy = is_bdy_t[x]
            p_non_ls_at_realized = non_ls_byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            # When prev_not_boundary AND has_leading_space[y]: use leading_space_sum
            # When prev_not_boundary AND NOT has_leading_space[y]: use non_ls_byte_dist[fb]
            # When prev_is_boundary: use byte_dist[fb] (= p_byte_dist, already computed = nll_C input)
            p_E_prev_notbdy = torch.where(has_ls_t[y], p_leading_space.clamp(min=1e-30), p_non_ls_at_realized)
            p_E = torch.where(prev_is_bdy, p_byte_dist, p_E_prev_notbdy)
            nll_E = -torch.log(p_E.clamp(min=1e-30))

            sum_A += float(nll_A[valid].sum())
            sum_B += float(nll_B[valid].sum())
            sum_C += float(nll_C[valid].sum())
            sum_D += float(nll_D[valid].sum())
            sum_E += float(nll_E[valid].sum())
            n_inc += int((valid & inc).sum())
            n_total += int(valid.sum())

            pos = end
    elapsed = time.time() - t_start
    LN2 = math.log(2.0)

    print(f"\n=== SPEED ===")
    print(f"Forward total:        {t_fwd:.2f}s ({N/t_fwd/1e6:.2f}M tok/s)")
    print(f"Marginalization total:{t_marg:.2f}s ({100*t_marg/t_fwd:.1f}% of forward)")
    print(f"Wall total:           {elapsed:.2f}s")

    print(f"\n=== POSITIONS ===")
    print(f"valid scored tokens: {n_total}")
    print(f"include_space:       {n_inc} ({100*n_inc/n_total:.2f}%)")
    print(f"non-include:         {n_total - n_inc} ({100*(n_total-n_inc)/n_total:.2f}%)")

    print(f"\n=== Σ nll_byte_0 (nats), avg per token (bits) ===")
    print(f"A flat-everywhere:           Σ={sum_A:.1f}  avg={sum_A/n_total/LN2:.5f} bits/tok")
    print(f"B #2039 hybrid (with if):    Σ={sum_B:.1f}  avg={sum_B/n_total/LN2:.5f} bits/tok")
    print(f"C no-if (byte_dist for all): Σ={sum_C:.1f}  avg={sum_C/n_total/LN2:.5f} bits/tok")
    print(f"D LS-marginal (not C1-clean):Σ={sum_D:.1f}  avg={sum_D/n_total/LN2:.5f} bits/tok")
    print(f"E TRULY PROPER (C1-clean):   Σ={sum_E:.1f}  avg={sum_E/n_total/LN2:.5f} bits/tok")
    print(f"\n=== Δ vs A (negative = better than flat) ===")
    print(f"B - A: {(sum_B - sum_A)/n_total/LN2:+.5f} bits/tok")
    print(f"C - A: {(sum_C - sum_A)/n_total/LN2:+.5f} bits/tok")
    print(f"D - A: {(sum_D - sum_A)/n_total/LN2:+.5f} bits/tok")
    print(f"E - A: {(sum_E - sum_A)/n_total/LN2:+.5f} bits/tok")
    print(f"\n=== Δ vs B (#2039 hybrid; negative = better than #2039) ===")
    print(f"D - B: {(sum_D - sum_B)/n_total/LN2:+.5f} bits/tok")
    print(f"E - B: {(sum_E - sum_B)/n_total/LN2:+.5f} bits/tok")
    print(f"\nE is the truly-proper, C1-clean construction (target-independent distribution).")
    print(f"D over-counts in non-include positions (uses full byte_dist that includes LS tokens).")
    print(f"If E < B, the C1-clean fix is strictly better than what #2039 ships at byte_0.")


if __name__ == "__main__":
    main()
