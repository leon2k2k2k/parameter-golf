"""Verify the remainder distribution sums to 1 and has all non-negative values.

For each position with a multi-byte realized token, the remainder probability
is computed via chain-rule shortcut: P_NN(rem | b_0) = P_NN(T) / P_NN(b_0).

Two checks:
1. ENUMERATIVE: at sampled positions, sum P_NN(T_i)/P_NN(b_0) over ALL tokens
   T_i consistent with the realized b_0 — should be 1.
2. AGGREGATE: across all positions, scan the realized P_NN_rem values and report
   min, max, count of >1.0 (clamping triggered), count of <0.
"""

import os, sys, math, argparse, importlib.util
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
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--n_tokens", type=int, default=200_000)
    ap.add_argument("--n_enum_samples", type=int, default=20)
    ap.add_argument("--seq_len", type=int, default=2048); ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    for k in ["SMEAR_GATE_ENABLED","SPARSE_ATTN_GATE_ENABLED","LQER_ENABLED","LQER_ASYM_ENABLED","ASYM_LOGIT_RESCALE","CASEOPS_ENABLED"]:
        os.environ.setdefault(k, "1")

    device = torch.device("cuda")
    tg = load_train_gpt(args.train_gpt); tg.BOS_ID = 1
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state.state_dict()
    h = tg.Hyperparameters()
    h.distributed=False; h.world_size=1; h.rank=0; h.local_rank=0; h.eval_seq_len=args.seq_len
    m = tg.GPT(h).to(device)
    m.load_state_dict(sd, strict=False); m.eval(); m.looping_active = True

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(); sp.Load(args.tokenizer)
    V = 8192
    fb_id = np.full(V, -1, dtype=np.int16)
    has_ls = np.zeros(V, dtype=np.bool_); is_bdy = np.zeros(V, dtype=np.bool_)
    tb_lut = [b""] * V
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_bdy[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_ls[tid] = True; piece = piece[1:]
            b = piece.encode("utf-8"); tb_lut[tid] = b
            if b: fb_id[tid] = b[0]
        except: pass

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    # Forward + capture full per-position softmax probs (for the enumerative check).
    # And capture nll_token + nn_p_b0_raw for the aggregate scan.
    print(f"[fwd] N={N}", flush=True)

    targets_arr = np.empty(N, dtype=np.int64)
    prevs_arr = np.empty(N, dtype=np.int64)
    nll_arr = np.empty(N, dtype=np.float32)
    nn_p_b0_arr = np.empty(N, dtype=np.float32)   # raw (case-appropriate)
    n_bytes_arr = np.empty(N, dtype=np.int32)     # token byte count incl. include_space

    # For the enumerative samples, save full softmax probs at those positions
    np.random.seed(42)
    sample_positions = sorted(np.random.choice(N, size=args.n_enum_samples * 5, replace=False).tolist())  # pad
    sample_set = set(sample_positions)
    sample_probs = {}  # pos -> np.array of shape (V,)

    chunk = args.seq_len * args.batch_seqs; pos = 0; BOS_ID = 1
    has_ls_arr = np.array([1 if has_ls[t] else 0 for t in range(V)], dtype=np.int32)
    is_bdy_arr = np.array([1 if is_bdy[t] else 0 for t in range(V)], dtype=np.int32)
    fb_id_arr = fb_id.astype(np.int32)
    fbm = np.zeros((V, 256), dtype=np.float32); nls_fbm = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if fb_id[tid] >= 0:
            fbm[tid, int(fb_id[tid])] = 1.0
            if not has_ls[tid]: nls_fbm[tid, int(fb_id[tid])] = 1.0
    fbm_t = torch.from_numpy(fbm).to(device); nls_t = torch.from_numpy(nls_fbm).to(device)
    lsm_t = torch.from_numpy(has_ls.astype(np.float32)).to(device)

    with torch.no_grad():
        while pos < N:
            end = min(pos + chunk, N)
            ids = torch.from_numpy(arr[pos:end+1]).to(device)
            x = ids[:-1]; y = ids[1:]
            bos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu, ms = tg._build_cu_seqlens(bos, x.numel(), x.device, args.seq_len, 64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = m.forward_logits(x[None], cu_seqlens=cu, max_seqlen=ms)
            logits = logits.reshape(-1, V).float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            nll_t = -log_probs.gather(1, y[:, None]).squeeze(1)
            byte_dist = probs @ fbm_t
            non_ls_dist = probs @ nls_t
            p_ls = probs @ lsm_t

            # Per-position case-split nn_p_b0_raw
            x_np = x.cpu().numpy(); y_np = y.cpu().numpy()
            byte_dist_np = byte_dist.cpu().numpy()
            non_ls_np = non_ls_dist.cpu().numpy()
            p_ls_np = p_ls.cpu().numpy()
            probs_np = probs.cpu().numpy()
            nll_np = nll_t.cpu().numpy()

            for j in range(end - pos):
                global_i = pos + j
                tid = int(y_np[j]); pid = int(x_np[j])
                tb = tb_lut[tid]
                hs = bool(has_ls_arr[tid])
                pb = (pid < 0) or bool(is_bdy_arr[pid])
                inc_sp = hs and not pb
                token_bytes = ([0x20] if inc_sp else []) + list(tb)
                nb = len(token_bytes)
                n_bytes_arr[global_i] = nb
                if nb == 0:
                    nn_p_b0_arr[global_i] = 1.0
                else:
                    b0 = token_bytes[0]
                    if pb:
                        nn_p_b0_arr[global_i] = float(byte_dist_np[j, b0])
                    elif inc_sp:
                        nn_p_b0_arr[global_i] = float(p_ls_np[j])
                    else:
                        nn_p_b0_arr[global_i] = float(non_ls_np[j, b0])

                if global_i in sample_set:
                    sample_probs[global_i] = probs_np[j].copy()

            targets_arr[pos:end] = y_np
            prevs_arr[pos:end] = x_np
            nll_arr[pos:end] = nll_np
            pos = end
    print(f"[fwd] done", flush=True)

    # ENUMERATIVE CHECK at sampled multi-byte positions
    print()
    print("=" * 70)
    print("ENUMERATIVE CHECK: Σ over all tokens consistent with realized b_0 of P(T)/P(b_0)")
    print("=" * 70)

    multibyte_samples = []
    for pos in sample_positions:
        tid = int(targets_arr[pos]); pid = int(prevs_arr[pos])
        tb = tb_lut[tid]
        hs = bool(has_ls_arr[tid])
        pb = (pid < 0) or bool(is_bdy_arr[pid])
        inc_sp = hs and not pb
        nb = len(tb) + (1 if inc_sp else 0)
        if nb < 2: continue
        multibyte_samples.append((pos, tid, pid, hs, pb, inc_sp))
        if len(multibyte_samples) >= args.n_enum_samples: break

    print(f"Sampled {len(multibyte_samples)} multi-byte positions:")
    print()
    for pos, tid, pid, hs, pb, inc_sp in multibyte_samples:
        probs = sample_probs[pos]   # (V,)
        if inc_sp:  # b_0 = SPACE; consistent T = all LS tokens
            mask = has_ls_arr.astype(bool)
            b0_label = "SPACE (LS class)"
        elif pb:    # any token with fb=fb_actual
            actual_b0 = (tb_lut[tid][0] if tb_lut[tid] else 0x20) if not inc_sp else 0x20
            mask = (fb_id_arr == actual_b0)
            b0_label = f"0x{actual_b0:02x} (any token w/ fb=b0)"
        else:       # non-LS w/ first piece byte = realized b_0
            actual_b0 = tb_lut[tid][0]
            mask = ((~has_ls_arr.astype(bool)) & (fb_id_arr == actual_b0))
            b0_label = f"0x{actual_b0:02x} (¬LS class, fb=b0)"

        consistent_ids = np.where(mask)[0]
        sum_prob = float(probs[consistent_ids].sum())
        nn_p_b0 = nn_p_b0_arr[pos]

        # Sum of P(T)/P(b_0) over consistent T
        rem_sum = sum_prob / max(nn_p_b0, 1e-30)
        # Each P(T)/P(b_0) value
        rem_values = probs[consistent_ids] / max(nn_p_b0, 1e-30)
        rem_min = float(rem_values.min()) if len(rem_values) else 0.0
        rem_max = float(rem_values.max()) if len(rem_values) else 0.0
        n_consistent = len(consistent_ids)

        try: piece = sp.id_to_piece(tid)
        except: piece = f"<{tid}>"
        print(f"  pos={pos:6d}  realized={piece!r:>15s}  case=({'A' if pb else ('B-LS' if inc_sp else 'B-other')})  b_0={b0_label}")
        print(f"           n_consistent_tokens={n_consistent}  Σ P(T)={sum_prob:.6f}  P(b_0)={nn_p_b0:.6f}")
        print(f"           Σ P(rem|b_0) = {rem_sum:.10f}  (should be 1.0)")
        print(f"           min P(rem|b_0) = {rem_min:.6e}   max P(rem|b_0) = {rem_max:.6f}   (each ∈ [0,1])")

    # AGGREGATE CHECK over all positions
    print()
    print("=" * 70)
    print("AGGREGATE CHECK across all multi-byte positions")
    print("=" * 70)
    multi_mask = (n_bytes_arr >= 2)
    nn_p_b0_use = nn_p_b0_arr[multi_mask]
    nll_use = nll_arr[multi_mask]
    p_token = np.exp(-nll_use.astype(np.float64))
    nn_rem_p = p_token / np.maximum(nn_p_b0_use.astype(np.float64), 1e-30)

    n_total = int(multi_mask.sum())
    n_neg = int((nn_rem_p < 0).sum())
    n_gt1 = int((nn_rem_p > 1.0).sum())
    n_gt1_001 = int((nn_rem_p > 1.001).sum())  # excluding fp32 noise
    print(f"  Multi-byte positions:                {n_total:,}")
    print(f"  P(rem|b_0) < 0:                      {n_neg:,}  (should be 0)")
    print(f"  P(rem|b_0) > 1.0:                    {n_gt1:,}  ({100*n_gt1/n_total:.4f}%)  ← fp32 noise possible")
    print(f"  P(rem|b_0) > 1.001:                  {n_gt1_001:,}  ({100*n_gt1_001/n_total:.4f}%)  ← would be a real violation")
    print(f"  min P(rem|b_0):                      {nn_rem_p.min():.6e}")
    print(f"  max P(rem|b_0):                      {nn_rem_p.max():.6f}")
    print(f"  mean P(rem|b_0):                     {nn_rem_p.mean():.4f}")

    if n_neg == 0 and n_gt1_001 == 0:
        print("\n  VERDICT: remainder probabilities are valid — non-negative and ≤ 1 within fp32 noise.")
    else:
        print(f"\n  WARNING: found {n_neg} negative or {n_gt1_001} >1.001 — investigate.")


if __name__ == "__main__":
    main()
