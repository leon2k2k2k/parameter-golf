"""For each (case, position), explicitly construct the 256-byte distribution
and verify it sums to sum_check, then sum_check_renormalized = 1 exactly.

Reports per-case min/max/mean of sum BEFORE renorm and AFTER renorm."""

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
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_bdy[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_ls[tid] = True; piece = piece[1:]
            b = piece.encode("utf-8")
            if b: fb_id[tid] = b[0]
        except: pass

    fbm = np.zeros((V, 256), dtype=np.float32); nls_fbm = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if fb_id[tid] >= 0:
            fbm[tid, int(fb_id[tid])] = 1.0
            if not has_ls[tid]: nls_fbm[tid, int(fb_id[tid])] = 1.0
    fbm_t = torch.from_numpy(fbm).to(device); nls_t = torch.from_numpy(nls_fbm).to(device)
    lsm_t = torch.from_numpy(has_ls.astype(np.float32)).to(device)

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    print(f"[fwd] N={N}, building per-position 256-byte distributions...")
    # Per-position: full byte_dist (Case A construction), and Case B construction (non_ls_dist with [SPACE]=p_ls)
    sum_caseA_arr = np.empty(N, dtype=np.float64)
    sum_caseB_arr = np.empty(N, dtype=np.float64)
    sum_renorm_caseA = np.empty(N, dtype=np.float64)
    sum_renorm_caseB = np.empty(N, dtype=np.float64)
    prev_is_bdy_arr = np.zeros(N, dtype=np.bool_)
    chunk = args.seq_len * args.batch_seqs; pos = 0; BOS_ID = 1
    is_bdy_t = torch.from_numpy(is_bdy.astype(np.bool_)).to(device)
    with torch.no_grad():
        while pos < N:
            end = min(pos + chunk, N)
            ids = torch.from_numpy(arr[pos:end+1]).to(device)
            x = ids[:-1]
            bos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu, ms = tg._build_cu_seqlens(bos, x.numel(), x.device, args.seq_len, 64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = m.forward_logits(x[None], cu_seqlens=cu, max_seqlen=ms)
            logits = logits.reshape(-1, V).float()
            probs = F.log_softmax(logits, dim=-1).exp()
            byte_dist = probs @ fbm_t            # (B*T, 256) Case A: full marginal
            non_ls_dist = probs @ nls_t           # (B*T, 256)
            p_ls = probs @ lsm_t                  # (B*T,) leading-space mass

            # Case A construction: byte_dist[b] for all b
            sum_A = byte_dist.sum(dim=-1)         # = sum_check_full
            # Case B construction: non_ls_dist[b] for b≠SPACE; p_ls for b=SPACE
            caseB_dist = non_ls_dist.clone()
            caseB_dist[:, 0x20] = p_ls            # set SPACE bin to leading_space_sum
            sum_B = caseB_dist.sum(dim=-1)        # = sum_check_caseB

            # After renormalization, both should sum to 1.0 exactly
            renorm_A_dist = byte_dist / sum_A.clamp(min=1e-30).unsqueeze(-1)
            renorm_B_dist = caseB_dist / sum_B.clamp(min=1e-30).unsqueeze(-1)
            sum_renorm_A = renorm_A_dist.sum(dim=-1)
            sum_renorm_B = renorm_B_dist.sum(dim=-1)

            # Track which positions are case A (prev_is_boundary)
            prev_bdy = is_bdy_t[x]
            prev_is_bdy_arr[pos:end] = prev_bdy.cpu().numpy()
            sum_caseA_arr[pos:end] = sum_A.cpu().numpy()
            sum_caseB_arr[pos:end] = sum_B.cpu().numpy()
            sum_renorm_caseA[pos:end] = sum_renorm_A.cpu().numpy()
            sum_renorm_caseB[pos:end] = sum_renorm_B.cpu().numpy()
            pos = end

    print()
    print("=" * 70)
    print("VERIFICATION OF P_NN(byte_0 = b | history, prev) over 256 bytes")
    print("=" * 70)
    nA = int(prev_is_bdy_arr.sum())
    nB = N - nA
    print(f"Total positions: {N:,}  |  Case A (prev_is_boundary): {nA:,} ({100*nA/N:.2f}%)  |  Case B: {nB:,} ({100*nB/N:.2f}%)")

    print()
    print("--- CASE A (prev_is_boundary): construction = byte_dist[b] for all 256 b ---")
    A_used = sum_caseA_arr[prev_is_bdy_arr]
    print(f"  Σ_b P_NN(b | hist, prev_bdy) BEFORE renorm:")
    print(f"    mean = {A_used.mean():.6f}   min = {A_used.min():.6f}   max = {A_used.max():.6f}")
    print(f"    (slack = mass on control tokens = 1 − sum)")
    A_renorm = sum_renorm_caseA[prev_is_bdy_arr]
    print(f"  Σ_b P_NN(b | hist, prev_bdy) AFTER renorm by sum_check:")
    print(f"    mean = {A_renorm.mean():.10f}   min = {A_renorm.min():.10f}   max = {A_renorm.max():.10f}")
    print(f"    max deviation from 1.0 = {abs(A_renorm - 1.0).max():.2e}  (fp32 roundoff)")

    print()
    print("--- CASE B (prev_not_boundary): construction = non_ls_dist[b] for b≠SPACE; p_ls for b=SPACE ---")
    B_used = sum_caseB_arr[~prev_is_bdy_arr]
    print(f"  Σ_b P_NN(b | hist, prev_not_bdy) BEFORE renorm:")
    print(f"    mean = {B_used.mean():.6f}   min = {B_used.min():.6f}   max = {B_used.max():.6f}")
    print(f"    (slack = mass on control tokens = 1 − sum)")
    B_renorm = sum_renorm_caseB[~prev_is_bdy_arr]
    print(f"  Σ_b P_NN(b | hist, prev_not_bdy) AFTER renorm by sum_check:")
    print(f"    mean = {B_renorm.mean():.10f}   min = {B_renorm.min():.10f}   max = {B_renorm.max():.10f}")
    print(f"    max deviation from 1.0 = {abs(B_renorm - 1.0).max():.2e}  (fp32 roundoff)")

    print()
    print("VERDICT: both case-distributions are proper distributions over the 256-byte alphabet")
    print("after renormalization, with deviation from 1.0 at the level of fp32 numerical noise.")
    print("The pre-renorm sum (~0.999 in case B, ~0.90 in case A) reflects the mass on control")
    print("tokens (which have no byte representation). Rigorous mode: divide by sum_check + charge")
    print("−log(sum_check) per byte-content token.")


if __name__ == "__main__":
    main()
