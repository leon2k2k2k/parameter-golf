"""Sweep α, β on the cached forward pass. Reports sidecar-normalized BPB for each combo."""

import os, sys, math, importlib.util, argparse, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/ngram_analysis")
sys.path.insert(0, "/tmp/ngram_analysis/c_port")
from proper_ppm_mixer_rigorous import load_train_gpt
from ppm_mixer_c_wrapper import ppm_mixer_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True); ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=2048); ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    for k in ["SMEAR_GATE_ENABLED","SPARSE_ATTN_GATE_ENABLED","LQER_ENABLED","LQER_ASYM_ENABLED","ASYM_LOGIT_RESCALE","CASEOPS_ENABLED"]:
        os.environ.setdefault(k, "1")

    device = torch.device("cuda")
    print(f"[load]", flush=True)
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

    fbm = np.zeros((V, 256), dtype=np.float32); nls_fbm = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if fb_id[tid] >= 0:
            fbm[tid, int(fb_id[tid])] = 1.0
            if not has_ls[tid]: nls_fbm[tid, int(fb_id[tid])] = 1.0
    fbm_t = torch.from_numpy(fbm).to(device); nls_t = torch.from_numpy(nls_fbm).to(device)
    lsm_t = torch.from_numpy(has_ls.astype(np.float32)).to(device)
    fb_t = torch.from_numpy(fb_id.astype(np.int64)).to(device)

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    print(f"[fwd] N={N}", flush=True)
    t0 = time.time()
    targets = np.empty(N, dtype=np.int64); prevs = np.empty(N, dtype=np.int64)
    nll_arr = np.empty(N, dtype=np.float32)
    p_full = np.empty(N, dtype=np.float32); p_nls = np.empty(N, dtype=np.float32); p_LS = np.empty(N, dtype=np.float32)
    sum_full = np.empty(N, dtype=np.float32); sum_caseB = np.empty(N, dtype=np.float32)
    chunk = args.seq_len * args.batch_seqs; pos = 0; BOS_ID = 1
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
            log_probs = F.log_softmax(logits, dim=-1); probs = log_probs.exp()
            nll_t = -log_probs.gather(1, y[:, None]).squeeze(1)
            byte_dist = probs @ fbm_t; non_ls_dist = probs @ nls_t; p_ls = probs @ lsm_t
            tgt_fb = fb_t[y].clamp(min=0)
            pf = byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            pn = non_ls_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            sc_full = byte_dist.sum(dim=-1); sc_caseB = p_ls + non_ls_dist.sum(dim=-1)
            targets[pos:end] = y.cpu().numpy(); prevs[pos:end] = x.cpu().numpy()
            nll_arr[pos:end] = nll_t.cpu().numpy()
            p_full[pos:end] = pf.cpu().numpy(); p_nls[pos:end] = pn.cpu().numpy(); p_LS[pos:end] = p_ls.cpu().numpy()
            sum_full[pos:end] = sc_full.cpu().numpy(); sum_caseB[pos:end] = sc_caseB.cpu().numpy()
            pos = end
    print(f"[fwd] done {time.time()-t0:.1f}s", flush=True)

    sidecar = np.fromfile(args.val_bytes_bin, dtype=np.uint16)
    OFFSET = 906
    sb_aligned = sidecar[OFFSET+1:OFFSET+1+N].astype(np.int64)
    SB = int(sb_aligned.sum())
    LOG2 = math.log(2)
    nn_total_nll = float(nll_arr.sum())
    nn_bpb = nn_total_nll / SB / LOG2

    print(f"\nSidecar bytes: {SB:,}")
    print(f"NN-only val_bpb (sidecar): {nn_bpb:.6f}\n")

    # Sweep
    alphas = [30.0, 50.0, 100.0, 200.0]
    betas = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

    print(f"=== α/β sweep on N={N} tokens, LOOSE mode ===")
    print(f"{'':<8}", end="");
    for b in betas: print(f"  β={b:.2f} ", end="")
    print()

    best = (None, None, float('inf'))
    for a in alphas:
        print(f"α={a:>4.0f}: ", end="", flush=True)
        for b in betas:
            res = ppm_mixer_c(targets, prevs, nll_arr, p_full, p_nls, p_LS,
                              sum_full, sum_caseB, tb_lut, has_ls, is_bdy,
                              alpha=a, beta=b, renormalize=False)
            bpb = res['total_nll_nats'] / SB / LOG2
            delta = bpb - nn_bpb
            if delta < best[2]: best = (a, b, delta)
            print(f"  {delta:+.4f}", end="", flush=True)
        print()

    print(f"\nBest combo: α={best[0]}, β={best[1]} → Δ vs NN = {best[2]:+.6f} BPB")
    if best[2] >= 0:
        print(f"WARNING: no combo improves over NN-only (best is +{best[2]:.4f}). Mixer hurts uniformly.")
    else:
        print(f"Mixer helps by {-best[2]:.6f} BPB at best.")


if __name__ == "__main__":
    main()
