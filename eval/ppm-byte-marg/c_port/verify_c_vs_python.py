"""Verify the C mixer produces the same numbers as the Python reference, on a small slice.
Run this on the inspect pod with the spec 250 checkpoint."""

import os, sys, time, math, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from proper_ppm_mixer_rigorous import ppm_mixer_with_norm_modes, load_train_gpt
sys.path.insert(0, str(Path(__file__).parent))
from ppm_mixer_c_wrapper import ppm_mixer_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True); ap.add_argument("--n_tokens", type=int, default=200_000)
    ap.add_argument("--seq_len", type=int, default=2048); ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    for k in ["SMEAR_GATE_ENABLED","SPARSE_ATTN_GATE_ENABLED","LQER_ENABLED","LQER_ASYM_ENABLED","ASYM_LOGIT_RESCALE","CASEOPS_ENABLED"]:
        os.environ.setdefault(k, "1")

    device = torch.device("cuda")
    print(f"[load] {args.train_gpt}", flush=True)
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
    print(f"[fwd] done", flush=True)

    for renorm in [False, True]:
        print(f"\n=== {'RIGOROUS' if renorm else 'LOOSE'} ===")
        t0 = time.time()
        py = ppm_mixer_with_norm_modes(
            targets, prevs, nll_arr, p_full, p_nls, p_LS,
            sum_full, sum_caseB,
            tb_lut, has_ls, is_bdy,
            order=5, alpha=15.0, beta=0.80, renormalize=renorm,
        )
        py_dt = time.time() - t0
        t0 = time.time()
        c = ppm_mixer_c(
            targets, prevs, nll_arr, p_full, p_nls, p_LS,
            sum_full, sum_caseB,
            tb_lut, has_ls, is_bdy,
            alpha=15.0, beta=0.80, renormalize=renorm,
        )
        c_dt = time.time() - t0
        print(f"  Python: bpb={py['bpb']:.6f}  tok={py['tokens']}  bytes={py['bytes']}  nll={py['total_nll_nats']:.4f}  renorm={py['renorm_correction_nats']:.4f}  ({py_dt:.2f}s)")
        print(f"  C:      bpb={c['bpb']:.6f}  tok={c['tokens']}  bytes={c['bytes']}  nll={c['total_nll_nats']:.4f}  renorm={c['renorm_correction_nats']:.4f}  ({c_dt:.2f}s)")
        print(f"  Δbpb = {abs(py['bpb']-c['bpb']):.2e}  Δnll = {abs(py['total_nll_nats']-c['total_nll_nats']):.4f} nats  speedup = {py_dt/max(c_dt,1e-6):.1f}x")
        assert py['tokens'] == c['tokens'], f"token count mismatch"
        assert py['bytes'] == c['bytes'], f"byte count mismatch"


if __name__ == "__main__":
    main()
