"""Mixer oracle on byte_0 positions only (where NN and PPM both make distributions).

For each byte_0 position:
  oracle_nll = min(-log P_NN(realized b_0), -log P_PPM(realized b_0))

Compare to NN-only byte_0 NLL and to the actual PPM-mix NLL.
This bounds what any NN+PPM linear mixer could achieve at byte_0."""

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
    ap.add_argument("--val_bytes_bin", required=True); ap.add_argument("--n_tokens", type=int, default=1_000_000)
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
    targets_arr = np.empty(N, dtype=np.int64); prevs_arr = np.empty(N, dtype=np.int64)
    nll_arr = np.empty(N, dtype=np.float32)
    nn_p_b0 = np.empty(N, dtype=np.float32)
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
            pf = byte_dist.gather(1, tgt_fb[:, None]).squeeze(1)
            pn = non_ls_dist.gather(1, tgt_fb[:, None]).squeeze(1)
            # Per-position: pick case-appropriate
            x_np = x.cpu().numpy(); y_np = y.cpu().numpy()
            pf_np = pf.cpu().numpy(); pn_np = pn.cpu().numpy(); pls_np = p_ls.cpu().numpy()
            for j in range(end - pos):
                pid = int(x_np[j]); tid = int(y_np[j])
                pb = (pid < 0) or bool(is_bdy[pid])
                hs = bool(has_ls[tid]); inc_sp = hs and not pb
                if pb: nn_p_b0[pos+j] = pf_np[j]
                elif inc_sp: nn_p_b0[pos+j] = pls_np[j]
                else: nn_p_b0[pos+j] = pn_np[j]
            targets_arr[pos:end] = y_np; prevs_arr[pos:end] = x_np
            nll_arr[pos:end] = nll_t.cpu().numpy()
            pos = end
    print(f"[fwd] done", flush=True)

    # PPM-D pass on actual byte stream (need full stream)
    print(f"[ppm] running PPM on byte stream", flush=True)
    LN = math.log; LOG2 = LN(2.0); UNIFORM_LOGP = LN(1/256.)
    order = 5; ctx_counts = {}; window = bytearray()
    def _ppm(b, win, counts):
        log_p = None; conf = 0.0; seen = False; esc = 0.0
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            unique = len(cs); total = sum(cs.values()); denom = total + unique
            if not seen: conf = max(cs.values())/denom; seen = True
            if b in cs: log_p = esc + LN(cs[b]/denom); break
            esc += LN(unique/denom) if unique > 0 else 0.0
        if log_p is None: log_p = esc + UNIFORM_LOGP
        return log_p, (conf if seen else 0.0)
    def _upd(b, win, counts):
        for K in range(0, min(order, len(win))+1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            d = counts.get(ctx)
            if d is None: d = {}; counts[ctx] = d
            d[b] = d.get(b, 0) + 1
        win.append(b)
        if len(win) > order: del win[0]

    # Score byte_0 of each token, accumulate NN/PPM/oracle/mix NLLs at byte_0
    nn_b0_nll_total = 0.0
    ppm_b0_nll_total = 0.0
    oracle_b0_nll_total = 0.0
    mix_b0_nll_total = 0.0
    n_byte_content = 0
    total_b0_bytes = 0

    for i in range(N):
        tid = int(targets_arr[i]); pid = int(prevs_arr[i])
        tb = tb_lut[tid]
        hs = bool(has_ls[tid])
        pb = (pid < 0) or bool(is_bdy[pid])
        inc_sp = hs and not pb
        token_bytes = ([0x20] if inc_sp else []) + list(tb)
        if not token_bytes: continue
        b0 = token_bytes[0]
        ppm_logp, conf = _ppm(b0, window, ctx_counts)
        nn_p = max(min(float(nn_p_b0[i]), 1.0), 1e-30)
        ppm_p = math.exp(ppm_logp)
        nn_nll = -math.log(nn_p)
        ppm_nll = -ppm_logp
        oracle_nll = min(nn_nll, ppm_nll)
        # actual mixer with α=15, β=0.80
        z = 15.0*(conf-0.80)
        sig = 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))
        lam = 1.0 - sig
        p_mix = lam * nn_p + (1-lam) * ppm_p
        mix_nll = -math.log(max(p_mix, 1e-30))
        nn_b0_nll_total += nn_nll
        ppm_b0_nll_total += ppm_nll
        oracle_b0_nll_total += oracle_nll
        mix_b0_nll_total += mix_nll
        n_byte_content += 1
        total_b0_bytes += 1
        # update PPM with all bytes
        for b in token_bytes:
            _upd(b, window, ctx_counts)

    print()
    print(f"=== BYTE_0 ORACLE ON {n_byte_content:,} POSITIONS ===")
    print(f"  NN-only NLL/byte_0:       {nn_b0_nll_total/n_byte_content/LOG2:.4f} bits/tok")
    print(f"  PPM-only NLL/byte_0:      {ppm_b0_nll_total/n_byte_content/LOG2:.4f} bits/tok")
    print(f"  Actual mix NLL/byte_0:    {mix_b0_nll_total/n_byte_content/LOG2:.4f} bits/tok")
    print(f"  ORACLE NLL/byte_0:        {oracle_b0_nll_total/n_byte_content/LOG2:.4f} bits/tok")
    print()
    print(f"  Mix savings vs NN:        {(nn_b0_nll_total-mix_b0_nll_total)/n_byte_content/LOG2:.4f} bits/tok")
    print(f"  Oracle savings vs NN:     {(nn_b0_nll_total-oracle_b0_nll_total)/n_byte_content/LOG2:.4f} bits/tok")
    print(f"  Mix / Oracle efficiency:  {(nn_b0_nll_total-mix_b0_nll_total)/(nn_b0_nll_total-oracle_b0_nll_total)*100:.1f}%")

    # Translate to BPB on byte_0 only (denom = total bytes via sidecar, byte_0 portion)
    sidecar = np.fromfile(args.val_bytes_bin, dtype=np.uint16)
    OFFSET = 906
    sb_aligned = sidecar[OFFSET+1:OFFSET+1+N].astype(np.float64)
    total_sidecar = float(sb_aligned.sum())
    print()
    print(f"  Per total sidecar bytes ({int(total_sidecar):,}):")
    print(f"    NN byte_0 BPB:        {nn_b0_nll_total/total_sidecar/LOG2:.6f}")
    print(f"    Mix byte_0 BPB:       {mix_b0_nll_total/total_sidecar/LOG2:.6f}  Δ={(mix_b0_nll_total-nn_b0_nll_total)/total_sidecar/LOG2:+.6f}")
    print(f"    Oracle byte_0 BPB:    {oracle_b0_nll_total/total_sidecar/LOG2:.6f}  Δ={(oracle_b0_nll_total-nn_b0_nll_total)/total_sidecar/LOG2:+.6f}")


if __name__ == "__main__":
    main()
