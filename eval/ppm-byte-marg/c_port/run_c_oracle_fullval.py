"""Full-val C-mixer with oracle decomposition."""

import os, sys, time, math, argparse, importlib.util
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
    ap.add_argument("--val_bytes_bin", required=True); ap.add_argument("--n_tokens", type=int, default=47_000_000)
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
    sb_aligned = sidecar[OFFSET+1:OFFSET+1+N].astype(np.float64)
    nn_total_nll = float(nll_arr.sum())
    total_bytes_full = float(sb_aligned.sum())
    LOG2 = math.log(2)
    nn_bpb = nn_total_nll / total_bytes_full / LOG2

    print(f"\n[NN-only val_bpb (token-level)] = {nn_bpb:.6f}", flush=True)
    print(f"[total sidecar bytes] = {int(total_bytes_full):,}", flush=True)

    print(f"\n[ppm-c] LOOSE (computing oracle decomposition)", flush=True)
    t0 = time.time()
    res = ppm_mixer_c(targets, prevs, nll_arr, p_full, p_nls, p_LS,
                      sum_full, sum_caseB, tb_lut, has_ls, is_bdy,
                      alpha=15.0, beta=0.80, renormalize=False)
    print(f"  {time.time()-t0:.1f}s", flush=True)
    print(f"  bpb (canonical-norm, NOT leaderboard): {res['bpb']:.6f}", flush=True)
    bpb_sidecar = res['total_nll_nats'] / total_bytes_full / math.log(2)
    print(f"  bpb (SIDECAR-normalized, leaderboard): {bpb_sidecar:.6f}", flush=True)
    print(f"  Δ vs NN (sidecar):                     {bpb_sidecar - nn_bpb:+.6f}", flush=True)

    SB = total_bytes_full
    print()
    print("=" * 78)
    print(f"FULL-VAL DECOMPOSITION ({int(total_bytes_full):,} sidecar bytes; {res['b0_positions']:,} byte-content tokens; {res['multibyte_tokens']:,} multi-byte tokens)")
    print("=" * 78)

    # All BPB contributions normalized by total sidecar bytes.
    nn_b0_bpb     = res['nn_b0_nll'] / SB / LOG2
    ppm_b0_bpb    = res['ppm_b0_nll'] / SB / LOG2
    mix_b0_bpb    = res['mix_b0_nll'] / SB / LOG2
    oracle_b0_bpb = res['oracle_b0_nll'] / SB / LOG2
    nn_rem_bpb    = res['nn_rem_nll'] / SB / LOG2
    ppm_rem_bpb   = res['ppm_rem_nll'] / SB / LOG2
    mix_rem_bpb   = res['mix_rem_nll'] / SB / LOG2
    oracle_rem_bpb = res['oracle_rem_nll'] / SB / LOG2

    # control-token contribution = total_NN - byte_0 NN - rem NN
    ctl_bpb = nn_bpb - nn_b0_bpb - nn_rem_bpb

    # totals (mix and oracle): byte_0 + rem + control (control unchanged across mix variants)
    nn_total      = nn_b0_bpb + nn_rem_bpb + ctl_bpb
    ppm_total     = ppm_b0_bpb + ppm_rem_bpb + ctl_bpb     # PPM-only with NN's control NLL passthrough
    mix_total     = mix_b0_bpb + mix_rem_bpb + ctl_bpb
    oracle_total  = oracle_b0_bpb + oracle_rem_bpb + ctl_bpb

    # Hypothetical: drop byte_0 mix (use NN at byte_0, mix at remainder)
    drop_b0_total = nn_b0_bpb + mix_rem_bpb + ctl_bpb
    # Hypothetical: byte_0 oracle + actual remainder mix
    b0_oracle_total = oracle_b0_bpb + mix_rem_bpb + ctl_bpb
    # Hypothetical: full oracle (byte_0 + remainder both at oracle)
    full_oracle_total = oracle_b0_bpb + oracle_rem_bpb + ctl_bpb

    print()
    print(f"{'':<32} {'byte_0 BPB':>11}  {'rem BPB':>10}  {'ctl BPB':>10}  {'total BPB':>11}  {'Δ vs NN':>10}")
    print(f"{'-'*32} {'-'*11}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*10}")
    print(f"{'NN-only':<32} {nn_b0_bpb:>11.6f}  {nn_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {nn_total:>11.6f}  {0.0:>+10.6f}")
    print(f"{'PPM-only (w/ NN control)':<32} {ppm_b0_bpb:>11.6f}  {ppm_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {ppm_total:>11.6f}  {ppm_total-nn_total:>+10.6f}")
    print(f"{'Actual mix (LOOSE α15β0.80)':<32} {mix_b0_bpb:>11.6f}  {mix_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {mix_total:>11.6f}  {mix_total-nn_total:>+10.6f}")
    print(f"{'PROJ: λ_b0=1 (drop b0 mix)':<32} {nn_b0_bpb:>11.6f}  {mix_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {drop_b0_total:>11.6f}  {drop_b0_total-nn_total:>+10.6f}")
    print(f"{'PROJ: byte_0 oracle + mix rem':<32} {oracle_b0_bpb:>11.6f}  {mix_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {b0_oracle_total:>11.6f}  {b0_oracle_total-nn_total:>+10.6f}")
    print(f"{'PROJ: full oracle (b0 + rem)':<32} {oracle_b0_bpb:>11.6f}  {oracle_rem_bpb:>10.6f}  {ctl_bpb:>10.6f}  {full_oracle_total:>11.6f}  {full_oracle_total-nn_total:>+10.6f}")

    print()
    print("=== Per-class efficiency ===")
    nn_minus_mix_b0 = nn_b0_bpb - mix_b0_bpb
    nn_minus_oracle_b0 = nn_b0_bpb - oracle_b0_bpb
    nn_minus_mix_rem = nn_rem_bpb - mix_rem_bpb
    nn_minus_oracle_rem = nn_rem_bpb - oracle_rem_bpb
    print(f"  byte_0:    mix savings vs NN = {nn_minus_mix_b0:+.6f} BPB    oracle savings = {nn_minus_oracle_b0:+.6f}    mix/oracle = {100*nn_minus_mix_b0/max(nn_minus_oracle_b0,1e-30):.1f}%")
    print(f"  remainder: mix savings vs NN = {nn_minus_mix_rem:+.6f} BPB    oracle savings = {nn_minus_oracle_rem:+.6f}    mix/oracle = {100*nn_minus_mix_rem/max(nn_minus_oracle_rem,1e-30):.1f}%")


if __name__ == "__main__":
    main()
