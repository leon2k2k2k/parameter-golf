"""Fully rigorous version: renormalize NN's byte_0 distribution to sum to 1
over the byte alphabet (conditioning on 'target has byte content'), so that
the mix with PPM is a proper convex combination of two normalized distributions.

For control tokens (target with no bytes): score with full token NLL directly,
consistent with the leaderboard BPB convention.

Reports both UNRENORM (loose, what proper_ppm_mixer.py does) and RENORM (rigorous)
BPB so we can see the size of the correction.
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


def ppm_mixer_with_norm_modes(
    target_ids, prev_ids, nll_nats,
    p_full_b0, p_non_ls_b0, p_leading_space,
    sum_check_full, sum_check_caseB,    # per-position renormalization factors
    token_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    order=5, alpha=15.0, beta=0.80,
    renormalize=False,
):
    _ln = math.log; LOG2 = _ln(2.0); UNIFORM_LOGP = _ln(1/256.)
    num = len(target_ids)
    ctx_counts = {}; window = bytearray()
    total_mix_nll = 0.0; total_canonical_bytes = 0
    total_scored_tokens = 0; total_renorm_correction_nats = 0.0

    def _ppm(b, win, counts):
        log_p = None; conf = 0.0; seen_any = False; escape_log = 0.0
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            unique = len(cs); total = sum(cs.values()); denom = total + unique
            if not seen_any: conf = max(cs.values()) / denom; seen_any = True
            if b in cs:
                log_p = escape_log + _ln(cs[b] / denom); break
            escape_log += _ln(unique / denom) if unique > 0 else 0.0
        if log_p is None: log_p = escape_log + UNIFORM_LOGP
        return log_p, (conf if seen_any else 0.0)

    def _update(b, win, counts):
        for K in range(0, min(order, len(win)) + 1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            d = counts.get(ctx)
            if d is None: d = {}; counts[ctx] = d
            d[b] = d.get(b, 0) + 1
        win.append(b)
        if len(win) > order: del win[0]

    for i in range(num):
        tid = int(target_ids[i]); pid = int(prev_ids[i])
        tb = token_bytes_lut[tid] if 0 <= tid < len(token_bytes_lut) else b""
        has_space = bool(has_leading_space_lut[tid]) if 0 <= tid < len(has_leading_space_lut) else False
        prev_is_boundary = pid < 0 or (bool(is_boundary_token_lut[pid]) if 0 <= pid < len(is_boundary_token_lut) else True)
        include_space = has_space and not prev_is_boundary

        token_bytes = []
        if include_space: token_bytes.append(0x20)
        token_bytes.extend(tb)
        n_bytes = len(token_bytes)
        if n_bytes == 0:
            # Control token: full NLL, no byte split. (Leaderboard convention.)
            total_mix_nll += float(nll_nats[i])
            total_scored_tokens += 1
            continue

        b0 = token_bytes[0]
        ppm_log_b0, conf_b0 = _ppm(b0, window, ctx_counts)

        # NN's raw P(byte_0 = b0 | history)
        if prev_is_boundary:
            nn_p_b0_raw = float(p_full_b0[i])
            sc = float(sum_check_full[i])
        elif include_space:
            nn_p_b0_raw = float(p_leading_space[i])
            sc = float(sum_check_caseB[i])
        else:
            nn_p_b0_raw = float(p_non_ls_b0[i])
            sc = float(sum_check_caseB[i])

        nn_p_b0_raw = max(min(nn_p_b0_raw, 1.0), 1e-30)
        sc = max(min(sc, 1.0), 1e-30)

        if renormalize:
            # Renormalized: P(b0 | history, target_has_bytes) = raw / sum_check
            nn_p_b0_use = nn_p_b0_raw / sc
            # Track the conditioning correction: -log(P(target_has_bytes | history)) = -log(sc)
            # This is the cost of the "byte content" event itself, charged once per byte-content token.
            renorm_correction = -_ln(sc)   # nats added per byte-content token (positive)
            total_renorm_correction_nats += renorm_correction
        else:
            nn_p_b0_use = nn_p_b0_raw
            renorm_correction = 0.0

        nn_p_b0_use = max(min(nn_p_b0_use, 1.0), 1e-30)

        ppm_p_b0 = max(min(math.exp(ppm_log_b0), 1.0), 0.0)
        z0 = alpha * (conf_b0 - beta)
        sig0 = 1.0/(1.0+math.exp(-z0)) if z0 >= 0 else math.exp(z0)/(1.0+math.exp(z0))
        lam0 = 1.0 - sig0
        p_mix0 = lam0 * nn_p_b0_use + (1.0 - lam0) * ppm_p_b0
        nll_b0 = -_ln(max(p_mix0, 1e-30))
        _update(b0, window, ctx_counts)

        nll_rem = 0.0
        if n_bytes > 1:
            ppm_rem_log = 0.0; confs = []
            for j in range(1, n_bytes):
                bj = token_bytes[j]
                lp, cf = _ppm(bj, window, ctx_counts)
                ppm_rem_log += lp; confs.append(cf)
                _update(bj, window, ctx_counts)
            nn_token_p = math.exp(-float(nll_nats[i]))
            # Chain rule: P(rem | b0) = P(token) / P(b0)
            # If renormalizing: P(rem | b0, byte_content) = P(token | byte_content) / P(b0 | byte_content)
            #                                              = (P(token)/sc) / (raw/sc) = P(token)/raw  → SAME
            nn_rem_p = max(nn_token_p / max(nn_p_b0_raw, 1e-30), 1e-30)
            ppm_rem_p = math.exp(ppm_rem_log) if ppm_rem_log > -700 else 0.0
            mean_conf = sum(confs)/len(confs) if confs else 0.0
            zr = alpha * (mean_conf - beta)
            sigr = 1.0/(1.0+math.exp(-zr)) if zr >= 0 else math.exp(zr)/(1.0+math.exp(zr))
            lam_r = 1.0 - sigr
            p_mix_r = lam_r * nn_rem_p + (1.0 - lam_r) * ppm_rem_p
            nll_rem = -_ln(max(p_mix_r, 1e-30))

        total_mix_nll += nll_b0 + nll_rem + renorm_correction
        total_canonical_bytes += n_bytes
        total_scored_tokens += 1

    return {
        "bpb": total_mix_nll / total_canonical_bytes / LOG2 if total_canonical_bytes else None,
        "tokens": total_scored_tokens,
        "bytes": total_canonical_bytes,
        "total_nll_nats": total_mix_nll,
        "renorm_correction_nats": total_renorm_correction_nats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True)
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("SMEAR_GATE_ENABLED", "1")
    os.environ.setdefault("SPARSE_ATTN_GATE_ENABLED", "1")
    os.environ.setdefault("LQER_ENABLED", "1")
    os.environ.setdefault("LQER_ASYM_ENABLED", "1")
    os.environ.setdefault("ASYM_LOGIT_RESCALE", "1")
    os.environ.setdefault("CASEOPS_ENABLED", "1")

    device = torch.device("cuda")
    print(f"[load] {args.train_gpt}", flush=True)
    tg = load_train_gpt(args.train_gpt)
    tg.BOS_ID = 1
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state.state_dict()
    h = tg.Hyperparameters()
    h.distributed=False; h.world_size=1; h.rank=0; h.local_rank=0; h.eval_seq_len=args.seq_len
    m = tg.GPT(h).to(device)
    miss, unex = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(miss)} unexpected={len(unex)}", flush=True)
    m.eval(); m.looping_active = True

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    V = 8192
    fb_id = np.full(V, -1, dtype=np.int16)
    has_ls = np.zeros(V, dtype=np.bool_)
    is_bdy = np.zeros(V, dtype=np.bool_)
    tb_lut = [b""] * V
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_bdy[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_ls[tid] = True; piece = piece[1:]
            b = piece.encode("utf-8")
            tb_lut[tid] = b
            if b: fb_id[tid] = b[0]
        except Exception:
            pass

    fbm = np.zeros((V, 256), dtype=np.float32)
    nls_fbm = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if fb_id[tid] >= 0:
            fbm[tid, int(fb_id[tid])] = 1.0
            if not has_ls[tid]:
                nls_fbm[tid, int(fb_id[tid])] = 1.0
    fbm_t = torch.from_numpy(fbm).to(device)
    nls_t = torch.from_numpy(nls_fbm).to(device)
    lsm_t = torch.from_numpy(has_ls.astype(np.float32)).to(device)
    fb_t = torch.from_numpy(fb_id.astype(np.int64)).to(device)

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    targets = np.empty(N, dtype=np.int64)
    prevs = np.empty(N, dtype=np.int64)
    nll_arr = np.empty(N, dtype=np.float32)
    p_full = np.empty(N, dtype=np.float32)
    p_nls = np.empty(N, dtype=np.float32)
    p_LS = np.empty(N, dtype=np.float32)
    sum_full = np.empty(N, dtype=np.float32)         # Σ_b byte_dist[b] = Σ_T P(T)·𝟙[fb≥0]
    sum_caseB = np.empty(N, dtype=np.float32)        # leading_space_sum + Σ_b non_ls_byte_dist[b] (= mass on byte-content tokens)

    chunk = args.seq_len * args.batch_seqs
    pos = 0
    BOS_ID = 1
    print(f"[fwd] N={N} starting", flush=True)
    t_start = time.time()
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
            tgt_fb = fb_t[y].clamp(min=0)
            pf = byte_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            pn = non_ls_dist.gather(1, tgt_fb[:, None]).squeeze(1).clamp(min=1e-30)
            sc_full = byte_dist.sum(dim=-1)              # Case A (prev_is_boundary)
            sc_caseB = p_ls + non_ls_dist.sum(dim=-1)    # Case B sum (since non_ls_dist[SPACE]=0 verified)
            targets[pos:end] = y.cpu().numpy()
            prevs[pos:end] = x.cpu().numpy()
            nll_arr[pos:end] = nll_t.cpu().numpy()
            p_full[pos:end] = pf.cpu().numpy()
            p_nls[pos:end] = pn.cpu().numpy()
            p_LS[pos:end] = p_ls.cpu().numpy()
            sum_full[pos:end] = sc_full.cpu().numpy()
            sum_caseB[pos:end] = sc_caseB.cpu().numpy()
            pos = end
    print(f"[fwd] done {time.time()-t_start:.1f}s", flush=True)

    print(f"\n[diag] mean sum_full (Case A): {float(sum_full.mean()):.6f}  min: {float(sum_full.min()):.6f}")
    print(f"[diag] mean sum_caseB (Case B): {float(sum_caseB.mean()):.6f}  min: {float(sum_caseB.min()):.6f}")
    print(f"[diag] avg renorm correction (loose vs rigorous): {-math.log(float(sum_caseB.mean())):.6f} nats/byte-content-token")

    # NN-only sanity (token-level val_bpb, no renorm)
    sidecar = np.fromfile(args.val_bytes_bin, dtype=np.uint16)
    OFFSET = 906
    sb_aligned = sidecar[OFFSET+1:OFFSET+1+N].astype(np.float64)
    valid_mask = sb_aligned > 0
    nn_total_nll = float(nll_arr[valid_mask].sum()) + float(nll_arr[~valid_mask].sum())
    total_bytes_full = float(sb_aligned.sum())
    nn_bpb = nn_total_nll / total_bytes_full / math.log(2)
    print(f"\n[NN-only val_bpb (token-level, no renorm)] = {nn_bpb:.6f}")

    print(f"\n[ppm] LOOSE (no renorm — what proper_ppm_mixer.py does):", flush=True)
    t = time.time()
    res_loose = ppm_mixer_with_norm_modes(
        targets, prevs, nll_arr, p_full, p_nls, p_LS,
        sum_full, sum_caseB,
        tb_lut, has_ls, is_bdy,
        order=5, alpha=15.0, beta=0.80, renormalize=False,
    )
    print(f"  {time.time()-t:.1f}s; bpb = {res_loose['bpb']:.6f}", flush=True)

    print(f"\n[ppm] RIGOROUS (renormalize NN to byte alphabet, add log(sum_check) correction):", flush=True)
    t = time.time()
    res_rigor = ppm_mixer_with_norm_modes(
        targets, prevs, nll_arr, p_full, p_nls, p_LS,
        sum_full, sum_caseB,
        tb_lut, has_ls, is_bdy,
        order=5, alpha=15.0, beta=0.80, renormalize=True,
    )
    print(f"  {time.time()-t:.1f}s; bpb = {res_rigor['bpb']:.6f}", flush=True)
    print(f"  renorm correction added: {res_rigor['renorm_correction_nats']:.1f} nats total")
    print(f"  per byte-content token: {res_rigor['renorm_correction_nats']/res_rigor['tokens']:.6f} nats/tok")

    print(f"\n=== SUMMARY ===")
    print(f"NN-only val_bpb (token-level):              {nn_bpb:.6f}")
    print(f"PPM-mix LOOSE (no renorm):                  {res_loose['bpb']:.6f}  Δ={res_loose['bpb']-nn_bpb:+.6f}")
    print(f"PPM-mix RIGOROUS (with renorm):             {res_rigor['bpb']:.6f}  Δ={res_rigor['bpb']-nn_bpb:+.6f}")
    print(f"Rigorous - Loose difference:                 {res_rigor['bpb']-res_loose['bpb']:+.6f} bpb")


if __name__ == "__main__":
    main()
