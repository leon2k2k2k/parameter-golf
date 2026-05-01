"""Find specific tokens where PPM-D mix gives massive savings vs NN-alone.
For each such token, dump the byte-by-byte breakdown so you can see WHERE
the gain comes from."""

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--n_tokens", type=int, default=500_000)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--top_k_gains", type=int, default=20)
    args = ap.parse_args()

    os.environ.setdefault("SMEAR_GATE_ENABLED", "1")
    os.environ.setdefault("SPARSE_ATTN_GATE_ENABLED", "1")
    os.environ.setdefault("LQER_ENABLED", "1")
    os.environ.setdefault("LQER_ASYM_ENABLED", "1")
    os.environ.setdefault("ASYM_LOGIT_RESCALE", "1")
    os.environ.setdefault("CASEOPS_ENABLED", "1")

    device = torch.device("cuda")
    tg = load_train_gpt(args.train_gpt)
    tg.BOS_ID = 1
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state) if isinstance(state, dict) else state.state_dict()
    h = tg.Hyperparameters()
    h.distributed=False; h.world_size=1; h.rank=0; h.local_rank=0; h.eval_seq_len=args.seq_len
    m = tg.GPT(h).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval(); m.looping_active = True

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    V = 8192
    first_byte_id = np.full(V, -1, dtype=np.int16)
    has_leading_space = np.zeros(V, dtype=np.bool_)
    is_boundary = np.zeros(V, dtype=np.bool_)
    token_bytes_lut = [b""] * V
    for tid in range(V):
        try:
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                is_boundary[tid] = True; continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("▁"):
                has_leading_space[tid] = True; piece = piece[1:]
            b = piece.encode("utf-8")
            token_bytes_lut[tid] = b
            if b: first_byte_id[tid] = b[0]
        except Exception:
            pass

    fbm = np.zeros((V, 256), dtype=np.float32)
    nls_fbm = np.zeros((V, 256), dtype=np.float32)
    for tid in range(V):
        if first_byte_id[tid] >= 0:
            fbm[tid, int(first_byte_id[tid])] = 1.0
            if not has_leading_space[tid]:
                nls_fbm[tid, int(first_byte_id[tid])] = 1.0
    fbm_t = torch.from_numpy(fbm).to(device)
    nls_t = torch.from_numpy(nls_fbm).to(device)
    lsm_t = torch.from_numpy(has_leading_space.astype(np.float32)).to(device)
    fb_t = torch.from_numpy(first_byte_id.astype(np.int64)).to(device)

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    OFFSET = 906
    arr = raw[OFFSET:OFFSET + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    targets = np.empty(N, dtype=np.int64)
    prevs = np.empty(N, dtype=np.int64)
    nll_arr = np.empty(N, dtype=np.float32)
    p_full = np.empty(N, dtype=np.float32)
    p_nls = np.empty(N, dtype=np.float32)
    p_LS = np.empty(N, dtype=np.float32)
    chunk = args.seq_len * args.batch_seqs
    pos = 0
    BOS_ID = 1
    print(f"[fwd] running...", flush=True)
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
            targets[pos:end] = y.cpu().numpy()
            prevs[pos:end] = x.cpu().numpy()
            nll_arr[pos:end] = nll_t.cpu().numpy()
            p_full[pos:end] = pf.cpu().numpy()
            p_nls[pos:end] = pn.cpu().numpy()
            p_LS[pos:end] = p_ls.cpu().numpy()
            pos = end
    print(f"[fwd] done, running mixer with byte-level capture...", flush=True)

    # PPM-D byte mixer with per-token detailed capture
    LN = math.log; LOG2 = LN(2.0); UNIFORM_LOGP = LN(1/256.0)
    order = 5; alpha = 15.0; beta = 0.80
    ctx_counts = {}
    window = bytearray()

    def _ppm(b, win, counts):
        log_p = None; conf = 0.0; seen_any = False; escape_log = 0.0
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            unique = len(cs); total = sum(cs.values()); denom = total + unique
            if not seen_any: conf = max(cs.values())/denom; seen_any = True
            if b in cs:
                log_p = escape_log + LN(cs[b]/denom); break
            escape_log += LN(unique/denom) if unique > 0 else 0.0
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

    # Per-token: nn_total_nll, mix_total_nll, gain, byte detail string
    per_tok_nn = np.zeros(N, dtype=np.float64)
    per_tok_mix = np.zeros(N, dtype=np.float64)
    per_tok_text = [None] * N
    per_tok_byte_detail = [None] * N

    for i in range(N):
        tid = int(targets[i]); pid = int(prevs[i])
        tb = token_bytes_lut[tid]
        has_space = bool(has_leading_space[tid])
        prev_is_boundary = pid < 0 or bool(is_boundary[pid])
        include_space = has_space and not prev_is_boundary
        token_bytes = []
        if include_space: token_bytes.append(0x20)
        token_bytes.extend(tb)
        n_bytes = len(token_bytes)
        if n_bytes == 0:
            per_tok_nn[i] = float(nll_arr[i])
            per_tok_mix[i] = float(nll_arr[i])
            continue

        b0 = token_bytes[0]
        ppm_log_b0, conf_b0 = _ppm(b0, window, ctx_counts)
        if prev_is_boundary:
            nn_p_b0 = float(p_full[i])
        elif include_space:
            nn_p_b0 = float(p_LS[i])
        else:
            nn_p_b0 = float(p_nls[i])
        nn_p_b0 = max(min(nn_p_b0, 1.0), 1e-30)
        ppm_p_b0 = max(min(math.exp(ppm_log_b0), 1.0), 0.0)
        z0 = alpha * (conf_b0 - beta)
        sig0 = 1.0/(1.0+math.exp(-z0)) if z0 >= 0 else math.exp(z0)/(1.0+math.exp(z0))
        lam0 = 1.0 - sig0
        p_mix0 = lam0 * nn_p_b0 + (1.0 - lam0) * ppm_p_b0
        nll_b0_mix = -LN(max(p_mix0, 1e-30))
        nll_b0_nn = -LN(max(nn_p_b0, 1e-30))
        _update(b0, window, ctx_counts)

        byte_detail = [(b0, -ppm_log_b0/LOG2, conf_b0, nll_b0_nn/LOG2, nll_b0_mix/LOG2, lam0)]

        nll_rem_mix = 0.0; nll_rem_nn_alone = 0.0
        if n_bytes > 1:
            ppm_rem_log = 0.0; confs = []
            ppm_per_byte = []
            for j in range(1, n_bytes):
                bj = token_bytes[j]
                lp, cf = _ppm(bj, window, ctx_counts)
                ppm_rem_log += lp; confs.append(cf)
                ppm_per_byte.append((bj, -lp/LOG2, cf))
                _update(bj, window, ctx_counts)
            nn_token_p = math.exp(-float(nll_arr[i]))
            nn_rem_p = max(nn_token_p / max(nn_p_b0, 1e-30), 1e-30)
            ppm_rem_p = math.exp(ppm_rem_log) if ppm_rem_log > -700 else 0.0
            mean_conf = sum(confs)/len(confs) if confs else 0.0
            zr = alpha * (mean_conf - beta)
            sigr = 1.0/(1.0+math.exp(-zr)) if zr >= 0 else math.exp(zr)/(1.0+math.exp(zr))
            lam_r = 1.0 - sigr
            p_mix_r = lam_r * nn_rem_p + (1.0 - lam_r) * ppm_rem_p
            nll_rem_mix = -LN(max(p_mix_r, 1e-30))
            nll_rem_nn_alone = -LN(max(nn_rem_p, 1e-30))
            for (bj, ppm_nll, cf) in ppm_per_byte:
                byte_detail.append((bj, ppm_nll, cf, None, None, lam_r))

        per_tok_nn[i] = nll_b0_nn + nll_rem_nn_alone   # NN's full token NLL split
        per_tok_mix[i] = nll_b0_mix + nll_rem_mix
        try: piece = sp.id_to_piece(tid)
        except Exception: piece = f"<id {tid}>"
        try: text = bytes(token_bytes).decode("utf-8")
        except Exception: text = repr(bytes(token_bytes))
        per_tok_text[i] = (piece, text, n_bytes)
        per_tok_byte_detail[i] = byte_detail

    gain = per_tok_nn - per_tok_mix   # positive = mix saved nats
    # Sort descending by gain
    valid_idx = np.array([i for i in range(N) if per_tok_text[i] is not None])
    sorted_idx = valid_idx[np.argsort(-gain[valid_idx])]

    print(f"\n=== TOP {args.top_k_gains} TOKENS BY MIX SAVINGS (NN_NLL - MIX_NLL) ===\n")
    for rank, idx in enumerate(sorted_idx[:args.top_k_gains]):
        piece, text, nb = per_tok_text[idx]
        nn_nll = per_tok_nn[idx]
        mix_nll = per_tok_mix[idx]
        g = gain[idx]
        # Show context around it
        ctx_lo = max(0, idx - 5)
        ctx_hi = min(N, idx + 1)
        ctx_pieces = []
        for j in range(ctx_lo, ctx_hi):
            try: ctx_pieces.append(sp.id_to_piece(int(targets[j])))
            except Exception: ctx_pieces.append("?")
        print(f"--- Rank #{rank+1}, val pos {idx} ---")
        print(f"context: {' '.join(ctx_pieces)}")
        print(f"current token: piece={repr(piece)[:30]:>30} bytes={repr(text)[:30]:>30} ({nb} bytes)")
        print(f"NN-only NLL:  {nn_nll/LOG2:8.4f} bits  ({nn_nll:.3f} nats)")
        print(f"MIX NLL:      {mix_nll/LOG2:8.4f} bits  ({mix_nll:.3f} nats)")
        print(f"SAVINGS:      {g/LOG2:+8.4f} bits  ({g:+.3f} nats)")
        print(f"byte breakdown:")
        for j, (b, ppm_nll, conf, nn_nll_b0, mix_nll_b0, lam) in enumerate(per_tok_byte_detail[idx]):
            ch = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            if j == 0:
                print(f"  byte[{j}] {ch!r:>6}: NN={nn_nll_b0:.3f}  PPM={ppm_nll:.3f}  conf={conf:.3f}  λ_NN={lam:.3f} → MIX={mix_nll_b0:.3f}  bits")
            else:
                print(f"  byte[{j}] {ch!r:>6}: PPM={ppm_nll:.3f}  conf={conf:.3f}  λ_NN={lam:.3f} bits  (rem chain-rule)")
        print()


if __name__ == "__main__":
    main()
