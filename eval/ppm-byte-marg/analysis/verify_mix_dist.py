"""Verify byte_0 mix distribution sums to 1, then dump examples where PPM saves
the most NLL at byte_0. For each example: realized byte, lambda, top-5 NN, top-5 PPM, top-5 mix."""

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
    ap.add_argument("--n_tokens", type=int, default=200000)
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_seqs", type=int, default=8)
    args = ap.parse_args()

    for k in ["SMEAR_GATE_ENABLED","SPARSE_ATTN_GATE_ENABLED","LQER_ENABLED","LQER_ASYM_ENABLED","ASYM_LOGIT_RESCALE","CASEOPS_ENABLED"]:
        os.environ.setdefault(k, "1")

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

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    # Forward, store full byte_dist[256], non_ls_dist[256], p_ls per position
    print(f"[fwd] N={N}", flush=True)
    byte_dist_all = np.empty((N, 256), dtype=np.float32)
    non_ls_dist_all = np.empty((N, 256), dtype=np.float32)
    p_ls_all = np.empty(N, dtype=np.float32)
    nll_all = np.empty(N, dtype=np.float32)
    chunk = args.seq_len * args.batch_seqs
    pos = 0
    BOS_ID = 1
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
            byte_dist_all[pos:end] = byte_dist.cpu().numpy()
            non_ls_dist_all[pos:end] = non_ls_dist.cpu().numpy()
            p_ls_all[pos:end] = p_ls.cpu().numpy()
            nll_all[pos:end] = nll_t.cpu().numpy()
            pos = end
    print(f"[fwd] done", flush=True)

    # Now run PPM and at each byte_0 position record:
    #   - nn_dist_byte0 (256, after renorm)
    #   - ppm_dist_byte0 (256)
    #   - lambda
    #   - mix_dist_byte0 (256)
    #   - sum check on mix_dist
    #   - savings: -log P_NN(b0) - (-log P_mix(b0))
    LN = math.log; LOG2 = LN(2.0); UNIFORM_LOGP = LN(1/256.)
    order = 5
    ctx_counts = {}
    window = bytearray()

    def _ppm_full_dist(win, counts):
        """Returns the full 256-dim PPM-D distribution at this byte position."""
        # Use Method-D escape blending across orders
        # Final dist[b] = sum over orders of escape_to_order * P(b | order) (one-shot escape model)
        dist = np.zeros(256, dtype=np.float64)
        # We'll do it the same way scoring does: walk down from longest, accumulate escape prob,
        # and at each order add escape*(count[b]/denom) for b's seen at that order, then escape.
        remaining = 1.0
        seen_overall = set()
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            unique = len(cs); total = sum(cs.values()); denom = total + unique
            # mass available at this order = remaining
            for b, c in cs.items():
                if b not in seen_overall:
                    dist[b] += remaining * (c / denom)
                    seen_overall.add(b)
            esc = unique / denom if unique > 0 else 1.0
            remaining *= esc
        # uniform fallback for unseen
        if remaining > 0:
            for b in range(256):
                if b not in seen_overall:
                    dist[b] += remaining * (1/256.)
        return dist

    def _ppm_conf(win, counts):
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            denom = sum(cs.values()) + len(cs)
            return max(cs.values()) / denom
        return 0.0

    def _update(b, win, counts):
        for K in range(0, min(order, len(win)) + 1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            d = counts.get(ctx)
            if d is None: d = {}; counts[ctx] = d
            d[b] = d.get(b, 0) + 1
        win.append(b)
        if len(win) > order: del win[0]

    # Track top savings positions
    candidates = []  # (savings_bits, pos, b0, lambda, nn_dist, ppm_dist, sum_mix)
    sum_checks_seen = []

    print(f"[ppm] scanning {N} positions for examples...", flush=True)
    n_byte_content = 0
    sum_dev_max = 0.0
    for i in range(N):
        tid = int(arr[i+1]); pid = int(arr[i])
        tb = tb_lut[tid]
        has_space = bool(has_ls[tid])
        prev_is_boundary = pid < 0 or bool(is_bdy[pid])
        include_space = has_space and not prev_is_boundary
        token_bytes = []
        if include_space: token_bytes.append(0x20)
        token_bytes.extend(tb)
        if not token_bytes:
            continue
        n_byte_content += 1
        b0 = token_bytes[0]

        # Build NN byte_0 dist over 256
        if prev_is_boundary:
            nn_dist = byte_dist_all[i].astype(np.float64)
        else:
            nn_dist = non_ls_dist_all[i].astype(np.float64).copy()
            nn_dist[0x20] = float(p_ls_all[i])
        sc = nn_dist.sum()
        if sc <= 0: continue
        nn_dist_renorm = nn_dist / sc

        # PPM dist
        ppm_dist = _ppm_full_dist(window, ctx_counts)
        ppm_sum = ppm_dist.sum()
        if abs(ppm_sum - 1.0) > 1e-3:
            # renormalize defensively
            ppm_dist = ppm_dist / max(ppm_sum, 1e-30)

        conf = _ppm_conf(window, ctx_counts)
        z = 15.0 * (conf - 0.80)
        sig = 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))
        lam = 1.0 - sig
        mix_dist = lam * nn_dist_renorm + (1.0 - lam) * ppm_dist
        sum_mix = mix_dist.sum()
        sum_dev_max = max(sum_dev_max, abs(sum_mix - 1.0))

        nn_p_b0 = max(nn_dist_renorm[b0], 1e-30)
        mix_p_b0 = max(mix_dist[b0], 1e-30)
        savings_bits = (-math.log(nn_p_b0) - (-math.log(mix_p_b0))) / LOG2

        if savings_bits > 0.5:
            try: piece = sp.id_to_piece(tid)
            except: piece = f"<{tid}>"
            candidates.append((
                savings_bits, i, b0, lam, conf,
                nn_dist_renorm.copy(), ppm_dist.copy(), mix_dist.copy(),
                sum_mix, piece, bytes(window)
            ))

        _update(b0, window, ctx_counts)
        # advance through remainder bytes for state update
        for j in range(1, len(token_bytes)):
            _update(token_bytes[j], window, ctx_counts)

    print(f"\n[verify] mix_dist sum max deviation from 1.0: {sum_dev_max:.2e}")
    print(f"[verify] {n_byte_content} byte-content tokens; {len(candidates)} positions with >0.5 bit saved at byte_0")

    candidates.sort(reverse=True)
    print(f"\n=== TOP {args.n_examples} POSITIONS WHERE PPM SAVED MOST AT BYTE_0 ===\n")
    for k, (sv, i, b0, lam, conf, nn_d, ppm_d, mix_d, sm, piece, ctx_window) in enumerate(candidates[:args.n_examples]):
        ch = chr(b0) if 32 <= b0 < 127 else f"\\x{b0:02x}"
        try: ctx_str = ctx_window.decode("utf-8", errors="replace")
        except: ctx_str = repr(ctx_window)
        print(f"--- Example {k+1}: pos={i} piece={piece!r} realized byte_0={ch!r} (0x{b0:02x}) ---")
        print(f"    Recent byte context (last 5): {ctx_str!r}")
        print(f"    PPM confidence = {conf:.4f}  →  λ_NN = {lam:.4f}  (1−λ = {1-lam:.4f} on PPM)")
        print(f"    sum(mix_dist) = {sm:.10f}  (sanity: should be 1.0)")
        print(f"    NLL at byte_0 — NN: {-math.log(max(nn_d[b0],1e-30))/LOG2:.3f} bits   MIX: {-math.log(max(mix_d[b0],1e-30))/LOG2:.3f} bits   savings: {sv:.3f} bits")
        # Top-5 NN
        nn_top = np.argsort(nn_d)[::-1][:5]
        print(f"    Top-5 NN:   ", end="")
        for b in nn_top:
            ch_b = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            print(f"{ch_b!r}={nn_d[b]:.4f}", end="  ")
        print()
        # Top-5 PPM
        ppm_top = np.argsort(ppm_d)[::-1][:5]
        print(f"    Top-5 PPM:  ", end="")
        for b in ppm_top:
            ch_b = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            print(f"{ch_b!r}={ppm_d[b]:.4f}", end="  ")
        print()
        # Top-5 MIX
        mix_top = np.argsort(mix_d)[::-1][:5]
        print(f"    Top-5 MIX:  ", end="")
        for b in mix_top:
            ch_b = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
            print(f"{ch_b!r}={mix_d[b]:.4f}", end="  ")
        print()
        print()


if __name__ == "__main__":
    main()
