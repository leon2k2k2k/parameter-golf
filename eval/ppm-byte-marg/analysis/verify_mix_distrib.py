"""Same as verify_mix_dist but bucketed by position decile + dump 3 examples per decile."""

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
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--val_tokens_bin", required=True)
    ap.add_argument("--val_bytes_bin", required=True); ap.add_argument("--n_tokens", type=int, default=1_000_000)
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

    raw = np.fromfile(args.val_tokens_bin, dtype=np.uint16)
    arr = raw[906:906 + args.n_tokens].astype(np.int64)
    N = len(arr) - 1

    print(f"[fwd] N={N}", flush=True)
    byte_dist_all = np.empty((N, 256), dtype=np.float32)
    non_ls_dist_all = np.empty((N, 256), dtype=np.float32)
    p_ls_all = np.empty(N, dtype=np.float32); nll_all = np.empty(N, dtype=np.float32)
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
            byte_dist_all[pos:end] = byte_dist.cpu().numpy()
            non_ls_dist_all[pos:end] = non_ls_dist.cpu().numpy()
            p_ls_all[pos:end] = p_ls.cpu().numpy(); nll_all[pos:end] = nll_t.cpu().numpy()
            pos = end
    print(f"[fwd] done", flush=True)

    LN = math.log; LOG2 = LN(2.0)
    order = 5; ctx_counts = {}; window = bytearray()

    def _ppm_full_dist(win, counts):
        dist = np.zeros(256, dtype=np.float64); remaining = 1.0; seen_overall = set()
        for K in range(min(order, len(win)), -1, -1):
            ctx = bytes(win[-K:]) if K > 0 else b""
            cs = counts.get(ctx)
            if cs is None: continue
            unique = len(cs); total = sum(cs.values()); denom = total + unique
            for b, c in cs.items():
                if b not in seen_overall:
                    dist[b] += remaining * (c / denom); seen_overall.add(b)
            esc = unique / denom if unique > 0 else 1.0
            remaining *= esc
        if remaining > 0:
            for b in range(256):
                if b not in seen_overall: dist[b] += remaining * (1/256.)
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

    # Track savings + which decile
    n_deciles = 10
    decile_savings_sum = np.zeros(n_deciles)
    decile_savings_count = np.zeros(n_deciles, dtype=np.int64)
    decile_byte_count = np.zeros(n_deciles, dtype=np.int64)
    # Examples per decile
    decile_examples = {d: [] for d in range(n_deciles)}

    for i in range(N):
        tid = int(arr[i+1]); pid = int(arr[i])
        tb = tb_lut[tid]
        has_space = bool(has_ls[tid])
        prev_is_boundary = pid < 0 or bool(is_bdy[pid])
        include_space = has_space and not prev_is_boundary
        token_bytes = []
        if include_space: token_bytes.append(0x20)
        token_bytes.extend(tb)
        if not token_bytes: continue
        b0 = token_bytes[0]
        if prev_is_boundary:
            nn_dist = byte_dist_all[i].astype(np.float64)
        else:
            nn_dist = non_ls_dist_all[i].astype(np.float64).copy()
            nn_dist[0x20] = float(p_ls_all[i])
        sc = nn_dist.sum()
        if sc <= 0:
            for j in range(len(token_bytes)): _update(token_bytes[j], window, ctx_counts)
            continue
        nn_dist_renorm = nn_dist / sc
        ppm_dist = _ppm_full_dist(window, ctx_counts)
        ppm_dist = ppm_dist / max(ppm_dist.sum(), 1e-30)
        conf = _ppm_conf(window, ctx_counts)
        z = 15.0 * (conf - 0.80)
        sig = 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))
        lam = 1.0 - sig
        mix_dist = lam * nn_dist_renorm + (1.0 - lam) * ppm_dist
        nn_p = max(nn_dist_renorm[b0], 1e-30); mix_p = max(mix_dist[b0], 1e-30)
        savings_bits = (-math.log(nn_p) - (-math.log(mix_p))) / LOG2
        d = min(int(i * n_deciles / N), n_deciles - 1)
        decile_byte_count[d] += 1
        if savings_bits > 0.5:
            decile_savings_sum[d] += savings_bits
            decile_savings_count[d] += 1
            try: piece = sp.id_to_piece(tid)
            except: piece = f"<{tid}>"
            try: ctx_str = bytes(window).decode("utf-8", errors="replace")
            except: ctx_str = repr(bytes(window))
            decile_examples[d].append((savings_bits, i, b0, lam, conf, nn_dist_renorm.copy(), ppm_dist.copy(), mix_dist.copy(), piece, ctx_str))

        for j in range(len(token_bytes)): _update(token_bytes[j], window, ctx_counts)

    print(f"\n=== SAVINGS BY POSITION DECILE ===")
    print(f"{'decile':<10} {'tok_range':<25} {'n_byte_tok':<12} {'n_high_save':<12} {'sum_save_bits':<15} {'mean_save/tok':<15}")
    for d in range(n_deciles):
        lo = d * N // n_deciles; hi = (d+1) * N // n_deciles
        mean = decile_savings_sum[d] / max(decile_byte_count[d], 1)
        print(f"  {d:<8} [{lo:>7d},{hi:>7d}]   {decile_byte_count[d]:<12d} {decile_savings_count[d]:<12d} {decile_savings_sum[d]:<15.1f} {mean:<15.4f}")

    print(f"\n=== TOP 2 EXAMPLES PER DECILE (positions ≥ decile 1) ===")
    for d in range(1, n_deciles):
        ex = sorted(decile_examples[d], reverse=True)[:2]
        if not ex: continue
        print(f"\n--- DECILE {d} (pos {d*N//n_deciles}..{(d+1)*N//n_deciles}) ---")
        for sv, i, b0, lam, conf, nn_d, ppm_d, mix_d, piece, ctx_str in ex:
            ch = chr(b0) if 32 <= b0 < 127 else f"\\x{b0:02x}"
            print(f"  pos={i} piece={piece!r} byte_0={ch!r}  PPM_conf={conf:.3f}  λ_NN={lam:.3f}  saved={sv:.2f} bits")
            print(f"    ctx='{ctx_str}'")
            nn_top = np.argsort(nn_d)[::-1][:5]
            ppm_top = np.argsort(ppm_d)[::-1][:5]
            print(f"    NN  top5: " + "  ".join(f"{(chr(b) if 32<=b<127 else f'\\x{b:02x}')!r}={nn_d[b]:.3f}" for b in nn_top))
            print(f"    PPM top5: " + "  ".join(f"{(chr(b) if 32<=b<127 else f'\\x{b:02x}')!r}={ppm_d[b]:.3f}" for b in ppm_top))


if __name__ == "__main__":
    main()
