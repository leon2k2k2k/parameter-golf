"""Dump top 50 byte_0 savings positions with bigger byte context (last 30 bytes) for manual categorization."""

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
    ap.add_argument("--top_k", type=int, default=50)
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
    p_ls_all = np.empty(N, dtype=np.float32)
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
            byte_dist = probs @ fbm_t; non_ls_dist = probs @ nls_t; p_ls = probs @ lsm_t
            byte_dist_all[pos:end] = byte_dist.cpu().numpy()
            non_ls_dist_all[pos:end] = non_ls_dist.cpu().numpy()
            p_ls_all[pos:end] = p_ls.cpu().numpy()
            pos = end
    print(f"[fwd] done", flush=True)

    # Build full byte stream + per-token byte ranges
    print(f"[stream] building byte stream", flush=True)
    byte_stream = bytearray()
    per_pos_b0_offset = np.full(N, -1, dtype=np.int64)
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
        per_pos_b0_offset[i] = len(byte_stream)
        for b in token_bytes: byte_stream.append(b)

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

    candidates = []
    for i in range(N):
        if per_pos_b0_offset[i] < 0: continue
        tid = int(arr[i+1]); pid = int(arr[i])
        tb = tb_lut[tid]
        has_space = bool(has_ls[tid])
        prev_is_boundary = pid < 0 or bool(is_bdy[pid])
        include_space = has_space and not prev_is_boundary
        token_bytes = []
        if include_space: token_bytes.append(0x20)
        token_bytes.extend(tb)
        b0 = token_bytes[0]
        if prev_is_boundary:
            nn_dist = byte_dist_all[i].astype(np.float64)
        else:
            nn_dist = non_ls_dist_all[i].astype(np.float64).copy()
            nn_dist[0x20] = float(p_ls_all[i])
        sc = nn_dist.sum()
        if sc <= 0:
            for b in token_bytes: _update(b, window, ctx_counts)
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
        if savings_bits > 3.0:
            try: piece = sp.id_to_piece(tid)
            except: piece = f"<{tid}>"
            candidates.append((savings_bits, i, b0, lam, conf, nn_dist_renorm, ppm_dist, piece))
        for b in token_bytes: _update(b, window, ctx_counts)

    candidates.sort(reverse=True)
    print(f"\n=== TOP {args.top_k} BYTE_0 SAVINGS POSITIONS ===")
    print(f"{'rank':<5} {'pos':<8} {'piece':<14} {'b0':<6} {'saved':<6} {'conf':<6} {'lamNN':<6} {'preceding 25 bytes':<30} {'next 8 bytes':<15}")
    for k, (sv, i, b0, lam, conf, nn_d, ppm_d, piece) in enumerate(candidates[:args.top_k]):
        ch = chr(b0) if 32 <= b0 < 127 else f"\\x{b0:02x}"
        ofs = per_pos_b0_offset[i]
        ctx = bytes(byte_stream[max(0, ofs-25):ofs])
        nxt = bytes(byte_stream[ofs:ofs+8])
        try: ctx_s = ctx.decode("utf-8", errors="replace")
        except: ctx_s = repr(ctx)
        try: nxt_s = nxt.decode("utf-8", errors="replace")
        except: nxt_s = repr(nxt)
        nn_top1 = int(np.argmax(nn_d)); ppm_top1 = int(np.argmax(ppm_d))
        nn_top1_ch = chr(nn_top1) if 32 <= nn_top1 < 127 else f"\\x{nn_top1:02x}"
        ppm_top1_ch = chr(ppm_top1) if 32 <= ppm_top1 < 127 else f"\\x{ppm_top1:02x}"
        print(f"{k+1:<5} {i:<8} {piece!r:<14} {ch!r:<6} {sv:<6.2f} {conf:<6.2f} {lam:<6.3f} ctx={ctx_s!r:<30} nxt={nxt_s!r:<15}  NN→{nn_top1_ch!r}({nn_d[nn_top1]:.2f})  PPM→{ppm_top1_ch!r}({ppm_d[ppm_top1]:.2f})")


if __name__ == "__main__":
    main()
