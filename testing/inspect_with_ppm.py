"""Forward-pass + PPM-D byte-mixture inspection.

Loads a post-EMA pre-quant final_model.pt, runs forward on N val tokens,
applies the PPM-D byte mixture (extracted from PR #1857), and outputs a
markdown report comparing NN-only vs NN+PPM bpb.

Run on a 1xH100 pod. Usage:
    python3 testing/inspect_with_ppm.py --ckpt <path> --train_gpt <path>
"""
import argparse, ctypes, importlib.util, math, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece as spm

PPM_C_SRC_FILE = Path(__file__).parent / "ppm_scorer.c"

# ------------------------- helpers -------------------------

def load_train_gpt_module(path):
    spec = importlib.util.spec_from_file_location("tg", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tg"] = m
    spec.loader.exec_module(m)
    return m

def seed_env_from_train_log(log_path):
    """Read 'Hyperparameters:' block from a train.log and set values as env vars
    BEFORE importing train_gpt.py. This makes the constructed model match the
    saved checkpoint's architecture exactly."""
    import re
    txt = Path(log_path).read_text()
    # Find the hparam block
    in_block = False
    n_set = 0
    for line in txt.split("\n"):
        if line.strip().startswith("Hyperparameters:"):
            in_block = True
            continue
        if in_block:
            if not line.startswith("  "):  # block ended
                break
            m = re.match(r"  (\w+): (.+)$", line)
            if not m: continue
            key, val = m.group(1).upper(), m.group(2).strip()
            # Convert booleans to 0/1 (env vars are strings)
            if val == "True": val = "1"
            elif val == "False": val = "0"
            elif val == "None": val = ""
            # Convert "4.0" -> "4" to match int() parsers (logger formats ints as floats)
            elif re.match(r"^-?\d+\.0$", val): val = val[:-2]
            os.environ[key] = val
            n_set += 1
    print(f"[hparams] seeded {n_set} env vars from {log_path}")

def build_token_bytes_lut(sp, vocab_size):
    sz = max(int(sp.vocab_size()), vocab_size)
    bytestrs = [b""] * sz
    has_space = np.zeros(sz, dtype=np.uint8)
    is_boundary = np.zeros(sz, dtype=np.uint8)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            is_boundary[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if sp.is_byte(tid):
            bytestrs[tid] = bytes([int(piece[3:-1], 16)])
            continue
        if piece.startswith("▁"):
            has_space[tid] = 1
            piece = piece[1:]
        bytestrs[tid] = piece.encode("utf-8")
    flat = b"".join(bytestrs)
    lens = np.array([len(b) for b in bytestrs], dtype=np.int32)
    offs = np.zeros(sz, dtype=np.int32)
    offs[1:] = np.cumsum(lens[:-1])
    return np.frombuffer(flat, dtype=np.uint8).copy(), offs, lens, has_space, is_boundary

def compile_ppm_lib(c_src_path):
    so_path = Path(tempfile.gettempdir()) / "ppm_scorer.so"
    cmd = ["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
           "-o", str(so_path), str(c_src_path), "-lm"]
    print(f"[compile] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so_path))
    lib.ppm_score_omp.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_uint32,
        ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ppm_score_omp.restype = ctypes.c_int
    return lib

def categorize(piece):
    p = piece[1:] if piece.startswith("▁") else piece
    if not p:
        return "empty"
    if any(s in p for s in ["http", "://", "www.", ".com", ".org", ".net", ".io", "@"]):
        return "URL"
    s = p.replace(".", "").replace(",", "").replace("-", "")
    if s and s.isdigit():
        return "NUMERIC"
    if any(c in p for c in ["{", "}", "[", "]", "()", ";", "==", "!=", "->", "::"]):
        return "CODE"
    if all(c in "0123456789abcdef" for c in p) and 4 <= len(p) <= 64:
        return "HEX"
    if "/" in p and all(c in "/_-" or c.isalnum() for c in p):
        return "PATH"
    return "PROSE"

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_gpt", required=True)
    ap.add_argument("--val_tok", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin")
    ap.add_argument("--val_bytes", default="/workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_bytes_000000.bin")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--out", default="/tmp/ppm_inspect.md")
    ap.add_argument("--ppm_order", type=int, default=4)
    ap.add_argument("--lambda_hi", type=float, default=0.9)
    ap.add_argument("--lambda_lo", type=float, default=0.05)
    ap.add_argument("--ppm_threshold", type=float, default=0.9)
    ap.add_argument("--ppm_omp_threads", type=int, default=8)
    ap.add_argument("--ppm_chunk_tokens", type=int, default=4194304)
    ap.add_argument("--ppm_log_cache", type=int, default=1048576)
    ap.add_argument("--train_log", default="", help="train.log to read hparams from (for env-var seeding)")
    args = ap.parse_args()

    # Seed env vars from train.log BEFORE importing train_gpt
    if args.train_log:
        seed_env_from_train_log(args.train_log)
    else:
        # Default: try to find a train.log next to the checkpoint
        cand = Path(args.ckpt).parent / "train.log"
        if cand.exists():
            seed_env_from_train_log(cand)

    device = torch.device("cuda")

    # ---- tokenizer ----
    tk_path = args.tokenizer
    if not tk_path:
        for p in [
            "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
            str(Path(args.train_gpt).parent / "tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"),
        ]:
            if Path(p).exists():
                tk_path = p
                break
    sp = spm.SentencePieceProcessor()
    sp.load(tk_path)
    vocab = sp.vocab_size()
    print(f"[tk] vocab={vocab}")

    # ---- val tokens ----
    # Format: 1024-byte header (256 int32s) + uint16 tokens
    raw = Path(args.val_tok).read_bytes()
    HEADER = 1024
    payload = raw[HEADER:]
    n_total = len(payload) // 2
    all_tokens = list(struct.unpack(f"<{n_total}H", payload))
    n = min(args.tokens, n_total)
    tokens = all_tokens[:n]
    print(f"[val] {n}/{n_total} tokens loaded (max id={max(tokens)})")
    assert max(tokens) < 8192, f"token id out of range: {max(tokens)} (expected < 8192)"

    # ---- val_bytes sidecar (authoritative bytes-per-token) ----
    val_bytes_per_tok_full = None
    if Path(args.val_bytes).exists():
        bts_raw = Path(args.val_bytes).read_bytes()
        # Same header convention; per-token byte counts as int32
        bts_payload = bts_raw[HEADER:]
        n_bts = len(bts_payload) // 4
        val_bytes_per_tok_full = np.frombuffer(bts_payload, dtype=np.int32)[:n]
        # Sanity: should be one int32 per token, total bytes ~150M for full val
        print(f"[val] sidecar bytes loaded: {n_bts} entries, sum first {n} = {val_bytes_per_tok_full.sum():,} bytes")
    else:
        print(f"[val] WARNING: sidecar {args.val_bytes} not found; will fall back to piece-encoding")

    # ---- model ----
    print(f"[model] importing {args.train_gpt}")
    mod = load_train_gpt_module(Path(args.train_gpt))
    H = mod.Hyperparameters()
    print(f"[model] constructing {mod.GPT.__name__}")
    # Try GPT(h) signature first (older banked-style); fall back to kwargs (newer style)
    try:
        model = mod.GPT(H).to(device).bfloat16()
        print("[model] constructed via GPT(h) signature")
    except TypeError as e:
        print(f"[model] GPT(h) failed ({e}); trying kwargs")
        gpt_kwargs = dict(
            vocab_size=H.vocab_size, num_layers=H.num_layers, model_dim=H.model_dim,
            num_heads=H.num_heads, num_kv_heads=H.num_kv_heads, mlp_mult=H.mlp_mult,
            tie_embeddings=H.tie_embeddings, tied_embed_init_std=H.tied_embed_init_std,
            logit_softcap=H.logit_softcap, rope_base=H.rope_base, qk_gain_init=H.qk_gain_init,
            recur_layers=H.recur_layers, recur_start_step=H.recur_start_step,
            parallel_start_layer=H.parallel_start_layer, rope_dims=H.rope_dims,
        )
        model = mod.GPT(**gpt_kwargs).to(device).bfloat16()
    # Mirror train_gpt.py post-construction precision tweaks
    if hasattr(mod, "CastedLinear"):
        for m in model.modules():
            if isinstance(m, mod.CastedLinear):
                m.float()
    for fn in ["restore_low_dim_params_to_fp32", "restore_fp32_params"]:
        if hasattr(mod, fn):
            getattr(mod, fn)(model)
            break
    print(f"[model] loading state from {args.ckpt}")
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "state_dict" in state: state = state["state_dict"]
        elif "model" in state: state = state["model"]
    try:
        model.load_state_dict(state, strict=True)
        print("[model] strict load OK")
    except Exception as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[model] non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) <= 5: print(f"  missing: {missing}")
        if len(unexpected) <= 5: print(f"  unexpected: {unexpected}")
    model.eval()
    # CRITICAL: enable loop layers (depth recurrence). Without this the model
    # runs in non-looped mode (much higher val_bpb, doesn't match training).
    if H.num_loops > 0 and hasattr(model, "looping_active"):
        model.looping_active = True
        print("[model] looping_active=True (depth recurrence enabled)")

    # ---- forward (chunked, MATCHES official eval_val: seq_len=2048 with varlen attn) ----
    seq_len = 2048
    n_pred = n - 1
    nll_nats = np.zeros(n_pred, dtype=np.float64)
    top_ids = np.zeros((n_pred, 5), dtype=np.int32)
    top_logp = np.zeros((n_pred, 5), dtype=np.float32)
    print(f"[fwd] running on {n} tokens in chunks of {seq_len} with varlen attention (BOS-aware)")

    # Find _build_cu_seqlens helper from the train_gpt module
    build_cu = getattr(mod, "_build_cu_seqlens", None)
    if build_cu is None:
        print("[fwd] WARNING: _build_cu_seqlens not found; falling back to no-mask attention")
    else:
        print("[fwd] using mod._build_cu_seqlens (BOS-aware varlen attention)")

    BOS_ID_VAL = getattr(mod, "BOS_ID", None) or 1
    t0 = time.time()
    pos = 0
    # We chunk so that each chunk has exactly seq_len input tokens (and seq_len target tokens shifted by 1).
    # Mirrors official eval: x = local[:-1], y = local[1:], chunks span seq_len each.
    while pos < n_pred:
        # Take seq_len+1 raw tokens; x = first seq_len, y = last seq_len shifted by 1
        end = min(pos + seq_len + 1, n)
        if end - pos < 2:
            break
        chunk_tokens = tokens[pos:end]
        x_t = torch.tensor(chunk_tokens[:-1], dtype=torch.long, device=device)
        # Build cu_seqlens based on BOS positions in x
        cu_seqlens, max_seqlen = None, 0
        if build_cu is not None:
            bos_pos = (x_t == BOS_ID_VAL).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = build_cu(bos_pos, x_t.numel(), x_t.device, seq_len, 64)
        with torch.no_grad():
            if cu_seqlens is not None:
                logits = model.forward_logits(x_t[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            else:
                logits = model.forward_logits(x_t[None])
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tgt = torch.tensor(chunk_tokens[1:], dtype=torch.long, device=device)
            nll_chunk = -log_probs[0].gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            n_chunk = nll_chunk.shape[0]
            nll_nats[pos:pos + n_chunk] = nll_chunk.cpu().numpy()
            tk = log_probs[0].topk(5, dim=-1)
            top_ids[pos:pos + n_chunk] = tk.indices.cpu().numpy()
            top_logp[pos:pos + n_chunk] = tk.values.cpu().numpy()
        pos += seq_len
    print(f"[fwd] done in {time.time()-t0:.1f}s; mean NLL/tok = {nll_nats.mean():.4f} nats")

    # ---- build PPM args ----
    print(f"[ppm] building byte LUT (vocab={vocab})")
    flat, offs, lens, has_space, is_boundary = build_token_bytes_lut(sp, vocab)
    target_ids = np.array(tokens[1:n], dtype=np.int64)         # next-token at each pos
    prev_ids = np.array(tokens[0:n-1], dtype=np.int64)         # previous token
    print(f"[ppm] target shape={target_ids.shape} flat={len(flat)}B")

    # Per-token NN log-prob = -nll_nats; passed to scorer as nll (positive)
    # The scorer expects nll in nats, will divide by n_bytes internally.

    out = np.zeros(6, dtype=np.float64)
    lib = compile_ppm_lib(PPM_C_SRC_FILE)

    print(f"[ppm] order={args.ppm_order} λ_hi={args.lambda_hi} λ_lo={args.lambda_lo} thr={args.ppm_threshold}")
    t1 = time.time()
    rc = lib.ppm_score_omp(
        target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        nll_nats.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(len(target_ids)),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        has_space.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        is_boundary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(vocab),
        ctypes.c_int(args.ppm_order),
        ctypes.c_double(args.lambda_hi),
        ctypes.c_double(args.lambda_lo),
        ctypes.c_double(args.ppm_threshold),
        ctypes.c_uint32(args.ppm_log_cache),
        ctypes.c_int64(args.ppm_chunk_tokens),
        ctypes.c_int(args.ppm_omp_threads),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    print(f"[ppm] score_omp returned {rc} in {time.time()-t1:.1f}s")
    if rc != 0:
        print(f"[ppm] ERROR: rc={rc}")
        sys.exit(1)

    mix_bpb, ppm_only_bpb, nn_byte_bpb, token_bpb, n_bytes, gate_high_frac = out
    print(f"[ppm] mix_bpb={mix_bpb:.5f} ppm_only={ppm_only_bpb:.5f} nn_byte_bpb={nn_byte_bpb:.5f} bytes={int(n_bytes)} gate_high={gate_high_frac:.4f}")

    # ---- SAVE RAW DATA IMMEDIATELY (before any report-writing that could crash) ----
    npz_path = Path(args.out).with_suffix(".npz")
    save_dict = dict(
        tokens=np.array(tokens, dtype=np.int32),
        nll_nats=nll_nats.astype(np.float32),
        top_ids=top_ids,
        top_logp=top_logp,
        ppm_out=out,
        ckpt_path=np.array([args.ckpt]),
        ppm_config=np.array([args.ppm_order, args.lambda_hi, args.lambda_lo, args.ppm_threshold], dtype=np.float64),
    )
    if val_bytes_per_tok_full is not None:
        save_dict["val_bytes_per_tok"] = val_bytes_per_tok_full.astype(np.int32)
    np.savez_compressed(npz_path, **save_dict)
    print(f"[save] raw data → {npz_path} ({npz_path.stat().st_size//1024//1024} MB)")

    # ---- official-style val_bpb computation using sidecar bytes ----
    if val_bytes_per_tok_full is not None:
        # Match official: byte budget for target tokens (positions 1..n-1)
        target_bytes = val_bytes_per_tok_full[1:n].astype(np.float64)
        total_nll_bits = nll_nats.sum() / math.log(2.0)
        total_bytes_official = float(target_bytes.sum())
        nn_bpb_official = total_nll_bits / total_bytes_official
        print(f"[official] NN val_bpb (sidecar bytes, varlen attn) = {nn_bpb_official:.5f}")
        print(f"           total bits = {total_nll_bits:,.0f}, total bytes = {int(total_bytes_official):,}")

    # ---- failure analysis breakdown by token category ----
    pieces = [sp.id_to_piece(t) if t < vocab else "<oor>" for t in tokens]
    actual_pieces = pieces[1:n]
    cats = [categorize(p) for p in actual_pieces]
    NLL_bits = nll_nats / math.log(2)
    bytes_per = lens[target_ids]
    bpb_per = NLL_bits / np.maximum(bytes_per, 1)

    from collections import defaultdict
    cat_count = defaultdict(int)
    cat_total_bits = defaultdict(float)
    cat_total_bytes = defaultdict(int)
    for i, c in enumerate(cats):
        cat_count[c] += 1
        cat_total_bits[c] += float(NLL_bits[i])
        cat_total_bytes[c] += int(bytes_per[i])
    total_bits = float(NLL_bits.sum())
    total_bytes_meas = int(bytes_per.sum())
    measured_nn_bpb = total_bits / max(total_bytes_meas, 1)

    # NLL distribution buckets
    buckets = [(0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 999)]
    bucket_count = [0]*len(buckets); bucket_sum = [0.0]*len(buckets)
    for x in bpb_per:
        for j, (lo, hi) in enumerate(buckets):
            if lo <= x < hi:
                bucket_count[j] += 1
                bucket_sum[j] += float(x)
                break

    # Top/bottom NLL positions for context
    sorted_idx = np.argsort(-NLL_bits)
    worst = sorted_idx[:25]
    best = sorted_idx[-25:][::-1]

    def ctx_str(pos, k=25):
        end = pos + 1
        start = max(0, end - k)
        s = "".join(pieces[i].replace("▁", " ") for i in range(start, end))
        return s.replace("\n", "↵")[-80:]

    # ---- write report ----
    out_lines = []
    out_lines.append(f"# PPM-D byte-mixture inspection")
    out_lines.append(f"")
    out_lines.append(f"**Checkpoint:** `{args.ckpt}`")
    out_lines.append(f"**Tokens scored:** {len(target_ids)} (bytes: {int(n_bytes)})")
    out_lines.append(f"**PPM config:** order={args.ppm_order} λ_hi={args.lambda_hi} λ_lo={args.lambda_lo} threshold={args.ppm_threshold}")
    out_lines.append(f"")
    out_lines.append(f"## Headline")
    out_lines.append(f"")
    out_lines.append(f"| Metric | bits/byte |")
    out_lines.append(f"|---|---:|")
    out_lines.append(f"| **NN only** (`nn_byte_bpb`) | **{nn_byte_bpb:.5f}** |")
    out_lines.append(f"| PPM only (`ppm_only`) | {ppm_only_bpb:.5f} |")
    out_lines.append(f"| **NN + PPM mix** (`mix_bpb`) | **{mix_bpb:.5f}** |")
    out_lines.append(f"| **Δ from PPM** | **{mix_bpb - nn_byte_bpb:+.5f}** |")
    out_lines.append(f"| Gate high-confidence fraction | {gate_high_frac:.4f} ({100*gate_high_frac:.1f}%) |")
    out_lines.append(f"| Token-level reference (`token_bpb`) | {token_bpb:.5f} |")
    out_lines.append(f"")
    out_lines.append(f"## Comparison to dexhunter's #1857")
    out_lines.append(f"")
    out_lines.append(f"| | #1857 | this run |")
    out_lines.append(f"|---|---:|---:|")
    out_lines.append(f"| nn_byte_bpb | 1.10020 | {nn_byte_bpb:.5f} |")
    out_lines.append(f"| ppm_only | 2.34028 | {ppm_only_bpb:.5f} |")
    out_lines.append(f"| mix_bpb | 1.03176 | {mix_bpb:.5f} |")
    out_lines.append(f"| gate_high_frac | 0.14241 | {gate_high_frac:.5f} |")
    out_lines.append(f"| Δ from PPM | -0.06844 | {mix_bpb - nn_byte_bpb:+.5f} |")
    out_lines.append(f"")
    out_lines.append(f"## Per-category contribution to NN-only val_bpb")
    out_lines.append(f"")
    out_lines.append(f"| category | count | % positions | mean bits/byte | bits | % NN val_bpb |")
    out_lines.append(f"|---|---:|---:|---:|---:|---:|")
    for cat in sorted(cat_count, key=lambda c: -cat_total_bits[c]):
        c = cat_count[cat]
        pct = 100*c/max(len(cats),1)
        bb = cat_total_bytes[cat]
        mean = cat_total_bits[cat] / max(bb, 1) if bb else 0
        contrib = 100 * cat_total_bits[cat] / max(total_bits, 1e-9)
        out_lines.append(f"| {cat} | {c} | {pct:.1f}% | {mean:.2f} | {cat_total_bits[cat]:.1f} | {contrib:.1f}% |")
    out_lines.append(f"")
    ppm_addr_pct = 100 * sum(cat_total_bits[c] for c in ['URL','NUMERIC','CODE','HEX','PATH']) / max(total_bits, 1e-9)
    out_lines.append(f"**PPM-addressable categories (URL+NUMERIC+CODE+HEX+PATH): {ppm_addr_pct:.1f}% of NN val_bpb**")
    out_lines.append(f"")
    out_lines.append(f"## NLL distribution (NN only, bits/byte)")
    out_lines.append(f"")
    out_lines.append(f"| bucket | count | % | mean | contribution |")
    out_lines.append(f"|---|---:|---:|---:|---:|")
    for j, (lo, hi) in enumerate(buckets):
        c = bucket_count[j]; pct = 100*c/max(len(cats),1)
        mean = bucket_sum[j]/max(c,1)
        contrib = bucket_sum[j]/max(len(cats),1)
        contrib_pct = 100*contrib/max(measured_nn_bpb, 1e-9)
        hi_str = "∞" if hi > 100 else f"{hi:.1f}"
        out_lines.append(f"| {lo:.1f}–{hi_str} | {c} | {pct:.1f}% | {mean:.3f} | {contrib:.4f} ({contrib_pct:.1f}%) |")
    out_lines.append(f"")
    out_lines.append(f"## Top-50 most catastrophic NN predictions")
    out_lines.append(f"")
    out_lines.append(f"These are the bytes where the model was most surprised — high NLL means the model assigned ~0% probability to what actually came next.")
    out_lines.append(f"")
    out_lines.append(f"| pos | NLL (bits) | actual | top-1 prediction (prob) | category | left context (last 50 chars) |")
    out_lines.append(f"|---|---:|---|---|---|---|")
    for pos in sorted_idx[:50]:
        top1_id = int(top_ids[pos][0])
        top1_piece = sp.id_to_piece(top1_id) if top1_id < vocab else "<oor>"
        top1_p = math.exp(float(top_logp[pos][0]))
        out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | `{top1_piece!r}` ({top1_p:.3f}) | {cats[pos]} | `{ctx_str(int(pos), k=50)}` |")
    out_lines.append(f"")
    out_lines.append(f"## Worst-10 per category")
    out_lines.append(f"")
    out_lines.append(f"Where each kind of byte fails most. PPM-addressable categories (URL/NUMERIC/CODE) are exactly where PPM helps.")
    out_lines.append(f"")
    for target_cat in ["URL", "NUMERIC", "CODE", "HEX", "PATH", "PROSE"]:
        cat_positions = [i for i, c in enumerate(cats) if c == target_cat]
        if not cat_positions: continue
        cat_sorted = sorted(cat_positions, key=lambda i: -NLL_bits[i])[:10]
        out_lines.append(f"### {target_cat} ({len(cat_positions)} positions, mean NLL {sum(NLL_bits[i] for i in cat_positions)/len(cat_positions):.2f} bits)")
        out_lines.append(f"")
        out_lines.append(f"| pos | NLL | actual | top-1 (prob) | left context |")
        out_lines.append(f"|---|---:|---|---|---|")
        for pos in cat_sorted:
            top1_id = int(top_ids[pos][0])
            top1_piece = sp.id_to_piece(top1_id) if top1_id < vocab else "<oor>"
            top1_p = math.exp(float(top_logp[pos][0]))
            out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | `{top1_piece!r}` ({top1_p:.3f}) | `{ctx_str(int(pos), k=50)}` |")
        out_lines.append(f"")
    out_lines.append(f"## Top-25 best NN predictions (contrast)")
    out_lines.append(f"")
    out_lines.append(f"| pos | NLL (bits) | actual | category | left context |")
    out_lines.append(f"|---|---:|---|---|---|")
    for pos in best:
        out_lines.append(f"| {int(pos)} | {NLL_bits[pos]:.2f} | `{actual_pieces[pos]!r}` | {cats[pos]} | `{ctx_str(int(pos))}` |")

    Path(args.out).write_text("\n".join(out_lines))
    print(f"\n[done] wrote {len(out_lines)} lines to {args.out}")

if __name__ == "__main__":
    main()
