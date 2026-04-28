"""Patch 052's train_gpt.py to add 1850's PPM-D + our anti-hijack mod.

052 base = records/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py + per-pass FFN scale.
Has TTT_EVAL_ONLY mode but not full EVAL_ONLY. We piggyback on TTT_EVAL_ONLY=1
to skip training + GPTQ, load artifacts, run diagnostic quantized eval, then
fire our PPM hook.
"""
import sys
from pathlib import Path

SRC = "worktrees/055-050-with-ppm-fullrun/records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"
C_SRC = "worktrees/055-050-with-ppm-fullrun/testing/ppm_scorer_antihijack.c"
OUT = "worktrees/055-050-with-ppm-fullrun/records/track_10min_16mb/2026-04-27_050_PR1797_Base_BOS_Fix/train_gpt.py"

src_text = Path(SRC).read_text()
c_src = Path(C_SRC).read_text()

# 1. Add ctypes, tempfile, hashlib to top imports
src_text = src_text.replace(
    "import base64, collections, copy, fcntl, glob, io, lzma, math, os",
    "import base64, collections, copy, ctypes, fcntl, glob, hashlib, io, lzma, math, os, tempfile",
    1,
)

# 2. Add PPM hyperparameters. Find an anchor near the end of Hyperparameters fields.
ppm_hparams = """    # PPM-D byte mixture (port from PR #1850 + our anti-hijack tuning).
    ppm_native_enabled = bool(int(os.environ.get("PPM_NATIVE_ENABLED", "0")))
    ppm_order = int(os.environ.get("PPM_ORDER", 4))
    ppm_lambda_hi = float(os.environ.get("PPM_LAMBDA_HI", 0.9))
    ppm_lambda_lo = float(os.environ.get("PPM_LAMBDA_LO", 0.05))
    ppm_conf_threshold = float(os.environ.get("PPM_CONF_THRESHOLD", 0.76))
    ppm_nn_skip_thr_nats = float(os.environ.get("PPM_NN_SKIP_THR_NATS", 0.277))
    ppm_log_cache_size = int(os.environ.get("PPM_LOG_CACHE_SIZE", 1048576))
    ppm_omp_threads = int(os.environ.get("PPM_OMP_THREADS", 8))
    ppm_omp_chunk_tokens = int(os.environ.get("PPM_OMP_CHUNK_TOKENS", 4194304))
"""

# Insert after the ttt_eval_only line in Hyperparameters (find a stable anchor)
# Use the eval_seq_len line that's in Hyperparameters
anchor_line = '    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 48))'
if anchor_line not in src_text:
    # try alternate
    print(f"Anchor for hparams not found exactly; looking for alternates")
    # use eval_stride line or similar
    anchor_line2 = '    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))'
    if anchor_line2 in src_text:
        anchor_line = anchor_line2
    else:
        print("ERROR: no hparam anchor found")
        sys.exit(1)
src_text = src_text.replace(anchor_line, anchor_line + "\n" + ppm_hparams, 1)

# 3. Add PPM block before def train_and_eval
ppm_block = '''
# ============================================================
# PPM-D byte-level mixture (1850 port + anti-hijack guard).
# ============================================================
_NATIVE_PPM_LIB = None
_NATIVE_PPM_C_SRC = "worktrees/055-050-with-ppm-fullrun/testing/ppm_scorer_antihijack.c"

def _build_native_ppm_lib():
    global _NATIVE_PPM_LIB
    if _NATIVE_PPM_LIB is not None:
        return _NATIVE_PPM_LIB
    code = _NATIVE_PPM_C_SRC
    d = tempfile.gettempdir()
    digest = hashlib.sha1(code.encode()).hexdigest()[:12]
    c_path = os.path.join(d, f"pg_ppm_antihijack_{digest}.c")
    so_path = os.path.join(d, f"pg_ppm_antihijack_{digest}.so")
    if not os.path.exists(so_path):
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(code)
        cmd = ["gcc", "-O3", "-march=native", "-fPIC", "-shared", "-fopenmp",
               c_path, "-o", so_path, "-lm", "-lgomp"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError("native ppm build failed: " + res.stderr[-1000:])
    lib = ctypes.CDLL(so_path)
    lib.ppm_score.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ppm_score.restype = ctypes.c_int
    lib.ppm_score_omp.argtypes = [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint32, ctypes.c_int64, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ppm_score_omp.restype = ctypes.c_int
    _NATIVE_PPM_LIB = lib
    return lib

def _build_token_bytes_lut_for_ppm(sp, vocab_size):
    sz = max(int(sp.vocab_size()), vocab_size)
    bytestrs = [b""] * sz
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        piece = sp.id_to_piece(tid)
        if sp.is_byte(tid):
            bytestrs[tid] = bytes([int(piece[3:-1], 16)])
            continue
        if piece.startswith("▁"):
            piece = piece[1:]
        bytestrs[tid] = piece.encode("utf-8")
    return bytestrs

def run_ppm_native_pass(h, device, val_data, base_model):
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    sp = spm.SentencePieceProcessor()
    sp.load(h.tokenizer_path)
    vocab = sp.vocab_size()
    token_bytes_py = _build_token_bytes_lut_for_ppm(sp, vocab)
    has_leading_np = val_data.has_leading_space_lut.detach().cpu().numpy().astype(bool)
    is_boundary_np = val_data.is_boundary_token_lut.detach().cpu().numpy().astype(bool)
    lens = np.array([len(b) for b in token_bytes_py], dtype=np.int32)
    offs = np.zeros(vocab, dtype=np.int32)
    total = int(lens.sum())
    flat = np.empty(total, dtype=np.uint8)
    p = 0
    for i, b in enumerate(token_bytes_py):
        offs[i] = p
        if len(b):
            flat[p : p + len(b)] = np.frombuffer(b, dtype=np.uint8)
        p += len(b)

    seq_len = h.eval_seq_len
    total_tokens = val_data.val_tokens.numel() - 1
    total_seqs = total_tokens // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    local_count = (seq_end - seq_start) * seq_len
    nll_np = np.empty(local_count, dtype=np.float64)
    tgt_np = np.empty(local_count, dtype=np.int32)
    prev_np = np.empty(local_count, dtype=np.int32)
    write_i = 0

    log(f"ppm_collect:start total_seqs={total_seqs} my_seqs={seq_end-seq_start} tokens={local_count} rank={h.rank}")
    t_collect = time.perf_counter()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    forward_logits = torch.compile(base_model.forward_logits, dynamic=True, fullgraph=False)
    global BOS_ID
    if BOS_ID is None:
        BOS_ID = 1
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end):
            raw_start = batch_seq_start * seq_len
            raw_end = raw_start + seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1]
            y = local[1:]
            bos_pos = (x == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            cu_seqlens, max_seqlen = _build_cu_seqlens(
                bos_pos, x.numel(), x.device, h.eval_seq_len, 64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = forward_logits(x[None], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen).detach()
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y.reshape(-1),
                reduction="none",
            )
            n = per_token_loss.numel()
            nll_np[write_i : write_i + n] = per_token_loss.to(torch.float64).cpu().numpy()
            tgt_np[write_i : write_i + n] = y.cpu().numpy().astype(np.int32, copy=False)
            prev_np[write_i : write_i + n] = x.cpu().numpy().astype(np.int32, copy=False)
            write_i += n
    log(f"ppm_collect:done tokens={write_i} seconds={time.perf_counter()-t_collect:.1f}")

    lib = _build_native_ppm_lib()
    target_ids = np.ascontiguousarray(tgt_np[:write_i].astype(np.int64))
    prev_ids = np.ascontiguousarray(prev_np[:write_i].astype(np.int64))
    nll_in = np.ascontiguousarray(nll_np[:write_i])
    has = np.ascontiguousarray(has_leading_np.astype(np.uint8))
    isb = np.ascontiguousarray(is_boundary_np.astype(np.uint8))
    out = np.zeros(6, dtype=np.float64)
    log(f"ppm_score:start order={h.ppm_order} thr={h.ppm_conf_threshold} "
        f"nn_skip_thr_nats={h.ppm_nn_skip_thr_nats:.4f} "
        f"omp_threads={h.ppm_omp_threads} omp_chunk={h.ppm_omp_chunk_tokens}")
    t_score = time.perf_counter()
    use_omp = h.ppm_omp_chunk_tokens > 0
    if use_omp:
        rc = lib.ppm_score_omp(
            target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            nll_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(target_ids.size),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            has.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            isb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(vocab), ctypes.c_int(h.ppm_order),
            ctypes.c_double(h.ppm_lambda_hi), ctypes.c_double(h.ppm_lambda_lo),
            ctypes.c_double(h.ppm_conf_threshold),
            ctypes.c_double(h.ppm_nn_skip_thr_nats),
            ctypes.c_uint32(h.ppm_log_cache_size),
            ctypes.c_int64(h.ppm_omp_chunk_tokens), ctypes.c_int(h.ppm_omp_threads),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        rc = lib.ppm_score(
            target_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            prev_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            nll_in.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(target_ids.size),
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            offs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            lens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            has.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            isb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(vocab), ctypes.c_int(h.ppm_order),
            ctypes.c_double(h.ppm_lambda_hi), ctypes.c_double(h.ppm_lambda_lo),
            ctypes.c_double(h.ppm_conf_threshold),
            ctypes.c_double(h.ppm_nn_skip_thr_nats),
            ctypes.c_uint32(h.ppm_log_cache_size),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    if rc != 0:
        raise RuntimeError(f"native ppm rc={rc}")
    elapsed = time.perf_counter() - t_score
    n_bytes_piece = int(out[4])
    if val_data.caseops_enabled and val_data.val_bytes is not None:
        target_bytes = val_data.val_bytes[1 : write_i + 1].cpu().numpy().astype(np.int64)
        n_bytes_sidecar = int(target_bytes.sum())
        total_bits = out[0] * n_bytes_piece
        mix_bpb_sidecar = total_bits / n_bytes_sidecar
        nn_total_bits = out[2] * n_bytes_piece
        nn_bpb_sidecar = nn_total_bits / n_bytes_sidecar
    else:
        n_bytes_sidecar = n_bytes_piece
        mix_bpb_sidecar = out[0]
        nn_bpb_sidecar = out[2]
    log(
        f"ppm_full_native time={elapsed:.1f}s "
        f"tokens={write_i} bytes_piece={n_bytes_piece} bytes_sidecar={n_bytes_sidecar} "
        f"mix_bpb_piece={out[0]:.6f} ppm_only={out[1]:.6f} nn_byte_bpb_piece={out[2]:.6f} "
        f"mix_bpb_sidecar={mix_bpb_sidecar:.6f} nn_byte_bpb_sidecar={nn_bpb_sidecar:.6f} "
        f"gate_high_frac={out[5]:.6f} order={h.ppm_order} "
        f"lambda_hi={h.ppm_lambda_hi} lambda_lo={h.ppm_lambda_lo} "
        f"thr={h.ppm_conf_threshold} nn_skip_thr_nats={h.ppm_nn_skip_thr_nats}"
    )
    log(f"ppm_native_submission_val_bpb: {mix_bpb_sidecar:.6f}")
    return mix_bpb_sidecar

'''

src_text = src_text.replace(
    "def train_and_eval(h, device):",
    ppm_block + "def train_and_eval(h, device):",
    1,
)

# 4. Hook PPM into train_and_eval AFTER deserialize+looping setup, BEFORE
#    the `if not ttt_eval_only:` block (so it runs whether or not TTT_EVAL_ONLY is set).
old_block = """    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    if not ttt_eval_only:"""
new_block = """    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    if h.ppm_native_enabled:
        log("\\nbeginning PPM native pass")
        torch.cuda.synchronize()
        t_ppm_total = time.perf_counter()
        run_ppm_native_pass(h, device, val_data, eval_model)
        torch.cuda.synchronize()
        log(f"ppm_native:total_pass_time:{time.perf_counter()-t_ppm_total:.1f}s")
    if not ttt_eval_only:"""

if old_block not in src_text:
    print("ERROR: could not find diagnostic-quantized hook anchor in 052")
    sys.exit(1)
src_text = src_text.replace(old_block, new_block, 1)

Path(OUT).write_text(src_text)
print(f"Wrote {OUT} ({len(src_text)} bytes, {src_text.count(chr(10))+1} lines)")
print(f"PPM block inserted: {'_NATIVE_PPM_C_SRC' in src_text}")
print(f"Hook inserted: {'run_ppm_native_pass(h, device, val_data, eval_model)' in src_text}")
print(f"Hyperparameters added: {'ppm_native_enabled' in src_text}")
