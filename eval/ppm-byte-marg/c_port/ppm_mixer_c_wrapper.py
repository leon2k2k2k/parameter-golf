"""ctypes wrapper for libppm_byte_mixer.so. Drop-in replacement for the Python
ppm_mixer_with_norm_modes() function. Same inputs, same outputs.

Usage:
    res = ppm_mixer_c(target_ids, prev_ids, nll_nats,
                      p_full_b0, p_non_ls_b0, p_leading_sp,
                      sum_check_full, sum_check_caseB,
                      tb_lut, has_ls_lut, is_bdy_lut,
                      alpha=15.0, beta=0.80, renormalize=False)
    res['bpb'], res['tokens'], res['bytes'], res['total_nll_nats'], res['renorm_correction_nats']
"""

import os, ctypes, math
import numpy as np
from pathlib import Path


class _MixerResult(ctypes.Structure):
    _fields_ = [
        ("total_mix_nll_nats", ctypes.c_double),
        ("total_canonical_bytes", ctypes.c_int64),
        ("total_scored_tokens", ctypes.c_int64),
        ("total_renorm_correction_nats", ctypes.c_double),
        ("total_nn_b0_nll", ctypes.c_double),
        ("total_ppm_b0_nll", ctypes.c_double),
        ("total_mix_b0_nll", ctypes.c_double),
        ("total_oracle_b0_nll", ctypes.c_double),
        ("total_nn_rem_nll", ctypes.c_double),
        ("total_ppm_rem_nll", ctypes.c_double),
        ("total_mix_rem_nll", ctypes.c_double),
        ("total_oracle_rem_nll", ctypes.c_double),
        ("total_b0_positions", ctypes.c_int64),
        ("total_rem_bytes", ctypes.c_int64),
        ("total_multibyte_tokens", ctypes.c_int64),
    ]


_lib = None


def _load():
    global _lib
    if _lib is not None: return _lib
    here = Path(__file__).parent
    so = here / "libppm_byte_mixer.so"
    if not so.exists():
        raise FileNotFoundError(f"{so} not built — run build.sh first")
    _lib = ctypes.CDLL(str(so))
    _lib.ppm_mixer_run.restype = None
    _lib.ppm_mixer_run.argtypes = [
        ctypes.POINTER(ctypes.c_int32),    # target_ids
        ctypes.POINTER(ctypes.c_int32),    # prev_ids
        ctypes.POINTER(ctypes.c_float),    # nll_nats
        ctypes.POINTER(ctypes.c_float),    # p_full_b0
        ctypes.POINTER(ctypes.c_float),    # p_non_ls_b0
        ctypes.POINTER(ctypes.c_float),    # p_leading_sp
        ctypes.POINTER(ctypes.c_float),    # sum_check_full
        ctypes.POINTER(ctypes.c_float),    # sum_check_caseB
        ctypes.c_int64,                    # N
        ctypes.POINTER(ctypes.c_uint8),    # token_bytes_concat
        ctypes.POINTER(ctypes.c_int32),    # token_bytes_offset (V+1)
        ctypes.c_int32,                    # V
        ctypes.POINTER(ctypes.c_uint8),    # token_has_leading_space
        ctypes.POINTER(ctypes.c_uint8),    # token_is_boundary
        ctypes.c_double,                   # alpha
        ctypes.c_double,                   # beta
        ctypes.c_int,                      # renormalize
        ctypes.POINTER(_MixerResult),      # out
    ]
    return _lib


def _as_c_array(arr, dtype):
    """Ensure C-contiguous array of given dtype, return ctypes pointer."""
    a = np.ascontiguousarray(arr, dtype=dtype)
    return a, a.ctypes.data_as(ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype)))


def build_token_lut_arrays(token_bytes_lut, has_leading_space_lut, is_boundary_lut):
    """Convert Python LUTs (list[bytes], np.bool array, np.bool array) to flat C-friendly arrays."""
    V = len(token_bytes_lut)
    offsets = np.zeros(V + 1, dtype=np.int32)
    total = 0
    for i, b in enumerate(token_bytes_lut):
        offsets[i] = total
        total += len(b)
    offsets[V] = total
    concat = np.zeros(total, dtype=np.uint8)
    for i, b in enumerate(token_bytes_lut):
        if len(b) > 0:
            concat[offsets[i]:offsets[i+1]] = np.frombuffer(b, dtype=np.uint8)
    has_ls_u8 = np.ascontiguousarray(has_leading_space_lut, dtype=np.uint8)
    is_bdy_u8 = np.ascontiguousarray(is_boundary_lut, dtype=np.uint8)
    return concat, offsets, has_ls_u8, is_bdy_u8


def ppm_mixer_c(
    target_ids, prev_ids, nll_nats,
    p_full_b0, p_non_ls_b0, p_leading_sp,
    sum_check_full, sum_check_caseB,
    token_bytes_lut, has_leading_space_lut, is_boundary_lut,
    alpha=15.0, beta=0.80, renormalize=False,
):
    lib = _load()
    N = len(target_ids)

    # Ensure ctypes-friendly arrays
    a_tid, p_tid = _as_c_array(target_ids, np.int32)
    a_pid, p_pid = _as_c_array(prev_ids, np.int32)
    a_nll, p_nll = _as_c_array(nll_nats, np.float32)
    a_pf,  p_pf  = _as_c_array(p_full_b0, np.float32)
    a_pn,  p_pn  = _as_c_array(p_non_ls_b0, np.float32)
    a_pls, p_pls = _as_c_array(p_leading_sp, np.float32)
    a_scf, p_scf = _as_c_array(sum_check_full, np.float32)
    a_scb, p_scb = _as_c_array(sum_check_caseB, np.float32)

    concat, offsets, has_ls_u8, is_bdy_u8 = build_token_lut_arrays(
        token_bytes_lut, has_leading_space_lut, is_boundary_lut)
    a_concat = concat
    a_offsets = offsets
    a_hls = has_ls_u8
    a_bdy = is_bdy_u8
    p_concat  = a_concat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    p_offsets = a_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    p_hls     = a_hls.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    p_bdy     = a_bdy.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    out = _MixerResult()
    lib.ppm_mixer_run(
        p_tid, p_pid, p_nll,
        p_pf, p_pn, p_pls,
        p_scf, p_scb,
        ctypes.c_int64(N),
        p_concat, p_offsets, ctypes.c_int32(len(token_bytes_lut)),
        p_hls, p_bdy,
        ctypes.c_double(alpha), ctypes.c_double(beta),
        ctypes.c_int(1 if renormalize else 0),
        ctypes.byref(out),
    )

    LOG2 = math.log(2.0)
    return {
        "bpb": out.total_mix_nll_nats / max(out.total_canonical_bytes, 1) / LOG2,
        "tokens": out.total_scored_tokens,
        "bytes": out.total_canonical_bytes,
        "total_nll_nats": out.total_mix_nll_nats,
        "renorm_correction_nats": out.total_renorm_correction_nats,
        # Decomposition (raw NLL totals in nats; divide by total bytes / log(2) for BPB contribution)
        "nn_b0_nll": out.total_nn_b0_nll,
        "ppm_b0_nll": out.total_ppm_b0_nll,
        "mix_b0_nll": out.total_mix_b0_nll,
        "oracle_b0_nll": out.total_oracle_b0_nll,
        "nn_rem_nll": out.total_nn_rem_nll,
        "ppm_rem_nll": out.total_ppm_rem_nll,
        "mix_rem_nll": out.total_mix_rem_nll,
        "oracle_rem_nll": out.total_oracle_rem_nll,
        "b0_positions": out.total_b0_positions,
        "rem_bytes": out.total_rem_bytes,
        "multibyte_tokens": out.total_multibyte_tokens,
    }
