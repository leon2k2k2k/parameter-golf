#!/usr/bin/env python3
"""Spec 002: SWA + EMA blend screen.

Builds several averaged-weight state dicts from spec-000's post-recurrence
warmdown checkpoints, then for each config runs GPTQ + quantized eval +
sliding-window eval. Hessian is computed once (cached) and reused.

Configurations evaluated (C0 is control):
  C0  EMA-only                               (ckpt_final_pre_ema + its ema_state)
  C1  SWA of {1500, 2275, 3412, 3849}        pure 4-snapshot SWA
  C2  SWA of {2275, 3412, 3849}              late-3 only (skip post-flip transient)
  C3  0.5·C1 + 0.5·EMA
  C4  0.25·C1 + 0.75·EMA
  C5  0.75·C1 + 0.25·EMA

Checkpoints BEFORE step 1378 (pre-recurrence) are intentionally excluded:
the computational graph flips at step 1378; averaging across the architectural
boundary pulls toward a non-recurrent minimum.
"""
import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/parameter-golf")
from train_gpt_sota import (
    GPT,
    Hyperparameters,
    ShuffledSequenceLoader,
    ValidationData,
    _compress,
    collect_hessians,
    dequantize_mixed,
    eval_val,
    eval_val_sliding,
    gptq_mixed_quantize,
    log,
    restore_fp32_params,
    set_logging_hparams,
)


# Paths relative to the spec-000 checkpoint dir; filled in at launch.
SWA_SNAPSHOTS = ["step1500", "step2275", "step3412", "step3849"]
EMA_SOURCE = "step3849"  # take ema_state from ckpt_final_pre_ema_step3849.pt


def load_model_state(ckpt_path, device):
    """Return the raw training state_dict from a checkpoint (no EMA applied)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return {k: v for k, v in ckpt["model_state_dict"].items()}, ckpt.get("ema_state")


def average_states(state_dicts, weights):
    """Element-wise weighted average of a list of state_dicts.

    Assumes all state_dicts have identical keys and tensor shapes (same model
    architecture, same physical layers). Output tensors are float32 to avoid
    accumulating bf16 rounding error across multiple adds.
    """
    assert len(state_dicts) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6, f"weights must sum to 1, got {sum(weights)}"
    out = {}
    ref = state_dicts[0]
    for k in ref.keys():
        acc = torch.zeros_like(ref[k], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc.add_(sd[k].to(torch.float32), alpha=w)
        out[k] = acc.to(ref[k].dtype)
    return out


def blend_states(state_a, state_b, weight_a):
    """weight_a * state_a + (1 - weight_a) * state_b, element-wise."""
    assert 0.0 <= weight_a <= 1.0
    return average_states([state_a, state_b], [weight_a, 1.0 - weight_a])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="Dir containing spec-000 checkpoints (ckpt_event_step1500.pt etc.)")
    p.add_argument("--run_dir", required=True,
                   help="Output dir for this spec (runs/002-swa-plus-ema/)")
    p.add_argument("--configs", default="C0,C1,C2,C3,C4,C5",
                   help="Comma-separated config IDs to run (order matters; C0 first for validity).")
    args = p.parse_args()

    random.seed(1337); np.random.seed(1337); torch.manual_seed(1337); torch.cuda.manual_seed_all(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    h = Hyperparameters()
    set_logging_hparams(h)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    os.makedirs(args.run_dir, exist_ok=True)

    # ---- Locate checkpoints ----
    ckpt_paths = {}
    for tag in SWA_SNAPSHOTS:
        # Naming varies: ckpt_event_step1500.pt, ckpt_final_pre_ema_step3849.pt
        candidates = [
            f"ckpt_event_{tag}.pt",
            f"ckpt_final_pre_ema_{tag}.pt",
            f"ckpt_{tag}.pt",
        ]
        found = None
        for c in candidates:
            pth = os.path.join(args.ckpt_dir, c)
            if os.path.exists(pth):
                found = pth
                break
        if found is None:
            raise FileNotFoundError(f"No checkpoint matching {tag} in {args.ckpt_dir}")
        ckpt_paths[tag] = found
        log(f"{tag}: {found}")

    # ---- Load all state dicts + the EMA state ----
    log("loading checkpoints...")
    raw_states = {}
    ema_state = None
    for tag, pth in ckpt_paths.items():
        sd, es = load_model_state(pth, device)
        raw_states[tag] = sd
        if tag == EMA_SOURCE:
            if es is None:
                raise RuntimeError(f"checkpoint {pth} has no ema_state — cannot run C0/C3/C4/C5")
            ema_state = {k: v.to(device) for k, v in es.items()}
    log(f"loaded {len(raw_states)} raw states + EMA state from {EMA_SOURCE}")

    # ---- Build config state dicts ----
    swa_all4 = average_states([raw_states[t] for t in SWA_SNAPSHOTS], [0.25] * 4)
    swa_late3 = average_states([raw_states[t] for t in SWA_SNAPSHOTS[1:]], [1/3] * 3)

    # For C0 (EMA-only): load ema_state into the model state-dict slot.
    # ema_state dtype may differ from model params (ema is fp32); cast to match.
    c0_state = {k: v.to(dtype=raw_states[EMA_SOURCE][k].dtype) for k, v in ema_state.items()}

    c3_state = blend_states(swa_all4, c0_state, 0.5)
    c4_state = blend_states(swa_all4, c0_state, 0.25)
    c5_state = blend_states(swa_all4, c0_state, 0.75)

    configs = {
        "C0": ("EMA-only (control)", c0_state, {"type": "ema", "snapshots": [], "blend": None}),
        "C1": ("SWA all 4 post-recurrence", swa_all4, {"type": "swa", "snapshots": SWA_SNAPSHOTS, "blend": None}),
        "C2": ("SWA late 3", swa_late3, {"type": "swa", "snapshots": SWA_SNAPSHOTS[1:], "blend": None}),
        "C3": ("0.5 SWA + 0.5 EMA", c3_state, {"type": "blend", "snapshots": SWA_SNAPSHOTS, "blend": 0.5}),
        "C4": ("0.25 SWA + 0.75 EMA", c4_state, {"type": "blend", "snapshots": SWA_SNAPSHOTS, "blend": 0.25}),
        "C5": ("0.75 SWA + 0.25 EMA", c5_state, {"type": "blend", "snapshots": SWA_SNAPSHOTS, "blend": 0.75}),
    }
    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in requested:
        if c not in configs:
            raise ValueError(f"unknown config {c}; valid: {list(configs.keys())}")

    # ---- Build a base model; we'll swap state dicts into it per config ----
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    if h.num_loops > 0:
        base_model.looping_active = True
        log(f"recurrence active (eval setting): encoder={base_model.encoder_indices} decoder={base_model.decoder_indices}")

    # ---- Hessian: computed once from C0 (EMA-only) weights. Cached to disk. ----
    hessian_path = os.path.join(args.run_dir, "hessians.pt")
    if os.path.exists(hessian_path):
        hessians = torch.load(hessian_path, map_location="cpu", weights_only=False)
        log(f"RELOADED Hessians from {hessian_path} ({len(hessians)} keys)")
    else:
        log("computing Hessians from C0 (EMA-only) weights — ONCE, reused across configs...")
        base_model.load_state_dict(c0_state, strict=True)
        calib_loader = ShuffledSequenceLoader(h, device)
        t0 = time.perf_counter()
        hessians = collect_hessians(base_model, calib_loader, h, device,
                                    n_calibration_batches=h.gptq_calibration_batches)
        log(f"Hessians collected in {time.perf_counter()-t0:.1f}s; saving...")
        torch.save(hessians, hessian_path)

    val_data = ValidationData(h, device)

    # ---- Per-config loop ----
    for cid in requested:
        jpath = os.path.join(args.run_dir, f"config_{cid}.json")
        if os.path.exists(jpath):
            log(f"{cid}: already done (json exists), skipping")
            continue

        desc, state, meta = configs[cid]
        log(f"{cid} ({desc}): starting")
        t0 = time.perf_counter()

        # Load averaged state into model for quant (GPTQ reads model's cpu state_dict below)
        base_model.load_state_dict(state, strict=True)
        sd_cpu = {k: v.detach().cpu() for (k, v) in base_model.state_dict().items()}

        # Quantize (shared hessians, default hessian_clip_lambda=0 via h defaults)
        quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)

        # Save ptz artifact
        buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, buf)
        quant_raw = buf.getvalue()
        quant_blob = _compress(quant_raw, h.compressor)
        ptz_path = os.path.join(args.run_dir, f"quantized_{cid}.ptz")
        with open(ptz_path, "wb") as fout:
            fout.write(quant_blob)
        size = len(quant_blob)

        # Build eval model + dequantized weights
        deq_state = dequantize_mixed(quant_result, quant_meta, sd_cpu)
        eval_model = GPT(h).to(device).bfloat16()
        restore_fp32_params(eval_model)
        eval_model.load_state_dict(deq_state, strict=True)
        if h.num_loops > 0:
            eval_model.looping_active = True

        # Quant eval (compiled)
        compiled = torch.compile(eval_model, dynamic=False, fullgraph=True)
        vloss_q, vbpb_q = eval_val(h, device, val_data, compiled)
        t_quant = time.perf_counter() - t0

        # Sliding-window eval (uses un-compiled model per sota code path)
        t1 = time.perf_counter()
        if h.sliding_window_enabled:
            vloss_s, vbpb_s = eval_val_sliding(h, device, val_data, eval_model)
        else:
            vloss_s, vbpb_s = float("nan"), float("nan")
        t_slide = time.perf_counter() - t1

        result = {
            "config_id": cid,
            "description": desc,
            "meta": meta,
            "val_bpb_quantized": float(vbpb_q),
            "val_loss_quantized": float(vloss_q),
            "val_bpb_sliding": float(vbpb_s),
            "val_loss_sliding": float(vloss_s),
            "artifact_size_bytes": size,
            "elapsed_quant_sec": t_quant,
            "elapsed_sliding_sec": t_slide,
        }
        with open(jpath, "w") as f:
            json.dump(result, f, indent=2)
        log(f"{cid}: quant_bpb={vbpb_q:.6f} sliding_bpb={vbpb_s:.6f} size={size} bytes "
            f"(quant {t_quant:.1f}s, sliding {t_slide:.1f}s)")

        del eval_model, compiled, deq_state
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    log("swa_sweep.py done")


if __name__ == "__main__":
    main()
