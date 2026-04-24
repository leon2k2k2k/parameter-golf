#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/parameter-golf}"
CONFIG_JSON="${CONFIG_JSON:-$ROOT/runs/038-smear-lqer-asym-8h/seed_42/config.json}"
PINNED_BRANCH="${PINNED_BRANCH:-exp/038-fullfloat-smear-lqer-asym}"
PINNED_COMMIT="${PINNED_COMMIT:-c8620b68fc6c53c9cf8c0a28a05587bd4fdca9ea}"

cd "$ROOT"

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "missing config json: $CONFIG_JSON" >&2
  exit 1
fi

eval "$(
CONFIG_JSON="$CONFIG_JSON" python3 - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads(Path(os.environ["CONFIG_JSON"]).read_text())

env_map = {
    'data_dir': 'DATA_DIR',
    'seed': 'SEED',
    'run_id': 'RUN_ID',
    'iterations': 'ITERATIONS',
    'warmdown_frac': 'WARMDOWN_FRAC',
    'warmup_steps': 'WARMUP_STEPS',
    'train_batch_tokens': 'TRAIN_BATCH_TOKENS',
    'fused_ce_enabled': 'FUSED_CE_ENABLED',
    'train_seq_len': 'TRAIN_SEQ_LEN',
    'train_log_every': 'TRAIN_LOG_EVERY',
    'max_wallclock_seconds': 'MAX_WALLCLOCK_SECONDS',
    'val_batch_tokens': 'VAL_BATCH_TOKENS',
    'eval_seq_len': 'EVAL_SEQ_LEN',
    'val_loss_every': 'VAL_LOSS_EVERY',
    'vocab_size': 'VOCAB_SIZE',
    'num_layers': 'NUM_LAYERS',
    'xsa_last_n': 'XSA_LAST_N',
    'model_dim': 'MODEL_DIM',
    'num_kv_heads': 'NUM_KV_HEADS',
    'num_heads': 'NUM_HEADS',
    'mlp_mult': 'MLP_MULT',
    'skip_gates_enabled': 'SKIP_GATES_ENABLED',
    'tie_embeddings': 'TIE_EMBEDDINGS',
    'logit_softcap': 'LOGIT_SOFTCAP',
    'rope_base': 'ROPE_BASE',
    'rope_dims': 'ROPE_DIMS',
    'rope_train_seq_len': 'ROPE_TRAIN_SEQ_LEN',
    'rope_yarn': 'ROPE_YARN',
    'ln_scale': 'LN_SCALE',
    'qk_gain_init': 'QK_GAIN_INIT',
    'num_loops': 'NUM_LOOPS',
    'loop_start': 'LOOP_START',
    'loop_end': 'LOOP_END',
    'enable_looping_at': 'ENABLE_LOOPING_AT',
    'loop_depth_upgrade_at': 'LOOP_DEPTH_UPGRADE_AT',
    'parallel_start_layer': 'PARALLEL_START_LAYER',
    'parallel_final_lane': 'PARALLEL_FINAL_LANE',
    'min_lr': 'MIN_LR',
    'embed_lr': 'EMBED_LR',
    'tied_embed_lr': 'TIED_EMBED_LR',
    'tied_embed_init_std': 'TIED_EMBED_INIT_STD',
    'matrix_lr': 'MATRIX_LR',
    'scalar_lr': 'SCALAR_LR',
    'muon_momentum': 'MUON_MOMENTUM',
    'muon_backend_steps': 'MUON_BACKEND_STEPS',
    'muon_momentum_warmup_start': 'MUON_MOMENTUM_WARMUP_START',
    'muon_momentum_warmup_steps': 'MUON_MOMENTUM_WARMUP_STEPS',
    'muon_row_normalize': 'MUON_ROW_NORMALIZE',
    'beta1': 'BETA1',
    'beta2': 'BETA2',
    'adam_eps': 'ADAM_EPS',
    'grad_clip_norm': 'GRAD_CLIP_NORM',
    'eval_stride': 'EVAL_STRIDE',
    'adam_wd': 'ADAM_WD',
    'muon_wd': 'MUON_WD',
    'embed_wd': 'EMBED_WD',
    'ema_decay': 'EMA_DECAY',
    'ttt_enabled': 'TTT_ENABLED',
    'ttt_lora_rank': 'TTT_LORA_RANK',
    'ttt_lora_alpha': 'TTT_LORA_ALPHA',
    'ttt_lora_lr': 'TTT_LORA_LR',
    'ttt_chunk_size': 'TTT_CHUNK_SIZE',
    'ttt_eval_seq_len': 'TTT_EVAL_SEQ_LEN',
    'ttt_batch_size': 'TTT_BATCH_SIZE',
    'ttt_grad_steps': 'TTT_GRAD_STEPS',
    'ttt_weight_decay': 'TTT_WEIGHT_DECAY',
    'ttt_beta1': 'TTT_BETA1',
    'ttt_beta2': 'TTT_BETA2',
    'ttt_k_lora': 'TTT_K_LORA',
    'ttt_mlp_lora': 'TTT_MLP_LORA',
    'ttt_o_lora': 'TTT_O_LORA',
    'ttt_optimizer': 'TTT_OPTIMIZER',
    'ttt_eval_batches': 'TTT_EVAL_BATCHES',
    'val_doc_fraction': 'VAL_DOC_FRACTION',
    'compressor': 'COMPRESSOR',
    'gptq_calibration_batches': 'GPTQ_CALIBRATION_BATCHES',
    'gptq_reserve_seconds': 'GPTQ_RESERVE_SECONDS',
    'phased_ttt_prefix_docs': 'PHASED_TTT_PREFIX_DOCS',
    'phased_ttt_num_phases': 'PHASED_TTT_NUM_PHASES',
    'global_ttt_lr': 'GLOBAL_TTT_LR',
    'global_ttt_momentum': 'GLOBAL_TTT_MOMENTUM',
    'global_ttt_epochs': 'GLOBAL_TTT_EPOCHS',
    'global_ttt_chunk_tokens': 'GLOBAL_TTT_CHUNK_TOKENS',
    'global_ttt_batch_seqs': 'GLOBAL_TTT_BATCH_SEQS',
    'global_ttt_warmup_start_lr': 'GLOBAL_TTT_WARMUP_START_LR',
    'global_ttt_warmup_chunks': 'GLOBAL_TTT_WARMUP_CHUNKS',
    'global_ttt_grad_clip': 'GLOBAL_TTT_GRAD_CLIP',
    'global_ttt_respect_doc_boundaries': 'GLOBAL_TTT_RESPECT_DOC_BOUNDARIES',
    'matrix_bits': 'MATRIX_BITS',
    'embed_bits': 'EMBED_BITS',
    'matrix_clip_sigmas': 'MATRIX_CLIP_SIGMAS',
    'embed_clip_sigmas': 'EMBED_CLIP_SIGMAS',
    'mlp_clip_sigmas': 'MLP_CLIP_SIGMAS',
    'attn_clip_sigmas': 'ATTN_CLIP_SIGMAS',
    'attn_out_gate_enabled': 'ATTN_OUT_GATE_ENABLED',
    'attn_out_gate_src': 'ATTN_OUT_GATE_SRC',
    'smear_gate_enabled': 'SMEAR_GATE_ENABLED',
    'gate_window': 'GATE_WINDOW',
    'recur_alpha_enabled': 'RECUR_ALPHA_ENABLED',
    'recur_diag_p2p_cos': 'RECUR_DIAG_P2P_COS',
    'gated_attn_enabled': 'GATED_ATTN_ENABLED',
    'gated_attn_init_std': 'GATED_ATTN_INIT_STD',
    'sparse_attn_gate_enabled': 'SPARSE_ATTN_GATE_ENABLED',
    'sparse_attn_gate_init_std': 'SPARSE_ATTN_GATE_INIT_STD',
    'sparse_attn_gate_scale': 'SPARSE_ATTN_GATE_SCALE',
    'lqer_enabled': 'LQER_ENABLED',
    'lqer_rank': 'LQER_RANK',
    'lqer_top_k': 'LQER_TOP_K',
    'lqer_factor_bits': 'LQER_FACTOR_BITS',
    'lqer_asym_enabled': 'LQER_ASYM_ENABLED',
    'lqer_asym_group': 'LQER_ASYM_GROUP',
    'gated_attn_quant_gate': 'GATED_ATTN_QUANT_GATE',
    'spinquant_enabled': 'SPINQUANT_ENABLED',
    'spinquant_seed': 'SPINQUANT_SEED',
    'spinquant_sites': 'SPINQUANT_SITES',
    'caseops_enabled': 'CASEOPS_ENABLED',
    'datasets_dir': 'DATASETS_DIR',
    'tokenizer_path': 'TOKENIZER_PATH',
    'train_files': 'TRAIN_FILES',
    'val_files': 'VAL_FILES',
    'val_bytes_files': 'VAL_BYTES_FILES',
}

def fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)

for key, env in env_map.items():
    if key in cfg and cfg[key] is not None:
        print(f"export {env}={json.dumps(fmt(cfg[key]))}")
PY
)"

export RUN_ID=043B-038a-faithful-4h-20m-replay
export MAX_WALLCLOCK_SECONDS=1200
export TTT_ENABLED=0
export TRAINING_ONLY_SCREEN=1

git fetch fork
git checkout "$PINNED_BRANCH"
git reset --hard "$PINNED_COMMIT"

torchrun --standalone --nproc_per_node=4 \
  records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py
