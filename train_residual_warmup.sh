#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Usage:
#   bash train_residual_warmup.sh [pi05_checkpoint_path] [checkpoint_base_dir] [extra train args...]
#
# Example:
#   bash train_residual_warmup.sh \
#     /home/pengguanqi/Data/rl4vla/checkpoints/pi05_warmup/warmup_lora_skeleton_actions_20260513_224206/10000/params \
#     /home/pengguanqi/Data/rl4vla/checkpoints \
#     --batch_size 16 --num_train_steps 10000

PI05_CHECKPOINT_PATH=${1:-${PI05_CHECKPOINT_PATH:-/home/pengguanqi/Data/rl4vla/checkpoints/pi05_warmup/warmup_lora_skeleton_actions_20260513_224206/10000/params}}
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ -d "${PI05_CHECKPOINT_PATH}/params" && ! -f "${PI05_CHECKPOINT_PATH}/_METADATA" ]]; then
    PI05_CHECKPOINT_PATH="${PI05_CHECKPOINT_PATH}/params"
fi

CHECKPOINT_DIR=${1:-${CHECKPOINT_DIR:-/home/pengguanqi/Data/rl4vla/checkpoints}}
if [[ $# -gt 0 ]]; then
    shift
fi

RLDS_DATA_DIR=${RLDS_DATA_DIR:-../datasets}
NORM_STATS_DIR=${NORM_STATS_DIR:-./assets/pi05_warmup/warmup}
EXP_NAME=${EXP_NAME:-residual_bc_skeleton_${TIMESTAMP}}
LOG_DIR=${LOG_DIR:-${CHECKPOINT_DIR}/logs/residual_warmup}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}

mkdir -p "${LOG_DIR}"
echo "Writing training log to ${LOG_FILE}"

uv run scripts/train_residual_warmup.py \
    --rlds_data_dir "${RLDS_DATA_DIR}" \
    --norm_stats_dir "${NORM_STATS_DIR}" \
    --pi05_checkpoint_path "${PI05_CHECKPOINT_PATH}" \
    --checkpoint_base_dir "${CHECKPOINT_DIR}" \
    --exp_name "${EXP_NAME}" \
    --lora full \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_train_steps "${NUM_TRAIN_STEPS:-20000}" \
    --lr "${LR:-1e-4}" \
    --res_scale "${RES_SCALE:-0.2}" \
    --res_encoded_dim "${RES_ENCODED_DIM:-256}" \
    --res_hidden_dims "${RES_HIDDEN_DIMS:-256,256,256}" \
    --feature_layer_idx "${FEATURE_LAYER_IDX:-12}" \
    --feature_capture_step "${FEATURE_CAPTURE_STEP:-0}" \
    --proprio_dim "${PROPRIO_DIM:-7}" \
    --action_dim "${ACTION_DIM:-6}" \
    --action_type skeleton_actions \
    --use_relative_state \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
