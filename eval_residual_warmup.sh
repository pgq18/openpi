#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Usage:
#   bash eval_residual_warmup.sh [pi05_step_checkpoint_dir] [residual_params_checkpoint_dir] [extra eval args...]
#
# Example:
#   bash eval_residual_warmup.sh \
#     /home/pengguanqi/Data/rl4vla/checkpoints/pi05_warmup/warmup_lora_skeleton_actions_20260513_224206/10000 \
#     /home/pengguanqi/Data/rl4vla/checkpoints/pi05_residual/residual_bc_skeleton_20260521_010422/19999/params \
#     --episode_idx 0 --stride 10 --res_scale 0.2

PI05_CHECKPOINT_DIR=${1:-${PI05_CHECKPOINT_DIR:-/home/pengguanqi/Data/rl4vla/checkpoints/pi05_warmup/warmup_lora_skeleton_actions_20260513_224206/10000}}
if [[ $# -gt 0 ]]; then
    shift
fi

RESIDUAL_CHECKPOINT_DIR=${1:-${RESIDUAL_CHECKPOINT_DIR:-/home/pengguanqi/Data/rl4vla/checkpoints/pi05_residual/residual_bc_skeleton_20260521_010422/19999}}
if [[ $# -gt 0 ]]; then
    shift
fi

# Pi05 policy loading expects the step checkpoint directory that contains params/ and assets/.
if [[ "$(basename "${PI05_CHECKPOINT_DIR}")" == "params" ]]; then
    PI05_CHECKPOINT_DIR=$(dirname "${PI05_CHECKPOINT_DIR}")
fi

# Residual actor loading expects the params item directory itself.
if [[ -d "${RESIDUAL_CHECKPOINT_DIR}/params" && ! -f "${RESIDUAL_CHECKPOINT_DIR}/_METADATA" ]]; then
    RESIDUAL_CHECKPOINT_DIR="${RESIDUAL_CHECKPOINT_DIR}/params"
fi

DATA_DIR=${DATA_DIR:-../datasets}
EVAL_SAVE_DIR=${EVAL_SAVE_DIR:-./eval_results/residual_warmup/${TIMESTAMP}}
LOG_DIR=${LOG_DIR:-${EVAL_SAVE_DIR}/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/eval_residual_warmup.log}

mkdir -p "${LOG_DIR}"
echo "Writing eval log to ${LOG_FILE}"

uv run scripts/eval_residual_warmup.py \
    --pi05_checkpoint_dir "${PI05_CHECKPOINT_DIR}" \
    --residual_checkpoint_dir "${RESIDUAL_CHECKPOINT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --episode_idx "${EPISODE_IDX:-30}" \
    --split "${SPLIT:-train}" \
    --lora "${LORA:-full}" \
    --res_scale "${RES_SCALE:-1}" \
    --stride "${STRIDE:-10}" \
    --save_dir "${EVAL_SAVE_DIR}" \
    --res_encoded_dim "${RES_ENCODED_DIM:-256}" \
    --res_hidden_dims "${RES_HIDDEN_DIMS:-256,256,256}" \
    --feature_layer_idx "${FEATURE_LAYER_IDX:-12}" \
    --feature_capture_step "${FEATURE_CAPTURE_STEP:-0}" \
    --proprio_dim "${PROPRIO_DIM:-7}" \
    --action_dim "${ACTION_DIM:-6}" \
    --action_type skeleton_actions \
    --use_relative_state \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
