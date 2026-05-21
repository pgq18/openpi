export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# Action type: raw_actions (default), skeleton_actions, or residual_actions
ACTION_TYPE=${1:-raw_actions}

CHECKPOINT_DIR="checkpoints/pi05_warmup/warmup_lora_raw_actions_20260506_192850/10000"

uv run scripts/eval_warmup_openloop.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --episode_idx 91 \
    --action_type ${ACTION_TYPE} \
    --use_relative_state
