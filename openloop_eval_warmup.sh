export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# Action type: raw_actions (default), skeleton_actions, or residual_actions
ACTION_TYPE=${1:-raw_actions}

uv run scripts/eval_warmup_openloop.py \
    --checkpoint_dir checkpoints/pi05_warmup/warmup_lora_skeleton_actions_20260513_003021/10000 \
    --episode_idx 92 \
    --action_type ${ACTION_TYPE} \
    --use_relative_state
