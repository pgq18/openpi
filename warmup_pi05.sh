export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Action type: raw_actions (default), skeleton_actions, or residual_actions
ACTION_TYPE=${1:-raw_actions}

# Checkpoint save directory (default: ./checkpoints) /home/pengguanqi/Data/rl4vla/checkpoints
CHECKPOINT_DIR=${2:-./checkpoints}

# Full fine-tuning (default)
# uv run scripts/train_warmup.py \
#     --rlds_data_dir ../datasets \
#     --norm_stats_dir ./assets/pi05_warmup/warmup \
#     --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
#     --checkpoint_base_dir ${CHECKPOINT_DIR} \
#     --exp_name warmup_${ACTION_TYPE}_${TIMESTAMP} \
#     --batch_size 32 \
#     --num_train_steps 20000 \
#     --action_type ${ACTION_TYPE} \
#     --use_relative_state

# LoRA fine-tuning (full -- both paligemma and action expert)
uv run scripts/train_warmup.py \
    --rlds_data_dir ../datasets \
    --norm_stats_dir ./assets/pi05_warmup/warmup \
    --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
    --checkpoint_base_dir ${CHECKPOINT_DIR} \
    --exp_name warmup_lora_${ACTION_TYPE}_${TIMESTAMP} \
    --lora full \
    --batch_size 32 \
    --num_train_steps 20000 \
    --action_type ${ACTION_TYPE} \
    --use_relative_state

# LoRA fine-tuning (action expert only)
# uv run scripts/train_warmup.py \
#     --rlds_data_dir ../datasets \
#     --norm_stats_dir ./assets/pi05_warmup/warmup \
#     --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
#     --checkpoint_base_dir ${CHECKPOINT_DIR} \
#     --exp_name warmup_lora_ae_${ACTION_TYPE}_${TIMESTAMP} \
#     --lora action_expert_only \
#     --batch_size 32 \
#     --num_train_steps 20000 \
#     --action_type ${ACTION_TYPE} \
#     --use_relative_state
