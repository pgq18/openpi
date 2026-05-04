export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Full fine-tuning (default)
# uv run scripts/train_warmup.py \
#     --rlds_data_dir ../datasets \
#     --norm_stats_dir ./assets/pi05_warmup/warmup \
#     --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
#     --exp_name warmup_test_${TIMESTAMP} \
#     --batch_size 32 \
#     --num_train_steps 20000

# LoRA fine-tuning (full -- both paligemma and action expert)
uv run scripts/train_warmup.py \
    --rlds_data_dir ../datasets \
    --norm_stats_dir ./assets/pi05_warmup/warmup \
    --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
    --exp_name warmup_lora_test_${TIMESTAMP} \
    --lora full \
    --batch_size 32 \
    --num_train_steps 20000

# LoRA fine-tuning (action expert only)
# uv run scripts/train_warmup.py \
#     --rlds_data_dir ../datasets \
#     --norm_stats_dir ./assets/pi05_warmup/warmup \
#     --checkpoint_path /home/pengguanqi/Models/pi05_base/params \
#     --exp_name warmup_lora_ae_${TIMESTAMP} \
#     --lora action_expert_only \
#     --batch_size 32 \
#     --num_train_steps 20000
