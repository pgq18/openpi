export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false 
export PYTHONUNBUFFERED=1

uv run scripts/eval_warmup_openloop.py \
    --checkpoint_dir checkpoints/pi05_warmup/warmup_lora_test_20260504_150746/2000 \
    --episode_idx 91 \
    --use_relative_state