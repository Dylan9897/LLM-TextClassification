#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="ckpt/Llama-3___2-1B-Instruct"
DATA="data/datasets/r8/data"
SEED=42
OUTPUT_DIR="output_full_r8_seed42"

echo "Training r8 dataset with full fine-tuning (seed: $SEED)"

export CUDA_VISIBLE_DEVICES=0
python main.py \
  --model_name_or_path $MODEL \
  --is_training True \
  --data_path $DATA \
  --fp16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --eval_strategy steps \
  --eval_steps 1000 \
  --save_strategy steps \
  --save_steps 2000 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --adam_beta2 0.999 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 100 \
  --report_to none \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --dataloader_pin_memory False \
  --seed $SEED
