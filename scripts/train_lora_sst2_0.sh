#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="ckpt/Llama-3___2-1B-Instruct"
DATA="data/datasets/sst2/data"
SEED=0
OUTPUT_DIR="output_lora_sst2_seed0"

echo "Training sst2 dataset with LoRA fine-tuning (seed: $SEED)"

export CUDA_VISIBLE_DEVICES=0
python main.py \
  --model_name_or_path $MODEL \
  --is_training True \
  --data_path $DATA \
  --fp16 True \
  --output_dir $OUTPUT_DIR \
  --num_train_epochs 8 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --eval_strategy steps \
  --eval_steps 100 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 5e-4 \
  --weight_decay 0.01 \
  --adam_beta2 0.999 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --report_to none \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --dataloader_pin_memory False \
  --use_lora \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --seed $SEED
