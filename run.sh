#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="ckpt/DeepSeek-R1-Distill-Qwen-1___5B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="data/datasets/longnews"

function usage() {
    echo '
Usage: bash finetune/finetune_lora_single_gpu.sh [-m MODEL_PATH] [-d DATA_PATH]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

export CUDA_VISIBLE_DEVICES=0
python lora_predict.py \
  --model_name_or_path $MODEL \
  --is_training True \
  --add_adapter True \
  --data_path $DATA \
  --bf16 True \
  --output_dir output_qwen/longnews\
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 10 \
  --learning_rate 5e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --warmup_steps 1000 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora

