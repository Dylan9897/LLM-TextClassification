#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --model MODEL_PATH      预训练模型路径 (默认: $DEFAULT_MODEL)"
    echo "  --data DATA_PATH        数据集路径 (默认: $DEFAULT_DATA)"
    echo "  --output OUTPUT_DIR     输出目录 (默认: $DEFAULT_OUTPUT_DIR)"
    echo "  --seed SEED             随机种子 (默认: $DEFAULT_SEED)"
    echo "  --max-samples N         最大样本数 (默认: $DEFAULT_MAX_SAMPLES)"
    echo "  --epochs EPOCHS         训练轮数 (默认: $DEFAULT_EPOCHS)"
    echo "  --batch-size BATCH_SIZE 批处理大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  --help, -h              显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 --max-samples 50                   # 使用50个样本"
    echo "  $0 --seed 123 --epochs 2              # 自定义种子和轮数"
}

# 默认配置
DEFAULT_MODEL="ckpt/Llama-3___2-1B-Instruct"
DEFAULT_DATA="data/datasets/r8/data"
DEFAULT_SEED=0
DEFAULT_OUTPUT_DIR="outputs/models/output_lora_r8_seed0"
DEFAULT_MAX_SAMPLES=100
DEFAULT_EPOCHS=1
DEFAULT_BATCH_SIZE=8

# 解析参数
parse_args() {
    MODEL="$DEFAULT_MODEL"
    DATA="$DEFAULT_DATA"
    SEED="$DEFAULT_SEED"
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    MAX_SAMPLES="$DEFAULT_MAX_SAMPLES"
    EPOCHS="$DEFAULT_EPOCHS"
    BATCH_SIZE="$DEFAULT_BATCH_SIZE"

    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --data)
                DATA="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --max-samples)
                MAX_SAMPLES="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "未知参数: $1"
                echo "使用 --help 查看帮助信息"
                exit 1
                ;;
        esac
    done
}

# 解析传入的参数
parse_args "$@"

echo "Training r8 dataset with LoRA Fine-tuning (seed: $SEED)"
echo "Working directory: $(pwd)"
echo "Model: $MODEL"
echo "Data: $DATA"
echo "Output: $OUTPUT_DIR"
echo "Max samples: $MAX_SAMPLES (quick test mode)"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 启动LoRA训练
python main.py \
    --model_name_or_path "$MODEL" \
    --is_training True \
    --data_path "$DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --max_samples "$MAX_SAMPLES" \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.999 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --logging_steps 25 \
    --report_to none \
    --model_max_length 512 \
    --log_level info \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --dataloader_pin_memory False \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --seed "$SEED"

echo ""
echo "LoRA fine-tuning completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "Note: LoRA weights are saved as adapter_model.safetensors"
