#!/bin/bash

# DeepSpeed模型评估脚本
# 用于合并ZeRO权重并生成测试结果

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 配置参数
CHECKPOINT_DIR="output_deepspeed_r8_seed0"  # DeepSpeed检查点目录
DATA_PATH="data/datasets/r8/data"           # 数据集路径
OUTPUT_DIR="eval_results_r8"                # 评估结果输出目录
BASE_MODEL_PATH="ckpt/Llama-3___2-1B-Instruct"  # 基础模型路径
SEED=0                                      # 随机种子

echo "Running DeepSpeed Model Evaluation..."
echo "Working directory: $(pwd)"
echo ""

# 检查检查点目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please check if the training has completed successfully."
    exit 1
fi

# 检查数据集路径是否存在
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path not found: $DATA_PATH"
    echo "Please check if the dataset path is correct."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"
echo "  Data Path: $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Base Model Path: $BASE_MODEL_PATH"
echo "  Seed: $SEED"
echo ""

# 运行评估
python scripts/eval_deepspeed_model.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --base_model_path "$BASE_MODEL_PATH" \
    --seed "$SEED" \
    --model_max_length 512

echo ""
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
