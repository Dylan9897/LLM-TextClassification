#!/bin/bash

# 测试集预测脚本
# 用于运行训练好的模型在测试集上的预测

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 配置参数
MODEL_PATH="output_deepspeed_r8_seed0"  # 训练好的模型路径
DATA_PATH="data/datasets/r8/data"       # 数据集路径
OUTPUT_DIR="test_results_r8"            # 测试结果输出目录
BASE_MODEL_PATH="ckpt/Llama-3___2-1B-Instruct"  # 基础模型路径（用于LoRA）
SEED=0                                  # 随机种子

echo "Running Test Set Prediction..."
echo "Working directory: $(pwd)"
echo ""

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path not found: $MODEL_PATH"
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
echo "  Model Path: $MODEL_PATH"
echo "  Data Path: $DATA_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Base Model Path: $BASE_MODEL_PATH"
echo "  Seed: $SEED"
echo ""

# 运行测试预测
python scripts/test_predict.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --base_model_path "$BASE_MODEL_PATH" \
    --seed "$SEED" \
    --model_max_length 512 \
    --batch_size 8

echo ""
echo "Test prediction completed!"
echo "Results saved to: $OUTPUT_DIR"
