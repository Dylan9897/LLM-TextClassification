# LLM文本分类项目

## 项目概述

这是一个基于大语言模型的文本分类项目，支持三种训练模式：**全参数微调**、**LoRA微调**和**DeepSpeed训练**。项目采用模块化设计，代码结构清晰，易于维护和扩展。

## 🚀 主要功能特性

### 训练模式
- **全参数微调 (Full Fine-tuning)**: 支持完整的模型参数训练
- **LoRA微调**: 使用低秩适应技术，大幅减少训练参数和显存占用
- **DeepSpeed训练**: 集成DeepSpeed ZeRO-3优化，支持大模型训练

### 数据集支持
- **SST-2**: 情感分析数据集 (2分类)
- **R8**: 新闻主题分类数据集 (8分类)
- **自动格式检测**: 支持JSON和Parquet格式数据集
- **智能数据切分**: 自动从训练集切分验证集 (15%比例)
- **提示词增强**: 为输入文本自动添加分类提示词

### 技术特性
- **混合精度训练**: 支持fp16模式，提升训练效率
- **梯度检查点**: 减少显存占用
- **最优模型保存**: 自动保存训练过程中的最佳模型
- **多随机种子**: 支持0、42、123三个随机种子，确保结果可复现
- **智能checkpoint管理**: 只保留最优的2个checkpoint
- **DeepSpeed ZeRO-3**: 支持模型状态分片，大幅减少显存占用

## 📁 项目结构

```
LLM-TextClassification/
├── main.py                    # 标准训练主脚本（全参微调 + LoRA）
├── main_deepspeed.py          # DeepSpeed训练主脚本
├── train.sh                   # 批量训练脚本
├── requirements.txt            # 依赖包列表
├── src/                       # 源代码目录
│   ├── config/               # 配置文件
│   │   ├── training_args.py  # 训练参数定义
│   │   └── dataset_config.py # 数据集配置
│   ├── data/                 # 数据处理模块
│   │   └── dataset_loader.py # 数据集加载器
│   ├── models/               # 模型管理模块
│   │   └── model_manager.py  # 模型管理器
│   ├── training/             # 训练模块
│   │   ├── trainer.py        # 标准训练器
│   │   └── deepspeed_trainer.py # DeepSpeed训练器
│   ├── evaluation/           # 评估模块
│   │   └── test_evaluator.py # 测试集评估器
│   └── utils/                # 工具模块
│       ├── helpers.py        # 辅助函数
│       └── deepspeed_utils.py # DeepSpeed工具
├── scripts/                   # 脚本目录
│   ├── train_full_*.sh      # 全参微调脚本
│   ├── train_lora_*.sh      # LoRA微调脚本
│   ├── eval_deepspeed.sh    # DeepSpeed模型评估脚本
│   └── eval_deepspeed_model.py # DeepSpeed评估核心逻辑
├── data/                     # 数据集目录
├── ckpt/                     # 预训练模型目录
└── outputs/                  # 训练输出目录
    └── models/              # 训练完成的模型
```

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (LoRA支持)
- DeepSpeed (DeepSpeed训练支持)
- Datasets
- Scikit-learn

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将数据集放置在 `data/datasets/` 目录下，支持以下结构：

```
data/datasets/sst2/data/
├── train.json
├── dev.json
└── test.json

# 或
data/datasets/r8/data/
├── train-00000-of-00001.parquet
└── test-00000-of-00001.parquet
```

### 3. 运行训练

#### 标准训练（全参微调 + LoRA）
```bash
# 全参微调
python main.py --use_lora false --max_samples 100

# LoRA微调
python main.py --use_lora true --max_samples 100
```

#### DeepSpeed训练
```bash
# DeepSpeed训练（自动检测并保存最优模型）
python main_deepspeed.py --deepspeed --max_samples 100
```

#### 批量训练
```bash
# 运行所有训练脚本
bash train.sh
```

## 📊 训练参数说明

### 全参数微调
- **学习率**: 2e-5 (适合大模型微调)
- **批次大小**: 4 (根据显存调整)
- **训练轮数**: 5 epochs
- **评估频率**: 每50步
- **保存频率**: 每100步
- **保存限制**: 最多2个checkpoint

### LoRA微调
- **学习率**: 5e-4 (LoRA通常需要更高学习率)
- **批次大小**: 8 (LoRA显存占用更少)
- **训练轮数**: 8 epochs
- **评估频率**: 每100步
- **保存频率**: 每200步
- **LoRA配置**: r=64, alpha=16, dropout=0.1

### DeepSpeed训练
- **ZeRO阶段**: ZeRO-3 (模型状态分片)
- **混合精度**: fp16
- **CPU卸载**: 支持
- **梯度累积**: 自动调整
- **最优模型保存**: 训练完成后自动保存最终模型

## 🔧 模型配置

### 自定义分类头 (LoRA模式)
```python
class CustomClassifier(nn.Module):
    def __init__(self, input_size, out_features, hidden_dim=512):
        # 三层全连接网络
        # input_size -> 512 -> 256 -> out_features
        # 使用Xavier均匀初始化
        # 包含ReLU激活和Dropout正则化
```

### LoRA目标模块
- **注意力层**: q_proj, k_proj, v_proj, o_proj
- **MLP层**: gate_proj, up_proj, down_proj

## 📈 模型评估

### 标准训练模式
- 训练完成后自动进行测试集评估
- 生成详细的分类报告和混淆矩阵
- 保存预测结果和性能指标

### DeepSpeed训练模式
- 训练完成后只保存模型，不进行测试集预测
- 使用单独的评估脚本进行测试集预测
- 支持智能模型检测（自动识别PyTorch模型或DeepSpeed检查点）

#### DeepSpeed模型评估
```bash
# 使用评估脚本
bash scripts/eval_deepspeed.sh

# 或直接运行Python脚本
python scripts/eval_deepspeed_model.py \
    --checkpoint_dir output_deepspeed_r8_seed0 \
    --data_path data/datasets/r8/data \
    --output_dir eval_results_r8 \
    --base_model_path ckpt/Llama-3___2-1B-Instruct \
    --seed 0 \
    --model_max_length 512
```

## 📤 输出说明

### 标准训练输出
训练完成后，每个实验会在对应的输出目录生成：
- `pytorch_model.bin`: 训练好的模型权重
- `adapter_model.safetensors`: LoRA模型权重
- `test_results.json`: 测试集评估结果
- `test_summary.txt`: 测试结果摘要
- `checkpoint-*/`: 训练过程中的checkpoint

### DeepSpeed训练输出
- `pytorch_model.bin`: 最终模型权重（已合并）
- `zero_to_fp32.py`: DeepSpeed权重合并脚本
- `checkpoint-*/`: 训练过程中的DeepSpeed检查点
- 评估结果保存在指定的输出目录中

## 💬 提示词格式

系统会自动为输入文本添加分类提示词：
```
Please carefully read the following reference text:"{text}", 
and determine which of the following categories it belongs to:{class_labels},answer:
```

其中 `{class_labels}` 根据数据集类型自动生成：
- **SST-2**: ["Negative", "Positive"]
- **R8**: ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship"]

## ⚠️ 注意事项

### 显存管理
- **全参微调**: 需要较大显存，建议使用LoRA模式
- **LoRA微调**: 显存占用较少，适合大多数场景
- **DeepSpeed训练**: 支持大模型训练，显存占用可控

### 数据格式
- 确保数据集包含正确的文本和标签列
- 支持JSON和Parquet两种格式
- 自动检测数据集类型和标签数量

### 模型路径
- 检查预训练模型路径是否正确
- DeepSpeed训练需要基础模型配置文件

### 随机种子
- 不同种子会产生不同的训练结果
- 支持0、42、123三个预设种子

## 🔍 故障排除

### 常见问题
1. **显存不足**: 减少batch_size或使用LoRA模式
2. **模型加载失败**: 检查模型路径和配置文件
3. **DeepSpeed评估错误**: 使用修复后的评估脚本
4. **数据加载问题**: 检查数据格式和路径

### 最近修复
- ✅ 修复了DeepSpeed训练后的预测错误
- ✅ 优化了模型加载逻辑，支持缺少配置文件的情况
- ✅ 改进了评估脚本的错误处理
- ✅ 分离了训练和评估流程，提高稳定性

## 📄 许可证

本项目采用MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 支持

如果您在使用过程中遇到问题，请：
1. 查看本文档的故障排除部分
2. 检查项目的Issue页面
3. 提交新的Issue描述问题

---

**项目状态**: ✅ 稳定可用  
**最后更新**: 2025-08-27  
**版本**: v2.0.0

