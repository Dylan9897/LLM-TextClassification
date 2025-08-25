# LLM文本分类项目

## 项目概述

这是一个基于大语言模型的文本分类项目，支持全参数微调和LoRA微调两种训练模式。项目采用模块化设计，代码结构清晰，易于维护和扩展。

## 主要功能特性

### 🚀 训练模式
- **全参数微调 (Full Fine-tuning)**: 支持完整的模型参数训练
- **LoRA微调**: 使用低秩适应技术，大幅减少训练参数和显存占用
- **自定义分类头**: LoRA模式下使用专门设计的分类网络结构

### 📊 数据集支持
- **SST-2**: 情感分析数据集 (2分类)
- **R8**: 新闻主题分类数据集 (8分类)
- **自动格式检测**: 支持JSON和Parquet格式数据集
- **智能数据切分**: 自动从训练集切分验证集 (15%比例)
- **提示词增强**: 为输入文本自动添加分类提示词

### 🔧 技术特性
- **混合精度训练**: 支持fp16模式，提升训练效率
- **梯度检查点**: 减少显存占用
- **最优模型保存**: 自动保存训练过程中的最佳模型
- **多随机种子**: 支持0、42、123三个随机种子，确保结果可复现
- **智能checkpoint管理**: 只保留最优的2个checkpoint

## 项目结构

```
LLM-TextClassification/
├── main.py                 # 主训练脚本
├── train.sh               # 批量训练脚本
├── requirements.txt        # 依赖包列表
├── module/                 # 核心模块
│   └── argument.py        # 参数定义
├── src/                   # 源代码目录
│   ├── config/           # 配置文件
│   │   └── dataset_config.py
│   ├── data/             # 数据处理模块
│   │   └── dataset_loader.py
│   ├── models/           # 模型管理模块
│   │   └── model_manager.py
│   ├── training/         # 训练模块
│   │   └── trainer.py
│   ├── evaluation/       # 评估模块
│   │   └── test_evaluator.py
│   └── utils/            # 工具模块
│       └── helpers.py
├── scripts/               # 训练脚本
│   ├── train_full_*.sh   # 全参微调脚本
│   └── train_lora_*.sh   # LoRA微调脚本
├── data/                  # 数据集目录
└── ckpt/                  # 预训练模型目录
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (LoRA支持)
- Datasets
- Scikit-learn

## 快速开始

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

#### 单个训练脚本
```bash
# 全参微调
bash scripts/train_full_sst2_0.sh

# LoRA微调
bash scripts/train_lora_sst2_0.sh
```

#### 批量训练
```bash
# 运行所有训练脚本
bash train.sh
```

## 训练参数说明

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

## 模型配置

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

## 输出说明

训练完成后，每个实验会在对应的输出目录生成：
- `model.safetensors` / `adapter_model.safetensors`: 训练好的模型
- `test_results.json`: 测试集评估结果
- `test_summary.txt`: 测试结果摘要
- `checkpoint-*/`: 训练过程中的checkpoint

## 提示词格式

系统会自动为输入文本添加分类提示词：
```
Please carefully read the following reference text:"{text}", 
and determine which of the following categories it belongs to:{class_labels},answer:
```

其中 `{class_labels}` 根据数据集类型自动生成：
- **SST-2**: ["Negative", "Positive"]
- **R8**: ["acq", "corn", "crude", "earn", "grain", "interest", "money-fx", "ship"]

## 注意事项

1. **显存管理**: 全参微调需要较大显存，建议使用LoRA模式
2. **数据格式**: 确保数据集包含正确的文本和标签列
3. **模型路径**: 检查预训练模型路径是否正确
4. **随机种子**: 不同种子会产生不同的训练结果

## 许可证

本项目采用MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目！

