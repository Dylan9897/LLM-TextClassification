# 项目结构说明

## 📁 整体架构

```
LLM-TextClassification/
├── main.py                      # 标准训练入口
├── main_deepspeed.py            # DeepSpeed训练入口
├── 📁 src/                      # 核心源代码
│   ├── 📁 config/               # 配置管理（包含训练参数定义）
│   ├── 📁 data/                 # 数据处理
│   ├── 📁 models/               # 模型定义
│   ├── 📁 training/             # 训练逻辑
│   ├── 📁 evaluation/           # 评估模块
│   └── 📁 utils/                # 工具函数
├── 📁 scripts/                  # 脚本文件
│   ├── train_deepspeed_r8.sh    # R8数据集DeepSpeed训练脚本
│   ├── train_deepspeed_sst2.sh  # SST2数据集DeepSpeed训练脚本
│   ├── eval_deepspeed_model.py  # DeepSpeed模型评估脚本
│   ├── run_eval_deepspeed.sh    # 运行DeepSpeed评估的Shell脚本
│   ├── test_predict.py          # 测试预测Python脚本
│   ├── run_test_prediction.sh   # 运行测试预测的Shell脚本
│   └── test_deepspeed.sh        # DeepSpeed测试脚本
├── 📁 configs/                  # 配置文件
│   └── 📁 deepspeed/            # DeepSpeed配置
│       └── ds_config.json
├── 📁 outputs/                  # 输出目录
│   ├── 📁 models/               # 训练好的模型
│   ├── 📁 results/              # 评估结果
│   └── 📁 logs/                 # 日志文件
├── 📁 data/                     # 数据集
├── 📁 ckpt/                     # 预训练模型
├── 📁 docs/                     # 文档

├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```

## 🎯 目录功能说明

### **主入口文件**
- `main.py`: 标准训练流程入口
- `main_deepspeed.py`: DeepSpeed训练流程入口

### **src/** - 核心源代码
- **config/**: 数据集配置、模型配置等
- **data/**: 数据加载、预处理、数据集管理
- **models/**: 模型定义、模型管理器
- **training/**: 训练器、训练逻辑
- **evaluation/**: 评估器、测试预测
- **utils/**: 工具函数、辅助功能

### **scripts/** - 脚本文件
- **train/**: 各种训练脚本（DeepSpeed、标准训练）
- **eval/**: 模型评估脚本
- **test/**: 测试预测脚本
- **deepspeed/**: DeepSpeed专用脚本

### **configs/** - 配置文件
- **deepspeed/**: DeepSpeed相关配置
- **training/**: 训练相关配置

### **outputs/** - 输出目录
- **models/**: 训练好的模型文件
- **results/**: 评估结果、预测结果
- **logs/**: 训练日志、错误日志

## 🚀 使用流程

### 1. 训练模型
```bash
# 使用最终训练脚本（推荐）
./train.sh                    # 默认DeepSpeed训练
./train.sh -n 50             # 快速测试模式
./train.sh -t standard       # 标准训练

# 或使用具体脚本
./scripts/train_deepspeed_r8.sh
./scripts/train_deepspeed_sst2.sh
```

### 2. 评估模型
```bash
# DeepSpeed模型评估
./scripts/run_eval_deepspeed.sh

# 测试预测
./scripts/run_test_prediction.sh
```

### 3. 配置文件
```bash
# DeepSpeed配置
configs/deepspeed/ds_config.json
```

## 📝 注意事项

1. **路径更新**: 重组后，所有脚本中的路径引用需要相应更新
2. **配置文件**: 确保配置文件路径正确
3. **输出目录**: 新的输出目录结构更加清晰
4. **模块导入**: 确保Python模块导入路径正确

## 🔧 维护建议

1. **定期清理**: 定期清理 `outputs/logs/` 中的旧日志
2. **模型管理**: 在 `outputs/models/` 中按项目组织模型文件
3. **结果归档**: 将重要的评估结果归档到 `outputs/results/`
4. **配置版本**: 为不同实验保存配置文件版本
