# 大模型文本分类工具包

本项目旨在提供一个灵活高效的文本分类解决方案，基于先进的大语言模型（LLM），包括Qwen和DeepSeek。项目支持两种主要模式：直接使用预训练的LLM结合自定义分类层进行文本分类，以及通过LoRA（Low-Rank Adaptation）技术对LLM进行微调后添加分类层以实现更精准的分类效果。

## 主要特性：
- **双模型支持**：集成Qwen与DeepSeek两大先进语言模型。
- **多样化部署方案**：支持纯LLM+分类层模式及LLM+LoRA+分类层模式。
- **易于扩展**：模块化设计便于根据需要调整或替换组件。

欢迎贡献代码、提出问题或分享您的使用案例！

## 配置环境、下载模型及运行项目

### 配置环境

#### 1. 安装依赖
首先，请确保您已安装了Python（推荐版本3.10及以上）。然后，通过以下命令安装所需的依赖项：

```bash
pip install -r requirements.txt
```

#### 2.下载模型

本项目支持Qwen和DeepSeek等多种预训练语言模型。您可以通过以下方式下载所需模型（以Qwen和DeepSeek为例）：

**使用Qwen模型**

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',cache_dir="./ckpt")
```

**使用DeepSeek模型**

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',cache_dir="./ckpt")
```

#### 3.准备数据集

​	本代码专注于实现文本分类任务，其数据源于一个专门构建的长文本分类数据集。在数据预处理阶段，我们采用了0.15的比例对原始数据进行划分，以构建评估模型性能的测试集与用于学习的训练集。具体而言，训练集包含5950条精心挑选的样本，而测试集则包括了1050条样本。该分类体系涵盖了七个核心领域，分别是：“时尚”、“财经”、“时政”、“家居”、“房产”、“教育”以及“科技”。

#### 4、训练

​	**LLM（全参微调）+分类层模式：**

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="Qwen2.5-0.5B-Instruct" 
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
python main.py \
  --model_name_or_path $MODEL \
  --is_training True \
  --data_path $DATA \
  --bf16 True \
  --output_dir output_qwen/longnews \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing
```

​	**LLM+LoRA+分类层模式**

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="Qwen2.5-0.5B-Instruct" 
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
python main.py \
  --model_name_or_path $MODEL \
  --is_training True \
  --data_path $DATA \
  --bf16 True \
  --output_dir output_qwen/longnews \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora
```

#### 5、实验结果

（1）qwen全参微调+分类层

```bash
              precision    recall  f1-score   support

          教育       0.98      0.94      0.96       154
          财经       0.95      0.96      0.95       130
          科技       0.95      0.98      0.96       135
          房产       0.99      0.94      0.96       156
          时政       0.90      0.95      0.92       130
          家居       0.95      0.94      0.95       158
          时尚       0.99      1.00      0.99       138

    accuracy                           0.96      1001
   macro avg       0.96      0.96      0.96      1001
weighted avg       0.96      0.96      0.96      1001
```

（2）lora+deepseek+分类层结果

```bash
              precision    recall  f1-score   support

        教育       0.92      0.89      0.90       154
        财经       0.82      0.93      0.87       130
        科技       0.84      0.96      0.90       135
        房产       0.87      0.88      0.87       156
        时政       0.91      0.79      0.85       130
        家居       0.91      0.76      0.83       158
        时尚       0.92      0.98      0.95       138

    accuracy                           0.88      1001
   macro avg       0.88      0.88      0.88      1001
weighted avg       0.89      0.88      0.88      1001
```

（3）lora+deepseek+TemporalAttention分类层的结果

```bash
              precision    recall  f1-score   support

        教育       0.92      0.87      0.90       154
        财经       0.81      0.90      0.85       130
        科技       0.78      0.97      0.87       135
        房产       0.94      0.85      0.90       156
        时政       0.85      0.87      0.86       130
        家居       0.91      0.74      0.82       158
        时尚       0.94      0.96      0.95       138

    accuracy                           0.88      1001
   macro avg       0.88      0.88      0.88      1001
weighted avg       0.88      0.88      0.88      1001
```

（4）[bert-base-chinese分类](https://github.com/Dylan9897/ai-nlp-project/tree/main/TextClassification)结果对比：

```bash
                precision    recall  f1-score   support

          教育       0.97      0.98      0.97       154
          财经       0.97      0.94      0.95       130
          科技       0.97      0.99      0.98       135
          房产       0.95      0.95      0.95       156
          时政       0.95      0.95      0.95       130
          家居       0.96      0.96      0.96       158
          时尚       0.99      1.00      1.00       138

    accuracy                           0.97      1001
   macro avg       0.97      0.97      0.97      1001
weighted avg       0.97      0.97      0.97      1001
```

在Thucnews数据集上的实验结果，以Qwen全参微调为例：

```bash
               precision    recall  f1-score   support

      finance       0.92      0.87      0.89      1000
       realty       0.92      0.93      0.92      1000
       stocks       0.83      0.84      0.84      1000
    education       0.94      0.94      0.94      1000
      science       0.83      0.86      0.85      1000
      society       0.88      0.91      0.89      1000
     politics       0.88      0.88      0.88      1000
       sports       0.95      0.94      0.94      1000
         game       0.93      0.91      0.92      1000
entertainment       0.90      0.91      0.91      1000

     accuracy                           0.90     10000
    macro avg       0.90      0.90      0.90     10000
 weighted avg       0.90      0.90      0.90     10000
```

[bert-base-chinese分类](https://github.com/Dylan9897/ai-nlp-project/tree/main/TextClassification)结果对比：

```bash
               precision    recall  f1-score   support

      finance       0.92      0.93      0.92      1000
       realty       0.96      0.95      0.95      1000
       stocks       0.91      0.89      0.90      1000
    education       0.96      0.97      0.97      1000
      science       0.91      0.90      0.91      1000
      society       0.90      0.95      0.93      1000
     politics       0.92      0.92      0.92      1000
       sports       0.98      0.98      0.98      1000
         game       0.97      0.94      0.95      1000
entertainment       0.95      0.97      0.96      1000

     accuracy                           0.94     10000
    macro avg       0.94      0.94      0.94     10000
 weighted avg       0.94      0.94      0.94     10000
```

#### 6.常问问题

**（1）项目完美吗？**

答：这是一个每周夜间马拉松项目，请向我们提供反馈，我们将改进它。

**（2）为什么不直接使用LLM**

答：分类器需要输出一个准确而有效的类，LLM可能会回答"**根据给定的内容，类别是\*\*\*，嗯……这取决于……**"，编写解析器很麻烦。

