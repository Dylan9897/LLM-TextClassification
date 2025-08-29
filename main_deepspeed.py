#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型文本分类训练脚本 - DeepSpeed 版本
支持全参微调和LoRA微调两种模式，集成 DeepSpeed ZeRO-3 优化
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import transformers
from transformers import AutoTokenizer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

# 导入自定义模块
from src.config.training_args import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from src.data.dataset_loader import DatasetLoader
from src.models.model_manager import ModelManager
from src.training.deepspeed_trainer import BestModelDeepSpeedTrainer
# from src.evaluation.test_evaluator import TestEvaluator  # 不再需要
from src.utils.helpers import set_seed, detect_dataset_name
from src.utils.deepspeed_utils import check_deepspeed_installation, get_memory_usage
from src.config.dataset_config import get_num_labels


def train_with_deepspeed(verbose=False):
    """使用 DeepSpeed 的主训练函数"""
    
    # 检查 DeepSpeed 安装
    if not check_deepspeed_installation():
        print("DeepSpeed is not installed. Please install it first:")
        print("pip install deepspeed")
        return
    
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    # 设置 DeepSpeed 配置
    if not hasattr(training_args, 'deepspeed') or not training_args.deepspeed:
        # 如果没有指定 DeepSpeed 配置，使用默认配置
        training_args.deepspeed = "ds_config.json"
        print(f"Using default DeepSpeed config: {training_args.deepspeed}")
    
    # 设置随机种子
    set_seed(training_args.seed)
    print(f"Random seed set to: {training_args.seed}")
    
    # 检测数据集类型
    dataset_name = detect_dataset_name(data_args.data_path)
    if dataset_name:
        num_labels = get_num_labels(dataset_name)
        print(f"Detected dataset: {dataset_name} with {num_labels} labels")
    else:
        num_labels = 7
        print(f"Could not detect dataset type, using default {num_labels} labels")
    
    # 显示 GPU 内存信息
    memory_info = get_memory_usage()
    print(f"GPU Memory: {memory_info['gpu_memory']:.2f} GB")
    print(f"GPU Memory Allocated: {memory_info['gpu_memory_allocated']:.2f} GB")
    
    # 创建模型
    model_manager = ModelManager(model_args, training_args, lora_args, num_labels)
    model = model_manager.create_model()
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    
    # 为Llama模型设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置DatasetLoader的tokenizer
    dataset_loader = DatasetLoader(data_args, training_args, dataset_name, tokenizer)
    dataset_loader._tokenize_text = lambda texts: tokenizer(texts, padding="max_length", truncation=True)
    
    # 加载和预处理数据
    processed_data = dataset_loader.load_and_split_data()
    
    # 检查是否有检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Found checkpoint: {last_checkpoint}")
            training_args.resume_from_checkpoint = last_checkpoint
    
    # 创建 DeepSpeed trainer
    trainer = BestModelDeepSpeedTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["valid"],
        deepspeed_config=training_args.deepspeed,
    )
    
    # 训练模型
    print("\n" + "="*60)
    print("Starting training with DeepSpeed...")
    print(f"Training arguments: {training_args}")
    print(f"DeepSpeed config: {training_args.deepspeed}")
    print(f"Model type: {'LoRA' if training_args.use_lora else 'Full Fine-tuning'}")
    print("="*60)
    
    try:
        print("Calling trainer.train()...")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        print("Training completed, saving model...")
        
        # 保存最终模型
        trainer.save_state()
        trainer.save_final_model(output_dir=training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to evaluate with available checkpoints...")
    
    # 训练完成，不进行测试集预测
    # 如需评估，请单独运行预测脚本
    print("\nTraining completed successfully!")
    print("To evaluate the model on test set, please run the evaluation script separately.")


def train_without_deepspeed(verbose=False):
    """不使用 DeepSpeed 的标准训练函数"""
    print("Running standard training without DeepSpeed...")
    
    # 导入原始训练函数
    from main import train
    train(verbose)


if __name__ == "__main__":
    # 检查命令行参数是否包含 --deepspeed
    use_deepspeed = "--deepspeed" in sys.argv
    
    if use_deepspeed:
        print("DeepSpeed mode enabled")
        train_with_deepspeed(verbose=True)
    else:
        print("Standard training mode")
        train_without_deepspeed(verbose=True)
