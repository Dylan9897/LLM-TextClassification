#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型文本分类训练脚本
支持全参微调和LoRA微调两种模式
"""

import transformers
from transformers import AutoTokenizer

# 导入自定义模块
from module.argument import ModelArguments, DataArguments, TrainingArguments, LoraArguments
from src.data.dataset_loader import DatasetLoader
from src.models.model_manager import ModelManager
from src.training.trainer import BestModelTrainer
from src.evaluation.test_evaluator import TestEvaluator
from src.utils.helpers import set_seed, detect_dataset_name
from src.config.dataset_config import get_num_labels


def train(verbose=False):
    """主训练函数"""
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
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
    
    # 创建trainer
    trainer = BestModelTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["valid"],
    )
    
    # 训练模型
    print("\n" + "="*60)
    print("Starting training...")
    print(f"Training arguments: {training_args}")
    print(f"Model type: {'LoRA' if training_args.use_lora else 'Full Fine-tuning'}")
    print("="*60)
    
    try:
        print("Calling trainer.train()...")
        trainer.train()
        print("Training completed, saving model...")
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to evaluate with available checkpoints...")
    
    # 评估测试集
    test_evaluator = TestEvaluator(model, tokenizer, dataset_loader, training_args.output_dir, model_args.model_name_or_path)
    test_evaluator.evaluate()


if __name__ == "__main__":
    train(verbose=True)
