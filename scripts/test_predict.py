#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集预测脚本
用于运行训练好的模型在测试集上的预测，并生成详细的评估结果
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.evaluation.test_predictor import TestPredictor
from src.utils.helpers import set_seed, detect_dataset_name
from src.config.dataset_config import get_num_labels


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Test Set Prediction Script")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True,
                       help="数据集路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    
    # 可选参数
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="基础模型路径（用于LoRA模型）")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="tokenizer路径（如果不指定，使用model_path）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--model_max_length", type=int, default=512,
                       help="模型最大输入长度")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # 检测数据集类型
    dataset_name = detect_dataset_name(args.data_path)
    if dataset_name:
        num_labels = get_num_labels(dataset_name)
        print(f"Detected dataset: {dataset_name} with {num_labels} labels")
    else:
        num_labels = 7
        print(f"Could not detect dataset type, using default {num_labels} labels")
    
    # 设置tokenizer路径
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建训练参数（用于dataset_loader）
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        model_max_length=args.model_max_length,
        per_device_eval_batch_size=args.batch_size,
        dataloader_pin_memory=False,
        seed=args.seed
    )
    
    # 创建数据参数
    from src.config.training_args import DataArguments
    data_args = DataArguments(
        data_path=args.data_path,
        model_max_length=args.model_max_length
    )
    
    # 创建dataset_loader
    dataset_loader = DatasetLoader(data_args, training_args, dataset_name)
    
    # 创建测试预测器
    predictor = TestPredictor(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_loader=dataset_loader,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path
    )
    
    # 执行预测
    print("\n" + "="*60)
    print("Starting Test Set Prediction...")
    print(f"Model Path: {args.model_path}")
    print(f"Tokenizer Path: {args.tokenizer_path}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)
    
    try:
        results, metrics = predictor.predict()
        
        if results and metrics:
            print("\n" + "="*60)
            print("Prediction Completed Successfully!")
            print("="*60)
            
            # 显示主要指标
            print(f"\nPerformance Summary:")
            print(f"  Total Samples: {len(results)}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            if metrics['auc'] is not None:
                print(f"  AUC: {metrics['auc']:.4f}")
            
            # 获取预测摘要
            summary = predictor.get_prediction_summary()
            if summary:
                print(f"\nPrediction Summary:")
                print(f"  Correct Predictions: {summary['correct_predictions']}")
                print(f"  Incorrect Predictions: {summary['total_samples'] - summary['correct_predictions']}")
                print(f"  Success Rate: {summary['correct_predictions']/summary['total_samples']:.2%}")
        
        else:
            print("Prediction failed or returned no results")
            
    except Exception as e:
        print(f"\nPrediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
