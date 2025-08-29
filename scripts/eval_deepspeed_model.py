#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSpeed模型评估脚本
专门用于处理DeepSpeed ZeRO训练后的模型，合并权重并生成测试结果
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.utils.helpers import set_seed, detect_dataset_name
from src.config.dataset_config import get_num_labels


def merge_deepspeed_checkpoint(checkpoint_dir, output_dir):
    """合并DeepSpeed ZeRO检查点"""
    print(f"正在合并DeepSpeed检查点: {checkpoint_dir}")
    
    # 检查是否有zero_to_fp32.py脚本（在根目录）
    checkpoint_root = os.path.dirname(checkpoint_dir)
    zero_to_fp32_script = os.path.join(checkpoint_root, "zero_to_fp32.py")
    if not os.path.exists(zero_to_fp32_script):
        print(f"错误: 找不到zero_to_fp32.py脚本: {zero_to_fp32_script}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行合并脚本
    import subprocess
    try:
        # 使用绝对路径
        cmd = [
            sys.executable, 
            os.path.abspath(zero_to_fp32_script), 
            os.path.abspath(checkpoint_dir), 
            os.path.abspath(os.path.join(output_dir, "pytorch_model.bin"))
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(zero_to_fp32_script))
        
        if result.returncode == 0:
            print("✅ DeepSpeed检查点合并成功")
            return True
        else:
            print(f"❌ 合并失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 执行合并脚本时出错: {e}")
        return False


def find_latest_checkpoint(checkpoint_dir):
    """找到最新的检查点目录"""
    print(f"搜索检查点目录: {checkpoint_dir}")
    checkpoints = []
    
    for item in os.listdir(checkpoint_dir):
        print(f"检查项目: {item}")
        if item.startswith("global_step"):
            try:
                # 提取步骤数字，格式是 "global_step291"
                step_str = item.replace("global_step", "")
                step = int(step_str)
                checkpoints.append((step, item))
                print(f"找到检查点: {item} (步骤 {step})")
            except Exception as e:
                print(f"解析检查点失败: {item}, 错误: {e}")
                continue
    
    if not checkpoints:
        print("没有找到检查点")
        return None
    
    # 按步骤排序，返回最新的
    checkpoints.sort(key=lambda x: x[0])
    latest_step, latest_dir = checkpoints[-1]
    
    print(f"找到最新检查点: {latest_dir} (步骤 {latest_step})")
    return os.path.join(checkpoint_dir, latest_dir)


def load_model_and_tokenizer(model_path, base_model_path, num_labels):
    """加载模型和tokenizer"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    print(f"加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 为Llama模型设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"加载模型: {model_path}")
    
    # 检查是否有配置文件
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"⚠️ 模型目录缺少config.json，使用基础模型配置")
        # 使用基础模型配置
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=num_labels,
            trust_remote_code=True
        )
        # 加载训练好的权重
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print("✅ 使用基础模型配置加载成功")
    else:
        # 使用模型目录的配置
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            trust_remote_code=True
        )
        print("✅ 使用模型目录配置加载成功")
    
    return model, tokenizer


def evaluate_model(model, tokenizer, dataset_loader, output_dir, model_path=None):
    """评估模型性能"""
    print("开始模型评估...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 加载测试集
    try:
        raw_dataset = dataset_loader._load_raw_dataset()
        if 'test' not in raw_dataset:
            print("❌ 测试集不存在")
            return None
        
        test_dataset = raw_dataset['test']
        print(f"✅ 加载测试集: {len(test_dataset)} 样本")
    except Exception as e:
        print(f"❌ 加载测试集失败: {e}")
        return None
    
    # 处理测试集
    test_data = []
    for example in test_dataset:
        # 使用dataset_loader的add_prompt方法
        if hasattr(dataset_loader, 'add_prompt'):
            text = dataset_loader.add_prompt(example['text'])
        else:
            text = example['text']
        
        # 编码文本
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        test_data.append({
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': example['label'],
            'text': text,
            'original_text': example['text']
        })
    
    print(f"✅ 处理测试集: {len(test_data)} 样本")
    
    # 进行预测
    from tqdm import tqdm
    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for data in tqdm(test_data, desc="预测中"):
            input_ids = data['input_ids'].unsqueeze(0).to(device)
            attention_mask = data['attention_mask'].unsqueeze(0).to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 获取预测类别
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            predictions.append(pred)
            
            # 获取概率分布
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probabilities.append(probs.tolist())
            
            # 真实标签
            true_labels.append(data['labels'])
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # 生成详细结果
    detailed_results = []
    for i, (data, pred, prob, true_label) in enumerate(zip(test_data, predictions, probabilities, true_labels)):
        result = {
            'sample_id': i,
            'original_text': data['original_text'],
            'processed_text': data['text'],
            'true_label': int(true_label),
            'predicted_label': int(pred),
            'prediction_correct': int(pred == true_label),
            'probabilities': prob,
            'confidence': float(max(prob)),
            'predicted_class_probability': float(prob[pred])
        }
        detailed_results.append(result)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细预测结果
    predictions_file = os.path.join(output_dir, f"test_predictions_{timestamp}.json")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # 保存评估指标
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'support': support.tolist() if support is not None else []
    }
    
    metrics_file = os.path.join(output_dir, f"test_metrics_{timestamp}.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # 保存摘要报告
    summary_file = os.path.join(output_dir, f"test_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("DeepSpeed模型测试集评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(detailed_results)}\n\n")
        
        f.write("性能指标:\n")
        f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"  精确率: {metrics['precision']:.4f}\n")
        f.write(f"  召回率: {metrics['recall']:.4f}\n")
        f.write(f"  F1分数: {metrics['f1_score']:.4f}\n")
        
        f.write(f"\n分类报告:\n")
        f.write(f"{report}\n")
        
        f.write(f"\n混淆矩阵:\n")
        f.write(f"{cm}\n")
    
    print(f"\n✅ 评估完成！结果已保存到:")
    print(f"  - 预测结果: {predictions_file}")
    print(f"  - 评估指标: {metrics_file}")
    print(f"  - 摘要报告: {summary_file}")
    
    # 显示主要指标
    print(f"\n📊 性能摘要:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    
    return detailed_results, metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSpeed模型评估脚本")
    
    # 必需参数
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="DeepSpeed检查点目录")
    parser.add_argument("--data_path", type=str, required=True,
                       help="数据集路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录")
    
    # 可选参数
    parser.add_argument("--base_model_path", type=str, default=None,
                       help="基础模型路径（用于LoRA模型）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--model_max_length", type=int, default=512,
                       help="模型最大输入长度")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"随机种子设置为: {args.seed}")
    
    # 检测数据集类型
    dataset_name = detect_dataset_name(args.data_path)
    if dataset_name:
        num_labels = get_num_labels(dataset_name)
        print(f"检测到数据集: {dataset_name}，标签数: {num_labels}")
    else:
        num_labels = 7
        print(f"无法检测数据集类型，使用默认标签数: {num_labels}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 找到最新检查点
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    if not latest_checkpoint:
        print("❌ 找不到检查点目录")
        return 1
    
    # 检查模型类型并选择合适的加载方式
    pytorch_model_file = os.path.join(args.checkpoint_dir, "pytorch_model.bin")
    
    if os.path.exists(pytorch_model_file):
        print("✅ 检测到标准PyTorch模型文件，直接使用")
        merged_model_dir = args.checkpoint_dir
    else:
        print("🔄 检测到DeepSpeed检查点，需要合并权重")
        merged_model_dir = os.path.join(args.output_dir, "merged_model")
        
        # 在合并之前，复制latest文件到检查点子目录
        latest_file = os.path.join(args.checkpoint_dir, "latest")
        if os.path.exists(latest_file):
            import shutil
            latest_dst = os.path.join(latest_checkpoint, "latest")
            shutil.copy2(latest_file, latest_dst)
            print(f"✅ 复制latest文件到检查点目录: {latest_dst}")
        
        if not merge_deepspeed_checkpoint(latest_checkpoint, merged_model_dir):
            print("❌ 合并检查点失败")
            return 1
        
        # 复制配置文件到合并后的模型目录
        import shutil
        config_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']
        for file in config_files:
            src_file = os.path.join(args.checkpoint_dir, file)
            dst_file = os.path.join(merged_model_dir, file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
                print(f"✅ 复制配置文件: {file}")
    
    # 创建训练参数（用于dataset_loader）
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_pin_memory=False,
        seed=args.seed
    )
    
    # 创建数据参数
    from src.config.training_args import DataArguments
    data_args = DataArguments(
        data_path=args.data_path
    )
    
    # 创建dataset_loader
    dataset_loader = DatasetLoader(data_args, training_args, dataset_name)
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(merged_model_dir, args.base_model_path, num_labels)
    
    # 评估模型
    results = evaluate_model(model, tokenizer, dataset_loader, args.output_dir, merged_model_dir)
    
    if results:
        print("\n🎉 评估成功完成！")
        return 0
    else:
        print("\n❌ 评估失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
