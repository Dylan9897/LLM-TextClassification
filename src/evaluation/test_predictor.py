#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试集预测模块
用于生成详细的预测结果和指标，并将结果保存为JSON文件
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd


class TestPredictor:
    """测试集预测器"""
    
    def __init__(self, model_path, tokenizer_path, dataset_loader, output_dir, base_model_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dataset_loader = dataset_loader
        self.output_dir = output_dir
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        print(f"Loading model from: {self.model_path}")
        print(f"Loading tokenizer from: {self.tokenizer_path}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True
        )
        
        # 为Llama模型设置padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        if os.path.exists(self.model_path):
            # 检查是否是LoRA模型目录
            lora_model_file = os.path.join(self.model_path, "adapter_model.safetensors")
            if os.path.exists(lora_model_file):
                # LoRA模型
                from peft import PeftModel
                if self.base_model_path:
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        self.base_model_path, 
                        trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(base_model, self.model_path)
                else:
                    # 尝试从模型配置获取基础模型路径
                    config_path = os.path.join(self.model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        base_path = config.get('base_model_name_or_path', None)
                        if base_path:
                            base_model = AutoModelForSequenceClassification.from_pretrained(
                                base_path, 
                                trust_remote_code=True
                            )
                            model = PeftModel.from_pretrained(base_model, self.model_path)
                        else:
                            raise ValueError("Cannot find base model path for LoRA model")
                    else:
                        raise ValueError("Cannot find config.json for LoRA model")
            else:
                # 全参微调模型
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        return model, tokenizer
    
    def _load_test_set(self):
        """加载测试集"""
        try:
            # 从dataset_loader获取测试集
            if hasattr(self.dataset_loader, '_load_raw_dataset'):
                raw_dataset = self.dataset_loader._load_raw_dataset()
                if 'test' in raw_dataset:
                    return raw_dataset['test']
            
            # 如果dataset_loader没有测试集，则重新加载
            raw_dataset = self.dataset_loader._load_raw_dataset()
            if 'test' in raw_dataset:
                return raw_dataset['test']
            else:
                raise ValueError("No test set found in dataset")
        except Exception as e:
            print(f"Error loading test set: {e}")
            return None
    
    def _process_test_set(self, test_dataset):
        """处理测试集数据"""
        processed_data = []
        
        for example in test_dataset:
            # 使用dataset_loader的add_prompt方法
            if hasattr(self.dataset_loader, 'add_prompt'):
                text = self.dataset_loader.add_prompt(example['text'])
            else:
                text = example['text']
            
            # 编码文本
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            processed_data.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': example['label'],
                'text': text,
                'original_text': example['text']
            })
        
        return processed_data
    
    def predict(self):
        """执行预测"""
        print("Starting test set prediction...")
        
        # 加载测试集
        test_dataset = self._load_test_set()
        if test_dataset is None:
            print("Failed to load test set")
            return None
        
        print(f"Loaded test set with {len(test_dataset)} samples")
        
        # 处理测试集
        test_data = self._process_test_set(test_dataset)
        print(f"Processed {len(test_data)} test samples")
        
        # 进行预测
        predictions, probabilities, true_labels = self._predict_batch(test_data)
        
        # 计算指标
        metrics = self._calculate_metrics(true_labels, predictions, probabilities)
        
        # 生成详细结果
        detailed_results = self._generate_detailed_results(
            test_data, predictions, probabilities, true_labels, metrics
        )
        
        # 保存结果
        self._save_results(detailed_results, metrics)
        
        return detailed_results, metrics
    
    def _predict_batch(self, test_data):
        """批量预测"""
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for data in tqdm(test_data, desc="Predicting"):
                input_ids = data['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # 获取预测类别
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                predictions.append(pred)
                
                # 获取概率分布
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probabilities.append(probs.tolist())
                
                # 真实标签
                true_labels.append(data['labels'])
        
        return predictions, probabilities, true_labels
    
    def _calculate_metrics(self, true_labels, predictions, probabilities):
        """计算评估指标"""
        # 基础指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 分类报告
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # 精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        # 尝试计算AUC（多分类情况）
        try:
            if len(np.unique(true_labels)) == 2:
                # 二分类
                auc = roc_auc_score(true_labels, [prob[1] for prob in probabilities])
            else:
                # 多分类
                auc = roc_auc_score(true_labels, probabilities, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc is not None else None,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'support': support.tolist()
        }
        
        return metrics
    
    def _generate_detailed_results(self, test_data, predictions, probabilities, true_labels, metrics):
        """生成详细结果"""
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
        
        return detailed_results
    
    def _save_results(self, detailed_results, metrics):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细预测结果
        predictions_file = os.path.join(self.output_dir, f"test_predictions_{timestamp}.json")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # 保存评估指标
        metrics_file = os.path.join(self.output_dir, f"test_metrics_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 保存摘要报告
        summary_file = os.path.join(self.output_dir, f"test_summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Test Set Prediction Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Tokenizers Path: {self.tokenizer_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(detailed_results)}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
            if metrics['auc'] is not None:
                f.write(f"  AUC: {metrics['auc']:.4f}\n")
            
            f.write(f"\nClassification Report:\n")
            f.write(f"{metrics['classification_report']}\n")
            
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"{metrics['confusion_matrix']}\n")
        
        # 保存CSV格式的预测结果（便于分析）
        csv_file = os.path.join(self.output_dir, f"test_predictions_{timestamp}.csv")
        df_data = []
        for result in detailed_results:
            df_data.append({
                'sample_id': result['sample_id'],
                'original_text': result['original_text'],
                'true_label': result['true_label'],
                'predicted_label': result['predicted_label'],
                'prediction_correct': result['prediction_correct'],
                'confidence': result['confidence'],
                'predicted_class_probability': result['predicted_class_probability']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Results saved to:")
        print(f"  - Predictions: {predictions_file}")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Summary: {summary_file}")
        print(f"  - CSV: {csv_file}")
    
    def get_prediction_summary(self):
        """获取预测结果摘要"""
        if not hasattr(self, '_last_results'):
            print("No prediction results available. Please run predict() first.")
            return None
        
        results, metrics = self._last_results
        
        summary = {
            'total_samples': len(results),
            'correct_predictions': sum(1 for r in results if r['prediction_correct']),
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc': metrics['auc']
        }
        
        return summary
