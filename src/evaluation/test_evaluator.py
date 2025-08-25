import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer
from tqdm import tqdm


class TestEvaluator:
    """测试集评估器"""
    
    def __init__(self, model, tokenizer, dataset_loader, output_dir, base_model_path=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_loader = dataset_loader
        self.output_dir = output_dir
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def _load_test_set(self):
        """加载测试集"""
        try:
            # 首先尝试从dataset_loader获取测试集
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
                'text': text
            })
        
        return processed_data
    
    def _evaluate_single_model(self, model_path, test_data):
        """评估单个模型"""
        print(f"Evaluating model: {model_path}")
        
        # 加载模型
        if os.path.exists(model_path):
            # 检查是否是LoRA模型目录
            lora_model_file = os.path.join(model_path, "adapter_model.safetensors")
            if os.path.exists(lora_model_file):
                # LoRA模型 - 需要从基础模型路径加载
                from peft import PeftModel
                if self.base_model_path:
                    # 先加载基础模型，然后加载LoRA adapter
                    if hasattr(self.model, 'get_base_model'):
                        # 如果当前模型是PeftModel，获取其基础模型类
                        base_model_class = self.model.get_base_model().__class__
                    else:
                        # 否则使用transformers的AutoModelForSequenceClassification
                        from transformers import AutoModelForSequenceClassification
                        base_model_class = AutoModelForSequenceClassification
                    
                    # 使用正确的num_labels加载基础模型
                    num_labels = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else 8
                    base_model = base_model_class.from_pretrained(self.base_model_path, num_labels=num_labels)
                    model = PeftModel.from_pretrained(base_model, model_path)
                else:
                    # 尝试从模型配置获取基础模型路径
                    base_path = getattr(self.model.config, '_name_or_path', None)
                    if base_path:
                        if hasattr(self.model, 'get_base_model'):
                            base_model_class = self.model.get_base_model().__class__
                        else:
                            from transformers import AutoModelForSequenceClassification
                            base_model_class = AutoModelForSequenceClassification
                        
                        # 使用正确的num_labels加载基础模型
                        num_labels = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else 8
                        base_model = base_model_class.from_pretrained(base_path, num_labels=num_labels)
                        model = PeftModel.from_pretrained(base_model, model_path)
                    else:
                        print(f"Warning: Cannot load LoRA model without base model path")
                        return None
            else:
                # 全参微调模型
                model = self.model.__class__.from_pretrained(model_path)
            
            model.to(self.device)
            model.eval()
        else:
            print(f"Model not found: {model_path}")
            return None
        
        # 进行预测
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data in tqdm(test_data, desc="Evaluating"):
                input_ids = data['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                
                predictions.append(pred)
                true_labels.append(data['labels'])
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # 转换为列表以确保JSON序列化
        true_labels = [int(label) for label in true_labels]
        predictions = [int(pred) for pred in predictions]
        
        results = {
            'model_path': model_path,
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return results
    
    def _evaluate_best_models(self, test_data):
        """评估最佳模型"""
        print("Evaluating best models...")
        
        # 查找模型文件
        model_files = []
        
        # 检查LoRA模型
        lora_model_path = os.path.join(self.output_dir, "adapter_model.safetensors")
        if os.path.exists(lora_model_path):
            print("=== LoRA Mode ===")
            model_files.append(self.output_dir)
            print(f"Found LoRA model: {self.output_dir}")
            print("LoRA mode: Only evaluating final LoRA model")
        else:
            # 检查全参微调模型
            print("=== Full Fine-tuning Mode ===")
            final_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            if os.path.exists(final_model_path):
                model_files.append(self.output_dir)
                print(f"Found final model: {self.output_dir}")
            
            # 检查checkpoint目录
            checkpoint_dir = os.path.join(self.output_dir, "checkpoint-*")
            import glob
            checkpoints = glob.glob(checkpoint_dir)
            if checkpoints:
                # 按数字排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                latest_checkpoint = checkpoints[-1]
                model_files.append(latest_checkpoint)
                print(f"Found checkpoint: {latest_checkpoint}")
        
        if not model_files:
            print("No model files found for evaluation")
            return
        
        print(f"\nTotal models to evaluate: {len(model_files)}")
        for i, model_path in enumerate(model_files, 1):
            print(f"  {i}. {model_path}")
        
        # 评估找到的模型
        all_results = []
        for model_path in model_files:
            results = self._evaluate_single_model(model_path, test_data)
            if results:
                all_results.append(results)
        
        # 保存结果
        self._save_results(all_results)
    
    def _save_results(self, results):
        """保存评估结果"""
        # 保存详细结果到JSON文件
        results_file = os.path.join(self.output_dir, "test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存摘要到文本文件
        summary_file = os.path.join(self.output_dir, "test_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Test Set Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Model: {result['model_path']}\n")
                f.write(f"Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"Classification Report:\n")
                f.write(f"{result['classification_report']}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Results saved to {results_file}")
        print(f"Summary saved to {summary_file}")
    
    def evaluate(self):
        """执行评估"""
        print("Starting test set evaluation...")
        
        # 加载测试集
        test_dataset = self._load_test_set()
        if test_dataset is None:
            print("Failed to load test set")
            return
        
        print(f"Loaded test set with {len(test_dataset)} samples")
        
        # 处理测试集
        test_data = self._process_test_set(test_dataset)
        print(f"Processed {len(test_data)} test samples")
        
        # 评估模型
        self._evaluate_best_models(test_data)
        
        print("Test set evaluation completed")
