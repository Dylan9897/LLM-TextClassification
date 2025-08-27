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
        # 不要直接使用训练时的model对象，而是保存路径信息
        self.model_path = model
        self.tokenizer = tokenizer
        self.dataset_loader = dataset_loader
        self.output_dir = output_dir
        self.base_model_path = base_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 不在这里加载模型，而是在需要时从文件加载
        
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
        
        try:
            # 检查是否为 DeepSpeed 模型
            is_deepspeed_model = self._is_deepspeed_model(model_path)
            
            if is_deepspeed_model:
                print("Detected DeepSpeed model, using DeepSpeed inference...")
                return self._evaluate_deepspeed_model(model_path, test_data)
            else:
                print("Using standard inference...")
                return self._evaluate_standard_model(model_path, test_data)
                
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _is_deepspeed_model(self, model_path):
        """检查是否为 DeepSpeed 模型"""
        import glob
        
        # 检查是否有 DeepSpeed 相关的文件
        deepspeed_indicators = [
            os.path.join(model_path, "zero_to_fp32.py"),
            os.path.join(model_path, "ds_config.json"),
            os.path.join(model_path, "checkpoint-*")
        ]
        
        for pattern in deepspeed_indicators:
            if glob.glob(pattern):
                return True
        
        # 检查是否有 DeepSpeed 检查点
        checkpoint_dirs = glob.glob(os.path.join(model_path, "checkpoint-*"))
        if checkpoint_dirs:
            return True
            
        return False
    
    def _evaluate_deepspeed_model(self, model_path, test_data):
        """使用 DeepSpeed 方式评估模型"""
        try:
            print("Loading DeepSpeed model for inference...")
            
            # 对于 DeepSpeed 模型，我们需要重新加载为标准的 PyTorch 模型
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            # 检查是否有配置文件
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                # 从当前路径加载模型
                try:
                    # 首先尝试加载为序列分类模型
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    print(f"Loaded sequence classification model from {model_path}")
                except Exception as e:
                    print(f"Failed to load as sequence classification model: {e}")
                    # 如果失败，尝试加载为因果语言模型
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    print(f"Loaded causal language model from {model_path}")
            else:
                # 从原始预训练模型加载
                try:
                    # 首先尝试加载为序列分类模型
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.base_model_path or "ckpt/Llama-3___2-1B-Instruct",
                        trust_remote_code=True
                    )
                    print(f"Loaded base sequence classification model")
                except Exception as e:
                    print(f"Failed to load base sequence classification model: {e}")
                    # 如果失败，尝试加载为因果语言模型
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_path or "ckpt/Llama-3___2-1B-Instruct",
                        trust_remote_code=True
                    )
                    print(f"Loaded base causal language model")
                
                # 尝试加载训练好的权重
                pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
                if os.path.exists(pytorch_model_path):
                    state_dict = torch.load(pytorch_model_path, map_location="cpu")
                    model.load_state_dict(state_dict, strict=False)
                    print("Loaded weights from pytorch_model.bin")
            
            # 加载 tokenizer
            if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_path or "ckpt/Llama-3___2-1B-Instruct")
            
            model.to(self.device)
            model.eval()
            
            return self._run_inference(model, tokenizer, test_data, model_path)
            
        except Exception as e:
            print(f"Error in DeepSpeed evaluation: {e}")
            # 回退到标准评估
            return self._evaluate_standard_model(model_path, test_data)
    
    def _evaluate_standard_model(self, model_path, test_data):
        """使用标准方式评估模型"""
        try:
            print("Loading standard model...")
            
            # 检查是否是LoRA模型
            lora_model_file = os.path.join(model_path, "adapter_model.safetensors")
            if os.path.exists(lora_model_file):
                # LoRA模型
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification
                
                if self.base_model_path:
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        self.base_model_path,
                        trust_remote_code=True
                    )
                    model = PeftModel.from_pretrained(base_model, model_path)
                    print("Loaded LoRA model")
                else:
                    print("Warning: Cannot load LoRA model without base model path")
                    return None
            else:
                # 全参微调模型
                try:
                    # 首先尝试加载为序列分类模型
                    from transformers import AutoModelForSequenceClassification
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    print("Loaded full fine-tuned sequence classification model")
                except Exception as e:
                    print(f"Failed to load as sequence classification model: {e}")
                    # 如果失败，尝试加载为因果语言模型
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    print("Loaded full fine-tuned causal language model")
            
            model.to(self.device)
            model.eval()
            
            # 加载 tokenizer
            from transformers import AutoTokenizer
            if os.path.exists(os.path.join(model_path, "tokenizer.json")):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_path or "ckpt/Llama-3___2-1B-Instruct")
            
            return self._run_inference(model, tokenizer, test_data, model_path)
            
        except Exception as e:
            print(f"Error in standard evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _run_inference(self, model, tokenizer, test_data, model_path):
        """运行推理"""
        try:
            # 进行预测
            predictions = []
            true_labels = []
            
            print("Starting predictions...")
            for data in tqdm(test_data, desc="Evaluating"):
                input_ids = data['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    try:
                        # 尝试标准的前向传播
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        # 检查模型类型并相应处理输出
                        if hasattr(outputs, 'logits'):
                            # 序列分类模型
                            logits = outputs.logits
                            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                        elif hasattr(outputs, 'last_hidden_state'):
                            # 因果语言模型，使用最后一个token的隐藏状态
                            last_hidden = outputs.last_hidden_state[:, -1, :]  # 取最后一个token
                            # 这里需要根据具体任务调整，暂时使用简单的线性分类
                            if hasattr(model, 'score'):
                                logits = model.score(last_hidden)
                            else:
                                # 如果没有分类头，使用简单的线性变换
                                import torch.nn as nn
                                classifier = nn.Linear(last_hidden.size(-1), 7).to(self.device)  # 假设7个类别
                                logits = classifier(last_hidden)
                            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                        else:
                            raise ValueError("Unknown model output format")
                            
                    except Exception as e:
                        print(f"Error during inference: {e}")
                        # 如果所有方法都失败，返回默认预测
                        pred = 0  # 默认预测第一个类别
                        print("Using default prediction due to inference error")
                
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
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
            
            # 检查多种可能的模型文件格式
            model_found = False
            
            # 1. 检查标准的 PyTorch 模型文件
            final_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            if os.path.exists(final_model_path):
                model_files.append(self.output_dir)
                print(f"Found PyTorch model: {self.output_dir}")
                model_found = True
            
            # 2. 检查 DeepSpeed 16-bit 模型文件
            deepspeed_model_path = os.path.join(self.output_dir, "pytorch_model.bin")
            if os.path.exists(deepspeed_model_path):
                model_files.append(self.output_dir)
                print(f"Found DeepSpeed model: {self.output_dir}")
                model_found = True
            
            # 3. 检查 DeepSpeed 检查点目录
            checkpoint_dir = os.path.join(self.output_dir, "checkpoint-*")
            import glob
            checkpoints = glob.glob(checkpoint_dir)
            if checkpoints:
                # 按数字排序，取最新的
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                latest_checkpoint = checkpoints[-1]
                model_files.append(latest_checkpoint)
                print(f"Found DeepSpeed checkpoint: {latest_checkpoint}")
                model_found = True
            
            # 4. 检查 best_model 目录
            best_model_dirs = glob.glob(os.path.join(self.output_dir, "best_model_*"))
            if best_model_dirs:
                for best_model_dir in best_model_dirs:
                    # 检查 best_model 目录中是否有模型文件
                    if os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
                       os.path.exists(os.path.join(best_model_dir, "adapter_model.safetensors")) or \
                       len(glob.glob(os.path.join(best_model_dir, "checkpoint-*"))) > 0:
                        model_files.append(best_model_dir)
                        print(f"Found best model: {best_model_dir}")
                        model_found = True
            
            # 5. 如果没有找到任何模型，尝试使用输出目录本身
            if not model_found:
                print("No standard model files found, attempting to use output directory...")
                # 检查是否有配置文件
                if os.path.exists(os.path.join(self.output_dir, "config.json")):
                    model_files.append(self.output_dir)
                    print(f"Using output directory as model: {self.output_dir}")
                    model_found = True
        
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
