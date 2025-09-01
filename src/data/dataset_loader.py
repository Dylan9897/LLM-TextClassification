#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集加载和预处理模块
"""

import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset


class DatasetLoader:
    """数据集加载和预处理类"""
    
    def __init__(self, data_args, training_args, dataset_name, tokenizer=None):
        self.data_args = data_args
        self.training_args = training_args
        self.dataset_name = dataset_name
        self._tokenizer = tokenizer
        
        # 初始化标签映射
        self.label_names = self._get_label_names()
    
    def _get_label_names(self):
        """根据数据集类型获取标签映射"""
        if self.dataset_name == "ag_news":
            return {
                0: "World",
                1: "Sports", 
                2: "Business",
                3: "Sci/Tech"
            }
        elif self.dataset_name == "imdb":
            return {
                0: "Negative",
                1: "Positive"
            }
        elif self.dataset_name == "sst2":
            return {
                0: "Negative",
                1: "Positive"
            }
        elif self.dataset_name == "r8":
            return {
                0: "acq",  # 收购
                1: "corn",  # 玉米
                2: "crude",  # 原油
                3: "earn",  # 收益
                4: "grain",  # 谷物
                5: "interest",  # 利率
                6: "money-fx",  # 外汇
                7: "ship"  # 船舶
            }
        else:
            # 默认标签映射
            return {i: f"Class_{i}" for i in range(10)}
    
    def add_prompt(self, text: str) -> str:
        """为文本添加提示词"""
        class_labels = json.dumps(list(self.label_names.values()), ensure_ascii=False)
        prompt = """
            Please carefully read the following reference text:"{text}", and determine which of the following categories it belongs to:{class_labels},answer:
        """
        return prompt.format(class_labels=class_labels, text=text)
    
    def load_and_split_data(self):
        """加载数据集并根据需要切分验证集"""
        # 加载原始数据集
        dataset = self._load_raw_dataset()
     

        # 显示原始数据集信息
        print(f"Original dataset splits:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data):,} examples")
        
        # 应用样本数量限制（如果设置了的话）
        max_samples = 0
        if hasattr(self.data_args, 'max_samples') and self.data_args.max_samples > 0:
            max_samples = self.data_args.max_samples
        elif hasattr(self.data_args, 'max_example') and self.data_args.max_example > 0:
            max_samples = self.data_args.max_example
        
        if max_samples > 0:
            print(f"Limiting dataset to {max_samples} examples per split for testing...")
            limited_dataset = {}
            for split_name, split_data in dataset.items():
                if len(split_data) > max_samples:
                    limited_dataset[split_name] = split_data.select(range(max_samples))
                else:
                    limited_dataset[split_name] = split_data
            dataset = limited_dataset
            
            # 显示限制后的数据集信息
            print(f"Limited dataset splits:")
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data):,} examples")
        else:
            print("Using full dataset for training (no sample limit)")
        
        # 处理验证集和测试集
        if "train" in dataset:
            if "valid" in dataset or "dev" in dataset:
                # 已经有验证集，不需要切分
                if "dev" in dataset:
                    dataset["valid"] = dataset.pop("dev")
                print("Using existing validation set")
            elif "test" in dataset:
                # 有测试集，从训练集中切分验证集
                print("Found test set, splitting training set to create validation set...")
                dataset = self._split_training_data(dataset)
                # 显示切分后的数据集信息
                print(f"After splitting dataset:")
                for split_name, split_data in dataset.items():
                    print(f"  {split_name}: {len(split_data):,} examples")
            else:
                # 没有验证集也没有测试集，从训练集中切分
                print("No validation set found, splitting training set...")
                dataset = self._split_training_data(dataset)
                # 显示切分后的数据集信息
                print(f"After splitting dataset:")
                for split_name, split_data in dataset.items():
                    print(f"  {split_name}: {len(split_data):,} examples")
        
        # 预处理数据
        processed_data = self._preprocess_dataset(dataset)
        
        # 显示最终处理后的数据集信息
        print(f"Final processed dataset splits:")
        for split_name, split_data in processed_data.items():
            print(f"  {split_name}: {len(split_data):,} examples")
        
        # 展示前5个训练样本
        if "train" in processed_data:
            self._show_sample_examples(processed_data["train"], 5)
        
        return processed_data
    
    def _load_raw_dataset(self):
        """加载原始数据集"""
        data_path = self.data_args.data_path
        
        # 检查数据格式并加载
        if os.path.exists(os.path.join(data_path, "train.json")) and os.path.exists(os.path.join(data_path, "dev.json")):
            print("Loading JSON format dataset...")
            return self._load_json_dataset(data_path)
        elif os.path.exists(os.path.join(data_path, "train-00000-of-00001.parquet")):
            print("Loading Parquet format dataset...")
            return self._load_parquet_dataset(data_path)
        else:
            return self._auto_detect_dataset(data_path)
    
    def _load_json_dataset(self, data_path):
        """加载JSON格式数据集"""
        from datasets import load_dataset
        return load_dataset("json", data_files={
            "train": os.path.join(data_path, "train.json"),
            "valid": os.path.join(data_path, "dev.json")
        })
    
    def _load_parquet_dataset(self, data_path):
        """加载Parquet格式数据集"""
        from datasets import load_dataset
        return load_dataset("parquet", data_files={
            "train": os.path.join(data_path, "train-00000-of-00001.parquet"),
            "test": os.path.join(data_path, "test-00000-of-00001.parquet")
        })
    
    def _auto_detect_dataset(self, data_path):
        """自动检测数据集格式"""
        from datasets import load_dataset
        print("Auto-detecting dataset format...")
        
        try:
            dataset = load_dataset("parquet", data_dir=data_path)
            print(f"Successfully loaded parquet dataset with splits: {list(dataset.keys())}")
            return dataset
        except Exception as e:
            print(f"Failed to load as parquet: {e}")
            try:
                dataset = load_dataset("json", data_dir=data_path)
                print(f"Successfully loaded json dataset with splits: {list(dataset.keys())}")
                return dataset
            except Exception as e2:
                print(f"Failed to load as json: {e2}")
                raise ValueError(f"Could not load dataset from {data_path}")
    
    def _split_training_data(self, dataset):
        """使用sklearn切分训练集"""
        print("No validation set found. Splitting training set with sklearn train_test_split...")
        
        # 获取数据并转换为Python列表
        texts = dataset["train"]["text"] if "text" in dataset["train"].features else dataset["train"]["content"]
        labels = dataset["train"]["label"]
        texts_list = list(texts)
        labels_list = list(labels)
        
        # 使用sklearn切分
        train_texts, valid_texts, train_labels, valid_labels = train_test_split(
            texts_list, labels_list, 
            test_size=0.15, 
            random_state=self.training_args.seed,
            stratify=labels_list if len(set(labels_list)) <= 10 else None
        )
        
        # 创建新的数据集
        train_dataset = Dataset.from_dict({
            "text": train_texts if "text" in dataset["train"].features else train_texts,
            "label": train_labels
        })
        valid_dataset = Dataset.from_dict({
            "text": valid_texts if "text" in dataset["train"].features else valid_texts,
            "label": valid_labels
        })
        
        print(f"Training set size: {len(train_dataset)}")
        print(f"Validation set size: {len(valid_dataset)}")
        print(f"Test set size: {len(dataset["test"])}")
        
        return {"train": train_dataset, "valid": valid_dataset,"test": dataset["test"]}
    
    def _preprocess_dataset(self, dataset):
        """预处理数据集"""
        def process_function(examples):
            # 根据数据集类型使用正确的列名
            if self.dataset_name:
                from src.config.dataset_config import get_text_column, get_label_column
                text_col = get_text_column(self.dataset_name)
                label_col = get_label_column(self.dataset_name)
            else:
                text_col = "content"
                label_col = "label"
            
            # 为文本添加提示词
            prompted_texts = [self.add_prompt(text) for text in examples[text_col]]
            
            # 处理带提示词的文本
            tokenized = self._tokenize_text(prompted_texts)
            
            # 添加标签
            tokenized["labels"] = [int(unit) for unit in examples[label_col]]
            
            return tokenized
        
        # 处理每个split
        processed_dataset = {}
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name} split...")
            
            # 应用预处理
            processed_split = split_data.map(process_function, batched=True, batch_size=16)
            
            # 移除不需要的列
            if self.dataset_name:
                from src.config.dataset_config import get_text_column, get_label_column
                text_col = get_text_column(self.dataset_name)
                label_col = get_label_column(self.dataset_name)
                columns_to_remove = [text_col, label_col]
            else:
                columns_to_remove = ["content", "metadata"]
            
            processed_dataset[split_name] = processed_split.remove_columns(columns_to_remove)
        
        return processed_dataset
    
    def _tokenize_text(self, texts):
        """文本tokenization（需要在外部设置tokenizer）"""
        # 这个方法需要在外部设置tokenizer后调用
        pass
    
    def _show_sample_examples(self, dataset, num_samples=5):
        """展示数据集中的前几个样本"""
        print(f"\n{'='*60}")
        print(f"Sample Examples (showing first {num_samples} training samples):")
        print(f"{'='*60}")
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i+1}:")
            print(f"  Input IDs length: {len(sample['input_ids'])}")
            print(f"  Attention Mask length: {len(sample['attention_mask'])}")
            print(f"  Label: {sample['labels']} ({self.label_names.get(sample['labels'], 'Unknown')})")
            
            # 如果有tokenizer，尝试解码显示原始文本
            if hasattr(self, '_tokenizer') and self._tokenizer is not None:
                try:
                    # 解码input_ids（去除padding）
                    input_ids = [id for id in sample['input_ids'] if id != self._tokenizer.pad_token_id]
                    decoded_text = self._tokenizer.decode(input_ids, skip_special_tokens=True)
                    print(f"  Decoded Text:")
                    print(f"    {decoded_text}")
                    
                    # 显示文本长度信息
                    print(f"  Text Length: {len(decoded_text)} characters")
                    
                    # 如果文本太长，显示截断信息
                    if len(decoded_text) > 500:
                        print(f"  Note: Text is long, showing full content above")
                    
                except Exception as e:
                    print(f"  Decoded Text: [Error decoding: {e}]")
            else:
                print(f"  Input IDs: {sample['input_ids'][:10]}{'...' if len(sample['input_ids']) > 10 else ''}")
        
        print(f"\n{'='*60}")
        print(f"Dataset Info:")
        print(f"  Total training samples: {len(dataset)}")
        print(f"  Label distribution: {self._get_label_distribution(dataset)}")
        print(f"{'='*60}")
    
    def _get_label_distribution(self, dataset):
        """获取数据集的标签分布"""
        from collections import Counter
        labels = [sample['labels'] for sample in dataset]
        label_counts = Counter(labels)
        
        # 转换为可读的格式
        distribution = {}
        for label, count in label_counts.items():
            label_name = self.label_names.get(label, f'Class_{label}')
            distribution[label_name] = count
        
        return distribution
