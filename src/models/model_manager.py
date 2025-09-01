#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理模块
"""

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForSequenceClassification,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


class CustomClassifier(nn.Module):
    """自定义分类头"""
    
    def __init__(self, input_size, out_features, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.actual_input_size = input_size
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.dropout = dropout
        
        self.fc1 = nn.Linear(self.actual_input_size, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_dim, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(256, self.out_features)
        self.softmax = nn.Softmax(dim=1)
        
        # 使用Xavier均匀初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class ModelManager:
    """模型管理类"""
    
    def __init__(self, model_args, training_args, lora_args, num_labels):
        self.model_args = model_args
        self.training_args = training_args
        self.lora_args = lora_args
        self.num_labels = num_labels
    
    def create_model(self):
        """创建和配置模型"""
        # 加载模型配置
        config = transformers.AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.training_args.cache_dir,
            trust_remote_code=True,
            is_training=self.model_args.is_training
        )
        config.use_cache = False
        # if not self.training_args.use_lora:
            # 创建模型
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path, 
            num_labels=self.num_labels
        )
        # else:
        #     model = AutoModelForCausalLM.from_pretrained(
        #         self.model_args.model_name_or_path,
        #         config=config
        #     )
        
        # 设置padding token ID，确保在词汇表范围内
        if hasattr(model.config, 'vocab_size'):
            # 使用词汇表大小减1作为padding token ID
            model.config.pad_token_id = model.config.vocab_size - 1
        else:
            # 默认值
            model.config.pad_token_id = 0
    
        # 配置训练模式
        if not self.training_args.use_lora:
            self._setup_full_finetuning(model)
        else:
            model = self._setup_lora_finetuning(model)
        
        # 应用其他配置
        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        return model
    
    def _setup_lora_finetuning(self, model,create=False):
        print("Setting up LoRA fine-tuning...")

        for param in model.parameters():
            param.requires_grad = False

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_args.lora_dropout,
            bias=self.lora_args.lora_bias,
            task_type="SEQ_CLS",
            modules_to_save=target_modules+["socre"]
        )

       
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # for name, module in model.named_modules():
        #     if "lm_head" in name:
        #         print(name, module)
        # s = input()

        if create:
            # 获取分类头的输入输出维度
            if hasattr(model, 'score') and hasattr(model.score, 'modules_to_save'):
                # 获取原始Linear层的in/out特征数
                old_linear = model.score.modules_to_save["default"]
                in_features = old_linear.in_features
                out_features = old_linear.out_features
                # 替换为自定义分类头
                model.score.original_module = CustomClassifier(in_features, out_features)
                model.score.modules_to_save["default"] = CustomClassifier(in_features, out_features)
                print("Replaced model.score.modules_to_save['default'] with CustomClassifier.")

            else:
                print("Warning: model.score.modules_to_save not found, fallback to direct replacement.")
                # 兜底方案
                hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.hidden_dim
                model.score = CustomClassifier(hidden_size, self.num_labels)
      
        # 只设置LoRA参数和分类层参数为可训练
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in ['lora_', 'score']):
                param.requires_grad = True
                print(f"Setting {name} as trainable")
            else:
                param.requires_grad = False

        print(model)
        # 打印可训练参数统计
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"LoRA setup complete:")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")

        return model
