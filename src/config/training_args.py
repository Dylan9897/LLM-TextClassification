#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练参数定义
包含模型、数据、训练和LoRA相关的参数配置
"""

import transformers
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen-7B",
        metadata={"help": "预训练模型的路径或名称"}
    )
    bert_name_or_path: Optional[str] = field(
        default="bert-base-chinese",
        metadata={"help": "BERT模型文件地址"}
    )
    is_training: bool = field(
        default=False,
        metadata={"help": "是否为训练模式"}
    )
    update_bmlp: bool = field(
        default=False,
        metadata={"help": "BERT模型是否更新"}
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径"}
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "评估数据路径"}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "是否使用延迟预处理"}
    )
    max_example: int = field(
        default=0,
        metadata={"help": "训练使用的最大样本数（0表示使用完整数据集）"}
    )
    max_samples: int = field(
        default=0,
        metadata={"help": "训练使用的最大样本数（0表示使用完整数据集），与max_example功能相同"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """训练相关参数，继承自Transformers的TrainingArguments"""
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "缓存目录"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "优化器类型"}
    )
    model_max_length: int = field(
        default=8192,
        metadata={"help": "最大序列长度，序列将被右填充（可能被截断）"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA微调"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "是否移除未使用的列"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子，用于可重现性"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "DeepSpeed配置文件路径"}
    )


@dataclass
class LoraArguments:
    """LoRA微调相关参数"""
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA的秩（rank）"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA的缩放参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA的dropout率"}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "LoRA目标模块列表"}
    )
    lora_weight_path: str = field(
        default="",
        metadata={"help": "LoRA权重路径"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA偏置处理方式"}
    )
    q_lora: bool = field(
        default=False,
        metadata={"help": "是否使用QLoRA"}
    )
