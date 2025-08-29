# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/3/2 11:39
# encoding : utf-8 -*-
# @author  : 冬瓜
# @mail    : dylan_han@126.com
# @Time    : 2024/4/8 23:15
import transformers
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    bert_name_or_path: Optional[str] = field(default="bert-base-chinese")  # bert模型文件地址
    is_training: bool = False  # 是否为训练模式
    update_bmlp: bool = False  # bert模型是否更新


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    remove_unused_columns:bool = False

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

