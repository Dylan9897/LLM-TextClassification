#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSpeed 工具模块
提供 DeepSpeed 配置管理和初始化功能
"""

import os
import json
import torch
from typing import Dict, Any, Optional


class DeepSpeedConfig:
    """DeepSpeed 配置管理类"""
    
    def __init__(self, config_path: str = "ds_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载 DeepSpeed 配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"DeepSpeed config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置字典"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(updates)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """保存配置到文件"""
        output_path = output_path or self.config_path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
    
    def set_zero_stage(self, stage: int) -> None:
        """设置 ZeRO 优化阶段"""
        if stage not in [1, 2, 3]:
            raise ValueError("ZeRO stage must be 1, 2, or 3")
        
        self.config["zero_optimization"]["stage"] = stage
        
        # 根据阶段调整配置
        if stage == 1:
            # ZeRO-1: 仅优化器状态分片
            self.config["zero_optimization"].pop("offload_param", None)
            self.config["zero_optimization"].pop("stage3_prefetch_bucket_size", None)
            self.config["zero_optimization"].pop("stage3_param_persistence_threshold", None)
            self.config["zero_optimization"].pop("stage3_max_live_parameters", None)
            self.config["zero_optimization"].pop("stage3_max_reuse_distance", None)
            self.config["zero_optimization"].pop("stage3_gather_16bit_weights_on_model_save", None)
        
        elif stage == 2:
            # ZeRO-2: 优化器状态和梯度分片
            self.config["zero_optimization"].pop("offload_param", None)
            self.config["zero_optimization"].pop("stage3_prefetch_bucket_size", None)
            self.config["zero_optimization"].pop("stage3_param_persistence_threshold", None)
            self.config["zero_optimization"].pop("stage3_max_live_parameters", None)
            self.config["zero_optimization"].pop("stage3_max_reuse_distance", None)
            self.config["zero_optimization"].pop("stage3_gather_16bit_weights_on_model_save", None)
        
        # ZeRO-3: 保持所有配置
    
    def enable_cpu_offload(self, enable: bool = True) -> None:
        """启用/禁用 CPU 卸载"""
        if self.config["zero_optimization"]["stage"] >= 2:
            if enable:
                self.config["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True
                }
                if self.config["zero_optimization"]["stage"] == 3:
                    self.config["zero_optimization"]["offload_param"] = {
                        "device": "cpu",
                        "pin_memory": True
                    }
            else:
                self.config["zero_optimization"].pop("offload_optimizer", None)
                self.config["zero_optimization"].pop("offload_param", None)
    
    def set_mixed_precision(self, precision: str = "fp16") -> None:
        """设置混合精度类型"""
        if precision == "fp16":
            self.config["fp16"]["enabled"] = True
            self.config["bf16"]["enabled"] = False
        elif precision == "bf16":
            self.config["fp16"]["enabled"] = False
            self.config["bf16"]["enabled"] = True
        else:
            self.config["fp16"]["enabled"] = False
            self.config["bf16"]["enabled"] = False


def create_deepspeed_config(
    zero_stage: int = 3,
    enable_cpu_offload: bool = True,
    mixed_precision: str = "none",
    output_path: str = "ds_config_custom.json"
) -> str:
    """创建自定义 DeepSpeed 配置文件
    
    Args:
        zero_stage: ZeRO 优化阶段 (1, 2, 3)
        enable_cpu_offload: 是否启用 CPU 卸载
        mixed_precision: 混合精度类型 ("fp16", "bf16", "none", 默认: "none")
        output_path: 输出配置文件路径
    
    Returns:
        配置文件路径
    """
    config = DeepSpeedConfig()
    config.set_zero_stage(zero_stage)
    config.enable_cpu_offload(enable_cpu_offload)
    config.set_mixed_precision(mixed_precision)
    config.save_config(output_path)
    
    return output_path


def get_deepspeed_launcher_args(
    num_gpus: int = 1,
    config_path: str = "ds_config.json",
    script_path: str = "main.py",
    script_args: Optional[str] = None
) -> str:
    """生成 DeepSpeed 启动命令
    
    Args:
        num_gpus: GPU 数量
        config_path: DeepSpeed 配置文件路径
        script_path: 训练脚本路径
        script_args: 脚本参数
    
    Returns:
        DeepSpeed 启动命令
    """
    cmd = f"deepspeed --num_gpus {num_gpus} --config {config_path} {script_path}"
    
    if script_args:
        cmd += f" {script_args}"
    
    return cmd


def check_deepspeed_installation() -> bool:
    """检查 DeepSpeed 是否正确安装"""
    try:
        import deepspeed
        return True
    except ImportError:
        return False


def get_memory_usage() -> Dict[str, float]:
    """获取 GPU 内存使用情况"""
    if not torch.cuda.is_available():
        return {"gpu_memory": 0.0, "gpu_memory_allocated": 0.0}
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    
    return {
        "gpu_memory": gpu_memory,
        "gpu_memory_allocated": gpu_memory_allocated
    }


if __name__ == "__main__":
    # 测试配置创建
    print("Creating DeepSpeed configurations...")
    
    # 创建 ZeRO-3 配置
    config_path = create_deepspeed_config(
        zero_stage=3,
        enable_cpu_offload=True,
        mixed_precision="fp16",
        output_path="ds_config_zero3.json"
    )
    print(f"Created ZeRO-3 config: {config_path}")
    
    # 创建 ZeRO-2 配置
    config_path = create_deepspeed_config(
        zero_stage=2,
        enable_cpu_offload=False,
        mixed_precision="fp16",
        output_path="ds_config_zero2.json"
    )
    print(f"Created ZeRO-2 config: {config_path}")
    
    # 检查安装
    if check_deepspeed_installation():
        print("DeepSpeed is properly installed")
    else:
        print("DeepSpeed is not installed")
    
    # 显示内存使用
    memory_info = get_memory_usage()
    print(f"GPU Memory: {memory_info['gpu_memory']:.2f} GB")
    print(f"GPU Memory Allocated: {memory_info['gpu_memory_allocated']:.2f} GB")
