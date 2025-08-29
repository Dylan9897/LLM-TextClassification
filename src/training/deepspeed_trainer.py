#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持 DeepSpeed 的训练器模块
"""

import os
import transformers
from transformers import Trainer
from typing import Optional, Dict, Any
from .base_trainer import BaseBestModelTrainer


class DeepSpeedTrainer(Trainer):
    """支持 DeepSpeed 的训练器类"""
    
    def __init__(self, *args, **kwargs):
        # 移除 deepspeed_config 参数，因为 Trainer 不接受
        deepspeed_config = kwargs.pop('deepspeed_config', None)
        super().__init__(*args, **kwargs)
        self.deepspeed_config = deepspeed_config
        self._setup_deepspeed()
    
    def _setup_deepspeed(self):
        """设置 DeepSpeed 配置"""
        if self.deepspeed_config and hasattr(self.args, 'deepspeed'):
            print(f"DeepSpeed configuration loaded: {self.args.deepspeed}")
            print(f"ZeRO stage: {self._get_zero_stage()}")
    
    def _get_zero_stage(self) -> Optional[int]:
        """获取 ZeRO 优化阶段"""
        if not hasattr(self.args, 'deepspeed') or not self.args.deepspeed:
            return None
        
        try:
            import json
            with open(self.args.deepspeed, 'r') as f:
                config = json.load(f)
            return config.get('zero_optimization', {}).get('stage', None)
        except:
            return None
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, *args, **kwargs):
        """重写评估和保存逻辑以支持 DeepSpeed"""
        if self.control.should_evaluate:
            # 在评估前同步 DeepSpeed 引擎
            if hasattr(self, 'deepspeed') and self.deepspeed is not None:
                self.deepspeed.eval()
            
            # 确保 ignore_keys 是列表类型
            if ignore_keys_for_eval is None:
                ignore_keys_for_eval = []
            elif not isinstance(ignore_keys_for_eval, list):
                ignore_keys_for_eval = [ignore_keys_for_eval]
            
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self.log(metrics)
            
            # 评估后恢复训练模式
            if hasattr(self, 'deepspeed') and self.deepspeed is not None:
                self.deepspeed.train()
        
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """保存检查点，支持 DeepSpeed"""
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 使用 DeepSpeed 保存
            self.deepspeed.save_checkpoint(self.args.output_dir)
            print(f"DeepSpeed checkpoint saved to {self.args.output_dir}")
        else:
            # 使用标准方法保存
            super()._save_checkpoint(model, trial, metrics)
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型，支持 DeepSpeed"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 使用 DeepSpeed 保存模型
            self.deepspeed.save_checkpoint(output_dir)
            print(f"DeepSpeed model saved to {output_dir}")
        else:
            # 使用标准方法保存
            super().save_model(output_dir, _internal_call)
    
    def save_final_model(self, output_dir: Optional[str] = None):
        """保存最终模型，兼容三种训练模式"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # DeepSpeed 模式：保存为 PyTorch 格式
            print("Saving final model in PyTorch format for DeepSpeed...")
            try:
                # 使用 DeepSpeed 的 save_16bit_model 方法
                if hasattr(self.deepspeed, 'save_16bit_model'):
                    self.deepspeed.save_16bit_model(output_dir)
                    print(f"DeepSpeed 16-bit model saved to {output_dir}")
                else:
                    # 尝试使用标准的 save_model 方法
                    print("Attempting to save using standard save_model...")
                    # 临时将模型设置为 CPU 模式
                    device = next(self.model.parameters()).device
                    self.model.to('cpu')
                    
                    # 保存模型
                    super().save_model(output_dir, _internal_call=False)
                    
                    # 恢复设备
                    self.model.to(device)
                    print(f"Final PyTorch model saved to {output_dir}")
                    
            except Exception as e:
                print(f"Warning: Failed to save PyTorch format: {e}")
                print("Falling back to DeepSpeed checkpoint format...")
                # 回退到 DeepSpeed 检查点格式
                self.deepspeed.save_checkpoint(output_dir)
                print(f"DeepSpeed checkpoint saved to {output_dir}")
        else:
            # 标准训练模式：直接保存
            print("Saving final model in standard format...")
            super().save_model(output_dir, _internal_call=False)
    
    def get_train_dataloader(self):
        """获取训练数据加载器"""
        if self.train_dataset is None:
            return None
        
        # 检查是否使用 DeepSpeed
        if hasattr(self.args, 'deepspeed') and self.args.deepspeed:
            # DeepSpeed 会自动处理数据并行
            return super().get_train_dataloader()
        else:
            return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        """获取评估数据加载器"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        if eval_dataset is None:
            return None
        
        # 检查是否使用 DeepSpeed
        if hasattr(self.args, 'deepspeed') and self.args.deepspeed:
            # DeepSpeed 会自动处理数据并行
            return super().get_eval_dataloader(eval_dataset)
        else:
            return super().get_eval_dataloader(eval_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失，支持 DeepSpeed"""
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 使用 DeepSpeed 计算损失
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss
        else:
            # 使用标准方法计算损失
            return super().compute_loss(model, inputs, return_outputs)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """训练步骤，支持 DeepSpeed"""
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 使用 DeepSpeed 进行训练步骤
            loss = self.deepspeed.backward(self.compute_loss(model, inputs))
            self.deepspeed.step()
            return loss.detach()
        else:
            # 使用标准方法进行训练步骤
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            else:
                return super().training_step(model, inputs)
    
    def _inner_training_loop(self, *args, **kwargs):
        """内部训练循环，支持 DeepSpeed"""
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 初始化 DeepSpeed
            self.deepspeed.init_distributed()
            print("DeepSpeed distributed training initialized")
        
        return super()._inner_training_loop(*args, **kwargs)


class BestModelDeepSpeedTrainer(DeepSpeedTrainer, BaseBestModelTrainer):
    """支持 DeepSpeed 的最优模型保存训练器，兼容所有训练模式"""
    
    def __init__(self, *args, **kwargs):
        DeepSpeedTrainer.__init__(self, *args, **kwargs)
        BaseBestModelTrainer.__init__(self, *args, **kwargs)
    
    def on_evaluate(self, *args, **kwargs):
        """评估完成后的回调，用于保存最优模型"""
        super().on_evaluate(*args, **kwargs)
        BaseBestModelTrainer.on_evaluate(self, *args, **kwargs)
    

    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型，支持 DeepSpeed 和最优模型保存"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        if hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # 使用 DeepSpeed 保存模型
            self.deepspeed.save_checkpoint(output_dir)
            print(f"DeepSpeed model saved to {output_dir}")
        else:
            # 使用标准方法保存
            super().save_model(output_dir, _internal_call)