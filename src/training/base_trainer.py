#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础训练器模块
提供统一的最优模型保存逻辑，兼容所有训练模式
"""

import os
import transformers
from typing import Optional


class BaseBestModelTrainer:
    """基础最优模型保存训练器，兼容所有训练模式"""
    
    def __init__(self, *args, **kwargs):
        self.best_models = []
        self.best_scores = []
        self.max_models_to_keep = 2
    
    def on_evaluate(self, *args, **kwargs):
        """评估完成后的回调，用于保存最优模型"""
        # 检查评估指标
        if hasattr(self, 'state') and hasattr(self.state, 'eval_metrics'):
            eval_metrics = self.state.eval_metrics
            print(f"评估指标: {eval_metrics}")
            
            # 尝试多种指标名称
            score = None
            if eval_metrics and 'eval_accuracy' in eval_metrics:
                score = eval_metrics['eval_accuracy']
                print(f"使用准确率指标: {score}")
            elif eval_metrics and 'eval_loss' in eval_metrics:
                # 对于损失，越小越好，所以取负值
                score = -eval_metrics['eval_loss']
                print(f"使用损失指标: {eval_metrics['eval_loss']} (转换为分数: {score})")
            elif eval_metrics and 'eval_f1' in eval_metrics:
                score = eval_metrics['eval_f1']
                print(f"使用F1指标: {score}")
            
            if score is not None:
                self._save_best_model(score)
            else:
                print("未找到合适的评估指标进行最优模型保存")
    
    def _save_best_model(self, current_score):
        """保存最优模型"""
        if len(self.best_scores) < self.max_models_to_keep or current_score > min(self.best_scores):
            # 保存模型
            model_save_path = os.path.join(self.args.output_dir, f"best_model_{len(self.best_models)}")
            self.save_model(model_save_path)
            
            # 更新最优模型列表
            if len(self.best_scores) >= self.max_models_to_keep:
                worst_idx = self.best_scores.index(min(self.best_scores))
                self.best_models.pop(worst_idx)
                self.best_scores.pop(worst_idx)
            
            self.best_models.append(model_save_path)
            self.best_scores.append(current_score)
            
            print(f"Saved best model {len(self.best_models)} with accuracy: {current_score:.4f}")
    
    def save_final_model(self, output_dir: Optional[str] = None):
        """保存最终模型，兼容三种训练模式"""
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        # 检查是否为 LoRA 模式
        if hasattr(self.args, 'use_lora') and self.args.use_lora:
            # LoRA 模式：保存 LoRA 权重
            print("Saving final model in LoRA format...")
            self.save_model(output_dir, _internal_call=False)
            print(f"LoRA model saved to {output_dir}")
        else:
            # 标准模式：直接保存
            print("Saving final model in standard format...")
            self.save_model(output_dir, _internal_call=False)
