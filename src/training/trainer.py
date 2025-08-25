#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
"""

import os
import transformers


class BestModelTrainer(transformers.Trainer):
    """自定义Trainer类，用于保存最优模型"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_models = []
        self.best_scores = []
        self.max_models_to_keep = 2
    
    def on_evaluate(self, *args, **kwargs):
        """评估完成后的回调，用于保存最优模型"""
        super().on_evaluate(*args, **kwargs)
        
        # 检查评估指标
        if hasattr(self, 'state') and hasattr(self.state, 'eval_metrics'):
            eval_metrics = self.state.eval_metrics
            if eval_metrics and 'eval_accuracy' in eval_metrics:
                self._save_best_model(eval_metrics['eval_accuracy'])
    
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
