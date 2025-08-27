#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
"""

import os
import transformers
from .base_trainer import BaseBestModelTrainer


class BestModelTrainer(transformers.Trainer, BaseBestModelTrainer):
    """自定义Trainer类，用于保存最优模型，兼容所有训练模式"""
    
    def __init__(self, *args, **kwargs):
        transformers.Trainer.__init__(self, *args, **kwargs)
        BaseBestModelTrainer.__init__(self, *args, **kwargs)
    
    def on_evaluate(self, *args, **kwargs):
        """评估完成后的回调，用于保存最优模型"""
        super().on_evaluate(*args, **kwargs)
        BaseBestModelTrainer.on_evaluate(self, *args, **kwargs)
