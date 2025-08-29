#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辅助工具模块
"""

import torch
import random
import numpy as np
import transformers


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def detect_dataset_name(data_path):
    """自动检测数据集名称"""
    if not data_path:
        return None
    
    path_parts = data_path.split('/')
    for part in path_parts:
        if part in ['sst2', 'ag_news', 'imdb', 'r8', 'longnews']:
            return part
    return None
