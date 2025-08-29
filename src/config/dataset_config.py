#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集配置文件
定义不同数据集的标签数量和配置信息
"""

DATASET_CONFIGS = {
    "sst2": {
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "text_column": "text",
        "label_column": "label"
    },
    "ag_news": {
        "num_labels": 4,
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "text_column": "text",
        "label_column": "label"
    },
    "imdb": {
        "num_labels": 2,
        "label_names": ["negative", "positive"],
        "text_column": "text",
        "label_column": "label"
    },
    "r8": {
        "num_labels": 8,
        "label_names": ["acq", "crude", "earn", "grain", "interest", "money-fx", "ship", "trade"],
        "text_column": "text",
        "label_column": "label"
    },
    "longnews": {
        "num_labels": 7,
        "label_names": ["时尚", "财经", "时政", "家居", "房产", "教育", "科技"],
        "text_column": "content",
        "label_column": "label"
    }
}

def get_dataset_config(dataset_name):
    """获取数据集配置"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]

def get_num_labels(dataset_name):
    """获取数据集的标签数量"""
    config = get_dataset_config(dataset_name)
    return config["num_labels"]

def get_label_names(dataset_name):
    """获取数据集的标签名称"""
    config = get_dataset_config(dataset_name)
    return config["label_names"]

def get_text_column(dataset_name):
    """获取数据集的文本列名"""
    config = get_dataset_config(dataset_name)
    return config["text_column"]

def get_label_column(dataset_name):
    """获取数据集的标签列名"""
    config = get_dataset_config(dataset_name)
    return config["label_column"]

def list_available_datasets():
    """列出所有可用的数据集"""
    return list(DATASET_CONFIGS.keys())

if __name__ == "__main__":
    print("Available datasets:")
    for name, config in DATASET_CONFIGS.items():
        print(f"  {name}: {config['num_labels']} labels - {config['label_names']}")
