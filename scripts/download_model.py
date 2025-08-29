# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/3/2 11:49

from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',cache_dir="./ckpt")
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',cache_dir="./ckpt")