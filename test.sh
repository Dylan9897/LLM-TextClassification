#!/bin/bash
# 主训练脚本 - 依次运行所有训练脚本

echo "Starting training scripts sequentially..."
echo "This will run 12 training jobs (2 datasets × 3 seeds × 2 modes) one by one"
echo ""

# 创建日志目录
mkdir -p logs

# 计数器
count=0
total_scripts=$(ls scripts/test_*.sh | wc -l)

# 依次运行所有训练脚本
for script in scripts/test_*.sh; do
    if [ -f "$script" ]; then
        count=$((count + 1))
        echo "=========================================="
        echo "Running script $count of $total_scripts: $script"
        echo "Log: logs/$(basename $script .sh).log"
        echo "=========================================="
        
        # 前台运行，等待完成后再运行下一个
        bash "$script" > "logs/$(basename $script .sh).log" 2>&1
        
        # 检查上一个脚本的退出状态
        if [ $? -eq 0 ]; then
            echo "✓ Script $script completed successfully"
        else
            echo "✗ Script $script failed with exit code $?"
        fi
        
        echo ""
        echo "Waiting 5 seconds before starting next script..."
        sleep 5
    fi
done

echo "=========================================="
echo "All training scripts completed!"
echo "Check logs/ directory for results."
echo "=========================================="
