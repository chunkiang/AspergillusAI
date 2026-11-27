#!/bin/zsh
# 对每个模型最新的交叉验证目录做错误分析并画混淆矩阵
cd "${0:A:h}"
RUNS=output/02_training_runs/runs

for BB in densenet121 resnet50 efficientnet_b0 convnext_tiny swin_t; do
    # (/om[1]) = 只匹配目录、按修改时间排序、取最新一个
    dirs=(${RUNS}/${BB}_cv*(/Nom[1]))
    if (( ${#dirs} == 0 )); then
        echo "[跳过] 找不到 ${BB} 的 CV 目录"
        continue
    fi
    run=${dirs[1]}
    echo "== ${run} =="

    python 02b_analyze_errors.py "$run" || { echo "[失败] analyze_errors: $run"; continue; }

    if [[ -f "$run/confusion.csv" ]]; then
        python 02b_plot_confusion.py "$run/confusion.csv"
    else
        echo "[缺失] $run/confusion.csv"
    fi
done

