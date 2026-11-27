#!/bin/zsh
# 02_run_all.sh —— 依次跑完所有模型的 5 折交叉验证（可整晚运行，中断后再次运行会从断点继续）
# 用法：caffeinate -i zsh 02_run_all.sh
set -e
DENSE_DIR=output/02_training_runs/runs/densenet121_cv_20261009_1529      # 已完成第 1 折的 DenseNet121 目录（续跑）

python 02_train.py --backbone densenet121 --mode cv --resume_dir $DENSE_DIR

for BB in resnet50 efficientnet_b0 convnext_tiny swin_t; do
  DIR=output/02_training_runs/runs/${BB}_cv_main
  python 02_train.py --backbone $BB --mode cv --resume_dir $DIR
done

python 02b_compare_runs.py
