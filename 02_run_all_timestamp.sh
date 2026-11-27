#!/bin/zsh
# 02_run_all.sh —— 依次跑完所有模型的 5 折交叉验证（可整晚运行，中断后再次运行会从断点继续）
# 输出目录统一为 output/02_training_runs/runs/<模型>_cv_<YYYYMMDD_HHMM>
# 用法：caffeinate -i zsh 02_run_all.sh
set -e
cd "${0:A:h}"
RUNS=output/02_training_runs/runs
mkdir -p $RUNS

# 为某个模型找到（或新建）输出目录：
#   1) 已有 <模型>_cv_<时间戳> 目录 → 取最新的一个续跑
#   2) 只有旧式 <模型>_cv_main 目录 → 按其修改时间重命名为时间戳格式后续跑
#   3) 都没有 → 用当前时间新建
get_run_dir() {
  local bb=$1
  local existing=( $RUNS/${bb}_cv_[0-9](#c8)_[0-9](#c4)(N/om) )
  if (( ${#existing} )); then
    print -r -- ${existing[1]}
    return
  fi
  if [[ -d $RUNS/${bb}_cv_main ]]; then
    local ts=$(date -r $RUNS/${bb}_cv_main +%Y%m%d_%H%M)
    mv $RUNS/${bb}_cv_main $RUNS/${bb}_cv_${ts}
    print -r -- $RUNS/${bb}_cv_${ts}
    return
  fi
  print -r -- $RUNS/${bb}_cv_$(date +%Y%m%d_%H%M)
}

setopt extendedglob   # 启用 (#c8) 这类重复次数匹配

# for BB in densenet121 resnet50 efficientnet_b0 convnext_tiny swin_t; do
# for BB in densenet121; do
# for BB in resnet50 efficientnet_b0 convnext_tiny swin_t; do
for BB in  efficientnet_b0  swin_t; do
  DIR=$(get_run_dir $BB)
  echo ">>> $BB -> $DIR"
  python 02_train.py --backbone $BB --mode cv --resume_dir $DIR
done

python 02b_compare_runs.py
