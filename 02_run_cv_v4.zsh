#!/bin/zsh
# =============================================================================
# 02_run_cv_v4.zsh —— 第 1 步：建 trainset_v4，多个 backbone 的菌株级 5 折交叉验证
#
# trainset_v4 = 本院规范截图 + 核对为合格的旧截图，不含 ✅ 文件夹截图（--exclude_sources dev_checked）
# 交叉验证：02_train.py --mode cv（StratifiedGroupKFold，按 cv_group 分组）
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh 02_run_cv_v4.zsh                       # 默认 6 个 backbone
#     zsh 02_run_cv_v4.zsh swin_t densenet121    # 只跑指定的
# 中断后重跑：已完成的 backbone（有 cv_summary.json）跳过；做了一半的从断点续跑
#
# 输出：
#   output/01_trainsets/trainset_v4/                       训练清单（已存在则不重建）
#   output/02_training_runs/runs_v4/<backbone>_cv/          每个 backbone 的 CV 结果 + 错误分析 + 混淆矩阵
#   output/02_training_runs/runs_v4/cv_汇总.txt             各 backbone 对比（论文 Table：backbone 比较）
#   output/06_logs/cv_v4_<时间>.log
# =============================================================================
set -e -o pipefail
cd "${0:A:h}"

TS=trainset_v4
OUT=output
RUNS=${OUT}/02_training_runs/runs_v4
BUILD_OPTS=(--replace_classes all_v2 --review review_images.csv --exclude_sources dev_checked)
if (( $# )); then BBS=("$@"); else BBS=(swin_t densenet121 efficientnet_b0 resnet50 convnext_tiny vit_b_16); fi

for f in paths.py 01_build_trainset.py 02_train.py 02b_analyze_errors.py 02b_plot_confusion.py 02b_oof_by_site.py; do
  [[ -e "${f}" ]] || { echo "找不到 ${f}"; exit 1; }
done
grep -q "exclude_sources" 01_build_trainset.py || { echo "01_build_trainset.py 不是 10-21 版（缺 --exclude_sources）"; exit 1; }

mkdir -p ${OUT}/06_logs ${RUNS}
LOG="${OUT}/06_logs/cv_v4_$(date +%Y%m%d_%H%M).log"
caffeinate -i -w $$ &

{
  # ---------- 1. 训练集 ----------
  if [[ -f "${OUT}/01_trainsets/${TS}/trainset_micro.csv" ]]; then
    echo "== ${TS} 已存在，不重建（要重建先把 ${OUT}/01_trainsets/${TS} 挪走）"
  else
    echo "\n==================== 建 ${TS} $(date '+%F %T') ===================="
    python 01_build_trainset.py "${BUILD_OPTS[@]}" --out "${TS}"
  fi
  python - "${OUT}/01_trainsets/${TS}/trainset_micro.csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
n = int((d["source"] == "dev_checked").sum())
print(f"核对：{sys.argv[1]} 共 {len(d)} 张，dev_checked {n} 张")
sys.exit("✗ trainset_v4 里还有 dev_checked，停止" if n else 0)
EOF

  # ---------- 2. 各 backbone 5 折 CV ----------
  for BB in "${BBS[@]}"; do
    DIR="${RUNS}/${BB}_cv"
    echo "\n==================== ${BB} 5 折 CV $(date '+%F %T') ===================="
    if [[ -f "${DIR}/cv_summary.json" ]]; then
      echo "== ${DIR}/cv_summary.json 已存在，跳过训练"
    else
      python 02_train.py --backbone "${BB}" --mode cv --trainset "${TS}" --resume_dir "${DIR}"
    fi
    # 下面三项只是训练后的分析：任何一项出错只给警告，不中断后面 backbone 的训练
    python 02b_analyze_errors.py "${DIR}" || echo "⚠️  ${BB}：02b_analyze_errors.py 出错（不影响训练结果）"
    python 02b_plot_confusion.py "${DIR}/confusion.csv" || echo "⚠️  ${BB}：02b_plot_confusion.py 出错（不影响训练结果）"
    python 02b_oof_by_site.py --run "${DIR}" --trainset "${TS}" || echo "⚠️  ${BB}：02b_oof_by_site.py 出错（不影响训练结果）"
  done

  # ---------- 3. 汇总 ----------
  echo "\n==================== backbone 汇总 ===================="
  python - "${RUNS}" <<'EOF' | tee "${RUNS}/cv_汇总.txt"
import sys, json, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_cv", "cv_summary.json"))):
    s = json.load(open(f))
    rows.append((s.get("backbone", os.path.basename(os.path.dirname(f))), s))
print(f"{'backbone':16s} {'折数':>4s} {'acc':>15s} {'macro-F1':>15s} {'balanced acc':>15s} {'top2':>15s} {'最佳epoch中位数':>14s}")
for bb, s in rows:
    cell = lambda c: f"{s[c + '_mean']:.3f} ± {s[c + '_sd']:.3f}"
    print(f"{bb:16s} {len(s.get('folds_done', [])):>4d} {cell('acc'):>15s} {cell('macro_f1'):>15s} "
          f"{cell('balanced_acc'):>15s} {cell('top2'):>15s} {s.get('median_best_epoch', float('nan')):>14.0f}")
print("\n各折均值 ± SD；菌株级划分（StratifiedGroupKFold，按 cv_group）")
EOF
} 2>&1 | tee "${LOG}"
