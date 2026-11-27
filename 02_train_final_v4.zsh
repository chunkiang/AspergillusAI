#!/bin/zsh
# =============================================================================
# 02_train_final_v4.zsh —— 第 2 步：用 trainset_v4 全部数据训练最终模型
#
# epoch 数 = 该 backbone 5 折 CV 各折最佳 epoch 的中位数（读 cv_summary.json 的 median_best_epoch），
# 不看外部数据，事先确定。必须先跑完 02_run_cv_v4.zsh。
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh 02_train_final_v4.zsh                 # 默认 swin_t（与 v3 主分析一致）
#     zsh 02_train_final_v4.zsh densenet121     # 换主模型（需在跑外部验证前决定，并在论文中说明理由）
# 已有 final.pt 则跳过训练，只做自检
#
# 输出：
#   output/02_training_runs/runs_v4/<backbone>_full_final/   final.pt、train_log.csv、args.json、class_names.json
#   output/02_training_runs/runs_v4/<backbone>_full_final/model_sha256.txt   权重指纹（外部验证时核对）
#   output/06_logs/final_v4_<时间>.log
# =============================================================================
set -e -o pipefail
cd "${0:A:h}"

BB=${1:-swin_t}
TS=trainset_v4
OUT=output
RUNS=${OUT}/02_training_runs/runs_v4
CV="${RUNS}/${BB}_cv"
DIR="${RUNS}/${BB}_full_final"

for f in paths.py 02_train.py 03_sanity_check.py; do
  [[ -e "${f}" ]] || { echo "找不到 ${f}"; exit 1; }
done
[[ -f "${CV}/cv_summary.json" ]] || { echo "找不到 ${CV}/cv_summary.json，先跑 zsh 02_run_cv_v4.zsh ${BB}"; exit 1; }

EPOCHS=$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(max(1, round(s['median_best_epoch'])))" "${CV}/cv_summary.json")

mkdir -p ${OUT}/06_logs
LOG="${OUT}/06_logs/final_v4_$(date +%Y%m%d_%H%M).log"
caffeinate -i -w $$ &

{
  echo "==================== ${BB} 最终模型 $(date '+%F %T') ===================="
  echo "训练集 ${TS}；epochs = ${EPOCHS}（${CV}/cv_summary.json 的 median_best_epoch）"
  if [[ -f "${DIR}/final.pt" ]]; then
    echo "== ${DIR}/final.pt 已存在，跳过训练"
  else
    # 02_train.py 的 full 模式可能不按 --resume_dir 存放，而是存到 runs/<backbone>_full_final；
    # 那里如果已有别的模型（如 v3），训练会把它覆盖，所以先检查
    DEFAULT="${OUT}/02_training_runs/runs/${BB}_full_final"
    if [[ -e "${DEFAULT}" ]]; then
      echo "✗ ${DEFAULT} 已存在，训练可能会覆盖它。请先改名保存，例如："
      echo "    mv ${DEFAULT} ${DEFAULT}_v3"
      exit 1
    fi
    TMP=$(mktemp)
    python 02_train.py --backbone "${BB}" --mode full --epochs "${EPOCHS}" --trainset "${TS}" --resume_dir "${DIR}" | tee "${TMP}"
    if [[ ! -f "${DIR}/final.pt" ]]; then
      SAVED=$(grep -o "最终模型已保存：.*final\.pt" "${TMP}" | tail -1 | sed 's/^最终模型已保存：//')
      [[ -n "${SAVED}" && -f "${SAVED}" ]] || { echo "✗ 找不到刚训练的 final.pt，请看上面的输出"; exit 1; }
      echo "== 模型被存到了 ${SAVED:h}，移到 ${DIR}"
      mkdir -p "${DIR:h}"
      mv "${SAVED:h}" "${DIR}"
    fi
    rm -f "${TMP}"
  fi
  shasum -a 256 "${DIR}/final.pt" | tee "${DIR}/model_sha256.txt"

  echo "\n==================== 自检（训练截图抽样，应 ≥ 0.95） ===================="
  python 03_sanity_check.py --models "${DIR}"
  echo "\n自检通过后再跑：zsh 03_eval_roi_v4.zsh ${BB}"
} 2>&1 | tee "${LOG}"
