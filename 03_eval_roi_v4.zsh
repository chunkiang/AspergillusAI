#!/bin/zsh
# =============================================================================
# 03_eval_roi_v4.zsh —— 第 3 步：v4 最终模型在 roi_test 上做外部验证（只跑一次）
#
# roi_test 已被旧模型、v3、LOSO 看过多次。为避免"在测试集上选模型"，本脚本：
#   · 运行前核对：trainset_v4 中没有 roi_test 的 ROI；模型权重与 02_train_final_v4.zsh 记录的指纹一致
#   · 结果文件夹已存在就拒绝再跑（真需要重跑，加 --force，并在论文中如实说明）
#   · 记录运行时间、模型指纹、命令到 eval_record.txt
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh 03_eval_roi_v4.zsh                    # 默认 swin_t
#     zsh 03_eval_roi_v4.zsh densenet121
#     zsh 03_eval_roi_v4.zsh swin_t --force     # 强制重跑（不建议）
#
# 输出：
#   output/03_external_eval/ext_eval_roi_v4/     metrics.txt、per_case.csv、per_image.csv、eval_record.txt
#   output/06_logs/eval_roi_v4_<时间>.log
# =============================================================================
set -e -o pipefail
cd "${0:A:h}"

BB=${1:-swin_t}
FORCE=0
if [[ "$2" == "--force" ]]; then FORCE=1; fi
TS=trainset_v4
OUT=output
DIR="${OUT}/02_training_runs/runs_v4/${BB}_full_final"
ROI_DIR="/Users/chunjiang/Documents/真菌AI/CNN/roi_test"
EVAL=ext_eval_roi_v4
EVAL_DIR="${OUT}/03_external_eval/${EVAL}"

for f in paths.py 02_train.py 03_eval_external.py 04_check_roi_overlap.py; do
  [[ -e "${f}" ]] || { echo "找不到 ${f}"; exit 1; }
done
[[ -f "${DIR}/final.pt" ]] || { echo "找不到 ${DIR}/final.pt，先跑 zsh 02_train_final_v4.zsh ${BB}"; exit 1; }
if [[ -f "${EVAL_DIR}/metrics.txt" && ${FORCE} -eq 0 ]]; then
  echo "✗ ${EVAL_DIR}/metrics.txt 已存在：roi_test 对 v4 只评估一次。"
  echo "  确需重跑请加 --force，并在论文中说明重跑原因。"
  exit 1
fi

mkdir -p ${OUT}/06_logs
LOG="${OUT}/06_logs/eval_roi_v4_$(date +%Y%m%d_%H%M).log"

{
  echo "==================== 运行前核对 $(date '+%F %T') ===================="
  # 1. 模型指纹
  if [[ -f "${DIR}/model_sha256.txt" ]]; then
    NOW=$(shasum -a 256 "${DIR}/final.pt" | cut -d' ' -f1)
    REC=$(cut -d' ' -f1 "${DIR}/model_sha256.txt")
    [[ "${NOW}" == "${REC}" ]] || { echo "✗ final.pt 与训练时记录的指纹不一致，停止"; exit 1; }
    echo "✓ 模型指纹一致 ${NOW:0:16}…"
  else
    echo "⚠ 没有 model_sha256.txt（不是用 02_train_final_v4.zsh 训练的？），继续"
  fi
  # 2. 训练集里不能有 roi_test 的 ROI
  python 04_check_roi_overlap.py --roi_dir "${ROI_DIR}" --trainset "${TS}"
  python - "${OUT}/01_trainsets/${TS}/trainset_micro.csv" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1])
bad = d[d["source"].isin(["roi_ext", "dev_checked"])]
sys.exit(f"✗ trainset_v4 含 roi_ext/dev_checked {len(bad)} 张，停止" if len(bad) else 0)
print("✓ trainset_v4 不含 roi_ext、dev_checked")
EOF

  echo "\n==================== 外部验证（盲法 ROI） ===================="
  python 03_eval_external.py --models "${DIR}" --roi_dir "${ROI_DIR}" --out "${EVAL}"

  {
    echo "运行时间: $(date '+%F %T')"
    echo "模型: ${DIR}/final.pt"
    echo "模型 sha256: $(shasum -a 256 "${DIR}/final.pt" | cut -d' ' -f1)"
    echo "训练集: ${TS}"
    echo "测试集: ${ROI_DIR}"
    echo "强制重跑: ${FORCE}"
  } > "${EVAL_DIR}/eval_record.txt"

  echo "\n主要结果：${EVAL_DIR}/metrics.txt"
} 2>&1 | tee "${LOG}"
