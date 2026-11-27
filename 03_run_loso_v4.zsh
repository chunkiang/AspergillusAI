#!/bin/zsh
# =============================================================================
# 03_run_loso_v4.zsh —— 第 4 步：按来源留一（leave-one-source-out）外部验证，v4 口径
#
# 与 03_run_loso.zsh 相同，只改两处：
#   · 训练集一律不含 ✅ 文件夹截图（--exclude_sources dev_checked）
#   · 输出另存为 *_v4，不覆盖已有的 LOSO 结果
#
# 对每个外部来源 S：
#   1) 训练集 = 本院 + 其他外部来源的 ROI（来源 S 的全部截图不进训练）
#   2) 训练 swin_t（full，EPOCHS 个 epoch）
#   3) 只在来源 S 的盲法 ROI 上测试
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh 03_run_loso_v4.zsh                 # 全部 6 个来源，约 6 × 训练时间
#     zsh 03_run_loso_v4.zsh TP HYY          # 只跑指定来源
#     EPOCHS=10 zsh 03_run_loso_v4.zsh       # 改 epoch 数（默认 14，与 v3 的 LOSO 一致）
# 中断后重跑：已有 final.pt 的来源跳过训练，直接测试
#
# 输出：
#   output/01_trainsets/trainset_loso_v4_<来源>/
#   output/02_training_runs/runs_loso_v4/swin_t_<来源>/final.pt
#   output/03_external_eval/ext_loso_v4/<来源>/metrics.txt、per_case.csv
#   output/03_external_eval/ext_loso_v4/汇总.txt
#   output/06_logs/loso_v4_<时间>.log
# =============================================================================
set -e -o pipefail
cd "${0:A:h}"

EPOCHS=${EPOCHS:-14}
BB=swin_t
OUT=output
ROI_DIR="/Users/chunjiang/Documents/真菌AI/CNN/roi_test"
BUILD_OPTS=(--replace_classes all_v2 --review review_images.csv --add_roi --exclude_sources dev_checked)

# 来源代码 → 03_eval_external.py 用的文件夹关键字（网络图 1 必须带下划线）
typeset -A KEY
KEY=(TP 谭盼  HYY 胡音音  MQQ 马琼琼  LYF 李一凡  WEB1 网络真菌图片_  WEB2 网络真菌图2)
if (( $# )); then SITES=("$@"); else SITES=(TP HYY MQQ LYF WEB1 WEB2); fi

for f in paths.py 01_build_trainset.py 02_train.py 03_eval_external.py; do
  [[ -e "${f}" ]] || { echo "找不到 ${f}"; exit 1; }
done
grep -q "exclude_sources" 01_build_trainset.py || { echo "01_build_trainset.py 不是 10-21 版（缺 --exclude_sources）"; exit 1; }

EXT=${OUT}/03_external_eval/ext_loso_v4
mkdir -p ${OUT}/06_logs ${OUT}/02_training_runs/runs_loso_v4 ${EXT}
LOG="${OUT}/06_logs/loso_v4_$(date +%Y%m%d_%H%M).log"
caffeinate -i -w $$ &

{
  for S in "${SITES[@]}"; do
    K="${KEY[$S]}"
    [[ -n "${K}" ]] || { echo "未知来源 ${S}（可选：${(k)KEY}）"; continue; }
    TS="trainset_loso_v4_${S}"
    RUN="${OUT}/02_training_runs/runs_loso_v4/${BB}_${S}"
    echo "\n==================== 留出 ${S}（${K}） $(date '+%F %T') ===================="
    python 01_build_trainset.py "${BUILD_OPTS[@]}" --exclude_sites "${S}" --out "${TS}" | \
      { grep -E "留出|剔除|外部 ROI|人工核对|显微截图|⚠️" || true; }
    python - "${OUT}/01_trainsets/${TS}/trainset_micro.csv" "${S}" <<'EOF'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); s = sys.argv[2]
n_dc, n_s = int((d["source"] == "dev_checked").sum()), int((d["site"] == s).sum())
print(f"核对：{len(d)} 张；dev_checked {n_dc} 张；来源 {s} {n_s} 张")
sys.exit("✗ 训练集里还有 dev_checked 或被留出来源的图，停止" if n_dc or n_s else 0)
EOF
    if [[ -f "${RUN}/final.pt" ]]; then
      echo "== ${RUN}/final.pt 已存在，跳过训练"
    else
      python 02_train.py --backbone "${BB}" --mode full --epochs "${EPOCHS}" --trainset "${TS}" --resume_dir "${RUN}"
    fi
    python 03_eval_external.py --models "${RUN}" --roi_dir "${ROI_DIR}" --sources "${K}" --out "ext_loso_v4/${S}"
  done

  echo "\n==================== 汇总（病例级，[全部] 行） ===================="
  for S in TP HYY MQQ LYF WEB1 WEB2; do
    if [[ -f "${EXT}/${S}/metrics.txt" ]]; then
      echo "${S}: $(grep -m1 '\[全部\]' "${EXT}/${S}/metrics.txt" || true)"
    fi
  done | tee "${EXT}/汇总.txt"
} 2>&1 | tee "${LOG}"
