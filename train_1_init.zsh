#!/bin/zsh
# =============================================================================
# train_1_init.zsh —— 第 1 步：建立训练用的文件夹（可反复运行；只新建缺少的，已有的文件夹和文件一概不动）
#
# 用法（fungal 环境，Audit 文件夹）：  zsh train_1_init.zsh
#
# 会检查 / 补建：
#   output/01_trainsets  02_training_runs/colony  04_audit_review/train_audit  06_logs …
#   原图_重截/<菌种>/（由 01_make_recrop_dirs.py 建，说明文件已存在就不覆盖）
#   补拍_2026/FUM FLA NIG TER NID/
#   output/01_trainsets/colony_annotations/
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
tr_start "${0:t}" "建立（补齐）训练用文件夹"

mk() {   # mk <文件夹>：已存在只提示
  if [[ -d "$1" ]]; then info "已存在 $1"; else mkdir -p "$1" && ok "新建 $1"; fi
}

step "1/4 output 下的文件夹"
for d in "${TRAINSETS}" "${RUNS_ROOT}" "${COLONY_RUNS}" "${OUT}/03_external_eval" "${AUDIT_REVIEW}" \
         "${AUDIT_REVIEW}/train_audit" "${OUT}/05_roi_diagnostics" "${LOG_DIR}" "${TRAINSETS}/${COLONY_ANN}"; do
  mk "$d"
done
info "output 里原有的其他文件（如 move_log.csv）不读、不改"

step "2/4 待截图的显微原图：原图_重截"
run_py - 01_make_recrop_dirs.py --root "${RECROP_ROOT}" --v2_log "${CROPS_V2_ROOT}/crop_log.csv" \
  || die "01_make_recrop_dirs.py 出错"

step "3/4 菌落原始照片：补拍_2026"
mk "${COLONY_RAW_ROOT}"
for g in FUM FLA NIG TER NID; do mk "${COLONY_RAW_ROOT}/$g"; done

step "4/4 现有内容一览（只读）"
nm=0 nc=0
for d in "${TRAINSETS}"/*(N/); do
  [[ -f "$d/trainset_micro.csv" ]] && (( nm++ ))
  [[ -f "$d/manifest.csv" ]] && (( nc++ ))
done
nmod=$(find "${RUNS_ROOT}" \( -name cv_summary.json -o -name final.pt -o -name locked_config.json \) -not -path '*/fold_*' 2>/dev/null | wc -l | tr -d ' ')
info "显微数据集 ${nm} 个、菌落数据集 ${nc} 个、模型 ${nmod} 个（详情：zsh train_review.zsh）"

README="${OUT}/训练流程_目录说明.txt"
if [[ ! -f "${README}" ]]; then
  cat > "${README}" <<EOF
训练流程目录说明（由 train_1_init.zsh 生成；以后不会覆盖）

脚本顺序（都在 Audit 文件夹运行）：
  zsh train_0_selftest.zsh    自检
  zsh train_1_init.zsh        建文件夹
  zsh train_2_audit.zsh       图片审计；可加入新的显微 / 菌落图片
  zsh train_3_crop.zsh        显微截图、菌落标注、菌落白平衡
  zsh train_4_dataset.zsh     建显微数据集 + 菌落数据集
  zsh train_5_micro.zsh       训练显微模型（CV / 最终模型）
  zsh train_6_colony.zsh      训练菌落模型
  zsh train_review.zsh        随时查看原图、数据集、模型

图片位置：
  ${RECROP_ROOT}/<菌种>/<菌株>/   新的显微原图（截图后进入 crops_v2）
  ${COLONY_RAW_ROOT}/<组>/          菌落照片 PUPH-FUM-001-B-O-1.jpg

输出：
  01_trainsets/micro_<时间>_<来源代号>/     显微数据集（含 DATASET_INFO.json）
  01_trainsets/colony_<时间>/               菌落数据集
  02_training_runs/<数据集名>/<backbone>_cv           交叉验证
  02_training_runs/<数据集名>/<backbone>_full_final   最终模型
  02_training_runs/colony/<时间>/                     菌落模型
  每个模型文件夹里：TRAIN_SUMMARY.md（总结）、RUN_RECORD.md（命令与脚本指纹）
  06_logs/train_ledger.tsv                  所有 python 调用的总账
EOF
  ok "写出 ${README}"
fi

next_step "zsh train_2_audit.zsh" "审计现有图片；要加入新图片：zsh train_2_audit.zsh <文件夹> [<文件夹>…]"
