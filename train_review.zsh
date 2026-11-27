#!/bin/zsh
# =============================================================================
# train_review.zsh —— 随时查看：训练原图 / 训练数据集 / 训练后的模型（只读，不改任何文件）
#
# 用法（fungal 环境，Audit 文件夹）：  zsh train_review.zsh
#
#   1) 训练原始图片汇总：每个来源多少张、在哪个文件夹；原图_重截 截图进度；补拍_2026 标注进度
#   2) 训练数据集：列出全部显微 / 菌落数据集，输入编号看创建时间、来源、各类张数、菌株数等
#   3) 训练后的模型：列出全部模型，输入编号看性能指标、创建时间、数据集、训练参数等
#   任何时候：Q 退出，B 返回上一层
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "查看原图、数据集、模型"
need_scripts train_review.py 00_train_check.py
LST="${LOG_DIR}/.review_$$.tsv"

pause() {   # 看完一屏：回车 / B 返回，Q 退出
  ask "按回车返回（Q 退出）" "" ; return 0
}

review_sources() {
  milestone "${BG_BLU}" "🖼️" "训练原始图片汇总"
  tool train_review.py sources --micro_root "${MICRO_DATASET_ROOT}" --colony_legacy "${COLONY_LEGACY_ROOT}" \
    --crops_new "${CROPS_NEW_ROOT}" --crops_v2 "${CROPS_V2_ROOT}" --roi_test "${ROI_TEST_ROOT}" \
    --recrop_root "${RECROP_ROOT}" --colony_raw "${COLONY_RAW_ROOT}" --ann_dir "${TRAINSETS}/${COLONY_ANN}"
  pause
}

review_list() {   # review_list datasets|models
  local what=$1 title
  [[ ${what} == datasets ]] && title="训练数据集（output/01_trainsets）" || title="训练后的模型（output/02_training_runs）"
  while true; do
    milestone "${BG_BLU}" "$([[ ${what} == datasets ]] && print -r -- 🗂️ || print -r -- 🧠)" "${title}"
    if [[ ${what} == datasets ]]; then
      tool train_review.py datasets --kind all --tsv "${LST}"
    else
      tool train_review.py models --tsv "${LST}"
    fi
    local -a P=("${(@f)$(<"${LST}")}")
    [[ -z "${P[1]}" ]] && { pause; return 0; }
    ask_one "输入编号查看详情（B 返回上一层，Q 退出）" ${#P} || return 0
    local d="${P[$ANS]}"
    print -r -- ""
    print -r -- "${CYN}${THIN}${R}"
    if [[ ${what} == datasets ]]; then tool train_review.py dataset --path "${d}"
    else tool train_review.py model --path "${d}"; fi
    print -r -- "${CYN}${THIN}${R}"
    for f in TRAIN_SUMMARY.md RUN_RECORD.md DATASET_INFO.json; do [[ -f "${d}/$f" ]] && info "${f}：${d}/${f}"; done
    ask "按回车返回列表（B 返回上一层，Q 退出）" "" || return 0
  done
}

while true; do
  print -r -- ""
  print -r -- "${BG_BLU}  🔍 训练资料查看  ${R}"
  print -r -- "   1) 用于训练的原始图片汇总"
  print -r -- "   2) 已有的训练数据集"
  print -r -- "   3) 训练后的模型"
  ask_one "选哪一项？（Q 退出）" 3 || continue
  case ${ANS} in
    1) review_sources ;;
    2) review_list datasets ;;
    3) review_list models ;;
  esac
done
