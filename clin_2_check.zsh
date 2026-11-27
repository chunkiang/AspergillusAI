#!/bin/zsh
# =============================================================================
# clin_2_check.zsh —— 第 2 步：检查照片命名 / 盘点素材 / 建立或更新答案表（不改动照片）
#
# 用法：  zsh clin_2_check.zsh
# 每次新加了菌株都可以再跑一次：新株自动追加到 answers_master.csv，已填内容不会被覆盖
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "检查照片命名、盘点素材、维护答案表"

[[ -d "${MICRO_DIR}" ]] || die "找不到 ${MICRO_DIR}，先运行 zsh clin_1_init.zsh"

step "检查与盘点"
rc=0
run_py - 00_clin_check.py \
  --micro_dir "${MICRO_DIR}" \
  --colony_dir "${COLONY_DIR}" \
  --answers "${ANS_MASTER}" \
  --micro_model "${MICRO_MODELS[1]}" \
  --id_regex "${ID_REGEX}" \
  --inventory "${CLIN_ROOT}/inventory.csv" || rc=$?

if [[ ${rc} -eq 2 ]]; then
  die "有命名或答案表错误（见上面的 ❌），改好后重跑本脚本"
elif [[ ${rc} -ne 0 ]]; then
  die "检查脚本异常退出"
fi
ok "素材清单 ${CLIN_ROOT}/inventory.csv"
ok "答案表 ${ANS_MASTER}（用 Excel 填写 reference / train_class / method，另存为 CSV UTF-8）"
print -r -- ""
info "下一步：zsh clin_3_roi.zsh（镜检截图）  和  zsh clin_4_colony.zsh（菌落标注）"
