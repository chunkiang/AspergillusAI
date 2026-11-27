#!/bin/zsh
# =============================================================================
# clin_lock.zsh —— 正式验证前锁定模型和判读规则（只运行一次；试运行 PILOT=1 不需要）
#
# 用法：  zsh clin_lock.zsh
# 生成 ${CLIN_ROOT}/clinical_lock.json（镜检模型指纹 + 阈值；fusion 段：α、ASP_MIN、菌落模型和融合脚本指纹）
# 和 lock_extra.txt（同样内容的文字版）。正式评估时两部分都会被自动核对。
# 锁定后请把这两个文件发邮件给自己留时间戳，再开始截图。
# 需要换模型重新锁定：先手动把旧的 clinical_lock.json 改名备份。
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "锁定模型与判读规则"
check_rules

[[ -f "${LOCK}" ]] && die "${LOCK} 已存在，锁定只做一次。确需重新锁定请先改名备份旧文件"
run_py - 03_eval_clinical.py --write_lock --models "${MICRO_MODELS[@]}" --lock "${LOCK}" \
  --p_min "${P_MIN}" --gap_min "${GAP_MIN}" || die "写锁定文件失败"

CM="output/02_training_runs/${COLONY_MODEL}"; [[ -d "${COLONY_MODEL}" ]] && CM="${COLONY_MODEL}"
[[ -f "${CM}/final_model.joblib" ]] || die "找不到菌落模型 ${CM}/final_model.joblib"
# 融合规则（ASP_MIN、ALPHA、阈值）+ 菌落模型指纹 + 融合脚本指纹 写进同一个 clinical_lock.json
run_py - 03_clin_fusion.py --write_lock --lock "${LOCK}" --colony_model "${CM}/final_model.joblib" \
  --asp_min "${ASP_MIN}" --alpha "${ALPHA}" --p_min "${P_MIN}" --gap_min "${GAP_MIN}" --out "${CLIN_ROOT}" \
  || die "写融合锁定失败（clinical_lock.json 里已有镜检部分；请把它改名备份后重新锁定）"
{
  print -r -- "锁定时间: $(now)"
  print -r -- "镜检模型: ${MICRO_MODELS[*]}"
  print -r -- "菌落模型: ${CM}  final_model.joblib sha256 $(sha "${CM}/final_model.joblib")"
  print -r -- "规则: P_MIN=${P_MIN} GAP_MIN=${GAP_MIN} ASP_MIN=${ASP_MIN} ALPHA=${ALPHA}"
  for f in 03_clin_fusion.py clin_sections.py 03_eval_clinical.py 03_colony_predict.py 03_roi_clinical.py; do
    print -r -- "脚本: ${f}  sha256 $(sha "${f}")"
  done
} > "${CLIN_ROOT}/lock_extra.txt"
ok "已写出 ${LOCK} 和 ${CLIN_ROOT}/lock_extra.txt"
[[ "${PILOT}" == 1 ]] && warn "现在 PILOT=1：把 clin_config.zsh 里的 PILOT 改成 0，正式评估才会核对锁定"
info "请把 clinical_lock.json 和 lock_extra.txt 发邮件给自己留时间戳，然后开始截图"
