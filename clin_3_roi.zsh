#!/bin/zsh
# =============================================================================
# clin_3_roi.zsh —— 第 3 步：镜检照片盲法截取产孢结构（所有菌株，包括曲霉和非曲霉）
#
# 用法：
#     zsh clin_3_roi.zsh           打开截图窗口（随时 q 退出，下次从断点继续）
#     zsh clin_3_roi.zsh status    只看进度
#
# 窗口按键：左键截取  +/- 缩放  ←/→ 旋转  1/2/3 完整/局部/多个  i 成熟度
#           u 撤销  n/空格 下一张  b 上一张  s 无产孢结构  x 质量差  q 退出
# 每张图截 1–5 个最典型的产孢结构；没有就按 s，不要勉强截菌丝
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "镜检盲法截图"

[[ -d "${MICRO_DIR}" ]] || die "找不到 ${MICRO_DIR}，先运行 zsh clin_1_init.zsh"
[[ -n "$(ls -A "${MICRO_DIR}" 2>/dev/null)" ]] || die "${MICRO_DIR} 里还没有照片"

if [[ "$1" == "status" ]]; then
  run_py - 03_roi_clinical.py --src "${MICRO_DIR}" --out "${ROI_DIR}" --status || die "查看进度失败"
  exit 0
fi

info "图片已随机打乱并匿名（CL-0001…），窗口里看不到株号和菌名"
run_py - 03_roi_clinical.py --src "${MICRO_DIR}" --out "${ROI_DIR}" || die "截图工具异常退出（已截的不会丢失，重新运行即可继续）"
ok "截图保存在 ${ROI_DIR}"
info "全部截完后（或试运行时随时）运行  zsh clin_5_eval.zsh"
