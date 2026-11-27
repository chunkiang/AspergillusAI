#!/bin/zsh
# =============================================================================
# fsim_0_check.zsh —— 第 0 步：诊断（只读）
#   找镜检 / 菌落的 OOF 预测表，识别列名，生成 类别 → 节 映射表，核对融合设置
#   用法：cd 真菌AI/Audit && zsh fsim_0_check.zsh
# =============================================================================
source "${0:A:h}/fsim_lib.zsh"

say_start "同节随机配对 · 第 0 步：诊断"
fsim_check_env

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${CHECK_ROOT}/${STAMP}"

args=(--audit "$FSIM_AUDIT" --micro_run "$MICRO_RUN" --backbone "$MICRO_BACKBONE"
      --colony_run "$COLONY_RUN" --out "$OUTDIR")
[[ -n "$MICRO_OOF" ]] && args+=(--micro_oof "$MICRO_OOF")
[[ -f "$TAXONOMY" ]] && args+=(--taxonomy "$TAXONOMY") || say_warn "没有 ${TAXONOMY}，节映射改用关键词推断"

if fsim_run "$OUTDIR" "fsim_0_check" "00_fusion_sim_check.py" "${args[@]}"; then
  cp "$OUTDIR/CHECK_RESULT.json" "${CHECK_ROOT}/LATEST_CHECK.json"
  cp "$OUTDIR/section_map_auto.csv" "${CHECK_ROOT}/LATEST_section_map.csv"
  say_ok "诊断结果：${OUTDIR}"
  print "👉 请打开 ${CHECK_ROOT}/LATEST_section_map.csv 核对每个曲霉类别对应的节"
  print "   （要改就直接改这个文件，存为 CSV UTF-8；第 1 步读的就是它）"
  say_done "第 0 步完成 → 下一步：zsh fsim_1_run.zsh"
else
  say_err "诊断没通过，结果在 ${FSIM_LAST_OUT}"
  print "   常见处理：自动找错文件 → 在 fsim_config.zsh 里填 MICRO_OOF；节没识别 → 改映射表后重跑本步"
  print "   处理不了就把 ${FSIM_LAST_OUT}/terminal.log 发给 Claude"
  exit 1
fi
