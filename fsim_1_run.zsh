#!/bin/zsh
# =============================================================================
# fsim_1_run.zsh —— 第 1 步：同节随机配对，估计融合增益
#   读第 0 步的 LATEST_CHECK.json 和 LATEST_section_map.csv（你核对或修改过的映射表）
#   用法：cd 真菌AI/Audit && zsh fsim_1_run.zsh
# =============================================================================
source "${0:A:h}/fsim_lib.zsh"

say_start "同节随机配对 · 第 1 步：融合增益分析"
fsim_check_env

CHECK="${CHECK_ROOT}/LATEST_CHECK.json"
SMAP="${CHECK_ROOT}/LATEST_section_map.csv"
[[ -f "$CHECK" ]] || { say_err "没有 ${CHECK}，请先运行 zsh fsim_0_check.zsh"; exit 1; }
[[ -f "$SMAP" ]]  || { say_err "没有 ${SMAP}，请先运行 zsh fsim_0_check.zsh"; exit 1; }

# 诊断之后改过设置（输入路径），提醒重跑诊断
if [[ "${FSIM_AUDIT}/fsim_config.zsh" -nt "$CHECK" ]]; then
  say_warn "fsim_config.zsh 在诊断之后改过；如果改的是输入路径，请先重跑 fsim_0_check.zsh"
fi
say_ok "规则：${FUSION_RULE}，菌落权重 ${COLONY_WEIGHT}，分流阈值 ${ASP_GATE}，配对 ${N_ITER} 次"

STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="${RUN_ROOT}/${STAMP}"
mkdir -p "$OUTDIR/_snapshot"
cp "$CHECK" "$SMAP" "$OUTDIR/_snapshot/"

if fsim_run "$OUTDIR" "fsim_1_run" "03_fusion_simpair.py" \
    --check "$CHECK" --section_map "$SMAP" \
    --rule "$FUSION_RULE" --weight "$COLONY_WEIGHT" --sens_weights "$SENS_WEIGHTS" \
    --asp_gate "$ASP_GATE" --n_iter "$N_ITER" --seed "$SEED" --out "$OUTDIR"; then
  say_ok "结果：${OUTDIR}/report.txt"
  say_done "第 1 步完成"
else
  say_err "运行出错，结果在 ${FSIM_LAST_OUT}；把其中的 terminal.log 发给 Claude"
  exit 1
fi
