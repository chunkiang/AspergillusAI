#!/bin/zsh
# =============================================================================
# clin_5_eval.zsh —— 第 5 步：一键评估（揭盲 → 镜检模型 → 菌落模型 → 曲霉融合 → 最终报告）
#
# 用法：  zsh clin_5_eval.zsh
# 每运行一次生成一个新的结果文件夹（不覆盖旧结果）：
#   output/03_external_eval/clinical/<PILOT_>YYYYMMDD_HHMMSS/
#   ├── RUN_RECORD.md          ★ 本次用了哪些 python 脚本（sha256）、什么参数、什么模型、什么答案表
#   ├── final/report.txt       ★ 最终结果（主要看这个）
#   ├── final/per_isolate_final.csv   每株：通路、最终报告、是否复核、复核原因、建议操作
#   ├── micro/                 镜检模型结果（03_eval_clinical.py：metrics.txt、per_isolate.csv …）
#   │                          confusion.csv / confusion_roi.csv + confusion_matrix(_roi).png/.pdf（株级 / ROI 级混淆矩阵）
#   ├── colony/                菌落模型结果（03_colony_predict.py：per_strain.csv、metrics.txt …）
#   ├── colony_preprocessed/   本次菌落预处理（质控拼图 qc_sheets/ 可以肉眼检查）
#   └── _snapshot/             本次用到的全部脚本、配置、答案表的副本
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "一键评估"
check_rules
keep_awake

RUN_ID="$([[ ${PILOT} == 1 ]] && echo PILOT_)$(date +%Y%m%d_%H%M%S)"
RUN="${EVAL_ROOT}/${RUN_ID}"
[[ -e "${RUN}" ]] && die "${RUN} 已存在，请稍后再试"
mkdir -p "${RUN}/_snapshot"
CLIN_RUN_DIR="${RUN}"
REC="${RUN}/RUN_RECORD.md"
CM="output/02_training_runs/${COLONY_MODEL}"; [[ -d "${COLONY_MODEL}" ]] && CM="${CLIN_KIT_DIR}/${COLONY_MODEL}"
UNBLIND_FORCE=(); LOCK_ARGS=(--lock "${LOCK}")
if [[ ${PILOT} == 1 ]]; then UNBLIND_FORCE=(--force); LOCK_ARGS=(--no_lock); fi

{
  print -r -- "# 临床验证运行记录 ${RUN_ID}"
  print -r -- ""
  print -r -- "- 开始时间：$(now)"
  print -r -- "- 模式：$([[ ${PILOT} == 1 ]] && echo '试运行 PILOT=1（不能写进论文）' || echo '正式 PILOT=0')"
  print -r -- "- 电脑：$(hostname)；python：$(python -c 'import sys; print(sys.executable, sys.version.split()[0])')"
  print -r -- "- 外层脚本：${WRAPPER}（sha256 $(sha "${WRAPPER}")）；配置 clin_config.zsh（sha256 $(sha clin_config.zsh)）"
  print -r -- "- 镜检模型：${MICRO_MODELS[*]}"
  print -r -- "- 菌落 python：${COLONY_PY}"
  print -r -- "- 菌落模型：${COLONY_MODEL}$([[ -f "${CM}/final_model.joblib" ]] && echo "（final_model.joblib sha256 $(sha "${CM}/final_model.joblib")）")"
  print -r -- "- 判读规则：P_MIN=${P_MIN} GAP_MIN=${GAP_MIN} ASP_MIN=${ASP_MIN} ALPHA=${ALPHA}（曲霉融合：菌落权重，线性加权）"
  print -r -- "- 素材：${CLIN_ROOT}"
  print -r -- ""
  print -r -- "## 依次运行的 python 脚本与参数"
  print -r -- ""
} > "${REC}"
info "本次结果文件夹 ${RUN}"

# ---------------------------------------------------------------- ① 检查 + 生成派生答案表
step "① 检查素材、生成派生答案表"
rc=0
run_py "${REC}" 00_clin_check.py --micro_dir "${MICRO_DIR}" --colony_dir "${COLONY_DIR}" --answers "${ANS_MASTER}" \
  --micro_model "${MICRO_MODELS[1]}" --id_regex "${ID_REGEX}" --inventory "${RUN}/inventory.csv" \
  --split --require_answers || rc=$?
[[ ${rc} -eq 3 ]] && die "答案表没填完：打开 ${ANS_MASTER} 填好 reference / train_class / method 再运行"
[[ ${rc} -ne 0 ]] && die "素材检查未通过（见上面的 ❌）"

# ---------------------------------------------------------------- ② 镜检
step "② 镜检：揭盲 → 锁定模型预测（所有菌株）"
HAVE_MICRO=0
if [[ -f "${ROI_DIR}/crop_log.csv" ]] && grep -q ",crop," "${ROI_DIR}/crop_log.csv"; then
  run_py "${REC}" 03_roi_clinical.py --src "${MICRO_DIR}" --out "${ROI_DIR}" --unblind \
    --answers "${ANS_DIR}/answers_clinical.csv" "${UNBLIND_FORCE[@]}" \
    || die "揭盲失败（正式模式下必须先截完全部图片；试运行请确认 PILOT=1）"
  cp "${ROI_DIR}/roi_clinical_table.csv" "${ROI_DIR}/unblind_summary.txt" "${RUN}/_snapshot/" 2>/dev/null
  run_py "${REC}" 03_eval_clinical.py --models "${MICRO_MODELS[@]}" --roi_dir "${ROI_DIR}" --out "${RUN}/micro" \
    "${LOCK_ARGS[@]}" --p_min "${P_MIN}" --gap_min "${GAP_MIN}" \
    || die "镜检模型评估失败"
  # 混淆矩阵图（株级 + ROI 级）；作图失败不影响其他结果
  for cf lvl in confusion "isolate level" confusion_roi "ROI level"; do
    [[ -f "${RUN}/micro/${cf}.csv" ]] || continue
    run_py "${REC}" 02b_plot_confusion.py "${RUN}/micro/${cf}.csv" \
      --title "Clinical validation $([[ ${PILOT} == 1 ]] && echo '(PILOT) ')— microscopy model, ${lvl}" \
      || warn "镜检混淆矩阵图（${cf}）生成失败，不影响其他结果"
  done
  HAVE_MICRO=1
else
  warn "还没有镜检截图（${ROI_DIR}/crop_log.csv），跳过镜检。先运行 zsh clin_3_roi.zsh"
fi

# ---------------------------------------------------------------- ③ 菌落
step "③ 菌落（曲霉）：预处理 → 锁定模型预测"
HAVE_COLONY=0
NCOL=$(( $(wc -l < "${ANS_DIR}/answers_colony.csv") - 1 ))
if [[ ${NCOL} -gt 0 ]] && [[ -n "$(find "${COLONY_ANN}" -name '*.json' 2>/dev/null | head -1)" ]]; then
  if check_colony_env "${REC}" && run_py_colony "${REC}" 01_colony_preprocess.py build --root "${COLONY_DIR}" --ann_dir "${COLONY_ANN}" \
       --out "${RUN}/colony_preprocessed" --views O \
     && run_py_colony "${REC}" 03_colony_predict.py --model "${COLONY_MODEL}" --pre "${RUN}/colony_preprocessed" \
       --answers "${ANS_DIR}/answers_colony.csv" --out "${RUN}/colony" --t1 "${P_MIN}" --t2 "${GAP_MIN}"; then
    HAVE_COLONY=1
  else
    warn "菌落部分失败，本次只用镜检结果（看上面的 ❌；常见原因：没做白平衡确认、菌落模型路径不对）"
  fi
else
  warn "没有已标注的菌落照片，跳过菌落（曲霉将只用镜检结果）。需要的话先运行 zsh clin_4_colony.zsh"
fi

# ---------------------------------------------------------------- ④ 融合
step "④ 融合：曲霉 = 菌落为主（α=${ALPHA}）+ 镜检；非曲霉 = 镜检"
[[ ${HAVE_MICRO} == 1 || ${HAVE_COLONY} == 1 ]] || die "镜检和菌落都没有结果，无法评估"
FUSE=(--out "${RUN}/final" --answers "${ANS_MASTER}" --asp_min "${ASP_MIN}" --alpha "${ALPHA}"
      --p_min "${P_MIN}" --gap_min "${GAP_MIN}")
[[ ${HAVE_MICRO} == 1 ]] && FUSE+=(--micro "${RUN}/micro")
[[ ${HAVE_COLONY} == 1 ]] && FUSE+=(--colony "${RUN}/colony")
[[ ${PILOT} == 1 ]] || FUSE+=(--lock "${LOCK}")      # 正式模式：核对锁定的 α、阈值、菌落模型和融合脚本
run_py "${REC}" 03_clin_fusion.py "${FUSE[@]}" || die "融合失败"

# ---------------------------------------------------------------- ⑤ 存档
step "⑤ 存档本次用到的脚本、配置和答案表"
cp clin_*.zsh clin_sections.py 00_clin_check.py 00_clin_selftest.py 03_clin_fusion.py 03_roi_clinical.py 03_eval_clinical.py \
   03_eval_external.py 02_train.py 02b_plot_confusion.py 01_colony_preprocess.py 02_colony_train.py 03_colony_predict.py paths.py \
   "${RUN}/_snapshot/" 2>/dev/null
cp "${ANS_DIR}"/answers_*.csv "${RUN}/_snapshot/" 2>/dev/null
[[ -f "${LOCK}" ]] && cp "${LOCK}" "${RUN}/_snapshot/"
cp "${CLIN_LOG}" "${RUN}/_snapshot/terminal.log" 2>/dev/null
{
  print -r -- "## 结果"
  print -r -- ""
  print -r -- "- 镜检：$([[ ${HAVE_MICRO} == 1 ]] && echo "micro/metrics.txt" || echo '本次未运行')"
  print -r -- "- 菌落：$([[ ${HAVE_COLONY} == 1 ]] && echo "colony/metrics.txt" || echo '本次未运行')"
  print -r -- "- 最终：final/report.txt、final/per_isolate_final.csv"
  print -r -- "- 结束时间：$(now)"
} >> "${REC}"
ok "已存档到 ${RUN}/_snapshot/"

print -r -- ""
print -r -- "${BG_GRN}  📊 最终报告：${RUN}/final/report.txt  ${R}"
print -r -- "${GRN}  📋 运行记录：${REC}${R}"
