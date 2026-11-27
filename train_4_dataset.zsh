#!/bin/zsh
# =============================================================================
# train_4_dataset.zsh —— 第 4 步：建立训练数据集（先显微，再菌落）
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh train_4_dataset.zsh            先建显微数据集，再问要不要接着建菌落数据集
#     zsh train_4_dataset.zsh micro      只建显微数据集
#     zsh train_4_dataset.zsh colony     只建菌落数据集
#
# 显微数据集（01_build_trainset.py）：
#   ① 打印当前参数（train_config.zsh 的 DS_* 项，默认与 trainset_v4 相同）
#   ② 按当前参数试建一次，列出 5 个截图来源各有多少张可用图
#   ③ 用编号选来源（如 1,3,4），自动起名 micro_<时间>_<来源代号>，可改成自己的名字
#   ④ 正式建立 → output/01_trainsets/<名字>/（trainset_micro.csv、class_names.json、summary.txt、DATASET_INFO.json）
# 菌落数据集（01_colony_preprocess.py build）：
#   已标注 + 已确认白平衡的正面照片 → 颜色校正、尺度统一、自动截图、质控拼图
#   → output/01_trainsets/colony_<时间>/（manifest.csv、crops/、qc_sheets/、DATASET_INFO.json）
# 已有的数据集一律不覆盖（重名会要求换名字）。
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "建立训练数据集"
need_scripts 01_build_trainset.py 01_colony_preprocess.py train_review.py 00_train_check.py
check_python
ANN="${TRAINSETS}/${COLONY_ANN}"

SRC_KEYS=(dev_original dev_checked crops_new crops_v2 roi_ext)
SRC_CODE=(O C N V R)
SRC_NAME=("旧截图" "✅ 外院投稿截图" "网络补截" "规范截图" "外部盲法 ROI")
SRC_LOC=("${MICRO_DATASET_ROOT}" "${MICRO_DATASET_ROOT}（✅ 文件夹）" "${CROPS_NEW_ROOT}" "${CROPS_V2_ROOT}" "${ROI_TEST_ROOT}")

print_code_table() {
  print -r -- "${B}   数据集名字里的来源代号：${R}"
  local i
  for (( i = 1; i <= 5; i++ )); do
    printf "     %s = %-13s %s\n" "${SRC_CODE[$i]}" "${SRC_KEYS[$i]}" "${SRC_NAME[$i]}"
  done
}

# =============================================================================
build_micro() {
  milestone "${BG_BLU}" "🧫" "显微数据集"
  [[ -f "${AUDIT_REVIEW}/train_audit/train_manifest.csv" ]] \
    || die "没有 output/04_audit_review/train_audit/train_manifest.csv：先运行 zsh train_2_audit.zsh"

  # ---------------------------------------------------------- ① 参数
  step "① 当前参数（在 train_config.zsh 里改；默认与 trainset_v4 相同）"
  local yes="${GRN}是${R}" no="${DIM}否${R}"
  print -r -- "   DS_REPLACE_ALL_V2   = ${DS_REPLACE_ALL_V2}  $([[ ${DS_REPLACE_ALL_V2} == 1 ]] && print -r -- ${yes} || print -r -- ${no})  有规范截图的菌种改用规范截图，不用旧截图  (--replace_classes all_v2)"
  print -r -- "   DS_USE_REVIEW       = ${DS_USE_REVIEW}  $([[ ${DS_USE_REVIEW} == 1 ]] && print -r -- ${yes} || print -r -- ${no})  按人工核对剔除\"不合格\"\"局部\"截图        (--review ${DS_REVIEW_CSV})"
  print -r -- "   DS_EXCLUDE_IMMATURE = ${DS_EXCLUDE_IMMATURE}  $([[ ${DS_EXCLUDE_IMMATURE} == 1 ]] && print -r -- ${yes} || print -r -- ${no})  不用标为\"未成熟\"的规范截图              (--exclude_immature)"
  print -r -- "   DS_MIN_CROP         = ${DS_MIN_CROP}     截图边长小于它的不用                     (--min_crop ${DS_MIN_CROP})"
  local same_v4=1
  [[ ${DS_REPLACE_ALL_V2} == 1 && ${DS_USE_REVIEW} == 1 && ${DS_EXCLUDE_IMMATURE} == 0 && ${DS_MIN_CROP} == 64 ]] || same_v4=0
  (( same_v4 )) && ok "与 trainset_v4 的设置相同" || warn "与 trainset_v4 的设置不同（会记进 DATASET_INFO.json）"

  local -a OPTS=(--min_crop "${DS_MIN_CROP}" --micro_root "${MICRO_DATASET_ROOT}" --colony_root "${COLONY_LEGACY_ROOT}"
                 --crop_log "${CROPS_NEW_ROOT}/crop_log.csv" --v2_log "${CROPS_V2_ROOT}/crop_log.csv"
                 --roi_log "${ROI_TEST_ROOT}/crop_log.csv" --recrop_root "${RECROP_ROOT}" --taxonomy "${TAXONOMY}")
  [[ ${DS_REPLACE_ALL_V2} == 1 ]] && OPTS+=(--replace_classes all_v2)
  if [[ ${DS_USE_REVIEW} == 1 ]]; then
    [[ -f "${DS_REVIEW_CSV}" || -f "${AUDIT_REVIEW}/${DS_REVIEW_CSV}" ]] \
      || die "DS_USE_REVIEW=1 但找不到 ${DS_REVIEW_CSV}（应在 Audit/ 或 output/04_audit_review/）；或把 DS_USE_REVIEW 改成 0"
    OPTS+=(--review "${DS_REVIEW_CSV}")
  fi
  [[ ${DS_EXCLUDE_IMMATURE} == 1 ]] && OPTS+=(--exclude_immature)
  local opts_json="{\"替换为规范截图 --replace_classes all_v2\": ${DS_REPLACE_ALL_V2}, \"人工核对剔除 --review\": \"$([[ ${DS_USE_REVIEW} == 1 ]] && print -r -- ${DS_REVIEW_CSV})\", \"排除未成熟 --exclude_immature\": ${DS_EXCLUDE_IMMATURE}, \"最小截图边长 --min_crop\": ${DS_MIN_CROP}}"

  # ---------------------------------------------------------- ② 试建，统计各来源
  step "② 按当前参数统计各来源可用的图（试建，用完即删）"
  local PV="${TRAINSETS}/_preview_$$" PVLOG="${LOG_DIR}/dataset_preview_$(date +%Y%m%d_%H%M%S).log" ROI_OK=1
  rm -rf "${PV}"
  if ! "${PYTHON}" -u 01_build_trainset.py "${OPTS[@]}" --add_roi --out "${PV}" > "${PVLOG%.log}_roi.log" 2>&1; then
    ROI_OK=0
    ROI_WHY=$(grep -v '^[[:space:]]*$' "${PVLOG%.log}_roi.log" | tail -1)
    rm -rf "${PV}"
    "${PYTHON}" -u 01_build_trainset.py "${OPTS[@]}" --out "${PV}" > "${PVLOG}" 2>&1 || {
      err "试建失败，最后几行："; tail -15 "${PVLOG}"; rm -rf "${PV}"
      die "01_build_trainset.py 出错（完整输出 ${PVLOG}）"
    }
    warn "加入外部 ROI 时出错，外部 ROI（R）本次不可选：${ROI_WHY}"
    hint "完整输出 ${PVLOG%.log}_roi.log；不打算纳入外部 ROI 的话可以忽略"
  fi
  local dev_original=0 dev_checked=0 crops_new=0 crops_v2=0 roi_ext=0
  tool train_review.py preview --dir "${PV}" --kv "${PV}.kv" && kv_read cat "${PV}.kv"
  local CNT=(${dev_original} ${dev_checked} ${crops_new} ${crops_v2} ${roi_ext})
  local NCLS=$(grep -m1 '^类别数' "${PV}/summary.txt" 2>/dev/null)
  rm -rf "${PV}" "${PV}.kv"
  grep -E '^⚠️|^人工核对|^按截图来源|^替换类别' "${PVLOG}" | sed 's/^/   /' | head -8

  local stage=1 i k mark okflag total auto chosen_codes name rc
  local -a CHOSEN def
  while true; do
    case ${stage} in
    1) # ------------------------------------------------------ ③ 选来源
      step "③ 选择纳入数据集的截图来源"
      print -r -- "   编号 代号  来源            说明              可用张数   位置"
      for (( i = 1; i <= 5; i++ )); do
        mark=""
        [[ ${i} == 2 ]] && mark="  ${YEL}⚠️ 外院投稿、非盲截取，v4 起不用${R}"
        [[ ${i} == 5 ]] && mark="  ${RED}🔴 选了以后，模型不能再用 roi_test 做外部验证${R}"
        [[ ${i} == 5 && ${ROI_OK} == 0 ]] && mark="  ${DIM}（本次不可用）${R}"
        printf "   %3d)  %s   %-14s  %-14s  %7d   %s%s\n" $i "${SRC_CODE[$i]}" "${SRC_KEYS[$i]}" "${SRC_NAME[$i]}" ${CNT[$i]} "${SRC_LOC[$i]}" "${mark}"
      done
      print -r -- "${DIM}   张数 = 按上面的参数筛选后实际会进入数据集的显微截图；${NCLS}${R}"
      def=()
      for k in "${DS_DEFAULT_SOURCES[@]}"; do def+=(${SRC_KEYS[(ie)$k]}); done
      ask_nums "纳入哪些来源？（如 1,3,4；默认来自 train_config.zsh 的 DS_DEFAULT_SOURCES）" 5 "${(j:,:)def}" || return 10
      CHOSEN=() chosen_codes=""
      for i in $(print -l -- "${NUMS[@]}" | sort -n); do
        if [[ ${i} == 5 && ${ROI_OK} == 0 ]]; then warn "外部 ROI 本次不可用，已去掉"; continue; fi
        (( CNT[$i] == 0 )) && warn "${SRC_CODE[$i]} ${SRC_KEYS[$i]} 没有可用的图（仍记为已选）"
        CHOSEN+=("${SRC_KEYS[$i]}"); chosen_codes+="${SRC_CODE[$i]}"
      done
      (( ${#CHOSEN} )) || { warn "至少要选一个来源"; continue; }
      if (( ${CHOSEN[(Ie)dev_checked]} )); then
        ask_yn "${YEL}你选了 C（✅ 外院投稿截图，非盲截取）。确定纳入？${R}" n; rc=$?
        (( rc != 0 )) && continue
      fi
      if (( ${CHOSEN[(Ie)roi_ext]} )); then
        print -r -- "${BG_RED}  🔴 纳入外部盲法 ROI 后，用这个数据集训练的模型不能在 roi_test 上评估，外部验证只能用按来源留一（LOSO）  ${R}"
        ask_yn "确定纳入 R（外部 ROI）？" n; rc=$?
        (( rc != 0 )) && continue
      fi
      total=0; for k in "${CHOSEN[@]}"; do (( total += CNT[${SRC_KEYS[(ie)$k]}] )); done
      ok "选定：${chosen_codes}（${CHOSEN[*]}），约 ${total} 张"
      stage=2 ;;

    2) # ------------------------------------------------------ ④ 起名
      step "④ 数据集名字"
      print_code_table
      auto="micro_$(stamp)_${chosen_codes}"
      ask "数据集名字（直接回车用自动名字；只能用字母、数字、_、-）" "${auto}"; rc=$?
      (( rc == 10 )) && { stage=1; continue; }
      valid_name "${ANS}" || continue
      name="${ANS}"
      stage=3 ;;

    3) # ------------------------------------------------------ 确认
      step "确认"
      print -r -- "   名字：${B}${name}${R}"
      print -r -- "   位置：${TRAINSETS}/${name}/"
      print -r -- "   来源：${chosen_codes}（${CHOSEN[*]}）"
      print -r -- "   参数：replace_all_v2=${DS_REPLACE_ALL_V2}  review=$([[ ${DS_USE_REVIEW} == 1 ]] && print -r -- ${DS_REVIEW_CSV} || print -r -- 不用)  exclude_immature=${DS_EXCLUDE_IMMATURE}  min_crop=${DS_MIN_CROP}"
      ask_yn "开始建立？" y; rc=$?
      (( rc == 10 )) && { stage=2; continue; }
      (( rc == 1 )) && { info "没有建立"; return 0; }
      break ;;
    esac
  done

  # ---------------------------------------------------------- 正式建立
  local DIR="${TRAINSETS}/${name}" REC
  mkdir -p "${DIR}"
  TR_RUN_DIR="${DIR}"
  REC="${DIR}/RUN_RECORD.md"
  new_record "${REC}" "显微数据集 ${name}"
  print -r -- "- 来源：${chosen_codes}（${CHOSEN[*]}）" >> "${REC}"
  print -r -- "" >> "${REC}"
  local -a EXCL=()
  for k in dev_original dev_checked crops_new crops_v2; do (( ${CHOSEN[(Ie)$k]} )) || EXCL+=($k); done
  local -a BUILD=("${OPTS[@]}")
  (( ${#EXCL} )) && BUILD+=(--exclude_sources "${EXCL[@]}")
  (( ${CHOSEN[(Ie)roi_ext]} )) && BUILD+=(--add_roi)
  step "建立 ${name}"
  run_py "${REC}" 01_build_trainset.py "${BUILD[@]}" --out "${DIR}" || die "01_build_trainset.py 出错"
  [[ -f "${DIR}/trainset_micro.csv" ]] || die "没有生成 trainset_micro.csv"
  step "数据集信息"
  tool train_review.py write-micro-info --dir "${DIR}" --sources "${CHOSEN[@]}" --options "${opts_json}" \
    --command "python 01_build_trainset.py ${(j: :)${(q-)BUILD[@]}} --out ${DIR}" || die "写 DATASET_INFO.json 失败"
  cp train_config.zsh "${DIR}/" 2>/dev/null
  TR_RUN_DIR=""
  print -r -- ""
  print_code_table
  milestone "${BG_GRN}" "💾" "显微数据集已保存：${DIR}"
  info "其中 DATASET_INFO.json（来源、参数、各类张数、清单指纹）、summary.txt、RUN_RECORD.md"
  MICRO_BUILT="${name}"
  return 0
}

# =============================================================================
build_colony() {
  milestone "${BG_BLU}" "🧫" "菌落数据集"
  check_colony_python || die "菌落用 python 不可用（见上面）"
  local NEED_ANN=0 NEED_WB=0 READY=0 PHOTO_O=0 rc
  step "① 标注与白平衡进度"
  tool 00_train_check.py colony-status --colony_raw "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" --kv "${LOG_DIR}/.colony_$$.kv"
  kv_read cat "${LOG_DIR}/.colony_$$.kv"; rm -f "${LOG_DIR}/.colony_$$.kv"
  if (( READY == 0 )); then
    err "没有已标注并确认白平衡的正面照片，建不了菌落数据集"
    hint "先运行 zsh train_3_crop.zsh（菌落标注 → 白平衡）"
    return 1
  fi
  local -a WBOPT=()
  if (( NEED_ANN > 0 )); then
    warn "还有 ${NEED_ANN} 张正面照片没标注：它们不会进入这个数据集"
  fi
  if (( NEED_WB > 0 )); then
    warn "还有 ${NEED_WB} 张已标注、但没确认白平衡的正面照片（01_colony_preprocess.py 默认会拒绝建立）"
    print -r -- "   1) 先去确认白平衡（zsh train_3_crop.zsh wb），本次不建"
    print -r -- "   2) 这些照片按\"无参照\"处理（--allow_missing_wb），不进入主分析"
    ask_one "怎么处理？" 2 1 || return 10
    [[ ${ANS} == 1 ]] && { next_step "zsh train_3_crop.zsh wb"; return 0; }
    WBOPT=(--allow_missing_wb)
  elif (( NEED_ANN > 0 )); then
    ask_yn "仍然现在建立？（否 = 先去 train_3_crop.zsh 做完标注）" y; rc=$?
    (( rc != 0 )) && { info "没有建立"; return 0; }
  fi
  local name
  step "② 数据集名字"
  while true; do
    ask "菌落数据集名字（直接回车用自动名字）" "colony_$(stamp)" || return 10
    valid_name "${ANS}" && break
  done
  name="${ANS}"
  local DIR="${TRAINSETS}/${name}" REC
  mkdir -p "${DIR}"
  TR_RUN_DIR="${DIR}"
  REC="${DIR}/RUN_RECORD.md"
  new_record "${REC}" "菌落数据集 ${name}"
  step "③ 预处理（颜色校正、尺度统一、自动截图、质控拼图）"
  local -a CB=(01_colony_preprocess.py build --root "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" --out "${DIR}" --views O "${WBOPT[@]}")
  run_py_colony "${REC}" "${CB[@]}" || die "01_colony_preprocess.py build 出错"
  [[ -f "${DIR}/manifest.csv" ]] || die "没有生成 manifest.csv"
  step "数据集信息"
  tool train_review.py write-colony-info --dir "${DIR}" --root "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" \
    --command "${COLONY_PY} ${(j: :)${(q-)CB[@]}}" || die "写 DATASET_INFO.json 失败"
  cp train_config.zsh "${DIR}/" 2>/dev/null
  TR_RUN_DIR=""
  milestone "${BG_GRN}" "💾" "菌落数据集已保存：${DIR}"
  print -r -- "${YEL}👀 建议肉眼看一遍质控拼图：${DIR}/qc_sheets/（颜色校正是否正常、截图是否落在菌落上）${R}"
  COLONY_BUILT="${name}"
  return 0
}

# =============================================================================
MICRO_BUILT="" COLONY_BUILT=""
case "$1" in
  micro)  build_micro ;;
  colony) build_colony ;;
  "")
    build_micro
    print -r -- ""
    ask_yn "接着建立菌落数据集？" y && build_colony ;;
  *) die "不认识的参数 $1（可用：micro / colony）" ;;
esac

[[ -n "${MICRO_BUILT}" ]] && next_step "zsh train_5_micro.zsh" "训练显微模型（默认用最新的数据集 ${MICRO_BUILT}）"
[[ -n "${COLONY_BUILT}" ]] && next_step "zsh train_6_colony.zsh" "训练菌落模型（默认用最新的数据集 ${COLONY_BUILT}）"
[[ -z "${MICRO_BUILT}${COLONY_BUILT}" ]] && info "本次没有建立新的数据集"
exit 0
