#!/bin/zsh
# =============================================================================
# train_2_audit.zsh —— 第 2 步：加入新图片（可选）+ 审计全部训练图片（不改动、不删除任何原图）
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh train_2_audit.zsh                         只审计；开始时会问有没有新图片要加入
#     zsh train_2_audit.zsh <文件夹> [<文件夹>…]     先加入这些文件夹的图片，再审计（每个文件夹会问是显微还是菌落）
#     zsh train_2_audit.zsh full                    强制重跑 00_audit_train.py（平时只在 CNN/dataset、CNN/normal 有变化时才跑）
#
# 加入新图片（只复制，原文件夹不动）：
#   显微镜下图片 → 原图_重截/<菌种>/<菌株>/        菌种从 taxonomy_map.csv 的列表里选，菌株号自己填
#   菌落图片     → 补拍_2026/<组>/                 文件名必须是 PUPH-FUM-001-B-O-1.jpg 这种格式
#   目标里已有相同的图自动跳过；有同名但内容不同的图则一张都不复制，并提示怎么改
#
# 审计（00_train_check.py audit）：打不开 / 损坏 / 0 字节 / 太小 / 菌种名不对 / 菌株号重复 /
#   跨菌种或跨菌株的重复图 / 菌落命名与分组 / 截图记录与文件是否一致；每类问题都附解决建议
#   CNN/dataset、CNN/normal 有新文件时另跑 00_audit_train.py（重复图、菌株分组推断）+ 00_clean_train.py（预演）
#
# 输出：output/04_audit_review/train_check/<时间>/report.txt、problems.csv
#       output/04_audit_review/train_audit/import_log.csv（每次复制了哪些图）
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "加入新图片 + 审计训练图片"
need_scripts 00_train_check.py 00_audit_train.py 00_clean_train.py
check_python

FORCE_FULL=0
SRCS=()
for x in "$@"; do [[ "$x" == full ]] && FORCE_FULL=1 || SRCS+=("$x"); done
IMPORT_LOG="${AUDIT_REVIEW}/train_audit/import_log.csv"
N_IMPORTED=0

# ---------------------------------------------------------------- 菌种列表
SPECIES=() SPCLASS=()
while IFS=$'\t' read -r sp tc; do SPECIES+=("$sp"); SPCLASS+=("$tc"); done < <(tool 00_train_check.py species --taxonomy "${TAXONOMY}")
show_species() {
  local i
  for (( i = 1; i <= ${#SPECIES}; i++ )); do
    printf "  %3d) %-12s %s\n" $i "${SPECIES[$i]}" "${DIM}${SPCLASS[$i]}${R}"
  done
}

# ---------------------------------------------------------------- 加入一个文件夹
import_one() {   # import_one <文件夹>；返回 0 完成/跳过，10 = 用户按 B
  local src="${1:A}" kind sp strain mode
  local -a imp
  if [[ ! -d "${src}" ]]; then err "找不到文件夹 ${1}"; hint "路径里有空格时，把文件夹直接拖进终端即可"; return 0; fi
  local DIRECT=0 SUBS=0 TOTAL=0 COLONY_NAMED=0 SUBNAMES=""
  kv_read tool 00_train_check.py inspect --src "${src}"
  milestone "${BG_MAG}" "📂" "新文件夹：${src}"
  info "共 ${TOTAL} 张图片：直接放在里面 ${DIRECT} 张；有图片的子文件夹 ${SUBS} 个${SUBNAMES:+（${SUBNAMES//|/、}）}"
  (( TOTAL == 0 )) && { warn "这个文件夹里没有图片，跳过"; return 0; }
  local guess=1; (( COLONY_NAMED * 2 > TOTAL )) && guess=2
  (( COLONY_NAMED )) && info "其中 ${COLONY_NAMED} 张符合菌落命名（PUPH-XXX-000-B-O/R-n）"

  while true; do
    print -r -- "   1) 显微镜下图片（乳酸酚棉蓝等，复制到 原图_重截，之后截图）"
    print -r -- "   2) 菌落图片（平皿照片，复制到 补拍_2026，之后标注和白平衡）"
    ask_one "这个文件夹里是哪种图片？" 2 ${guess} || return 10
    kind=${ANS}

    if [[ ${kind} == 2 ]]; then
      step "检查菌落照片（预演，不复制）"
      run_py - 00_train_check.py import-colony --src "${src}" --colony_raw "${COLONY_RAW_ROOT}" --log "${IMPORT_LOG}" --dry_run
      [[ $? -eq 2 ]] && { err "这个文件夹没有加入（见上面的 ❌ 和 💡）"; return 0; }
      ask_yn "确认复制到 ${COLONY_RAW_ROOT}/<组>/ ？" y; local rc=$?
      (( rc == 10 )) && continue
      (( rc == 1 )) && { info "没有复制"; return 0; }
      run_py - 00_train_check.py import-colony --src "${src}" --colony_raw "${COLONY_RAW_ROOT}" --log "${IMPORT_LOG}" \
        && { (( N_IMPORTED++ )); ok "菌落照片已加入"; } || err "复制出错（见上面）"
      return 0
    fi

    # ---- 显微
    while true; do
      print -r -- "${B}本期训练的菌种（taxonomy_map.csv）：${R}"
      show_species
      ask_one "这些图是哪个菌种？（输入编号）" ${#SPECIES} || continue 2
      sp="${SPECIES[$ANS]}"
      info "菌种：${sp} → 训练类 ${SPCLASS[$ANS]}"

      imp=(00_train_check.py import-micro --src "${src}" --species "${sp}" --recrop_root "${RECROP_ROOT}"
           --taxonomy "${TAXONOMY}" --log "${IMPORT_LOG}")
      if (( SUBS > 0 && DIRECT == 0 )); then
        ask_yn "每个子文件夹是一个菌株吗？（是：子文件夹名 = 菌株号；否：整个文件夹算一个菌株）" y; local rc=$?
        (( rc == 10 )) && continue
        (( rc == 0 )) && mode=sub || mode=one
      elif (( SUBS > 0 && DIRECT > 0 )); then
        warn "既有直接放的图片，又有子文件夹"
        print -r -- "   1) 子文件夹 = 菌株（直接放着的 ${DIRECT} 张不复制）"
        print -r -- "   2) 整个文件夹算一个菌株"
        ask_one "怎么处理？" 2 1 || continue
        [[ ${ANS} == 1 ]] && mode=sub || mode=one
      else
        mode=one
      fi
      if [[ ${mode} == one ]]; then
        while true; do
          ask "菌株号（会成为 原图_重截/${sp}/<菌株号>/；同一菌株的图放在一起，交叉验证按它分组）" "${src:t}" || continue 2
          strain="${ANS}"
          [[ "${strain}" == */* || -z "${strain}" ]] && { warn "菌株号不能为空，也不能含 /"; continue; }
          break
        done
        imp+=(--strain "${strain}")
      else
        imp+=(--subfolders)
      fi
      break
    done

    step "检查显微图片（预演，不复制）"
    run_py - "${imp[@]}" --dry_run
    [[ $? -eq 2 ]] && { err "这个文件夹没有加入（见上面的 ❌ 和 💡）"; return 0; }
    ask_yn "确认复制？" y; local rc=$?
    (( rc == 10 )) && continue
    (( rc == 1 )) && { info "没有复制"; return 0; }
    run_py - "${imp[@]}" && { (( N_IMPORTED++ )); ok "显微图片已加入 ${RECROP_ROOT}/${sp}/"; } || err "复制出错（见上面）"
    return 0
  done
}

# ---------------------------------------------------------------- ① 加入新图片
step "① 加入新图片"
if (( ${#SRCS} == 0 )); then
  while true; do
    ask "有新图片要加入吗？输入文件夹路径（可把文件夹拖进终端），直接回车跳过" "" || break
    [[ -z "${ANS}" ]] && break
    p="${(Q)ANS}"     # 去掉拖进终端时加的引号和反斜杠转义
    import_one "${p}"
    info "可以继续输入下一个文件夹；直接回车结束加入、开始审计"
  done
else
  for s in "${SRCS[@]}"; do
    import_one "$s"
  done
fi
(( N_IMPORTED )) && ok "本次加入了 ${N_IMPORTED} 个文件夹（复制记录 ${IMPORT_LOG}）" || info "本次没有加入新图片"

# ---------------------------------------------------------------- ② 审计
step "② 审计全部训练图片（只读）"
CK="${AUDIT_REVIEW}/train_check/$(date +%Y%m%d_%H%M%S)"
AUDIT_RC=0
run_py - 00_train_check.py audit --taxonomy "${TAXONOMY}" --recrop_root "${RECROP_ROOT}" \
  --colony_raw "${COLONY_RAW_ROOT}" --ann_dir "${TRAINSETS}/${COLONY_ANN}" \
  --crops_v2 "${CROPS_V2_ROOT}" --crops_new "${CROPS_NEW_ROOT}" --out "${CK}" \
  --min_raw "${AUDIT_MIN_RAW_SIDE}" --min_crop "${AUDIT_MIN_CROP_SIDE}" || AUDIT_RC=$?
(( AUDIT_RC == 2 )) && print -r -- "${YEL}   （退出码 2 = 审计发现了 ❌ 问题，不是脚本本身出错；先把后面的步骤做完再统一处理）${R}"
(( AUDIT_RC != 0 && AUDIT_RC != 2 )) && die "审计脚本异常退出"

# ---------------------------------------------------------------- ③ 旧截图库
step "③ 旧截图库（CNN/dataset、CNN/normal）：00_audit_train.py"
NEWER=0 MANIFEST_TIME=""
kv_read tool 00_train_check.py fresh --micro_root "${MICRO_DATASET_ROOT}" --colony_root "${COLONY_LEGACY_ROOT}"
if [[ ${FORCE_FULL} == 1 || "${NEWER}" == -1 || ${NEWER} -gt 0 ]]; then
  [[ "${NEWER}" == -1 ]] && info "还没有 train_manifest.csv，现在生成"
  (( NEWER > 0 )) && warn "上次审计（${MANIFEST_TIME}）之后 CNN/dataset、CNN/normal 里有 ${NEWER} 个新文件，重新审计"
  [[ ${FORCE_FULL} == 1 ]] && info "按 full 参数重新审计"
  run_py - 00_audit_train.py --micro_root "${MICRO_DATASET_ROOT}" --colony_root "${COLONY_LEGACY_ROOT}" \
    || die "00_audit_train.py 出错"
  run_py - 00_clean_train.py --micro_root "${MICRO_DATASET_ROOT}" --colony_root "${COLONY_LEGACY_ROOT}" \
    || warn "00_clean_train.py 预演出错（不影响后续）"
  hint "完全重复的副本要真正移到隔离目录，先看 review_cross_class.html，再手动运行：python 00_clean_train.py --apply"
else
  ok "上次审计（${MANIFEST_TIME}）之后没有变化，跳过 00_audit_train.py（要强制重跑：zsh train_2_audit.zsh full）"
fi

# ---------------------------------------------------------------- ④ 进度
step "④ 截图与菌落标注进度"
tool 00_train_check.py crop-status --taxonomy "${TAXONOMY}" --recrop_root "${RECROP_ROOT}" --v2_log "${CROPS_V2_ROOT}/crop_log.csv" | tail -1
tool 00_train_check.py colony-status --colony_raw "${COLONY_RAW_ROOT}" --ann_dir "${TRAINSETS}/${COLONY_ANN}"

if (( AUDIT_RC == 2 )); then
  milestone "${BG_RED}" "🔎" "审计发现 ❌ 错误：按上面每类问题下的 💡 改好后，重跑 zsh train_2_audit.zsh"
  info "完整报告 ${CK}/report.txt；逐条清单 ${CK}/problems.csv（Excel 可打开）"
  exit 1
fi
milestone "${BG_GRN}" "🔎" "审计通过（⚠️ 警告不阻止后续步骤，但建议看一眼）：${CK}/report.txt"
next_step "zsh train_3_crop.zsh" "截还没截的显微原图、标注菌落照片、确认白平衡"
