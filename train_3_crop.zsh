#!/bin/zsh
# =============================================================================
# train_3_crop.zsh —— 第 3 步：显微原图截图 + 菌落照片几何标注 + 菌落白平衡确认（交互窗口）
#
# 用法（fungal 环境，Audit 文件夹）：
#     zsh train_3_crop.zsh             先显示进度，再从菜单选
#     zsh train_3_crop.zsh micro       直接进入显微截图
#     zsh train_3_crop.zsh annotate    直接进入菌落几何标注
#     zsh train_3_crop.zsh wb          直接进入菌落白平衡确认
#
# 显微截图：列出 原图_重截 里还没截完的菌株，用编号选（可多选），逐株打开 01_crop_batch_training.py；
#           截图存到 CNN/crops_v2/，记录在 crops_v2/crop_log.csv（中途按 q 退出，下次从断点继续）
# 菌落标注：01_colony_annotate.py，只标正面（O）：平皿边界 → 菌落中心 → 菌落边缘
# 白平衡：  01_colony_preprocess.py pick，点选 / 拖框选白纸参照；a 培养基模式；x 无参照
# 标注和白平衡都存在 output/01_trainsets/colony_annotations/，与临床验证的标注分开
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "显微截图、菌落标注、白平衡"
need_scripts 00_train_check.py 01_crop_batch_training.py 01_colony_annotate.py 01_colony_preprocess.py
check_python
ANN="${TRAINSETS}/${COLONY_ANN}"
PEND="${LOG_DIR}/.crop_pending_$$.tsv"
# 注意：不能用 trap … EXIT（会顶掉 train_lib.zsh 的结束提示）；临时文件在结尾删除

micro_status() {
  tool 00_train_check.py crop-status --taxonomy "${TAXONOMY}" --recrop_root "${RECROP_ROOT}" \
    --v2_log "${CROPS_V2_ROOT}/crop_log.csv" --tsv "${PEND}"
}
colony_status() {
  tool 00_train_check.py colony-status --colony_raw "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" --kv "${PEND}.kv"
}

# ---------------------------------------------------------------- 显微截图
do_micro() {
  step "显微截图：原图_重截 → crops_v2"
  micro_status > /dev/null
  local -a SP ST PA NI ND NC PF
  local sp st pa ni nd nc pf i
  while IFS=$'\t' read -r sp st pa ni nd nc pf; do
    SP+=("$sp"); ST+=("$st"); PA+=("$pa"); NI+=("$ni"); ND+=("$nd"); NC+=("$nc"); PF+=("$pf")
  done < "${PEND}"
  if (( ${#SP} == 0 )); then
    ok "原图_重截 里的菌株都已截完（或还没放原图）"
    hint "新原图先用 zsh train_2_audit.zsh <文件夹> 加入"
    return 0
  fi
  print -r -- "${B}还没截完的菌株：${R}"
  for (( i = 1; i <= ${#SP}; i++ )); do
    printf "  %3d) %-10s %-16s 原图 %4d   已看 %4d   截图 %4d\n" $i "${SP[$i]}" "${ST[$i]}" ${NI[$i]} ${ND[$i]} ${NC[$i]}
  done
  print -r -- "${DIM}  （已看 = 截完按 n 或按 s 跳过的原图）${R}"
  ask_nums "要截哪些菌株？（如 1,3 或 2-5；a = 全部）" ${#SP} "a" || return 10
  local -a pick=("${NUMS[@]}")
  print -r -- ""
  print -r -- "${YEL}🖱️  截图窗口按键：左键 截取 | + / - 或右键 调框 | ← → 旋转 | 1 完整 2 局部 3 多个 | i 成熟/未成熟${R}"
  print -r -- "${YEL}              u 撤销 | n / 空格 下一张 | b 上一张 | s 跳过本图 | q 保存并退出（下次从断点继续）${R}"
  local k=0 j
  for j in "${pick[@]}"; do
    (( k++ ))
    milestone "${BG_MAG}" "🔬" "截图 ${k}/${#pick}：${SP[$j]} / ${ST[$j]}（${NI[$j]} 张原图，已看 ${ND[$j]}）"
    run_py - 01_crop_batch_training.py --folder "${PA[$j]}" --label "${SP[$j]}" --code_prefix "${PF[$j]}-${ST[$j]}" \
      --out_root "${CROPS_V2_ROOT}" || { err "截图工具异常退出（已截的不会丢失，重新运行即可继续）"; }
    if (( k < ${#pick} )); then
      ask_yn "继续截下一个菌株（${SP[${pick[$((k+1))]}]} / ${ST[${pick[$((k+1))]}]}）？" y; local rc=$?
      (( rc != 0 )) && break
    fi
  done
  step "显微截图进度"
  micro_status | tail -2
  return 0
}

# ---------------------------------------------------------------- 菌落标注
do_annotate() {
  step "菌落几何标注（只标正面 O）"
  check_colony_python || return 1
  local NEED_ANN=0 NEED_WB=0 READY=0 PHOTO_O=0
  colony_status; kv_read cat "${PEND}.kv"
  if (( PHOTO_O == 0 )); then warn "${COLONY_RAW_ROOT} 里没有符合命名的菌落正面照片"; hint "用 zsh train_2_audit.zsh <文件夹> 加入"; return 0; fi
  if (( NEED_ANN == 0 )); then ok "正面照片都已标注"; return 0; fi
  info "待标注 ${NEED_ANN} 张"
  print -r -- "${YEL}🖱️  Enter 确认每一步 | Backspace 上一步 | 方向键 / + - 微调平皿 | f 菌落铺满 | m 多菌落${R}"
  print -r -- "${YEL}    c 污染/混合 → 剔除 | b 图像不合格 → 剔除 | n 跳过 | q / Esc 退出（已保存的不会丢失）${R}"
  run_py_colony - 01_colony_annotate.py --root "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" --annotator "${ANNOTATOR}" --only=-B-O- \
    || err "标注工具异常退出（已保存的不会丢失，重新运行即可继续）"
  colony_status
  return 0
}

# ---------------------------------------------------------------- 白平衡
do_wb() {
  step "菌落白平衡参照确认（正面 O）"
  check_colony_python || return 1
  local NEED_ANN=0 NEED_WB=0 READY=0 PHOTO_O=0
  colony_status; kv_read cat "${PEND}.kv"
  if (( NEED_WB == 0 )); then
    ok "已标注的正面照片都已确认白平衡"
    (( NEED_ANN )) && warn "还有 ${NEED_ANN} 张没标注：先做菌落标注（标注后才能确认白平衡）"
    return 0
  fi
  info "待确认 ${NEED_WB} 张"
  print -r -- "${YEL}🖱️  先切到英文输入法、点一下窗口 | 点击/拖框 选白纸 | a 白纸/培养基模式 | t 文字遮罩 | x 无参照${R}"
  print -r -- "${YEL}    r 恢复建议 | z 撤销 | Enter 保存并下一张 | p 上一张 | n 跳过 | q / Esc 退出${R}"
  run_py_colony - 01_colony_preprocess.py pick --root "${COLONY_RAW_ROOT}" --ann_dir "${ANN}" --views O --annotator "${ANNOTATOR}" \
    || err "白平衡工具异常退出（已保存的不会丢失，重新运行即可继续）"
  colony_status
  return 0
}

# ---------------------------------------------------------------- 入口
case "$1" in
  micro)    do_micro; rm -f "${PEND}" "${PEND}.kv"; next_step "zsh train_4_dataset.zsh"; exit 0 ;;
  annotate) do_annotate; rm -f "${PEND}" "${PEND}.kv"; next_step "zsh train_3_crop.zsh wb" "确认白平衡"; exit 0 ;;
  wb)       do_wb; rm -f "${PEND}" "${PEND}.kv"; next_step "zsh train_4_dataset.zsh"; exit 0 ;;
  "")       ;;
  *)        die "不认识的参数 $1（可用：micro / annotate / wb）" ;;
esac

while true; do
  step "当前进度"
  print -r -- "${B}【显微原图 → 截图】${R}"; micro_status | tail -1
  print -r -- "${B}【菌落照片】${R}"; colony_status
  print -r -- ""
  print -r -- "   1) 显微截图"
  print -r -- "   2) 菌落几何标注"
  print -r -- "   3) 菌落白平衡确认"
  print -r -- "   4) 菌落：标注 → 白平衡 依次做"
  print -r -- "   5) 完成，去建数据集"
  ask_one "做哪一项？（Q 退出）" 5 5 || continue
  case ${ANS} in
    1) do_micro ;;
    2) do_annotate ;;
    3) do_wb ;;
    4) do_annotate && do_wb ;;
    5) break ;;
  esac
done

NEED_ANN=0 NEED_WB=0 READY=0 PHOTO_O=0
colony_status > /dev/null; kv_read cat "${PEND}.kv"
(( NEED_ANN + NEED_WB > 0 )) && warn "还有 $(( NEED_ANN + NEED_WB )) 张菌落正面照片没标注或没确认白平衡：建菌落数据集时它们不会进入主分析"
rm -f "${PEND}" "${PEND}.kv"
next_step "zsh train_4_dataset.zsh" "先建显微数据集，再建菌落数据集"
