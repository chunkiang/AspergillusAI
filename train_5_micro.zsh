#!/bin/zsh
# =============================================================================
# train_5_micro.zsh —— 第 5 步：训练显微模型（菌株级交叉验证 / 最终模型）
#
# 用法（fungal 环境，Audit 文件夹）：  zsh train_5_micro.zsh
#   ① 选 backbone（可多选，如 1,2）
#   ② 选数据集（列出 output/01_trainsets 下全部显微数据集，默认最新的）
#   ③ 选模式：1 只做交叉验证  2 只训练最终模型（需已有 CV）  3 先 CV 再最终模型
#
# 交叉验证：02_train.py --mode cv（StratifiedGroupKFold，按 cv_group 分组），之后自动做错误分析、混淆矩阵、按来源拆分
# 最终模型：02_train.py --mode full，epoch 数 = 该 backbone CV 各折最佳 epoch 的中位数（事先确定，不看外部数据）；
#           之后用 03_sanity_check.py 在训练截图上自检（应 ≥ 0.95）
#
# 输出：output/02_training_runs/<数据集名>/<backbone>_cv/          （中断后重跑：已完成的折跳过）
#       output/02_training_runs/<数据集名>/<backbone>_full_final/  （已有 final.pt 则跳过）
#       每个模型文件夹：TRAIN_SUMMARY.md（总结）、RUN_RECORD.md（命令与脚本指纹）、train_terminal.log
#       output/02_training_runs/<数据集名>/cv_汇总.txt                各 backbone 对比
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "训练显微模型"
need_scripts 02_train.py 02b_analyze_errors.py 02b_plot_confusion.py 02b_oof_by_site.py 03_sanity_check.py train_review.py
check_python
LST="${LOG_DIR}/.micro_ds_$$.tsv"

stage=1
while true; do
  case ${stage} in
  1) # ---------------------------------------------------------- ① backbone
    step "① 选择 backbone（ImageNet 预训练，torchvision）"
    for (( i = 1; i <= ${#MICRO_BACKBONES}; i++ )); do
      bb=${MICRO_BACKBONES[$i]}
      printf "   %2d) %-16s%s\n" $i "${bb}" "$( (( ${DEFAULT_BACKBONES[(Ie)$bb]} )) && print -r -- '（默认）')"
    done
    def=(); for bb in "${DEFAULT_BACKBONES[@]}"; do def+=(${MICRO_BACKBONES[(ie)$bb]}); done
    ask_nums "训练哪些 backbone？（可多选，如 1,2；a = 全部）" ${#MICRO_BACKBONES} "${(j:,:)def}" || continue
    BBS=(); for i in "${NUMS[@]}"; do BBS+=(${MICRO_BACKBONES[$i]}); done
    ok "backbone：${BBS[*]}"
    stage=2 ;;

  2) # ---------------------------------------------------------- ② 数据集
    step "② 选择显微数据集（按创建时间，最新的在最上面）"
    tool train_review.py datasets --kind micro --tsv "${LST}"
    DSS=("${(@f)$(<"${LST}")}")
    [[ -z "${DSS[1]}" ]] && die "output/01_trainsets 下没有显微数据集：先运行 zsh train_4_dataset.zsh"
    ask_one "用哪个数据集？" ${#DSS} 1 || { stage=1; continue; }
    DSDIR="${DSS[$ANS]}"; DS="${DSDIR:t}"
    ok "数据集：${DS}"
    if grep -q ',roi_ext,' "${DSDIR}/trainset_micro.csv" 2>/dev/null; then
      print -r -- "${BG_RED}  🔴 这个数据集含外部 ROI（roi_ext）：训练出的模型不能在 roi_test 上评估  ${R}"
    fi
    stage=3 ;;

  3) # ---------------------------------------------------------- ③ 模式
    step "③ 训练模式"
    print -r -- "   1) 只做 ${CV_FOLDS} 折交叉验证（选模型、报告内部性能）"
    print -r -- "   2) 只训练最终模型（需要该 backbone 已完成交叉验证）"
    print -r -- "   3) 先交叉验证，再训练最终模型"
    ask_one "选哪种？" 3 3 || { stage=2; continue; }
    MODE=${ANS}
    stage=4 ;;

  4) # ---------------------------------------------------------- 确认
    RUNS="${RUNS_ROOT}/${DS}"
    step "确认训练计划"
    print -r -- "   数据集：${DS}（${DSDIR}）"
    print -r -- "   输出：  ${RUNS}/"
    print -r -- "   参数：  folds=${CV_FOLDS} max_epochs=${MAX_EPOCHS} patience=${PATIENCE} batch=${BATCH} img_size=${IMG_SIZE} workers=${WORKERS} seed=${SEED} ${MICRO_EXTRA_ARGS[*]}"
    blocked=0
    for bb in "${BBS[@]}"; do
      line="   ${B}${bb}${R}："
      if (( MODE != 2 )); then
        [[ -f "${RUNS}/${bb}_cv/cv_summary.json" ]] && line+="CV 已完成（跳过）" \
          || { [[ -d "${RUNS}/${bb}_cv" ]] && line+="CV 续跑（已完成的折跳过）" || line+="CV 新训练"; }
      fi
      if (( MODE != 1 )); then
        (( MODE == 3 )) && line+=" → "
        if [[ -f "${RUNS}/${bb}_full_final/final.pt" ]]; then line+="最终模型已存在（跳过）"
        elif [[ -f "${RUNS}/${bb}_cv/cv_summary.json" ]]; then
          e=$("${PYTHON}" -c "import json,sys; print(max(1, round(json.load(open(sys.argv[1]))['median_best_epoch'])))" "${RUNS}/${bb}_cv/cv_summary.json")
          line+="最终模型 ${e} 个 epoch"
        elif (( MODE == 3 )); then line+="最终模型（epoch 数等 CV 完成后确定）"
        else line+="${RED}❌ 没有 CV 结果，不能训练最终模型${R}"; blocked=1; fi
      fi
      print -r -- "${line}"
    done
    if (( blocked )); then
      warn "有 backbone 还没做交叉验证：返回上一步改成模式 3，或先做 CV"
      stage=3; continue
    fi
    ask_yn "开始训练？（训练期间电脑不会休眠）" y; rc=$?
    (( rc == 10 )) && { stage=3; continue; }
    (( rc == 1 )) && { info "没有训练"; rm -f "${LST}"; exit 0; }
    break ;;
  esac
done
rm -f "${LST}"

# =============================================================================
keep_awake
mkdir -p "${RUNS}"
TRAIN_ARGS=(--trainset "${DSDIR}" --batch "${BATCH}" --img_size "${IMG_SIZE}" --workers "${WORKERS}" --seed "${SEED}" "${MICRO_EXTRA_ARGS[@]}")
RESULTS=()

rec_of() {   # rec_of <模型文件夹>：RUN_RECORD.md（没有就建）
  local r="$1/RUN_RECORD.md"
  [[ -f "$r" ]] || new_record "$r" "训练记录 ${1:t}（数据集 ${DS}）"
  print -r -- "## $(now) ${WRAPPER} 运行" >> "$r"
  print -r -- "" >> "$r"
  REC="$r"
}

run_cv() {
  local bb=$1 DIR="${RUNS}/$1_cv"
  milestone "${BG_MAG}" "🧪" "${bb}：${CV_FOLDS} 折交叉验证  →  ${DIR}"
  if [[ -f "${DIR}/cv_summary.json" ]]; then
    info "cv_summary.json 已存在，跳过训练（要重训，先把 ${DIR:t} 改名挪走）"
  else
    mkdir -p "${DIR}"; rec_of "${DIR}"
    TR_TEE="${DIR}/train_terminal.log" run_py "${REC}" 02_train.py --backbone "${bb}" --mode cv \
      --folds "${CV_FOLDS}" --epochs "${MAX_EPOCHS}" --patience "${PATIENCE}" "${TRAIN_ARGS[@]}" --resume_dir "${DIR}" \
      || { err "${bb} 交叉验证出错：已完成的折会保留，重跑本脚本即可续跑"; RESULTS+=("❌ ${bb}_cv 出错"); return 1; }
  fi
  rec_of "${DIR}"
  step "${bb}：训练后分析（出错只警告，不影响训练结果）"
  run_py "${REC}" 02b_analyze_errors.py "${DIR}" || warn "02b_analyze_errors.py 出错"
  [[ -f "${DIR}/confusion.csv" ]] && { run_py "${REC}" 02b_plot_confusion.py "${DIR}/confusion.csv" || warn "02b_plot_confusion.py 出错"; }
  run_py "${REC}" 02b_oof_by_site.py --run "${DIR}" --trainset "${DSDIR}" || warn "02b_oof_by_site.py 出错"
  step "${bb}：交叉验证总结"
  tool train_review.py summary --run "${DIR}"
  RESULTS+=("✅ ${bb}_cv：$("${PYTHON}" -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"macro-F1 {s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f}，acc {s['acc_mean']:.3f} ± {s['acc_sd']:.3f}\")" "${DIR}/cv_summary.json" 2>/dev/null)")
  return 0
}

run_full() {
  local bb=$1 CV="${RUNS}/$1_cv" DIR="${RUNS}/$1_full_final" e
  milestone "${BG_MAG}" "🏗️" "${bb}：最终模型（全部数据）  →  ${DIR}"
  [[ -f "${CV}/cv_summary.json" ]] || { err "没有 ${CV}/cv_summary.json，不能确定 epoch 数"; RESULTS+=("❌ ${bb}_full_final：缺 CV"); return 1; }
  e=$("${PYTHON}" -c "import json,sys; print(max(1, round(json.load(open(sys.argv[1]))['median_best_epoch'])))" "${CV}/cv_summary.json")
  info "epoch 数 = ${e}（${CV:t}/cv_summary.json 的 median_best_epoch）"
  if [[ -f "${DIR}/final.pt" ]]; then
    info "final.pt 已存在，跳过训练（要重训，先把 ${DIR:t} 改名挪走）"
  else
    mkdir -p "${DIR}"; rec_of "${DIR}"
    print -r -- "- epoch 数 ${e} 取自 ${CV}/cv_summary.json 的 median_best_epoch" >> "${REC}"
    TR_TEE="${DIR}/train_terminal.log" run_py "${REC}" 02_train.py --backbone "${bb}" --mode full --epochs "${e}" \
      "${TRAIN_ARGS[@]}" --resume_dir "${DIR}" \
      || { err "${bb} 最终模型训练出错"; RESULTS+=("❌ ${bb}_full_final 出错"); return 1; }
    [[ -f "${DIR}/final.pt" ]] || { err "训练结束但没有 ${DIR}/final.pt"; RESULTS+=("❌ ${bb}_full_final 无 final.pt"); return 1; }
  fi
  rec_of "${DIR}"
  sha "${DIR}/final.pt" | tee "${DIR}/model_sha256.txt" | sed 's/^/   final.pt sha256 /'
  step "${bb}：自检（训练截图抽样，应 ≥ 0.95）"
  rm -f "${DIR}/sanity_check.txt"
  TR_TEE="${DIR}/sanity_check.txt" run_py "${REC}" 03_sanity_check.py --models "${DIR}" --trainset "${DSDIR}" \
    || warn "03_sanity_check.py 出错（模型可能载入失败，外部验证前务必查明）"
  step "${bb}：最终模型总结"
  tool train_review.py summary --run "${DIR}"
  local acc=$(grep -oE '准确率 [0-9.]+' "${DIR}/sanity_check.txt" 2>/dev/null | tail -1)
  if [[ -n "${acc}" ]] && (( ${acc#准确率 } < 0.95 )); then
    RESULTS+=("⚠️  ${bb}_full_final：自检${acc#准确率}（< 0.95，需查明原因）")
    print -r -- "${BG_RED}  ⚠️ ${bb} 自检准确率 ${acc#准确率 } < 0.95：模型载入 / 类别顺序 / 预处理可能有问题，外部验证结果不可信  ${R}"
  else
    RESULTS+=("✅ ${bb}_full_final：自检${acc#准确率}")
  fi
  return 0
}

for bb in "${BBS[@]}"; do
  if (( MODE != 2 )); then run_cv "${bb}" || continue; fi
  (( MODE != 1 )) && run_full "${bb}"
done

# =============================================================================
step "全部结果"
if ls "${RUNS}"/*_cv/cv_summary.json >/dev/null 2>&1; then
  tool train_review.py cvtable --root "${RUNS}"
fi
print -r -- ""
for x in "${RESULTS[@]}"; do print -r -- "   $x"; done
if (( ${RESULTS[(I)❌*]} )); then
  milestone "${BG_RED}" "🧠" "有训练出错（见上面 ❌）；修好后重跑本脚本，已完成的会自动跳过"
  exit 1
fi
milestone "${BG_GRN}" "🧠" "显微模型训练完成：${RUNS}/"
info "每个模型文件夹里有 TRAIN_SUMMARY.md；以后可用 zsh train_review.zsh 查看"
(( MODE != 1 )) && hint "临床验证要用新模型：clin_config.zsh 里写 MICRO_MODELS=(${DS}/${BBS[1]}_full_final)，然后重跑 zsh clin_0_selftest.zsh"
next_step "zsh train_6_colony.zsh" "训练菌落模型；或 zsh train_review.zsh 查看模型"
