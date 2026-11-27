#!/bin/zsh
# =============================================================================
# train_6_colony.zsh —— 第 6 步：训练菌落模型（株级留一组交叉验证 → 选方案 → 温度缩放 → 全部株重训 → 锁定）
#
# 用法（菌落用 python 由 train_config.zsh 的 COLONY_PYTHON 指定，Audit 文件夹）：  zsh train_6_colony.zsh
#   ① 选菌落数据集（列出 output/01_trainsets 下全部菌落数据集，默认最新的）
#   ② 确认参数（train_config.zsh 的 COLONY_* 项）
#   ③ 02_colony_train.py：方案 A（CIELAB 颜色 + 直径 → 逻辑回归）、方案 B（冻结网络特征 → 逻辑回归），
#      预设规则选 OOF 株级 NLL 最低者，温度缩放后用全部株重训并锁定
#
# 输出：output/02_training_runs/colony/<时间，如 20261022_2215>/
#       metrics.txt（中文结果）、final_model.joblib、locked_config.json、confusion_*.png、oof_*.csv、
#       TRAIN_SUMMARY.md（总结）、RUN_RECORD.md（命令与脚本指纹）
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
TR_INTERACTIVE=1
tr_start "${0:t}" "训练菌落模型"
need_scripts 02_colony_train.py train_review.py
check_colony_python || die "菌落用 python 不可用（见上面）"
LST="${LOG_DIR}/.colony_ds_$$.tsv"

stage=1
SKIP_B=${COLONY_SKIP_B}
while true; do
  case ${stage} in
  1) # ---------------------------------------------------------- ① 数据集
    step "① 选择菌落数据集（按创建时间，最新的在最上面）"
    tool train_review.py datasets --kind colony --tsv "${LST}"
    DSS=("${(@f)$(<"${LST}")}")
    [[ -z "${DSS[1]}" ]] && die "output/01_trainsets 下没有菌落数据集：先运行 zsh train_4_dataset.zsh colony"
    ask_one "用哪个数据集？" ${#DSS} 1 || continue
    DSDIR="${DSS[$ANS]}"
    ok "数据集：${DSDIR:t}"
    stage=2 ;;

  2) # ---------------------------------------------------------- ② 参数
    step "② 参数（在 train_config.zsh 里改）"
    print -r -- "   COLONY_MIN_STRAINS = ${COLONY_MIN_STRAINS}     株数少于它的组本轮不纳入"
    print -r -- "   COLONY_DUP_INLIERS = ${COLONY_DUP_INLIERS}   疑似同一平皿的照片合并到同一 CV 组（0 = 不合并）"
    print -r -- "   COLONY_BACKBONE    = ${COLONY_BACKBONE}  方案 B 的冻结特征网络"
    print -r -- "   COLONY_SKIP_B      = ${SKIP_B}       1 = 只跑方案 A"
    print -r -- "   COLONY_RUN_C       = ${COLONY_RUN_C}       1 = 加跑方案 C（探索性，不参与选择）"
    [[ -n "${COLONY_WEIGHTS}" ]] && print -r -- "   COLONY_WEIGHTS     = ${COLONY_WEIGHTS}"
    [[ -n "${COLONY_REGISTRY}" ]] && print -r -- "   COLONY_REGISTRY    = ${COLONY_REGISTRY}"
    print -r -- "   python             = ${COLONY_PY}"
    if [[ ${SKIP_B} != 1 ]] && ! "${COLONY_PY}" -c "import torch, torchvision" 2>/dev/null; then
      warn "菌落用 python 里没有 torch，方案 B 会失败"
      ask_yn "只跑方案 A？（否 = 仍然尝试方案 B）" y; rc=$?
      (( rc == 10 )) && { stage=1; continue; }
      (( rc == 0 )) && SKIP_B=1
    fi
    STAMP=$(stamp)
    OUTDIR="${COLONY_RUNS}/${STAMP}"
    if [[ -e "${OUTDIR}" ]]; then
      warn "${OUTDIR} 已存在（同一分钟内刚训练过）。等一分钟后再确认"
      ask_yn "现在重试？" y; rc=$?; (( rc == 10 )) && { stage=1; continue; }; (( rc == 1 )) && exit 0
      continue
    fi
    print -r -- "   输出：${OUTDIR}"
    ask_yn "开始训练？" y; rc=$?
    (( rc == 10 )) && { stage=1; continue; }
    (( rc == 1 )) && { info "没有训练"; rm -f "${LST}"; exit 0; }
    break ;;
  esac
done
rm -f "${LST}"

# =============================================================================
keep_awake
mkdir -p "${OUTDIR}"
TR_RUN_DIR="${OUTDIR}"
REC="${OUTDIR}/RUN_RECORD.md"
new_record "${REC}" "菌落模型训练 ${STAMP}（数据集 ${DSDIR:t}）"
CT=(02_colony_train.py --pre "${DSDIR}" --out "${STAMP}" --min_strains "${COLONY_MIN_STRAINS}"
    --dup_inliers "${COLONY_DUP_INLIERS}" --backbone "${COLONY_BACKBONE}")
[[ ${SKIP_B} == 1 ]] && CT+=(--skip_b)
[[ ${COLONY_RUN_C} == 1 ]] && CT+=(--run_c)
[[ -n "${COLONY_WEIGHTS}" ]] && CT+=(--weights "${COLONY_WEIGHTS}")
[[ -n "${COLONY_REGISTRY}" ]] && CT+=(--registry "${COLONY_REGISTRY}")

milestone "${BG_MAG}" "🧫" "菌落模型训练  →  ${OUTDIR}"
TR_TEE="${OUTDIR}/train_terminal.log" run_py_colony "${REC}" "${CT[@]}" || die "02_colony_train.py 出错（看上面的报错）"
[[ -f "${OUTDIR}/locked_config.json" && -f "${OUTDIR}/final_model.joblib" ]] || die "训练结束但没有 locked_config.json / final_model.joblib"
cp train_config.zsh "${OUTDIR}/" 2>/dev/null
TR_RUN_DIR=""

step "菌落模型总结"
tool train_review.py summary --run "${OUTDIR}"
milestone "${BG_GRN}" "🧫" "菌落模型训练完成：${OUTDIR}"
hint "临床验证要用这个模型：clin_config.zsh 里写 COLONY_MODEL=\"colony/${STAMP}\"，然后重跑 zsh clin_0_selftest.zsh"
print -r -- "${YEL}🔒 如果临床验证已经正式锁定（clinical_lock.json），换模型就必须重新锁定，之前的正式结果不能混用${R}"
next_step "zsh train_review.zsh" "查看全部数据集和模型"
