#!/bin/zsh
# =============================================================================
# methods_collect.zsh —— 为论文 Materials and Methods 收集需要核对的数字和参数（只读，不改任何原有文件）
#
# 用法（fungal 环境，Audit 文件夹）：
#     conda activate fungal
#     cd /Users/chunjiang/Documents/真菌AI/Audit
#     zsh methods_collect.zsh
#
# 依次输出 6 段：🧫 开发集株数  🗂️ 显微数据集  ⚙️ 显微训练参数  🎨 菌落模型  💻 软件与电脑  🏥 外部图像来源
# 最后 📋 汇总，并把结果存到 output/04_audit_review/methods_collect/<时间>/（另打一个 zip，方便发给 Claude）
#
# 放置：本文件和 00_methods_collect.py 直接放进 Audit 文件夹（与 train_config.zsh、train_lib.zsh 在一起）
# =============================================================================

# ---------------------------------------------------------------- 只需核对这 4 项
# 论文要报告的两个模型（与 clin_config.zsh 里临床验证用的模型应当相同；不同时程序会提醒）
MICRO_MODEL="runs_v4/swin_t_full_final"      # 显微最终模型：output/02_training_runs/ 下的位置
COLONY_MODEL="colony/20261020_2108"          # 菌落模型：output/02_training_runs/ 下的位置
MICRO_TRAINSET="trainset_v4"                 # 仅当模型的 args.json 里找不到数据集时才用
EXT_ROOT="/Users/chunjiang/Desktop/丝状真菌"  # 回顾性外部图像文件夹
# ----------------------------------------------------------------

TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
for f in train_config.zsh train_lib.zsh 00_methods_collect.py; do
  [[ -f "./$f" ]] || { print -r -- "❌ 缺少 $f：请把 methods_collect.zsh 和 00_methods_collect.py 放进 Audit 文件夹（与 train_config.zsh 在一起）"; exit 1; }
done
source ./train_config.zsh; source ./train_lib.zsh
tr_start "${0:t}" "收集论文 Methods 需要的数字和参数（只读）"
need_scripts 00_methods_collect.py
check_python

# 读 clin_config.zsh 里临床验证用的模型（只用来比对，不改它；在子 shell 里读，避免变量互相覆盖）
CLIN_MICRO=""; CLIN_COLONY=""
if [[ -f ./clin_config.zsh ]]; then
  CLIN_MICRO=$(zsh -c 'source ./clin_config.zsh 2>/dev/null; print -r -- "${MICRO_MODELS[1]}"' 2>/dev/null)
  CLIN_COLONY=$(zsh -c 'source ./clin_config.zsh 2>/dev/null; print -r -- "${COLONY_MODEL}"' 2>/dev/null)
  info "clin_config.zsh（临床验证）：显微 ${CLIN_MICRO:-—}；菌落 ${CLIN_COLONY:-—}"
else
  warn "Audit 里没有 clin_config.zsh，跳过与临床验证设置的比对"
fi

RUN_DIR="${AUDIT_REVIEW}/methods_collect/$(stamp)"
mkdir -p "${RUN_DIR}"
TR_RUN_DIR="${RUN_DIR}"
REC="${RUN_DIR}/RUN_RECORD.md"
new_record "${REC}" "Methods 资料收集"
{
  print -r -- "- 显微模型 MICRO_MODEL=${MICRO_MODEL}"
  print -r -- "- 菌落模型 COLONY_MODEL=${COLONY_MODEL}"
  print -r -- "- 备用数据集 MICRO_TRAINSET=${MICRO_TRAINSET}"
  print -r -- "- 外部图像 EXT_ROOT=${EXT_ROOT}"
  print -r -- ""
} >> "${REC}"

step "本次统计的对象"
info "🔬 显微模型：${RUNS_ROOT}/${MICRO_MODEL}"
info "🎨 菌落模型：${RUNS_ROOT}/${COLONY_MODEL}"
[[ -d "${RUNS_ROOT}/${MICRO_MODEL}" ]]  || warn "显微模型文件夹不存在，请改本文件开头的 MICRO_MODEL（可用 zsh train_review.zsh 选 3 查看全部模型）"
[[ -d "${RUNS_ROOT}/${COLONY_MODEL}" ]] || warn "菌落模型文件夹不存在，请改本文件开头的 COLONY_MODEL"

rc=0
run_py "${REC}" 00_methods_collect.py \
  --audit "${AUDIT}" --out_root "${OUT}" \
  --micro_model "${MICRO_MODEL}" --micro_trainset "${MICRO_TRAINSET}" \
  --colony_model "${COLONY_MODEL}" \
  --recrop_root "${RECROP_ROOT}" --colony_raw "${COLONY_RAW_ROOT}" --ext_root "${EXT_ROOT}" \
  --colony_python "${COLONY_PY}" \
  --clin_micro_model "${CLIN_MICRO}" --clin_colony_model "${CLIN_COLONY}" \
  --out "${RUN_DIR}" || rc=$?

cp ./train_config.zsh "${RUN_DIR}/copies/train_config.zsh" 2>/dev/null
[[ -f ./clin_config.zsh ]] && cp ./clin_config.zsh "${RUN_DIR}/copies/clin_config.zsh" 2>/dev/null
[[ -f ./03_clin_fusion.py ]] && cp ./03_clin_fusion.py "${RUN_DIR}/copies/03_clin_fusion.py" 2>/dev/null && ok "已附上 03_clin_fusion.py 和 clin_config.zsh（核对融合公式用，清单第 7 项）"

if [[ -f "${RUN_DIR}/METHODS_INFO.md" ]]; then
  ZIP="${RUN_DIR:h}/methods_collect_${RUN_DIR:t}.zip"
  ( cd "${RUN_DIR:h}" && zip -qr "${ZIP}" "${RUN_DIR:t}" ) && milestone "${BG_GRN}" "📦" "压缩包（发给 Claude）：${ZIP}"
  TR_RUN_DIR=""          # 有结果就保留文件夹，即使个别段落出错
  if (( rc != 0 )); then
    warn "有段落出错（见上面的 ❌），其余段落的结果仍然可用"
    rc=0
  fi
  next_step "把上面的压缩包上传到项目，并填好 external_sources_待填.csv" "不想传整个包时，只传 METHODS_INFO.md 也可以"
fi
exit ${rc}
