#!/bin/zsh
# =============================================================================
# clin_0_selftest.zsh —— 自检：不需要任何临床素材，用训练数据把整条流程真实跑一遍
#
# 用法（fungal 环境，Audit 文件夹）：  zsh clin_0_selftest.zsh
#
# 检查什么：
#   ① 脚本是否放在 Audit 文件夹、python 环境与依赖、所需脚本是否齐全
#   ② 菌落模型：锁定指纹核对 + 在训练株上预测（03_colony_predict.py --sanity），应接近全对
#   ③ 镜检模型：从训练清单每类抽 1 株（3 张截图）冒充临床菌株，
#      用真实的 03_eval_clinical.py 预测（与正式临床验证同一代码路径），应接近全对
#   ④ 融合：曲霉株借用菌落训练株的株号，用真实的 03_clin_fusion.py 融合，检查两边能对上
#   ⑤ 汇总判定：✅ 通过 / ⚠️ 警告 / ❌ 失败
#
# 输出：output/03_external_eval/clinical/_SELFTEST_<时间>/
#       SELFTEST_RESULT.txt（结论）、RUN_RECORD.md（用了哪些脚本和参数）、micro/、colony/、final/
# 自检结果只说明"流程和模型能正常工作"，训练数据上的准确率不能写进论文。
# 什么时候跑：第一次安装后、换了模型后、升级了 python 包后、正式锁定之前。
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "自检（用训练数据跑通整条流程）"
check_rules
keep_awake

ST="${EVAL_ROOT}/_SELFTEST_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ST}"
CLIN_RUN_DIR="${ST}"
REC="${ST}/RUN_RECORD.md"
{
  print -r -- "# 自检运行记录 ${ST:t}"
  print -r -- ""
  print -r -- "- 时间：$(now)；电脑：$(hostname)"
  print -r -- "- 镜检模型：${MICRO_MODELS[*]}；镜检训练清单（备用）：${MICRO_TRAINSET}"
  print -r -- "- 菌落模型：${COLONY_MODEL}；菌落训练预处理：${COLONY_TRAIN_PRE}；菌落 python：${COLONY_PY}"
  print -r -- "- 判读规则：P_MIN=${P_MIN} GAP_MIN=${GAP_MIN} ASP_MIN=${ASP_MIN} ALPHA=${ALPHA}"
  print -r -- ""
  print -r -- "## 依次运行的 python 脚本与参数"
  print -r -- ""
} > "${REC}"
info "自检结果文件夹 ${ST}"

# ---------------------------------------------------------------- ① 环境
step "① 环境与脚本"
check_python
python - <<'EOF'
import torch
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"   torch {torch.__version__}，计算设备 {dev}")
EOF
[[ $? -eq 0 ]] || warn "读取 torch 版本/设备失败（继续，镜检步骤会给出具体报错）"
for f in 03_roi_clinical.py 03_eval_clinical.py 03_eval_external.py 02_train.py paths.py \
         01_colony_annotate.py 01_colony_preprocess.py 02_colony_train.py 03_colony_predict.py \
         00_clin_check.py 00_clin_selftest.py 03_clin_fusion.py clin_sections.py; do
  [[ -f "$f" ]] || die "Audit 文件夹里缺少 $f"
done
ok "所需脚本齐全"
command -v shasum >/dev/null || warn "没有 shasum 命令（macOS 自带；Linux 会改用 sha256sum）"

# ---------------------------------------------------------------- ② 菌落
step "② 菌落模型：运行环境核对 + 锁定核对 + 训练株预测"
CPRE="${AUDIT}/output/01_trainsets/${COLONY_TRAIN_PRE}"
HAVE_COLONY=0
COLONY_ENV_OK=1
check_colony_env "${REC}" || { COLONY_ENV_OK=0; err "菌落运行环境与训练时不一致（按上面的提示改 clin_config.zsh 的 COLONY_PYTHON 后重跑自检）"; }
if [[ ${COLONY_ENV_OK} == 0 ]]; then
  :
elif [[ -f "${CPRE}/manifest.csv" ]]; then
  if run_py_colony "${REC}" 03_colony_predict.py --model "${COLONY_MODEL}" --pre "${CPRE}" --sanity \
       --out "${ST}/colony" --t1 "${P_MIN}" --t2 "${GAP_MIN}"; then
    HAVE_COLONY=1
  else
    err "菌落模型自检失败（看上面的报错：模型指纹不一致？预处理参数不一致？路径不对？）"
  fi
else
  warn "找不到菌落训练预处理 ${CPRE}/manifest.csv，跳过菌落自检（改 clin_config.zsh 的 COLONY_TRAIN_PRE）"
fi

# ---------------------------------------------------------------- ③ 镜检
step "③ 镜检模型：训练截图冒充临床菌株 → 03_eval_clinical.py"
PREP=(prepare --micro_model "${MICRO_MODELS[1]}" --trainset "${MICRO_TRAINSET}" --n_roi 3 --out "${ST}/roi")
[[ ${HAVE_COLONY} == 1 ]] && PREP+=(--colony_pred "${ST}/colony" --colony_pre "${CPRE}")
run_py "${REC}" 00_clin_selftest.py "${PREP[@]}" || die "准备自检数据失败"
run_py "${REC}" 03_eval_clinical.py --models "${MICRO_MODELS[@]}" --roi_dir "${ST}/roi" \
  --table "${ST}/roi/roi_clinical_table.csv" --out "${ST}/micro" --no_lock \
  --p_min "${P_MIN}" --gap_min "${GAP_MIN}" || die "镜检模型预测失败（这就是正式评估第②步会遇到的问题，把上面的报错发给 Claude）"

# ---------------------------------------------------------------- ④ 融合
step "④ 融合：03_clin_fusion.py"
FUSE=(--micro "${ST}/micro" --answers "${ST}/roi/answers_master.csv" --out "${ST}/final"
      --asp_min "${ASP_MIN}" --alpha "${ALPHA}" --p_min "${P_MIN}" --gap_min "${GAP_MIN}")
[[ ${HAVE_COLONY} == 1 ]] && FUSE+=(--colony "${ST}/colony")
run_py "${REC}" 03_clin_fusion.py "${FUSE[@]}" || die "融合失败"

# ---------------------------------------------------------------- ⑤ 判定
step "⑤ 汇总判定"
cp clin_config.zsh "${ST}/" 2>/dev/null
cp "${CLIN_LOG}" "${ST}/terminal.log" 2>/dev/null
VERIFY=(verify --dir "${ST}" --colony_pre "${CPRE}")
[[ ${COLONY_ENV_OK} == 0 ]] && VERIFY+=(--colony_env_failed)
run_py "${REC}" 00_clin_selftest.py "${VERIFY[@]}" \
  || die "自检未通过（见上面的 ❌ 项）"
print -r -- ""
print -r -- "${BG_GRN}  🩺 自检结论：${ST}/SELFTEST_RESULT.txt  ${R}"
