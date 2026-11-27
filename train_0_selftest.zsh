#!/bin/zsh
# =============================================================================
# train_0_selftest.zsh —— 自检：环境、依赖、脚本、路径；默认再做一次约 2 分钟的冒烟训练
#
# 用法（fungal 环境，Audit 文件夹）：  zsh train_0_selftest.zsh
#
# 检查什么：
#   ① 脚本是否放在 Audit 文件夹、所需 python 脚本是否齐全
#   ② 显微用 python（fungal）：依赖、torch 版本、计算设备（MPS / CUDA / CPU）
#   ③ 菌落用 python（COLONY_PYTHON）：依赖、scikit-learn 版本、是否与 clin_config.zsh 一致、有无 torch（方案 B 需要）
#   ④ 图片和清单的位置是否存在、磁盘剩余空间
#   ⑤ 冒烟训练（SELFTEST_QUICK=0 时）：从最新的显微数据集抽 12 张，真实调用 02_train.py（cv + full，1 个 epoch，
#      不加载预训练权重）和 03_sanity_check.py；从最新的菌落数据集用 02_colony_train.py 跑方案 A
#   ⑥ 汇总判定：✅ 通过 / ⚠️ 警告 / ❌ 失败
#
# 输出：output/02_training_runs/_SELFTEST_<时间>/SELFTEST_RESULT.txt、RUN_RECORD.md
#       （以 _ 开头，不会被当作模型列出；冒烟训练的权重用完即删）
# 冒烟训练的准确率没有意义，只说明"整条训练代码能跑通"。
# 什么时候跑：第一次安装后、升级了 python 包后、换了电脑后。
# =============================================================================
TRAIN_KIT_DIR="${0:A:h}"; cd "${TRAIN_KIT_DIR}" || exit 1
source ./train_config.zsh; source ./train_lib.zsh
tr_start "${0:t}" "自检（环境、依赖、路径、冒烟训练）"

ST="${RUNS_ROOT}/_SELFTEST_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ST}"
TR_RUN_DIR="${ST}"
REC="${ST}/RUN_RECORD.md"
new_record "${REC}" "训练流程自检 ${ST:t}"
PASS=() WARNS=() FAILS=()
pass_() { ok "$1"; PASS+=("$1"); }
warn_() { warn "$1"; [[ -n "$2" ]] && hint "$2"; WARNS+=("$1${2:+ → $2}"); }
fail_() { err "$1"; [[ -n "$2" ]] && hint "$2"; FAILS+=("$1${2:+ → $2}"); }

if [[ "${SELFTEST_QUICK}" == 1 ]]; then
  print -r -- "${YEL}⏩ 快速模式（train_config.zsh 里 SELFTEST_QUICK=1）：跳过冒烟训练。要做完整自检，把它改成 0${R}"
else
  print -r -- "${YEL}🧪 完整模式：最后会做约 2 分钟的冒烟训练。想跳过，把 train_config.zsh 里的 SELFTEST_QUICK 改成 1${R}"
fi

# ---------------------------------------------------------------- ① 脚本
step "① 脚本"
SCRIPTS=(paths.py 00_train_check.py train_review.py 00_audit_train.py 00_clean_train.py 01_make_recrop_dirs.py
         01_crop_batch_training.py 01_build_trainset.py 01_colony_annotate.py 01_colony_preprocess.py
         02_train.py 02b_analyze_errors.py 02b_plot_confusion.py 02b_oof_by_site.py 03_sanity_check.py 03_eval_external.py
         02_colony_train.py 00_colony_inventory.py)
miss=()
for f in "${SCRIPTS[@]}"; do [[ -f "$f" ]] || miss+=("$f"); done
if (( ${#miss} )); then fail_ "Audit 文件夹里缺少：${miss[*]}" "从 script.zip 或项目里补回这些文件"
else pass_ "所需 python 脚本齐全（${#SCRIPTS} 个）"; fi
grep -q "exclude_sources" 01_build_trainset.py 2>/dev/null || fail_ "01_build_trainset.py 是旧版（没有 --exclude_sources）" "换成 2026-10-21 之后的版本"
[[ -f "${TAXONOMY}" ]] && pass_ "taxonomy_map.csv：$(($(wc -l < "${TAXONOMY}") - 1)) 行" \
                       || fail_ "找不到 ${TAXONOMY}" "taxonomy_map.csv 应放在 Audit 文件夹"

# ---------------------------------------------------------------- ② 显微 python
step "② 显微用 python（${PYTHON}）"
check_python
dev=$("${PYTHON}" - <<'EOF' 2>&1
import torch, torchvision, sklearn
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}，torchvision {torchvision.__version__}，scikit-learn {sklearn.__version__}，计算设备 {dev}")
EOF
)
if [[ $? -eq 0 ]]; then
  pass_ "${dev}"
  [[ "${dev}" == *"设备 cpu"* ]] && warn_ "没有可用的 GPU（MPS/CUDA），训练会很慢" "确认是在 Apple 芯片 Mac 上、torch 版本支持 MPS"
else
  fail_ "读取 torch 失败：${dev}" "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision"
fi
"${PYTHON}" -c "import pillow_heif" 2>/dev/null || warn_ "显微用 python 没有 pillow-heif：审计时 HEIC 照片无法检查" \
  "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow-heif"

# ---------------------------------------------------------------- ③ 菌落 python
step "③ 菌落用 python（${COLONY_PY}）"
COLONY_OK=1
if check_colony_python; then
  PASS+=("菌落用 python 依赖齐全")
  CLIN_CP=$(grep -E '^COLONY_PYTHON=' clin_config.zsh 2>/dev/null | head -1 | sed -E 's/^COLONY_PYTHON=//; s/^"//; s/"[[:space:]]*(#.*)?$//')
  if [[ -z "${CLIN_CP}" ]]; then
    info "没找到 clin_config.zsh 的 COLONY_PYTHON（没有临床验证脚本时可忽略）"
  elif [[ "${CLIN_CP:A}" == "${COLONY_PY:A}" ]]; then
    pass_ "与 clin_config.zsh 的 COLONY_PYTHON 一致"
  else
    warn_ "与 clin_config.zsh 的 COLONY_PYTHON 不一致（clin：${CLIN_CP}）" "两处应写同一个 python，否则菌落模型在临床验证时版本对不上"
  fi
  if ! "${COLONY_PY}" -c "import torch, torchvision" 2>/dev/null; then
    if [[ "${COLONY_SKIP_B}" == 1 ]]; then info "菌落 python 没有 torch；COLONY_SKIP_B=1，菌落训练只跑方案 A"
    else warn_ "菌落 python 没有 torch：菌落训练的方案 B 会失败（方案 A 不受影响）" \
               "装 torch（${COLONY_PY} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision），或把 COLONY_SKIP_B 改成 1"; fi
  fi
else
  COLONY_OK=0
  FAILS+=("菌落用 python 不可用或缺依赖 → 见上面的提示")
fi

# ---------------------------------------------------------------- ④ 路径
step "④ 图片与清单的位置"
for pair in "旧截图(dev_original/dev_checked):${MICRO_DATASET_ROOT}" "旧菌落图:${COLONY_LEGACY_ROOT}" \
            "网络补截(crops_new):${CROPS_NEW_ROOT}" "规范截图(crops_v2):${CROPS_V2_ROOT}" "外部ROI(roi_ext):${ROI_TEST_ROOT}" \
            "待截原图(原图_重截):${RECROP_ROOT}" "菌落原始照片(补拍_2026):${COLONY_RAW_ROOT}"; do
  n="${pair%%:*}"; p="${pair#*:}"
  [[ -d "$p" ]] && pass_ "${n}  ${p}" || warn_ "${n} 不存在：${p}" "检查 train_config.zsh 里的路径"
done
[[ -f "${AUDIT_REVIEW}/train_audit/train_manifest.csv" ]] && pass_ "train_manifest.csv（00_audit_train 的输出）存在" \
  || warn_ "没有 output/04_audit_review/train_audit/train_manifest.csv" "建数据集需要它：先运行 zsh train_2_audit.zsh"
if [[ "${DS_USE_REVIEW}" == 1 ]]; then
  if [[ -f "${DS_REVIEW_CSV}" || -f "${AUDIT_REVIEW}/${DS_REVIEW_CSV}" ]]; then pass_ "人工核对表 ${DS_REVIEW_CSV} 存在"
  else warn_ "DS_USE_REVIEW=1 但找不到 ${DS_REVIEW_CSV}" "放到 Audit/ 或 output/04_audit_review/，或把 DS_USE_REVIEW 改成 0"; fi
fi
free_gb=$(df -k "${AUDIT}" | awk 'NR==2 {printf "%d", $4/1024/1024}')
(( free_gb < 20 )) && warn_ "磁盘剩余 ${free_gb} GB（建议 ≥ 20 GB：每个 backbone 的 CV 约占 0.5–2 GB）" "清理旧的 runs 或外接硬盘" \
                   || pass_ "磁盘剩余 ${free_gb} GB"

# ---------------------------------------------------------------- ⑤ 冒烟训练
if [[ "${SELFTEST_QUICK}" != 1 ]]; then
  step "⑤ 冒烟训练（显微：02_train.py cv + full + 03_sanity_check.py）"
  LST="${ST}/datasets.tsv"
  tool train_review.py datasets --kind micro --tsv "${LST}" >/dev/null 2>&1
  DS=$(head -1 "${LST}" 2>/dev/null)
  if [[ -z "${DS}" ]]; then
    warn_ "还没有显微数据集，跳过显微冒烟训练" "建好数据集（train_4_dataset.zsh）后再跑一次自检"
  else
    info "用数据集 ${DS:t} 抽样"
    if run_py "${REC}" 00_train_check.py smoke-subset --trainset "${DS}" --out "${ST}/subset" \
       && run_py "${REC}" 02_train.py --backbone "${DEFAULT_BACKBONES[1]}" --mode cv --trainset "${ST}/subset" \
            --folds 2 --only_fold 1 --epochs 1 --batch 4 --workers 0 --no_pretrained --resume_dir "${ST}/smoke_cv" \
       && run_py "${REC}" 02_train.py --backbone "${DEFAULT_BACKBONES[1]}" --mode full --trainset "${ST}/subset" \
            --epochs 1 --batch 4 --workers 0 --no_pretrained --resume_dir "${ST}/smoke_full" \
       && run_py "${REC}" 03_sanity_check.py --models "${ST}/smoke_full" --trainset "${ST}/subset" --per_class 2; then
      pass_ "显微训练代码跑通（${DEFAULT_BACKBONES[1]}：cv、full、载入模型预测）"
    else
      fail_ "显微冒烟训练失败（正式训练会遇到同样的问题）" "把上面第一个 ❌ 附近的报错发给 Claude"
    fi
    find "${ST}" -name '*.pt' -delete 2>/dev/null
  fi

  step "⑤ 冒烟训练（菌落：02_colony_train.py 方案 A）"
  tool train_review.py datasets --kind colony --tsv "${LST}" >/dev/null 2>&1
  CDS=$(head -1 "${LST}" 2>/dev/null)
  if [[ ${COLONY_OK} == 0 ]]; then
    warn_ "菌落 python 不可用，跳过菌落冒烟训练"
  elif [[ -z "${CDS}" ]]; then
    warn_ "还没有菌落数据集，跳过菌落冒烟训练"
  else
    SNAME="_SELFTEST_$(date +%H%M%S)"
    if run_py_colony "${REC}" 02_colony_train.py --pre "${CDS}" --out "${SNAME}" --skip_b; then
      pass_ "菌落训练代码跑通（数据集 ${CDS:t}，方案 A）"
    else
      fail_ "菌落冒烟训练失败" "常见原因：菌落 python 缺包、数据集 manifest.csv 不完整；看上面的报错"
    fi
    [[ -d "${COLONY_RUNS}/${SNAME}" ]] && mv "${COLONY_RUNS}/${SNAME}" "${ST}/smoke_colony"
  fi
fi

# ---------------------------------------------------------------- ⑥ 判定
step "⑥ 汇总判定"
{
  print -r -- "训练流程自检  $(now)  ${ST:t}"
  print -r -- "模式：$([[ "${SELFTEST_QUICK}" == 1 ]] && echo '快速（无冒烟训练）' || echo '完整（含冒烟训练）')"
  print -r -- ""
  for x in "${FAILS[@]}"; do print -r -- "❌ $x"; done
  for x in "${WARNS[@]}"; do print -r -- "⚠️  $x"; done
  for x in "${PASS[@]}"; do print -r -- "✅ $x"; done
  print -r -- ""
  print -r -- "合计：✅ ${#PASS}  ⚠️ ${#WARNS}  ❌ ${#FAILS}"
} > "${ST}/SELFTEST_RESULT.txt"
cp train_config.zsh "${ST}/" 2>/dev/null
cp "${TR_LOG}" "${ST}/terminal.log" 2>/dev/null
print -r -- "${GRN}✅ 通过 ${#PASS}${R}   ${YEL}⚠️ 警告 ${#WARNS}${R}   ${RED}❌ 失败 ${#FAILS}${R}"
if (( ${#FAILS} )); then
  milestone "${BG_RED}" "🩺" "自检未通过：先解决上面的 ❌ 项（结论 ${ST}/SELFTEST_RESULT.txt）"
  exit 1
fi
if (( ${#WARNS} )); then
  milestone "${BG_YEL}" "🩺" "自检通过，但有 ${#WARNS} 个警告（结论 ${ST}/SELFTEST_RESULT.txt）"
else
  milestone "${BG_GRN}" "🩺" "自检全部通过（结论 ${ST}/SELFTEST_RESULT.txt）"
fi
TR_RUN_DIR=""
next_step "zsh train_1_init.zsh" "建立（补齐）文件夹；已有的文件夹和文件不会被改动"
