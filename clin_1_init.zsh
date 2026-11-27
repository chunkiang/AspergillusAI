#!/bin/zsh
# =============================================================================
# clin_1_init.zsh —— 第 1 步：建立临床验证文件夹 + 检查环境和模型（可反复运行，不会删除任何东西）
#
# 用法（fungal 环境，Audit 文件夹）：  zsh clin_1_init.zsh
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "建立文件夹、检查环境与模型"

step "1/4 建立文件夹"
for d in "${MICRO_DIR}" "${COLONY_DIR}" "${ANS_DIR}" "${ROI_DIR}" "${COLONY_ANN}" "${EVAL_ROOT}" "${LOG_DIR}"; do
  if [[ -d "$d" ]]; then info "已存在 $d"; else mkdir -p "$d" && ok "新建 $d"; fi
done
README="${CLIN_ROOT}/目录说明.txt"
if [[ ! -f "${README}" ]]; then
  cat > "${README}" <<EOF
临床验证素材目录（由 clin_1_init.zsh 生成）

00_raw/micro/PUPH-PCL-001/   该株的镜检照片（乳酸酚棉蓝，任意文件名，如 PUPH-PCL-001-C-01.jpg）
00_raw/colony/               曲霉（或疑似曲霉）的菌落照片，命名 PUPH-PCL-001-B-O-1.jpg（O 正面，R 反面）
answers/answers_master.csv   参考鉴定答案表（唯一需要手填）；answers_clinical.csv / answers_colony.csv 自动生成
01_roi/                      镜检盲法截图（clin_3_roi.zsh 生成，不要手动改）
02_colony_annotations/       菌落标注（clin_4_colony.zsh 生成，不要手动改）
clinical_lock.json           正式验证的锁定文件（clin_lock.zsh 生成，只生成一次）

文件夹名和文件名里不要出现菌名（会破坏盲法）。
EOF
  ok "写出 ${README}"
fi

step "2/4 检查 python 环境"
check_python
for f in 03_roi_clinical.py 03_eval_clinical.py 03_eval_external.py 02_train.py paths.py \
         01_colony_annotate.py 01_colony_preprocess.py 02_colony_train.py 03_colony_predict.py \
         00_clin_check.py 00_clin_selftest.py 03_clin_fusion.py clin_sections.py; do
  [[ -f "$f" ]] || die "Audit 文件夹里缺少 $f"
done
ok "所需 python 脚本齐全"

step "3/4 检查镜检模型"
for m in "${MICRO_MODELS[@]}"; do
  d=""
  for c in "$m" "output/02_training_runs/$m" "output/02_training_runs/runs/$m"; do [[ -d "$c" ]] && { d="$c"; break; }; done
  [[ -n "$d" ]] || die "找不到镜检模型 $m（改 clin_config.zsh 的 MICRO_MODELS）"
  for f in final.pt class_names.json args.json; do [[ -f "$d/$f" ]] || die "$d 里缺少 $f"; done
  ok "镜检模型 $d（final.pt sha256 $(sha "$d/final.pt" | cut -c1-12)…）"
done
run_py - 00_clin_check.py --show_mapping --micro_model "${MICRO_MODELS[1]}" || die "镜检模型类别检查失败"

step "4/4 检查菌落模型"
CM="output/02_training_runs/${COLONY_MODEL}"
[[ -d "${COLONY_MODEL}" ]] && CM="${COLONY_MODEL}"
if [[ -f "${CM}/final_model.joblib" && -f "${CM}/locked_config.json" ]]; then
  ok "菌落模型 ${CM}（final_model.joblib sha256 $(sha "${CM}/final_model.joblib" | cut -c1-12)…）"
  check_colony_env - || die "菌落运行环境与训练时不一致：按上面的提示设置 clin_config.zsh 的 COLONY_PYTHON"
else
  err "找不到锁定的菌落模型 ${CM}（需要 final_model.joblib 和 locked_config.json）"
  if [[ -d output/02_training_runs/colony ]]; then
    warn "现有的菌落训练输出（把其中一个写进 clin_config.zsh 的 COLONY_MODEL）："
    for d in output/02_training_runs/colony/*(N/); do
      [[ -f "$d/locked_config.json" ]] && print -r -- "     🔒 ${d#output/02_training_runs/}" || print -r -- "        ${d#output/02_training_runs/}（未锁定）"
    done
  fi
  exit 1
fi

print -r -- ""
ok "准备就绪。下一步：把照片放进 ${CLIN_ROOT}/00_raw/ 后运行  zsh clin_2_check.zsh"
