#!/bin/zsh
# =============================================================================
# clin_4_colony.zsh —— 第 4 步：曲霉菌落照片的几何标注 + 白平衡参照确认（两个交互窗口，依次进行）
#
# 用法：
#     zsh clin_4_colony.zsh            先标注（平皿边界/菌落中心/菌落边缘），再确认白平衡参照
#     zsh clin_4_colony.zsh pick       只做白平衡确认（标注已做完时）
#
# 与训练集完全相同的工具和参数，只是标注存到临床验证文件夹，不与训练集混放。
# 只标正面（O）；反面照片可以放着，不参与模型。
# =============================================================================
CLIN_KIT_DIR="${0:A:h}"; cd "${CLIN_KIT_DIR}" || exit 1
source ./clin_config.zsh; source ./clin_lib.zsh
clin_start "${0:t}" "菌落标注与白平衡确认"

[[ -d "${COLONY_DIR}" ]] || die "找不到 ${COLONY_DIR}，先运行 zsh clin_1_init.zsh"
n=$(ls "${COLONY_DIR}" | grep -ciE '\-B-O-[0-9]+\.(jpe?g|png|heic|heif|tiff?)$')
[[ ${n} -gt 0 ]] || die "${COLONY_DIR} 里没有符合命名的菌落正面照片（PUPH-PCL-001-B-O-1.jpg）"
info "菌落正面照片 ${n} 张"

if [[ "$1" != "pick" ]]; then
  step "① 几何标注（Enter 确认每一步；c 污染  b 不合格  q 退出）"
  run_py_colony - 01_colony_annotate.py --root "${COLONY_DIR}" --ann_dir "${COLONY_ANN}" --annotator "${ANNOTATOR}" --only=-B-O- \
    || die "标注工具异常退出（已保存的不会丢失，重新运行即可继续）"
fi

step "② 白平衡参照确认（点击/拖框选白纸；a 切换培养基模式；x 无参照；Enter 下一张）"
run_py_colony - 01_colony_preprocess.py pick --root "${COLONY_DIR}" --ann_dir "${COLONY_ANN}" --views O --annotator "${ANNOTATOR}" \
  || die "白平衡工具异常退出（已保存的不会丢失，重新运行即可继续）"

ok "菌落标注保存在 ${COLONY_ANN}"
info "下一步：zsh clin_5_eval.zsh"
