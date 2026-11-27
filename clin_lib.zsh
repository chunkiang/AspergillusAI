# =============================================================================
# clin_lib.zsh —— clin_*.zsh 共用的函数（不用改，也不要单独运行）
#
#   醒目提示：🚀 开始  ✅ 完成  ⚠️ 警告  ❌ 错误  🏁 结束  💥 异常结束
#   run_py：运行一个 python 脚本，并把"哪个脚本（sha256）+ 完整参数 + 退出码 + 用时"
#           记到总账 output/06_logs/clin_ledger.tsv，以及（若指定）本次结果文件夹的 RUN_RECORD.md
# =============================================================================

# ---------------------------------------------------------------- 颜色
if [[ -t 1 || -n "${CLIN_FORCE_COLOR}" ]]; then
  R=$'\e[0m'; B=$'\e[1m'
  RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'; BLU=$'\e[1;34m'; MAG=$'\e[1;35m'; CYN=$'\e[1;36m'
  BG_RED=$'\e[1;97;41m'; BG_GRN=$'\e[1;97;42m'; BG_BLU=$'\e[1;97;44m'
else
  R= B= RED= GRN= YEL= BLU= MAG= CYN= BG_RED= BG_GRN= BG_BLU=
fi
LINE="════════════════════════════════════════════════════════════════════"

now()    { date '+%F %T'; }
info()   { print -r -- "${CYN}ℹ️  $*${R}"; }
ok()     { print -r -- "${GRN}✅ $*${R}"; }
warn()   { print -r -- "${YEL}⚠️  $*${R}"; }
err()    { print -r -- "${RED}❌ $*${R}"; }
step()   { print -r -- ""; print -r -- "${MAG}▶️  ${B}$*${R}"; }
die()    { err "$*"; exit 1; }

banner() {   # banner <颜色> <emoji> <文字>
  print -r -- ""
  print -r -- "$1${LINE}${R}"
  print -r -- "$1  $2  $3  $(now)${R}"
  print -r -- "$1${LINE}${R}"
}

sha() {      # 文件的 sha256（macOS 用 shasum，Linux 兼容）
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}

# ---------------------------------------------------------------- 开始 / 结束
# clin_start <本脚本名> <说明>：建日志、屏幕输出同时写入日志、注册结束提示
clin_start() {
  WRAPPER="$1"; WRAPPER_DESC="$2"
  # 位置检查：clin_* 必须和原有脚本放在同一个 Audit 文件夹里（不能隔一层 clin_kit 子文件夹）
  if [[ ! -f "${AUDIT}/paths.py" || ! -f "${AUDIT}/03_eval_clinical.py" ]]; then
    print -r -- "${RED}❌ ${WRAPPER} 没放在 Audit 文件夹里（当前位置 ${AUDIT}）${R}"
    print -r -- "${YEL}   请把 clin_kit 里的全部文件直接复制到 Audit 文件夹（和 paths.py、03_eval_clinical.py 放在一起），${R}"
    print -r -- "${YEL}   然后在 Audit 文件夹里运行，例如：cd /Users/chunjiang/Documents/真菌AI/Audit && zsh ${WRAPPER}${R}"
    exit 1
  fi
  mkdir -p "${LOG_DIR}"
  CLIN_LOG="${LOG_DIR}/${WRAPPER:r}_$(date +%Y%m%d_%H%M%S).log"
  CLIN_T0=$(date +%s)
  exec > >(tee -a "${CLIN_LOG}") 2>&1
  banner "${BG_BLU}" "🚀" "开始 ${WRAPPER}：${WRAPPER_DESC}"
  info "外层脚本 ${WRAPPER}（sha256 $(sha "${AUDIT}/${WRAPPER}" | cut -c1-12)）  日志 ${CLIN_LOG}"
  if [[ "${PILOT}" == 1 ]]; then
    print -r -- "${YEL}🧪 试运行模式（PILOT=1）：结果只用于跑通流程，不能写进论文${R}"
  else
    print -r -- "${GRN}🔒 正式模式（PILOT=0）${R}"
  fi
  CLIN_STARTED=1
}

# 结束提示：TRAPEXIT 必须定义在最外层（定义在函数里只会在函数返回时触发）；子 shell 退出时不提示
TRAPEXIT() {
  local rc=$?
  [[ -n "${CLIN_STARTED}" && ${ZSH_SUBSHELL} -eq 0 ]] && clin_end ${rc}
}

clin_end() {
  local rc=$1 mins=$(( ($(date +%s) - CLIN_T0) / 60 )) secs=$(( ($(date +%s) - CLIN_T0) % 60 ))
  if [[ ${rc} -eq 0 ]]; then
    banner "${BG_GRN}" "🏁" "结束 ${WRAPPER}：全部完成（用时 ${mins} 分 ${secs} 秒）"
  else
    banner "${BG_RED}" "💥" "结束 ${WRAPPER}：出错退出（退出码 ${rc}，用时 ${mins} 分 ${secs} 秒）"
    print -r -- "${RED}   看上面第一个 ❌ 的提示；完整日志 ${CLIN_LOG}${R}"
    if [[ -n "${CLIN_RUN_DIR}" && -d "${CLIN_RUN_DIR}" ]]; then     # 评估中途出错：结果文件夹加 _FAILED 后缀，避免误用
      print -r -- "- ❌ 中途出错退出（退出码 ${rc}），$(now)" >> "${CLIN_RUN_DIR}/RUN_RECORD.md"
      cp "${CLIN_LOG}" "${CLIN_RUN_DIR}/terminal.log" 2>/dev/null
      mv "${CLIN_RUN_DIR}" "${CLIN_RUN_DIR}_FAILED" && print -r -- "${RED}   未完成的结果文件夹已改名为 ${CLIN_RUN_DIR:t}_FAILED${R}"
    fi
  fi
  sleep 0.3      # 等 tee 把最后几行写进日志
}

# ---------------------------------------------------------------- 运行 python 并记录
# run_py <RUN_RECORD.md 路径或 -> <python 脚本> [参数…]
run_py() {
  local rec="$1"; shift
  local script="$1"
  [[ -f "${AUDIT}/${script}" ]] || { err "找不到 ${script}（应放在 ${AUDIT}）"; return 1; }
  local h=$(sha "${AUDIT}/${script}") t0=$(date +%s) rc=0
  local -a qa=(${(q-)@})
  local py="${PYBIN:-python}"
  local cmd="${py} ${qa[*]}"
  print -r -- ""
  print -r -- "${BLU}🚀 [$(date '+%T')] 开始 ${B}${script}${R}"
  print -r -- "${BLU}   命令：${cmd}${R}"
  "${py}" -u "$@" || rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [[ ${rc} -eq 0 ]]; then
    print -r -- "${GRN}✅ [$(date '+%T')] 完成 ${script}（${dt} 秒）${R}"
  else
    print -r -- "${RED}❌ [$(date '+%T')] 出错 ${script}（退出码 ${rc}，${dt} 秒）${R}"
  fi
  # 总账（制表符分隔，Excel 可直接打开）
  [[ -f "${LEDGER}" ]] || print -r -- $'时间\t外层脚本\tpython脚本\tsha256\t退出码\t秒\t完整命令' > "${LEDGER}"
  print -r -- "$(now)	${WRAPPER}	${script}	${h:0:16}	${rc}	${dt}	${cmd}" >> "${LEDGER}"
  # 本次结果文件夹的记录
  if [[ "${rec}" != "-" ]]; then
    {
      print -r -- "### ${script}  $([[ ${rc} -eq 0 ]] && echo ✅ || echo ❌)"
      print -r -- "- 时间：$(now)（用时 ${dt} 秒，退出码 ${rc}）"
      print -r -- "- 外层脚本：${WRAPPER}（sha256 $(sha "${AUDIT}/${WRAPPER}")）"
      print -r -- "- python 脚本：${script}（sha256 ${h}）"
      print -r -- "- 完整命令："
      print -r -- '  ```'
      print -r -- "  cd ${AUDIT}"
      print -r -- "  ${cmd}"
      print -r -- '  ```'
      print -r -- ""
    } >> "${rec}"
  fi
  return ${rc}
}

# 菌落相关脚本用 COLONY_PYTHON 指定的解释器（须与训练菌落模型时的 scikit-learn 版本一致）
run_py_colony() { PYBIN="${COLONY_PY}" run_py "$@"; }

# 菌落环境核对：解释器能否运行、scikit-learn 版本是否与训练菌落模型时一致
check_colony_env() {   # check_colony_env <记录文件或 ->
  local cm="output/02_training_runs/${COLONY_MODEL}"
  [[ -d "${COLONY_MODEL}" ]] && cm="${COLONY_MODEL}"
  run_py_colony "$1" 00_clin_selftest.py envcheck --colony_model "${cm}"
}

# ---------------------------------------------------------------- 检查
check_python() {
  local miss
  miss=$(python - <<'EOF' 2>/dev/null
import importlib.util
need = {"torch": "torch", "torchvision": "torchvision", "sklearn": "scikit-learn", "pandas": "pandas",
        "numpy": "numpy", "scipy": "scipy", "PIL": "pillow", "cv2": "opencv-python", "joblib": "joblib",
        "matplotlib": "matplotlib"}
print(" ".join(p for m, p in need.items() if importlib.util.find_spec(m) is None))
EOF
) || die "python 无法运行。请先 conda activate fungal"
  if [[ -n "${miss}" ]]; then
    err "当前 python 环境缺少：${miss}"
    print -r -- "${YEL}   安装（清华源）：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ${miss}${R}"
    exit 1
  fi
  ok "python 环境：$(python -c 'import sys; print(sys.executable)')"
}

# 判读规则核对：ALPHA 必须在 clin_config.zsh 里写明（不再用隐含默认值），且在 0–1 之间
check_rules() {
  [[ -n "${ALPHA}" ]] || die "clin_config.zsh 里没有设置 ALPHA（曲霉融合的菌落权重）。请加一行：ALPHA=0.8"
  python -c "import sys; a=float(sys.argv[1]); sys.exit(0 if 0<=a<=1 else 1)" "${ALPHA}" 2>/dev/null \
    || die "ALPHA=${ALPHA} 不是 0–1 之间的数"
  [[ -n "${W_COLONY}" ]] && warn "clin_config.zsh 里的 W_COLONY=${W_COLONY} 已停用（融合 v1.2 起改用 ALPHA），可以删掉这一行"
  return 0
}

keep_awake() {   # macOS 防休眠，本脚本结束自动解除
  command -v caffeinate >/dev/null && { caffeinate -i -w $$ & } 2>/dev/null
}
