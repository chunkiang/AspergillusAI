# =============================================================================
# train_lib.zsh —— train_*.zsh 共用的函数（不用改，也不要单独运行）
#
#   醒目提示：🚀 开始  ✅ 完成  ⚠️ 警告  ❌ 错误  🏁 结束  💥 异常结束  👉 下一步
#   run_py：运行一个 python 脚本，并把"哪个脚本（sha256）+ 完整参数 + 退出码 + 用时"
#           记到总账 output/06_logs/train_ledger.tsv，以及（若指定）本次输出文件夹的 RUN_RECORD.md
#   交互：任何提问处输入 Q 退出、B 返回上一步；直接回车 = 方括号里的默认值
# =============================================================================

# ---------------------------------------------------------------- 颜色
if [[ -t 1 || -n "${TRAIN_FORCE_COLOR}" ]]; then
  R=$'\e[0m'; B=$'\e[1m'; DIM=$'\e[2m'
  RED=$'\e[1;31m'; GRN=$'\e[1;32m'; YEL=$'\e[1;33m'; BLU=$'\e[1;34m'; MAG=$'\e[1;35m'; CYN=$'\e[1;36m'
  BG_RED=$'\e[1;97;41m'; BG_GRN=$'\e[1;97;42m'; BG_BLU=$'\e[1;97;44m'; BG_YEL=$'\e[1;30;43m'; BG_MAG=$'\e[1;97;45m'
else
  R= B= DIM= RED= GRN= YEL= BLU= MAG= CYN= BG_RED= BG_GRN= BG_BLU= BG_YEL= BG_MAG=
fi
LINE="════════════════════════════════════════════════════════════════════"
THIN="────────────────────────────────────────────────────────────────────"

now()    { date '+%F %T'; }
stamp()  { date '+%Y%m%d_%H%M'; }
info()   { print -r -- "${CYN}ℹ️  $*${R}"; }
ok()     { print -r -- "${GRN}✅ $*${R}"; }
warn()   { print -r -- "${YEL}⚠️  $*${R}"; }
err()    { print -r -- "${RED}❌ $*${R}"; }
hint()   { print -r -- "${YEL}   💡 $*${R}"; }
step()   { print -r -- ""; print -r -- "${MAG}▶️  ${B}$*${R}"; }
die()    { err "$*"; exit 1; }

banner() {   # banner <颜色> <emoji> <文字>
  print -r -- ""
  print -r -- "$1${LINE}${R}"
  print -r -- "$1  $2  $3  $(now)${R}"
  print -r -- "$1${LINE}${R}"
}

milestone() {   # 重要节点：milestone <颜色> <emoji> <文字>
  print -r -- ""
  print -r -- "$1  $2  $3  ${R}"
}

next_step() {   # next_step <命令> [说明]
  print -r -- ""
  print -r -- "${BG_GRN}  👉 下一步：$1  ${R}"
  [[ -n "$2" ]] && print -r -- "${GRN}     $2${R}"
}

sha() {      # 文件的 sha256（macOS 用 shasum，Linux 兼容）
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}

# ---------------------------------------------------------------- 开始 / 结束
# tr_start <本脚本名> <说明>：建日志、屏幕输出同时写入日志、注册结束提示
tr_start() {
  WRAPPER="$1"; WRAPPER_DESC="$2"
  # 位置检查：train_* 必须和原有脚本放在同一个 Audit 文件夹里
  if [[ ! -f "${AUDIT}/paths.py" || ! -f "${AUDIT}/02_train.py" || ! -f "${AUDIT}/01_build_trainset.py" ]]; then
    print -r -- "${RED}❌ ${WRAPPER} 没放在 Audit 文件夹里（当前位置 ${AUDIT}）${R}"
    print -r -- "${YEL}   请把 train_kit 里的全部文件直接复制到 Audit 文件夹（和 paths.py、02_train.py 放在一起），${R}"
    print -r -- "${YEL}   然后在 Audit 文件夹里运行，例如：cd /Users/chunjiang/Documents/真菌AI/Audit && zsh ${WRAPPER}${R}"
    exit 1
  fi
  mkdir -p "${LOG_DIR}"
  TR_LOG="${LOG_DIR}/${WRAPPER:r}_$(date +%Y%m%d_%H%M%S).log"
  TR_T0=$(date +%s)
  exec > >(tee -a "${TR_LOG}") 2>&1
  banner "${BG_BLU}" "🚀" "开始 ${WRAPPER}：${WRAPPER_DESC}"
  info "外层脚本 ${WRAPPER}（sha256 $(sha "${AUDIT}/${WRAPPER}" | cut -c1-12)）  日志 ${TR_LOG}"
  [[ -n "${TR_INTERACTIVE}" ]] && print -r -- "${DIM}   提问处：直接回车 = 方括号里的默认值；输入 Q 退出，B 返回上一步${R}"
  TR_STARTED=1
}

# 结束提示：TRAPEXIT 必须定义在最外层；子 shell 退出时不提示
TRAPEXIT() {
  local rc=$?
  [[ -n "${TR_STARTED}" && ${ZSH_SUBSHELL} -eq 0 ]] && tr_end ${rc}
}

tr_end() {
  local rc=$1 mins=$(( ($(date +%s) - TR_T0) / 60 )) secs=$(( ($(date +%s) - TR_T0) % 60 ))
  if [[ -n "${TR_QUIT}" ]]; then
    banner "${BG_YEL}" "🚪" "结束 ${WRAPPER}：已按 Q 退出（用时 ${mins} 分 ${secs} 秒）"
    [[ -n "${TR_RUN_DIR}" && -d "${TR_RUN_DIR}" ]] && _mark_failed "按 Q 中途退出"
  elif [[ ${rc} -eq 0 ]]; then
    banner "${BG_GRN}" "🏁" "结束 ${WRAPPER}：全部完成（用时 ${mins} 分 ${secs} 秒）"
  else
    banner "${BG_RED}" "💥" "结束 ${WRAPPER}：出错退出（退出码 ${rc}，用时 ${mins} 分 ${secs} 秒）"
    print -r -- "${RED}   看上面第一个 ❌ 的提示；完整日志 ${TR_LOG}${R}"
    [[ -n "${TR_RUN_DIR}" && -d "${TR_RUN_DIR}" ]] && _mark_failed "出错退出（退出码 ${rc}）"
  fi
  rm -f "${LOG_DIR}"/.*_$$*(N) 2>/dev/null     # 本次运行的临时文件
  sleep 0.3      # 等 tee 把最后几行写进日志
}

_mark_failed() {   # 未完成的输出文件夹加 _FAILED 后缀，避免误用
  print -r -- "- ❌ $1，$(now)" >> "${TR_RUN_DIR}/RUN_RECORD.md"
  cp "${TR_LOG}" "${TR_RUN_DIR}/terminal.log" 2>/dev/null
  local dst="${TR_RUN_DIR}_FAILED"
  [[ -e "${dst}" ]] && dst="${TR_RUN_DIR}_FAILED_$(date +%H%M%S)"
  mv "${TR_RUN_DIR}" "${dst}" && print -r -- "${RED}   未完成的文件夹已改名为 ${dst:t}（不会被当作可用结果列出）${R}"
}

tr_quit() { TR_QUIT=1; exit 0; }

# ---------------------------------------------------------------- 交互
# ask <提示> [默认值]：结果放在 $ANS；Q → 退出；B → 返回 10；读到文件结尾（无终端）→ 按 Q 处理
ask() {
  local prompt="$1" def="$2" line
  if [[ -n "${def}" ]]; then print -rn -- "${CYN}❓ ${prompt} [${def}]：${R}"
  else print -rn -- "${CYN}❓ ${prompt}：${R}"; fi
  if ! read -r line; then print -r -- ""; tr_quit; fi
  line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"   # 去掉首尾空白
  case "${line}" in
    q|Q) tr_quit ;;
    b|B) ANS=""; print -r -- "${DIM}   ↩️  返回上一步${R}"; return 10 ;;
  esac
  ANS="${line:-${def}}"
  [[ -t 0 ]] || print -r -- ""                         # 非终端输入（测试）时补换行
  [[ -z "${line}" && -n "${def}" ]] && print -r -- "${DIM}   → 用默认值 ${def}${R}"
  return 0
}

# ask_yn <提示> <y|n 默认>：是 → 0，否 → 1，B → 10
ask_yn() {
  local d="$2"
  while true; do
    ask "$1（y/n）" "${d}"; local rc=$?; (( rc == 10 )) && return 10
    case "${ANS}" in y|Y|yes|是) return 0 ;; n|N|no|否) return 1 ;; esac
    warn "请输入 y 或 n（或 Q 退出、B 返回）"
  done
}

# parse_nums <输入> <最大编号>：支持 1,2,4  1 2 4  2-5  a/all；结果放在 $NUMS（去重、按输入顺序）
parse_nums() {
  local s="${1//，/,}" max=$2 tok a b i
  NUMS=()
  if [[ "${s}" == (a|all|A|ALL|全部) ]]; then NUMS=({1..${max}}); return 0; fi
  for tok in ${(s:,:)${s// /,}}; do
    [[ -z "${tok}" ]] && continue
    if [[ "${tok}" == <->-<-> ]]; then
      a=${tok%-*}; b=${tok#*-}
      (( a >= 1 && b <= max && a <= b )) || return 1
      for (( i = a; i <= b; i++ )); do (( ${NUMS[(Ie)$i]} )) || NUMS+=($i); done
    elif [[ "${tok}" == <-> ]]; then
      (( tok >= 1 && tok <= max )) || return 1
      (( ${NUMS[(Ie)$tok]} )) || NUMS+=($tok)
    else
      return 1
    fi
  done
  (( ${#NUMS} > 0 ))
}

# ask_nums <提示> <最大编号> [默认]：反复提问直到输入合法；结果在 $NUMS；B → 10
ask_nums() {
  while true; do
    ask "$1" "$3"; local rc=$?; (( rc == 10 )) && return 10
    parse_nums "${ANS}" $2 && return 0
    warn "输入不对：请写 1 到 $2 之间的编号，例如 1,2,4 或 2-5（a = 全部）"
  done
}

# ask_one <提示> <最大编号> [默认]：只能选一个；结果在 $ANS（数字）
ask_one() {
  while true; do
    ask "$1" "$3"; local rc=$?; (( rc == 10 )) && return 10
    [[ "${ANS}" == <-> ]] && (( ANS >= 1 && ANS <= $2 )) && return 0
    warn "请输入 1 到 $2 之间的一个编号"
  done
}

# ---------------------------------------------------------------- 运行 python 并记录
# run_py <RUN_RECORD.md 路径或 -> <python 脚本> [参数…]
#   设置了 TR_TEE=<文件> 时，屏幕输出另存一份到该文件
run_py() {
  local rec="$1"; shift
  local script="$1"
  [[ -f "${AUDIT}/${script}" ]] || { err "找不到 ${script}（应放在 ${AUDIT}）"; return 1; }
  local h=$(sha "${AUDIT}/${script}") t0=$(date +%s) rc=0
  local -a qa=(${(q-)@})
  local py="${PYBIN:-${PYTHON}}"
  local cmd="${py} ${qa[*]}"
  print -r -- ""
  print -r -- "${BLU}🚀 [$(date '+%T')] 开始 ${B}${script}${R}"
  print -r -- "${BLU}   命令：${cmd}${R}"
  if [[ -n "${TR_TEE}" ]]; then
    "${py}" -u "$@" 2>&1 | tee -a "${TR_TEE}"; rc=${pipestatus[1]}
  else
    "${py}" -u "$@" || rc=$?
  fi
  local dt=$(( $(date +%s) - t0 ))
  if [[ ${rc} -eq 0 ]]; then
    print -r -- "${GRN}✅ [$(date '+%T')] 完成 ${script}（$(_dur ${dt})）${R}"
  else
    print -r -- "${RED}❌ [$(date '+%T')] 出错 ${script}（退出码 ${rc}，$(_dur ${dt})）${R}"
  fi
  # 总账（制表符分隔，Excel 可直接打开）
  [[ -f "${LEDGER}" ]] || print -r -- $'时间\t外层脚本\tpython脚本\tsha256\t退出码\t秒\t完整命令' > "${LEDGER}"
  print -r -- "$(now)	${WRAPPER}	${script}	${h:0:16}	${rc}	${dt}	${cmd}" >> "${LEDGER}"
  # 本次输出文件夹的记录
  if [[ "${rec}" != "-" ]]; then
    {
      print -r -- "### ${script}  $([[ ${rc} -eq 0 ]] && echo ✅ || echo ❌)"
      print -r -- "- 时间：$(now)（用时 $(_dur ${dt})，退出码 ${rc}）"
      print -r -- "- 外层脚本：${WRAPPER}（sha256 $(sha "${AUDIT}/${WRAPPER}")）"
      print -r -- "- python 脚本：${script}（sha256 ${h}）"
      print -r -- "- python：$(command -v "${py}" 2>/dev/null || print -r -- "${py}")"
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

_dur() { local s=$1; (( s >= 3600 )) && { print -r -- "$((s/3600)) 小时 $((s%3600/60)) 分"; return; }
         (( s >= 60 )) && { print -r -- "$((s/60)) 分 $((s%60)) 秒"; return; }; print -r -- "${s} 秒"; }

# 菌落相关脚本用 COLONY_PYTHON（须与临床验证时的 scikit-learn 版本一致）
run_py_colony() { PYBIN="${COLONY_PY}" run_py "$@"; }

# 工具脚本（00_train_check.py / train_review.py）的简单调用：不进总账
tool()        { "${PYTHON}" -u "${AUDIT}/$1" "${@:2}"; }
tool_colony() { "${COLONY_PY}" -u "${AUDIT}/$1" "${@:2}"; }

# kv_read <命令…>：读取命令输出里的 KEY=值 行，赋给同名变量（调用前先 local 声明）
kv_read() {
  local line k v
  while IFS= read -r line; do
    [[ "${line}" =~ '^[A-Za-z_][A-Za-z0-9_]*=' ]] || continue
    k="${line%%=*}"; v="${line#*=}"
    eval "${k}=\${v}"
  done < <("$@")
}

# 新建 RUN_RECORD.md：new_record <文件> <标题>
new_record() {
  {
    print -r -- "# $2"
    print -r -- ""
    print -r -- "- 时间：$(now)；电脑：$(hostname)；外层脚本：${WRAPPER}"
    print -r -- "- 终端日志：${TR_LOG}"
    print -r -- ""
    print -r -- "## 依次运行的 python 脚本与参数"
    print -r -- ""
  } > "$1"
}

# ---------------------------------------------------------------- 检查
check_python() {
  local miss
  miss=$("${PYTHON}" - <<'EOF' 2>/dev/null
import importlib.util
need = {"torch": "torch", "torchvision": "torchvision", "sklearn": "scikit-learn", "pandas": "pandas",
        "numpy": "numpy", "PIL": "pillow", "cv2": "opencv-python", "matplotlib": "matplotlib"}
print(" ".join(p for m, p in need.items() if importlib.util.find_spec(m) is None))
EOF
) || die "python 无法运行。请先 conda activate fungal"
  if [[ -n "${miss}" ]]; then
    err "当前 python 环境缺少：${miss}"
    hint "安装（清华源）：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ${miss}"
    exit 1
  fi
  ok "显微用 python：$("${PYTHON}" -c 'import sys; print(sys.executable)')"
}

check_colony_python() {   # 菌落 python 能运行、依赖齐全；返回 0/1，不退出
  if ! "${COLONY_PY}" -c 'import sys' 2>/dev/null; then
    err "菌落用 python 无法运行：${COLONY_PY}"
    hint "改 train_config.zsh 的 COLONY_PYTHON（应与 clin_config.zsh 的 COLONY_PYTHON 相同）"
    return 1
  fi
  local miss
  miss=$("${COLONY_PY}" - <<'EOF' 2>/dev/null
import importlib.util
need = {"sklearn": "scikit-learn", "pandas": "pandas", "numpy": "numpy", "scipy": "scipy", "PIL": "pillow",
        "cv2": "opencv-python", "joblib": "joblib", "matplotlib": "matplotlib"}
print(" ".join(p for m, p in need.items() if importlib.util.find_spec(m) is None))
EOF
)
  if [[ -n "${miss}" ]]; then
    err "菌落用 python（${COLONY_PY}）缺少：${miss}"
    hint "安装（清华源）：${COLONY_PY} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ${miss}"
    return 1
  fi
  ok "菌落用 python：${COLONY_PY}（scikit-learn $("${COLONY_PY}" -c 'import sklearn; print(sklearn.__version__)')）"
  "${COLONY_PY}" -c "import pillow_heif" 2>/dev/null || { warn "菌落用 python 没有 pillow-heif：HEIC 格式的菌落照片读不了（JPG 不受影响）"
    hint "${COLONY_PY} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow-heif"; }
  return 0
}

need_scripts() {   # need_scripts <脚本…>：缺任何一个就退出
  local f miss=()
  for f in "$@"; do [[ -f "${AUDIT}/$f" ]] || miss+=("$f"); done
  (( ${#miss} )) && die "Audit 文件夹里缺少：${miss[*]}"
  return 0
}

keep_awake() {   # macOS 防休眠，本脚本结束自动解除
  command -v caffeinate >/dev/null && { caffeinate -i -w $$ & } 2>/dev/null
}

# 数据集名字检查：只允许字母、数字、_、-；不能已存在
valid_name() {
  [[ "$1" =~ '^[A-Za-z0-9][A-Za-z0-9_-]*$' ]] || { warn "名字只能用英文字母、数字、_ 和 -（不能有空格、中文、+），且以字母或数字开头"; return 1; }
  [[ -e "${TRAINSETS}/$1" ]] && { warn "${TRAINSETS}/$1 已存在，请换一个名字"; return 1; }
  return 0
}
