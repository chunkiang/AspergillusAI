# =============================================================================
# fsim_lib.zsh —— fsim_*.zsh 共用的函数（不要直接运行）
# =============================================================================
setopt pipefail 2>/dev/null

FSIM_AUDIT="${0:A:h}"
FSIM_OUT="${FSIM_AUDIT}/output"
source "${FSIM_AUDIT}/fsim_config.zsh" || { print "❌ 读不到 fsim_config.zsh"; exit 1; }

say_start() { print -P "\n%F{cyan}🚀 ====== $1 ======%f"; }
say_ok()    { print -P "%F{green}✅ $1%f"; }
say_warn()  { print -P "%F{yellow}⚠️  $1%f"; }
say_err()   { print -P "%F{red}❌ $1%f"; }
say_done()  { print -P "\n%F{green}🎉 ====== $1 ======%f\n"; }

sha() { shasum -a 256 "$1" 2>/dev/null | cut -c1-16; }

# 检查 python 和依赖
fsim_check_env() {
  command -v ${PYTHON} >/dev/null 2>&1 || { say_err "找不到 python：${PYTHON}（先 conda activate fungal）"; exit 1; }
  ${PYTHON} -c "import pandas, numpy, scipy" 2>/dev/null || {
    say_err "缺少依赖。请运行："
    print "    ${PYTHON} -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy scipy matplotlib"
    exit 1; }
  ${PYTHON} -c "import matplotlib" 2>/dev/null || say_warn "没有 matplotlib：不画图，其他照常"
}

# 用法：fsim_run <输出目录> <标签> <python 脚本> [参数...]
# 终端日志、RUN_RECORD.md、脚本和配置快照写进输出目录；总账写进 LEDGER；出错时目录加 _FAILED
fsim_run() {
  local out="$1" tag="$2" script="$3"; shift 3
  local spath="${FSIM_AUDIT}/${script}"
  [[ -f "$spath" ]] || { say_err "找不到脚本 ${spath}（所有文件要平铺在 Audit 下）"; exit 1; }
  mkdir -p "$out/_snapshot" "$LOG_DIR"
  cp "$spath" "${FSIM_AUDIT}/fsim_config.zsh" "${FSIM_AUDIT}/fsim_lib.zsh" "$out/_snapshot/" 2>/dev/null
  local t0=$(date +%s) start=$(date "+%Y-%m-%d %H:%M:%S")
  print "▶️  ${PYTHON} ${script} $*"
  ${PYTHON} "$spath" "$@" 2>&1 | tee "$out/terminal.log"
  local rc=${pipestatus[1]}
  local dur=$(( $(date +%s) - t0 ))
  {
    print -r -- "# RUN_RECORD — ${tag}"
    print -r -- ""
    print -r -- "- 开始：${start}；用时 ${dur} 秒；退出码 ${rc}"
    print -r -- "- 脚本：${script}（sha256 前 16 位 $(sha "$spath")）"
    print -r -- "- python：$(command -v ${PYTHON})（$(${PYTHON} --version 2>&1)）"
    print -r -- "- 完整命令："
    print -r -- '```'
    print -r -- "${PYTHON} ${spath} $*"
    print -r -- '```'
    print -r -- "- fsim_config.zsh（sha256 前 16 位 $(sha "${FSIM_AUDIT}/fsim_config.zsh")）："
    print -r -- '```'
    grep -v '^\s*#' "${FSIM_AUDIT}/fsim_config.zsh" | grep -v '^\s*$'
    print -r -- '```'
  } > "$out/RUN_RECORD.md"
  [[ -f "$LEDGER" ]] || print -r -- $'time\twrapper\tscript\tsha256_16\texit\tseconds\tout\targs' > "$LEDGER"
  print -r -- "${start}"$'\t'"${tag}"$'\t'"${script}"$'\t'"$(sha "$spath")"$'\t'"${rc}"$'\t'"${dur}"$'\t'"${out}"$'\t'"$*" >> "$LEDGER"
  if (( rc != 0 )); then
    mv "$out" "${out}_FAILED" 2>/dev/null
    FSIM_LAST_OUT="${out}_FAILED"
  else
    FSIM_LAST_OUT="$out"
  fi
  return $rc
}
