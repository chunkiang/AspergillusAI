# =============================================================================
# fsim_config.zsh —— 同节随机配对融合分析的全部设置（唯一需要改的文件）
#
# 改了这里的任何一项，下一次运行的 RUN_RECORD.md 里都会记下新值。
# 等号两边不要加空格；路径里有空格或中文时用双引号括起来。
# =============================================================================

# ---------------------------------------------------------------- 1) python
PYTHON="python"        # 当前 conda 环境（fungal）的 python；只需要 pandas numpy scipy matplotlib

# ---------------------------------------------------------------- 2) 输入
# 镜检：交叉验证所在的目录。诊断会在它下面递归查找 OOF 预测表（不是 swin_t_full_final！）
MICRO_RUN="${FSIM_OUT}/02_training_runs/runs_v4"
MICRO_BACKBONE="swin_t"     # 多个 backbone 时优先选路径里含它的那个
MICRO_OOF=""                # 自动查找选错时，在这里直接写 OOF 文件的完整路径；留空 = 自动查找

# 菌落：与临床验证锁定的是同一个模型
COLONY_RUN="${FSIM_OUT}/02_training_runs/colony/20261020_2108"

TAXONOMY="${FSIM_AUDIT}/taxonomy_map.csv"

# ---------------------------------------------------------------- 3) 融合规则（与临床验证预设一致）
FUSION_RULE="linear"        # linear = (1−w)·镜检 + w·菌落；product = 乘积
COLONY_WEIGHT=0.8           # 主分析的菌落权重（预设，不按结果调）
SENS_WEIGHTS="0,0.5,0.6,0.7,0.8,0.9,1.0"   # 敏感性分析（只放补充材料）
ASP_GATE=0.5                # 镜检 P(曲霉) ≥ 它才进入曲霉通路

# ---------------------------------------------------------------- 4) 随机配对
N_ITER=1000
SEED=20261113

# ---------------------------------------------------------------- 以下一般不用改
CHECK_ROOT="${FSIM_OUT}/04_audit_review/fusion_sim_check"     # 诊断输出
RUN_ROOT="${FSIM_OUT}/02_training_runs/fusion_sim"            # 正式结果
LOG_DIR="${FSIM_OUT}/06_logs"
LEDGER="${LOG_DIR}/fsim_ledger.tsv"                           # 每次调用 python 记一行
