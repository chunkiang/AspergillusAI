# =============================================================================
# clin_config.zsh —— 临床验证的全部设置（唯一需要改的文件；其他 clin_*.zsh 都读它）
#
# 改了这里的任何一项，下一次运行的 RUN_RECORD.md 里都会记下新值，可追溯。
# =============================================================================

# ---------------------------------------------------------------- 需要确认的 4 项
# 1) 临床验证素材的总文件夹（照片、答案表、截图都在这里面）
CLIN_ROOT="/Users/chunjiang/Documents/真菌AI/临床验证_2026"

# 2) 镜检模型（output/02_training_runs 下的文件夹，含 final.pt / class_names.json / args.json）
#    写多个 = 集成，例如 (runs_v4/swin_t_full_final runs_v4/densenet121_full_final)
MICRO_MODELS=(runs_v4/swin_t_full_final)	# ⚠️
# MICRO_MODELS=(micro_20261023_0851_ONV/efficientnet_b0_full_final)

# 3) 菌落模型（02_colony_train.py 锁定后的输出，含 final_model.joblib / locked_config.json）
COLONY_MODEL="colony/20261020_2108"			# ⚠️ swin_t
#COLONY_MODEL="colony/20261023_0924"			# efficientnet_b0

# 4) 运行模式
#    PILOT=1 试运行：允许没截完就揭盲、不核对锁定文件；结果文件夹以 PILOT_ 开头，不能写进论文
#    PILOT=0 正式：必须先 zsh clin_lock.zsh 锁定，必须全部截完才能揭盲
PILOT=1

ANNOTATOR="赵"            # 菌落标注人（写进标注 JSON）

# 菌落脚本用的 python。留空 = 和镜检一样用当前环境（fungal）的 python。
# 菌落模型必须用"训练它时的 scikit-learn 版本"运行；版本不一致时 clin_0_selftest.zsh 会报错并推荐该填哪个，
# 例如 COLONY_PYTHON="/Users/chunjiang/miniforge3/bin/python"
COLONY_PYTHON="/Users/chunjiang/miniforge3/envs/ai/bin/python" # ⚠️

# 自检（clin_0_selftest.zsh）用的训练数据：镜检模型的训练清单、菌落模型的训练预处理文件夹
MICRO_TRAINSET="trainset_v4"              # 优先用镜检模型 args.json 里记录的；找不到才用这个
COLONY_TRAIN_PRE="colony_preprocessed"    # output/01_trainsets/ 下

# ---------------------------------------------------------------- 预先设定的判读规则（正式验证前定好，不再改）
P_MIN=0.70                # top1 概率低于它 → 需复核
GAP_MIN=0.20              # top1 − top2 低于它 → 需复核
ASP_MIN=0.5               # 镜检曲霉类概率之和 ≥ 它 → 走曲霉通路（镜检 + 菌落融合）
ALPHA=0.8                 # 曲霉融合时菌落的权重（线性加权）：s = (1−α)·镜检 + α·菌落；0 = 只看镜检，1 = 只看菌落
                          # α 会写进锁定文件；正式模式下与锁定值不一致会拒绝运行
ID_REGEX='^PUPH-PCL-[0-9]{3}$'   # 株号规则：PUPH-PCL-001 … PUPH-PCL-999

# ---------------------------------------------------------------- 以下由上面推出，一般不用改
MICRO_DIR="${CLIN_ROOT}/00_raw/micro"              # 镜检：每株一个子文件夹
COLONY_DIR="${CLIN_ROOT}/00_raw/colony"            # 菌落：只放曲霉（或疑似曲霉）的平皿照片
ANS_DIR="${CLIN_ROOT}/answers"
ANS_MASTER="${ANS_DIR}/answers_master.csv"         # 唯一需要手填的答案表
ROI_DIR="${CLIN_ROOT}/01_roi"                      # 镜检盲法截图（03_roi_clinical.py）
COLONY_ANN="${CLIN_ROOT}/02_colony_annotations"    # 菌落几何标注 + 白平衡（01_colony_annotate / preprocess pick）
LOCK="${CLIN_ROOT}/clinical_lock.json"             # 正式验证的锁定文件

COLONY_PY="${COLONY_PYTHON:-python}"
AUDIT="${CLIN_KIT_DIR}"                            # = 脚本所在的 Audit 文件夹
EVAL_ROOT="${AUDIT}/output/03_external_eval/clinical"   # 每次评估一个子文件夹
LOG_DIR="${AUDIT}/output/06_logs"
LEDGER="${LOG_DIR}/clin_ledger.tsv"                # 所有 clin_*.zsh 调用 python 的总账
