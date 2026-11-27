# =============================================================================
# train_config.zsh —— 模型训练流程的全部设置（唯一需要改的文件；其他 train_*.zsh 都读它）
#
# 改了这里的任何一项，下一次运行的 RUN_RECORD.md / DATASET_INFO.json 里都会记下新值，可追溯。
# 行首 # 是注释；等号两边不要加空格；路径里有空格或中文时用双引号括起来。
# =============================================================================

# ---------------------------------------------------------------- 1) 图片所在位置
AI_ROOT="/Users/chunjiang/Documents/真菌AI"

MICRO_DATASET_ROOT="${AI_ROOT}/CNN/dataset"     # 旧截图（dev_original）和 ✅ 文件夹截图（dev_checked）
COLONY_LEGACY_ROOT="${AI_ROOT}/CNN/normal"      # 旧菌落图（只用于 00_audit_train 审计，菌落模型不用）
CROPS_NEW_ROOT="${AI_ROOT}/CNN/crops_new"       # 网络补截（crops_new）
CROPS_V2_ROOT="${AI_ROOT}/CNN/crops_v2"         # 规范截图（crops_v2），01_crop_batch_training.py 的输出
ROI_TEST_ROOT="${AI_ROOT}/CNN/roi_test"         # 外部盲法 ROI（roi_ext）
RECROP_ROOT="${AI_ROOT}/原图_重截"               # 待截图的显微原图：<菌种>/<菌株>/原图
COLONY_RAW_ROOT="${AI_ROOT}/CNN/补拍_2026"       # 菌落原始照片：<组>/PUPH-FUM-001-B-O-1.jpg
COLONY_ANN="colony_annotations"                  # 菌落标注 + 白平衡（output/01_trainsets/ 下，与临床验证分开）

# ---------------------------------------------------------------- 2) python
PYTHON="python"     # 显微相关：当前 conda 环境（fungal）的 python
# 菌落相关（标注、白平衡、建数据集、训练）用的 python。必须和 clin_config.zsh 的 COLONY_PYTHON 一致，
# 否则训练出的菌落模型在临床验证时 scikit-learn 版本对不上（自检会核对）
COLONY_PYTHON="/Users/chunjiang/miniforge3/envs/ai/bin/python"

ANNOTATOR="赵"      # 菌落标注人（写进标注 JSON）

# ---------------------------------------------------------------- 3) 自检
# SELFTEST_QUICK=0：自检时做约 2 分钟的冒烟训练（小样本、1 个 epoch，真实调用 02_train.py 和 02_colony_train.py）
# SELFTEST_QUICK=1：跳过冒烟训练，只查环境、依赖、脚本和路径（约 10 秒）
SELFTEST_QUICK=0

# ---------------------------------------------------------------- 4) 审计
AUDIT_MIN_RAW_SIDE=300     # 显微原图短边小于这个像素数 → ⚠️（截不出 224 的框）
AUDIT_MIN_CROP_SIDE=64     # 截图短边小于这个像素数 → ⚠️（与建数据集的 DS_MIN_CROP 保持一致）

# ---------------------------------------------------------------- 5) 显微数据集（train_4_dataset.zsh）
# 默认值 = trainset_v4 的设置（2026-10-21）。建数据集时会把下面的当前值打印出来。
#
# DS_REPLACE_ALL_V2=1   某个菌种只要在 crops_v2（规范截图）里有截图，这个菌种的旧截图（dev_original、dev_checked）
#                       就不再使用，改用规范截图。菌落图不受影响。      → --replace_classes all_v2
#                       （和 DS_USE_REVIEW=1 一起用时，人工核对为"合格"的旧截图仍保留）
# DS_USE_REVIEW=1       用 04_review_train.py 的人工核对结果：标为"不合格""局部"的截图剔除，"合格"保留；
#                       "标签可疑"默认不剔除；新截的图没核对过，全部保留。    → --review review_images.csv
# DS_EXCLUDE_IMMATURE=0 crops_v2 里截图时按 i 标为"未成熟"（无孢子或孢子很少）的结构不用。 → --exclude_immature
# DS_MIN_CROP=64        截图边长小于这个像素数的不用。                       → --min_crop 64
DS_REPLACE_ALL_V2=1
DS_USE_REVIEW=1
DS_REVIEW_CSV="review_images.csv"      # 在 Audit/ 或 output/04_audit_review/ 下找
DS_EXCLUDE_IMMATURE=0
DS_MIN_CROP=64

# 建数据集时默认选中的截图来源（直接回车就用这些）。v4 = 旧截图 + 网络补截 + 规范截图，不含 ✅ 文件夹截图
#   dev_original 旧截图(O)  dev_checked ✅外院投稿截图(C)  crops_new 网络补截(N)  crops_v2 规范截图(V)  roi_ext 外部ROI(R)
DS_DEFAULT_SOURCES=(dev_original crops_new crops_v2)

# ---------------------------------------------------------------- 6) 显微模型训练（train_5_micro.zsh）
MICRO_BACKBONES=(swin_t densenet121 efficientnet_b0 resnet50 convnext_tiny vit_b_16)   # 可选的 backbone
DEFAULT_BACKBONES=(swin_t)                                                             # 直接回车时用的
CV_FOLDS=5          # 菌株级 StratifiedGroupKFold 折数
MAX_EPOCHS=30       # CV 每折最多训练几个 epoch（早停）
PATIENCE=8          # 连续多少个 epoch 没提升就早停
BATCH=32
IMG_SIZE=224
WORKERS=4
SEED=42
MICRO_EXTRA_ARGS=()   # 额外传给 02_train.py 的参数，一般留空（例：(--lr 5e-5)）；会记进 RUN_RECORD.md
# 最终模型（full）的 epoch 数不在这里设：自动取该 backbone CV 各折最佳 epoch 的中位数（与 v4 相同，事先确定）

# ---------------------------------------------------------------- 7) 菌落模型训练（train_6_colony.zsh）
COLONY_BACKBONE="efficientnet_b0"   # 方案 B 冻结特征用的网络：swin_t / densenet121 / resnet50 / efficientnet_b0 / resnet50 / convnext_tiny / vit_b_16
COLONY_MIN_STRAINS=3       # 株数少于它的组本轮不纳入（如 Nidulantes）
COLONY_DUP_INLIERS=300     # 疑似同一平皿的照片（特征点内点 ≥ 它）合并到同一 CV 组；0 = 不合并
COLONY_SKIP_B=0            # 1 = 不跑方案 B（COLONY_PYTHON 环境里没有 torch 时）
COLONY_RUN_C=0             # 1 = 加跑方案 C 端到端微调（探索性，较慢，不参与选择）
COLONY_WEIGHTS=""          # 方案 B 的离线权重路径；留空 = 自动下载 ImageNet 权重
COLONY_REGISTRY=""         # 菌株登记表.xlsx（可选，剔除 状态=剔除 的株）

# ---------------------------------------------------------------- 以下由上面推出，一般不用改
AUDIT="${TRAIN_KIT_DIR}"                                # = 脚本所在的 Audit 文件夹
OUT="${AUDIT}/output"
TRAINSETS="${OUT}/01_trainsets"
RUNS_ROOT="${OUT}/02_training_runs"
COLONY_RUNS="${RUNS_ROOT}/colony"
AUDIT_REVIEW="${OUT}/04_audit_review"
LOG_DIR="${OUT}/06_logs"
LEDGER="${LOG_DIR}/train_ledger.tsv"                    # 所有 train_*.zsh 调用 python 的总账
TAXONOMY="${AUDIT}/taxonomy_map.csv"
COLONY_PY="${COLONY_PYTHON:-${PYTHON}}"
