# 🍄 真菌 AI 识别项目使用总览

> 📌 **文档定位**：本 README 汇总了 `Audit` 目录下全部现有使用说明与最新脚本流程，涵盖**模型训练一键流程**、**前瞻临床验证一键流程**、**回顾性外部验证与 LOSO**、**同节随机配对融合分析**以及**底层编号脚本速查**。
>
> 📅 **汇总日期**：2026-11-27｜💻 **运行环境**：macOS（Apple Silicon MPS 加速）

---

## 📚 现有说明文档索引

本仓库已有以下专项文档，本 README 已将其核心内容整合；需要查看更细的参数或历史对照时，可直接查阅原文件：

| 文档 | 核心内容 |
|---|---|
| 🚀 [训练一键流程_使用说明_2026-10-22.md](./训练一键流程_使用说明_2026-10-22.md) | 显微 + 菌落模型训练全流程（`train_0` ~ `train_6` + `train_review`） |
| 🏥 [使用说明_临床验证.md](./使用说明_临床验证.md) | 前瞻临床双模态验证全流程（`clin_0` ~ `clin_5` + 锁定机制） |
| 📖 [脚本说明.md](./脚本说明.md) | `00_` ~ `99_` 底层编号脚本参数、快捷键、工作流与旧名对照 |
| 📋 [脚本目录.txt](./脚本目录.txt) | 编号脚本单页速查表 |
| 📂 [output/训练流程_目录说明.txt](./output/训练流程_目录说明.txt) | 训练输入图片位置与 `output/` 输出目录速查 |

---

## 🧠 核心识别架构

所有丝状真菌菌株均做**镜检盲法 ROI 截图**，由镜检模型输出 **26 类**概率：
- **非曲霉通路**（`P(曲霉) < 0.5`）：直接采用镜检结果，报告到**种**。
- **曲霉通路**（`P(曲霉) ≥ 0.5`）：将镜检概率按曲霉组（Section）汇总，与**菌落模型**的组概率进行线性加权融合（`03_clin_fusion.py v1.3` 默认 `α = 0.8`，即**菌落为主、镜检为辅**），报告到**组**并附**种提示**。

```text
          ┌── 🔬 镜检 ROI ──► 镜检模型 (26类) ──┬─ P(曲霉) < 0.5 ──► 非曲霉：报告到种
所有菌株 ─┤                                      └─ P(曲霉) ≥ 0.5 ──┐
          └── 🎨 菌落正面照 ─► 菌落模型 (组级) ────────────────────► 融合 s = (1−α)·s镜检 + α·s菌落 ──► 曲霉：报告到组 + 种提示
```

> ⚠️ **复核触发条件**（满足任一项即标为需人工复核）：
> - `top1 < 0.70` 或 `top1 − top2 < 0.20`
> - 镜检首位组与菌落首位组不一致
> - 菌落特征超出训练集分布范围（`D²` 超限）或曲霉株缺失菌落照片（仅报告 `Aspergillus spp.`）
> - 镜检首位组不在菌落模型覆盖组内（如 `Nidulantes`、`Other`）
> - 💡 需复核且涉及 **Fumigati / Terrei / Flavi** 时，系统自动建议加做 **50 °C 生长试验**。

---

## ⚙️ 运行环境与目录约定

### 1️⃣ 每次运行前的标准起手式

```zsh
conda activate fungal
cd /Users/chunjiang/Documents/真菌AI/Audit
```

- 🔬 **显微分支**：默认使用 `fungal` 环境的 `python`（PyTorch + Apple MPS）。
- 🎨 **菌落分支**：通过 `COLONY_PYTHON` 单独指定（当前配置为 `/Users/chunjiang/miniforge3/envs/ai/bin/python`）。
  > 🚨 **关键提醒**：`train_config.zsh` 与 `clin_config.zsh` 中的 `COLONY_PYTHON` **必须保持完全一致**，确保训练与临床验证使用相同版本的 `scikit-learn`！
- 🔍 任意 `.py` 脚本均可加 `-h` 查看完整参数（如 `python 02_train.py -h`）。

### 2️⃣ 目录放置与 `paths.py` 短路径规则

> ❗ **平铺铁律**：所有 `00_`~`99_` 脚本、`train_*.zsh`、`clin_*.zsh`、`fsim_*.zsh`、`paths.py`、`taxonomy_map.csv`、`answers.csv` **必须直接平铺在 `Audit/` 根目录**，不要再套子文件夹，也**不要重命名 `paths.py`**。

```text
Audit/
├── 00_*.py … 99_*.py, paths.py           # 底层编号脚本与公共路径约定
├── train_*.zsh, train_config.zsh         # 🚀 模型训练一键流程
├── clin_*.zsh, clin_config.zsh           # 🏥 前瞻临床验证一键流程
├── fsim_*.zsh, fsim_config.zsh           # 📊 同节随机配对融合分析
├── methods_collect.zsh                   # 📝 论文 Methods 数据一键收集
├── taxonomy_map.csv, answers*.csv        # 菌种映射与参考答案表
├── 真菌逐图判读台.html                     # 🖥️ 浏览器交互式逐图判读台
└── output/                               # 所有结果按步骤编号归档
    ├── 01_trainsets/                     # 显微/菌落训练集、标注与预处理结果
    ├── 02_training_runs/                 # 交叉验证、最终模型、菌落模型、融合模拟
    ├── 03_external_eval/                 # 回顾性外部验证、LOSO、临床验证结果
    ├── 04_audit_review/                  # 数据审计、人工核对、Methods 收集
    ├── 05_roi_diagnostics/               # ROI 深入诊断与颜色偏移分析
    └── 06_logs/                          # 终端日志 + 总账表 (*_ledger.tsv)
```

✨ **`paths.py` 短路径简写**：命令行里只需写短名，脚本会自动去对应编号目录查找（写绝对路径或当前目录存在的路径则按原样使用）：
- `--trainset trainset_v4` → `output/01_trainsets/trainset_v4`
- `--models runs_v4/swin_t_full_final` → `output/02_training_runs/runs_v4/swin_t_full_final`
- `--out ext_eval_roi_v4` → `output/03_external_eval/ext_eval_roi_v4`
- `--review review_images.csv` → `output/04_audit_review/review_images.csv`

### 3️⃣ 原始图像数据路径速查（Mac 本地，位于 `output/` 之外）

| 用途 | 本地路径 |
|---|---|
| 🔬 显微旧截图 & ✅ 文件夹 | `/Users/chunjiang/Documents/真菌AI/CNN/dataset` |
| 🔬 显微规范截图（主训练源） | `/Users/chunjiang/Documents/真菌AI/CNN/crops_v2/` |
| 🔬 显微网络补截 | `/Users/chunjiang/Documents/真菌AI/CNN/crops_new/` |
| 🔬 外部盲法 ROI 测试集 | `/Users/chunjiang/Documents/真菌AI/CNN/roi_test/` |
| 🔬 待重截显微原图（按菌种/菌株） | `/Users/chunjiang/Documents/真菌AI/原图_重截/<菌种>/<菌株>/` |
| 🎨 菌落原始照片（新标准） | `/Users/chunjiang/Documents/真菌AI/CNN/补拍_2026/<组>/` |
| 🎨 旧菌落图（仅审计用） | `/Users/chunjiang/Documents/真菌AI/CNN/normal` |
| 🏥 回顾性外部原图 | `/Users/chunjiang/Desktop/丝状真菌` |
| 🏥 前瞻临床验证素材库 | `/Users/chunjiang/Documents/真菌AI/临床验证_2026` |
| 🗑️ 清理隔离区（仅移动不删除） | `/Users/chunjiang/Documents/真菌AI/CNN/_quarantine/` |

---

## 🚀 工作流一：模型训练一键流程（`train_*.zsh`）

适用于**加入新图 → 质控审计 → 截图/标注 → 构建数据集 → 训练显微与菌落模型**。
🔧 **唯一配置文件**：[`train_config.zsh`](./train_config.zsh)（所有改动都会自动记入 `RUN_RECORD.md` 与 `DATASET_INFO.json`）。

```text
train_0_selftest ─► train_1_init ─► train_2_audit ─► train_3_crop
─► train_4_dataset（先显微，再菌落）─► train_5_micro ─► train_6_colony
                      🔎 随时查看：zsh train_review.zsh
```

| 步骤 | 命令 | 核心功能与注意事项 |
|---|---|---|
| **0. 自检** | `zsh train_0_selftest.zsh` | 检查脚本、依赖、MPS、`COLONY_PYTHON` 一致性与磁盘空间；默认含约 2 分钟冒烟训练（设 `SELFTEST_QUICK=1` 可跳过） |
| **1. 初始化** | `zsh train_1_init.zsh` | 仅创建缺失文件夹，不覆盖任何已有文件 |
| **2. 入库与审计** | `zsh train_2_audit.zsh [文件夹…]` | 只复制不移动、先预演再确认；检测坏图、跨菌种重复、命名错误等并给 💡 建议（加 `full` 强制重跑全量审计） |
| **3. 截图/标注** | `zsh train_3_crop.zsh` | 交互菜单：① 显微规范截图 ② 菌落几何标注 ③ 菌落白平衡确认 ④ 标注+白平衡连做（随时按 `q` 断点续跑） |
| **4. 建数据集** | `zsh train_4_dataset.zsh` | 按来源代号构建显微集（默认 `O+N+V`，同 `trainset_v4`）与菌落集，生成 `DATASET_INFO.json` |
| **5. 训显微模型** | `zsh train_5_micro.zsh` | 支持多选 backbone、5 折菌株级 CV / 最终模型（epoch 自动取 CV 最佳中位数）；自动跑错误分析、混淆矩阵与自检 |
| **6. 训菌落模型** | `zsh train_6_colony.zsh` | 留一组 CV 对比方案 A（颜色+直径）与方案 B（冻结特征），按最低 NLL 锁定并提示 `COLONY_MODEL` 填法 |
| **🔎 随时盘点** | `zsh train_review.zsh` | 交互查看 ① 原始图片与截图进度 ② 所有数据集 ③ 所有模型指标与参数（支持旧版兼容） |

### 🏷️ 显微数据集来源代号说明（`train_4_dataset.zsh`）
- **`O`**（`dev_original`）：旧截图
- **`C`**（`dev_checked`）：✅ 外院投稿截图（选择时会二次确认）
- **`N`**（`crops_new`）：网络补截
- **`V`**（`crops_v2`）：规范截图（核心主力）
- **`R`**（`roi_ext`）：外部 ROI —— 🚨 **红色警告**：纳入 `R` 训练的模型**严禁再在 `roi_test` 上做外部评估**！

---

## 🏥 工作流二：前瞻临床验证一键流程（`clin_*.zsh`）

适用于**严格盲法入组、双模态预测、融合报告与可追溯归档**。
🔧 **唯一配置文件**：[`clin_config.zsh`](./clin_config.zsh)（先确认 4 项：`CLIN_ROOT`、`MICRO_MODELS`、`COLONY_MODEL`、`PILOT`，以及 `COLONY_PYTHON`）。

### 1️⃣ 素材摆放与命名铁律（最重要！）

```text
临床验证_2026/
├── 00_raw/
│   ├── micro/                      # 🔬 镜检：每株一个子文件夹，文件夹名 = 株号
│   │   ├── PUPH-PCL-001/           #    内放照片，建议命名 PUPH-PCL-001-C-01.jpg …
│   │   └── PUPH-PCL-002/ …
│   └── colony/                     # 🎨 菌落：全部照片平铺在一层，不建子文件夹！
│       ├── PUPH-PCL-001-B-O-1.jpg  #    O = 正面（参与模型）
│       └── PUPH-PCL-001-B-R-1.jpg  #    R = 反面（可选，不参与模型）
├── answers/answers_master.csv      # 📝 唯一手填答案表（clin_2 自动生成/追加）
├── 01_roi/                         # 🤖 自动生成，勿手改
└── 02_colony_annotations/          # 🤖 自动生成，勿手改
```

- 🆔 **统一株号**：严格使用 `PUPH-PCL-三位数字`（如 `PUPH-PCL-001`），镜检和菌落靠它自动对齐。
- 🙈 **严禁泄密**：文件夹名和文件名中**绝对不能出现菌名**（`clin_2` 会自动拦截）。
- 📸 **拍摄标准**（须与训练集一致）：
  - 🔬 镜检：乳酸酚棉蓝染色，每株 3–6 个视野，包含成熟完整产孢结构。
  - 🎨 菌落：SDA，28 °C，**第 3 天**，正面，平皿置于**白色 A4 纸**上（作白平衡参照），每株 1–2 张。建议**所有入组丝状真菌均拍菌落**以避免选择偏倚。
- 📝 **答案表 `answers_master.csv`**（另存为 CSV UTF-8，最好由不参与截图者填写或全部截完后填写）：
  - `reference`：参考鉴定名（必填）｜`train_class`：照抄 `clin_1` 列出的类名，**集外菌种留空**｜`section`：曲霉可留空自动推断｜`method`：`MALDI-TOF` / `ITS` / `ITS+BenA`。

### 2️⃣ 临床验证运行顺序

| 步骤 | 命令 | 交互 | 说明 |
|---|---|---|---|
| **0. 自检** | `zsh clin_0_selftest.zsh` | 否 | 无需临床素材，用训练数据真跑全流程（含融合）；看到 🎉 全部 ✅ 再开始（结果不可写进论文） |
| **1. 初始化** | `zsh clin_1_init.zsh` | 否 | 建目录、查环境与模型、打印镜检类名与曲霉组对应表（只需一次） |
| **🔒 锁定** | `zsh clin_lock.zsh` | 否 | **正式验证前运行且仅运行一次**，写入模型与规则指纹至 `clinical_lock.json`（试运行跳过） |
| **2. 盘点** | `zsh clin_2_check.zsh` | 否 | 检查命名与泄密风险、盘点素材、建立或追加 `answers_master.csv` |
| **3. 盲法 ROI** | `zsh clin_3_roi.zsh` | **是** | 镜检盲法截图；运行 `zsh clin_3_roi.zsh status` 可随时查进度 |
| **4. 菌落处理** | `zsh clin_4_colony.zsh` | **是** | 菌落几何标注 + 白平衡参照确认（只处理正面照） |
| **5. 一键评估** | `zsh clin_5_eval.zsh` | 否 | 揭盲 → 镜检预测 → 菌落预测 → 双模态融合 → 输出完整报告与快照（每次新建目录不覆盖） |

> 💡 **后期追加新菌株**：把新照片放入 `00_raw/`，按 `2 → 3 → 4 → 5` 重跑即可，已截的图与已做的标注会自动保留。

### 3️⃣ 试运行（`PILOT=1`）vs 正式验证（`PILOT=0`）

| 对比项 | 🧪 `PILOT=1` 试运行 | 🏛️ `PILOT=0` 正式验证 |
|---|---|---|
| 锁定校验 | 不核对 `clinical_lock.json` | 强制核对，模型/阈值/脚本有任何改动即拒绝运行 |
| 揭盲门槛 | 未截完也可运行 | 必须全部截完或显式跳过 |
| 输出目录名 | `PILOT_<时间戳>/` | `<时间戳>/`（无前缀） |
| 能否写入论文 | ❌ 不可（且试运行株**不能**再计入正式验证） | ✅ 可以 |

### 4️⃣ 结果重点看什么？

位于 `output/03_external_eval/clinical/<运行目录>/`：
- 🏆 **`final/report.txt`**：**主报告**（最终准确率、曲霉配对比较 McNemar 检验、分流准确率、集外菌种与错判株明细）
- 📊 **`final/per_isolate_final.csv`**：每株一行（通路、最终报告、是否复核、复核原因、50 °C 试验建议）
- 🖼️ **`colony_preprocessed/qc_sheets/`**：菌落白平衡校正质控拼图（强烈建议肉眼复核一遍）
- 🧾 **`RUN_RECORD.md` & `_snapshot/`**：完整命令、所有脚本与模型 SHA-256 指纹、配置与答案表快照

---

## 📊 工作流三：论文验证、融合模拟与实用工具

### 1️⃣ v4 显微标准复现流程（不含 ✅ 外院投稿截图）
```zsh
zsh 02_run_cv_v4.zsh           # ① 建 trainset_v4 + 6 个 backbone 5折 CV + 错误分析/混淆矩阵
zsh 02_train_final_v4.zsh      # ② 取 CV 最佳 epoch 中位数训练 swin_t 最终模型 + 记录权重指纹
zsh 03_eval_roi_v4.zsh         # ③ 在 roi_test 上做严格单次外部验证（自带防泄漏核对与防重跑保护）
zsh 03_run_loso_v4.zsh         # ④ 按来源留一（LOSO）外部验证（默认 TP HYY MQQ LYF WEB1 WEB2）
```

### 2️⃣ 同节随机配对融合增益分析（`fsim_*.zsh`）
在无配对临床数据前，利用镜检与菌落的 OOF 预测按同节随机配对（默认 1000 次）估计双模态融合增益与权重敏感性：
- 🔧 配置文件：[`fsim_config.zsh`](./fsim_config.zsh)（默认 `linear`、`COLONY_WEIGHT=0.8`、`ASP_GATE=0.5`）
```zsh
zsh fsim_0_check.zsh           # 第 0 步：自动定位 OOF 表并生成 LATEST_section_map.csv 供人工核对
zsh fsim_1_run.zsh             # 第 1 步：执行随机配对分析 → output/02_training_runs/fusion_sim/<时间>/report.txt
```

### 3️⃣ 论文资料收集、结果打包与可视化台
- 📝 **收集论文 Methods 数据**（只读，输出 6 段参数并自动打 zip）：
  ```zsh
  zsh methods_collect.zsh      # 输出到 output/04_audit_review/methods_collect/<时间>/
  ```
- 📦 **一键打包全部统计结果与代码**（仅复制、不改原文件、排除大权重文件）：
  ```zsh
  python3 collect_fungal_results.py --root .
  ```
- 🖥️ **逐图判读与图片浏览**：
  - 双击打开 [`真菌逐图判读台.html`](./真菌逐图判读台.html)：导入 CSV 即可在浏览器中按病例/概率热图逐图判读。
  - `python 04_train_gallery.py --root output/01_trainsets`：本地网页按菌种（支持中文别名）浏览训练集图片。

---

## ⌨️ 交互式截图与标注快捷键速查

### 🔬 显微截图工具（`01_crop_batch_training.py` / `03_roi_blind.py` / `03_roi_clinical.py`）
| 按键 | 功能 |
|---|---|
| `鼠标左键` | 截取当前框（规范：1 框 = 1 个完整产孢结构，每图 3–8 个） |
| `+` / `-` 或 `右键` | 放大 / 缩小截取框（96–800 px） |
| `←` / `→`（或 `,` / `.`） | 逆 / 顺时针旋转 10°；`↑` 回正 |
| `1` / `2` / `3`、`i` | （仅训练规范截图）`1` 完整、`2` 局部、`3` 多结构；`i` 切换成熟 ↔ 未成熟 |
| `u` | 撤销本图最后一次截取 |
| `n` / `空格`；`b` | 下一张；上一张 |
| `s` | 跳过本图（无可用结构/质量差） |
| `q` | 保存并退出（下次自动断点续跑） |

### 🎨 菌落几何标注（`01_colony_annotate.py`）& 白平衡确认（`01_colony_preprocess.py pick`）
- **几何标注**：`左键` 点/拖 → `Enter` 确认下一步 → `Backspace` 返回上一步｜`方向键` / `+-` 微调平皿｜`f` 铺满平皿｜`m` 多菌落｜`c` 污染剔除｜`b` 不合格剔除｜`z` 撤销标签框｜`q/Esc` 退出。
- **白平衡确认**（输入法切英文）：`左键点击/拖框` 选白纸参照区｜`a` 切换白纸/培养基模式｜`t` 开关**文字遮罩**（遮挡菌名/手写字/标签）｜`x` 无可用参照｜`r` 恢复程序建议｜`z` 撤销｜`Enter` 保存下一张｜`p`/`n` 上/下一张。

---

## 🗂️ 底层编号脚本速查（`00_` ~ `99_`）

| 编号 | 阶段 | 核心脚本 | 主要输出目录 |
|---|---|---|---|
| **00_** | 🔍 数据审计与清单 | `00_audit_train.py`, `00_clean_train.py`, `00_audit_external.py`, `00_build_manifest.py`, `00_colony_inventory.py`, `00b_colony_inventory.py`, `00_train_check.py`, `00_clin_check.py`, `00_clin_selftest.py`, `00_fusion_sim_check.py`, `00_methods_collect.py` | `output/04_audit_review/` |
| **01_** | ✂️ 截图/标注与建集 | `01_make_recrop_dirs.py`, `01_crop_batch_training.py`, `01_crop_batch.py`, `01_crop_helper.py`, `01_build_trainset.py`, `01_colony_annotate.py`, `01_colony_preprocess.py` | `CNN/crops_v2/`、`output/01_trainsets/` |
| **02_** | 🏋️ 模型训练 | `02_train.py`, `02_colony_train.py`, `02_run_cv_v4.zsh`, `02_train_final_v4.zsh`, `02_run_all.sh`, `02_run_all_timestamp.sh` | `output/02_training_runs/` |
| **02b_** | 📈 训练内部分析 | `02b_analyze_errors.py`, `02b_analyze_errors_batch.sh`, `02b_plot_confusion.py`, `02b_compare_runs.py`, `02b_oof_by_site.py` | 写回对应训练目录 |
| **03_** | 🧪 外部与临床验证 | `03_roi_blind.py`, `03_roi_clinical.py`, `03_sanity_check.py`, `03_eval_external.py`, `03_eval_clinical.py`, `03_colony_predict.py`, `03_clin_fusion.py`, `03_fusion_simpair.py`, `03_eval_roi_v4.zsh`, `03_run_loso_v4.zsh` | `output/03_external_eval/` |
| **04_** | 👁️ 人工核对与查漏 | `04_review_train.py`, `04_review_sites.py`, `04_check_roi_overlap.py`, `04_train_gallery.py` | `output/04_audit_review/` |
| **05_** | 🔬 ROI 深入诊断 | `05_roi_analysis.py` | `output/05_roi_diagnostics/` |
| **99_** | 🛠️ 维护工具 | `99_organize_outputs.py`（归档旧目录）, `99_check_versions.py`（脚本版本指纹核对） | — |

---

## 🧾 可追溯性与防错机制

- 📒 **三大调用总账**（Excel 可直接打开，记录每次运行的时间、脚本 SHA-256、退出码、耗时与完整命令）：
  - `output/06_logs/train_ledger.tsv`（训练流程）
  - `output/06_logs/clin_ledger.tsv`（临床验证）
  - `output/06_logs/fsim_ledger.tsv`（融合模拟）
- 📄 **自动留痕**：每次训练/建集/评估都会在输出目录内写入 `RUN_RECORD.md`、`TRAIN_SUMMARY.md` / `DATASET_INFO.json` 及当时配置文件的副本。
- 🛡️ **失败隔离**：中途出错或按 `Q` 退出的新建数据集、菌落模型或临床评估目录会自动追加 **`_FAILED`** 后缀，避免误用（显微 CV 例外，保留原目录方便断点续跑）。

---

## ❓ 常见问题与避坑指南（FAQ）

1. **💥 报错：`'LogisticRegression' object has no attribute 'multi_class'`**
   - **原因**：菌落模型训练环境与运行环境的 `scikit-learn` 版本不一致。
   - **解决**：运行 `zsh clin_0_selftest.zsh` 查看本机各 conda 环境的版本匹配建议，并将对应路径填入 `clin_config.zsh` 与 `train_config.zsh` 的 `COLONY_PYTHON`。
2. **⚠️ 菌落模型没有 `Nidulantes`（仅覆盖 4 个组）**
   - **原因**：训练时 `Nidulantes` 株数 `< COLONY_MIN_STRAINS`（默认 3 株）被自动剔除。
   - **处理**：融合时若镜检首位组为未覆盖组，系统不强行融合，自动报告 `Aspergillus spp.` 并标为需复核；后续补齐菌株重训菌落模型后会自动生效。
3. **❌ `train_2_audit.zsh` 提示 `00_train_check.py（退出码 2）`**
   - 退出码 `2` **不是程序崩溃**，而是审计发现了需处理的 ❌ 数据问题。打开 `output/04_audit_review/train_check/<时间>/report.txt`，按每条问题下的 💡 建议修正后重跑即可。
4. **🙈 盲法铁律与防数据泄漏**
   - 外部盲法截图（`03_roi_blind.py`）完成前**绝不要打开 `roi_key.csv`**。
   - 含 `--add_roi`（来源 `R`）的训练集**绝不能**在 `roi_test` 上评估，必须改用 `03_run_loso_v4.zsh` 做留一验证，或用 `04_check_roi_overlap.py` 核查重叠。
5. **🔄 想重训同名模型怎么办？**
   - 脚本默认保护已有成果、不会覆盖。如需重训，请先将对应输出文件夹改名或移走后再运行。
6. **⚠️ 临床验证时菌落分支失败，但评估仍跑完了？**
   - 这是有意设计的降级保护（退回仅镜检并标复核）。最常见原因是漏做了 `clin_4_colony.zsh` 的第②步（白平衡确认），补做后重跑第 5 步即可。
