# 临床验证运行记录 PILOT_20261118_134439

- 开始时间：2026-11-18 13:44:39
- 模式：试运行 PILOT=1（不能写进论文）
- 电脑：Mac.lan；python：/Users/chunjiang/miniforge3/envs/ai/bin/python 3.11.14
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）；配置 clin_config.zsh（sha256 a0f190b484bf5e1787f66894217f02648870366b5adf1c2cd6e1313dce270d9e）
- 镜检模型：runs_v4/swin_t_full_final
- 菌落 python：/Users/chunjiang/miniforge3/envs/ai/bin/python
- 菌落模型：colony/20261020_2108（final_model.joblib sha256 a0bfde8db2feb3024aa9e2e71d27b0a434e05ccc6da74f07644fc0520d0d3b47）
- 判读规则：P_MIN=0.70 GAP_MIN=0.20 ASP_MIN=0.5 ALPHA=0.8（曲霉融合：菌落权重，线性加权）
- 素材：/Users/chunjiang/Documents/真菌AI/临床验证_2026

## 依次运行的 python 脚本与参数

### 00_clin_check.py  ✅
- 时间：2026-11-18 13:44:40（用时 1 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：00_clin_check.py（sha256 7f233aba24768d09bd310a90b19df9feab99e457dac4156a86f084266a8a967c）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 00_clin_check.py --micro_dir /Users/chunjiang/Documents/真菌AI/临床验证_2026/00_raw/micro --colony_dir /Users/chunjiang/Documents/真菌AI/临床验证_2026/00_raw/colony --answers /Users/chunjiang/Documents/真菌AI/临床验证_2026/answers/answers_master.csv --micro_model runs_v4/swin_t_full_final --id_regex '^PUPH-PCL-[0-9]{3}$' --inventory /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/inventory.csv --split --require_answers
  ```

### 03_roi_clinical.py  ✅
- 时间：2026-11-18 13:44:40（用时 0 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：03_roi_clinical.py（sha256 24b9d0ae8b30d2b81e06b10cf6f1f6305be56cb411d1ef490632e7b7bf5d02dd）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 03_roi_clinical.py --src /Users/chunjiang/Documents/真菌AI/临床验证_2026/00_raw/micro --out /Users/chunjiang/Documents/真菌AI/临床验证_2026/01_roi --unblind --answers /Users/chunjiang/Documents/真菌AI/临床验证_2026/answers/answers_clinical.csv --force
  ```

### 03_eval_clinical.py  ✅
- 时间：2026-11-18 13:44:53（用时 13 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：03_eval_clinical.py（sha256 cdf81670675272f7ea99e635118f2da79489df8dcf30281be5f436208aecfe06）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 03_eval_clinical.py --models runs_v4/swin_t_full_final --roi_dir /Users/chunjiang/Documents/真菌AI/临床验证_2026/01_roi --out /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/micro --no_lock --p_min 0.70 --gap_min 0.20
  ```

### 02b_plot_confusion.py  ✅
- 时间：2026-11-18 13:44:55（用时 2 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：02b_plot_confusion.py（sha256 52cdcc34c28da9d92e84115a542c105549899ebaecfca52019bf83bc45a403e5）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 02b_plot_confusion.py /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/micro/confusion.csv --title 'Clinical validation (PILOT) — microscopy model, isolate level'
  ```

### 02b_plot_confusion.py  ✅
- 时间：2026-11-18 13:44:56（用时 1 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：02b_plot_confusion.py（sha256 52cdcc34c28da9d92e84115a542c105549899ebaecfca52019bf83bc45a403e5）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 02b_plot_confusion.py /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/micro/confusion_roi.csv --title 'Clinical validation (PILOT) — microscopy model, ROI level'
  ```

### 00_clin_selftest.py  ✅
- 时间：2026-11-18 13:44:58（用时 2 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：00_clin_selftest.py（sha256 e582ebc2c634bdae54b883d6add41e2d7ffdf4d6040a9455993b3b519f14bcdb）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  /Users/chunjiang/miniforge3/envs/ai/bin/python 00_clin_selftest.py envcheck --colony_model output/02_training_runs/colony/20261020_2108
  ```

### 01_colony_preprocess.py  ✅
- 时间：2026-11-18 13:45:17（用时 19 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：01_colony_preprocess.py（sha256 bf26dcdf719534eb97f03f070ce4ae632a5ed6684fa19acd43f4a14a6af8d7e5）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  /Users/chunjiang/miniforge3/envs/ai/bin/python 01_colony_preprocess.py build --root /Users/chunjiang/Documents/真菌AI/临床验证_2026/00_raw/colony --ann_dir /Users/chunjiang/Documents/真菌AI/临床验证_2026/02_colony_annotations --out /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/colony_preprocessed --views O
  ```

### 03_colony_predict.py  ✅
- 时间：2026-11-18 13:45:20（用时 3 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：03_colony_predict.py（sha256 076d09b7f78354f925dbda709dfd4b4ebe590251c0e671b1c865a083b9615727）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  /Users/chunjiang/miniforge3/envs/ai/bin/python 03_colony_predict.py --model colony/20261020_2108 --pre /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/colony_preprocessed --answers /Users/chunjiang/Documents/真菌AI/临床验证_2026/answers/answers_colony.csv --out /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/colony --t1 0.70 --t2 0.20
  ```

### 03_clin_fusion.py  ✅
- 时间：2026-11-18 13:45:21（用时 1 秒，退出码 0）
- 外层脚本：clin_5_eval.zsh（sha256 bfeb9c43bd4f788c51bda94b061dbf40700d8c9002698e79a5b2fc4b79703c2a）
- python 脚本：03_clin_fusion.py（sha256 99c0c6dcb10afe6205c7164027d6597176f79d7ac3804fdc21410da524ed30fb）
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 03_clin_fusion.py --out /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/final --answers /Users/chunjiang/Documents/真菌AI/临床验证_2026/answers/answers_master.csv --asp_min 0.5 --alpha 0.8 --p_min 0.70 --gap_min 0.20 --micro /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/micro --colony /Users/chunjiang/Documents/真菌AI/Audit/output/03_external_eval/clinical/PILOT_20261118_134439/colony
  ```

## 结果

- 镜检：micro/metrics.txt
- 菌落：colony/metrics.txt
- 最终：final/report.txt、final/per_isolate_final.csv
- 结束时间：2026-11-18 13:45:21
