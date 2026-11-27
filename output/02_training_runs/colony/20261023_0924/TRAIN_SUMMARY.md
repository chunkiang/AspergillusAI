# 训练总结：20261023_0924

- 生成时间：2026-10-23 09:25（train_review.py summary）
- 运行记录：RUN_RECORD.md（本文件夹）

```
菌落模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/colony/20261023_0924
锁定时间：2026-10-23T09:25:25    版本：02_colony_train v1.2 (2026-10-23)
数据集：colony_20261023_0853（/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/colony_20261023_0853）
类别：Fumigati、Flavi、Nigri、Terrei；36 株，33 个 CV 组（留一组交叉验证）
选定方案 A（规则：A、B 中株级 OOF NLL 最低者）；温度 T = 1.150
方案 B 骨干网络：efficientnet_b0；min_strains=3；dup_inliers=300

方案         top1   top2  平衡准确率    NLL    ECE
--------------------------------------------------
A           0.944  1.000       0.950  0.268  0.061
A_无直径    0.944  1.000       0.950  0.299  0.059
A_仅中心块  0.944  1.000       0.933  0.185  0.028
B           0.833  0.917       0.794  0.512  0.039
B_仅中心块  0.778  0.861       0.716  0.725  0.074
final_model.joblib sha256：00abd99c684e1d27…
完整报告：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/colony/20261023_0924/metrics.txt

临床验证引用写法（clin_config.zsh 的 COLONY_MODEL）：colony/20261023_0924
```
