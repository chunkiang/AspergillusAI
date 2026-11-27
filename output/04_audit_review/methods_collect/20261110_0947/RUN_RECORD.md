# Methods 资料收集

- 时间：2026-11-10 09:47:42；电脑：ChunkiangMBP-M3.local；外层脚本：methods_collect.zsh
- 终端日志：/Users/chunjiang/Documents/真菌AI/Audit/output/06_logs/methods_collect_20261110_094742.log

## 依次运行的 python 脚本与参数

- 显微模型 MICRO_MODEL=runs_v4/swin_t_full_final
- 菌落模型 COLONY_MODEL=colony/20261020_2108
- 备用数据集 MICRO_TRAINSET=trainset_v4
- 外部图像 EXT_ROOT=/Users/chunjiang/Desktop/丝状真菌

### 00_methods_collect.py  ✅
- 时间：2026-11-10 09:47:47（用时 5 秒，退出码 0）
- 外层脚本：methods_collect.zsh（sha256 3e351ad8b0ab9ca3c24ba1aeb476a9d41781968f29b86b118a2e0b4cc71935ee）
- python 脚本：00_methods_collect.py（sha256 6b673d22dc9d26ad66369e85de790851e13e04e413353f547e804aa59f9eb5c0）
- python：/Users/chunjiang/miniforge3/envs/fungal/bin/python
- 完整命令：
  ```
  cd /Users/chunjiang/Documents/真菌AI/Audit
  python 00_methods_collect.py --audit /Users/chunjiang/Documents/真菌AI/Audit --out_root /Users/chunjiang/Documents/真菌AI/Audit/output --micro_model runs_v4/swin_t_full_final --micro_trainset trainset_v4 --colony_model colony/20261020_2108 --recrop_root /Users/chunjiang/Documents/真菌AI/原图_重截 --colony_raw /Users/chunjiang/Documents/真菌AI/CNN/补拍_2026 --ext_root /Users/chunjiang/Desktop/丝状真菌 --colony_python /Users/chunjiang/miniforge3/envs/ai/bin/python --clin_micro_model runs_v4/swin_t_full_final --clin_colony_model colony/20261020_2108 --out /Users/chunjiang/Documents/真菌AI/Audit/output/04_audit_review/methods_collect/20261110_0947
  ```

