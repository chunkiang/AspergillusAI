#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_compare_runs.py —— 汇总 runs/ 下所有交叉验证结果，生成模型比较表（论文 Table 1 的原始数据）

用法：
    python 02b_compare_runs.py                          # 读取 output/02_training_runs/runs/*_cv_*/
    python 02b_compare_runs.py runs_20261012_加入规范的镰刀菌等等图片   # 比较另一批（02_training_runs 下的子文件夹名）
输出：
    runs/comparison.csv           每个模型：各折均值 ± SD、合并 OOF 指标、训练耗时
    runs/per_class_f1.csv         每类 F1 × 每个模型（看哪些类是共同的弱项）
"""
import os
import glob
import json
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ROOT = OP.find_run(sys.argv[1]) if len(sys.argv) > 1 else OP.RUNS
rows, per_class = [], {}
for d in sorted(glob.glob(os.path.join(ROOT, "*_cv_*"))):
    f = os.path.join(d, "cv_summary.json")
    if not os.path.exists(f):
        continue
    s = json.load(open(f, encoding="utf-8"))
    secs = 0.0
    for log in glob.glob(os.path.join(d, "fold_*", "train_log.csv")):
        secs += pd.read_csv(log)["seconds"].sum()
    name = s["backbone"]
    rows.append({
        "模型": name, "目录": os.path.basename(d), "完成折数": len(s["folds_done"]),
        "准确率": f"{s['acc_mean']:.3f} ± {s['acc_sd']:.3f}",
        "macro-F1": f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f}",
        "平衡准确率": f"{s['balanced_acc_mean']:.3f} ± {s['balanced_acc_sd']:.3f}",
        "Top-2": f"{s['top2_mean']:.3f} ± {s['top2_sd']:.3f}",
        "_f1": s["macro_f1_mean"],
        "最佳epoch中位数": s["median_best_epoch"],
        "训练总时长(分钟)": round(secs / 60, 1),
    })
    pc = os.path.join(d, "per_class.csv")
    if os.path.exists(pc):
        t = pd.read_csv(pc, index_col=0, encoding="utf-8-sig")
        t = t.drop(index=[i for i in ["accuracy", "macro avg", "weighted avg"] if i in t.index])
        per_class[name] = t["f1-score"]
        per_class.setdefault("验证张数", t["support"])

if not rows:
    print(f"{ROOT} 下没有找到交叉验证结果")
else:
    df = pd.DataFrame(rows).sort_values("_f1", ascending=False).drop(columns="_f1")
    df.to_csv(os.path.join(ROOT, "comparison.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(OP.table(df, index=False))
    if per_class:
        pcf = pd.DataFrame(per_class)
        models = [c for c in pcf.columns if c != "验证张数"]
        pcf = pcf.sort_values(models[0])
        pcf.to_csv(os.path.join(ROOT, "per_class_f1.csv"), encoding="utf-8-sig")
        print("\n每类 F1（按第一个模型从低到高排序，前 10 个最弱的类）：")
        print(OP.table(pcf.head(10).round(3)))
    print(f"\n已写出 {ROOT}/comparison.csv、{ROOT}/per_class_f1.csv")
