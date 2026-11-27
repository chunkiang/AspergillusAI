#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_analyze_errors.py —— 交叉验证错误分析：错成了什么、是不是整株/整批来源都错、错分图长什么样

用法：
    python 02b_analyze_errors.py runs/densenet121_cv_20261009_1529
输出（在该 run 目录下）：
    error_summary.txt     每类召回率、主要被错分成哪些类；按分组（菌株/来源）统计的准确率
    confusion.csv         混淆矩阵（行=真实，列=预测）
    errors.html           错分图片缩略图（浏览器打开，按真实类别分组，显示预测类别和概率）
"""
import os
import sys
import html
import json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

run = OP.find_run(sys.argv[1]) if len(sys.argv) > 1 else OP.latest_cv_run()
o = pd.read_csv(os.path.join(run, "oof_predictions.csv"), encoding="utf-8-sig")
names = json.load(open(os.path.join(run, "class_names.json"), encoding="utf-8"))
short = {c: c.replace("Aspergillus ", "A. ").split(" (")[0] for c in names}
pcols = [f"p::{c}" for c in names]
o["correct"] = o["train_class"] == o["pred_class"]
o["p_true"] = [r[f"p::{r['train_class']}"] for _, r in o.iterrows()]
# 正确类别排第几
rank = o[pcols].rank(axis=1, ascending=False)
o["true_rank"] = [int(rank.at[i, f"p::{c}"]) for i, c in zip(o.index, o["train_class"])]

L = []
L.append(f"run: {run}    验证图片: {len(o)}    准确率: {o['correct'].mean():.3f}\n")
L.append("=== 每类：召回率 / 主要错分去向 / 错分时正确答案排第几 ===")
for c in names:
    d = o[o["train_class"] == c]
    if not len(d):
        continue
    wrong = d[~d["correct"]]
    dest = wrong["pred_class"].map(short).value_counts().head(3)
    dest_s = "、".join(f"{k}×{v}" for k, v in dest.items()) or "—"
    rk = wrong["true_rank"].value_counts().sort_index().head(3)
    rk_s = " ".join(f"第{k}名×{v}" for k, v in rk.items()) or ""
    flag = "  ←" if len(d) and d["correct"].mean() < 0.8 else ""
    L.append(f"{short[c]:28s} n={len(d):3d}  召回 {d['correct'].mean():.2f}  错成: {dest_s:40s} {rk_s}{flag}")
L.append("\n=== 被误判进来最多的类（精确率低的原因） ===")
fp = o[~o["correct"]].groupby("pred_class")["train_class"].apply(
    lambda s: "、".join(f"{short[k]}×{v}" for k, v in s.value_counts().head(4).items()))
for c, s in fp.sort_values(key=lambda x: x.str.len(), ascending=False).head(8).items():
    L.append(f"{short[c]:28s} ← {s}")
L.append("\n=== 按分组（菌株 / 原图 / 来源文件夹）统计：整组都错说明是“没见过的菌株”泛化失败 ===")
g = o.groupby(["train_class", "cv_group"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                                 top=("pred_class", lambda s: short[s.mode().iat[0]]))
g = g[g["n"] >= 3].sort_values("acc").head(25)
for (c, grp), r in g.iterrows():
    L.append(f"{short[c]:22s} {grp[:60]:60s} n={r['n']:3d}  正确 {r['acc']:.2f}  多数被判为 {r['top']}")
L.append("\n=== 按来源 ===")
L.append(OP.table(o.groupby("source")["correct"].agg(["size", "mean"]).round(3)))
txt = "\n".join(L)
open(os.path.join(run, "error_summary.txt"), "w", encoding="utf-8").write(txt)
print(txt)

cm = pd.crosstab(o["train_class"].map(short), o["pred_class"].map(short))
cm.to_csv(os.path.join(run, "confusion.csv"), encoding="utf-8-sig")

# 错分缩略图
parts = ["<!doctype html><meta charset='utf-8'><title>错分图片</title><style>"
         "body{font-family:-apple-system,sans-serif;margin:16px}h3{margin:18px 0 6px}"
         ".g{display:flex;flex-wrap:wrap;gap:8px}figure{margin:0;width:170px;font-size:11px}"
         "img{width:170px;height:170px;object-fit:contain;background:#eee}</style>",
         f"<h2>错分图片（{(~o['correct']).sum()} 张）— {html.escape(run)}</h2>"]
for c, d in o[~o["correct"]].groupby("train_class"):
    parts.append(f"<h3>真实：{html.escape(c)}（{len(d)} 张错分）</h3><div class='g'>")
    for _, r in d.sort_values("p_top1", ascending=False).iterrows():
        src = "file://" + os.path.abspath(r["path"])
        parts.append(f"<figure><img src='{html.escape(src)}' loading='lazy'><figcaption>"
                     f"判为 <b>{html.escape(short[r['pred_class']])}</b> {r['p_top1']:.2f}<br>"
                     f"正确类概率 {r['p_true']:.2f}（第{r['true_rank']}名）<br>"
                     f"{html.escape(os.path.basename(r['path']))}</figcaption></figure>")
    parts.append("</div>")
open(os.path.join(run, "errors.html"), "w", encoding="utf-8").write("\n".join(parts))
print(f"\n已写出 {run}/error_summary.txt、confusion.csv、errors.html")
