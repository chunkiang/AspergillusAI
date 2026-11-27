#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_plot_confusion.py —— 混淆矩阵热图（行归一化 = 召回率；格子里写张数），可直接用作论文图

用法：
    # 交叉验证（与原来完全相同）
    python 02b_plot_confusion.py runs/densenet121_cv_20261009_1529/confusion.csv
    # 临床验证（03_eval_clinical.py 输出的 confusion.csv / confusion_roi.csv；类名自动缩写）
    python 02b_plot_confusion.py <RUN>/micro/confusion.csv --title "Clinical validation, microscopy, isolate level"
输出：同目录下 confusion_matrix*.png（300 dpi）和 .pdf
      confusion.csv → confusion_matrix.png；confusion_roi.csv → confusion_matrix_roi.png
依赖：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib

说明：
  · 全名（如 "Aspergillus fumigatus (sect. Fumigati)"）自动缩写为 "A. fumigatus"，与 CV 图一致；
  · 类群分隔线按实际出现的类别画（只出现 3 个曲霉时不会画错位的线）；
  · 类别少时自动缩小画布、放大字号。
"""
import argparse
import os
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

# 按类群排序，便于看出"错误集中在同一类群内"
ORDER = [
    "A. fumigatus", "A. flavus", "A. niger", "A. terreus", "A. nidulans", "A. versicolor", "A. glaucus",
    "Rhizopus", "Mucor", "Lichtheimia", "Cunninghamella",
    "Fusarium", "Scedosporium", "Lomentospora prolificans", "Penicillium", "Paecilomyces", "Scopulariopsis",
    "Alternaria", "Curvularia", "Bipolaris", "Fonsecaea",
    "Talaromyces marneffei", "Sporothrix schenckii complex", "Histoplasma capsulatum",
    "Schizophyllum", "Coprinellus/Coprinopsis",
]
GROUPS = [("Aspergillus", 7), ("Mucorales", 4), ("Hyaline", 6), ("Dematiaceous", 4), ("Dimorphic", 3), ("Basidio.", 2)]
GROUP_OF, _i = {}, 0
for _g, _n in GROUPS:
    for _l in ORDER[_i:_i + _n]:
        GROUP_OF[_l] = _g
    _i += _n
# 单色顺序色阶（浅 → 深蓝）
CMAP = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DEFAULT_TITLE = "Confusion matrix — out-of-fold predictions, 5-fold CV"


def short(c):
    """与 02b_analyze_errors.py 相同的缩写规则；已是缩写时不变。"""
    return str(c).replace("Aspergillus ", "A. ").split(" (")[0].strip()


ap = argparse.ArgumentParser(description="混淆矩阵热图")
ap.add_argument("path", nargs="?", default=None, help="confusion.csv（行 = 真实，列 = 预测）")
ap.add_argument("--title", default=DEFAULT_TITLE, help="图标题（后面会自动加 n 和准确率）")
ap.add_argument("--out", default=None, help="输出文件名（不含扩展名），默认与输入同目录")
a = ap.parse_args()

path = a.path or os.path.join(OP.latest_cv_run(), "confusion.csv")
if not os.path.exists(path):
    d = OP.find_run(os.path.dirname(path) or ".")
    path = os.path.join(d, os.path.basename(path))
cm = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
cm.index = [short(c) for c in cm.index]
cm.columns = [short(c) for c in cm.columns]
cm = cm.groupby(level=0).sum().T.groupby(level=0).sum().T      # 缩写后若有重名则合并
labels = [c for c in ORDER if c in cm.index or c in cm.columns] + \
         [c for c in sorted(set(cm.index) | set(cm.columns)) if c not in ORDER]
cm = cm.reindex(index=labels, columns=labels).fillna(0).astype(int)
K = len(labels)
big = K >= 20
# 类别少（临床验证）时只保留实际出现的参考类别作为行，避免一排排 n=0 的空行；
# 列仍保留全部出现过的类别（参考或预测），所以对角线 = 行名与列名相同的格子
rows = labels if big else [l for l in labels if cm.loc[l].sum() > 0] or labels
cm = cm.loc[rows]
R = len(rows)
n_row = cm.sum(axis=1).replace(0, 1)
norm = cm.div(n_row, axis=0).values


def it(l):
    return l if l.startswith("$") else "$\\it{" + l.replace(" ", "\\ ") + "}$"


ylab = [f"{it(l)}  (n={int(n)})" for l, n in zip(rows, cm.sum(axis=1))]

if big:
    figsize, f_cell, f_tick, f_title = (11.5, 10), 7, 8, 11
else:
    figsize, f_cell, f_tick, f_title = (3.8 + 0.6 * K, 2.4 + 0.55 * R), 10, 9, 9
fig, ax = plt.subplots(figsize=figsize)
im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
for i in range(R):
    for j in range(K):
        v = cm.iat[i, j]
        if v:
            ax.text(j, i, str(v), ha="center", va="center", fontsize=f_cell,
                    color="white" if norm[i, j] > 0.55 else "#1f1f1f",
                    fontweight="bold" if rows[i] != labels[j] else "normal")
# 类群分隔线：按实际出现的类别，在类群变化处画线（行、列分别判断）
for seq, line in [(labels, ax.axvline), (rows, ax.axhline)]:
    grp = [GROUP_OF.get(l, "Other") for l in seq]
    for k in range(1, len(seq)):
        if grp[k] != grp[k - 1]:
            line(k - 0.5, color="#9a9a9a", lw=0.8)
ax.set_xticks(range(K))
ax.set_yticks(range(R))
ax.set_xticklabels([it(l) for l in labels], rotation=60 if big or K > 4 else 30, ha="right", fontsize=f_tick)
ax.set_yticklabels(ylab, fontsize=f_tick)
ax.set_xlabel("Predicted class", fontsize=10)
ax.set_ylabel("True class", fontsize=10)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035 if big else 0.05, pad=0.02)
cb.set_label("Proportion of true class (recall)", fontsize=9)
cb.outline.set_visible(False)
tot = int(cm.values.sum())
acc = sum(int(cm.at[r, r]) for r in rows) / tot if tot else float("nan")
title = f"{a.title} (n={tot}, accuracy {acc:.3f})"
ax.set_title(title if big else textwrap.fill(title, width=int(figsize[0] * 9)), fontsize=f_title, loc="left")
fig.tight_layout()
if a.out:
    out = os.path.splitext(a.out)[0]
else:
    d, b = os.path.split(os.path.splitext(path)[0])
    out = os.path.join(d, b.replace("confusion", "confusion_matrix", 1))
fig.savefig(out + ".png", dpi=300)
fig.savefig(out + ".pdf")
print(f"已写出 {out}.png / .pdf")
