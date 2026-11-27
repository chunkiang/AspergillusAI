#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_clean_train.py —— 根据 train_audit/train_manifest.csv 清理训练集（只移动，不删除）

用法：
    python 00_clean_train.py            # 预演：只打印将要做什么，并生成人工复核网页
    python 00_clean_train.py --apply    # 真正执行：把完全重复的副本移到隔离目录

做三件事：
  1. 完全相同的重复图（md5 相同，如 000.png 与 000(1).png）：保留一张，副本移到 CNN/_quarantine/exact_dup/
  2. 跨类别重复（同一张图出现在两个类别）：不自动处理，生成 review_cross_class.html 并排显示，供人工判断哪个标签正确
  3. 文件名菌名与类别不符（如 根霉(27) 菌落目录里的 小孢根霉）：同样列入复核网页
  恢复：把隔离目录里的文件按原相对路径移回即可（move_log.csv 记录了每一步）
"""
import os
import csv
import html
import shutil
import argparse
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=OP.TRAIN_MANIFEST)
ap.add_argument("--micro_root", default="/Users/chunjiang/Documents/真菌AI/CNN/dataset")
ap.add_argument("--colony_root", default="/Users/chunjiang/Documents/真菌AI/CNN/normal")
ap.add_argument("--quarantine", default="/Users/chunjiang/Documents/真菌AI/CNN/_quarantine")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
a.manifest = OP.find(a.manifest, OP.TRAIN_AUDIT, OP.AUDIT)

df = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
root = {"micro": a.micro_root, "colony": a.colony_root}


def abspath(r):
    return os.path.join(root[r["modality"]], r["rel_path"])


# ---------- 1. 完全重复 ----------
dups = df[df.exact_dup_of != ""]
print(f"完全重复的副本: {len(dups)} 张" + ("（将移入隔离目录）" if a.apply else "（预演，未移动）"))
log = []
for _, r in dups.iterrows():
    src = abspath(r)
    dst = os.path.join(a.quarantine, "exact_dup", r["modality"], r["rel_path"])
    log.append([src, dst, r["exact_dup_of"]])
    if a.apply and os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
if a.apply:
    os.makedirs(a.quarantine, exist_ok=True)
    with open(os.path.join(a.quarantine, "move_log.csv"), "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["from", "to", "kept_copy"]] + log)
for c, n in dups.groupby("cls").size().items():
    print(f"  类别 {c}: {n}")

# ---------- 2/3. 复核网页 ----------
by_rel = {r.rel_path: r for _, r in df.iterrows()}
cards = []
seen = set()
for _, r in df[df.cross_class_dup != ""].iterrows():
    others = [o for o in r.cross_class_dup.split(";") if o]
    key = tuple(sorted([r.rel_path] + others))
    if key in seen:
        continue
    seen.add(key)
    imgs = [r] + [by_rel[o] for o in others if o in by_rel]
    cards.append(("像素级几乎相同的图出现在不同类别（请目视核对）", imgs))
for _, r in df[df.label_conflict != ""].iterrows():
    cards.append((r.label_conflict, [r]))


def fig(r):
    p = "file://" + abspath(r)
    return (f'<figure><img src="{html.escape(p)}" loading="lazy"><figcaption>'
            f'<b>[{html.escape(r["cls"])} {html.escape(r["species"])}]</b><br>'
            f'{html.escape(r["rel_path"])}</figcaption></figure>')


page = ["<!doctype html><meta charset='utf-8'><title>训练集复核</title>",
        "<style>body{font-family:-apple-system,sans-serif;margin:16px;background:#fafafa}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}"
        ".row{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;width:260px}"
        "img{width:260px;height:260px;object-fit:contain;background:#eee;border-radius:4px}"
        "figcaption{font-size:12px;word-break:break-all}h3{margin:0 0 8px;font-size:14px}</style>",
        f"<h2>训练集人工复核（{len(cards)} 组）</h2>",
        "<p>请判断每组图片的正确类别；标签错的移到正确类别，无法判断的移到 _quarantine/uncertain/。</p>"]
for i, (title, imgs) in enumerate(cards, 1):
    page.append(f"<div class='card'><h3>{i}. {html.escape(title)}</h3><div class='row'>"
                + "".join(fig(x) for x in imgs) + "</div></div>")
os.makedirs(OP.AUDIT, exist_ok=True)
out = os.path.join(OP.AUDIT, "review_cross_class.html")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(page))
print(f"\n需要人工复核: {len(cards)} 组 → 用浏览器打开 {out}")
if not a.apply:
    print("\n确认无误后运行: python 00_clean_train.py --apply")
