#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_audit_train.py —— 训练集审计：重复图 / 跨类别重复 / 菌株分组推断 / 菌落图标签核对

用法（默认路径已按你的 Mac 设置）：
    python 00_audit_train.py
    python 00_audit_train.py --micro_root /Users/chunjiang/Documents/真菌AI/CNN/dataset \
                          --colony_root /Users/chunjiang/Documents/真菌AI/CNN/normal \
                          --out ./train_audit

依赖（清华源）：
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow numpy pandas

输出：
    train_audit/train_manifest.csv   每张图一行：类别、来源类型、md5、拍摄日期、推断的 group_id 及可信度
    train_audit/summary.txt          每类图片数 / 推断菌株数 / 重复情况 / 跨类别重复 / 菌落标签疑点

group_id 推断规则（宁大勿小：分组偏大只会让评估更保守，偏小才会泄漏）：
    1. 文件名带菌株号：烟曲霉-02_006 → 烟曲霉-02；微小根霉_01_005 → 微小根霉_01；W-B-14_烟曲霉_001 → W-B-14
    2. tile_15_3 → tile_15（同一原图切出的小图）
    3. ✅ 文件夹截图（文件名含 "XXpics"）→ 文件夹 × 菌种 为一组（无法区分同一文件夹内的不同菌株时的保守做法）
    4. 其余（species_N、Untitled-N、000.png）→ 同类别 + 同一拍摄日期 为一组（EXIF 优先，其次文件修改日期）
    5. 内容完全相同（md5）或近似重复（dHash+互相关）的图强制并入同一组
"""
import os
import re
import hashlib
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DUP_HAMMING = 6
NCC_THR = 0.95           # 第一步：48x48 灰度相关系数（只做候选，单个细胞/简单构图极易误报）
THUMB = 48
FINE = 96                # 第二步：96x96 彩色逐像素比较
FINE_MAD = 4.0
FINE_NCC = 0.995           # 平均绝对像素差 < 4（0-255 灰阶）才判为"同一张图"（只差压缩/缩放）

LABELS = {
    "00": "土曲霉", "01": "链格孢", "02": "阿萨希毛孢子菌", "04": "布鲁菌革兰染色", "05": "黑曲霉",
    "06": "黄曲霉", "07": "尖端赛多孢", "08": "烟曲霉", "09": "微小根霉", "10": "白念珠菌",
    "11": "光滑念珠菌", "13": "克柔念珠菌", "15": "热带念珠菌", "19": "耳念珠菌", "21": "罗伦特隐球菌",
    "22": "酿酒酵母", "27": "根霉", "28": "构巢", "29": "马尔尼菲", "30": "青霉", "31": "少根根霉",
    "32": "镰刀", "33": "多育赛多孢", "34": "申克孢子丝", "35": "小克银汉", "36": "帚霉",
    "37": "弯孢霉", "38": "拟青霉", "39": "无绿藻", "40": "杂色曲霉", "41": "毛霉", "42": "灰绿曲霉",
    "43": "着色霉", "44": "离蠕孢", "45": "组织胞浆菌", "46": "裂褶菌", "47": "波氏赛多孢",
    "48": "鬼伞", "49": "犁头霉", "50": "隐球菌",
}
# 文件名中的菌名与类别不一致时提示（例：根霉(27) 里出现 小孢根霉 → 应属 微小根霉(09)）
NAME_TO_CLASS = {"小孢根霉": "09", "微小根霉": "09", "少根根霉": "31", "烟曲霉": "08", "黄曲霉": "06",
                 "黑曲霉": "05", "土曲霉": "00", "构巢": "28", "杂色曲霉": "40", "灰绿曲霉": "42",
                 "多育赛多孢": "33", "波氏赛多孢": "47", "尖端赛多孢": "07", "拟青": "38", "镰刀": "32",
                 "耳念珠菌": "19", "小克银": "35", "申克孢子丝": "34", "马尔尼菲": "29"}


def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def fine(img):
    return np.asarray(img.convert("RGB").resize((FINE, FINE), Image.BILINEAR), dtype=np.uint8)


def same_image(a, b):
    """像素级比较 96x96 彩色图：平均绝对差 < 4 且 相关系数 >= 0.995 才算同一张图。
    真正的同一张图（重存 / 缩放 / JPEG 压缩）相关系数 > 0.999；
    构图相似的不同图（如居中的单个酵母细胞）背景占大部分，平均差可能很小，但相关系数通常 < 0.99"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if float(np.abs(a - b).mean()) >= FINE_MAD:
        return False
    a -= a.mean()
    b -= b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return den > 0 and float((a * b).sum()) / den >= FINE_NCC


def thumb(img):
    a = np.asarray(img.convert("L").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def hamming(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def shot_date(img, path):
    try:
        ex = img.getexif()
        v = ex.get(36867) or ex.get_ifd(0x8769).get(36867) or ex.get(306)
        if v:
            return str(v)[:10].replace(":", "-"), "exif"
    except Exception:
        pass
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d"), "mtime"


def strain_rule(cls, fn):
    """返回 (group_id, 规则名)，无法从文件名推断时返回 (None, None)"""
    stem = os.path.splitext(fn)[0]
    stem = re.sub(r"\(\d+\)$", "", stem).strip()          # 000(1) → 000
    m = re.match(r"^([A-Za-z]+-[ABC]-\d+)_", stem)       # W-B-14_烟曲霉_001
    if m:
        return m.group(1), "contrib_code"
    m = re.match(r"^(.+?pics)(.*?)_\d+$", stem)          # 真菌图（…）_56pics烟曲霉_033
    if m:
        return f"{m.group(1)}|{m.group(2) or cls}", "contrib_folder"
    m = re.match(r"^tile_(\d+)_\d+$", stem)              # tile_15_3
    if m:
        return f"{cls}|tile_{m.group(1)}", "tile_source"
    m = re.match(r"^(\D+?)[-_](\d+)_\d+$", stem)         # 烟曲霉-02_006 / 微小根霉_01_005
    if m:
        return f"{m.group(1)}-{m.group(2)}", "strain_in_name"
    return None, None


def scan(root, modality):
    rows = []
    if not root or not os.path.isdir(root):
        return rows
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for dp, _, fns in os.walk(cdir):
            for fn in sorted(fns):
                if not fn.lower().endswith(IMG_EXTS) or fn.startswith("."):
                    continue
                p = os.path.join(dp, fn)
                r = dict(modality=modality, cls=cls, species=LABELS.get(cls, ""),
                         rel_path=os.path.relpath(p, root), file=fn)
                try:
                    with open(p, "rb") as f:
                        r["md5"] = hashlib.md5(f.read()).hexdigest()
                    with Image.open(p) as im:
                        r["date"], r["date_src"] = shot_date(im, p)
                        im = im.convert("RGB")
                        r["width"], r["height"] = im.size
                        r["_dh"], r["_th"], r["_fi"] = dhash(im), thumb(im), fine(im)
                    r["error"] = ""
                except Exception as e:
                    r.update(md5="", date="", date_src="", width=None, height=None,
                             _dh=None, _th=None, _fi=None, error=str(e))
                rows.append(r)
    return rows


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_root", default="/Users/chunjiang/Documents/真菌AI/CNN/dataset")
    ap.add_argument("--colony_root", default="/Users/chunjiang/Documents/真菌AI/CNN/normal")
    ap.add_argument("--out", default=OP.TRAIN_AUDIT)
    a = ap.parse_args()
    a.out = OP.place(a.out, OP.AUDIT)
    os.makedirs(a.out, exist_ok=True)

    rows = scan(a.micro_root, "micro") + scan(a.colony_root, "colony")
    df = pd.DataFrame(rows)
    print(f"扫描完成：{len(df)} 张")

    # ---------- 初步 group ----------
    gid, rule = [], []
    for _, r in df.iterrows():
        g, ru = strain_rule(r.cls, r.file)
        if g is None:
            g, ru = f"{r.modality}|{r.cls}|date:{r.date}", f"date_{r.date_src}"
        gid.append(g)
        rule.append(ru)
    df["group_id"], df["group_rule"] = gid, rule

    # ---------- 重复检测（同一模态内）----------
    uf = UF(len(df))
    df["exact_dup_of"] = ""
    df["near_dup_of"] = ""
    df["cross_class_dup"] = ""
    for mod in ("micro", "colony"):
        idx = df.index[(df.modality == mod) & (df.error == "")].tolist()
        by_md5 = defaultdict(list)
        for i in idx:
            by_md5[(df.at[i, "cls"], df.at[i, "md5"])].append(i)
        for lst in by_md5.values():
            # 保留"原名"那张（不带 (1) 之类后缀、文件名最短），其余记为副本
            lst = sorted(lst, key=lambda i: ("(" in df.at[i, "file"], len(df.at[i, "file"]), df.at[i, "file"]))
            for j in lst[1:]:
                uf.union(lst[0], j)
                df.at[j, "exact_dup_of"] = df.at[lst[0], "rel_path"]
        H = np.array([df.at[i, "_dh"] for i in idx], dtype=np.uint64)
        T = np.stack([df.at[i, "_th"] for i in idx]) if idx else np.zeros((0, THUMB * THUMB))
        for s in range(0, len(idx), 400):
            D = hamming(H[s:s + 400], H)
            for k, row in enumerate(D):
                ii = s + k
                for jj in np.where(row <= DUP_HAMMING)[0]:
                    if jj <= ii or float(T[ii] @ T[jj]) < NCC_THR:
                        continue
                    i, j = idx[ii], idx[jj]
                    if not same_image(df.at[i, "_fi"], df.at[j, "_fi"]):
                        continue
                    if df.at[i, "cls"] == df.at[j, "cls"]:
                        if df.at[i, "md5"] == df.at[j, "md5"]:
                            continue
                        uf.union(i, j)
                        df.at[j, "near_dup_of"] = df.at[i, "rel_path"]
                    else:
                        df.at[i, "cross_class_dup"] += df.at[j, "rel_path"] + ";"
                        df.at[j, "cross_class_dup"] += df.at[i, "rel_path"] + ";"
    # 重复图所在的 group 整体合并（重复图跨越两个组时，两组合为一组）
    comp = defaultdict(list)
    for i in df.index:
        comp[uf.find(i)].append(i)
    g_uf = {}

    def gf(g):
        while g_uf.get(g, g) != g:
            g = g_uf[g]
        return g
    for members in comp.values():
        gs = sorted(set(gid[i] for i in members))
        for g in gs[1:]:
            g_uf[gf(g)] = gf(gs[0])
    df["group_id"] = df["group_id"].map(gf)

    conf = {"contrib_code": "high", "strain_in_name": "high", "tile_source": "medium",
            "contrib_folder": "conservative", "date_exif": "medium", "date_mtime": "low"}
    df["group_confidence"] = df["group_rule"].map(conf).fillna("low")

    # ---------- 菌落标签核对 ----------
    df["label_conflict"] = ""
    for i, r in df.iterrows():
        for name, c in NAME_TO_CLASS.items():
            if name in r.file and c != r.cls and not (name == "镰刀" and r.cls == "32"):
                # 更长的名字优先（"小孢根霉" 优先于 "根霉"）
                df.at[i, "label_conflict"] = f"文件名含'{name}'→应属{c}"
                break

    out = df.drop(columns=["_dh", "_th", "_fi"])
    out.to_csv(os.path.join(a.out, "train_manifest.csv"), index=False, encoding="utf-8-sig")

    # ---------- 汇总 ----------
    lines = []
    W = lines.append
    W(f"总图片: {len(df)}  (显微 {int((df.modality=='micro').sum())} / 菌落 {int((df.modality=='colony').sum())})  读取失败: {int((df.error!='').sum())}\n")
    W("拍摄日期来源: " + df.date_src.value_counts().to_dict().__repr__())
    W("  (mtime 只是文件修改日期，若图片被批量复制过，同一天会出现大量不同菌株，此时日期分组不可信)\n")
    m = df[df.modality == "micro"]
    t = m.groupby("cls").agg(species=("species", "first"), images=("file", "size"),
                             groups=("group_id", "nunique"),
                             exact_dups=("exact_dup_of", lambda s: int((s != "").sum())),
                             near_dups=("near_dup_of", lambda s: int((s != "").sum())),
                             rule=("group_rule", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().items())))
    W("=== 显微图：每类图片数 / 推断菌株组数 / 重复 ===")
    W(OP.table(t))
    W("\n推断菌株组数 ≥5 的类别（可做 isolate-level 5 折）: " +
      ", ".join(f"{c}{t.at[c,'species']}" for c in t.index if t.at[c, "groups"] >= 5))
    W("推断菌株组数 <5 的类别（无法可靠做 5 折，需补菌株或剔除）: " +
      ", ".join(f"{c}{t.at[c,'species']}({t.at[c,'groups']})" for c in t.index if t.at[c, "groups"] < 5))
    W(f"\n完全相同的重复图（md5）: {int((df.exact_dup_of!='').sum())}   近似重复: {int((df.near_dup_of!='').sum())}")
    W("  旧代码按文件名分组，000.png 与 000(1).png 会被分进不同组 → 同一张图同时出现在训练折和验证折。\n")
    cc = df[df.cross_class_dup != ""]
    W(f"=== 跨类别：像素级几乎相同的图（MAD<4 且相关≥0.995）: {len(cc)} ===")
    for _, r in cc.iterrows():
        W(f"  [{r.cls}] {r.rel_path}  <->  {r.cross_class_dup}")
    lc = df[df.label_conflict != ""]
    W(f"\n=== 文件名菌名与类别不符: {len(lc)} ===")
    for _, r in lc.iterrows():
        W(f"  [{r.modality} {r.cls}] {r.rel_path}  {r.label_conflict}")
    col = df[df.modality == "colony"].groupby("cls").agg(images=("file", "size"), groups=("group_id", "nunique"))
    W("\n=== 菌落图：每类图片数 / 推断菌株组数 ===")
    W(OP.table(col))
    txt = "\n".join(lines)
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
