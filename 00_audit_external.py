#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_audit_external.py —— 外部验证图片审计（清点 / 查重 / 与训练集重叠 / 质量筛查）

用法：
    python 00_audit_external.py            # 默认路径即你的 Mac 路径
    python 00_audit_external.py --ext_root /Users/chunjiang/Desktop/丝状真菌 \
                             --train_root /Users/chunjiang/Documents/真菌AI/CNN/dataset \
                             --colony_root /Users/chunjiang/Documents/真菌AI/CNN/normal --out ./ext_audit

依赖（清华源）：
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow numpy pandas pillow-heif

输出：
    ext_audit/inventory.csv        每张图一行；末尾空列需人工填写（参考鉴定、菌株号等）
    ext_audit/summary.txt          按来源/类别统计、重复图、与训练集重叠、低质量图
    ext_audit/converted/           HEIC 转出的 JPG
"""
import os
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic")
CODE_RE = re.compile(r"^([A-Za-z]+)-([ABC])-(\d+)", re.I)
DUP_HAMMING = 6          # dHash 汉明距离阈值：<=6 视为疑似重复
MIN_SIDE = 400           # 短边像素 < 400 记为低分辨率
BLUR_PCTL = 10           # 拉普拉斯方差处于全体最低 10% 记为疑似模糊（显微图背景柔和，绝对阈值误报多）
NCC_THR = 0.90           # dHash 候选对再用 48x48 灰度归一化互相关确认，>=0.90 才判为重复
THUMB = 48
FINE = 96                # 第二步：96x96 彩色逐像素比较，平均绝对差 < 4 才判为同一张图
FINE_MAD = 4.0
FINE_NCC = 0.995


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def laplacian_var(img: Image.Image) -> float:
    g = img.convert("L")
    g.thumbnail((1024, 1024))
    a = np.asarray(g, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4 * a[1:-1, 1:-1])
    return float(lap.var())


def thumb(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((THUMB, THUMB), Image.BILINEAR), dtype=np.float32).ravel()
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


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


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def iter_images(root):
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().strip().endswith(IMG_EXTS) and not fn.startswith("."):
                yield os.path.join(dp, fn)


def guess_fields(path, ext_root):
    rel = os.path.relpath(path, ext_root)
    folder = rel.split(os.sep)[0]
    stem = os.path.splitext(os.path.basename(path))[0].strip()
    m = CODE_RE.match(stem)
    code = cat = None
    if m:
        code = f"{m.group(1).upper()}-{m.group(2).upper()}-{int(m.group(3))}"
        cat = m.group(2).upper()
    # 编号规则：姓名首字母-A/B/C-序号
    #   A = 原始标本镜下（直接镜检，不属于本模型任务，默认排除）
    #   B = 纯菌落（宏观菌落图）
    #   C = 纯菌落镜下（纯培养物显微图，模型主输入）
    # 同一株的 B 与 C 序号不对应，配对只能依据答案 Word 文档
    if cat == "A":
        modality = "direct_specimen"
    elif cat == "B" or "菌落" in stem:
        modality = "colony"
    elif cat == "C":
        modality = "micro"
    else:
        modality = "unknown"
    label_hint = ""
    if not m and re.search(r"[\u4e00-\u9fff]", stem):
        label_hint = re.sub(r"(菌落形态|菌落|高倍|低倍|\d+)", "", stem)
    source_type = "web" if "网络" in folder else "clinical"
    return dict(folder=folder, rel_path=rel, code=code, category=cat,
                modality_guess=modality, label_hint=label_hint,
                source_type=source_type)


def load_img(path, conv_dir):
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    if path.lower().endswith(".heic"):
        os.makedirs(conv_dir, exist_ok=True)
        out = os.path.join(conv_dir, os.path.splitext(os.path.basename(path))[0] + ".jpg")
        img.save(out, quality=95)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_root", default="/Users/chunjiang/Desktop/丝状真菌")
    ap.add_argument("--train_root", default="/Users/chunjiang/Documents/真菌AI/CNN/dataset")
    ap.add_argument("--colony_root", default="/Users/chunjiang/Documents/真菌AI/CNN/normal")
    ap.add_argument("--out", default=OP.EXT_AUDIT)
    args = ap.parse_args()
    args.out = OP.place(args.out, OP.AUDIT)
    os.makedirs(args.out, exist_ok=True)
    conv_dir = os.path.join(args.out, "converted")

    # ---------- 外部图片 ----------
    rows, hashes, thumbs, fines = [], [], [], []
    for p in iter_images(args.ext_root):
        r = guess_fields(p, args.ext_root)
        try:
            if p.lower().endswith(".heic") and not HEIF_OK:
                raise RuntimeError("需安装 pillow-heif")
            img = load_img(p, conv_dir)
            w, h = img.size
            r.update(width=w, height=h, blur_var=round(laplacian_var(img), 1),
                     mean_brightness=round(float(np.asarray(img.convert("L")).mean()), 1),
                     read_error="")
            hashes.append(dhash(img))
            thumbs.append(thumb(img))
            fines.append(fine(img))
        except Exception as e:
            r.update(width=None, height=None, blur_var=None,
                     mean_brightness=None, read_error=str(e))
            hashes.append(None)
            thumbs.append(None)
            fines.append(None)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["low_res"] = df["width"].notna() & (df[["width", "height"]].min(axis=1) < MIN_SIDE)
    blur_cut = float(np.nanpercentile(df["blur_var"].astype(float), BLUR_PCTL)) if df["blur_var"].notna().any() else 0.0
    df["maybe_blurry"] = df["blur_var"].notna() & (df["blur_var"] < blur_cut)

    ok = [i for i, h in enumerate(hashes) if h is not None]
    ext_h = np.array([hashes[i] for i in ok], dtype=np.uint64)
    ext_t = np.stack([thumbs[i] for i in ok]) if ok else np.zeros((0, THUMB * THUMB), np.float32)

    # ---------- 外部集内部重复 ----------
    dup_of = defaultdict(list)
    if len(ext_h) > 1:
        D = hamming_matrix(ext_h, ext_h)
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                if (D[a, b] <= DUP_HAMMING and float(ext_t[a] @ ext_t[b]) >= NCC_THR
                        and same_image(fines[ok[a]], fines[ok[b]])):
                    dup_of[ok[a]].append(df.at[ok[b], "rel_path"])
                    dup_of[ok[b]].append(df.at[ok[a], "rel_path"])
    df["dup_within_ext"] = [";".join(dup_of.get(i, [])) for i in range(len(df))]

    # ---------- 与训练集重叠 ----------
    df["near_dup_in_train"] = ""
    df["code_seen_in_train"] = ""
    df["folder_in_train"] = 0
    if args.train_root and os.path.isdir(args.train_root):
        tr_paths, tr_h, tr_t, tr_f, tr_codes = [], [], [], [], defaultdict(list)
        tr_files = list(iter_images(args.train_root))
        if args.colony_root and os.path.isdir(args.colony_root):
            tr_files += list(iter_images(args.colony_root))
        for p in tr_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            m = CODE_RE.match(stem)
            if m:
                tr_codes[f"{m.group(1).upper()}-{m.group(2).upper()}-{int(m.group(3))}"].append(
                    os.path.relpath(p, os.path.dirname(args.train_root)))
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    tr_h.append(dhash(im))
                    tr_t.append(thumb(im))
                    tr_f.append(fine(im))
                tr_paths.append(os.path.relpath(p, os.path.dirname(args.train_root)))
            except Exception:
                pass
        tr_h = np.array(tr_h, dtype=np.uint64)
        tr_t = np.stack(tr_t) if tr_t else np.zeros((0, THUMB * THUMB), np.float32)
        if len(tr_h) and len(ext_h):
            for start in range(0, len(ext_h), 200):
                D = hamming_matrix(ext_h[start:start + 200], tr_h)
                for k, row in enumerate(D):
                    cand = np.where(row <= DUP_HAMMING)[0]
                    hits = [j for j in cand if float(ext_t[start + k] @ tr_t[j]) >= NCC_THR
                            and same_image(fines[ok[start + k]], tr_f[j])]
                    if len(hits):
                        df.at[ok[start + k], "near_dup_in_train"] = ";".join(
                            tr_paths[j] for j in hits[:5])
        for i, c in df["code"].items():
            if c and c in tr_codes:
                df.at[i, "code_seen_in_train"] = ";".join(tr_codes[c][:5])
        # 整个文件夹是否已并入训练集：训练集文件名里出现该文件夹名
        tr_names = [os.path.basename(p) for p in tr_paths]
        merged = {}
        for fd in df["folder"].unique():
            key = re.sub(r"[✅❌\s]", "", fd)
            n = sum(1 for t in tr_names if key and key in t)
            if n:
                merged[fd] = n
        df["folder_in_train"] = df["folder"].map(lambda x: merged.get(x, 0))

    # ---------- 人工填写列 ----------
    for col in ["species_ref", "isolate_id", "centre", "ref_method",
                "modality_confirmed", "include", "exclude_reason"]:
        df[col] = ""
    is_a = df["modality_guess"] == "direct_specimen"
    df.loc[is_a, "include"] = "N"
    df.loc[is_a, "exclude_reason"] = "A类原始标本直接镜检，非纯培养物"
    df.to_csv(os.path.join(args.out, "inventory.csv"), index=False, encoding="utf-8-sig")

    # ---------- 汇总 ----------
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"图片总数: {len(df)}  读取失败: {(df.read_error != '').sum()}\n\n")
        f.write("按来源类型:\n" + OP.table(df.source_type.value_counts()) + "\n\n")
        f.write("按文件夹 × 类别(A/B/C):\n")
        f.write(OP.table(pd.crosstab(df.folder, df.category.fillna("无编码"))) + "\n\n")
        f.write("按模态猜测:\n" + OP.table(df.modality_guess.value_counts()) + "\n\n")
        f.write(f"低分辨率(短边<{MIN_SIDE}): {int(df.low_res.sum())}\n")
        f.write(f"疑似模糊(拉普拉斯方差最低{BLUR_PCTL}%, <{blur_cut:.1f}): {int(df.maybe_blurry.sum())}\n")
        f.write(f"外部集内部疑似重复: {int((df.dup_within_ext != '').sum())}\n")
        f.write(f"与训练集近似重复(dHash): {int((df.near_dup_in_train != '').sum())}\n")
        f.write(f"编号与训练集文件名相同: {int((df.code_seen_in_train != '').sum())}\n\n")
        fm = df[df.folder_in_train > 0].groupby("folder").folder_in_train.first()
        f.write("已截图放入训练目录的文件夹（✅；若用于训练则不能再作外部验证）:\n")
        f.write((OP.table(fm) if len(fm) else "  无") + "\n\n")
        for col, title in [("near_dup_in_train", "与训练集近似重复"),
                           ("code_seen_in_train", "编号与训练集相同"),
                           ("dup_within_ext", "外部集内部重复")]:
            sub = df[df[col] != ""]
            if len(sub):
                f.write(f"=== {title} ===\n")
                for _, r in sub.iterrows():
                    f.write(f"{r.rel_path}  ->  {r[col]}\n")
                f.write("\n")
    print(open(os.path.join(args.out, "summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
