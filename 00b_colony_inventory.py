#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00b_colony_inventory.py — 菌落图数据盘点（只读，不修改任何原图）

用途
  盘点训练用菌落图：每个曲霉组有多少株、正面/反面各多少张、
  有多少株能和镜检训练图对上株号、拍摄设备、能否检测到平皿、背景是否为白纸。
  输出补拍清单（每组距离目标株数的缺口）。

安装（清华源）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python pillow pillow-heif pandas openpyxl

运行
  python 00b_colony_inventory.py \
      --colony_root "/Users/chunjiang/Documents/真菌AI/CNN/normal" \
      --micro_root  "/Users/chunjiang/Documents/真菌AI/CNN/dataset" \
      --out_dir     ./00b_inventory_out \
      --qc_overlay

株号推断规则（运行后请先看 strain_mapping_examples.txt 核对）
  · 若路径为 菌种/株号/图片 → 第二级文件夹即株号
  · 若路径为 菌种/图片     → 从文件名去掉结尾的 正/反/front/back/序号 等后缀得到株号
    规则不对时用 --strain_regex 自定义（第一个捕获组 = 株号）
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python，请先按文件头部说明安装")

from PIL import Image, ExifTags

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".heif"}
TARGET_STRAINS = 10  # 每组目标株数

# ---------- 菌种/文件夹名 → 曲霉组 ----------
# 按顺序匹配（小写后做子串匹配），先匹配到的生效
SECTION_RULES = [
    ("Fumigati",   ["fumigatus", "fumigati", "lentulus", "烟曲"]),
    ("Flavi",      ["flavus", "flavi", "oryzae", "parasiticus", "黄曲"]),
    ("Nigri",      ["niger", "nigri", "tubingensis", "awamori", "welwitschiae", "黑曲"]),
    ("Terrei",     ["terreus", "terrei", "土曲"]),
    ("Nidulantes", ["nidulans", "nidulantes", "versicolor", "versicolores",
                    "sydowii", "构巢", "杂色", "聚多"]),
]

# ---------- 正面/反面 关键词（文件名或文件夹名中出现即判定） ----------
REVERSE_KEYS = ["反面", "背面", "反", "back", "reverse", "rev", "bottom", "底"]
OBVERSE_KEYS = ["正面", "正", "front", "obverse", "obv", "top"]
# 单字母 F / R 只在作为独立词元时识别
TOKEN_SPLIT = re.compile(r"[-_\s.()（）\[\]]+")

DEFAULT_STRAIN_RE = None  # 默认使用下方 infer_strain_from_name


def section_of(text: str) -> str:
    t = text.lower()
    for sec, keys in SECTION_RULES:
        if any(k in t for k in keys):
            return sec
    return "其他/未识别"


def view_of(path: Path) -> str:
    names = [path.stem] + [p.name for p in path.parents][:2]
    for n in names:
        low = n.lower()
        toks = [x for x in TOKEN_SPLIT.split(low) if x]
        if any(k in low for k in REVERSE_KEYS if len(k) > 1) or "r" in toks or "反" in low:
            return "反面"
        if any(k in low for k in OBVERSE_KEYS if len(k) > 1) or "f" in toks or "正" in low:
            return "正面"
    return "未标注"


SUFFIX_TOKENS = {"正", "反", "正面", "反面", "背面", "front", "back", "reverse", "rev",
                 "obverse", "obv", "top", "bottom", "f", "r", "copy", "副本"}


_CN = [x for x in SUFFIX_TOKENS if not x.isascii()]
_EN = [x for x in SUFFIX_TOKENS if x.isascii()]
CN_RE = "|".join(sorted(map(re.escape, _CN), key=len, reverse=True))
EN_RE = "|".join(sorted(map(re.escape, _EN), key=len, reverse=True))
# 结尾可去掉的部分：
#   中文视角词（可直接粘连）；英文视角词/单字母 F、R（必须有分隔符）；
#   "_数字"、" 数字"、"(数字)"
# 注意：不去掉 "-数字"，因为株号本身常为 HQM-01、YQM-3 这种形式
TRAIL_RE = re.compile(
    rf"(?:[\s_\-]*(?:{CN_RE})|[\s_\-]+(?:{EN_RE})|_\d{{1,3}}|\s+\d{{1,3}}|\s*[(（]\d{{1,3}}[)）])+$",
    re.IGNORECASE)


def infer_strain_from_name(stem: str) -> str:
    s = TRAIL_RE.sub("", stem.strip())
    s = s.strip(" _-")
    return s or stem


def strain_of(path: Path, root: Path, strain_re):
    rel = path.relative_to(root).parts
    if len(rel) >= 3:                      # 菌种/株号/图片
        return rel[1], "子文件夹"
    if strain_re is not None:
        m = strain_re.search(path.stem)
        if m:
            return m.group(1), "正则"
    return infer_strain_from_name(path.stem), "文件名推断"


def exif_info(img: Image.Image):
    try:
        ex = img.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        model = str(tags.get("Model", "")).strip()
        make = str(tags.get("Make", "")).strip()
        dt = str(tags.get("DateTime", "")).strip()
        return f"{make} {model}".strip(), dt
    except Exception:
        return "", ""


def load_image(path: Path):
    """返回 (BGR ndarray 缩到长边 1024, 设备, 拍摄时间, 原始尺寸, 错误)"""
    try:
        with Image.open(path) as im:
            dev, dt = exif_info(im)
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            s = 1024 / max(w, h)
            if s < 1:
                im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
            arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            return arr, dev, dt, (w, h), ""
    except Exception as e:
        hint = "（HEIC 需安装 pillow-heif）" if path.suffix.lower() in {".heic", ".heif"} and not HEIF_OK else ""
        return None, "", "", (0, 0), f"{type(e).__name__}: {e}{hint}"


def detect_plate(bgr):
    """霍夫圆检测平皿。返回 (cx, cy, r) 或 None。"""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    m = min(h, w)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m,
                               param1=100, param2=40,
                               minRadius=int(m * 0.25), maxRadius=int(m * 0.55))
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r)


def background_check(bgr, plate):
    """平皿外 1.05R–1.25R 环形区域：是否为白纸。返回 (是否白背景, 中位亮度, 中位饱和度, 有效像素比例)"""
    if plate is None:
        return None, np.nan, np.nan, np.nan
    h, w = bgr.shape[:2]
    cx, cy, r = plate
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (d > 1.05 * r) & (d < 1.25 * r)
    frac_in_frame = ring.sum() / max(1, (np.pi * ((1.25 * r) ** 2 - (1.05 * r) ** 2)))
    if ring.sum() < 500:
        return False, np.nan, np.nan, frac_in_frame
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = np.median(hsv[..., 2][ring])
    s = np.median(hsv[..., 1][ring])
    is_white = (v >= 150) and (s <= 40)
    return bool(is_white), float(v), float(s), float(frac_in_frame)


def draw_overlay(bgr, plate, is_white, out_path: Path, label: str):
    vis = bgr.copy()
    if plate is not None:
        cx, cy, r = plate
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
        cv2.circle(vis, (int(cx), int(cy)), int(1.05 * r), (255, 200, 0), 1)
        cv2.circle(vis, (int(cx), int(cy)), int(1.25 * r), (255, 200, 0), 1)
    color = (0, 160, 0) if (plate is not None and is_white) else (0, 0, 255)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    s = 400 / max(vis.shape[:2])
    vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", vis)[1].tofile(str(out_path))


def collect_micro_strains(micro_root: Path, strain_re):
    """镜检训练图：返回 {section: set(株号)}"""
    res = defaultdict(set)
    if micro_root is None or not micro_root.exists():
        return res
    for p in micro_root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            rel = p.relative_to(micro_root).parts
            sec = section_of("/".join(rel[:-1]) or p.stem)
            sid, _ = strain_of(p, micro_root, strain_re)
            res[sec].add(norm_id(sid))
    return res


# 贡献者命名规则：<姓名缩写>-B-<序号> = 纯培养菌落；能匹配到即视为可追溯到株
TRACE_RE = re.compile(r"^([A-Za-z]{1,8}-B-\d+)", re.IGNORECASE)


def load_labels(path):
    """labels.txt：每行 '编号<TAB>菌种名[<TAB>...]'，返回 {编号: 菌种名}"""
    res = {}
    if not path:
        return res
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        parts = [x.strip() for x in line.split("\t")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            res[parts[0]] = parts[1]
    return res


def norm_id(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s).upper()


def main():
    ap = argparse.ArgumentParser(description="菌落图数据盘点（只读）")
    ap.add_argument("--colony_root", required=True)
    ap.add_argument("--micro_root", default=None, help="镜检训练图根目录（用于配对统计，可省略）")
    ap.add_argument("--out_dir", default="./00b_inventory_out")
    ap.add_argument("--strain_regex", default=None, help="自定义株号正则，第一个捕获组为株号")
    ap.add_argument("--qc_overlay", action="store_true", help="输出平皿检测缩略图，便于人工核查")
    ap.add_argument("--target", type=int, default=TARGET_STRAINS)
    ap.add_argument("--labels", default=None, help="labels.txt：文件夹编号 → 菌种名（制表符分隔）")
    args = ap.parse_args()
    labels = load_labels(args.labels)

    root = Path(args.colony_root).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    strain_re = re.compile(args.strain_regex) if args.strain_regex else None

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"在 {root} 下没有找到图片")
    print(f"共找到 {len(files)} 张图片，开始盘点 …")

    rows = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(root)
        folder_text = "/".join(rel.parts[:-1])
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        species = labels.get(top, "")
        sec = section_of(species) if species else "其他/未识别"
        if sec == "其他/未识别":
            sec = section_of(folder_text) if folder_text else section_of(p.stem)
        if sec == "其他/未识别":
            sec = section_of(p.stem)
        m_trace = TRACE_RE.match(p.stem)
        if m_trace:
            sid, sid_src, traceable = m_trace.group(1), "贡献者编号", True
        else:
            sid, sid_src = strain_of(p, root, strain_re)
            traceable = False
        view = view_of(p)
        bgr, dev, dt, (w, h), err = load_image(p)
        plate = detect_plate(bgr) if bgr is not None else None
        is_white, bg_v, bg_s, ring_frac = background_check(bgr, plate) if bgr is not None else (None, np.nan, np.nan, np.nan)
        rows.append(dict(
            相对路径=str(rel), 菌种文件夹=top, 菌种=species,
            组=sec, 株号=sid, 株号来源=sid_src, 可追溯株号=traceable, 视角=view,
            设备=dev, 拍摄时间=dt, 宽=w, 高=h,
            检测到平皿=plate is not None,
            平皿半径占短边比=(round(plate[2] / min(bgr.shape[:2]), 3) if plate is not None else np.nan),
            白纸背景=is_white, 背景亮度=round(bg_v, 1) if bg_v == bg_v else np.nan,
            背景饱和度=round(bg_s, 1) if bg_s == bg_s else np.nan,
            背景环在画面内比例=round(ring_frac, 2) if ring_frac == ring_frac else np.nan,
            读取错误=err,
        ))
        if args.qc_overlay and bgr is not None:
            tag = f"{sec} | {view} | {'plate' if plate else 'NO plate'} | {'white' if is_white else 'bg?'}"
            draw_overlay(bgr, plate, is_white, out / "qc_overlay" / sec.replace("/", "_") / (p.stem + ".jpg"), tag)
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    df = pd.DataFrame(rows)
    df["株号_规范"] = df["株号"].map(norm_id)
    df["iPhone"] = df["设备"].str.contains("iPhone", case=False, na=False)
    df.to_csv(out / "per_image.csv", index=False, encoding="utf-8-sig")

    # ---------- 镜检配对 ----------
    micro = collect_micro_strains(Path(args.micro_root).expanduser() if args.micro_root else None, strain_re)

    # ---------- 株级表 ----------
    g = df.groupby(["组", "株号_规范"])
    strain_df = g.agg(
        株号=("株号", "first"),
        菌种文件夹=("菌种文件夹", "first"),
        菌种=("菌种", "first"),
        可追溯=("可追溯株号", "first"),
        图片数=("相对路径", "size"),
        正面=("视角", lambda x: (x == "正面").sum()),
        反面=("视角", lambda x: (x == "反面").sum()),
        未标注视角=("视角", lambda x: (x == "未标注").sum()),
        可用图_平皿且白底=("白纸背景", lambda x: int((x == True).sum())),
    ).reset_index()
    strain_df["有镜检图"] = strain_df.apply(lambda r: r["株号_规范"] in micro.get(r["组"], set()), axis=1)
    strain_df["可用株"] = strain_df["可用图_平皿且白底"] > 0
    strain_df.to_csv(out / "per_strain.csv", index=False, encoding="utf-8-sig")

    # ---------- 组级汇总 ----------
    sections = ["Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes", "其他/未识别"]
    summ = []
    for sec in sections:
        s = strain_df[strain_df["组"] == sec]
        d = df[df["组"] == sec]
        if len(d) == 0 and sec == "其他/未识别":
            continue
        n_strain = len(s)
        n_usable = int(s["可用株"].sum())
        summ.append(dict(
            组=sec, 株数_含不明来源图组=n_strain,
            可追溯株数=int(s["可追溯"].sum()),
            不明来源图数=int((~d["可追溯株号"]).sum()),
            有正面株数=int((s["正面"] + s["未标注视角"] > 0).sum()),
            有反面株数=int((s["反面"] > 0).sum()),
            正面图数=int((d["视角"] == "正面").sum()),
            反面图数=int((d["视角"] == "反面").sum()),
            未标注视角图数=int((d["视角"] == "未标注").sum()),
            有镜检图的株数=int(s["有镜检图"].sum()),
            iPhone占比=round(d["iPhone"].mean(), 2) if len(d) else np.nan,
            平皿检出率=round(d["检测到平皿"].mean(), 2) if len(d) else np.nan,
            白底占比=round((d["白纸背景"] == True).mean(), 2) if len(d) else np.nan,
            可用株数_平皿且白底=n_usable,
            距目标缺口=(max(0, args.target - n_usable) if sec != "其他/未识别" else np.nan),
        ))
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(out / "summary_by_section.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out / "00b_inventory.xlsx") as xw:
        summ_df.to_excel(xw, sheet_name="组级汇总", index=False)
        strain_df.to_excel(xw, sheet_name="株级", index=False)
        df.to_excel(xw, sheet_name="逐图", index=False)

    # ---------- 株号推断示例（供人工核对） ----------
    with open(out / "strain_mapping_examples.txt", "w", encoding="utf-8") as f:
        f.write("每组随机抽 8 张：文件名 → 推断株号（来源）。如有错误请用 --strain_regex 修正。\n\n")
        for sec, d in df.groupby("组"):
            f.write(f"== {sec} ==\n")
            for _, r in d.sample(min(8, len(d)), random_state=0).iterrows():
                f.write(f"  {r['相对路径']}  [{r['菌种']}]  →  {r['株号']}  ({r['株号来源']}, {r['视角']})\n")
            f.write("\n")
        unk = df[df["组"] == "其他/未识别"][["菌种文件夹", "菌种"]].drop_duplicates().apply(lambda r: f"{r['菌种文件夹']} {r['菌种']}", axis=1)
        if len(unk):
            f.write("未能归入 5 个组的菌种文件夹（如属曲霉请在 SECTION_RULES 中补充关键词）：\n")
            for u in unk:
                f.write(f"  {u}\n")

    # ---------- 屏幕输出 ----------
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n===== 组级汇总 =====")
    print(summ_df.to_string(index=False))
    n_err = (df["读取错误"] != "").sum()
    if n_err:
        print(f"\n⚠ {n_err} 张图片读取失败，详见 per_image.csv 的“读取错误”列")
    print(f"\n输出目录：{out.resolve()}")
    print("请先打开 strain_mapping_examples.txt 核对株号推断是否正确，")
    if args.qc_overlay:
        print("并抽查 qc_overlay/ 中的平皿检测缩略图（绿圈 = 平皿，蓝色细环 = 白平衡参照区）。")


if __name__ == "__main__":
    main()
