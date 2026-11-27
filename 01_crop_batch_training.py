#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_crop_batch_training.py —— 规范化重建训练截图（以曲霉为首），操作方式与 01_crop_batch.py 相同

与 01_crop_batch.py 的区别：
  · 只处理"开发集"原图：manifest 中 role 为 dev_source（已整理、eXPORT）、dev_set_checked（✅ 文件夹），
    或用 --folder 指定任意文件夹（如本院 烟曲霉-01/02 的原图）；绝不会列出外部测试图
  · 输出到 CNN/crops_v2/，日志 crops_v2/crop_log.csv，与旧截图完全分开
  · 每张截图都标注【结构类型】和【生长阶段】，训练时可据此筛选，论文方法里也能写清楚截图规范
  · 多一个"预览"窗口：显示当前框缩放到 224×224 后的样子，也就是模型实际看到的图

截图规范（建议）：
  一个框 = 一个产孢结构。优先截"完整、成熟"的结构：顶囊 + 小梗 + 分生孢子链都清楚可见；
  结构太大装不下时截最有鉴别意义的部分并标为"局部"；没有孢子的幼稚结构标为"未成熟"。

按键：
  左键            截取当前框
  + / -  或 右键   放大 / 缩小截取框（右键循环放大）
  ← / →           逆时针 / 顺时针旋转 10°（或 , 和 .）；↑ 回正
  1 / 2 / 3       结构类型：1 = 完整产孢结构（默认）  2 = 局部  3 = 多个结构
  i               切换生长阶段：成熟（有孢子，默认） ↔ 未成熟（无孢子 / 孢子很少）
  u   撤销本图最后一次截取      n / 空格  下一张      b  上一张
  s   跳过本图                 q  保存并退出（下次从断点继续）

用法：
  # 1) 按 manifest 列出开发集原图（曲霉 7 组）
  python 01_crop_batch_training.py --classes "Aspergillus fumigatus (sect. Fumigati)" \
      "Aspergillus flavus (sect. Flavi)" "Aspergillus niger (sect. Nigri)" "Aspergillus terreus (sect. Terrei)" \
      "Aspergillus nidulans (sect. Nidulantes)" "Aspergillus versicolor (sect. Versicolores)" \
      "Aspergillus glaucus (sect. Aspergillus)"

  # 2) 任意文件夹（所有图同一菌种）；会递归子文件夹，建议 <菌种文件夹>/<菌株编号>/原图
  python 01_crop_batch_training.py --folder "/Users/chunjiang/Documents/真菌AI/原图_重截/弯孢霉" --label 弯孢霉 --code_prefix PUPH-CUR

  # 3) 网络图补截（会占用网络测试图，需显式加 --allow_web）
  python 01_crop_batch_training.py --classes "Aspergillus fumigatus (sect. Fumigati)" --allow_web

依赖（清华源）：
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python pandas pillow
"""
import os
import sys
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

BOX = 160                      # 默认框大小（原图像素）
BOX_SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, \
			210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, \
			460, 480, 500, 520, 540, 560, 580, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, \
			1000, 1040, 1080, 1120, 1160, 1200, ]   # 按 + / - 切换；保存时保留原图像素，训练时统一缩放
EXT_ROOT = "/Users/chunjiang/Desktop/丝状真菌"
OUT_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/crops_v2"
SCREEN_W, SCREEN_H = 1600, 950
ANGLE_STEP = 10                # 每按一次 ← / → 旋转的角度
LOG_FIELDS = ["time", "status", "source_image", "source_code", "species_cn", "train_class",
              "x", "y", "w", "h", "cx", "cy", "angle", "structure", "stage", "crop_file"]
STRUCTURE = {"1": "完整", "2": "局部", "3": "多个"}
STAGE = {True: "成熟", False: "未成熟"}
DEV_ROLES = ("dev_source", "dev_set_checked")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# cv2.waitKeyEx 返回的方向键码（macOS / Linux / Windows）
KEYS_LEFT = {63234, 65361, 2424832}
KEYS_RIGHT = {63235, 65363, 2555904}
KEYS_UP = {63232, 65362, 2490368}

FOLDER_TAG = {"网络真菌图片": "WEB1", "网络真菌图2": "WEB2", "谭盼": "TP", "胡音音": "HYY",
              "马琼琼": "MQQ", "李一凡": "LYF", "李艳": "LY", "田蕴琦": "TYQ",
              "已整理": "PUPH-ZL", "eXPORT": "PUPH-EX", "麦老师": "MCW", "HSJ": "HSJ",
              "蒋宇杰": "JYJ", "申克": "SK"}


FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_FONT = None


def get_font(size=18):
    """cv2.putText 不支持中文，改用 PIL + 系统中文字体绘制"""
    global _FONT
    if _FONT is None:
        for p in FONT_CANDIDATES:
            if os.path.exists(p):
                try:
                    _FONT = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
        if _FONT is None:
            _FONT = False
    return _FONT or None


def make_banner(text, width):
    """生成一条深色底、白字的状态栏（BGR numpy），每张图/每次截取后重绘一次"""
    font = get_font()
    lines = text.split("\n")
    h = 6 + 24 * len(lines)
    im = Image.new("RGB", (max(width, 200), h), (30, 30, 30))
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        if font is None:
            d.text((8, 8 + 24 * k), line.encode("ascii", "replace").decode(), fill=(255, 255, 255))
        else:
            d.text((8, 3 + 24 * k), line, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def folder_tag(folder):
    for k, v in FOLDER_TAG.items():
        if k in folder:
            return v
    return "".join(ch for ch in folder if ch.isalnum())[:8]


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s)).strip("_")


def box_at(cx, cy, w, h, box=BOX):
    """（不旋转时）以点击点为中心取 box×box，返回左上角；靠边时整体内移。图比框还小时返回 None"""
    if w < box or h < box:
        return None
    x0 = int(min(max(cx - box // 2, 0), w - box))
    y0 = int(min(max(cy - box // 2, 0), h - box))
    return x0, y0


def _rot(angle):
    """画面上逆时针为正（图像坐标 y 轴向下）。返回把框内坐标 (dx, dy) 映射到图像坐标的 2×2 矩阵"""
    a = np.deg2rad(angle)
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, s_], [-s_, c]])


def place_box(cx, cy, w, h, box, angle):
    """返回旋转框的中心（靠边时内移，保证四个角都在图内）；放不下返回 None"""
    a = np.deg2rad(angle)
    half = box / 2 * (abs(np.cos(a)) + abs(np.sin(a)))
    if w < 2 * half or h < 2 * half:
        return None
    return float(min(max(cx, half), w - half)), float(min(max(cy, half), h - half))


def box_corners(cx, cy, box, angle):
    """旋转框四个角（左上、右上、右下、左下）的图像坐标"""
    d = box / 2
    off = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
    return off @ _rot(angle).T + np.array([cx, cy])


def extract_rotated(img, cx, cy, box, angle):
    """把旋转框里的内容摆正后取出，得到 box×box 的图"""
    A = _rot(angle)
    t = np.array([cx, cy]) - A @ np.array([box / 2, box / 2])
    M = np.hstack([A, t[:, None]]).astype(np.float64)
    return cv2.warpAffine(img, M, (box, box), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def load_log(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def append_log(path, row):
    new = not os.path.exists(path)
    if not new:
        with open(path, encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")
        if header != LOG_FIELDS:      # 旧版日志缺少 cx/cy/angle 列：按新表头重写一遍
            rows = load_log(path)
            for r in rows:
                if r.get("status") == "crop" and not r.get("cx"):
                    r["cx"] = int(r["x"]) + int(r["w"]) / 2
                    r["cy"] = int(r["y"]) + int(r["h"]) / 2
                    r["angle"] = 0
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_taxonomy(path):
    import pandas as pd
    t = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    return dict(zip(t["species_cn"], t["train_class"]))


def build_jobs(a):
    if a.folder:
        root = os.path.expanduser(a.folder.strip().strip("'\""))
        if not os.path.isdir(root):
            sys.exit(f"找不到文件夹：{root}")
        if not a.label:
            sys.exit("--folder 模式需要 --label（菌名，如 烟曲霉）")
        tax = load_taxonomy(a.taxonomy)
        label = tax.get(a.label, "")
        if not label:
            valid = sorted(set(v for v in tax.values() if v))
            if a.label in valid:
                label = a.label
            else:
                sys.exit(f"--label {a.label} 不在 taxonomy_map.csv 的 species_cn 里（或在但本期不训练）。"
                         f"请用表中的中文名，或直接写训练类名，如：{valid[:6]} …")
        prefix = a.code_prefix or safe(os.path.basename(root.rstrip("/")))
        jobs = []
        # 递归：<root>/<菌株编号>/*.jpg —— 子文件夹名进入截图编号，build_trainset 据此按菌株分组
        for d, subs, files in os.walk(root):
            subs[:] = sorted(x for x in subs if not x.startswith("."))
            rel = os.path.relpath(d, root)
            for fn in sorted(files):
                if fn.lower().endswith(IMG_EXTS) and not fn.startswith("."):
                    tag = "" if rel == "." else safe(rel.replace(os.sep, "-")) + "-"
                    jobs.append(dict(path=os.path.join(d, fn),
                                     code=f"{prefix}-{tag}{safe(os.path.splitext(fn)[0])}",
                                     species_cn=a.label, train_class=label))
        n_sub = len({os.path.dirname(j["path"]) for j in jobs})
        print(f"{root}：{len(jobs)} 张原图，分布在 {n_sub} 个文件夹（菌株）；训练标签 → {label}")
        return jobs
    import pandas as pd
    df = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
    roles = DEV_ROLES + (("external_micro",) if a.allow_web else ())
    df = df[df["role"].isin(roles) & (df["modality_guess"] != "colony") & (df["train_class"] != "")]
    if a.allow_web:   # 只放开网络图，临床外部图永远不列出
        df = df[(df["role"] != "external_micro") | df["folder"].str.contains("网络")]
    if a.classes:
        df = df[df["train_class"].isin(a.classes)]
    if a.folders:
        df = df[df["folder"].apply(lambda f: any(k in f for k in a.folders))]
    df = df[~df["rel_path"].str.contains("_small")]      # HSJ 文件夹里的缩小副本
    jobs = []
    for _, r in df.sort_values(["train_class", "role", "folder", "rel_path"]).iterrows():
        stem = os.path.splitext(os.path.basename(r["rel_path"]))[0].strip()
        jobs.append(dict(path=os.path.join(a.ext_root, r["rel_path"]),
                         code=f"{folder_tag(r['folder'])}-{safe(stem)}",
                         species_cn=r["species_cn"], train_class=r["train_class"]))
    return jobs


def crop_and_save(img, cx, cy, job, out_root, idx, box=BOX, angle=0):
    if angle % 360 == 0:
        x0, y0 = int(round(cx - box / 2)), int(round(cy - box / 2))
        crop = img[y0:y0 + box, x0:x0 + box]
    else:
        crop = extract_rotated(img, cx, cy, box, angle)
    d = os.path.join(out_root, safe(job["train_class"]) or "unlabeled")
    os.makedirs(d, exist_ok=True)
    fn = f"{job['code']}__{safe(job['species_cn'])}__{idx:03d}.jpg"
    path = os.path.join(d, fn)
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def run(jobs, out_root, log_path):
    log = load_log(log_path)
    done = {r["source_image"] for r in log if r["status"] in ("done", "skipped")}
    pending = [j for j in jobs if j["path"] not in done]
    print(f"共 {len(jobs)} 张，已完成 {len(jobs) - len(pending)} 张，待处理 {len(pending)} 张")
    if not pending:
        return
    state = {"mouse": None, "click": None}

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            # 右键：框循环放大（到 800 后回到 96），单手操作
            state["resize"] = "cycle"

    cv2.namedWindow("crop", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("preview 224", cv2.WINDOW_AUTOSIZE)
    state.setdefault("structure", "1")
    state.setdefault("mature", True)
    cv2.setMouseCallback("crop", on_mouse)
    i = 0
    while 0 <= i < len(pending):
        job = pending[i]
        img = cv2.imread(job["path"])
        if img is None:
            print(f"无法读取，跳过：{job['path']}")
            i += 1
            continue
        h, w = img.shape[:2]
        sc = min(1.0, SCREEN_W / w, SCREEN_H / h)
        disp = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else img.copy()
        prev = [r for r in load_log(log_path) if r["source_image"] == job["path"] and r["status"] == "crop"]
        boxes = []
        for r in prev:
            bs = int(r.get("w") or BOX)
            cx = float(r["cx"]) if r.get("cx") else int(r["x"]) + bs / 2
            cy = float(r["cy"]) if r.get("cy") else int(r["y"]) + bs / 2
            boxes.append((cx, cy, bs, float(r.get("angle") or 0), r["crop_file"]))
        box = state.get("box", BOX)
        angle = state.get("angle", 0)
        title = f"[{i + 1}/{len(pending)}] {job['code']}  {job['species_cn']}  →  {job['train_class']}"
        print("\n" + title + f"   原图 {w}x{h}  显示比例 {sc:.2f}")
        state["click"] = None
        action = None
        banner_key = None
        while action is None:
            view = disp.copy()
            ov = view.copy()
            for (bcx, bcy, bs, ba, _) in boxes:
                pts = (box_corners(bcx, bcy, bs, ba) * sc).astype(np.int32)
                cv2.fillPoly(ov, [pts], (0, 0, 255))
                cv2.polylines(view, [pts], True, (0, 255, 0), 1)
            cv2.addWeighted(ov, 0.3, view, 0.7, 0, view)
            if state["mouse"]:
                c = place_box(state["mouse"][0] / sc, state["mouse"][1] / sc, w, h, box, angle)
                if c:
                    pts = (box_corners(c[0], c[1], box, angle) * sc).astype(np.int32)
                    ov2 = view.copy()
                    cv2.fillPoly(ov2, [pts], (219, 152, 52))
                    cv2.addWeighted(ov2, 0.3, view, 0.7, 0, view)
                    cv2.polylines(view, [pts], True, (219, 152, 52), 1)
                    # 框的"上边"画粗一点，提示截出来的图哪边朝上
                    cv2.line(view, tuple(pts[0]), tuple(pts[1]), (219, 152, 52), 3)
            tag = f"{STRUCTURE[state['structure']]}·{STAGE[state['mature']]}"
            if banner_key != (len(boxes), box, angle, tag):
                banner = make_banner(f"{i + 1}/{len(pending)}   {job['code']}   {job['species_cn']} → {job['train_class']}"
                                     f"   已截 {len(boxes)} 张   框 {box}×{box}  角度 {angle:+.0f}°   标注【{tag}】\n1完整 2局部 3多个 i成熟/未成熟  左键截取 +-或右键调框 ←→旋转 ↑回正 u撤销 n下一张 b上一张 s跳过 q退出",
                                     view.shape[1])
                banner_key = (len(boxes), box, angle, tag)
            bh, bw = banner.shape[:2]
            bw = min(bw, view.shape[1])
            view[:bh, :bw] = cv2.addWeighted(view[:bh, :bw], 0.25, banner[:, :bw], 0.75, 0)
            cv2.imshow("crop", view)
            if state["mouse"]:
                c = place_box(state["mouse"][0] / sc, state["mouse"][1] / sc, w, h, box, angle)
                if c:
                    prev_img = extract_rotated(img, c[0], c[1], box, angle)
                    cv2.imshow("preview 224", cv2.resize(prev_img, (224, 224), interpolation=cv2.INTER_AREA))
            kx = cv2.waitKeyEx(20)
            k = kx & 0xFF if 0 <= kx < 0x10000 and kx not in KEYS_LEFT | KEYS_RIGHT | KEYS_UP else -1
            if kx in KEYS_LEFT or k in (ord(","), ord("<")):
                angle = state["angle"] = (angle + ANGLE_STEP + 180) % 360 - 180      # 逆时针
            elif kx in KEYS_RIGHT or k in (ord("."), ord(">")):
                angle = state["angle"] = (angle - ANGLE_STEP + 180) % 360 - 180      # 顺时针
            elif kx in KEYS_UP:
                angle = state["angle"] = 0
            if k in (ord("1"), ord("2"), ord("3")):
                state["structure"] = chr(k)
            elif k == ord("i"):
                state["mature"] = not state["mature"]
            if state["click"]:
                c = place_box(state["click"][0] / sc, state["click"][1] / sc, w, h, box, angle)
                state["click"] = None
                if c is None:
                    print(f"  原图放不下 {box}×{box}（旋转 {angle:+.0f}°）的框，请缩小框或回正")
                elif any(abs(c[0] - x) < 8 and abs(c[1] - y) < 8 and bs == box and ba == angle
                         for x, y, bs, ba, _ in boxes):
                    print("  该区域已截过，跳过")
                else:
                    path = crop_and_save(img, c[0], c[1], job, out_root, len(boxes), box, angle)
                    boxes.append((c[0], c[1], box, angle, path))
                    append_log(log_path, dict(time=datetime.now().isoformat(timespec="seconds"), status="crop",
                                              source_image=job["path"], source_code=job["code"],
                                              species_cn=job["species_cn"], train_class=job["train_class"],
                                              x=int(round(c[0] - box / 2)), y=int(round(c[1] - box / 2)),
                                              w=box, h=box, cx=round(c[0], 1), cy=round(c[1], 1),
                                              angle=angle, structure=STRUCTURE[state["structure"]],
                                              stage=STAGE[state["mature"]], crop_file=path))
                    print(f"  保存 {os.path.basename(path)}  ({box}×{box} {angle:+.0f}° {STRUCTURE[state['structure']]}·{STAGE[state['mature']]})")
            step = state.pop("resize", None)
            if k in (ord("+"), ord("=")):
                step = 1
            elif k in (ord("-"), ord("_")):
                step = -1
            if step is not None:
                j = BOX_SIZES.index(box) if box in BOX_SIZES else BOX_SIZES.index(BOX)
                if step == "cycle":
                    j = (j + 1) % len(BOX_SIZES)
                else:
                    j = min(max(j + step, 0), len(BOX_SIZES) - 1)
                box = state["box"] = BOX_SIZES[j]
                print(f"  框大小 → {box}×{box}")
            if k == ord("u") and boxes:
                *_, path = boxes.pop()
                if os.path.exists(path):
                    os.remove(path)
                rows = [r for r in load_log(log_path)
                        if not (r["status"] == "crop" and r["crop_file"] == path)]
                with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
                    wtr = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
                    wtr.writeheader()
                    wtr.writerows(rows)
                print(f"  撤销 {os.path.basename(path)}")
            elif k in (ord("n"), ord(" ")):
                action = "next"
            elif k == ord("b"):
                action = "back"
            elif k == ord("s"):
                action = "skip"
            elif k == ord("q"):
                action = "quit"
        base = dict(time=datetime.now().isoformat(timespec="seconds"), source_image=job["path"],
                    source_code=job["code"], species_cn=job["species_cn"], train_class=job["train_class"],
                    x="", y="", w="", h="", cx="", cy="", angle="", structure="", stage="", crop_file="")
        if action == "skip":
            append_log(log_path, {**base, "status": "skipped"})
            i += 1
        elif action == "next":
            if boxes:
                append_log(log_path, {**base, "status": "done"})
            i += 1
        elif action == "back":
            i = max(0, i - 1)
        elif action == "quit":
            if boxes:
                append_log(log_path, {**base, "status": "done"})
            break
    cv2.destroyAllWindows()
    log = load_log(log_path)
    n_crop = sum(r["status"] == "crop" for r in log)
    n_img = len({r["source_image"] for r in log if r["status"] == "crop"})
    print(f"\n累计：{n_img} 张原图，{n_crop} 张截图。记录在 {log_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=OP.MANIFEST)
    ap.add_argument("--taxonomy", default=os.path.join(here, "taxonomy_map.csv"))
    ap.add_argument("--ext_root", default=EXT_ROOT)
    ap.add_argument("--classes", nargs="*", default=[], help="训练标签（与 taxonomy_map.csv 的 train_class 一致）")
    ap.add_argument("--folders", nargs="*", default=[], help="只看这些文件夹（名称关键字）")
    ap.add_argument("--folder", default="", help="任意原图文件夹（需配合 --label）")
    ap.add_argument("--label", default="", help="--folder 模式下的菌名，如 烟曲霉")
    ap.add_argument("--code_prefix", default="", help="--folder 模式下的来源编号前缀，如 PUPH-AF01")
    ap.add_argument("--allow_web", action="store_true", help="也列出网络图（截过后不再用于网络测试）")
    ap.add_argument("--out_root", default=OUT_ROOT)
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    if not a.folder and not a.classes:
        ap.error("请用 --classes 指定类别，或用 --folder + --label 指定原图文件夹")
    jobs = build_jobs(a)
    if not jobs:
        sys.exit("没有找到符合条件的原图")
    os.makedirs(a.out_root, exist_ok=True)
    run(jobs, a.out_root, os.path.join(a.out_root, "crop_log.csv"))


if __name__ == "__main__":
    main()
