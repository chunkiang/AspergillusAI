#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_crop_batch.py —— 批量点选截图（在你原来 cutter_3_small.py 基础上改的）

操作和原来一样：鼠标移动显示 160×160 框，左键点击保存。新增：
  · 按 manifest 自动列出要截的图，一张接一张，不用每次输入路径和前缀
  · 文件名自动带来源编号和菌名：WEB1-C-36__黄曲霉__000.jpg（以后能追溯到原图）
  · 大图自动缩放到屏幕，但截取仍按原图像素的 160×160；靠近边缘时框会内移，保证截图都是完整的 160×160
  · 每次截取都记录到 crop_log.csv（原图、坐标、菌种），写论文方法和复现都要用
  · 截过的原图记入 crop_log.csv，评估脚本会自动把它们从外部测试集里排除

按键：
  左键   截取当前框
  + / -  或 右键   放大 / 缩小截取框；右键 = 循环放大（800 之后回到 96），方便单手操作
  ← / →         截取框逆时针 / 顺时针旋转 10°（键盘没反应时可用 , 和 .）；↑ 或 0 回正
                旋转后截出的图会摆正保存，框的粗线一边就是截图的上边
         截取框（96–800 像素）：结构大或放大倍数高时用大框框住完整产孢结构；保存原图像素，训练时统一缩放
  u      撤销本图最后一次截取（删除文件）
  n / 空格  下一张图
  b      上一张图
  s      跳过本图（不适合截取），记为 skipped
  q      保存并退出（下次运行自动从没做完的图继续）

用法：
  # 1) 按 manifest 批量截：网络图片中的黄曲霉、黑曲霉、横梗霉
  python 01_crop_batch.py --manifest ext_audit/manifest.csv \
      --classes "Aspergillus flavus (sect. Flavi)" "Aspergillus niger (sect. Nigri)" Lichtheimia \
      --folders 网络

  # 2) 单张图（和原来的用法一样）
  python 01_crop_batch.py --image /path/to/xxx.jpg --label 黄曲霉

依赖（清华源）：
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python pandas
"""
import os
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

BOX = 160                      # 默认框大小（原图像素）
BOX_SIZES = [96, 128, 160, 224, 320, 448, 640, 800]   # 按 + / - 切换；保存时保留原图像素，训练时统一缩放
EXT_ROOT = "/Users/chunjiang/Desktop/丝状真菌"
OUT_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/crops_new"
SCREEN_W, SCREEN_H = 1600, 950
ANGLE_STEP = 10                # 每按一次 ← / → 旋转的角度
LOG_FIELDS = ["time", "status", "source_image", "source_code", "species_cn", "train_class",
              "x", "y", "w", "h", "cx", "cy", "angle", "crop_file"]
# cv2.waitKeyEx 返回的方向键码（macOS / Linux / Windows）
KEYS_LEFT = {63234, 65361, 2424832}
KEYS_RIGHT = {63235, 65363, 2555904}
KEYS_UP = {63232, 65362, 2490368}

FOLDER_TAG = {"网络真菌图片": "WEB1", "网络真菌图2": "WEB2", "谭盼": "TP", "胡音音": "HYY",
              "马琼琼": "MQQ", "李一凡": "LYF", "李艳": "LY", "田蕴琦": "TYQ"}


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


def build_jobs(a):
    if a.image:
        p = os.path.expanduser(a.image.strip().strip("'\""))
        code = os.path.splitext(os.path.basename(p))[0]
        return [dict(path=p, code=safe(code), species_cn=a.label or "", train_class=a.label or "")]
    import pandas as pd
    df = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
    df = df[df["role"] == "external_micro"]
    if a.classes:
        df = df[df["train_class"].isin(a.classes)]
    if a.folders:
        df = df[df["folder"].apply(lambda f: any(k in f for k in a.folders))]
    jobs = []
    for _, r in df.sort_values(["train_class", "folder", "rel_path"]).iterrows():
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
            if banner_key != (len(boxes), box, angle):
                banner = make_banner(f"{i + 1}/{len(pending)}   {job['code']}   {job['species_cn']} → {job['train_class']}"
                                     f"   已截 {len(boxes)} 张   框 {box}×{box}  角度 {angle:+.0f}°\n左键截取 +-或右键调框 ←→旋转 ↑回正 u撤销 n下一张 b上一张 s跳过 q退出",
                                     view.shape[1])
                banner_key = (len(boxes), box, angle)
            bh, bw = banner.shape[:2]
            bw = min(bw, view.shape[1])
            view[:bh, :bw] = cv2.addWeighted(view[:bh, :bw], 0.25, banner[:, :bw], 0.75, 0)
            cv2.imshow("crop", view)
            kx = cv2.waitKeyEx(20)
            k = kx & 0xFF if 0 <= kx < 0x10000 and kx not in KEYS_LEFT | KEYS_RIGHT | KEYS_UP else -1
            if kx in KEYS_LEFT or k in (ord(","), ord("<")):
                angle = state["angle"] = (angle + ANGLE_STEP + 180) % 360 - 180      # 逆时针
            elif kx in KEYS_RIGHT or k in (ord("."), ord(">")):
                angle = state["angle"] = (angle - ANGLE_STEP + 180) % 360 - 180      # 顺时针
            elif kx in KEYS_UP or k == ord("0"):
                angle = state["angle"] = 0
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
                                              angle=angle, crop_file=path))
                    print(f"  保存 {os.path.basename(path)}  (中心 {c[0]:.0f},{c[1]:.0f}  {box}×{box}  {angle:+.0f}°)")
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
                    x="", y="", w="", h="", cx="", cy="", angle="", crop_file="")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=OP.MANIFEST)
    ap.add_argument("--ext_root", default=EXT_ROOT)
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--folders", nargs="*", default=[], help="文件夹名关键字，如 网络")
    ap.add_argument("--image", default="", help="单张图路径（兼容原来用法）")
    ap.add_argument("--label", default="", help="单张图模式下的菌名")
    ap.add_argument("--out_root", default=OUT_ROOT)
    ap.add_argument("--all", action="store_true", help="不按类别筛选，列出全部外部镜下图（一般不要用）")
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    if not a.image and not a.classes and not a.all:
        ap.error("请用 --classes 指定要补截图的类别（外部测试图一旦截取就不能再用于测试）；"
                 "确实要列出全部外部图，请加 --all")
    os.makedirs(a.out_root, exist_ok=True)
    run(build_jobs(a), a.out_root, os.path.join(a.out_root, "crop_log.csv"))


if __name__ == "__main__":
    main()
