#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_roi_clinical.py —— 临床验证菌株的盲法产孢结构截取（前瞻性临床验证专用）

用途
====
对前瞻性入组的临床菌株镜下图（纯培养，乳酸酚棉蓝染色）做盲法 ROI 截取：
所有图片随机打乱、匿名编号为 CL-0001、CL-0002……，截图窗口里不显示文件名、
文件夹名和菌种。截出的产孢结构用于评估模型能否按产孢结构识别和鉴别菌种。

本脚本独立运行，不调用、不修改 03_roi_blind.py；输出也与 roi_test 分开，
因此已用于论文的回顾性外部 ROI（roi_test）保持不变。

一、截图前必须先做好的事（写进验证方案并存档）
================================================
1. 锁定模型：选定一个最终模型，复制为固定名称（例如 runs/clinical_locked），
   验证期间不再重训或更换。
2. 锁定判读规则：病例级预测 = 该株全部 ROI 的平均概率；
   top1 < 0.70 或 top1 − top2 < 0.20 标记为"需复核"。
3. 入组：新分离、从未进入训练集的菌株，按时间连续入组；集外菌种也纳入。
4. 参考鉴定：MALDI-TOF 或 ITS 测序（曲霉加 BenA/CaM），结果写进单独的答案表，
   截图人员在截图完成前不能查看。

二、图片怎么放
==============
推荐：每株一个子文件夹，文件夹名只写株号，**不要写菌名**（文件夹名会写进对照表）。

    /Users/chunjiang/Desktop/临床验证_2026/
    ├── P001/   P001-01.jpg  P001-02.jpg  …
    ├── P002/   …
    └── P003/   …

也可以不建子文件夹，把图直接放在一个文件夹里，文件名以株号开头：
P001-01.jpg、P001-02.jpg、P002-01.jpg……脚本用 --id_regex 从文件名提取株号
（默认取最后一个 "-数字" 或 "_数字" 之前的部分，P001-01 → P001）。

支持 jpg / jpeg / png / tif / tiff / bmp；子文件夹可以多层，株号取第一层文件夹名。

三、使用方法（fungal 环境，Audit 文件夹）
==========================================
1) 截图（第一次运行会建立匿名对照表；中断后再运行，从断点继续）
    python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026"

   指定输出位置（默认 /Users/chunjiang/Documents/真菌AI/CNN/roi_clinical）：
    python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026" --out "/Users/chunjiang/Documents/真菌AI/CNN/roi_clinical"

2) 后续新入组了菌株：把新株文件夹放进同一个 --src，再运行同一条命令。
   新图会被单独打乱后接在原有编号后面（已有编号不变），然后接着截。

3) 查看进度（不打开窗口）
    python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026" --status

4) 生成空白答案表（按已识别的株号列好，只需填写参考鉴定结果）
    python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026" --make_answers
   在当前文件夹生成 answers_clinical.csv（已存在则不覆盖）；同时在屏幕上列出
   taxonomy_map.csv 里全部可用的 train_class 类名，填写时照抄。
   答案表由不参与截图的人填写，或截图全部完成后再填。

5) 全部截完后揭盲，生成评估用的总表
    python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026" --unblind --answers answers_clinical.csv
   还有未截的图时会拒绝揭盲；确需提前揭盲加 --force（论文里要说明）。

   answers_clinical.csv 的列（UTF-8，可以用 Excel 另存为"CSV UTF-8"）：
     isolate_id   株号，和文件夹名/文件名前缀一致，例如 P001
     reference    参考鉴定结果，例如 Aspergillus fumigatus
     train_class  对应模型 26 类的类名（与 taxonomy_map.csv 的 train_class 完全一致）；
                  集外菌种留空
     method       鉴定方法，例如 MALDI-TOF / ITS / ITS+BenA

四、截图窗口按键
================
    鼠标移动        移动截取框（框中心跟随鼠标）
    左键            截取当前框
    + / - 或右键    放大 / 缩小截取框（30–1200 px；右键循环放大，到最大后回到最小）
    ← / → 或 , / .  逆时针 / 顺时针旋转 10°；↑ 或 0 回正；↓ 在当前角度上再转 180°
    1 / 2 / 3       结构类型：完整产孢结构（默认）/ 局部 / 多个结构
    i               切换 成熟 ↔ 未成熟
    u               撤销本图最后一次截取（同时删除截图文件）
    n 或 空格       下一张（本图没截也没跳过的话，下次还会出现）
    b               上一张
    s               跳过本图：没有可用的产孢结构
    x               跳过本图：图像质量差（失焦、过曝、染色差等）
    q 或 Esc        保存并退出

    另有一个 224×224 预览窗口，显示模型实际看到的样子。

截图规范（与训练截图一致）：一个框 = 一个产孢结构，优先截顶囊、小梗、分生孢子链
都清楚的成熟结构；每张图截 1–5 个最典型的结构；没有就按 s，不要勉强截菌丝。

五、输出
========
<out>/
├── CL-0001__r00.jpg …      截图（文件名只含匿名编号）
├── crop_log.csv            每次截取/跳过一行：roi_file, image_id, cx, cy, size,
│                           angle, structure, stage, status, time
├── clinical_key.csv        匿名编号 ↔ 株号 ↔ 原图路径（不含菌名）
├── roi_clinical_table.csv  揭盲后生成：每个 ROI 一行，带株号、参考鉴定、train_class
└── unblind_summary.txt     揭盲后生成：入组株数、截图株数、跳过率及原因、每株 ROI 数

论文需要报告：入组株数、至少有 1 个 ROI 的株数、跳过的图数及原因（s / x）、
每株 ROI 数（中位数和范围）、结构类型和成熟度的构成。
"""
import argparse
import csv
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

try:
    import cv2
except ImportError:  # 揭盲和 --status 不需要 OpenCV
    cv2 = None

DEFAULT_OUT = "/Users/chunjiang/Documents/真菌AI/CNN/roi_clinical"
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SIZES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, \
			210, 224, 230, 240, 250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, \
			460, 480, 500, 520, 540, 560, 580, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, \
			1000, 1040, 1080, 1120, 1160, 1200, ]   # 按 + / - 切换；保存时保留原图像素，训练时统一缩放
STRUCT = {ord("1"): "complete", ord("2"): "partial", ord("3"): "multiple"}
SEED = 20261017
LOG_COLS = ["roi_file", "image_id", "cx", "cy", "size", "angle",
            "structure", "stage", "status", "time"]
KEY_COLS = ["image_id", "isolate_id", "rel_path"]

# 方向键码：macOS / Linux / Windows（cv2.waitKeyEx）
K_LEFT = {63234, 65361, 2424832, 2}
K_RIGHT = {63235, 65363, 2555904, 3}
K_UP = {63232, 65362, 2490368, 0}
K_DOWN = {63233, 65364, 2621440, 1}


# --------------------------------------------------------------------------- 文件读写
def imread(path):
    """支持中文路径的读图。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img,
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise IOError("写入失败: " + path)
    buf.tofile(path)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- 对照表
def scan_src(src, id_regex):
    """返回 [(rel_path, isolate_id)]。"""
    pat = re.compile(id_regex)
    items = []
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "_")))
        for fn in sorted(files):
            if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in EXTS:
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, src)
            parts = rel.split(os.sep)
            if len(parts) > 1:
                iso = parts[0]
            else:
                stem = os.path.splitext(fn)[0]
                m = pat.match(stem)
                iso = m.group(1) if m else stem
            items.append((rel, iso))
    return items


def build_or_update_key(src, out, id_regex):
    key_path = os.path.join(out, "clinical_key.csv")
    key = read_csv(key_path)
    known = {r["rel_path"] for r in key}
    found = scan_src(src, id_regex)
    if not found and not key:
        sys.exit(f"在 {src} 里没有找到图片（支持 {', '.join(sorted(EXTS))}）")
    new = [(rel, iso) for rel, iso in found if rel not in known]
    if new:
        rng = random.Random(SEED + len(key))  # 新增批次单独打乱，已有编号不变
        rng.shuffle(new)
        start = len(key)
        for i, (rel, iso) in enumerate(new, start=start + 1):
            key.append({"image_id": f"CL-{i:04d}", "isolate_id": iso, "rel_path": rel})
        write_csv(key_path, key, KEY_COLS)
        if start:
            print(f"新增 {len(new)} 张图（{len({i for _, i in new})} 株），编号接在 CL-{start:04d} 之后")
        else:
            print(f"已建立对照表：{len(new)} 张图，{len({i for _, i in new})} 株 → {key_path}")
    found_set = {rel for rel, _ in found}
    missing = [r for r in key if r["rel_path"] not in found_set]
    if missing:
        print(f"注意：对照表里有 {len(missing)} 张图在 --src 中找不到了（被移动或删除？），将跳过")
    return key, found_set


# --------------------------------------------------------------------------- 截取几何
def crop_matrix(cx, cy, size, angle):
    """把原图上以 (cx,cy) 为中心、旋转 angle 的方框映射到 size×size 输出。"""
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
    M[0, 2] += size / 2.0 - cx
    M[1, 2] += size / 2.0 - cy
    return M


def extract(img, cx, cy, size, angle):
    M = crop_matrix(cx, cy, size, angle)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def box_corners(cx, cy, size, angle):
    Minv = cv2.invertAffineTransform(crop_matrix(cx, cy, size, angle))
    pts = np.array([[0, 0, 1], [size, 0, 1], [size, size, 1], [0, size, 1]], float)
    return pts @ Minv.T  # 原图坐标


# --------------------------------------------------------------------------- 状态
def image_status(log):
    """每张图的最终状态：crop / skip_nostructure / skip_quality。"""
    st = {}
    for r in log:
        s = r["status"]
        if s == "crop":
            st[r["image_id"]] = "crop"
        elif s.startswith("skip") and st.get(r["image_id"]) != "crop":
            st[r["image_id"]] = s
    return st


def print_status(key, log, found_set):
    st = image_status(log)
    active = [r for r in key if r["rel_path"] in found_set]
    n = len(active)
    c = Counter(st.get(r["image_id"], "todo") for r in active)
    rois = sum(1 for r in log if r["status"] == "crop")
    iso_all = {r["isolate_id"] for r in active}
    iso_roi = {r["isolate_id"] for r in active if st.get(r["image_id"]) == "crop"}
    print(f"图片 {n} 张：已截 {c['crop']}，跳过（无结构）{c['skip_nostructure']}，"
          f"跳过（质量差）{c['skip_quality']}，未处理 {c['todo']}")
    print(f"ROI 共 {rois} 个；菌株 {len(iso_all)} 株，其中已有 ROI 的 {len(iso_roi)} 株")
    return c["todo"]


# --------------------------------------------------------------------------- 截图界面
class Cropper:
    WIN = "clinical ROI (blind)"
    PREV = "224 preview"

    def __init__(self, src, out, key, found_set, max_w, max_h):
        self.src, self.out = src, out
        self.items = [r for r in key if r["rel_path"] in found_set]
        self.log_path = os.path.join(out, "crop_log.csv")
        self.log = read_csv(self.log_path)
        self.max_w, self.max_h = max_w, max_h
        self.size_i = SIZES.index(224)
        self.angle = 0
        self.mouse = None
        self.dirty = True

    # ---- 日志
    def save_log(self):
        write_csv(self.log_path, self.log, LOG_COLS)

    def rois_of(self, iid):
        return [r for r in self.log if r["image_id"] == iid and r["status"] == "crop"]

    def next_roi_name(self, iid):
        used = {r["roi_file"] for r in self.log if r["image_id"] == iid}
        k = 0
        while f"{iid}__r{k:02d}.jpg" in used or os.path.exists(os.path.join(self.out, f"{iid}__r{k:02d}.jpg")):
            k += 1
        return f"{iid}__r{k:02d}.jpg"

    def add_row(self, iid, status, roi_file="", cx="", cy="", size="", angle=""):
        self.log.append({"roi_file": roi_file, "image_id": iid, "cx": cx, "cy": cy,
                         "size": size, "angle": angle, "structure": self.structure,
                         "stage": self.stage, "status": status,
                         "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save_log()

    # ---- 鼠标
    def on_mouse(self, ev, x, y, flags, param):
        if self.scale <= 0:
            return
        ox, oy = x / self.scale, y / self.scale
        if ev == cv2.EVENT_MOUSEMOVE:
            self.mouse = (ox, oy)
            self.dirty = True
        elif ev == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (ox, oy)
            self.do_crop()
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.size_i = (self.size_i + 1) % len(SIZES)
            self.dirty = True

    def do_crop(self):
        if self.mouse is None:
            return
        size = SIZES[self.size_i]
        cx, cy = self.mouse
        h, w = self.img.shape[:2]
        cx, cy = min(max(cx, 0), w - 1), min(max(cy, 0), h - 1)
        roi = extract(self.img, cx, cy, size, self.angle)
        name = self.next_roi_name(self.iid)
        imwrite(os.path.join(self.out, name), roi)
        self.add_row(self.iid, "crop", name, round(cx, 1), round(cy, 1), size, self.angle)
        self.flash = f"saved {name}"
        self.dirty = True

    def undo(self):
        for i in range(len(self.log) - 1, -1, -1):
            r = self.log[i]
            if r["image_id"] == self.iid:
                if r["status"] == "crop" and r["roi_file"]:
                    p = os.path.join(self.out, r["roi_file"])
                    if os.path.exists(p):
                        os.remove(p)
                del self.log[i]
                self.save_log()
                self.flash = "undo"
                self.dirty = True
                return
        self.flash = "nothing to undo"
        self.dirty = True

    # ---- 绘制
    def render(self):
        disp = cv2.resize(self.img, None, fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_AREA) if self.scale != 1 else self.img.copy()
        # 已截的框
        for r in self.rois_of(self.iid):
            pts = box_corners(float(r["cx"]), float(r["cy"]), int(r["size"]), float(r["angle"]))
            p = np.int32(pts * self.scale)
            cv2.polylines(disp, [p], True, (0, 200, 0), 2)
            # 截图左上角（正方向标记）：白边绿点，避免与绿色框线混在一起
            cv2.circle(disp, tuple(int(v) for v in p[0]), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(disp, tuple(int(v) for v in p[0]), 3, (0, 200, 0), -1, cv2.LINE_AA)
        size = SIZES[self.size_i]
        if self.mouse is not None:
            cx, cy = self.mouse
            pts = box_corners(cx, cy, size, self.angle)
            p = np.int32(pts * self.scale)
            cv2.polylines(disp, [p], True, (0, 0, 255), 2)
            # 截取框左上角 = 截图（及 224 预览）的左上角，用绿点标出正方向
            cv2.circle(disp, tuple(int(v) for v in p[0]), 6, (0, 255, 0), -1, cv2.LINE_AA)
            prev = cv2.resize(extract(self.img, cx, cy, size, self.angle), (224, 224),
                              interpolation=cv2.INTER_AREA)
            cv2.imshow(self.PREV, prev)
        st = image_status(self.log).get(self.iid, "todo")
        n_done = sum(1 for r in self.items if image_status(self.log).get(r["image_id"]))
        lines = [
            f"{self.iid}  ({self.pos + 1}/{len(self.items)}, done {n_done})  status: {st}",
            f"box {size}px  angle {self.angle:+d}  struct {self.structure}  stage {self.stage}"
            f"  ROIs here {len(self.rois_of(self.iid))}",
            "L-click crop | +/- size | arrows rotate (down=180) | 1/2/3 struct | i stage | u undo | "
            "n next | b prev | s no-structure | x poor-quality | q quit",
        ]
        if self.flash:
            lines.append(self.flash)
        y = 22
        for t in lines:
            cv2.putText(disp, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22
        cv2.imshow(self.WIN, disp)
        self.dirty = False

    # ---- 主循环
    def load(self, pos):
        self.pos = pos
        r = self.items[pos]
        self.iid = r["image_id"]
        self.img = imread(os.path.join(self.src, r["rel_path"]))
        self.structure, self.stage = "complete", "mature"
        self.angle, self.flash, self.mouse = 0, "", None
        if self.img is None:
            self.flash = "cannot read image -> press x to mark poor quality"
            self.img = np.zeros((400, 600, 3), np.uint8)
        h, w = self.img.shape[:2]
        self.scale = min(1.0, self.max_w / w, self.max_h / h)
        self.dirty = True

    def first_todo(self, start=0):
        st = image_status(self.log)
        for i in range(start, len(self.items)):
            if self.items[i]["image_id"] not in st:
                return i
        return None

    def run(self):
        start = self.first_todo()
        if start is None:
            print("全部图片都已截取或跳过。需要修改某张图，可以用 b 往回翻：从最后一张开始。")
            start = len(self.items) - 1
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.PREV, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WIN, self.on_mouse)
        self.load(start)
        while True:
            if self.dirty:
                self.render()
            k = cv2.waitKeyEx(30)
            if k == -1:
                continue
            ch = k & 0xFF if k < 256 else None
            if k in K_LEFT or ch == ord(","):
                self.angle -= 10                     # 逆时针
                self.angle = int(((self.angle + 180) % 360) - 180)
                self.dirty = True
            elif k in K_RIGHT or ch == ord("."):
                self.angle += 10                     # 顺时针
                self.angle = int(((self.angle + 180) % 360) - 180)
                self.dirty = True
            elif k in K_UP or ch == ord("0"):
                self.angle, self.dirty = 0, True
            elif k in K_DOWN:
                self.angle += 180                    # 在当前角度上翻转 180°
                self.angle = int(((self.angle + 180) % 360) - 180)
                self.dirty = True
            elif ch in (ord("+"), ord("=")):
                self.size_i = min(self.size_i + 1, len(SIZES) - 1); self.dirty = True
            elif ch in (ord("-"), ord("_")):
                self.size_i = max(self.size_i - 1, 0); self.dirty = True
            elif ch is not None and ch in STRUCT:
                self.structure, self.dirty = STRUCT[ch], True
            elif ch == ord("i"):
                self.stage = "immature" if self.stage == "mature" else "mature"; self.dirty = True
            elif ch == ord("u"):
                self.undo()
            elif ch in (ord("s"), ord("x")):
                status = "skip_nostructure" if ch == ord("s") else "skip_quality"
                if self.rois_of(self.iid):
                    self.flash = "this image already has ROIs; press u to remove them before skipping"
                    self.dirty = True
                    continue
                self.add_row(self.iid, status)
                nxt = self.first_todo(self.pos + 1)
                if nxt is None:
                    print_status(self.items, self.log, {r["rel_path"] for r in self.items})
                    print("已到最后一张未处理的图。")
                    break
                self.load(nxt)
            elif ch in (ord("n"), ord(" ")):
                if self.pos + 1 >= len(self.items):
                    self.flash = "last image"; self.dirty = True
                else:
                    self.load(self.pos + 1)
            elif ch == ord("b"):
                if self.pos > 0:
                    self.load(self.pos - 1)
            elif ch in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        self.save_log()


# --------------------------------------------------------------------------- 答案表模板
def make_answers(key, found_set, path):
    if os.path.exists(path):
        sys.exit(f"{os.path.abspath(path)} 已存在，不覆盖。要重新生成请先改名或删除旧文件。")
    isos = sorted({r["isolate_id"] for r in key if r["rel_path"] in found_set})
    write_csv(path, [{"isolate_id": i, "reference": "", "train_class": "", "method": ""} for i in isos],
              ["isolate_id", "reference", "train_class", "method"])
    print(f"已生成 {os.path.abspath(path)}：{len(isos)} 株（{'、'.join(isos[:10])}{' …' if len(isos) > 10 else ''}）")
    print("填写说明：reference 写参考鉴定到种；train_class 写下面列出的类名之一（照抄），集外菌种留空；"
          "method 写 MALDI-TOF / ITS / ITS+BenA 等。用 Excel 编辑后另存为\"CSV UTF-8\"。")
    tm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy_map.csv")
    rows = read_csv(tm)
    classes = sorted({(r.get("train_class") or "").strip() for r in rows} - {""})
    if classes:
        print(f"\n可用的 train_class（{len(classes)} 类，来自 taxonomy_map.csv）：")
        for c in classes:
            print("  " + c)
    else:
        print("\n（没找到 taxonomy_map.csv，train_class 请照抄模型的 class_names.json）")


# --------------------------------------------------------------------------- 揭盲
def unblind(out, key, found_set, answers_path, force):
    log = read_csv(os.path.join(out, "crop_log.csv"))
    todo = print_status(key, log, found_set)
    if todo and not force:
        sys.exit(f"还有 {todo} 张图未截取或跳过，拒绝揭盲。确需提前揭盲请加 --force（论文里要说明）。")
    if not answers_path:
        sys.exit("请用 --answers 指定答案表（isolate_id, reference, train_class, method）")
    answers_path = os.path.expanduser(answers_path.strip().strip("'\""))
    if not os.path.exists(answers_path):
        sys.exit(f"找不到答案表：{os.path.abspath(answers_path)}\n"
                 f"（相对路径是相对于当前文件夹 {os.getcwd()}；可以改写成完整路径）")
    ans_rows = read_csv(answers_path)
    need = {"isolate_id", "reference", "train_class", "method"}
    have = set(ans_rows[0].keys()) if ans_rows else set()
    if not need <= have:
        sys.exit(f"答案表缺少列：{', '.join(sorted(need - have))}（现有列：{', '.join(sorted(have)) or '无'}）\n"
                 "第一行必须是表头：isolate_id,reference,train_class,method")
    ans = {r["isolate_id"].strip(): r for r in ans_rows}
    active = [r for r in key if r["rel_path"] in found_set]
    by_img = {r["image_id"]: r for r in active}
    st = image_status(log)

    rows, miss = [], set()
    for r in log:
        if r["status"] != "crop" or r["image_id"] not in by_img:
            continue
        k = by_img[r["image_id"]]
        a = ans.get(k["isolate_id"])
        if a is None:
            miss.add(k["isolate_id"])
            a = {}
        tc = (a.get("train_class") or "").strip()
        rows.append({
            "roi_file": r["roi_file"], "roi_path": os.path.join(out, r["roi_file"]),
            "image_id": r["image_id"], "isolate_id": k["isolate_id"], "src_image": k["rel_path"],
            "reference": (a.get("reference") or "").strip(), "train_class": tc,
            "in_panel": (int(bool(tc)) if a else ""), "method": (a.get("method") or "").strip(),
            "structure": r["structure"], "stage": r["stage"], "size": r["size"], "angle": r["angle"],
        })
    cols = ["roi_file", "roi_path", "image_id", "isolate_id", "src_image", "reference",
            "train_class", "in_panel", "method", "structure", "stage", "size", "angle"]
    table = os.path.join(out, "roi_clinical_table.csv")
    write_csv(table, rows, cols)

    # 汇总
    iso_imgs = defaultdict(list)
    for r in active:
        iso_imgs[r["isolate_id"]].append(r["image_id"])
    roi_per_iso = Counter(r["isolate_id"] for r in rows)
    n_iso = len(iso_imgs)
    with_roi = [i for i in iso_imgs if roi_per_iso[i] > 0]
    without = sorted(i for i in iso_imgs if roi_per_iso[i] == 0)
    c = Counter(st.get(r["image_id"], "todo") for r in active)
    per_iso = [roi_per_iso[i] for i in with_roi]
    in_panel = [i for i in iso_imgs if (ans.get(i, {}).get("train_class") or "").strip()]
    no_ans = sorted(i for i in iso_imgs if i not in ans)

    L = []
    L.append(f"揭盲时间 {time.strftime('%Y-%m-%d %H:%M')}" + ("（--force 提前揭盲）" if todo else ""))
    L.append(f"入组菌株 {n_iso} 株（集内 {len(in_panel)}，集外 {n_iso - len(in_panel) - len(no_ans)}，"
             f"答案表缺失 {len(no_ans)}）；图片 {len(active)} 张")
    L.append(f"图片处理：截取 {c['crop']}，跳过-无产孢结构 {c['skip_nostructure']}，"
             f"跳过-质量差 {c['skip_quality']}，未处理 {c['todo']}")
    L.append(f"至少有 1 个 ROI 的菌株 {len(with_roi)}/{n_iso}；ROI 共 {len(rows)} 个")
    if per_iso:
        L.append(f"每株 ROI 数：中位数 {float(np.median(per_iso)):.1f}，范围 {min(per_iso)}–{max(per_iso)}")
    L.append("结构类型：" + "，".join(f"{k} {v}" for k, v in Counter(r['structure'] for r in rows).most_common()))
    L.append("成熟度：" + "，".join(f"{k} {v}" for k, v in Counter(r['stage'] for r in rows).most_common()))
    if without:
        L.append("没有 ROI 的菌株（不进入模型评估，论文需报告）：" + "、".join(without))
    if no_ans:
        L.append("答案表里找不到的株号（请补全后重新揭盲）：" + "、".join(no_ans))
    L.append("\n按参考类别（集内按 train_class，集外按 reference）的菌株数：")
    cls = Counter((ans.get(i, {}).get("train_class") or "").strip()
                  or "[集外] " + (ans.get(i, {}).get("reference") or "?").strip()
                  for i in with_roi if i in ans)
    for k, v in sorted(cls.items()):
        L.append(f"  {k}: {v}")
    txt = "\n".join(L)
    with open(os.path.join(out, "unblind_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt)
    print(f"\n已写出 {table}\n已写出 {os.path.join(out, 'unblind_summary.txt')}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="临床验证菌株的盲法产孢结构截取（详细用法见脚本开头注释）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="临床验证镜下图所在文件夹（每株一个子文件夹，或文件名以株号开头）")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"输出文件夹，默认 {DEFAULT_OUT}")
    ap.add_argument("--id_regex", default=r"^(.+?)[-_]\d+$",
                    help="图片直接放在 --src 下时，从文件名（不含扩展名）提取株号的正则，默认取最后一个 -数字/_数字 之前的部分")
    ap.add_argument("--status", action="store_true", help="只显示进度，不打开窗口")
    ap.add_argument("--unblind", action="store_true", help="揭盲：合并答案表，生成 roi_clinical_table.csv")
    ap.add_argument("--answers", help="答案表 CSV（isolate_id, reference, train_class, method）")
    ap.add_argument("--force", action="store_true", help="还有未截的图时也揭盲")
    ap.add_argument("--make_answers", action="store_true", help="按株号生成空白答案表 answers_clinical.csv")
    ap.add_argument("--max_w", type=int, default=1400, help="窗口最大宽度（像素）")
    ap.add_argument("--max_h", type=int, default=850, help="窗口最大高度（像素）")
    a = ap.parse_args()

    src = os.path.expanduser(a.src.strip().strip("'\""))
    out = os.path.expanduser(a.out.strip().strip("'\""))
    if not os.path.isdir(src):
        sys.exit(f"找不到 --src 文件夹：{src}")
    if os.path.abspath(out).rstrip("/").endswith("roi_test"):
        sys.exit("输出不能放在 roi_test（那是回顾性外部 ROI），请换一个文件夹")
    os.makedirs(out, exist_ok=True)

    key, found_set = build_or_update_key(src, out, a.id_regex)
    if a.make_answers:
        make_answers(key, found_set, a.answers or "answers_clinical.csv")
        return
    if a.unblind:
        unblind(out, key, found_set, a.answers, a.force)
        return
    if a.status:
        print_status(key, read_csv(os.path.join(out, "crop_log.csv")), found_set)
        return
    if cv2 is None:
        sys.exit("需要 OpenCV：pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple")
    Cropper(src, out, key, found_set, a.max_w, a.max_h).run()
    print_status(key, read_csv(os.path.join(out, "crop_log.csv")), found_set)


if __name__ == "__main__":
    main()
