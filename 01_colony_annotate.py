#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_colony_annotate.py — 菌落照片几何标注工具（只读原图，标注另存 JSON）

每张菌落照片只标几何信息，截图由后续 01_colony_preprocess.py 按这些信息自动生成：
  正面：① 平皿边界  ② 菌落中心  ③ 菌落边缘
  反面：① 平皿边界  ② 菌落中心  ③ 框选标签/文字区域（可 0 个）  ④ 菌落边缘

安装（清华源；tkinter 随 Python 自带）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python pillow pillow-heif

运行
  python 01_colony_annotate.py --root "/path/to/补拍_2026" --annotator 赵
  python 01_colony_annotate.py --root ... --redo                 # 重新标注已标过的图（载入原标注作为起点）
  python 01_colony_annotate.py --root ... --only PUPH-FUM-007    # 只标某一株
  python 01_colony_annotate.py --root ... --include_nonstandard  # 也处理不符合命名规则的旧图

标注保存在  Audit/output/01_trainsets/colony_annotations/<组>/<原文件名>.json
（旧版存放在 <root>/annotations 的标注，首次运行时会自动复制过来）

操作
  鼠标左键   在当前步骤中点击 / 拖动（见窗口顶部提示）
  Enter      确认当前步骤，进入下一步（最后一步确认后自动保存并打开下一张）
  Backspace  回到上一步
  z          撤销最后一个标签框（第 ③ 步，反面）
  方向键     微调平皿圆心（第 ① 步），+ / - 微调平皿半径
  f          菌落铺满平皿（第 ③/④ 步：菌落半径 = 平皿半径）
  m          标记/取消"多菌落"
  v          切换 正面/反面（仅在文件名无法判断时使用）
  c          污染或混合生长 → 剔除并保存，跳到下一张
  b          图像不合格（模糊/反光/无白纸）→ 剔除并保存，跳到下一张
  n          跳过（不保存），下一张
  q / Esc    退出（已保存的不会丢失）
"""

import argparse
import datetime as dt
import getpass
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python，请先按文件头部说明安装")
import tkinter as tk
from PIL import Image, ImageOps, ImageTk

# ---------------------------------------------------------------- 输出路径约定（与 Audit/output 编号文件夹一致）
#   output/01_trainsets/      colony_annotations/（01 标注）  colony_preprocessed/（01 预处理）
#   output/02_training_runs/  colony/<时间>/（02 训练）  colony/feature_cache/
#   output/04_audit_review/   colony_inventory/（00 盘点）
#   output/06_logs/           每次运行的终端日志
AUDIT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = AUDIT_DIR / "output"


def out_path(sub, name=None, make=True):
    p = OUTPUT_DIR / sub / name if name else OUTPUT_DIR / sub
    if make:
        p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_in(sub, value):
    """命令行写短名字时到 output/<sub>/ 下找；写绝对路径或当前目录下存在的路径则原样使用"""
    p = Path(value).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    return (OUTPUT_DIR / sub / value).resolve()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, x):
        for s in self.streams:
            s.write(x)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def start_log(name):
    import datetime as _dt
    f = open(out_path("06_logs") / f"{name}_{_dt.datetime.now():%Y%m%d_%H%M%S}.log", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print("# " + " ".join(sys.argv))
    return f


def migrate_annotations(root: Path, ann_dir: Path):
    """旧版把标注放在 <照片根目录>/annotations；新位置为空时自动复制过来（不删除旧的）"""
    import shutil
    old = root / "annotations"
    if old.is_dir() and old.resolve() != ann_dir.resolve() and not any(ann_dir.rglob("*.json")):
        n = 0
        for f in old.rglob("*.json"):
            dst = ann_dir / f.relative_to(old)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
        if n:
            print(f"已把旧位置的 {n} 个标注复制到 {ann_dir}（旧文件保留在 {old}，确认无误后可自行删除）")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

TOOL_VERSION = "01_colony_annotate v1.0 (2026-10-19)"
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
STD_RE = re.compile(r"^(PUPH-[A-Z]{3}-\d{3})-B-([OR])-(\d+)$", re.IGNORECASE)

STEPS = {
    "O": ["plate", "center", "edge"],
    "R": ["plate", "center", "masks", "edge"],
}
PROMPTS = {
    "plate":  "① 确认平皿边界：绿圈为自动检测结果。拖动圆心可移动，拖动圆周可缩放，在空白处拖动可重画（圆心→边缘）。确认后按 Enter",
    "center": "② 点击菌落中心（可重复点击修正）。确认后按 Enter",
    "masks":  "③ 拖动框选反面的标签 / 文字区域，可框多个，z 撤销最后一个；没有就直接按 Enter",
    "edge":   "{n} 点击菌落边缘上的一点（确定菌落半径；菌落铺满平皿按 f）。确认后按 Enter 保存",
}


# ---------------------------------------------------------------- 工具函数
def sha1_of(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_name(path: Path):
    m = STD_RE.match(path.stem)
    if m:
        return m.group(1).upper(), m.group(2).upper(), True
    low = path.stem.lower()
    view = "R" if any(k in low for k in ["反", "back", "reverse", "-r-", "_r_"]) else "O"
    return path.stem, view, False


def detect_plate(rgb: np.ndarray):
    """霍夫圆检测平皿。输入 RGB（原尺寸），返回原图坐标 (cx, cy, r) 或 None。"""
    h, w = rgb.shape[:2]
    s = min(1.0, 1024 / max(h, w))
    small = cv2.resize(rgb, (int(w * s), int(h * s))) if s < 1 else rgb
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), 5)
    m = min(gray.shape)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m,
                               param1=100, param2=40,
                               minRadius=int(m * 0.25), maxRadius=int(m * 0.55))
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx / s), float(cy / s), float(r / s)


# ---------------------------------------------------------------- 标注界面
class Annotator:
    MAX_W, MAX_H = 1100, 760

    def __init__(self, files, root: Path, ann_dir: Path, annotator: str, redo: bool):
        self.files, self.root, self.ann_dir = files, root, ann_dir
        self.annotator, self.redo = annotator, redo
        self.i = -1
        self.n_saved = self.n_excluded = self.n_skipped = 0

        self.tk = tk.Tk()
        self.tk.title("菌落标注 01")
        self.prompt = tk.Label(self.tk, text="", font=("PingFang SC", 15, "bold"),
                               wraplength=self.MAX_W, justify="left", anchor="w", fg="#1f3b57")
        self.prompt.pack(fill="x", padx=10, pady=(8, 2))
        self.canvas = tk.Canvas(self.tk, width=self.MAX_W, height=self.MAX_H, bg="#333", highlightthickness=0)
        self.canvas.pack(padx=10, pady=4)
        self.status = tk.Label(self.tk, text="", font=("PingFang SC", 12), anchor="w", fg="#444")
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        for key, fn in [("<Return>", self.next_step), ("<KP_Enter>", self.next_step),
                        ("<BackSpace>", self.prev_step), ("<Escape>", self.quit),
                        ("q", self.quit), ("n", self.skip), ("c", lambda e: self.exclude("contaminated")),
                        ("b", lambda e: self.exclude("bad_image")), ("m", self.toggle_multi),
                        ("f", self.set_confluent), ("z", self.undo_mask), ("v", self.toggle_view),
                        ("<Left>", lambda e: self.nudge(-1, 0)), ("<Right>", lambda e: self.nudge(1, 0)),
                        ("<Up>", lambda e: self.nudge(0, -1)), ("<Down>", lambda e: self.nudge(0, 1)),
                        ("<plus>", lambda e: self.grow(1)), ("<equal>", lambda e: self.grow(1)),
                        ("<minus>", lambda e: self.grow(-1))]:
            self.tk.bind(key, fn)
        self.load_next()

    # ---------- 载入 ----------
    def load_next(self):
        self.i += 1
        if self.i >= len(self.files):
            self.finish()
            return
        p = self.files[self.i]
        self.path = p
        self.strain, self.view, self.std = parse_name(p)
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
        self.img = im
        self.W, self.H = im.size
        self.scale = min(self.MAX_W / self.W, self.MAX_H / self.H)
        disp = im.resize((int(self.W * self.scale), int(self.H * self.scale)), Image.BILINEAR)
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width, height=disp.height)

        self.plate = self.center = None
        self.colony_r = None
        self.masks = []
        self.flags = {"multi_colony": False, "confluent": False}
        self.plate_auto = False
        self.plate_adjusted = False

        prev = self.ann_path(p)
        if self.redo and prev.exists():
            try:
                d = json.loads(prev.read_text(encoding="utf-8"))
                if d.get("plate"):
                    pl = d["plate"]; self.plate = [pl["cx"], pl["cy"], pl["r"]]
                if d.get("colony"):
                    co = d["colony"]; self.center = [co["cx"], co["cy"]]; self.colony_r = co.get("r")
                self.masks = [list(m) for m in d.get("masks", [])]
                self.flags.update({k: v for k, v in d.get("flags", {}).items() if k in self.flags})
                self.view = d.get("view", self.view)
            except Exception:
                pass
        if self.plate is None:
            det = detect_plate(np.asarray(im))
            if det:
                self.plate, self.plate_auto = list(det), True

        self.step_i = 0
        self.drag = None
        self.redraw()

    def ann_path(self, p: Path) -> Path:
        rel = p.relative_to(self.root)
        return self.ann_dir / rel.parent / (p.stem + ".json")

    # ---------- 坐标 ----------
    def to_img(self, x, y):
        return x / self.scale, y / self.scale

    def to_disp(self, x, y):
        return x * self.scale, y * self.scale

    @property
    def step(self):
        return STEPS[self.view][self.step_i]

    # ---------- 鼠标 ----------
    def on_press(self, e):
        x, y = self.to_img(e.x, e.y)
        s = self.step
        if s == "plate":
            tol = 15 / self.scale
            if self.plate:
                cx, cy, r = self.plate
                d = math.hypot(x - cx, y - cy)
                if d < max(tol, 0.08 * r):
                    self.drag = ("move", x - cx, y - cy)
                    return
                if abs(d - r) < tol:
                    self.drag = ("resize",)
                    return
            self.drag = ("new", x, y)
            self.plate = [x, y, 1.0]
        elif s == "center":
            self.center = [x, y]
        elif s == "masks":
            self.drag = ("rect", x, y)
            self.masks.append([x, y, x, y])
        elif s == "edge":
            self.set_edge(x, y)
        self.redraw()

    def on_drag(self, e):
        x, y = self.to_img(e.x, e.y)
        s = self.step
        if s == "plate" and self.drag:
            kind = self.drag[0]
            if kind == "move":
                self.plate[0], self.plate[1] = x - self.drag[1], y - self.drag[2]
            elif kind == "resize":
                self.plate[2] = math.hypot(x - self.plate[0], y - self.plate[1])
            elif kind == "new":
                self.plate[2] = math.hypot(x - self.drag[1], y - self.drag[2])
            self.plate_adjusted = True
        elif s == "center":
            self.center = [x, y]
        elif s == "masks" and self.drag:
            x0, y0 = self.drag[1], self.drag[2]
            self.masks[-1] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        elif s == "edge":
            self.set_edge(x, y)
        self.redraw()

    def on_release(self, e):
        if self.step == "plate" and self.drag and self.drag[0] == "new" and self.plate[2] < 20 / self.scale:
            self.plate = None  # 误点，不算画圆
        if self.step == "masks" and self.masks:
            x0, y0, x1, y1 = self.masks[-1]
            if (x1 - x0) * (y1 - y0) < (10 / self.scale) ** 2:
                self.masks.pop()
        self.drag = None
        self.redraw()

    def set_edge(self, x, y):
        if self.center:
            self.colony_r = math.hypot(x - self.center[0], y - self.center[1])
            self.flags["confluent"] = False

    # ---------- 键盘 ----------
    def next_step(self, e=None):
        s = self.step
        if s == "plate" and not self.plate:
            return self.warn("还没有平皿圆：在图上从平皿中心拖到边缘画一个圆")
        if s == "center" and not self.center:
            return self.warn("请先点击菌落中心")
        if s == "center" and self.plate and math.hypot(self.center[0] - self.plate[0],
                                                      self.center[1] - self.plate[1]) > self.plate[2]:
            return self.warn("菌落中心在平皿圆外，请重新点击")
        if s == "edge" and not self.colony_r:
            return self.warn("请点击菌落边缘上的一点，或按 f 表示菌落铺满平皿")
        if self.step_i < len(STEPS[self.view]) - 1:
            self.step_i += 1
            self.redraw()
        else:
            self.save()

    def prev_step(self, e=None):
        if self.step_i > 0:
            self.step_i -= 1
            self.redraw()

    def nudge(self, dx, dy):
        if self.step == "plate" and self.plate:
            k = max(1.0, 0.002 * self.plate[2])
            self.plate[0] += dx * k; self.plate[1] += dy * k
            self.plate_adjusted = True
            self.redraw()

    def grow(self, sgn):
        if self.step == "plate" and self.plate:
            self.plate[2] *= 1 + 0.005 * sgn
            self.plate_adjusted = True
            self.redraw()

    def set_confluent(self, e=None):
        if self.step == "edge" and self.plate:
            self.flags["confluent"] = True
            self.colony_r = self.plate[2]
            self.redraw()

    def toggle_multi(self, e=None):
        self.flags["multi_colony"] = not self.flags["multi_colony"]
        self.redraw()

    def toggle_view(self, e=None):
        if self.std:
            return self.warn("文件名已标明正/反面，不能切换")
        self.view = "R" if self.view == "O" else "O"
        self.step_i = min(self.step_i, len(STEPS[self.view]) - 1)
        self.redraw()

    def undo_mask(self, e=None):
        if self.step == "masks" and self.masks:
            self.masks.pop()
            self.redraw()

    def warn(self, msg):
        self.status.config(text="⚠ " + msg, fg="#b3261e")

    # ---------- 保存 ----------
    def record(self, excluded=None):
        p = self.path
        d = {
            "file": str(p.relative_to(self.root)),
            "strain_id": self.strain,
            "standard_name": self.std,
            "view": self.view,
            "image_size": [self.W, self.H],
            "sha1": sha1_of(p),
            "plate": None if not self.plate else {
                "cx": round(self.plate[0], 1), "cy": round(self.plate[1], 1), "r": round(self.plate[2], 1),
                "auto_detected": self.plate_auto, "adjusted": self.plate_adjusted},
            "colony": None if not self.center else {
                "cx": round(self.center[0], 1), "cy": round(self.center[1], 1),
                "r": None if self.colony_r is None else round(self.colony_r, 1)},
            "colony_diam_mm": None,
            "masks": [[round(v, 1) for v in m] for m in self.masks],
            "flags": dict(self.flags),
            "excluded": excluded,
            "annotator": self.annotator,
            "annotated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "tool_version": TOOL_VERSION,
        }
        if self.plate and self.colony_r:
            d["colony_diam_mm"] = round(90.0 * self.colony_r / self.plate[2], 1)  # 90 mm 平皿
        return d

    def write(self, d):
        out = self.ann_path(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)

    def save(self):
        self.write(self.record())
        self.n_saved += 1
        self.load_next()

    def exclude(self, reason):
        self.write(self.record(excluded=reason))
        self.n_excluded += 1
        self.load_next()

    def skip(self, e=None):
        self.n_skipped += 1
        self.load_next()

    def quit(self, e=None):
        self.summary()
        self.tk.destroy()

    def finish(self):
        self.summary()
        self.canvas.delete("all")
        self.prompt.config(text="全部完成，可以关闭窗口")
        self.tk.after(1500, self.tk.destroy)

    def summary(self):
        print(f"本次：保存 {self.n_saved}，剔除 {self.n_excluded}，跳过 {self.n_skipped}；"
              f"标注目录 {self.ann_dir}")

    # ---------- 绘制 ----------
    def redraw(self):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        sc = self.scale

        if self.plate:
            cx, cy, r = self.plate
            X, Y = self.to_disp(cx, cy)
            R = r * sc
            for k in (1.05, 1.25):  # 白平衡参照环
                c.create_oval(X - k * R, Y - k * R, X + k * R, Y + k * R, outline="#4fc3f7", dash=(4, 4))
            c.create_oval(X - R, Y - R, X + R, Y + R, outline="#00e676", width=3)
            c.create_oval(X - 4, Y - 4, X + 4, Y + 4, fill="#00e676", outline="")

        for m in self.masks:
            x0, y0 = self.to_disp(m[0], m[1]); x1, y1 = self.to_disp(m[2], m[3])
            c.create_rectangle(x0, y0, x1, y1, outline="#ff5252", width=2, fill="#ff5252", stipple="gray25")

        if self.center:
            X, Y = self.to_disp(*self.center)
            c.create_line(X - 10, Y, X + 10, Y, fill="#ff1744", width=2)
            c.create_line(X, Y - 10, X, Y + 10, fill="#ff1744", width=2)
            if self.colony_r:
                R = self.colony_r * sc
                c.create_oval(X - R, Y - R, X + R, Y + R, outline="#ffab00", width=2)

        s = self.step
        n_steps = len(STEPS[self.view])
        text = PROMPTS[s].format(n="③" if n_steps == 3 else "④")
        self.prompt.config(text=text)

        view_txt = "正面" if self.view == "O" else "反面"
        name_tip = "" if self.std else "  ［非标准文件名：v 切换正/反面］"
        fl = []
        if self.flags["multi_colony"]: fl.append("多菌落")
        if self.flags["confluent"]: fl.append("铺满平皿")
        if s == "plate":
            fl.append("自动检测" if self.plate_auto and not self.plate_adjusted else
                      ("已手动调整" if self.plate else "未检测到平皿，请手动画圆"))
        diam = ""
        if self.plate and self.colony_r:
            diam = f"  菌落直径≈{90.0 * self.colony_r / self.plate[2]:.1f} mm"
        self.status.config(
            fg="#444",
            text=(f"[{self.i + 1}/{len(self.files)}]  {self.path.name}  |  株号 {self.strain}  |  {view_txt}"
                  f"  |  步骤 {self.step_i + 1}/{n_steps}{diam}"
                  f"{'  |  ' + '、'.join(fl) if fl else ''}{name_tip}"
                  f"      Enter 确认  Backspace 上一步  c 污染  b 不合格  n 跳过  q 退出"))


# ---------------------------------------------------------------- 主程序
def main():
    ap = argparse.ArgumentParser(description="菌落照片几何标注")
    ap.add_argument("--root", required=True, help="补拍数据根目录（含 FUM/FLA/... 子文件夹）")
    ap.add_argument("--ann_dir", default="colony_annotations", help="短名字 → output/01_trainsets/<名字>")
    ap.add_argument("--annotator", default=getpass.getuser())
    ap.add_argument("--redo", action="store_true", help="重新标注已标过的图")
    ap.add_argument("--only", default=None, help="只处理文件名包含该字符串的图（如某个株号）")
    ap.add_argument("--include_nonstandard", action="store_true", help="也处理不符合 PUPH-XXX-000-B-O/R-n 命名的图")
    args = ap.parse_args()
    start_log("01_colony_annotate")

    root = Path(args.root).expanduser().resolve()
    ann_dir = resolve_in("01_trainsets", args.ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    migrate_annotations(root, ann_dir)

    files, nonstd = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT or ann_dir in p.parents \
                or {"annotations", "preprocessed"} & set(p.relative_to(root).parts[:-1]):
            continue
        if args.only and args.only.lower() not in p.name.lower():
            continue
        is_std = bool(STD_RE.match(p.stem))
        is_colony = is_std or ("-C-" not in p.stem.upper())
        if not is_colony:
            continue  # 镜检图不在这里标注
        if not is_std and not args.include_nonstandard:
            nonstd.append(p)
            continue
        rel = p.relative_to(root)
        if not args.redo and (ann_dir / rel.parent / (p.stem + ".json")).exists():
            continue
        files.append(p)

    if nonstd:
        print(f"跳过 {len(nonstd)} 张不符合命名规则的图（如需标注加 --include_nonstandard），例如：")
        for p in nonstd[:5]:
            print("   ", p.relative_to(root))
    if not files:
        print("没有需要标注的菌落图（已全部标注，或路径/命名不对）。")
        return
    print(f"待标注 {len(files)} 张，标注保存到 {ann_dir}")
    app = Annotator(files, root, ann_dir, args.annotator, args.redo)
    app.tk.mainloop()


if __name__ == "__main__":
    main()
