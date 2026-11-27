#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_colony_preprocess.py — 菌落照片预处理（不修改原图）

  色彩空间统一（ICC → sRGB） → 人工确认白平衡参照 → 颜色校正 → 尺度统一（平皿直径 = 1024 px）
  → 自动截图（整皿 / 中心 / 边缘 ×6） → 质控拼图 + 清单

两个子命令，先 pick 再 build：

  1) pick  交互式确认白平衡参照区，结果写入 annotations/*.json 的 "wb" 字段
       python 02_colony_preprocess.py pick  --root "/.../补拍_2026"
       python 02_colony_preprocess.py pick  --root ... --only PUPH-TER-003 --redo

     操作（输入法切到英文，先点一下窗口）
       鼠标点击 / 拖框   选择参照区（点击 = 小方块；可选多处）
       a                在 白纸模式 / 培养基模式 之间切换（会清空已选区域）
       t                进入 / 退出「文字遮罩」：拖框盖住菌种名、手写字、标签、手指等（正面也可用）
       x                无可用参照（该图不进入主分析）
       r                恢复程序建议（白纸环）
       z                撤销最后一个框（遮罩模式下撤销最后一个遮罩）
       Enter            保存并下一张
       p                上一张        n  跳过        q / Esc  退出

  2) build  批量预处理（无界面）
       python 02_colony_preprocess.py build --root "/.../补拍_2026"
     输出到 <root>/preprocessed/ ：
       plates/<组>/<文件名>.png          颜色校正 + 尺度统一后的整幅图（1280×1280，平皿居中）
       crops/<组>/<文件名>__whole.png   整皿（平皿外置灰）
                         __center.png  菌落中心块
                         __edge0..5.png 菌落边缘块
       qc_sheets/<组>.jpg               每株一行：原图(参照区标出) | 校正后 | 截图
       qc_sheets/_center_by_group.jpg   各组中心块并排，用于肉眼比较颜色
       duplicates_check.csv              疑似同一平皿的照片对（特征点匹配）
       manifest.csv                      每张图一行：白平衡方式、增益、菌落直径、质控标记、截图路径
       preprocess_config.json            参数、版本、培养基标准色

安装（清华源）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python pillow pillow-heif pandas
"""

import argparse
import datetime as dt
import getpass
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("缺少 opencv-python，请先按文件头部说明安装")
from PIL import Image, ImageCms, ImageDraw, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

VERSION = "02_colony_preprocess v1.1 (2026-10-20)"
GROUPS = {"FUM": "Fumigati", "FLA": "Flavi", "NIG": "Nigri", "TER": "Terrei", "NID": "Nidulantes"}

PLATE_R = 512          # 归一化后平皿半径（px）
CANVAS = 1280          # 归一化整幅图边长，平皿圆心位于 (640, 640)
CROP = 384             # 截图输出边长
WHITE_SRGB = 0.90      # 白纸校正目标亮度（sRGB 0–1，约 230/255）
GRAY = 0.5             # 平皿外 / 标签遮罩处的中性灰
MIN_REF_PX = 300       # 参照区最少有效像素（工作分辨率下）
EDGE_MAX_BAD = 0.20    # 边缘块中平皿外 + 遮罩像素比例上限
RING = (1.05, 1.25)    # 白纸环：平皿半径的倍数
AGAR_AUTO = (1.15, 0.85)  # 自动培养基环：> 1.15×菌落半径 且 < 0.85×平皿半径
SAT_WARN = 0.30         # 白纸参照饱和度上限（暖光/冷光下白纸本身会偏色，阈值不宜过严）
LOW_RES_PLATE_R = 350  # 原图平皿半径 < 350 px（约 7.8 px/mm）标记为低分辨率
DUP_INLIERS = 300      # 两张平皿图特征点匹配内点数 ≥ 此值 → 疑似同一平皿（实测同一皿 500–2000，不同皿 < 200）
HARD_FLAGS = {"wb_none", "ref_too_small", "agar_target_unavailable", "no_geometry"}


# =============================================================== 颜色工具
def srgb_to_lin(x):
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, dtype=np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


WHITE_LIN = float(srgb_to_lin(WHITE_SRGB))


def load_srgb(path: Path):
    """读图 → 按 EXIF 摆正 → 按内嵌 ICC 转为 sRGB。返回 (float32 HxWx3, 0–1), 元信息"""
    with Image.open(path) as im:
        fmt = im.format or path.suffix.upper().lstrip(".")
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
    icc_name = "none (treated as sRGB)"
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            icc_name = ImageCms.getProfileDescription(src).strip() or "embedded"
            if "srgb" not in icc_name.lower():
                im = ImageCms.profileToProfile(im, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception as e:  # ICC 损坏时按 sRGB 处理并记录
            icc_name = f"icc_error:{type(e).__name__}"
    return np.asarray(im, dtype=np.float32) / 255.0, {"format": fmt, "icc": icc_name}


def to_working(img, ann):
    """缩放到平皿半径 = PLATE_R；返回工作分辨率图像和几何（工作坐标）"""
    pl = ann["plate"]
    s = PLATE_R / pl["r"]
    h, w = img.shape[:2]
    work = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    co = ann.get("colony") or {}
    geo = {
        "s": s,
        "plate": (pl["cx"] * s, pl["cy"] * s, PLATE_R),
        "colony": None if co.get("cx") is None else (co["cx"] * s, co["cy"] * s, (co.get("r") or 0) * s),
        "labels": [[v * s for v in m] for m in ann.get("masks", []) + ann.get("text_masks", [])],
        "confluent": bool(ann.get("flags", {}).get("confluent")),
    }
    return work, geo


def label_mask(shape, geo):
    m = np.zeros(shape[:2], bool)
    for x0, y0, x1, y1 in geo["labels"]:
        m[max(0, int(y0)):int(math.ceil(y1)), max(0, int(x0)):int(math.ceil(x1))] = True
    return m


def ref_mask(shape, geo, wb):
    """按白平衡设置得到参照区布尔掩膜（工作坐标）"""
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    pcx, pcy, pr = geo["plate"]
    dp = np.hypot(xx - pcx, yy - pcy)
    m = np.zeros((h, w), bool)
    mode, rois, s = wb.get("mode"), wb.get("rois", []), geo["s"]
    for r in rois:
        if r["type"] == "ring":
            m |= (dp > r["r_in"] * pr) & (dp < r["r_out"] * pr)
        elif r["type"] == "box":
            x0, y0, x1, y1 = [v * s for v in r["box"]]
            m[max(0, int(y0)):int(math.ceil(y1)), max(0, int(x0)):int(math.ceil(x1))] = True
    if mode == "paper":
        m &= dp > 1.02 * pr
    elif mode == "agar":
        if not rois:
            m = dp < AGAR_AUTO[1] * pr
            if geo["colony"] and not geo["confluent"]:
                ccx, ccy, cr = geo["colony"]
                m &= np.hypot(xx - ccx, yy - ccy) > AGAR_AUTO[0] * cr
            elif geo["confluent"]:
                m[:] = False
        m &= dp < 0.97 * pr
    else:
        m[:] = False
    m &= ~label_mask(shape, geo)
    return m


def ref_stats(work, m):
    px = work[m]
    clip_frac = 0.0
    if len(px):
        clipped = px.max(1) >= 0.99
        clip_frac = float(clipped.mean())
        keep = ~clipped & (px.mean(1) > 0.12)  # 去掉完全过曝和过暗像素
        if keep.sum() >= MIN_REF_PX or keep.sum() >= 0.5 * len(px):
            px = px[keep]
        # 否则（白纸大面积过曝）保留全部像素，并通过 clip_frac 报告
    n = int(len(px))
    if n == 0:
        return {"n": 0, "clip": round(clip_frac, 3)}
    lin = srgb_to_lin(px)
    med_lin = np.median(lin, 0)
    med_srgb = lin_to_srgb(med_lin)
    mx, mn = float(med_srgb.max()), float(med_srgb.min())
    sat = 0.0 if mx <= 0 else (mx - mn) / mx
    lum = px.mean(1)
    q1, q2, q3 = np.percentile(lum, [25, 50, 75])
    return {"n": n, "med_lin": med_lin.tolist(), "med_srgb": med_srgb.tolist(), "clip": round(clip_frac, 3),
            "sat": round(sat, 3), "lum": round(float(q2), 3), "uneven": round(float((q3 - q1) / max(q2, 1e-6)), 3)}


def suggest_wb(work, geo):
    ring = {"type": "ring", "r_in": RING[0], "r_out": RING[1]}
    st = ref_stats(work, ref_mask(work.shape, geo, {"mode": "paper", "rois": [ring]}))
    if st["n"] >= MIN_REF_PX and st["sat"] < SAT_WARN and st["lum"] > 0.55 and st["uneven"] < 0.25:
        return {"mode": "paper", "rois": [ring], "source": "auto_ring"}
    return {"mode": "paper", "rois": [], "source": "manual"}


def gains_for(wb_mode, st, agar_target_lin):
    ref = np.maximum(np.asarray(st["med_lin"], dtype=np.float32), 1e-4)
    if wb_mode == "paper":
        return WHITE_LIN / ref
    return np.asarray(agar_target_lin, dtype=np.float32) / ref


def apply_gains(work, g):
    return lin_to_srgb(srgb_to_lin(work) * np.asarray(g, dtype=np.float32)[None, None, :])


# =============================================================== 文件
def ann_files(root: Path, views, only=None):
    out = []
    for jp in sorted((root / "annotations").rglob("*.json")):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            print(f"  无法读取 {jp}，跳过")
            continue
        if d.get("view") not in views or d.get("excluded"):
            continue
        if only and only.lower() not in jp.stem.lower():
            continue
        out.append((jp, d))
    return out


def write_json(jp: Path, d):
    tmp = jp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(jp)


# =============================================================== pick（交互）
def cmd_pick(args):
    import tkinter as tk
    from PIL import ImageTk

    root = Path(args.root).expanduser().resolve()
    items = [(jp, d) for jp, d in ann_files(root, set(args.views.split(",")), args.only)
             if args.redo or "wb" not in d]
    items = [(jp, d) for jp, d in items if d.get("plate")]
    if not items:
        print("没有需要确认白平衡的图（已全部确认，或没有可用标注）。加 --redo 可重新确认。")
        return
    print(f"待确认 {len(items)} 张")

    MAXW, MAXH, PV = 860, 660, 300

    class Picker:
        def __init__(self):
            self.tk = tk.Tk()
            self.tk.title("02 白平衡参照确认")
            self.prompt = tk.Label(self.tk, font=("PingFang SC", 15, "bold"), fg="#1f3b57",
                                   anchor="w", justify="left", wraplength=MAXW + PV)
            self.prompt.pack(fill="x", padx=10, pady=(8, 2))
            row = tk.Frame(self.tk)
            row.pack(padx=10, pady=4)
            self.cv = tk.Canvas(row, width=MAXW, height=MAXH, bg="#333", highlightthickness=0)
            self.cv.pack(side="left")
            side = tk.Frame(row)
            side.pack(side="left", padx=(10, 0), anchor="n")
            tk.Label(side, text="校正后预览", font=("PingFang SC", 12)).pack(anchor="w")
            self.pv = tk.Canvas(side, width=PV, height=PV, bg="#666", highlightthickness=0)
            self.pv.pack()
            self.info = tk.Label(side, font=("Menlo", 11), justify="left", anchor="w", wraplength=PV)
            self.info.pack(anchor="w", pady=6)
            self.status = tk.Label(self.tk, font=("PingFang SC", 12), anchor="w", fg="#444")
            self.status.pack(fill="x", padx=10, pady=(0, 8))
            self.cv.bind("<ButtonPress-1>", self.press)
            self.cv.bind("<B1-Motion>", self.drag)
            self.cv.bind("<ButtonRelease-1>", self.release)
            for k, f in [("<Return>", self.save), ("<KP_Enter>", self.save), ("a", self.toggle),
                         ("t", self.mask_toggle), ("x", self.none), ("r", self.reset), ("z", self.undo), ("n", self.skip),
                         ("p", self.prev), ("q", self.quit), ("<Escape>", self.quit)]:
                self.tk.bind(k, f)
            self.i = 0
            self.n_saved = 0
            self.load()

        # ---------- 载入 ----------
        def load(self):
            if self.i >= len(items):
                print(f"完成：本次保存 {self.n_saved} 张")
                self.tk.destroy()
                return
            self.jp, self.d = items[self.i]
            self.d = json.loads(self.jp.read_text(encoding="utf-8"))  # 重新读，防止 p 返回后用旧内容
            img, _ = load_srgb(root / self.d["file"])
            self.work, self.geo = to_working(img, self.d)
            wh, ww = self.work.shape[:2]
            self.ds = min(MAXW / ww, MAXH / wh)
            self.disp = cv2.resize(self.work, (int(ww * self.ds), int(wh * self.ds)), interpolation=cv2.INTER_AREA)
            self.photo = ImageTk.PhotoImage(Image.fromarray((self.disp * 255).astype(np.uint8)))
            self.cv.config(width=self.disp.shape[1], height=self.disp.shape[0])
            self.wb = json.loads(json.dumps(self.d["wb"])) if "wb" in self.d else suggest_wb(self.work, self.geo)
            self.text_masks = [list(m) for m in self.d.get("text_masks", [])]
            self.mask_mode = False
            self.cur = None
            self.redraw()

        # ---------- 鼠标（工作坐标 = 显示坐标 / ds；保存为原图坐标 = 工作坐标 / s） ----------
        def press(self, e):
            if self.mask_mode:
                self.cur = [e.x, e.y, e.x, e.y]
                return
            if self.wb["mode"] == "none":
                return
            if any(r["type"] == "ring" for r in self.wb["rois"]):
                self.wb["rois"] = []  # 手动选区时替换程序建议
            self.cur = [e.x, e.y, e.x, e.y]

        def drag(self, e):
            if self.cur:
                self.cur[2], self.cur[3] = e.x, e.y
                self.redraw()

        def release(self, e):
            if not self.cur:
                return
            x0, y0, x1, y1 = self.cur
            if abs(x1 - x0) < 6 and abs(y1 - y0) < 6:  # 点击 → 40 px 小方块
                x0, y0, x1, y1 = x0 - 20, y0 - 20, x0 + 20, y0 + 20
            k = 1 / (self.ds * self.geo["s"])
            box = [min(x0, x1) * k, min(y0, y1) * k, max(x0, x1) * k, max(y0, y1) * k]
            if self.mask_mode:
                self.text_masks.append([round(v, 1) for v in box])
                self.sync_masks()
                self.cur = None
                self.redraw()
                return
            self.wb["rois"].append({"type": "box", "box": [round(v, 1) for v in box]})
            self.wb["source"] = "manual"
            self.cur = None
            self.redraw()

        # ---------- 键盘 ----------
        def toggle(self, e=None):
            self.wb = {"mode": "agar" if self.wb["mode"] == "paper" else "paper", "rois": [], "source": "manual"}
            self.redraw()

        def none(self, e=None):
            self.wb = {"mode": "none", "rois": [], "source": "manual"}
            self.redraw()

        def reset(self, e=None):
            self.wb = suggest_wb(self.work, self.geo)
            self.redraw()

        def mask_toggle(self, e=None):
            self.mask_mode = not self.mask_mode
            self.redraw()

        def sync_masks(self):
            s_ = self.geo["s"]
            self.geo["labels"] = [[v * s_ for v in m] for m in self.d.get("masks", []) + self.text_masks]

        def undo(self, e=None):
            if self.mask_mode:
                if self.text_masks:
                    self.text_masks.pop()
                    self.sync_masks()
                    self.redraw()
                return
            if self.wb["rois"]:
                self.wb["rois"].pop()
                self.redraw()

        def save(self, e=None):
            st = self.stats
            if self.wb["mode"] == "paper" and st["n"] < MIN_REF_PX:
                return self.warn("白纸参照区太小或没有选：请在白纸上点击或拖框；没有白纸按 a 改用培养基")
            if self.wb["mode"] == "agar" and st["n"] < MIN_REF_PX:
                return self.warn("培养基参照区太小：请框选干净的空白培养基；都没有按 x")
            rec = dict(self.wb)
            rec.update({"ref_srgb": [round(v, 4) for v in st.get("med_srgb", [])], "ref_n_px": st["n"],
                        "ref_sat": st.get("sat"), "ref_uneven": st.get("uneven"),
                        "picked_by": args.annotator, "picked_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "tool_version": VERSION})
            self.d["wb"] = rec
            self.d["text_masks"] = self.text_masks
            write_json(self.jp, self.d)
            self.n_saved += 1
            self.i += 1
            self.load()

        def skip(self, e=None):
            self.i += 1
            self.load()

        def prev(self, e=None):
            if self.i > 0:
                self.i -= 1
                self.load()

        def quit(self, e=None):
            print(f"退出：本次保存 {self.n_saved} 张")
            self.tk.destroy()

        def warn(self, msg):
            self.status.config(text="⚠ " + msg, fg="#b3261e")

        # ---------- 绘制 ----------
        def redraw(self):
            cv, ds = self.cv, self.ds
            cv.delete("all")
            cv.create_image(0, 0, image=self.photo, anchor="nw")
            pcx, pcy, pr = [v * ds for v in self.geo["plate"]]
            cv.create_oval(pcx - pr, pcy - pr, pcx + pr, pcy + pr, outline="#00e676", width=2)
            if self.geo["colony"] and self.geo["colony"][2]:
                ccx, ccy, cr = [v * ds for v in self.geo["colony"]]
                cv.create_oval(ccx - cr, ccy - cr, ccx + cr, ccy + cr, outline="#ffab00", width=1, dash=(3, 3))
            for x0, y0, x1, y1 in self.geo["labels"]:
                cv.create_rectangle(x0 * ds, y0 * ds, x1 * ds, y1 * ds, outline="#ff5252", dash=(3, 3))

            mode = self.wb["mode"]
            col = {"paper": "#40c4ff", "agar": "#ff9100"}.get(mode, "#999")
            for r in self.wb["rois"]:
                if r["type"] == "ring":
                    for k in (r["r_in"], r["r_out"]):
                        cv.create_oval(pcx - k * pr, pcy - k * pr, pcx + k * pr, pcy + k * pr, outline=col, width=2)
                else:
                    k = ds * self.geo["s"]
                    x0, y0, x1, y1 = [v * k for v in r["box"]]
                    cv.create_rectangle(x0, y0, x1, y1, outline=col, width=2)
            if mode == "agar" and not self.wb["rois"] and self.geo["colony"]:
                ccx, ccy, cr = [v * ds for v in self.geo["colony"]]
                a, b = AGAR_AUTO[0] * cr, AGAR_AUTO[1] * pr
                cv.create_oval(ccx - a, ccy - a, ccx + a, ccy + a, outline=col, dash=(5, 3))
                cv.create_oval(pcx - b, pcy - b, pcx + b, pcy + b, outline=col, dash=(5, 3))
            if self.cur:
                cv.create_rectangle(*self.cur, outline=col, width=2, dash=(4, 2))

            # 统计 + 预览
            self.stats = ref_stats(self.work, ref_mask(self.work.shape, self.geo, self.wb))
            st = self.stats
            self.pv.delete("all")
            lines = []
            if mode == "none":
                lines.append("无参照：不做白平衡，不进入主分析")
            elif st["n"] == 0:
                lines.append("参照区：尚未选择")
            else:
                r, g, b = [int(round(v * 255)) for v in st["med_srgb"]]
                lines += [f"参照像素 {st['n']}", f"中位色 RGB {r},{g},{b}",
                          f"饱和度 {st['sat']:.2f}  亮度 {st['lum']:.2f}", f"不均匀度 {st['uneven']:.2f}"]
                warn = []
                if mode == "paper" and st["sat"] >= SAT_WARN:
                    warn.append("⚠ 不像白纸（偏色）")
                if mode == "paper" and st["lum"] < 0.45:
                    warn.append("⚠ 偏暗，可能有阴影")
                if st["uneven"] >= 0.25:
                    warn.append("⚠ 区域不均匀（反光/阴影）")
                if mode == "paper" and st.get("clip", 0) >= 0.5:
                    warn.append("⚠ 白纸大面积过曝，颜色校正可能不准")
                lines += warn
                if mode == "paper" and st["n"] >= MIN_REF_PX:
                    g_ = gains_for("paper", st, None)
                    lines.append("增益 R/G/B " + "/".join(f"{v:.2f}" for v in g_))
                    pcx_, pcy_, pr_ = self.geo["plate"]
                    x0, y0 = int(max(0, pcx_ - 1.1 * pr_)), int(max(0, pcy_ - 1.1 * pr_))
                    sub = self.work[y0:int(pcy_ + 1.1 * pr_), x0:int(pcx_ + 1.1 * pr_)]
                    if sub.size:
                        sub = apply_gains(cv2.resize(sub, (PV, PV), interpolation=cv2.INTER_AREA), g_)
                        self.pvphoto = ImageTk.PhotoImage(Image.fromarray((sub * 255).astype(np.uint8)))
                        self.pv.create_image(0, 0, image=self.pvphoto, anchor="nw")
                elif mode == "agar":
                    self.pv.create_text(PV / 2, PV / 2, fill="white", width=PV - 20,
                                        text="培养基模式：校正目标色由白纸组图像计算，build 时生成")
            self.info.config(text="\n".join(lines))

            prompts = {
                "paper": "白纸模式：确认蓝色区域只覆盖白纸（无阴影、无反光、不含平皿）。不对就在白纸上点击或拖框重选。确认按 Enter",
                "agar": "培养基模式：在干净、无反光的空白培养基上点击或拖框（不要碰菌落）；不框则用虚线环内的区域。确认按 Enter",
                "none": "无可用参照：按 Enter 保存（该图不进入主分析）；按 a 返回选择",
            }
            self.prompt.config(text=prompts[mode] if not self.mask_mode else
                               "文字遮罩模式：拖框盖住菌种名、手写字、标签、手指、贴纸等（红色虚线）。z 撤销，t 返回白平衡选择，Enter 保存")
            src = {"auto_ring": "程序建议（白纸环）", "manual": "手动"}.get(self.wb.get("source"), "")
            self.status.config(fg="#444", text=(
                f"[{self.i + 1}/{len(items)}]  {Path(self.d['file']).name}  |  {self.d['strain_id']}  |  "
                f"{'正面' if self.d['view'] == 'O' else '反面'}  |  {dict(paper='白纸', agar='培养基', none='无参照')[mode]} · {src}"
                f"{'  |  【遮罩模式】' if self.mask_mode else ''}  遮罩 {len(self.text_masks)} 个"
                f"      a 切换模式  t 文字遮罩  x 无参照  r 恢复建议  z 撤销  p 上一张  n 跳过  q 退出"))

    app = Picker()
    app.tk.mainloop()


# =============================================================== build（批量）
def crop_square(img, cx, cy, half, out=CROP, fill=GRAY):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                 value=(fill,) * 3 if img.ndim == 3 else fill)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    sub = img[y0:y1, x0:x1]
    interp = cv2.INTER_AREA if sub.shape[0] > out else cv2.INTER_CUBIC
    return cv2.resize(sub, (out, out), interpolation=interp)


def process_one(img_path, ann, agar_target_lin, out_root):
    """返回 manifest 行 + 供质控拼图用的小图"""
    group = GROUPS.get(ann["strain_id"].split("-")[1], "Unknown") if ann["strain_id"].count("-") >= 2 else "Unknown"
    row = {"strain_id": ann["strain_id"], "group": group, "view": ann["view"], "file": ann["file"]}
    flags = []
    if not ann.get("plate") or not ann.get("colony") or not ann["colony"].get("r"):
        row["flags"] = "no_geometry"
        row["main_analysis"] = False
        return row, None

    img, meta = load_srgb(img_path)
    row.update(meta)
    work, geo = to_working(img, ann)
    wb = ann.get("wb", {"mode": "none", "rois": []})
    row["wb_mode"], row["wb_source"] = wb.get("mode"), wb.get("source")
    st = ref_stats(work, ref_mask(work.shape, geo, wb)) if wb.get("mode") in ("paper", "agar") else {"n": 0}
    row["ref_n_px"] = st["n"]

    if wb.get("mode") == "none":
        flags.append("wb_none")
        g = np.ones(3, np.float32)
    elif st["n"] < MIN_REF_PX:
        flags.append("ref_too_small")
        g = np.ones(3, np.float32)
    elif wb["mode"] == "agar" and agar_target_lin is None:
        flags.append("agar_target_unavailable")
        g = np.ones(3, np.float32)
    else:
        g = gains_for(wb["mode"], st, agar_target_lin)
        row.update({"ref_r": round(st["med_srgb"][0], 4), "ref_g": round(st["med_srgb"][1], 4),
                    "ref_b": round(st["med_srgb"][2], 4), "ref_sat": st["sat"], "ref_uneven": st["uneven"],
                    "ref_clip": st.get("clip")})
        if wb["mode"] == "paper" and st["sat"] >= SAT_WARN:
            flags.append("ref_not_neutral")
        if st["uneven"] >= 0.25:
            flags.append("ref_uneven")
        if wb["mode"] == "paper" and st.get("clip", 0) >= 0.5:
            flags.append("ref_overexposed")
        if g.max() / g.min() > 2.0 or g.max() > 3.0 or g.min() < 0.4:
            flags.append("gain_extreme")
    row.update({"gain_r": round(float(g[0]), 3), "gain_g": round(float(g[1]), 3), "gain_b": round(float(g[2]), 3)})

    corr = apply_gains(work, g)

    # 平移到 1280 画布，平皿圆心居中
    pcx, pcy, _ = geo["plate"]
    M = np.float32([[1, 0, CANVAS / 2 - pcx], [0, 1, CANVAS / 2 - pcy]])
    canvas = cv2.warpAffine(corr, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(GRAY,) * 3)
    lab = cv2.warpAffine(label_mask(work.shape, geo).astype(np.uint8), M, (CANVAS, CANVAS),
                         flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    yy, xx = np.ogrid[:CANVAS, :CANVAS]
    outside = np.hypot(xx - CANVAS / 2, yy - CANVAS / 2) > PLATE_R
    bad = outside | lab
    plate_only = canvas.copy()
    plate_only[bad] = GRAY

    ccx, ccy, cr = geo["colony"]
    ccx, ccy = ccx + CANVAS / 2 - pcx, ccy + CANVAS / 2 - pcy
    confluent = geo["confluent"] or cr >= 0.9 * PLATE_R
    if confluent:
        cr = min(cr, PLATE_R)
    row.update({"colony_diam_mm": ann.get("colony_diam_mm"), "confluent": confluent,
                "multi_colony": bool(ann.get("flags", {}).get("multi_colony"))})
    # 过曝只在菌落区域内统计（白纸校正到 0.9 后本身不应计入）
    yy_, xx_ = np.ogrid[:work.shape[0], :work.shape[1]]
    cx0, cy0, cr0 = geo["colony"]
    colmask = (np.hypot(xx_ - cx0, yy_ - cy0) < cr0) & ~label_mask(work.shape, geo)
    clipped = float((corr[colmask].max(1) >= 0.995).mean()) if colmask.any() else 0.0
    row["colony_clip_frac"] = round(clipped, 3)
    if clipped > 0.05:
        flags.append("overexposed")
    row["plate_r_px"] = round(ann["plate"]["r"], 1)
    row["px_per_mm"] = round(2 * ann["plate"]["r"] / 90.0, 2)
    if ann["plate"]["r"] < LOW_RES_PLATE_R:
        flags.append("low_resolution")
    row["n_text_masks"] = len(ann.get("text_masks", []))

    stem = Path(ann["file"]).stem
    pdir, cdir = out_root / "plates" / group, out_root / "crops" / group
    pdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)

    def save(arr, p):
        Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)

    save(canvas, pdir / f"{stem}.png")
    crops = {"whole": crop_square(plate_only, CANVAS / 2, CANVAS / 2, PLATE_R)}
    crops["center"] = crop_square(plate_only, ccx, ccy, max(0.5 * cr, 48))
    n_edge = 0
    if not confluent:
        half = max(0.3 * cr, 48)
        for k in range(6):
            th = math.radians(60 * k)
            ex, ey = ccx + cr * math.cos(th), ccy + cr * math.sin(th)
            frac_bad = float(crop_square(bad.astype(np.float32), ex, ey, half, out=64, fill=1.0).mean())
            if frac_bad > EDGE_MAX_BAD:
                continue
            crops[f"edge{k}"] = crop_square(plate_only, ex, ey, half)
            n_edge += 1
    paths = []
    for name, arr in crops.items():
        p = cdir / f"{stem}__{name}.png"
        save(arr, p)
        paths.append(str(p.relative_to(out_root)))
    row.update({"n_edge": n_edge, "plate_png": str((pdir / f'{stem}.png').relative_to(out_root)),
                "crops": ";".join(paths)})
    row["flags"] = ";".join(flags)
    row["main_analysis"] = not (set(flags) & HARD_FLAGS)

    # 质控用小图：原图（标出参照区）
    thumb = work.copy()
    rm = ref_mask(work.shape, geo, wb) if wb.get("mode") in ("paper", "agar") else np.zeros(work.shape[:2], bool)
    thumb[rm] = thumb[rm] * 0.4 + np.array([0.0, 0.6, 1.0]) * 0.6
    x0 = int(max(0, pcx - 1.3 * PLATE_R)); y0 = int(max(0, pcy - 1.3 * PLATE_R))
    thumb = thumb[y0:int(pcy + 1.3 * PLATE_R), x0:int(pcx + 1.3 * PLATE_R)]
    qc = {"orig": thumb, "corr": canvas, "crops": crops, "row": row, "orb": orb_features(canvas, bad)}
    return row, qc


def orb_features(canvas, bad):
    """平皿内（去掉文字遮罩）的 ORB 特征，用于检查是否为同一平皿的重复照片"""
    g = cv2.cvtColor((np.clip(canvas, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (CANVAS // 2, CANVAS // 2), interpolation=cv2.INTER_AREA)
    m = cv2.resize((~bad).astype(np.uint8) * 255, (CANVAS // 2, CANVAS // 2), interpolation=cv2.INTER_NEAREST)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    kp, des = cv2.ORB_create(nfeatures=3000).detectAndCompute(g, m)
    return (np.float32([k.pt for k in kp]), des) if des is not None and len(kp) >= 20 else None


def match_inliers(fa, fb):
    if fa is None or fb is None:
        return 0
    (pa, da), (pb, db) = fa, fb
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return int(inl.sum()) if inl is not None else 0


def duplicate_check(qcs, out_root):
    import itertools
    import pandas as pd
    res = []
    for a, b in itertools.combinations(qcs, 2):
        n = match_inliers(a["orb"], b["orb"])
        if n >= 20:
            res.append({"file_a": a["row"]["file"], "strain_a": a["row"]["strain_id"],
                        "file_b": b["row"]["file"], "strain_b": b["row"]["strain_id"],
                        "inliers": n, "same_strain_id": a["row"]["strain_id"] == b["row"]["strain_id"],
                        "suspected_same_plate": n >= DUP_INLIERS})
    df = pd.DataFrame(res, columns=["file_a", "strain_a", "file_b", "strain_b", "inliers",
                                    "same_strain_id", "suspected_same_plate"]).sort_values("inliers", ascending=False)
    df.to_csv(out_root / "duplicates_check.csv", index=False, encoding="utf-8-sig")
    return df


def agar_color_after(img_path, ann):
    """白纸组图像：校正后自动培养基环的中位色（线性），用于确定培养基标准色"""
    img, _ = load_srgb(img_path)
    work, geo = to_working(img, ann)
    st = ref_stats(work, ref_mask(work.shape, geo, ann["wb"]))
    if st["n"] < MIN_REF_PX or ann["wb"].get("ref_sat", 0) and ann["wb"]["ref_sat"] >= SAT_WARN:
        return None
    corr = apply_gains(work, gains_for("paper", st, None))
    sa = ref_stats(corr, ref_mask(work.shape, geo, {"mode": "agar", "rois": []}))
    return None if sa["n"] < MIN_REF_PX else np.asarray(sa["med_lin"])


def to_pil(arr, size):
    a = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def qc_sheets(qcs, out_root):
    qdir = out_root / "qc_sheets"
    qdir.mkdir(parents=True, exist_ok=True)
    by_g = {}
    for q in qcs:
        by_g.setdefault(q["row"]["group"], []).append(q)
    T, C, LBL = 240, 120, 260
    for g, lst in by_g.items():
        lst.sort(key=lambda q: q["row"]["file"])
        W = LBL + 2 * T + 8 * C
        sheet = Image.new("RGB", (W, len(lst) * (T + 10) + 10), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for i, q in enumerate(lst):
            y = 10 + i * (T + 10)
            r = q["row"]
            txt = [r["strain_id"], Path(r["file"]).name[:30], f"WB: {r.get('wb_mode')} ({r.get('wb_source')})",
                   f"gain R/G/B {r.get('gain_r')}/{r.get('gain_g')}/{r.get('gain_b')}",
                   f"diam {r.get('colony_diam_mm')} mm  edges {r.get('n_edge')}",
                   f"ICC {str(r.get('icc'))[:24]}", f"flags: {r.get('flags') or '-'}",
                   "MAIN" if r.get("main_analysis") else "EXCLUDED"]
            for j, t in enumerate(txt):
                dr.text((8, y + 4 + j * 16), t, fill=(180, 0, 0) if (j >= 6 and r.get("flags")) else (0, 0, 0))
            o = q["orig"]
            s = T / max(o.shape[:2])
            sheet.paste(to_pil(o, (max(1, int(o.shape[1] * s)), max(1, int(o.shape[0] * s)))), (LBL, y))
            sheet.paste(to_pil(q["corr"], (T, T)), (LBL + T + 4, y))
            x = LBL + 2 * T + 12
            for k in ["whole", "center"] + [f"edge{i}" for i in range(6)]:
                if k in q["crops"]:
                    sheet.paste(to_pil(q["crops"][k], (C, C)), (x, y + (T - C) // 2))
                    dr.text((x, y + (T - C) // 2 + C + 2), k, fill=(90, 90, 90))
                x += C + 2
        sheet.save(qdir / f"{g}.jpg", quality=92)

    # 各组中心块并排
    order = [g for g in GROUPS.values() if g in by_g]
    ncol = max(len(by_g[g]) for g in order)
    C2, L2 = 128, 110
    sheet = Image.new("RGB", (L2 + ncol * (C2 + 4), len(order) * (C2 + 22) + 10), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for i, g in enumerate(order):
        y = 10 + i * (C2 + 22)
        dr.text((6, y + C2 // 2), g, fill=(0, 0, 0))
        for j, q in enumerate(sorted(by_g[g], key=lambda q: q["row"]["file"])):
            x = L2 + j * (C2 + 4)
            sheet.paste(to_pil(q["crops"]["center"], (C2, C2)), (x, y))
            col = (0, 0, 0) if q["row"].get("main_analysis") else (200, 0, 0)
            dr.text((x, y + C2 + 2), q["row"]["strain_id"].split("-", 1)[-1] + ("" if q["row"].get("wb_mode") == "paper" else f" [{q['row'].get('wb_mode')}]"), fill=col)
    sheet.save(qdir / "_center_by_group.jpg", quality=92)


def cmd_build(args):
    import pandas as pd

    root = Path(args.root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve() if args.out else root / "preprocessed"
    out_root.mkdir(parents=True, exist_ok=True)
    items = ann_files(root, set(args.views.split(",")))
    if not items:
        sys.exit("没有找到标注（annotations/*.json）")
    missing = [jp.stem for jp, d in items if "wb" not in d]
    if missing and not args.allow_missing_wb:
        print(f"⚠ {len(missing)} 张尚未确认白平衡（先运行 pick），例如：{missing[:5]}")
        print("  如确认这些图按'无参照'处理，加 --allow_missing_wb")
        sys.exit(1)

    # 第 1 轮：由白纸组图像确定培养基标准色
    agar_vals = []
    for jp, d in items:
        if d.get("wb", {}).get("mode") == "paper" and d.get("plate") and d.get("colony"):
            v = agar_color_after(root / d["file"], d)
            if v is not None:
                agar_vals.append(v)
    agar_target = np.median(np.stack(agar_vals), 0) if len(agar_vals) >= 3 else None
    n_agar_imgs = sum(1 for _, d in items if d.get("wb", {}).get("mode") == "agar")
    if agar_target is None and n_agar_imgs:
        print(f"⚠ 白纸组中可测培养基颜色的图只有 {len(agar_vals)} 张（需 ≥3），培养基模式的 {n_agar_imgs} 张图无法校正")

    # 第 2 轮：逐张处理
    rows, qcs = [], []
    for i, (jp, d) in enumerate(items, 1):
        try:
            row, qc = process_one(root / d["file"], d, agar_target, out_root)
        except Exception as e:
            row, qc = {"strain_id": d.get("strain_id"), "file": d.get("file"), "flags": f"error:{type(e).__name__}:{e}",
                       "main_analysis": False}, None
        rows.append(row)
        if qc:
            qcs.append(qc)
        print(f"  [{i}/{len(items)}] {Path(d['file']).name}  {row.get('wb_mode', '-')}  {row.get('flags') or 'ok'}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifest.csv", index=False, encoding="utf-8-sig")
    dup = None
    if qcs:
        qc_sheets(qcs, out_root)
        print("检查重复平皿 …")
        dup = duplicate_check(qcs, out_root)
    cfg = {"version": VERSION, "built_at": dt.datetime.now().isoformat(timespec="seconds"),
           "root": str(root), "views": args.views, "PLATE_R": PLATE_R, "CANVAS": CANVAS, "CROP": CROP,
           "WHITE_SRGB": WHITE_SRGB, "GRAY": GRAY, "RING": RING, "AGAR_AUTO": AGAR_AUTO,
           "EDGE_MAX_BAD": EDGE_MAX_BAD, "MIN_REF_PX": MIN_REF_PX,
           "agar_target_srgb": None if agar_target is None else [round(float(v), 4) for v in lin_to_srgb(agar_target)],
           "agar_target_n_images": len(agar_vals)}
    (out_root / "preprocess_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总
    print("\n===== 预处理汇总 =====")
    if "group" in df:
        s = df.groupby("group").agg(图片=("file", "size"), 株=("strain_id", "nunique"),
                                    白纸=("wb_mode", lambda x: (x == "paper").sum()),
                                    培养基=("wb_mode", lambda x: (x == "agar").sum()),
                                    无参照=("wb_mode", lambda x: (x == "none").sum()),
                                    有质控标记=("flags", lambda x: (x.fillna("") != "").sum()),
                                    进入主分析=("main_analysis", "sum"))
        print(s.to_string())
    if cfg["agar_target_srgb"]:
        print(f"培养基标准色 sRGB = {[round(v * 255) for v in cfg['agar_target_srgb']]}（来自 {len(agar_vals)} 张白纸组图像）")
    if dup is not None:
        sus = dup[dup["suspected_same_plate"] & ~dup["same_strain_id"]]
        if len(sus):
            print(f"\n⚠ 疑似同一平皿但株号不同（{len(sus)} 对，请核对；若属实应合并为同一株）：")
            for _, r in sus.iterrows():
                print(f"   {r['strain_a']}  ↔  {r['strain_b']}   匹配内点 {r['inliers']}")
        else:
            print("\n未发现株号不同的重复平皿")
    print(f"\n输出：{out_root}")
    print("请打开 qc_sheets/ 逐组检查：校正后颜色是否与平皿实物一致、截图位置是否正确；"
          "_center_by_group.jpg 看同组颜色是否接近、组间是否可区分。")


# =============================================================== 入口
def main():
    ap = argparse.ArgumentParser(description="菌落照片预处理")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pick", help="交互确认白平衡参照区")
    p1.add_argument("--root", required=True)
    p1.add_argument("--views", default="O", help="O=正面（默认）；O,R=正反面")
    p1.add_argument("--only", default=None)
    p1.add_argument("--redo", action="store_true")
    p1.add_argument("--annotator", default=getpass.getuser())
    p2 = sub.add_parser("build", help="批量预处理")
    p2.add_argument("--root", required=True)
    p2.add_argument("--out", default=None, help="默认 <root>/preprocessed")
    p2.add_argument("--views", default="O")
    p2.add_argument("--allow_missing_wb", action="store_true")
    args = ap.parse_args()
    {"pick": cmd_pick, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
