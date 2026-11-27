#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_check_roi_overlap.py —— 检查训练集里有没有、有多少 roi_test 的 ROI（外部测试是否被"泄漏"进训练）

两种比对，互相印证：
  1) 路径：训练清单的 path 与 roi_test 里的截图文件路径
  2) 内容：文件 md5（即使复制、改名、换了文件夹也能认出是同一张图）

用法（Audit 文件夹）：
    python 04_check_roi_overlap.py
    python 04_check_roi_overlap.py --roi_dir /Users/chunjiang/Documents/真菌AI/CNN/roi_test --trainset trainset_roi_v3
输出：屏幕汇总 + <trainset>/roi_overlap.csv（每个 ROI 一行：是否在训练集、集内/集外、来源、原图）
"""
import os
import sys
import hashlib
import argparse
import unicodedata
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def dw(s):
    """显示宽度：中文等全角字符占 2 格"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def table(df):
    """按显示宽度对齐的表格（pandas 的 to_string 遇到中文会错位）"""
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))) for v in r] for r in df.values]
    w = [max(dw(c), *(dw(r[i]) for r in rows)) for i, c in enumerate(cols)]
    num = [all(r[i].replace(".", "", 1).isdigit() for r in rows if r[i]) for i in range(len(cols))]

    def fmt(vals):
        out = []
        for i, v in enumerate(vals):
            pad = " " * (w[i] - dw(v))
            out.append(pad + v if num[i] and i > 0 else v + pad)
        return "  ".join(out)
    line = "-" * dw(fmt(cols))
    return "\n".join([fmt(cols), line] + [fmt(r) for r in rows])


def norm(p):
    return unicodedata.normalize("NFC", os.path.abspath(os.path.expanduser(str(p).strip())))


def md5(p, cache={}):
    if p not in cache:
        try:
            with open(p, "rb") as f:
                cache[p] = hashlib.md5(f.read()).hexdigest()
        except OSError:
            cache[p] = ""
    return cache[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_dir", default="/Users/chunjiang/Documents/真菌AI/CNN/roi_test")
    ap.add_argument("--trainset", default="trainset_roi_v3",
                    help="训练集文件夹或 trainset_micro.csv")
    ap.add_argument("--manifest", default=OP.MANIFEST, help="用来区分集内（26 类）/集外 ROI")
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    a.trainset = OP.find_trainset_csv(a.trainset)

    ts_csv = os.path.join(a.trainset, "trainset_micro.csv") if os.path.isdir(a.trainset) else a.trainset
    if not os.path.exists(ts_csv):
        sys.exit(f"找不到 {ts_csv}")
    ts = pd.read_csv(ts_csv, encoding="utf-8-sig", dtype=str).fillna("")
    ts["_p"] = ts["path"].map(norm)
    print(f"训练清单：{ts_csv}  共 {len(ts)} 张；其中 source=roi_ext 的 {int((ts.get('source', '') == 'roi_ext').sum())} 张")

    # roi_test：日志里的截图 + 磁盘上实际存在的图片
    log_path = os.path.join(a.roi_dir, "crop_log.csv")
    lg = pd.read_csv(log_path, encoding="utf-8-sig", dtype=str).fillna("")
    lg = lg[lg["status"] == "crop"].drop_duplicates("crop_file").copy()
    lg["_p"] = lg["crop_file"].map(norm)
    on_disk = set()
    for d, subs, files in os.walk(a.roi_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS) and not f.startswith("."):
                on_disk.add(norm(os.path.join(d, f)))
    lg["exists"] = lg["_p"].isin(on_disk)
    extra = on_disk - set(lg["_p"])
    print(f"roi_test：日志记录 ROI {len(lg)} 个（文件存在 {int(lg['exists'].sum())}）；磁盘上另有未记录的图片 {len(extra)} 张")

    # 集内 / 集外
    lg["panel"] = "未知"
    if os.path.exists(a.manifest):
        m = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
        rel2 = dict(zip(m["rel_path"], zip(m["role"], m["train_class"], m["folder"])))
        def info(src):
            for rel, v in rel2.items():
                if src.endswith("/" + rel):
                    return v
            return ("", "", "")
        inf = lg["source_image"].map(info)
        lg["role"] = [x[0] for x in inf]
        lg["truth"] = [x[1] for x in inf]
        lg["folder"] = [x[2] for x in inf]
        lg["panel"] = ["集内(26类)" if r == "external_micro" and t else ("集外" if r else "未知") for r, t in zip(lg["role"], lg["truth"])]
    else:
        print(f"（找不到 {a.manifest}，不区分集内/集外）")

    # 1) 路径比对
    lg["in_train_path"] = lg["_p"].isin(set(ts["_p"]))
    # 2) 内容比对（只算训练集里可能相关的图：全部算，md5 有缓存）
    print("计算 md5 …（训练集图片较多时需要一两分钟）")
    roi_md5 = {p: md5(p) for p in lg.loc[lg["exists"], "_p"]}
    ts_md5 = {}
    for p in ts["_p"]:
        h = md5(p)
        if h:
            ts_md5.setdefault(h, p)
    lg["in_train_md5"] = lg["_p"].map(lambda p: bool(roi_md5.get(p)) and roi_md5.get(p) in ts_md5)
    lg["in_train"] = lg["in_train_path"] | lg["in_train_md5"]

    print("\n===== ROI 是否在训练集里（按 ROI 个数） =====")
    t = lg.groupby("panel").agg(ROI数=("_p", "size"), 在训练集=("in_train", "sum"),
                                 路径命中=("in_train_path", "sum"), 内容命中=("in_train_md5", "sum"))
    t["比例"] = (t["在训练集"] / t["ROI数"]).round(3)
    print(table(t))

    src = lg.groupby(["panel", "source_image"])["in_train"].agg(["size", "sum"]).reset_index()
    src["状态"] = ["全部在训练集" if s == n else ("部分在" if s > 0 else "都不在") for n, s in zip(src["size"], src["sum"])]
    print("\n===== 按原图（外部测试图）统计 =====")
    print(table(src.groupby(["panel", "状态"]).size().unstack(fill_value=0)))

    if "folder" in lg:
        print("\n===== 集内 ROI 按来源 =====")
        d = lg[lg["panel"] == "集内(26类)"]
        print(table(d.groupby("folder")["in_train"].agg(ROI数="size", 在训练集="sum")))

    miss = lg[(lg["panel"] == "集内(26类)") & ~lg["in_train"]]
    if len(miss):
        print(f"\n集内但不在训练集的 ROI（{len(miss)} 个），前 10 个：")
        for _, r in miss.head(10).iterrows():
            print(f"  {os.path.basename(r['crop_file'])}  宽 {r.get('w', '')}  文件存在 {r['exists']}")

    out = os.path.join(os.path.dirname(ts_csv), "roi_overlap.csv")
    lg.drop(columns=["_p"]).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细：{out}")


if __name__ == "__main__":
    main()
