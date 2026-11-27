#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_roi_analysis.py —— 盲法 ROI 外部验证的诊断分析（找出"为什么只有 0.33"）

回答 5 个问题：
  1. 单个 ROI 的准确率是多少？3 个模型各自 vs 集成
  2. 截图质量的影响：结构类型（完整/局部/多个）、生长阶段（成熟/未成熟）、框大小
  3. 汇总方式的影响：置信加权 / 平均 / 多数投票 / "至少一个 ROI 判对"（上限）
  4. 来源（实验室）的影响：每个文件夹的准确率 —— 判断是否存在实验室特异的域偏移
  5. 颜色域偏移：外部 ROI 与训练截图的亮度 / 饱和度 / 色相对比
另输出最常见的误判方向（真实 → 预测）

用法：
    python 05_roi_analysis.py --models runs/swin_t_full_final runs/efficientnet_b0_full_final runs/densenet121_full_final
输出：roi_diag/report.txt、roi_diag/per_roi.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib                                                  # noqa: E402
_ev = importlib.import_module("03_eval_external")
load_models, predict_tiles, EXT_ROOT = _ev.load_models, _ev.predict_tiles, _ev.EXT_ROOT
get_device = importlib.import_module("02_train").get_device
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ROI_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/roi_test"


def color_stats(paths, n=None, seed=0):
    if n and len(paths) > n:
        paths = list(pd.Series(paths).sample(n, random_state=seed))
    rows = []
    for p in paths:
        try:
            im = ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((128, 128))
        except Exception:
            continue
        hsv = np.asarray(im.convert("HSV"), dtype=float)
        rgb = np.asarray(im, dtype=float)
        rows.append(dict(亮度V=hsv[..., 2].mean(), 饱和度S=hsv[..., 1].mean(),
                         R=rgb[..., 0].mean(), G=rgb[..., 1].mean(), B=rgb[..., 2].mean()))
    return pd.DataFrame(rows)


def lab_stats(imgs):
    import cv2
    v = [cv2.cvtColor(np.asarray(im.resize((128, 128))), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float) for im in imgs]
    v = np.concatenate(v)
    return v.mean(0), v.std(0)


def reinhard(im, tgt_mean, tgt_std):
    """Reinhard 颜色归一化：把每个 ROI 在 LAB 空间的均值/标准差对齐到训练截图"""
    import cv2
    lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB).astype(float)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - m) / s * tgt_std + tgt_mean
    return Image.fromarray(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--roi_dir", default=ROI_ROOT)
    ap.add_argument("--manifest", default=OP.MANIFEST)
    ap.add_argument("--trainset", default="trainset")
    ap.add_argument("--out", default="roi_diag", help="只写名字时放进 output/05_roi_diagnostics/")
    ap.add_argument("--norm", choices=["none", "reinhard"], default="none",
                    help="reinhard：测试时把外部 ROI 的颜色对齐到训练截图（不需重新训练），检验颜色域偏移")
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    a.trainset = OP.find_trainset_csv(a.trainset)
    a.out = OP.place(a.out, OP.DIAG)
    a.models = [OP.find_run(d) for d in a.models]
    os.makedirs(a.out, exist_ok=True)

    m = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
    m = m[m["role"] == "external_micro"]
    lg = pd.read_csv(os.path.join(a.roi_dir, "crop_log.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    lg = lg[lg["status"] == "crop"].copy()
    rel = {r: r for r in m["rel_path"]}
    lg["rel_path"] = [next((r for r in rel if s.endswith("/" + r) or s == r), "") for s in lg["source_image"]]
    lg = lg.merge(m[["rel_path", "folder", "train_class"]].rename(columns={"train_class": "truth"}), on="rel_path")
    lg = lg[lg["truth"] != ""].reset_index(drop=True)
    if lg.empty:
        sys.exit("没有找到集内（本期 26 类）外部图的 ROI")
    print(f"集内 ROI {len(lg)} 个，来自 {lg['rel_path'].nunique()} 张原图")

    device = get_device()
    models, names, sizes = load_models(a.models, device)
    imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in lg["crop_file"]]
    if a.norm == "reinhard":
        tr0 = pd.read_csv(a.trainset, encoding="utf-8-sig")
        ref = [ImageOps.exif_transpose(Image.open(p)).convert("RGB")
               for p in tr0["path"].sample(min(400, len(tr0)), random_state=0)]
        tm, ts = lab_stats(ref)
        imgs = [reinhard(im, tm, ts) for im in imgs]
        print("已对外部 ROI 做 Reinhard 颜色归一化（对齐训练截图）")
        os.makedirs(os.path.join(a.out, "examples"), exist_ok=True)
        for k in range(0, len(imgs), max(1, len(imgs) // 12)):
            imgs[k].save(os.path.join(a.out, "examples", f"norm_{k:04d}.jpg"))
    per_model = {}
    for d, mdl, sz in zip(a.models, models, sizes):
        per_model[os.path.basename(d.rstrip("/"))] = predict_tiles(imgs, [mdl], [sz], device)
    P = np.mean(list(per_model.values()), 0)
    lg["pred"] = [names[i] for i in P.argmax(1)]
    lg["p_top1"] = P.max(1)
    lg["ok"] = lg["pred"] == lg["truth"]
    for k, p in per_model.items():
        lg[f"ok_{k}"] = [names[i] == t for i, t in zip(p.argmax(1), lg["truth"])]
    lg["box"] = pd.to_numeric(lg["w"], errors="coerce")
    lg["genus_ok"] = lg["pred"].str.split().str[0] == lg["truth"].str.split().str[0]
    lg.drop(columns=[]).to_csv(os.path.join(a.out, "per_roi.csv"), index=False, encoding="utf-8-sig")

    L = []
    L.append("===== 1. 单个 ROI 的准确率 =====")
    L.append(f"集成：{lg['ok'].mean():.3f}   属一级：{lg['genus_ok'].mean():.3f}   (n={len(lg)})")
    for k in per_model:
        L.append(f"  {k:<28s}{lg[f'ok_{k}'].mean():.3f}")

    L.append("\n===== 2. 截图质量的影响（ROI 级准确率） =====")
    for col in ["structure", "stage"]:
        if col in lg and lg[col].ne("").any():
            L.append(OP.table(lg[lg[col] != ""].groupby(col)["ok"].agg(准确率="mean", n="size").round(3)))
    lg["box_bin"] = pd.cut(lg["box"], [0, 160, 320, 480, 2000], labels=["≤160", "161–320", "321–480", ">480"])
    L.append(OP.table(lg.groupby("box_bin", observed=True)["ok"].agg(准确率="mean", n="size").round(3)))
    lg["conf_bin"] = pd.cut(lg["p_top1"], [0, .4, .6, .8, 1.01], labels=["<0.4", "0.4–0.6", "0.6–0.8", "≥0.8"])
    L.append(OP.table(lg.groupby("conf_bin", observed=True)["ok"].agg(准确率="mean", n="size").round(3)))

    L.append("\n===== 3. 汇总方式（按原图，n=%d） =====" % lg["rel_path"].nunique())
    pc = pd.DataFrame(P, columns=names)
    res = {"置信加权": [], "平均": [], "多数投票": [], "至少一个 ROI 判对（上限）": [], "单个最高置信 ROI": []}
    img_rows = []
    for rp, idx in lg.groupby("rel_path").groups.items():
        p = P[idx]
        truth = lg.loc[idx[0], "truth"]
        w = p.max(1) ** 2
        conf = names[int(((p * w[:, None]).sum(0) / w.sum()).argmax())]
        mean = names[int(p.mean(0).argmax())]
        vote = pd.Series([names[i] for i in p.argmax(1)]).value_counts().index[0]
        best = names[int(p[p.max(1).argmax()].argmax())]
        res["置信加权"].append(conf == truth)
        res["平均"].append(mean == truth)
        res["多数投票"].append(vote == truth)
        res["至少一个 ROI 判对（上限）"].append(any(names[i] == truth for i in p.argmax(1)))
        res["单个最高置信 ROI"].append(best == truth)
        img_rows.append(dict(rel_path=rp, folder=lg.loc[idx[0], "folder"], truth=truth, pred=conf,
                             ok=conf == truth, n_roi=len(idx)))
    for k, v in res.items():
        L.append(f"  {k:<26s}{np.mean(v):.3f}")
    im = pd.DataFrame(img_rows)

    L.append("\n===== 4. 各来源（文件夹）准确率（按原图，置信加权） =====")
    L.append(OP.table(im.groupby("folder")["ok"].agg(准确率="mean", n="size").sort_values("n", ascending=False).round(3)))

    L.append("\n===== 5. 最常见的误判（真实 → 预测，按原图） =====")
    err = im[~im["ok"]].groupby(["truth", "pred"]).size().sort_values(ascending=False).head(20)
    L.append(OP.table(err))
    L.append("\n被预测最多的类别（是否有\"吸收类\"）：")
    L.append(OP.table(im["pred"].value_counts().head(8)))

    L.append("\n===== 6. 颜色域偏移：外部 ROI vs 训练截图 =====")
    tr = pd.read_csv(a.trainset, encoding="utf-8-sig")
    cs = [("训练截图", color_stats(list(tr["path"]), n=400))]
    for f, d in lg.groupby("folder"):
        if len(d) >= 10:
            cs.append((f"外部 {f}", color_stats(list(d["crop_file"]), n=200)))
    L.append(OP.table(pd.DataFrame({k: v.mean() for k, v in cs}).T.round(1)))
    L.append("（若某来源的亮度/饱和度/RGB 与训练截图差别很大且该来源准确率低 → 颜色/染色域偏移）")

    txt = "\n".join(L)
    print("\n" + txt)
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write(txt)
    im.to_csv(os.path.join(a.out, "per_image.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已写出 {a.out}/report.txt、per_roi.csv、per_image.csv")


if __name__ == "__main__":
    main()
