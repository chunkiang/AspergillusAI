#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_eval_external.py —— 外部验证：多尺度自动切片 → 集成预测 → 按病例汇总 → 不确定性标记 → 集外菌种分析

用法：
  python 03_eval_external.py --models runs/swin_t_full_XXXX runs/efficientnet_b0_full_XXXX runs/densenet121_full_XXXX
  （每个目录里要有 final.pt、args.json、class_names.json —— 即 02_train.py --mode full 的输出）

输入：ext_audit/manifest.csv（00_build_manifest.py 生成）
  · role = external_micro                     → 外部验证主分析（本期 26 类）
  · role = out_of_panel 且为镜下图             → 集外菌种（开放集）：期望被标记为"需复核"
外部原图根目录默认 /Users/chunjiang/Desktop/丝状真菌

处理流程：
  1. 每张原图按 160 / 320 / 480 像素（原图像素）三种尺度、半步长滑窗切片；去掉空白片
     （按灰度标准差 + 边缘密度给每片打"前景分"，每个尺度保留前景分最高的若干片），外加整图一张
  2. 每片缩放到模型输入尺寸，多个模型的 softmax 概率取平均（集成）
  3. 一张图 = 各片概率的置信加权平均；一个病例 = 同一来源、同一菌种、编号连续的图取平均
  4. 不确定性标记：top1 概率 < 0.70 或 top1 − top2 < 0.20 → 需复核（阈值可调，论文里须预先设定）

输出（默认 ./ext_eval/）：
  per_tile.csv（可选）、per_image.csv、per_case.csv、metrics.json、metrics.txt、
  coverage_curve.csv（不同阈值下的"自动报告比例 vs 准确率"）
依赖：torch torchvision pandas numpy pillow pillow-heif scikit-learn
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import re
import sys
import json
import math
import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib                                                     # noqa: E402
_train = importlib.import_module("02_train")                          # noqa: E402  （文件名以数字开头，只能这样导入）
build_model, get_device = _train.build_model, _train.get_device
IMAGENET_MEAN, IMAGENET_STD = _train.IMAGENET_MEAN, _train.IMAGENET_STD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

EXT_ROOT = "/Users/chunjiang/Desktop/丝状真菌"
# 临床上更关心的"组"一级（用于分层准确率）
GROUP_OF = {
    "Rhizopus": "Mucorales", "Mucor": "Mucorales", "Lichtheimia": "Mucorales", "Cunninghamella": "Mucorales",
}


def group_of(c):
    if c.startswith("Aspergillus"):
        return "Aspergillus"
    return GROUP_OF.get(c, c)


# =============================================================================
# 模型
# =============================================================================
def load_models(dirs, device):
    models, names, sizes = [], None, []
    for d in dirs:
        d = OP.find_run(d)
        args = json.load(open(os.path.join(d, "args.json"), encoding="utf-8"))
        cn = json.load(open(os.path.join(d, "class_names.json"), encoding="utf-8"))
        if names is None:
            names = cn
        elif cn != names:
            sys.exit(f"{d} 的类别顺序与第一个模型不同，不能集成")
        ck = os.path.join(d, "final.pt")
        if not os.path.exists(ck):
            sys.exit(f"{d} 里没有 final.pt（需要 02_train.py --mode full 的输出）")
        m = build_model(args["backbone"], len(cn), pretrained=False)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        models.append(m.to(device).eval())
        sizes.append(int(args.get("img_size", 224)))
        print(f"载入 {args['backbone']:16s} ← {d}")
    return models, names, sizes


# =============================================================================
# 切片
# =============================================================================
def fg_score(tile):
    g = np.asarray(tile.convert("L").resize((64, 64)), dtype=np.float32)
    edges = np.asarray(tile.convert("L").resize((64, 64)).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float(g.std() + edges.mean())


def make_tiles(img, scales, per_scale, min_score, whole=True):
    w, h = img.size
    tiles, meta = [], []
    for s in scales:
        if min(w, h) < s * 0.8:          # 原图比切片还小：这个尺度跳过
            continue
        s_eff = min(s, w, h)
        step = max(1, s_eff // 2)
        cand = []
        for y in range(0, max(1, h - s_eff + 1), step):
            for x in range(0, max(1, w - s_eff + 1), step):
                t = img.crop((x, y, x + s_eff, y + s_eff))
                cand.append((fg_score(t), x, y, t))
        cand.sort(key=lambda c: -c[0])
        for sc, x, y, t in cand[:per_scale]:
            if sc >= min_score:
                tiles.append(t)
                meta.append(dict(scale=s, x=x, y=y, fg=round(sc, 2)))
    if whole or not tiles:               # 整图一张，兼顾"整张就是一个结构"的情况
        tiles.append(img)
        meta.append(dict(scale="whole", x=0, y=0, fg=round(fg_score(img), 2)))
    return tiles, meta


@torch.no_grad()
def predict_tiles(tiles, models, sizes, device, batch=48):
    probs = 0
    for m, sz in zip(models, sizes):
        tf = transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(),
                                 transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        out = []
        for i in range(0, len(tiles), batch):
            x = torch.stack([tf(t) for t in tiles[i:i + batch]]).to(device)
            out.append(torch.softmax(m(x).float(), 1).cpu().numpy())
        probs = probs + np.concatenate(out)
    return probs / len(models)


def aggregate(p, how):
    if how == "mean":
        return p.mean(0)
    w = p.max(1) ** 2                    # 置信加权：清楚的片说了算，空白/模糊片权重低
    return (p * w[:, None]).sum(0) / w.sum()


# =============================================================================
# 病例分组与统计
# =============================================================================
def case_ids(df):
    """同一文件夹、同一菌种、编号连续（相差 ≤ 1）的图归为同一病例；没有编号的图各自一例"""
    num = df["rel_path"].map(lambda p: re.search(r"[-_](\d+)\D*$", os.path.splitext(os.path.basename(p))[0]))
    df = df.assign(_n=[int(m.group(1)) if m else -1 for m in num])
    ids = {}
    for (folder, sp), d in df.groupby(["folder", "species_cn"]):
        d = d.sort_values("_n")
        k, prev = 0, None
        for i, n in zip(d.index, d["_n"]):
            if n < 0 or prev is None or prev < 0 or n - prev > 1:
                k += 1
            ids[i] = f"{folder}|{sp}|{k}"
            prev = n
    return df.index.map(ids)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def summarize(cases, names, thr_p, thr_gap):
    from sklearn.metrics import f1_score
    res = {}
    inp = cases[cases["in_panel"]]
    for scope, d in [("全部", inp)] + [(s, g) for s, g in inp.groupby("source_type")]:
        n = len(d)
        k1 = int((d["pred"] == d["truth"]).sum())
        k2 = int(d["top2_hit"].sum())
        acc = d[~d["flag"]]
        r = dict(病例数=n, top1=round(k1 / max(n, 1), 3), top1_95CI=wilson(k1, n),
                 top2=round(k2 / max(n, 1), 3), top2_95CI=wilson(k2, n),
                 macro_F1=round(float(f1_score(d["truth"], d["pred"], labels=sorted(d["truth"].unique()),
                                               average="macro", zero_division=0)), 3) if n else None,
                 组一级准确率=round(float((d["truth"].map(group_of) == d["pred"].map(group_of)).mean()), 3) if n else None,
                 标记复核比例=round(float(d["flag"].mean()), 3) if n else None,
                 未标记病例准确率=round(float((acc["pred"] == acc["truth"]).mean()), 3) if len(acc) else None,
                 被标记病例中实际错误比例=round(float((d[d["flag"]]["pred"] != d[d["flag"]]["truth"]).mean()), 3)
                 if d["flag"].any() else None,
                 ECE=round(ece(d["p_top1"].values, (d["pred"] == d["truth"]).values.astype(float)), 3) if n else None)
        res[scope] = r
    # 每类敏感度 / PPV
    per = []
    for c in names:
        t = inp[inp["truth"] == c]
        pp = inp[inp["pred"] == c]
        if len(t) or len(pp):
            per.append(dict(类别=c, 病例数=len(t),
                            敏感度=round(float((t["pred"] == c).mean()), 3) if len(t) else None,
                            敏感度_95CI=wilson(int((t["pred"] == c).sum()), len(t)),
                            PPV=round(float((pp["truth"] == c).mean()), 3) if len(pp) else None))
    res["每类"] = per
    # 开放集：集外菌种应被标记
    oos = cases[~cases["in_panel"]]
    if len(oos):
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(inp)), np.ones(len(oos))]
        s = np.r_[1 - inp["p_top1"].values, 1 - oos["p_top1"].values]
        res["开放集"] = dict(集外病例数=len(oos), 集外被标记比例=round(float(oos["flag"].mean()), 3),
                          集内被标记比例=round(float(inp["flag"].mean()), 3),
                          AUROC_以top1概率区分集内外=round(float(roc_auc_score(y, s)), 3) if len(inp) else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default=OP.MANIFEST)
    ap.add_argument("--ext_root", default=EXT_ROOT)
    ap.add_argument("--out", default="ext_eval", help="只写名字时放进 output/03_external_eval/")
    ap.add_argument("--scales", nargs="+", type=int, default=[160, 320, 480])
    ap.add_argument("--per_scale", type=int, default=24, help="每个尺度保留前景分最高的切片数")
    ap.add_argument("--min_fg", type=float, default=8.0, help="前景分低于此值的切片丢弃")
    ap.add_argument("--agg", choices=["conf", "mean"], default="conf")
    ap.add_argument("--thr_p", type=float, default=0.70)
    ap.add_argument("--thr_gap", type=float, default=0.20)
    ap.add_argument("--sources", nargs="*", default=[], help="只评估这些文件夹（关键字）")
    ap.add_argument("--save_tiles", action="store_true")
    ap.add_argument("--whole_only", action="store_true", help="诊断用：只用整图，不切片")
    ap.add_argument("--no_whole", action="store_true", help="诊断用：只用切片，不加整图")
    ap.add_argument("--roi_dir", default="", help="用 03_roi_blind.py 盲法截取的 ROI 代替自动切片（含 crop_log.csv 与 roi_key.csv）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试）")
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    a.out = OP.place(a.out, OP.EXT)
    a.models = [OP.find_run(d) for d in a.models]
    os.makedirs(a.out, exist_ok=True)

    device = get_device()
    models, names, sizes = load_models(a.models, device)
    K = len(names)

    m = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
    keep = (m["role"] == "external_micro") | (
        (m["role"] == "out_of_panel") & ~m["modality_guess"].isin(["colony", "direct_specimen"]))
    m = m[keep].copy()
    if a.sources:
        m = m[m["folder"].apply(lambda f: any(k in f for k in a.sources))]
    if a.limit:
        m = m.head(a.limit)
    m["in_panel"] = m["role"] == "external_micro"
    m["source_type"] = np.where(m["folder"].str.contains("网络"), "网络", "临床")
    print(f"设备 {device} | 模型 {len(models)} 个 | 外部镜下图 {int(m['in_panel'].sum())} 张 + 集外菌种 {int((~m['in_panel']).sum())} 张")

    roi = {}
    if a.roi_dir:
        lg = pd.read_csv(os.path.join(a.roi_dir, "crop_log.csv"), encoding="utf-8-sig", dtype=str).fillna("")
        lg = lg[lg["status"] == "crop"]
        for src, d in lg.groupby("source_image"):
            roi[src] = d["crop_file"].tolist()
        print(f"ROI 模式：{len(roi)} 张原图有盲法截取的 ROI，共 {len(lg)} 个")
    rows, tile_rows = [], []
    for n_done, (i, r) in enumerate(m.iterrows(), 1):
        path = os.path.join(a.ext_root, r["rel_path"])
        if a.roi_dir:
            files = roi.get(path) or next((v for k, v in roi.items() if k.endswith("/" + r["rel_path"])), [])
            if not files:
                continue                 # 没有截取 ROI（跳过的图）：不纳入 ROI 分析，另行报告
            tiles = [ImageOps.exif_transpose(Image.open(f)).convert("RGB") for f in files]
            meta = [dict(scale="roi", x=0, y=0, fg=0)] * len(tiles)
        else:
            try:
                img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception as e:
                print(f"  读取失败 {r['rel_path']}: {e}")
                continue
            if a.whole_only:
                tiles, meta = [img], [dict(scale="whole", x=0, y=0, fg=0)]
            else:
                tiles, meta = make_tiles(img, a.scales, a.per_scale, a.min_fg, whole=not a.no_whole)
        p = predict_tiles(tiles, models, sizes, device)
        pi = aggregate(p, a.agg)
        rows.append(dict(idx=i, **{f"p::{c}": float(pi[k]) for k, c in enumerate(names)}, n_tiles=len(tiles)))
        if a.save_tiles:
            for t, mt in zip(p, meta):
                tile_rows.append(dict(rel_path=r["rel_path"], **mt, pred=names[int(t.argmax())], p=float(t.max())))
        if n_done % 20 == 0:
            print(f"  {n_done}/{len(m)}")
    P = pd.DataFrame(rows).set_index("idx")
    m = m.join(P, how="inner")
    pc = [f"p::{c}" for c in names]

    def finish(d, level):
        pr = d[pc].values
        order = np.argsort(-pr, 1)
        d = d.copy()
        d["pred"] = [names[o[0]] for o in order]
        d["pred2"] = [names[o[1]] for o in order]
        d["p_top1"] = pr[np.arange(len(d)), order[:, 0]]
        d["p_top2"] = pr[np.arange(len(d)), order[:, 1]]
        d["truth"] = d["train_class"].where(d["in_panel"], "（集外）" + d["species_cn"])
        d["top2_hit"] = (d["pred"] == d["truth"]) | (d["pred2"] == d["truth"])
        d["flag"] = (d["p_top1"] < a.thr_p) | (d["p_top1"] - d["p_top2"] < a.thr_gap)
        front = ["truth", "pred", "p_top1", "pred2", "p_top2", "top2_hit", "flag"]
        d[front + [c for c in d.columns if c not in front]].to_csv(
            os.path.join(a.out, f"per_{level}.csv"), index=False, encoding="utf-8-sig")
        return d

    img_df = finish(m, "image")
    m["case_id"] = case_ids(m)
    agg = {c: "mean" for c in pc}
    agg.update({k: "first" for k in ["folder", "species_cn", "train_class", "in_panel", "source_type"]})
    agg["rel_path"] = lambda s: ";".join(os.path.basename(x) for x in s)
    cases = m.groupby("case_id").agg(agg).reset_index()
    cases["n_images"] = m.groupby("case_id").size().values
    cases = finish(cases, "case")
    if a.save_tiles:
        pd.DataFrame(tile_rows).to_csv(os.path.join(a.out, "per_tile.csv"), index=False, encoding="utf-8-sig")

    if a.roi_dir:
        print(f"ROI 分析纳入 {len(m)} 张图（其余为截取时跳过的图）")
    res = {"设置": dict(models=a.models, scales=a.scales, per_scale=a.per_scale, agg=a.agg,
                       mode="roi" if a.roi_dir else ("whole_only" if a.whole_only else "tiles"),
                       thr_p=a.thr_p, thr_gap=a.thr_gap),
           "按病例": summarize(cases, names, a.thr_p, a.thr_gap),
           "按图片": summarize(img_df, names, a.thr_p, a.thr_gap)}
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 覆盖率–准确率曲线（论文图：阈值越严，自动报告越少，但准确率越高）
    inp = cases[cases["in_panel"]]
    cur = []
    for t in np.round(np.arange(0.0, 1.0, 0.05), 2):
        acc = inp[inp["p_top1"] >= t]
        cur.append(dict(阈值=t, 自动报告比例=round(len(acc) / max(len(inp), 1), 3),
                        自动报告准确率=round(float((acc["pred"] == acc["truth"]).mean()), 3) if len(acc) else None))
    pd.DataFrame(cur).to_csv(os.path.join(a.out, "coverage_curve.csv"), index=False, encoding="utf-8-sig")

    lines = []
    for lvl in ["按病例", "按图片"]:
        lines.append(f"===== {lvl} =====")
        for k, v in res[lvl].items():
            if k == "每类":
                continue
            lines.append(f"[{k}] " + "  ".join(f"{a_}={b}" for a_, b in v.items()))
        lines.append("")
    lines.append("===== 每类（按病例） =====")
    lines.append(OP.table(pd.DataFrame(res["按病例"]["每类"]), index=False))
    txt = "\n".join(lines)
    open(os.path.join(a.out, "metrics.txt"), "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print(f"\n已写出 {a.out}/")


if __name__ == "__main__":
    main()
