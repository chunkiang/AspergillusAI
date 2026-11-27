#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_eval_clinical.py —— 前瞻临床验证：用锁定的模型评估盲法截取的产孢结构 ROI

与 03_roi_clinical.py 配套：读取它揭盲后生成的 roi_clinical_table.csv，
用锁定的模型预测每个 ROI，按"株"汇总（株号来自文件夹名/文件名，不再靠编号连续推断），
输出论文需要的全部临床验证指标。不修改 03_eval_external.py，但沿用它的
load_models / predict_tiles，保证预处理与外部验证完全相同。

一、流程总览（在 Audit 文件夹、fungal 环境）
============================================
  ① 截图开始之前：锁定模型和判读规则（只做一次，生成 clinical_lock.json 并存档）
      python 03_eval_clinical.py --write_lock --models runs/clinical_locked

  ② 截图（03_roi_clinical.py，盲法）→ 全部截完 → 揭盲，生成 roi_clinical_table.csv
      python 03_roi_clinical.py --src "/Users/chunjiang/Desktop/临床验证_2026" --unblind --answers answers_clinical.csv

  ③ 评估（自动核对锁定文件：模型权重、类别、阈值必须与锁定时一致，否则拒绝运行）
      python 03_eval_clinical.py --models runs/clinical_locked

  多个模型集成（锁定时就要写成多个，评估时完全一样）：
      python 03_eval_clinical.py --write_lock --models runs/swin_t_full_final runs/efficientnet_b0_full_final
      python 03_eval_clinical.py --models runs/swin_t_full_final runs/efficientnet_b0_full_final

二、参数
========
  --models        一个或多个模型文件夹（含 final.pt）；短名自动到 output/02_training_runs 里找
  --roi_dir       03_roi_clinical.py 的输出文件夹，默认 /Users/chunjiang/Documents/真菌AI/CNN/roi_clinical
  --table         揭盲总表，默认 <roi_dir>/roi_clinical_table.csv
  --out           结果文件夹，默认 output/03_external_eval/ext_eval_clinical
  --lock          锁定文件，默认 Audit/clinical_lock.json；存在就核对
  --write_lock    只写锁定文件后退出（截图之前运行）
  --no_lock       不核对锁定文件（仅用于调试，结果不能写进论文）
  --p_min         复核阈值：病例 top1 概率低于它 → 需复核（默认 0.70，须与锁定一致）
  --gap_min       复核阈值：top1 − top2 低于它 → 需复核（默认 0.20，须与锁定一致）

三、预先设定的判读规则（与锁定文件一致）
========================================
  · 每个 ROI 得到 26 类概率（多个模型时取平均）；
  · 病例（株）级概率 = 该株全部 ROI 概率的算术平均；top1 即模型给出的鉴定；
  · top1 < p_min 或 top1 − top2 < gap_min → "需复核"，其余为"可直接报告"；
  · 集外菌种（答案表 train_class 为空）不计入准确率，只用于开放集评估
    （理想情况下应被标记为需复核）。

四、输出（<out>/）
=================
  metrics.txt          主要结果（中文），可直接整理进论文
  metrics.json         同上，机器可读
  per_isolate.csv      每株一行：参考鉴定、预测前 3 名及概率、是否复核、各层级是否正确、各类概率
  per_roi.csv          每个 ROI 一行：ROI 级预测、结构类型、成熟度
  confusion.csv        集内菌株的混淆矩阵（株级；行 = 参考，列 = 预测）
  confusion_roi.csv    集内菌株全部 ROI 的混淆矩阵（ROI 级）
                       两者也会打印在 metrics.txt 中；作图用 02b_plot_confusion.py
  coverage_curve.csv   按置信度排序，不同覆盖率下的准确率
  run_info.json        本次运行的模型文件指纹、阈值、输入表指纹、时间

五、metrics.txt 包含的指标
==========================
  主要终点：集内菌株 top1 准确率（95% CI，Clopper–Pearson）
  次要终点：top2；属一级；临床类群一级（曲霉属合一、毛霉目合一、其余按属）；
            macro-F1；平衡准确率；每类敏感度（95% CI）和 PPV；
            复核标记比例；未标记（可直接报告）病例的准确率及占比；标记病例的准确率；
            ECE（10 箱）；覆盖率–准确率；
            集外菌种：被标记复核的比例；开放集 AUROC（以 1 − top1 概率区分集内/集外）；
  辅助分析：ROI 级准确率；按结构类型（完整/局部/多个）和成熟度分层；
            汇总方式的敏感性分析（平均 / 多数投票 / 最高置信 ROI）—— 仅供讨论，主要结果以平均为准。
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

try:
    import paths as OP          # 输出文件夹约定（Audit 文件夹里的 paths.py）
    RUNS_ROOT, EXT_DIR = OP.RUNS_ROOT, OP.EXT
except Exception:               # 没有 paths.py 时退回到当前目录结构
    RUNS_ROOT = os.path.join(BASE, "output", "02_training_runs")
    EXT_DIR = os.path.join(BASE, "output", "03_external_eval")

DEFAULT_ROI = "/Users/chunjiang/Documents/真菌AI/CNN/roi_clinical"
DEFAULT_LOCK = os.path.join(BASE, "clinical_lock.json")
MUCORALES = {"Rhizopus", "Mucor", "Lichtheimia", "Cunninghamella", "Rhizomucor",
             "Apophysomyces", "Saksenaea", "Syncephalastrum"}


# --------------------------------------------------------------------------- 小工具
def resolve_model(p):
    p = os.path.expanduser(str(p).strip().strip("'\""))
    for c in (p, os.path.join(BASE, p), os.path.join(RUNS_ROOT, p), os.path.join(RUNS_ROOT, "runs", p)):
        if os.path.isdir(c):
            return os.path.abspath(c)
    sys.exit(f"找不到模型文件夹：{p}")


def resolve_out(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(EXT_DIR, p)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def model_fingerprint(d):
    pts = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    if not pts:
        sys.exit(f"{d} 里没有 .pt 模型文件")
    fp = {"dir": os.path.basename(d.rstrip("/")), "weights": {f: md5(os.path.join(d, f)) for f in pts}}
    cn = os.path.join(d, "class_names.json")
    if os.path.exists(cn):
        fp["class_names"] = json.load(open(cn, encoding="utf-8"))
    return fp


def exact_ci(k, n, alpha=0.05):
    """Clopper–Pearson 精确 95% CI。"""
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
        return (float(lo), float(hi))
    except ImportError:          # 退回 Wilson
        z, p = 1.959964, k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    if n == 0:
        return "—（n=0）"
    lo, hi = exact_ci(k, n)
    return f"{k / n:.3f}（{k}/{n}，95% CI {lo:.3f}–{hi:.3f}）"


def genus(c):
    return str(c).split()[0] if c else ""


def clin_group(c):
    g = genus(c)
    return "Mucorales" if g in MUCORALES else g


def auroc(pos, neg):
    """pos 应得分更高。Mann–Whitney 计算，含并列。"""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


# --------------------------------------------------------------------------- 锁定
def lock_content(model_dirs, p_min, gap_min):
    return {
        "models": [model_fingerprint(d) for d in model_dirs],
        "rule": {"aggregation": "mean of ROI probabilities per isolate",
                 "p_min": p_min, "gap_min": gap_min},
    }


def write_lock(path, model_dirs, p_min, gap_min):
    if os.path.exists(path):
        sys.exit(f"{path} 已存在。锁定文件只应生成一次；确需重新锁定，请先手动改名备份旧文件。")
    c = lock_content(model_dirs, p_min, gap_min)
    c["locked_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=2)
    print(f"已锁定 → {path}")
    for m in c["models"]:
        print(f"  模型 {m['dir']}：" + "，".join(f"{k} {v[:8]}…" for k, v in m["weights"].items()))
    print(f"  判读规则：病例级平均概率；top1 < {p_min} 或 top1 − top2 < {gap_min} → 需复核")
    print("请把 clinical_lock.json 连同验证方案一起存档（例如发邮件给自己留时间戳），再开始截图。")


def check_lock(path, model_dirs, p_min, gap_min):
    if not os.path.exists(path):
        sys.exit(f"找不到锁定文件 {path}。\n截图前应先运行：python 03_eval_clinical.py --write_lock --models …\n"
                 "（调试可加 --no_lock，但结果不能写进论文）")
    L = json.load(open(path, encoding="utf-8"))
    now = lock_content(model_dirs, p_min, gap_min)
    errs = []
    if [m["weights"] for m in L["models"]] != [m["weights"] for m in now["models"]]:
        errs.append("模型权重与锁定时不同（换了模型、重训过，或 --models 的顺序/数量不同）")
    if [m.get("class_names") for m in L["models"]] != [m.get("class_names") for m in now["models"]]:
        errs.append("类别列表与锁定时不同")
    if L["rule"]["p_min"] != p_min or L["rule"]["gap_min"] != gap_min:
        errs.append(f"复核阈值与锁定时不同（锁定 p_min={L['rule']['p_min']}, gap_min={L['rule']['gap_min']}）")
    if errs:
        sys.exit("锁定核对未通过：\n  - " + "\n  - ".join(errs))
    print(f"锁定核对通过（锁定于 {L.get('locked_at', '?')}）")
    return L.get("locked_at", "")


# --------------------------------------------------------------------------- 预测
def predict(model_dirs, roi_paths):
    """调用 03_eval_external 的 load_models / predict_tiles，返回 (概率矩阵, 类名, 各模型概率)。"""
    from PIL import Image, ImageOps
    ev = importlib.import_module("03_eval_external")
    get_device = importlib.import_module("02_train").get_device
    device = get_device()
    models, names, sizes = ev.load_models(model_dirs, device)
    imgs = []
    for p in roi_paths:
        with Image.open(p) as im:
            imgs.append(ImageOps.exif_transpose(im).convert("RGB"))
    per_model = [np.asarray(ev.predict_tiles(imgs, [m], [s], device), float)
                 for m, s in zip(models, sizes)]
    return np.mean(per_model, 0), list(names), per_model


# --------------------------------------------------------------------------- 汇总与指标
def isolate_table(roi, P, names, p_min, gap_min, how="mean"):
    rows = []
    for iso, g in roi.groupby("isolate_id", sort=True):
        idx = g.index.values
        if how == "mean":
            p = P[idx].mean(0)
        elif how == "vote":
            votes = np.bincount(P[idx].argmax(1), minlength=len(names)).astype(float)
            p = votes / votes.sum()
        elif how == "maxconf":
            p = P[idx[P[idx].max(1).argmax()]]
        o = np.argsort(-p)
        r = g.iloc[0]
        rows.append({
            "isolate_id": iso, "reference": r["reference"], "train_class": r["train_class"],
            "in_panel": int(r["in_panel"]), "method": r.get("method", ""), "n_roi": len(idx),
            "pred": names[o[0]], "p_top1": float(p[o[0]]),
            "pred2": names[o[1]], "p_top2": float(p[o[1]]),
            "pred3": names[o[2]] if len(o) > 2 else "", "p_top3": float(p[o[2]]) if len(o) > 2 else np.nan,
            **{f"p::{n}": float(v) for n, v in zip(names, p)},
        })
    d = pd.DataFrame(rows)
    d["gap"] = d["p_top1"] - d["p_top2"]
    d["flag"] = ((d["p_top1"] < p_min) | (d["gap"] < gap_min)).astype(int)
    tc = d["train_class"]
    d["top1_ok"] = (d["pred"] == tc).astype(int)
    d["top2_ok"] = ((d["pred"] == tc) | (d["pred2"] == tc)).astype(int)
    d["genus_ok"] = (d["pred"].map(genus) == tc.map(genus)).astype(int)
    d["group_ok"] = (d["pred"].map(clin_group) == tc.map(clin_group)).astype(int)
    for c in ["top1_ok", "top2_ok", "genus_ok", "group_ok"]:
        d.loc[d["in_panel"] == 0, c] = np.nan
    return d


def class_metrics(ip, names):
    rows = []
    for c in sorted(set(ip["train_class"]) | set(ip["pred"])):
        tp = int(((ip["train_class"] == c) & (ip["pred"] == c)).sum())
        n_true = int((ip["train_class"] == c).sum())
        n_pred = int((ip["pred"] == c).sum())
        sens = tp / n_true if n_true else np.nan
        ppv = tp / n_pred if n_pred else np.nan
        lo, hi = exact_ci(tp, n_true) if n_true else (np.nan, np.nan)
        f1 = 2 * sens * ppv / (sens + ppv) if n_true and n_pred and (sens + ppv) > 0 else (0.0 if n_true else np.nan)
        rows.append({"类别": c, "菌株数": n_true, "判对": tp, "敏感度": sens,
                     "敏感度95CI": f"{lo:.2f}–{hi:.2f}" if n_true else "", "预测为该类": n_pred,
                     "PPV": ppv, "F1": f1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="前瞻临床验证评估（详细用法见脚本开头注释）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--roi_dir", default=DEFAULT_ROI)
    ap.add_argument("--table", default=None)
    ap.add_argument("--out", default="ext_eval_clinical")
    ap.add_argument("--lock", default=DEFAULT_LOCK)
    ap.add_argument("--write_lock", action="store_true")
    ap.add_argument("--no_lock", action="store_true")
    ap.add_argument("--p_min", type=float, default=0.70)
    ap.add_argument("--gap_min", type=float, default=0.20)
    a = ap.parse_args()

    model_dirs = [resolve_model(m) for m in a.models]
    if a.write_lock:
        write_lock(a.lock, model_dirs, a.p_min, a.gap_min)
        return
    locked_at = "未核对（--no_lock，仅供调试）" if a.no_lock else check_lock(a.lock, model_dirs, a.p_min, a.gap_min)

    roi_dir = os.path.expanduser(a.roi_dir)
    table = a.table or os.path.join(roi_dir, "roi_clinical_table.csv")
    if not os.path.exists(table):
        sys.exit(f"找不到 {table}\n请先揭盲：python 03_roi_clinical.py --src <图片文件夹> --unblind --answers answers_clinical.csv")
    roi = pd.read_csv(table, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    roi = roi[roi["roi_file"] != ""].reset_index(drop=True)
    miss_ans = roi[roi["in_panel"] == ""]
    if len(miss_ans):
        sys.exit(f"有 {miss_ans['isolate_id'].nunique()} 株没有参考鉴定："
                 f"{'、'.join(sorted(miss_ans['isolate_id'].unique()))}\n请补全答案表后重新揭盲。")
    roi["in_panel"] = roi["in_panel"].astype(int)
    roi["path"] = [p if os.path.exists(p) else os.path.join(roi_dir, f)
                   for p, f in zip(roi["roi_path"], roi["roi_file"])]
    lost = roi[~roi["path"].map(os.path.exists)]
    if len(lost):
        sys.exit(f"{len(lost)} 个 ROI 文件找不到，例如 {lost['path'].iloc[0]}")

    # ---- 预测
    print(f"ROI {len(roi)} 个，菌株 {roi['isolate_id'].nunique()} 株；开始预测 …")
    P, names, per_model = predict(model_dirs, roi["path"].tolist())
    bad = sorted(set(roi.loc[roi["in_panel"] == 1, "train_class"]) - set(names))
    if bad:
        sys.exit("答案表里的 train_class 不在模型类别中（检查拼写，须与 taxonomy_map.csv 完全一致）：\n  "
                 + "\n  ".join(bad))

    out = resolve_out(a.out)
    os.makedirs(out, exist_ok=True)

    # ---- ROI 级
    o = np.argsort(-P, 1)
    roi["pred"] = [names[i] for i in o[:, 0]]
    roi["p_top1"] = P.max(1)
    roi["ok"] = np.where(roi["in_panel"] == 1, (roi["pred"] == roi["train_class"]).astype(float), np.nan)
    roi.drop(columns=["path"]).assign(**{f"p::{n}": P[:, j] for j, n in enumerate(names)}) \
        .to_csv(os.path.join(out, "per_roi.csv"), index=False, encoding="utf-8-sig")

    # ---- 株级（预先设定：平均）
    iso = isolate_table(roi, P, names, a.p_min, a.gap_min, "mean")
    cols = ["isolate_id", "reference", "train_class", "in_panel", "method", "n_roi", "pred", "p_top1",
            "pred2", "p_top2", "pred3", "p_top3", "gap", "flag", "top1_ok", "top2_ok", "genus_ok", "group_ok"]
    iso[cols + [c for c in iso.columns if c.startswith("p::")]] \
        .to_csv(os.path.join(out, "per_isolate.csv"), index=False, encoding="utf-8-sig")

    ip = iso[iso["in_panel"] == 1]
    oop = iso[iso["in_panel"] == 0]
    n = len(ip)
    k1, k2 = int(ip["top1_ok"].sum()), int(ip["top2_ok"].sum())
    kg, kc = int(ip["genus_ok"].sum()), int(ip["group_ok"].sum())
    cm = class_metrics(ip, names) if n else pd.DataFrame()
    present = cm[cm["菌株数"] > 0] if n else cm
    macro_f1 = float(present["F1"].mean()) if n else float("nan")
    bal_acc = float(present["敏感度"].mean()) if n else float("nan")
    unf, flg = ip[ip["flag"] == 0], ip[ip["flag"] == 1]

    if n:
        conf = pd.crosstab(pd.Categorical(ip["train_class"], categories=sorted(set(ip["train_class"]))),
                           pd.Categorical(ip["pred"], categories=names), dropna=False)
        conf = conf.loc[:, conf.sum(axis=0) > 0]
        conf.index.name, conf.columns.name = "参考\\预测", None
        conf.to_csv(os.path.join(out, "confusion.csv"), encoding="utf-8-sig")
        # ROI 级混淆矩阵（集内菌株的全部 ROI；株数少时比株级更能看出模型把哪类结构认错）
        ri0 = roi[roi["in_panel"] == 1]
        conf_roi = pd.crosstab(pd.Categorical(ri0["train_class"], categories=sorted(set(ri0["train_class"]))),
                               pd.Categorical(ri0["pred"], categories=names), dropna=False)
        conf_roi = conf_roi.loc[:, conf_roi.sum(axis=0) > 0]
        conf_roi.index.name, conf_roi.columns.name = "参考\\预测", None
        conf_roi.to_csv(os.path.join(out, "confusion_roi.csv"), encoding="utf-8-sig")
        srt = ip.sort_values("p_top1", ascending=False).reset_index(drop=True)
        cov = pd.DataFrame({"coverage": (np.arange(1, n + 1)) / n,
                            "threshold_p_top1": srt["p_top1"],
                            "accuracy": srt["top1_ok"].cumsum() / np.arange(1, n + 1)})
        cov.to_csv(os.path.join(out, "coverage_curve.csv"), index=False, encoding="utf-8-sig")

    # 开放集：分数 = 1 − top1 概率，越高越像集外
    au = auroc(1 - oop["p_top1"].values, 1 - ip["p_top1"].values)

    L, J = [], {}
    L.append("===== 前瞻临床验证：结果 =====")
    L.append(f"评估时间 {time.strftime('%Y-%m-%d %H:%M')}；锁定 {locked_at}")
    L.append("模型：" + "、".join(os.path.basename(d) for d in model_dirs))
    L.append(f"判读规则：株级 = ROI 概率平均；top1 < {a.p_min} 或 top1 − top2 < {a.gap_min} → 需复核")
    L.append(f"菌株：集内 {n} 株，集外 {len(oop)} 株；ROI {len(roi)} 个"
             f"（每株中位数 {iso['n_roi'].median():.0f}，范围 {iso['n_roi'].min()}–{iso['n_roi'].max()}）")
    L.append("（截图阶段被跳过、没有 ROI 的菌株见 roi_clinical 文件夹里的 unblind_summary.txt，论文需一并报告）")

    L.append("\n----- 主要终点（集内菌株） -----")
    L.append(f"top1 准确率          {fmt(k1, n)}")
    L.append("\n----- 次要终点 -----")
    L.append(f"top2 准确率          {fmt(k2, n)}")
    L.append(f"属一级准确率         {fmt(kg, n)}")
    L.append(f"临床类群一级准确率   {fmt(kc, n)}   （曲霉属合一、毛霉目合一、其余按属）")
    L.append(f"macro-F1             {macro_f1:.3f}   （仅计参考集中出现的类别）")
    L.append(f"平衡准确率           {bal_acc:.3f}")
    L.append(f"复核标记比例         {fmt(len(flg), n)}")
    L.append(f"未标记（可直接报告）的准确率  {fmt(int(unf['top1_ok'].sum()), len(unf))}")
    L.append(f"标记复核的准确率     {fmt(int(flg['top1_ok'].sum()), len(flg))}")
    L.append(f"标记复核中 top2 含正确答案   {fmt(int(flg['top2_ok'].sum()), len(flg))}")
    L.append(f"ECE（10 箱）         {ece(ip['p_top1'], ip['top1_ok']):.3f}")
    L.append("\n----- 集外菌种（开放集） -----")
    L.append(f"被标记复核的比例     {fmt(int(oop['flag'].sum()), len(oop))}")
    L.append(f"开放集 AUROC         {au:.3f}   （1 − top1 概率区分集外/集内；n 集外={len(oop)}）")
    if len(oop):
        L.append("集外菌株的预测：")
        for _, r in oop.iterrows():
            L.append(f"  {r['isolate_id']:<10s}{r['reference']:<32s}→ {r['pred']}（{r['p_top1']:.2f}）"
                     f"{'  已标记' if r['flag'] else '  未标记 ⚠'}")

    if n:
        L.append("\n----- 每类（集内菌株） -----")
        show = cm.copy()
        for c in ["敏感度", "PPV", "F1"]:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        L.append(show.to_string(index=False))
        sh = lambda c: str(c).replace("Aspergillus ", "A. ").split(" (")[0]
        for lab, m in [("株级", conf), ("ROI 级", conf_roi)]:
            L.append(f"\n----- 混淆矩阵（{lab}；行 = 参考，列 = 预测） -----")
            L.append(m.rename(index=sh, columns=sh).to_string())
        L.append("\n----- 覆盖率–准确率（按株级 top1 概率从高到低） -----")
        for c in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
            m = max(1, int(round(c * n)))
            L.append(f"  覆盖 {c:.0%}（{m} 株）：准确率 {cov['accuracy'].iloc[m - 1]:.3f}，"
                     f"对应阈值 p_top1 ≥ {cov['threshold_p_top1'].iloc[m - 1]:.2f}")
        L.append("\n----- 判错的集内菌株 -----")
        for _, r in ip[ip["top1_ok"] == 0].iterrows():
            L.append(f"  {r['isolate_id']:<10s}{r['train_class']:<42s}→ {r['pred']}（{r['p_top1']:.2f}）"
                     f" 第2 {r['pred2']}（{r['p_top2']:.2f}）{'  已标记' if r['flag'] else '  未标记 ⚠'}")

    # ---- 辅助分析
    L.append("\n===== 辅助分析（仅供讨论） =====")
    ri = roi[roi["in_panel"] == 1]
    if len(ri):
        L.append(f"ROI 级 top1 准确率   {fmt(int(ri['ok'].sum()), len(ri))}")
        for col, lab in [("structure", "结构类型"), ("stage", "成熟度")]:
            t = ri.groupby(col)["ok"].agg(["mean", "size"])
            L.append(f"按{lab}：" + "；".join(f"{k} {v['mean']:.3f}（n={int(v['size'])}）" for k, v in t.iterrows()))
        if len(per_model) > 1:
            L.append("各模型单独（ROI 级）：" + "；".join(
                f"{os.path.basename(d)} {np.mean([names[i] == t for i, t in zip(pm[ri.index].argmax(1), ri['train_class'])]):.3f}"
                for d, pm in zip(model_dirs, per_model)))
        L.append("株级汇总方式的敏感性分析（top1；主要结果以'平均'为准）：")
        for how, lab in [("mean", "平均（预设）"), ("vote", "多数投票"), ("maxconf", "最高置信 ROI")]:
            t = isolate_table(roi, P, names, a.p_min, a.gap_min, how)
            t = t[t["in_panel"] == 1]
            L.append(f"  {lab:<12s}{fmt(int(t['top1_ok'].sum()), len(t))}")

    txt = "\n".join(L)
    with open(os.path.join(out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    J = {"n_in_panel": n, "n_out_of_panel": int(len(oop)), "n_roi": int(len(roi)),
         "top1": [k1, n, *exact_ci(k1, n)], "top2": [k2, n, *exact_ci(k2, n)],
         "genus": [kg, n, *exact_ci(kg, n)], "group": [kc, n, *exact_ci(kc, n)],
         "macro_f1": macro_f1, "balanced_accuracy": bal_acc,
         "flag_rate": [int(len(flg)), n], "unflagged_acc": [int(unf["top1_ok"].sum()), int(len(unf))],
         "flagged_acc": [int(flg["top1_ok"].sum()), int(len(flg))],
         "ece": ece(ip["p_top1"], ip["top1_ok"]), "ood_flagged": [int(oop["flag"].sum()), int(len(oop))],
         "open_set_auroc": au, "p_min": a.p_min, "gap_min": a.gap_min}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, ensure_ascii=False, indent=2, default=float)
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "locked_at": locked_at,
                   "models": [model_fingerprint(d) for d in model_dirs],
                   "table": table, "table_md5": md5(table),
                   "p_min": a.p_min, "gap_min": a.gap_min}, f, ensure_ascii=False, indent=2)
    print("\n" + txt)
    print(f"\n已写出 {out}/  (metrics.txt, metrics.json, per_isolate.csv, per_roi.csv, confusion.csv, confusion_roi.csv, coverage_curve.csv, run_info.json)")


if __name__ == "__main__":
    main()
