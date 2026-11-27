#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_colony_train.py — 菌落分支训练与株级交叉验证评估

读取 01_colony_preprocess.py 的输出（output/01_trainsets/colony_preprocessed/），在株水平做交叉验证，比较：
  方案 A  菌落区域 CIELAB 颜色统计 + 菌落直径 → 多项逻辑回归        （主分析候选）
  方案 B  冻结预训练网络特征（8 向旋转/翻转平均）→ 多项逻辑回归     （主分析候选）
  方案 C  端到端微调                                                （--run_c，仅探索性对照）
预先设定的选择规则：A、B 中 OOF 株级 NLL 最低者 → 温度缩放 → 全部株重训 → 锁定。

交叉验证
  · 以"株"为单位；预处理重复检测中匹配内点 ≥ --dup_inliers 的照片对视为同一平皿，合并到同一 CV 组
  · A、B：留一组交叉验证（Leave-One-Group-Out）；正则强度在训练折内部再做分层交叉验证选择
  · C：分层分组 5 折（计算量原因）
  · 株数 < --min_strains 的类别自动剔除（本轮 Nidulantes 只有 1 株会被剔除）

安装（清华源；torch/torchvision 沿用现有环境）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn scipy matplotlib joblib pandas

运行
  python 02_colony_train.py                                   # 默认读 output/01_trainsets/colony_preprocessed
  python 02_colony_train.py --backbone densenet121            # 换方案 B 的骨干网络
        可选：swin_t / densenet121 / resnet50 / efficientnet_b0 / convnext_tiny / vit_b_16
  python 02_colony_train.py --run_c                           # 加跑方案 C（较慢）
  python 02_colony_train.py --classes Fumigati,Flavi,Nigri,Terrei
  python 02_colony_train.py --weights /path/swin_t.pth        # 离线权重

输出（默认 Audit/output/02_training_runs/colony/<时间戳>/；终端日志在 output/06_logs/）
  metrics.txt                   各方案株级 OOF 结果（中文，可直接用于论文）
  oof_A.csv / oof_B.csv / ...   每株一行：真实类别、各类概率、预测、CV 组
  confusion_*.png               混淆矩阵
  coef_A.png                    方案 A 系数热图（颜色特征 × 类别）
  final_model.joblib            选定方案在全部株上重训的模型 + 温度 T + 类别 + 特征说明
  locked_config.json            选择结果、参数、文件哈希（前瞻验证前锁定）
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cv2
    from PIL import Image
    from scipy.optimize import minimize_scalar
    from scipy.stats import beta
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    import joblib
except ImportError as e:
    sys.exit(f"缺少依赖：{e}\n请运行：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
             "scikit-learn scipy matplotlib joblib pandas opencv-python pillow")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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



VERSION = "02_colony_train v1.2 (2026-10-23)"
# 方案 B/C 可用的骨干网络（均为 torchvision ImageNet-1K 预训练；输入统一 224×224）
BACKBONES = ["swin_t", "densenet121", "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
ORDER = ["Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes"]
GRAY8 = 128            # 02 中遮罩/平皿外填充值
C_GRID_A = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
C_GRID_B = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]
SEED = 20261020
T_MIN = 0.5            # 温度缩放下限


# ====================================================================== 数据
def load_manifest(pre: Path, args):
    m = pd.read_csv(pre / "manifest.csv")
    m = m[(m["view"] == "O")].copy()
    n0 = m["strain_id"].nunique()
    excl = m[~m["main_analysis"].astype(bool)]
    m = m[m["main_analysis"].astype(bool)]
    if args.registry:
        reg = pd.read_excel(args.registry, sheet_name="登记表")
        drop = set(reg.loc[reg["状态"].astype(str).str.strip() == "剔除", "株号"].astype(str))
        m = m[~m["strain_id"].isin(drop)]
    log = [f"正面照来自 {n0} 株；主分析排除 {excl['strain_id'].nunique()} 株"
           f"（{', '.join(sorted(excl['strain_id'].unique()))}）" if len(excl) else f"正面照来自 {n0} 株"]

    counts = m.groupby("group")["strain_id"].nunique()
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
    else:
        classes = [c for c in ORDER if counts.get(c, 0) >= args.min_strains]
    dropped = {c: int(counts.get(c, 0)) for c in counts.index if c not in classes}
    if dropped:
        log.append("株数不足、本轮不纳入的类别：" + "，".join(f"{c}（{n} 株）" for c, n in dropped.items()))
    m = m[m["group"].isin(classes)].copy()

    # CV 组：同一株 + 重复平皿合并（并查集）
    parent = {s: s for s in m["strain_id"].unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = []
    dpath = pre / "duplicates_check.csv"
    if dpath.exists() and args.dup_inliers > 0:
        d = pd.read_csv(dpath)
        for _, r in d[d["inliers"] >= args.dup_inliers].iterrows():
            a, b = r["strain_a"], r["strain_b"]
            if a in parent and b in parent and find(a) != find(b):
                parent[find(a)] = find(b)
                merged.append(f"{a}+{b}({int(r['inliers'])})")
    m["cv_group"] = m["strain_id"].map(find)
    if merged:
        log.append(f"疑似同一平皿，合并为同一 CV 组（不改株号）：{'，'.join(merged)}")
    return m, classes, log


def read_crop(pre: Path, rel):
    return np.asarray(Image.open(pre / rel).convert("RGB"), dtype=np.float32) / 255.0


def crops_of(row):
    out = {}
    for p in str(row["crops"]).split(";"):
        name = p.rsplit("__", 1)[-1].rsplit(".", 1)[0]
        out[name] = p
    return out


# ====================================================================== 方案 A：颜色特征
def colony_mask_edge(k, size=384):
    """边缘块中菌落内部像素：块中心位于菌落边缘（角度 60k°），块边长 0.6r → 菌落半径 = size/0.6"""
    th = math.radians(60 * k)
    R = size / 0.6
    c = size / 2
    cx, cy = c - R * math.cos(th), c - R * math.sin(th)
    yy, xx = np.ogrid[:size, :size]
    return np.hypot(xx - cx, yy - cy) < R


def lab_stats(px):
    """px: N×3 sRGB 0–1 → L*, a*, b* 的中位数和四分位距"""
    if len(px) < 50:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    q1, q2, q3 = np.percentile(lab, [25, 50, 75], axis=0)
    return np.concatenate([q2, q3 - q1])


def not_gray(img):
    return ~(np.abs(img * 255 - GRAY8).max(2) <= 1.5)


FEAT_A = ([f"中心_{c}_{s}" for s in ("中位", "IQR") for c in ("L", "a", "b")]
          + [f"边缘_{c}_{s}" for s in ("中位", "IQR") for c in ("L", "a", "b")]
          + ["菌落直径mm"])


def features_A(pre: Path, row):
    cr = crops_of(row)
    c = read_crop(pre, cr["center"])
    fc = lab_stats(c[not_gray(c)])
    edge_px = []
    for k in range(6):
        if f"edge{k}" in cr:
            e = read_crop(pre, cr[f"edge{k}"])
            mk = colony_mask_edge(k, e.shape[0]) & not_gray(e)
            edge_px.append(e[mk])
    fe = lab_stats(np.concatenate(edge_px)) if edge_px else None
    if fc is None:
        return None
    if fe is None:            # 菌落铺满平皿等无边缘块：用中心值代替
        fe = fc.copy()
    return np.concatenate([fc, fe, [float(row["colony_diam_mm"])]])


# ====================================================================== 方案 B：冻结骨干特征
def get_backbone(name, weights_path, device):
    import torch
    import torchvision.models as tvm
    if name == "swin_t":
        w = None if weights_path else tvm.Swin_T_Weights.IMAGENET1K_V1
        net = tvm.swin_t(weights=w)
        net.head = torch.nn.Identity()
    elif name == "densenet121":
        w = None if weights_path else tvm.DenseNet121_Weights.IMAGENET1K_V1
        net = tvm.densenet121(weights=w)
        net.classifier = torch.nn.Identity()
    elif name == "resnet50":
        w = None if weights_path else tvm.ResNet50_Weights.IMAGENET1K_V2
        net = tvm.resnet50(weights=w)
        net.fc = torch.nn.Identity()
    elif name == "efficientnet_b0":                       # 输出 1280 维
        w = None if weights_path else tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        net = tvm.efficientnet_b0(weights=w)
        net.classifier = torch.nn.Identity()              # 去掉 Dropout + Linear，保留全局平均池化
    elif name == "convnext_tiny":                         # 输出 768 维
        w = None if weights_path else tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        net = tvm.convnext_tiny(weights=w)
        net.classifier[2] = torch.nn.Identity()           # 保留 LayerNorm2d + Flatten，只去 Linear
    elif name == "vit_b_16":                              # 输出 768 维（CLS token）；输入必须 224×224
        w = None if weights_path else tvm.ViT_B_16_Weights.IMAGENET1K_V1
        net = tvm.vit_b_16(weights=w)
        net.heads = torch.nn.Identity()
    else:
        sys.exit(f"不支持的骨干网络：{name}（可选：{', '.join(BACKBONES)}）")
    if weights_path and weights_path != "random":
        sd = torch.load(weights_path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"  载入权重 {weights_path}（缺失 {len(missing)}，多余 {len(unexpected)} 个参数，分类头已移除）")
    return net.eval().to(device)


def pick_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor_batch(imgs, size=224):
    import torch
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = np.stack([(cv2.resize(i, (size, size), interpolation=cv2.INTER_AREA) - mean) / std for i in imgs])
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def dihedral(img):
    out = []
    for f in (img, img[:, ::-1]):
        for k in range(4):
            out.append(np.rot90(f, k))
    return out


def extract_B(pre: Path, df, backbone, weights_path, cache_dir: Path):
    """每张图：中心块特征 ‖ 边缘块特征均值（每块 8 向旋转/翻转平均）。带缓存。"""
    import torch
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = backbone + ("_" + hashlib.md5(str(weights_path).encode()).hexdigest()[:6] if weights_path else "")
    cpath = cache_dir / f"featB_{tag}.npz"
    cache = dict(np.load(cpath, allow_pickle=True)) if cpath.exists() else {}
    need = [r for _, r in df.iterrows() if r["file"] not in cache]
    if need:
        device = pick_device()
        print(f"  方案 B：用 {backbone} 提取 {len(need)} 张图的特征（设备 {device}）…")
        net = get_backbone(backbone, weights_path, device)
        with torch.no_grad():
            for i, r in enumerate(need, 1):
                cr = crops_of(r)
                feats = {}
                for name, rel in cr.items():
                    if name == "whole":
                        continue
                    x = to_tensor_batch(dihedral(read_crop(pre, rel))).to(device)
                    feats[name] = net(x).float().mean(0).cpu().numpy()
                fc = feats["center"]
                edges = [v for k, v in feats.items() if k.startswith("edge")]
                fe = np.mean(edges, 0) if edges else fc
                cache[r["file"]] = np.concatenate([fc, fe])
                if i % 10 == 0:
                    print(f"    {i}/{len(need)}")
        np.savez(cpath, **cache)
    return np.stack([cache[f] for f in df["file"]])


# ====================================================================== 株级 CV（A、B 通用）
def strain_table(df, X):
    """同一株多张图 → 特征取平均"""
    g = defaultdict(list)
    for i, s in enumerate(df["strain_id"]):
        g[s].append(i)
    sids = list(g)
    Xs = np.stack([X[g[s]].mean(0) for s in sids])
    meta = df.groupby("strain_id").first().loc[sids, ["group", "cv_group"]].reset_index()
    return sids, Xs, meta


def fit_lr(Xtr, ytr, c_grid, n_classes):
    """训练折内：标准化 + 分层 3 折内部交叉验证选 C（按 NLL）→ 全训练折重训"""
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    n_min = np.bincount(ytr, minlength=n_classes).min()
    best_c = c_grid[len(c_grid) // 2]
    if n_min >= 2:
        k = int(min(3, n_min))
        skf = StratifiedKFold(k, shuffle=True, random_state=SEED)
        scores = []
        for c in c_grid:
            nll = []
            for a, b in skf.split(Z, ytr):
                m = LogisticRegression(C=c, max_iter=5000).fit(Z[a], ytr[a])
                p = np.full((len(b), n_classes), 1e-6)
                p[:, m.classes_] = m.predict_proba(Z[b])
                nll.append(-np.log(np.clip(p[np.arange(len(b)), ytr[b]], 1e-6, 1)).mean())
            scores.append(np.mean(nll))
        best_c = c_grid[int(np.argmin(scores))]
    m = LogisticRegression(C=best_c, max_iter=5000).fit(Z, ytr)
    return sc, m, best_c


def predict_lr(sc, m, X, n_classes):
    p = np.full((len(X), n_classes), 0.0)
    p[:, m.classes_] = m.predict_proba(sc.transform(X))
    lg = np.full((len(X), n_classes), -30.0)
    lg[:, m.classes_] = m.decision_function(sc.transform(X)) if len(m.classes_) > 2 else np.c_[
        -m.decision_function(sc.transform(X)), m.decision_function(sc.transform(X))]
    return p, lg


def run_cv_lr(Xs, y, groups, c_grid, n_classes, cols=None):
    X = Xs if cols is None else Xs[:, cols]
    P = np.zeros((len(y), n_classes))
    LG = np.zeros((len(y), n_classes))
    cs = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc, m, c = fit_lr(X[tr], y[tr], c_grid, n_classes)
        P[te], LG[te] = predict_lr(sc, m, X[te], n_classes)
        cs.append(c)
    return P, LG, cs


# ====================================================================== 方案 C：端到端微调（探索性）
def run_C(pre, df, y_img, classes, backbone, weights_path, epochs, out):
    import torch
    import torch.nn as nn
    device = pick_device()
    sids = df["strain_id"].values
    groups = df["cv_group"].values
    n_classes = len(classes)
    items = []  # (图片索引, 截图路径)
    for i, (_, r) in enumerate(df.iterrows()):
        for name, rel in crops_of(r).items():
            if name != "whole":
                items.append((i, rel))
    imgs = {rel: read_crop(pre, rel) for _, rel in items}
    rng = np.random.default_rng(SEED)

    def augment(a):
        a = np.rot90(a, rng.integers(4))
        if rng.random() < 0.5:
            a = a[:, ::-1]
        a = a * rng.uniform(0.9, 1.1) + rng.uniform(-0.04, 0.04)      # 亮度/对比度 ±10%
        g = a.mean(2, keepdims=True)
        a = g + (a - g) * rng.uniform(0.95, 1.05)                       # 饱和度 ±5%，不动色相
        return np.clip(a, 0, 1).astype(np.float32)

    P_img = np.zeros((len(df), n_classes))
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(sgkf.split(np.zeros(len(df)), y_img, groups), 1):
        print(f"  方案 C 第 {fold}/5 折 …")
        net = get_backbone(backbone, weights_path, device)
        dim = net(torch.zeros(1, 3, 224, 224, device=device)).shape[1]
        head = nn.Linear(dim, n_classes).to(device)
        model = nn.Sequential(net, head).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
        tr_set = set(tr)
        tr_items = [(i, rel) for i, rel in items if i in tr_set]
        # 株平衡采样：每个 epoch 每株抽相同数量截图
        by_s = defaultdict(list)
        for i, rel in tr_items:
            by_s[sids[i]].append((i, rel))
        for ep in range(epochs):
            batch = []
            for s, lst in by_s.items():
                for j in rng.choice(len(lst), 8, replace=True):
                    batch.append(lst[j])
            rng.shuffle(batch)
            for b0 in range(0, len(batch), 32):
                bb = batch[b0:b0 + 32]
                x = to_tensor_batch([augment(imgs[rel]) for _, rel in bb]).to(device)
                t = torch.tensor([y_img[i] for i, _ in bb], device=device)
                opt.zero_grad()
                loss = lossf(model(x), t)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            for i in te:
                rels = [rel for j, rel in items if j == i]
                x = to_tensor_batch([imgs[r] for r in rels]).to(device)
                P_img[i] = torch.softmax(model(x), 1).mean(0).cpu().numpy()
    # 图片 → 株
    sid_list = list(dict.fromkeys(sids))
    P = np.stack([P_img[sids == s].mean(0) for s in sid_list])
    return sid_list, P


# ====================================================================== 评估
def cp_ci(k, n, a=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def ece(P, y, bins=5):
    conf = P.max(1)
    acc = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1)).mean())


def fit_temperature(LG, y):
    def f(T):
        z = LG / T
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return nll(p, y)
    # 株数少时 OOF 常接近完全可分，T 会被推向 0（过度自信）；下限设 0.5 防止过度锐化
    r = minimize_scalar(f, bounds=(T_MIN, 10), method="bounded")
    return float(r.x)


def softmax_T(LG, T):
    z = LG / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def evaluate(P, y, classes, title):
    n = len(y)
    pred = P.argmax(1)
    top2 = np.argsort(-P, 1)[:, :2]
    k1 = int((pred == y).sum())
    k2 = int(sum(y[i] in top2[i] for i in range(n)))
    lines = [f"----- {title} -----"]
    lo, hi = cp_ci(k1, n)
    lines.append(f"top1 准确率          {k1 / n:.3f}（{k1}/{n}，95% CI {lo:.3f}–{hi:.3f}）")
    lo, hi = cp_ci(k2, n)
    lines.append(f"top2 准确率          {k2 / n:.3f}（{k2}/{n}，95% CI {lo:.3f}–{hi:.3f}）")
    sens = []
    for c in range(len(classes)):
        m = y == c
        sens.append((pred[m] == c).mean() if m.any() else np.nan)
    lines.append(f"平衡准确率           {np.nanmean(sens):.3f}")
    lines.append(f"macro-F1             {f1_score(y, pred, average='macro', labels=list(range(len(classes)))):.3f}")
    lines.append(f"NLL                  {nll(P, y):.3f}")
    lines.append(f"ECE（5 箱）          {ece(P, y):.3f}")
    lines.append("每类：")
    lines.append(f"  {'类别':<12}{'株数':>5}{'判对':>5}{'敏感度':>8}  {'95% CI':<12}{'预测为该类':>8}{'PPV':>7}")
    for c, name in enumerate(classes):
        m = y == c
        kk = int((pred[m] == c).sum())
        lo, hi = cp_ci(kk, int(m.sum()))
        npred = int((pred == c).sum())
        ppv = kk / npred if npred else float("nan")
        lines.append(f"  {name:<12}{int(m.sum()):>5}{kk:>5}{kk / max(m.sum(), 1):>8.2f}  {lo:.2f}–{hi:.2f}   {npred:>6}{ppv:>8.2f}")
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    lines.append("混淆矩阵（行 = 真实，列 = 预测）：")
    lines.append("  " + " " * 12 + "".join(f"{c[:8]:>10}" for c in classes))
    for i, c in enumerate(classes):
        lines.append(f"  {c:<12}" + "".join(f"{v:>10}" for v in cm[i]))
    # 重点关注的两对
    for a, b in [("Fumigati", "Terrei"), ("Flavi", "Nigri")]:
        if a in classes and b in classes:
            ia, ib = classes.index(a), classes.index(b)
            lines.append(f"  {a}→{b} 误判 {cm[ia, ib]}，{b}→{a} 误判 {cm[ib, ia]}")
    return lines, cm, {"top1": k1 / n, "top2": k2 / n, "bal_acc": float(np.nanmean(sens)),
                       "nll": nll(P, y), "ece": ece(P, y)}


def plot_cm(cm, classes, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_coef(model, scaler, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names_en = ([f"center {c} {s}" for s in ("median", "IQR") for c in ("L*", "a*", "b*")]
                + [f"edge {c} {s}" for s in ("median", "IQR") for c in ("L*", "a*", "b*")] + ["diameter (mm)"])
    W = np.zeros((len(classes), len(names_en)))
    W[model.classes_] = model.coef_ if len(model.classes_) > 2 else np.r_[-model.coef_, model.coef_]
    v = np.abs(W).max()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xticks(range(len(names_en)), names_en, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="standardized coefficient")
    ax.set_title("Method A: colour/diameter coefficients (final model)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


# ====================================================================== 主程序
def main():
    ap = argparse.ArgumentParser(description="菌落分支训练与株级交叉验证")
    ap.add_argument("--pre", default="colony_preprocessed", help="短名字 → output/01_trainsets/<名字>（含 manifest.csv）")
    ap.add_argument("--out", default=None, help="默认 output/02_training_runs/colony/<时间戳>；短名字 → output/02_training_runs/colony/<名字>")
    ap.add_argument("--registry", default=None, help="菌株登记表.xlsx（可选，剔除 状态=剔除 的株）")
    ap.add_argument("--classes", default=None, help="逗号分隔；默认自动选株数 ≥ --min_strains 的组")
    ap.add_argument("--min_strains", type=int, default=3)
    ap.add_argument("--dup_inliers", type=int, default=300, help="≥ 此内点数的重复照片合并 CV 组；0 = 不合并")
    ap.add_argument("--backbone", default="swin_t", choices=BACKBONES)
    ap.add_argument("--weights", default=None, help="离线权重路径；'random' = 随机初始化（仅测试流程）")
    ap.add_argument("--skip_b", action="store_true", help="不跑方案 B（无 torch 时）")
    ap.add_argument("--run_c", action="store_true", help="加跑方案 C（探索性，较慢）")
    ap.add_argument("--epochs_c", type=int, default=15)
    args = ap.parse_args()
    start_log("02_colony_train")

    pre = resolve_in("01_trainsets", args.pre)
    if not (pre / "manifest.csv").exists():
        sys.exit(f"找不到 {pre / 'manifest.csv'}，请先运行 01_colony_preprocess.py build")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out = resolve_in("02_training_runs", f"colony/{args.out or stamp}")
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    df, classes, log = load_manifest(pre, args)
    if len(classes) < 2:
        sys.exit("可用类别少于 2 个，无法训练")
    cidx = {c: i for i, c in enumerate(classes)}
    K = len(classes)

    report = [f"===== 菌落分支：株级交叉验证结果 =====",
              f"运行时间 {dt.datetime.now():%Y-%m-%d %H:%M}；{VERSION}",
              f"数据：{pre}"] + log
    counts = df.groupby("group")["strain_id"].nunique()
    report.append("纳入类别与株数：" + "，".join(f"{c} {counts.get(c, 0)}" for c in classes)
                  + f"；共 {df['strain_id'].nunique()} 株、{df['cv_group'].nunique()} 个 CV 组")
    report.append("交叉验证：A/B 为留一组交叉验证（LOGO），正则强度在训练折内分层 3 折选择（按 NLL）")
    report.append("预设选择规则：A、B 中 OOF NLL 最低者进入融合；其后温度缩放并以全部株重训")
    report.append("")

    # ---------------- 方案 A ----------------
    print("方案 A：计算颜色特征 …")
    XA_img, keep = [], []
    for i, (_, r) in enumerate(df.iterrows()):
        f = features_A(pre, r)
        if f is not None:
            XA_img.append(f)
            keep.append(i)
    if len(keep) < len(df):
        report.append(f"方案 A：{len(df) - len(keep)} 张图菌落像素不足，跳过")
    dfA = df.iloc[keep].reset_index(drop=True)
    sids, XA, meta = strain_table(dfA, np.stack(XA_img))
    y = meta["group"].map(cidx).values
    groups = meta["cv_group"].values

    results = {}
    P, LG, cs = run_cv_lr(XA, y, groups, C_GRID_A, K)
    results["A"] = (P, LG)
    lines, cm, sm = evaluate(P, y, classes, "方案 A：菌落颜色（CIELAB）+ 直径 → 逻辑回归【主分析候选】")
    report += lines + [f"  所选正则强度 C（各折）：中位数 {np.median(cs):g}", ""]
    plot_cm(cm, classes, "Method A (colour + diameter), LOGO OOF", out / "confusion_A.png")
    summary = {"A": sm}

    # 消融
    abl = {"A_无直径": list(range(12)), "A_仅中心块": list(range(6)) + [12]}
    for name, cols in abl.items():
        Pa, _, _ = run_cv_lr(XA, y, groups, C_GRID_A, K, cols)
        _, _, s2 = evaluate(Pa, y, classes, name)
        summary[name] = s2
        report.append(f"消融 {name}：top1 {s2['top1']:.3f}，NLL {s2['nll']:.3f}")
    report.append("")

    oofA = meta.copy()
    oofA["true"] = [classes[i] for i in y]
    for j, c in enumerate(classes):
        oofA[f"p_{c}"] = P[:, j].round(4)
    oofA["pred"] = [classes[i] for i in P.argmax(1)]
    oofA.to_csv(out / "oof_A.csv", index=False, encoding="utf-8-sig")

    # ---------------- 方案 B ----------------
    if not args.skip_b:
        try:
            XB_img = extract_B(pre, dfA, args.backbone, args.weights, out_path("02_training_runs", "colony/feature_cache"))
            _, XB, _ = strain_table(dfA, XB_img)
            P, LG, cs = run_cv_lr(XB, y, groups, C_GRID_B, K)
            results["B"] = (P, LG)
            lines, cm, sm = evaluate(P, y, classes, f"方案 B：冻结 {args.backbone} 特征 → 逻辑回归【主分析候选】")
            report += lines + [f"  所选正则强度 C（各折）：中位数 {np.median(cs):g}", ""]
            plot_cm(cm, classes, f"Method B (frozen {args.backbone}), LOGO OOF", out / "confusion_B.png")
            summary["B"] = sm
            dim = XB.shape[1] // 2
            Pb, _, _ = run_cv_lr(XB, y, groups, C_GRID_B, K, list(range(dim)))
            _, _, s2 = evaluate(Pb, y, classes, "B_仅中心块")
            summary["B_仅中心块"] = s2
            report.append(f"消融 B_仅中心块：top1 {s2['top1']:.3f}，NLL {s2['nll']:.3f}\n")
            oofB = meta.copy()
            oofB["true"] = oofA["true"]
            for j, c in enumerate(classes):
                oofB[f"p_{c}"] = P[:, j].round(4)
            oofB["pred"] = [classes[i] for i in P.argmax(1)]
            oofB.to_csv(out / "oof_B.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            report.append(f"方案 B 未完成：{type(e).__name__}: {e}")
            report.append("  （如为权重下载失败，可用 --weights 指定本地权重，或 --skip_b 只跑方案 A）\n")
            print(f"⚠ 方案 B 失败：{e}")

    # ---------------- 方案 C（探索性） ----------------
    if args.run_c:
        try:
            y_img = dfA["group"].map(cidx).values
            sid_c, Pc = run_C(pre, dfA, y_img, classes, args.backbone, args.weights, args.epochs_c, out)
            order = [sid_c.index(s) for s in sids]
            Pc = Pc[order]
            lines, cm, sm = evaluate(Pc, y, classes, f"方案 C：端到端微调 {args.backbone}（分层分组 5 折）【探索性】")
            report += lines + [""]
            plot_cm(cm, classes, f"Method C (fine-tuned {args.backbone}), 5-fold", out / "confusion_C.png")
            summary["C"] = sm
        except Exception as e:
            report.append(f"方案 C 未完成：{type(e).__name__}: {e}\n")

    # ---------------- 选择 → 校准 → 全量重训 → 锁定 ----------------
    cand = {k: summary[k]["nll"] for k in ("A", "B") if k in summary}
    sel = min(cand, key=cand.get)
    P_sel, LG_sel = results[sel]
    T = fit_temperature(LG_sel, y)
    P_cal = softmax_T(LG_sel, T)
    report.append("===== 选择与校准 =====")
    report.append("OOF NLL：" + "，".join(f"{k} {v:.3f}" for k, v in cand.items()) + f" → 选定方案 {sel}")
    report.append(f"温度缩放 T = {T:.3f}；校准前 NLL {nll(P_sel, y):.3f}、ECE {ece(P_sel, y):.3f} → "
                  f"校准后 NLL {nll(P_cal, y):.3f}、ECE {ece(P_cal, y):.3f}")
    report.append("（T 在 OOF 预测上拟合；前瞻验证前随模型一并锁定）"
                  + (f"；T 触及下限 {T_MIN}，说明 OOF 几乎完全可分，样本量小时概率可能偏自信" if T <= T_MIN + 1e-3 else ""))

    Xsel = XA if sel == "A" else XB
    grid = C_GRID_A if sel == "A" else C_GRID_B
    sc, model, c_final = fit_lr(Xsel, y, grid, K)
    bundle = {"method": sel, "classes": classes, "scaler": sc, "model": model, "T": T, "C": c_final,
              "features": FEAT_A if sel == "A" else f"{args.backbone} frozen: center‖edge-mean, dihedral TTA",
              "backbone": None if sel == "A" else args.backbone, "weights": args.weights,
              "preprocess_config": json.loads((pre / "preprocess_config.json").read_text(encoding="utf-8"))
              if (pre / "preprocess_config.json").exists() else None,
              "version": VERSION}
    mpath = out / "final_model.joblib"
    joblib.dump(bundle, mpath)
    if "A" in summary:
        scA, mA, _ = fit_lr(XA, y, C_GRID_A, K)
        plot_coef(mA, scA, classes, out / "coef_A.png")
        coef = pd.DataFrame(np.zeros((K, len(FEAT_A))), index=classes, columns=FEAT_A)
        coef.loc[[classes[i] for i in mA.classes_]] = mA.coef_
        coef.round(3).to_csv(out / "coef_A.csv", encoding="utf-8-sig")

    oof_sel = meta.copy()
    oof_sel["true"] = oofA["true"]
    for j, c in enumerate(classes):
        oof_sel[f"p_{c}"] = P_cal[:, j].round(4)
    oof_sel["pred"] = [classes[i] for i in P_cal.argmax(1)]
    oof_sel.to_csv(out / "oof_colony_selected_calibrated.csv", index=False, encoding="utf-8-sig")

    locked = {"locked_at": dt.datetime.now().isoformat(timespec="seconds"), "version": VERSION,
              "selected_method": sel, "selection_rule": "lowest OOF strain-level NLL among A and B",
              "temperature": T, "C_final": c_final, "classes": classes,
              "n_strains": int(len(y)), "n_cv_groups": int(len(set(groups))),
              "summary": summary, "args": vars(args),
              "sha256": {"final_model.joblib": sha256(mpath), "manifest.csv": sha256(pre / "manifest.csv")}}
    (out / "locked_config.json").write_text(json.dumps(locked, ensure_ascii=False, indent=2), encoding="utf-8")
    report.append(f"\n最终模型：{mpath.name}（sha256 {locked['sha256']['final_model.joblib'][:16]}…）")

    txt = "\n".join(report)
    (out / "metrics.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n输出目录：{out}")


if __name__ == "__main__":
    main()
