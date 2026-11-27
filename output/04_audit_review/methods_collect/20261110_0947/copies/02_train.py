#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train.py —— 26 类丝状真菌显微截图分类（PyTorch + Apple MPS）

两种模式：
  python 02_train.py --backbone densenet121 --mode cv     # 5 折交叉验证（按来源分组），用于比较/选择模型
  python 02_train.py --backbone densenet121 --mode full   # 用全部开发集训练最终模型，供外部验证

可选 backbone（全部来自 torchvision，ImageNet 预训练）：
  densenet121  resnet50  efficientnet_b0  convnext_tiny  swin_t  vit_b_16

输入：trainset/trainset_micro.csv、trainset/class_names.json（01_build_trainset.py 生成）
输出：runs/<backbone>_<mode>_<时间>/
  cv 模式：fold_k/best.pt、oof_predictions.csv（每张图的验证概率）、cv_summary.json、per_class.csv
  full 模式：final.pt、train_log.csv

依赖（清华源）：
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision scikit-learn pandas pillow
训练时防止休眠：caffeinate -i python 02_train.py ...
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # MPS 不支持的算子自动回退 CPU

import io
import sys
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
import json
import time
import math
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, classification_report
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# 模型
# =============================================================================
def build_model(name, num_classes, pretrained=True):
    W = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = models.densenet121(weights=W)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=W)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=W)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=W)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name == "swin_t":
        m = models.swin_t(weights=W)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=W)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif name == "resnet18":          # 仅用于快速调试
        m = models.resnet18(weights=W)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的 backbone: {name}")
    return m


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 数据增强（解决尺度、方向、染色、分辨率差异）
# =============================================================================
class RandomZoomOut:
    """把图缩小后用镜像填充回原尺寸：模拟"低倍镜下结构显得更小" """
    def __init__(self, p=0.3, min_scale=0.6):
        self.p, self.min_scale = p, min_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, 1.0)
        small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        arr = np.asarray(small)
        pw, ph = w - small.size[0], h - small.size[1]
        l, t = random.randint(0, pw), random.randint(0, ph)
        arr = np.pad(arr, ((t, ph - t), (l, pw - l), (0, 0)), mode="reflect")
        return Image.fromarray(arr)


class RandomRotate:
    """任意角度旋转，镜像填充四角（菌丝方向没有意义）"""
    def __call__(self, img):
        ang = random.uniform(0, 360)
        w, h = img.size
        pad = int(math.ceil(max(w, h) * 0.21))
        arr = np.pad(np.asarray(img), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        big = Image.fromarray(arr).rotate(ang, resample=Image.BILINEAR)
        return big.crop((pad, pad, pad + w, pad + h))


class RandomLowRes:
    """随机降低分辨率再放大：让"模糊/低分辨率"不再是某一类独有的特征"""
    def __init__(self, p=0.25, lo=48, hi=128):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        t = random.randint(self.lo, self.hi)
        if min(w, h) <= t:
            return img
        s = t / min(w, h)
        return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class RandomJPEG:
    def __init__(self, p=0.25, q=(30, 90)):
        self.p, self.q = p, q

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(*self.q))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomBlur:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))


def make_transforms(size, train):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return transforms.Compose([
        RandomRotate(),
        RandomZoomOut(p=0.3, min_scale=0.6),
        transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.8, 1.25)),   # 多尺度：随机放大局部
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
        RandomLowRes(p=0.25),
        RandomBlur(p=0.2),
        RandomJPEG(p=0.25),
        transforms.ToTensor(),
        norm,
    ])


class ImgDataset(Dataset):
    def __init__(self, df, size, train):
        self.paths = df["path"].tolist()
        self.labels = df["class_idx"].astype(int).tolist()
        self.tf = make_transforms(size, train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i])
        img = ImageOps.exif_transpose(img).convert("RGB")
        return self.tf(img), self.labels[i], i


# =============================================================================
# 采样权重：类别平衡 + 同一原图截图过多时降权
# =============================================================================
def sample_weights(df, num_classes):
    counts = df["class_idx"].value_counts().reindex(range(num_classes)).fillna(0).values
    cw = 1.0 / np.sqrt(np.maximum(counts, 1))             # 类别：1/√n
    w = cw[df["class_idx"].values]
    gs = df["group_size"].astype(float).values
    w = w / np.sqrt(np.maximum(gs, 1))                    # 同一原图/菌株截了很多张：1/√组大小
    return torch.as_tensor(w / w.mean(), dtype=torch.double)


# =============================================================================
# 训练 / 评估
# =============================================================================
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, idxs = [], []
    for x, _, i in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        idxs.append(i.numpy())
    return np.concatenate(probs), np.concatenate(idxs)


def metrics(y, p):
    pred = p.argmax(1)
    top2 = np.argsort(-p, 1)[:, :2]
    return dict(acc=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                balanced_acc=float(balanced_accuracy_score(y, pred)),
                top2=float(np.mean([y[k] in top2[k] for k in range(len(y))])))


def cosine_lr(opt, base_lr, epoch, step, steps_per_epoch, epochs, warmup=1):
    t = epoch + step / steps_per_epoch
    lr = base_lr * t / warmup if t < warmup else 0.5 * base_lr * (1 + math.cos(math.pi * (t - warmup) / max(epochs - warmup, 1e-6)))
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
    return lr


def train_one(df_tr, df_va, a, num_classes, device, out_dir, log_prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(a.backbone, num_classes, pretrained=not a.no_pretrained).to(device)
    head_names = {"classifier", "fc", "head", "heads"}
    head, body = [], []
    for n, p in model.named_parameters():
        (head if n.split(".")[0] in head_names else body).append(p)
    opt = torch.optim.AdamW([{"params": body, "lr_scale": 1.0}, {"params": head, "lr_scale": 10.0}],
                            lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    ds_tr = ImgDataset(df_tr, a.img_size, True)
    sampler = WeightedRandomSampler(sample_weights(df_tr, num_classes), num_samples=len(df_tr), replacement=True)
    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0, pin_memory=False)
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, sampler=sampler, drop_last=True, **kw)
    dl_va = DataLoader(ImgDataset(df_va, a.img_size, False), batch_size=a.batch * 2, shuffle=False, **kw) if df_va is not None else None

    best, best_ep, bad, log = -1, 0, 0, []
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, y, _) in enumerate(dl_tr):
            lr = cosine_lr(opt, a.lr, ep, step, len(dl_tr), a.epochs)
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        row = dict(epoch=ep + 1, train_loss=tot / max(n, 1), lr=lr, seconds=round(time.time() - t0, 1))
        if dl_va is not None:
            p, idx = predict(model, dl_va, device)
            y = df_va["class_idx"].values[idx]
            row.update(metrics(y, p))
            score = row["macro_f1"]
            if score > best:
                best, best_ep, bad = score, ep + 1, 0
                torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            else:
                bad += 1
        log.append(row)
        pd.DataFrame(log).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
        msg = f"{log_prefix}ep {ep + 1:02d}/{a.epochs} loss {row['train_loss']:.3f} lr {lr:.1e} {row['seconds']}s"
        if dl_va is not None:
            msg += f" | acc {row['acc']:.3f} F1 {row['macro_f1']:.3f} top2 {row['top2']:.3f}  (best F1 {best:.3f} @ep{best_ep})"
        print(msg, flush=True)
        if dl_va is not None and bad >= a.patience:
            print(f"{log_prefix}早停：{a.patience} 个 epoch 无提升")
            break
    if dl_va is None:
        torch.save(model.state_dict(), os.path.join(out_dir, "final.pt"))
        return model, None, None, ep + 1
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    p, idx = predict(model, dl_va, device)
    return model, p, idx, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="densenet121")
    ap.add_argument("--mode", choices=["cv", "full"], default="cv")
    ap.add_argument("--trainset", default="trainset",
                    help="训练清单 CSV，或直接写文件夹（如 trainset_v3），会自动用其中的 trainset_micro.csv 和 class_names.json")
    ap.add_argument("--class_names", default="", help="默认与 --trainset 同一文件夹的 class_names.json")
    ap.add_argument("--out_root", default=OP.RUNS, help="默认 output/02_training_runs/runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=0, help="只跑第几折（1–5），调试用")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="不加载 ImageNet 权重（调试用）")
    ap.add_argument("--resume_dir", default="", help="续跑：指定上次的输出目录，已完成的折自动跳过")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = get_device()
    a.trainset = OP.find_trainset_csv(a.trainset)
    a.out_root = OP.place(a.out_root, OP.RUNS_ROOT)
    if a.resume_dir:
        given = a.resume_dir
        explicit = os.path.isabs(given) or given.replace("\\", "/").lstrip("./").startswith("output/")
        if explicit:
            # 写明了完整路径（绝对路径或 output/… 开头）：原样使用，不存在就新建；
            # 不再按文件夹名到别处"找"——2026-10-22 前会因此把 runs_v4/… 解析到已有的 runs/… 并覆盖旧模型
            a.resume_dir = os.path.abspath(given)
        else:
            q = OP.find_run(given)
            if os.path.exists(q):
                a.resume_dir = q
            else:   # 新目录：runs_loso/xxx → output/02_training_runs/runs_loso/xxx；只写名字 → output/02_training_runs/runs/xxx
                a.resume_dir = OP.place(given, OP.RUNS_ROOT if os.sep in given.strip("./") else OP.RUNS)
        if os.path.abspath(given) != os.path.abspath(a.resume_dir):
            print(f"⚠️  --resume_dir {given} 解析为 {a.resume_dir}")
        if a.mode == "full" and os.path.exists(os.path.join(a.resume_dir, "final.pt")):
            sys.exit(f"✗ {a.resume_dir}/final.pt 已存在，full 模式不会覆盖已有模型。请换输出目录或先把它改名。")
    if not a.class_names:
        a.class_names = os.path.join(os.path.dirname(a.trainset) or ".", "class_names.json")
    for f in (a.trainset, a.class_names):
        if not os.path.exists(f):
            sys.exit(f"找不到 {f}")
    print(f"训练清单：{a.trainset}\n类别表：{a.class_names}")
    df = pd.read_csv(a.trainset, encoding="utf-8-sig")
    class_names = json.load(open(a.class_names, encoding="utf-8"))
    K = len(class_names)
    out = a.resume_dir or os.path.join(a.out_root, f"{a.backbone}_{a.mode}_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(class_names, open(os.path.join(out, "class_names.json"), "w"), indent=2, ensure_ascii=False)
    print(f"设备 {device} | backbone {a.backbone} | 模式 {a.mode} | {len(df)} 张图 {K} 类 | 输出 {out}")

    if a.mode == "full":
        epochs = a.epochs
        train_one(df, None, a, K, device, out)
        print(f"最终模型已保存：{out}/final.pt（训练 {epochs} 个 epoch）")
        return

    sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    splits = list(sgkf.split(np.zeros(len(df)), df["class_idx"], df["cv_group"]))
    oof = np.full((len(df), K), np.nan, dtype=np.float32)
    fold_of = np.zeros(len(df), dtype=int)
    rows = []
    for k, (tr, va) in enumerate(splits, 1):
        if a.only_fold and k != a.only_fold:
            continue
        fdir = os.path.join(out, f"fold_{k}")
        pfile = os.path.join(fdir, "val_probs.npy")
        assert not set(df["cv_group"].iloc[tr]) & set(df["cv_group"].iloc[va]), "分组泄漏！"
        if os.path.exists(pfile):
            print(f"== 第 {k} 折已完成，跳过")
            p = np.load(pfile)
            best_ep = json.load(open(os.path.join(fdir, "fold.json")))["best_epoch"]
        else:
            print(f"\n== 第 {k}/{a.folds} 折：训练 {len(tr)} 张 / 验证 {len(va)} 张 ==")
            _, p, idx, best_ep = train_one(df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True),
                                           a, K, device, fdir, log_prefix=f"[F{k}] ")
            p = p[np.argsort(idx)]
            np.save(pfile, p)
            json.dump({"best_epoch": best_ep}, open(os.path.join(fdir, "fold.json"), "w"))
        oof[va] = p
        fold_of[va] = k
        m = metrics(df["class_idx"].values[va], p)
        m.update(fold=k, best_epoch=best_ep, n_val=len(va))
        rows.append(m)
        print(f"[F{k}] 验证：acc {m['acc']:.3f}  macro-F1 {m['macro_f1']:.3f}  balanced acc {m['balanced_acc']:.3f}  top2 {m['top2']:.3f}")

    fr = pd.DataFrame(rows)
    fr.to_csv(os.path.join(out, "cv_folds.csv"), index=False)
    done = ~np.isnan(oof).any(1)
    y = df["class_idx"].values[done]
    p = oof[done]
    pooled = metrics(y, p)
    summary = {"backbone": a.backbone, "folds_done": fr["fold"].tolist(),
               **{f"{c}_mean": float(fr[c].mean()) for c in ["acc", "macro_f1", "balanced_acc", "top2"]},
               **{f"{c}_sd": float(fr[c].std(ddof=1)) if len(fr) > 1 else 0.0 for c in ["acc", "macro_f1", "balanced_acc", "top2"]},
               "pooled_oof": pooled, "median_best_epoch": float(fr["best_epoch"].median())}
    json.dump(summary, open(os.path.join(out, "cv_summary.json"), "w"), indent=2, ensure_ascii=False)
    o = df.loc[done, ["path", "train_class", "class_idx", "cv_group", "source"]].copy()
    o["fold"] = fold_of[done]
    o["pred_idx"] = p.argmax(1)
    o["pred_class"] = [class_names[i] for i in o["pred_idx"]]
    o["p_top1"] = p.max(1)
    for i, c in enumerate(class_names):
        o[f"p::{c}"] = p[:, i]
    o.to_csv(os.path.join(out, "oof_predictions.csv"), index=False, encoding="utf-8-sig")
    rep = classification_report(y, p.argmax(1), labels=list(range(K)), target_names=class_names,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(os.path.join(out, "per_class.csv"), encoding="utf-8-sig")
    print("\n== 交叉验证汇总 ==")
    for c in ["acc", "macro_f1", "balanced_acc", "top2"]:
        print(f"  {c:13s} {summary[c + '_mean']:.3f} ± {summary[c + '_sd']:.3f}")
    print(f"  最佳 epoch 中位数 {summary['median_best_epoch']:.0f}（full 模式可用 --epochs 设为这个值）")
    print(f"  结果目录 {out}")


if __name__ == "__main__":
    main()
