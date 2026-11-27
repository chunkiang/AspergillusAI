# -*- coding: utf-8 -*-
"""
paths.py —— 全部脚本共用的输出文件夹约定（放在 Audit 文件夹，和其他脚本在一起）

Audit/
├── *.py, *.zsh                  脚本
├── taxonomy_map.csv, answers.csv
└── output/
    ├── 01_trainsets/            trainset, trainset_v3, trainset_roi_v3, trainset_loso_* …
    ├── 02_training_runs/        runs/（新训练默认放这里）, runs_20261009_…, runs_loso/
    ├── 03_external_eval/        ext_eval*, ext_loso/
    ├── 04_audit_review/         ext_audit/, train_audit/, review_*/, review_images.csv …
    ├── 05_roi_diagnostics/      roi_diag*
    └── 06_logs/

命令行里可以只写短名字，脚本会自动到对应文件夹里找：
    --trainset trainset_v3                → output/01_trainsets/trainset_v3
    --models runs/swin_t_full_final       → output/02_training_runs/runs/swin_t_full_final
    --out ext_eval_roi_v3                 → output/03_external_eval/ext_eval_roi_v3
写绝对路径或当前目录下确实存在的路径，则按原样使用（兼容旧的文件夹结构）。
"""
import os
import sys
import math
import unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
TRAINSETS = os.path.join(OUT, "01_trainsets")
RUNS_ROOT = os.path.join(OUT, "02_training_runs")
RUNS = os.path.join(RUNS_ROOT, "runs")                 # train.py 新训练的默认位置
EXT = os.path.join(OUT, "03_external_eval")
AUDIT = os.path.join(OUT, "04_audit_review")
DIAG = os.path.join(OUT, "05_roi_diagnostics")
LOGS = os.path.join(OUT, "06_logs")

EXT_AUDIT = os.path.join(AUDIT, "ext_audit")
TRAIN_AUDIT = os.path.join(AUDIT, "train_audit")
MANIFEST = os.path.join(EXT_AUDIT, "manifest.csv")
INVENTORY = os.path.join(EXT_AUDIT, "inventory.csv")
TRAIN_MANIFEST = os.path.join(TRAIN_AUDIT, "train_manifest.csv")

ALL_DIRS = [TRAINSETS, RUNS_ROOT, EXT, AUDIT, DIAG, LOGS]


def _clean(p):
    return os.path.expanduser(str(p).strip().strip("'\""))


def find(p, *bases):
    """输入用：按原样 → 当前目录 → 各基准文件夹 依次找已存在的路径；找不到就返回按原样的路径（让后面报错时路径清楚）"""
    if not p:
        return p
    p = _clean(p)
    if os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p) if os.path.exists(p) else p
    tail = p[2:] if p.startswith("./") else p
    for b in bases:
        for cand in (os.path.join(b, tail), os.path.join(b, os.path.basename(tail.rstrip("/")))):
            if os.path.exists(cand):
                return cand
    return p


def place(p, base):
    """输出用：绝对路径 → 原样；当前目录下已存在 → 原样（旧结构）；只写了名字 → 放进 base"""
    p = _clean(p)
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    parent = os.path.dirname(p.rstrip("/"))
    if parent and os.path.isdir(parent):          # 写了带文件夹的路径且上级文件夹存在（如 output/03_external_eval/xxx）
        return os.path.abspath(p)
    tail = p[2:] if p.startswith("./") else p
    return os.path.join(base, tail)


def find_trainset_csv(p, default="trainset"):
    """训练清单：可写文件夹名（trainset_v3）或 CSV 路径"""
    p = find(p or default, TRAINSETS, BASE)
    if os.path.isdir(p):
        p = os.path.join(p, "trainset_micro.csv")
    if not os.path.exists(p):
        sys.exit(f"找不到训练清单：{p}\n（训练集应在 {TRAINSETS} 下，可只写文件夹名，如 trainset_v3）")
    return p


def find_run(p):
    """模型 / 交叉验证输出目录：在 02_training_runs 下（含 runs/、runs_日期/、runs_loso/ 等子文件夹）找"""
    q = find(p, RUNS_ROOT, RUNS)
    if os.path.exists(q):
        return q
    name = os.path.basename(_clean(p).rstrip("/"))
    if os.path.isdir(RUNS_ROOT):
        for sub in sorted(os.listdir(RUNS_ROOT)):
            cand = os.path.join(RUNS_ROOT, sub, name)
            if os.path.isdir(cand):
                return cand
    return q


def latest_cv_run():
    cands = []
    for root in (RUNS_ROOT, os.path.join(BASE, "runs")):
        if not os.path.isdir(root):
            continue
        for d, subs, files in os.walk(root):
            if "_cv_" in os.path.basename(d) and "oof_predictions.csv" in files:
                cands.append((os.path.getmtime(os.path.join(d, "oof_predictions.csv")), d))
    if not cands:
        sys.exit(f"在 {RUNS_ROOT} 下没有找到交叉验证结果")
    return max(cands)[1]


# ---------------------------------------------------------------------------
# 中文对齐的表格（pandas 的 to_string 把中文当 1 格，终端里会错位）
# ---------------------------------------------------------------------------
def _dw(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _fmt(v):
    try:
        import numpy as np
        if isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
    except ImportError:
        pass
    if v is None:
        return "—"
    if isinstance(v, float):
        return "—" if math.isnan(v) else f"{v:.3f}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return str(v)


def table(obj, index=True):
    """DataFrame / Series → 按显示宽度对齐的文本表格；数字列右对齐"""
    import pandas as pd
    if isinstance(obj, pd.Series):
        obj = obj.to_frame(obj.name if obj.name is not None else "数量")
    keep_index = index and not isinstance(obj.index, pd.RangeIndex)
    df = obj.reset_index() if keep_index else obj
    cols = ["" if str(c) == "index" else str(c) for c in df.columns]
    rows = [[_fmt(v) for v in r] for r in df.itertuples(index=False)]
    if not rows:
        return "（无数据）"
    w = [max([_dw(c)] + [_dw(r[i]) for r in rows]) for i, c in enumerate(cols)]

    def isnum(x):
        x = x.replace("—", "").replace("-", "", 1).replace(".", "", 1)
        return x == "" or x.isdigit() or x.startswith("(")
    right = [all(isnum(r[i]) for r in rows) for i in range(len(cols))]

    def line(vals):
        out = []
        for i, v in enumerate(vals):
            pad = " " * (w[i] - _dw(v))
            out.append(pad + v if right[i] else v + pad)
        return "  ".join(out).rstrip()
    head = line(cols)
    return "\n".join([head, "-" * _dw(head)] + [line(r) for r in rows])
