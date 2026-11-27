#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_sanity_check.py —— 检查最终模型是否正常：在训练截图上应该几乎全对

用法：
    python 03_sanity_check.py --models runs/swin_t_full_final runs/efficientnet_b0_full_final runs/densenet121_full_final
若准确率远低于 0.95，说明模型载入 / 类别顺序 / 预处理有问题，外部验证结果不可信。
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib                                         # noqa: E402
_ev = importlib.import_module("03_eval_external")
load_models, predict_tiles = _ev.load_models, _ev.predict_tiles
get_device = importlib.import_module("02_train").get_device
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--trainset", default="trainset", help="训练集文件夹名或 CSV（在 output/01_trainsets/ 下找）")
ap.add_argument("--per_class", type=int, default=15)
a = ap.parse_args()
a.trainset = OP.find_trainset_csv(a.trainset)
a.models = [OP.find_run(d) for d in a.models]

device = get_device()
models, names, sizes = load_models(a.models, device)
df = pd.read_csv(a.trainset, encoding="utf-8-sig")
idx = []
for _, d in df.groupby("train_class"):
    idx += list(d.sample(min(len(d), a.per_class), random_state=0).index)
df = df.loc[idx].copy()
imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in df["path"]]
p = predict_tiles(imgs, models, sizes, device)
df["pred"] = [names[i] for i in p.argmax(1)]
df["ok"] = df["pred"] == df["train_class"]
print(f"\n训练截图抽样 {len(df)} 张：准确率 {df['ok'].mean():.3f}（正常应 ≥ 0.95）")
print(OP.table(df.groupby("train_class")["ok"].mean().round(2)))
print("\n预测类别分布：")
print(OP.table(df["pred"].value_counts().head(8)))
