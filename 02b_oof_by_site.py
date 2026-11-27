#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_oof_by_site.py —— 按数据来源（site）拆开看交叉验证的 out-of-fold 结果

重点看外部来源（TP、HYY、MQQ、WEB1/2…）在交叉验证中的准确率：
它们和本院（PUPH）的差距，就是跨实验室泛化能力的初步估计。

用法：
    python 02b_oof_by_site.py --run runs/<本次交叉验证的输出目录> --trainset trainset_roi/trainset_micro.csv
"""
import os
import argparse
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--trainset", default="trainset_roi")
    a = ap.parse_args()
    a.run = OP.find_run(a.run)
    a.trainset = OP.find_trainset_csv(a.trainset)
    o = pd.read_csv(os.path.join(a.run, "oof_predictions.csv"), encoding="utf-8-sig")
    t = pd.read_csv(a.trainset, encoding="utf-8-sig")
    if "site" not in t:
        print("注意：训练清单没有 site 列（旧版 build_trainset 生成），全部按 PUPH 处理")
        t["site"] = "PUPH"
    t = t[["path", "site", "source", "cv_group"]]
    o = o.drop(columns=[c for c in ("source", "cv_group") if c in o]).merge(t, on="path", how="left")
    pred = "pred_class"
    o["ok"] = o[pred] == o["train_class"]
    o["genus_ok"] = o[pred].str.split().str[0] == o["train_class"].str.split().str[0]

    L = ["===== 按来源（截图级） ====="]
    L.append(OP.table(o.groupby("site").agg(截图数=("ok", "size"), 准确率=("ok", "mean"), 属一级=("genus_ok", "mean"))
             .sort_values("截图数", ascending=False).round(3)))
    g = o.groupby(["site", "cv_group"]).agg(ok=("ok", "mean")).reset_index()
    g["ok"] = g["ok"] >= 0.5
    L.append("\n===== 按来源（病例/分组级，组内过半截图判对算对） =====")
    L.append(OP.table(g.groupby("site").agg(分组数=("ok", "size"), 准确率=("ok", "mean"))
             .sort_values("分组数", ascending=False).round(3)))
    ext = o[o["site"] != "PUPH"]
    L.append(f"\n本院 PUPH 截图准确率 {o.loc[o['site'] == 'PUPH', 'ok'].mean():.3f}；"
             f"外部来源合计 {ext['ok'].mean():.3f}（n={len(ext)}）")
    L.append("\n===== 外部来源每类准确率（截图级） =====")
    L.append(OP.table(ext.groupby("train_class")["ok"].agg(准确率="mean", n="size").round(3)))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.run, "oof_by_site.txt"), "w", encoding="utf-8").write(txt)
    print(f"\n已写出 {a.run}/oof_by_site.txt")


if __name__ == "__main__":
    main()
