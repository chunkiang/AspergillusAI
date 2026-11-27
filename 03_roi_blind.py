#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_roi_blind.py —— 外部测试图的"盲法"感兴趣区（ROI）截取：模拟临床上由技师选取视野、再交给模型判读

· 列出全部外部测试镜下图（本期 26 类 + 集外菌种），随机打乱顺序，编号匿名化为 ROI-0001…
· 窗口和终端都不显示菌名、文件夹名，截图文件名也不含菌名 —— 截取者对答案保持盲态
· 操作与 01_crop_batch_training.py 完全相同（缩放、旋转、撤销、续截、224×224 预览窗口、
  1/2/3 结构类型、i 成熟/未成熟）；每张图截 1–5 个最典型的产孢结构
· 截图存到 CNN/roi_test/，日志 roi_test/crop_log.csv；匿名编号与原图的对照在 roi_test/roi_key.csv
  （截图过程中请不要打开 roi_key.csv）
· 这些 ROI 只用于测试，不进入训练；00_build_manifest.py 也不会因此把原图排除出外部测试集

用法：
    python 03_roi_blind.py
    python 03_eval_external.py --models runs/*_full_final --roi_dir /Users/chunjiang/Documents/真菌AI/CNN/roi_test --out ext_eval_roi
"""
import os
import sys
import random
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib                    # noqa: E402
cb = importlib.import_module("01_crop_batch_training")   # noqa: E402  （带 224×224 预览窗口和结构/阶段标注）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ROI_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/roi_test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=OP.MANIFEST)
    ap.add_argument("--ext_root", default=cb.EXT_ROOT)
    ap.add_argument("--out_root", default=ROI_ROOT)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    os.makedirs(a.out_root, exist_ok=True)
    key_path = os.path.join(a.out_root, "roi_key.csv")

    if os.path.exists(key_path):            # 续截：沿用第一次生成的匿名编号和顺序
        key = pd.read_csv(key_path, encoding="utf-8-sig", dtype=str)
    else:
        m = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
        keep = (m["role"] == "external_micro") | (
            (m["role"] == "out_of_panel") & ~m["modality_guess"].isin(["colony", "direct_specimen"]))
        m = m[keep].sample(frac=1, random_state=a.seed).reset_index(drop=True)
        key = pd.DataFrame({"roi_code": [f"ROI-{i + 1:04d}" for i in range(len(m))],
                            "rel_path": m["rel_path"]})
        key.to_csv(key_path, index=False, encoding="utf-8-sig")
        print(f"已生成匿名对照表 {key_path}（截图期间请勿打开）")

    jobs = [dict(path=os.path.join(a.ext_root, r.rel_path), code=r.roi_code, species_cn="", train_class="ROI")
            for r in key.itertuples()]
    print(f"外部测试镜下图 {len(jobs)} 张，顺序已随机打乱。每张截 1–5 个最典型的产孢结构；没有可用结构按 s 跳过。")
    cb.run(jobs, a.out_root, os.path.join(a.out_root, "crop_log.csv"))


if __name__ == "__main__":
    main()
