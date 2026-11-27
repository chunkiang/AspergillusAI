#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
99_organize_outputs.py —— 把 Audit 下散落的输出文件夹，按 paths.py 的约定移进 output/ 的 6 个分类文件夹

    python 99_organize_outputs.py            # 预演：只列出将怎样移动，不动任何文件
    python 99_organize_outputs.py --apply    # 真正移动，并记录到 output/move_log.csv
    python 99_organize_outputs.py --undo     # 按 move_log.csv 全部移回原处

只移动输出（训练清单、模型、评估结果、核对网页、日志），脚本、taxonomy_map.csv、answers.csv 不动。
文件夹整体移动（mv），同一磁盘内瞬间完成，不复制、不删除。
"""
import os
import re
import csv
import shutil
import argparse
from datetime import datetime

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths as OP

RULES = [   # (匹配规则, 目标文件夹, 说明)
    (r"^trainset.*$", OP.TRAINSETS, "训练清单"),
    (r"^runs(_.*)?$", OP.RUNS_ROOT, "训练结果 / 模型"),
    (r"^ext_eval.*$|^ext_loso$", OP.EXT, "外部验证"),
    (r"^(ext_audit|train_audit|crop_review)$|^review_.*", OP.AUDIT, "审计与人工核对"),
    (r"^roi_diag.*$", OP.DIAG, "ROI 诊断"),
]
FILES = {"review_images.csv": OP.AUDIT, "review_classes.csv": OP.AUDIT, "review_result.csv": OP.AUDIT,
         "review_cross_class.html": OP.AUDIT}
LOG = os.path.join(OP.OUT, "move_log.csv")


def plan():
    moves, skipped = [], []
    for name in sorted(os.listdir(OP.BASE)):
        src = os.path.join(OP.BASE, name)
        if name in ("output", "__pycache__") or name.startswith("."):
            continue
        dst_dir = None
        if os.path.isdir(src):
            if name == "logs":                      # 日志：把里面的文件移进 06_logs
                for f in sorted(os.listdir(src)):
                    if not f.startswith("."):
                        moves.append((os.path.join(src, f), os.path.join(OP.LOGS, f), "日志"))
                continue
            for pat, d, why in RULES:
                if re.match(pat, name):
                    dst_dir = (d, why)
                    break
        elif name in FILES:
            dst_dir = (FILES[name], "人工核对结果")
        if dst_dir:
            dst = os.path.join(dst_dir[0], name)
            if os.path.exists(dst):
                skipped.append((src, dst))
            else:
                moves.append((src, dst, dst_dir[1]))
    return moves, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()

    if a.undo:
        if not os.path.exists(LOG):
            raise SystemExit(f"没有 {LOG}，无法撤销")
        rows = list(csv.DictReader(open(LOG, encoding="utf-8-sig")))
        for r in reversed(rows):
            if os.path.exists(r["to"]) and not os.path.exists(r["from"]):
                os.makedirs(os.path.dirname(r["from"]), exist_ok=True)
                shutil.move(r["to"], r["from"])
                print(f"移回 {os.path.relpath(r['from'], OP.BASE)}")
        os.rename(LOG, LOG + f".undone_{datetime.now():%Y%m%d_%H%M}")
        return

    moves, skipped = plan()
    rel = lambda p: os.path.relpath(p, OP.BASE)
    w = max([len(rel(s)) for s, _, _ in moves] + [10])
    print(f"Audit：{OP.BASE}\n")
    for s, d, why in moves:
        print(f"  {rel(s):<{w}}  →  {rel(d):<55s} {why}")
    for s, d in skipped:
        print(f"  ⚠ 跳过 {rel(s)}：目标 {rel(d)} 已存在")
    left = [n for n in sorted(os.listdir(OP.BASE)) if os.path.isdir(os.path.join(OP.BASE, n))
            and n not in ("output", "__pycache__", "logs") and not n.startswith(".")
            and not any(rel(s) == n for s, _, _ in moves)]
    if left:
        print(f"\n不在规则内、保留原处的文件夹：{'、'.join(left)}")
    if not a.apply:
        print(f"\n以上是预演，共 {len(moves)} 项。确认无误后运行：python 99_organize_outputs.py --apply")
        return

    for d in OP.ALL_DIRS:
        os.makedirs(d, exist_ok=True)
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["time", "from", "to"])
        for s, d, _ in moves:
            shutil.move(s, d)
            wr.writerow([datetime.now().isoformat(timespec="seconds"), s, d])
    logs_dir = os.path.join(OP.BASE, "logs")
    if os.path.isdir(logs_dir) and not os.listdir(logs_dir):
        os.rmdir(logs_dir)
    print(f"\n已移动 {len(moves)} 项，记录在 {rel(LOG)}；如需撤销：python 99_organize_outputs.py --undo")


if __name__ == "__main__":
    main()
