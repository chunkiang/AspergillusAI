#!/usr/bin/env python3
"""
整理输出文件夹：把同类子目录归入分类文件夹。
用法：
    python organize_outputs.py /path/to/output            # 只预览，不移动（dry run）
    python organize_outputs.py /path/to/output --apply    # 实际执行
执行后会在该目录生成 _move_log.csv，并可用 --undo 按日志还原。
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

# (分类文件夹, 匹配规则) —— 按顺序匹配，先命中先归类
RULES = [
    ("01_trainsets",        re.compile(r"^trainset(_.*)?$")),
    ("02_training_runs",    re.compile(r"^runs(_.*)?$")),
    ("03_external_eval",    re.compile(r"^ext_eval(_.*)?$")),
    ("04_audit_review",     re.compile(r"^(ext_audit|train_audit|review_.*)$")),
    ("05_roi_diagnostics",  re.compile(r"^roi_diag(_.*)?$")),
    ("06_logs",             re.compile(r"^logs$")),
]
DELETE = {"__pycache__"}          # Python 缓存，可随时重新生成
LOG_NAME = "_move_log.csv"


def plan(root: Path):
    moves, deletes, unmatched = [], [], []
    categories = {c for c, _ in RULES}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name in categories:
            continue
        if p.name in DELETE:
            deletes.append(p)
            continue
        for cat, pat in RULES:
            if pat.match(p.name):
                moves.append((p, root / cat / p.name))
                break
        else:
            unmatched.append(p)
    return moves, deletes, unmatched


def undo(root: Path):
    log = root / LOG_NAME
    if not log.exists():
        sys.exit(f"未找到 {log}")
    with log.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    for src, dst in reversed(rows):
        if Path(dst).exists() and not Path(src).exists():
            shutil.move(dst, src)
            print(f"还原  {dst}  ->  {src}")
    for cat, _ in RULES:
        d = root / cat
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    log.rename(log.with_suffix(".undone.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--apply", action="store_true", help="实际执行移动")
    ap.add_argument("--undo", action="store_true", help="按日志还原")
    a = ap.parse_args()
    root = a.root.resolve()

    if a.undo:
        undo(root)
        return

    moves, deletes, unmatched = plan(root)
    for src, dst in moves:
        print(f"移动  {src.name:<45} -> {dst.parent.name}/")
    for d in deletes:
        print(f"删除  {d.name}")
    for u in unmatched:
        print(f"未归类（保持原位）  {u.name}")

    if not a.apply:
        print("\n以上为预览。确认无误后加 --apply 执行。")
        return

    conflicts = [d for _, d in moves if d.exists()]
    if conflicts:
        sys.exit(f"目标已存在，已中止：{conflicts}")

    with (root / LOG_NAME).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst"])
        for src, dst in moves:
            dst.parent.mkdir(exist_ok=True)
            shutil.move(str(src), str(dst))
            w.writerow([str(src), str(dst)])
    for d in deletes:
        shutil.rmtree(d)
    print(f"\n完成。移动记录：{root / LOG_NAME}（可用 --undo 还原）")


if __name__ == "__main__":
    main()
