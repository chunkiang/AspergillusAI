#!/usr/bin/env python3
"""Collect FungalAI results using Python 3 standard library only.
Usage: python3 collect_fungal_results.py --root /path/to/project
Copies only; never edits source files. Does not anonymize or upload data.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime
import zipfile

TEXT_EXT = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.jsonl', '.yaml', '.yml', '.txt', '.md', '.log'}
ROOT_NAMES = {'answers.csv', 'answers_clinical.csv', 'clinical_lock.json',
              'taxonomy_map.csv', '菌株登记表.xlsx', '菌种总表_taxonomy_master.xlsx'}
ARRAY_WORDS = ('prob', 'pred', 'label', 'target', 'split', 'fold', 'oof', 'class')
PLOT_WORDS = ('confusion', 'roc', 'coverage', 'calibration', 'comparison', 'accuracy', 'f1', 'weight')

def reason(rel, plots=False):
    parts = rel.parts
    name = rel.name.lower()
    ext = rel.suffix.lower()
    if any(p.startswith('.') or p in {'__pycache__', '_old_scripts'} or p.startswith('FungalAI_results_') for p in parts):
        return None
    if len(parts) == 1:
        if rel.name in ROOT_NAMES:
            return 'reference_labels_and_mapping'
        if ext in {'.py', '.sh', '.zsh', '.md'} and rel.name != Path(__file__).name:
            return 'code_and_configuration'
        return None
    if parts[0] not in {'output', '00b_inventory_out'}:
        return None
    if 'colony_annotations' in parts:
        return None
    if name in {'list_of_output.txt', 'lst.txt'}:
        return None
    if ext in TEXT_EXT:
        return 'results_metadata_and_logs'
    if ext in {'.npy', '.npz'} and any(w in name for w in ARRAY_WORDS) and not name.startswith('feat'):
        return 'predictions_and_splits'
    if plots and ext in {'.png', '.pdf', '.svg'} and any(w in name for w in PLOT_WORDS):
        return 'result_figures'
    return None

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description='复制真菌AI统计所需结果，保留目录结构并生成ZIP；不修改原文件。')
    ap.add_argument('--root', default='.', help='包含 output 和 02_train.py 的项目目录；也可指定 output 目录')
    ap.add_argument('--dest', help='新的目标文件夹；必须尚不存在')
    ap.add_argument('--max-file-mb', type=float, default=100, help='单文件上限，默认100MB；超过会记录而不复制')
    ap.add_argument('--plots', action='store_true', help='额外收集名称匹配的统计图')
    ap.add_argument('--dry-run', action='store_true', help='仅预览数量和体积，不创建文件')
    ap.add_argument('--no-zip', action='store_true', help='只创建文件夹')
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    if root.name == 'output' and (root / '02_training_runs').is_dir():
        root = root.parent
    if not (root / 'output').is_dir():
        ap.error('找不到 output 文件夹，请用 --root 指定项目根目录。')
    if a.max_file_mb <= 0:
        ap.error('--max-file-mb 必须大于0')
    dest = Path(a.dest).expanduser().resolve() if a.dest else root.parent / ('FungalAI_results_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    archive = dest.with_name(dest.name + '.zip')
    if dest == root or dest in root.parents or dest.exists() or (not a.no_zip and archive.exists()):
        ap.error('目标目录或ZIP已存在，或目标为源目录的祖先。请指定新的目标目录。')
    selected, skipped = [], []
    def walk_error(error):
        skipped.append({'path': str(error.filename), 'reason': str(error)})
    # Only inspect known result trees and project-level scripts/tables.
    candidates = [p for p in root.iterdir() if p.is_file()]
    for sub in ('output', '00b_inventory_out'):
        base = root / sub
        if not base.is_dir():
            continue
        for folder, dirs, files in os.walk(base, followlinks=False, onerror=walk_error):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__' and not d.startswith('FungalAI_results_') and not (Path(folder) / d).is_symlink())
            candidates.extend(Path(folder) / n for n in sorted(files))
    for path in sorted(candidates):
        rel = path.relative_to(root)
        why = reason(rel, a.plots)
        if not why:
            continue
        try:
            if path.is_symlink():
                skipped.append({'path': str(rel), 'reason': 'symbolic_link_not_followed'})
                continue
            size = path.stat().st_size
            if size > a.max_file_mb * 1024 ** 2:
                skipped.append({'path': str(rel), 'reason': 'over_size_limit', 'bytes': size})
                continue
            selected.append((path, rel, size, why))
        except OSError as e:
            skipped.append({'path': str(rel), 'reason': str(e)})
    print('项目目录:', root)
    print('拟复制: {} 个文件，{:.1f} MB'.format(len(selected), sum(v[2] for v in selected) / 1024 ** 2))
    print('异常或超过限制:', len(skipped))
    if a.dry_run:
        for item in skipped:
            print('跳过:', item)
        return 0
    if not selected:
        ap.error('未找到符合条件的结果文件。')
    dest.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path, rel, size, why in selected:
        target = dest / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            before = path.stat()
            shutil.copy2(path, target)
            source_hash, copied_hash = digest(path), digest(target)
            after = path.stat()
            if source_hash != copied_hash or before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                raise OSError('源文件在复制期间变化或校验不一致；请在运行结束后重新收集')
            manifest.append({'relative_path': str(rel), 'bytes': target.stat().st_size,
                             'sha256': copied_hash, 'category': why})
        except OSError as e:
            if target.exists():
                target.unlink()
            skipped.append({'path': str(rel), 'reason': str(e)})
    with (dest / 'COLLECTION_MANIFEST.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256', 'category'])
        w.writeheader()
        w.writerows(manifest)
    (dest / 'COLLECTION_SKIPPED.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'created_at': datetime.now().astimezone().isoformat(), 'copied_files': len(manifest),
               'copied_bytes': sum(x['bytes'] for x in manifest), 'skipped_candidates': len(skipped),
               'max_file_mb': a.max_file_mb, 'selection': 'Multiple runs retained; final run not inferred from timestamps.'}
    (dest / 'COLLECTION_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (dest / 'README_COLLECTION.txt').write_text(
        '本包用于真菌AI结果统计与方法核对。\n'
        '保留各运行目录；未根据日期擅自判断最终版本，也未合并不同运行。\n'
        '包括：数据清单、参考标签、逐株/逐图预测、折外预测、模型比较、配置、日志、当前脚本。\n'
        '不包括：原始图像、ROI图片、模型权重、特征缓存、旧脚本目录。\n'
        'COLLECTION_MANIFEST.csv 为已复制文件和SHA256校验值。\n'
        'COLLECTION_SKIPPED.json 为超限、符号链接或复制失败的候选文件；常规排除项不逐个列出。\n'
        '文件内容原样复制，未自动脱敏；上传前请检查登记表、参考标签及路径字段中的患者身份信息。\n'
        '匿名菌株ID及其对应关系请保留，以支持配对统计。\n', encoding='utf-8')
    print('已复制:', len(manifest), '个文件')
    print('结果文件夹:', dest)
    if not a.no_zip:
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            for p in sorted(dest.rglob('*')):
                if p.is_file():
                    z.write(p, arcname=str(Path(dest.name) / p.relative_to(dest)))
        print('上传这个ZIP:', archive)
    if skipped:
        print('注意：有候选文件未复制，请查看 COLLECTION_SKIPPED.json；超限可提高 --max-file-mb 后重新运行。')
    return 0

if __name__ == '__main__':
    sys.exit(main())
