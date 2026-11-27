#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_train_check.py —— train_*.zsh 用的图片检查与盘点工具（只读原图；import 只复制，不移动、不删除）

一般不直接运行，由 train_2_audit.zsh / train_3_crop.zsh / train_0_selftest.zsh 调用。

子命令
  species        列出 taxonomy_map.csv 里本期训练的菌种（中文名 \t 训练类）
  inspect        看一个待加入的文件夹：多少图片、有无子文件夹、像显微还是菌落
  import-micro   把新的显微原图复制到 原图_重截/<菌种>/<菌株>/
  import-colony  把新的菌落照片复制到 补拍_2026/<组>/
  audit          逐张检查全部训练原图和截图记录，写出 report.txt / problems.csv
  crop-status    原图_重截 里每个菌株的截图进度；--tsv 写出还没截完的菌株
  colony-status  补拍_2026 里每组的标注 / 白平衡进度
  fresh          CNN/dataset、CNN/normal 里有没有比 train_manifest.csv 更新的图（需要重跑 00_audit_train）
  smoke-subset   自检用：从训练清单抽一个很小的子集

退出码：0 正常；2 有 ❌ 错误（audit、import）；1 用法或运行错误
依赖（清华源）：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow pandas
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.exit("缺少 pillow：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow")
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAVE_HEIF = True
except ImportError:
    HAVE_HEIF = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
COLONY_RE = re.compile(r"^(PUPH-([A-Z]{3})-\d{3})-B-([OR])-(\d+)$", re.IGNORECASE)
COLONY_GROUPS = {"FUM": "Fumigati", "FLA": "Flavi", "NIG": "Nigri", "TER": "Terrei", "NID": "Nidulantes"}
SKIP_DIRS = {"annotations", "preprocessed", "_quarantine", "__MACOSX"}

# 问题类别：等级, 说明, 解决建议
CATS = {
    "dir_missing":        ("❌", "文件夹不存在", "检查 train_config.zsh 里对应的路径；外接硬盘是否已接上"),
    "corrupt":            ("❌", "图片打不开或已损坏", "用看图软件打开确认；损坏的从原始备份重新复制，或移出该文件夹"),
    "empty":              ("❌", "0 字节文件", "复制中断造成，删除后重新复制"),
    "species_unknown":    ("❌", "菌种文件夹名不在 taxonomy_map.csv 里", "文件夹改成 taxonomy_map.csv 的 species_cn 中文名（用 train_2_audit.zsh 加图时可从列表选）"),
    "dup_cross_species":  ("❌", "同一张图出现在不同菌种下", "确认正确菌种后删去另一份（会造成标签矛盾）"),
    "strain_two_species": ("❌", "同一菌株号出现在两个菌种下", "株号应唯一；确认哪个对，改掉另一个文件夹名"),
    "colony_folder":      ("❌", "文件名里的组代号与所在文件夹不一致", "把照片移到与文件名一致的组文件夹（如 PUPH-FUM-… 放进 FUM/）"),
    "name_conflict":      ("❌", "目标位置已有同名、但内容不同的文件", "不会覆盖。确认是不同的图后给新图改名（如序号 +1），再重新加入"),
    "heic_nolib":         ("⚠️", "HEIC 照片无法检查（缺 pillow-heif）", "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow-heif"),
    "small_raw":          ("⚠️", "显微原图太小", "可能是缩略图，换成原始分辨率的图"),
    "small_crop":         ("⚠️", "截图太小", "建数据集时会按 DS_MIN_CROP 自动剔除；需要的话重截"),
    "loose_images":       ("⚠️", "原图直接放在菌种文件夹里（没有菌株子文件夹）", "移入菌株子文件夹，否则交叉验证只能按原图分组"),
    "dup_cross_strain":   ("⚠️", "同一张图出现在同一菌种的不同菌株下", "会让交叉验证泄漏：保留一份，或确认两个文件夹其实是同一菌株后合并"),
    "colony_name":        ("⚠️", "菌落照片命名不规范", "改为 PUPH-FUM-001-B-O-1.jpg（组 FUM/FLA/NIG/TER/NID；O 正面，R 反面）；不改则不会被标注"),
    "colony_group":       ("⚠️", "组代号不属于菌落模型的 5 个曲霉组", "只有 FUM/FLA/NIG/TER/NID 进入菌落模型；其他菌种的菌落照片可以放着，不参与训练"),
    "colony_no_front":    ("⚠️", "该株只有反面照片", "菌落模型只用正面（O），请补拍正面"),
    "crop_missing":       ("⚠️", "截图记录里的截图文件不存在", "截图被删除或移动；建数据集时会自动剔除，如是误删请从备份恢复"),
    "dup_same_strain":    ("ℹ️", "同一菌株内有完全相同的图", "可删除副本（不影响分组）"),
    "colony_has_micro":   ("ℹ️", "菌落文件夹里有镜检图（文件名含 -C-）", "不会被当作菌落照片；可以移走"),
    "crop_src_missing":   ("ℹ️", "截图对应的原图已不在原位置", "不影响已截的图；只是无法回看原图"),
    "hidden":             ("ℹ️", "系统隐藏文件（._*、.DS_Store）", "已自动忽略"),
}
ORDER = ["❌", "⚠️", "ℹ️"]


# =============================================================================
def load_taxonomy(path):
    if not os.path.exists(path):
        sys.exit(f"找不到 {path}（taxonomy_map.csv 应放在 Audit 文件夹）")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows or "species_cn" not in rows[0] or "train_class" not in rows[0]:
        sys.exit(f"{path} 里没有 species_cn / train_class 列")
    return {r["species_cn"].strip(): r["train_class"].strip() for r in rows
            if r["species_cn"].strip() and r["train_class"].strip()}


def is_img(fn):
    return os.path.splitext(fn)[1].lower() in IMG_EXT and not fn.startswith(".")


def walk_imgs(root):
    for d, subs, files in os.walk(root):
        subs[:] = sorted(s for s in subs if not s.startswith(".") and s not in SKIP_DIRS)
        for fn in sorted(files):
            if is_img(fn):
                yield os.path.join(d, fn)


def md5(p, block=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(block), b""):
            h.update(ch)
    return h.hexdigest()


def check_image(p):
    """→ (问题类别或 None, 宽, 高)"""
    try:
        if os.path.getsize(p) == 0:
            return "empty", 0, 0
    except OSError:
        return "corrupt", 0, 0
    if os.path.splitext(p)[1].lower() in (".heic", ".heif") and not HAVE_HEIF:
        return "heic_nolib", 0, 0
    try:
        with Image.open(p) as im:
            w, h = im.size
            im.verify()
        return None, w, h
    except Exception:
        return "corrupt", 0, 0


def recrop_prefixes():
    """01_make_recrop_dirs.py 里的 菌种 → 截图编号前缀"""
    try:
        import importlib
        m = importlib.import_module("01_make_recrop_dirs")
        return {p[0]: p[2] for p in m.PLAN}
    except Exception:
        return {}


def default_prefix(species, train_class):
    pre = recrop_prefixes().get(species)
    if pre:
        return pre
    letters = re.sub(r"[^A-Za-z]", "", train_class.split("(")[0])[:3].upper() or "NEW"
    return f"PUPH-{letters}"


class Problems:
    def __init__(self):
        self.items = []

    def add(self, cat, path, detail=""):
        self.items.append((cat, path, detail))

    def count(self, level):
        return sum(1 for c, _, _ in self.items if CATS[c][0] == level)

    def report(self, max_show=8):
        lines = []
        by = defaultdict(list)
        for c, p, d in self.items:
            by[c].append((p, d))
        for lv in ORDER:
            for c, (l, desc, fix) in CATS.items():
                if l != lv or c not in by or c == "hidden":
                    continue
                its = by[c]
                lines.append(f"{l} {desc}：{len(its)} 个")
                for p, d in its[:max_show]:
                    lines.append(f"     {p}" + (f"   （{d}）" if d else ""))
                if len(its) > max_show:
                    lines.append(f"     … 另有 {len(its) - max_show} 个，见 problems.csv")
                lines.append(f"   💡 {fix}")
        return lines


# =============================================================================
def cmd_species(a):
    tax = load_taxonomy(a.taxonomy)
    for sp, tc in sorted(tax.items(), key=lambda x: (x[1], x[0])):
        print(f"{sp}\t{tc}")


def cmd_inspect(a):
    src = os.path.expanduser(a.src)
    if not os.path.isdir(src):
        print("ERROR=找不到文件夹")
        return 1
    direct = [f for f in os.listdir(src) if is_img(f) and os.path.isfile(os.path.join(src, f))]
    subs = {}
    for s in sorted(os.listdir(src)):
        sp = os.path.join(src, s)
        if os.path.isdir(sp) and not s.startswith(".") and s not in SKIP_DIRS:
            n = sum(1 for _ in walk_imgs(sp))
            if n:
                subs[s] = n
    allimgs = list(walk_imgs(src))
    n_colony = sum(1 for p in allimgs if COLONY_RE.match(os.path.splitext(os.path.basename(p))[0]))
    print(f"DIRECT={len(direct)}")
    print(f"SUBS={len(subs)}")
    print(f"TOTAL={len(allimgs)}")
    print(f"COLONY_NAMED={n_colony}")
    print("SUBNAMES=" + "|".join(f"{k}:{v}" for k, v in subs.items()))
    return 0


def _copy_plan(pairs, kind, log_path, dry=False):
    """pairs: [(源, 目标)] → 复制；目标已存在且内容相同 → 跳过；内容不同 → ❌ 不覆盖"""
    probs = Problems()
    todo, same = [], 0
    for s, d in pairs:
        bad, w, h = check_image(s)
        if bad in ("corrupt", "empty"):
            probs.add(bad, s)
            continue
        if bad == "heic_nolib":
            probs.add(bad, s)
        if os.path.exists(d):
            if md5(s) == md5(d):
                same += 1
                continue
            probs.add("name_conflict", s, f"目标 {d}")
            continue
        todo.append((s, d))
    for line in probs.report():
        print(line)
    n_err = probs.count("❌")
    if n_err:
        print(f"\n❌ 有 {n_err} 个问题，本次一张都没有复制。改好后重新加入（已有的不会重复复制）。")
        return 2
    if dry:
        print(f"\n预演：将复制 {len(todo)} 张；目标里已有相同的 {same} 张（会跳过）。")
        return 0
    new_log = not os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["time", "kind", "source", "dest", "md5"])
        for s, d in todo:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)
            w.writerow([dt.datetime.now().isoformat(timespec="seconds"), kind, s, d, md5(d)])
    print(f"\n✅ 复制 {len(todo)} 张；目标里已有相同的 {same} 张（跳过）。原文件夹未改动。")
    print(f"   复制记录：{log_path}")
    return 0


def cmd_import_micro(a):
    tax = load_taxonomy(a.taxonomy)
    if a.species not in tax:
        print(f"❌ {a.species} 不在 taxonomy_map.csv 的本期训练菌种里")
        return 2
    src = os.path.expanduser(a.src)
    dest = os.path.join(a.recrop_root, a.species)
    pairs = []
    if a.subfolders:
        for s in sorted(os.listdir(src)):
            sp = os.path.join(src, s)
            if os.path.isdir(sp) and not s.startswith(".") and s not in SKIP_DIRS:
                for p in walk_imgs(sp):
                    pairs.append((p, os.path.join(dest, s, os.path.relpath(p, sp))))
        loose = [f for f in os.listdir(src) if is_img(f)]
        if loose:
            print(f"⚠️  {len(loose)} 张图直接放在 {src} 下（不在菌株子文件夹里），本次不复制它们")
    else:
        if not re.match(r"^[^/\\]+$", a.strain or ""):
            print("❌ 菌株号不能为空，也不能含 / ")
            return 2
        for p in walk_imgs(src):
            pairs.append((p, os.path.join(dest, a.strain, os.path.relpath(p, src))))
    if not pairs:
        print("❌ 没有找到图片")
        return 2
    # 同一菌株号已在别的菌种下 → 拒绝
    strains = {os.path.relpath(d, dest).split(os.sep)[0] for _, d in pairs}
    clash = []
    if os.path.isdir(a.recrop_root):
        for sp in os.listdir(a.recrop_root):
            if sp != a.species and os.path.isdir(os.path.join(a.recrop_root, sp)):
                clash += [f"{s}（已在 {sp} 下）" for s in strains if os.path.isdir(os.path.join(a.recrop_root, sp, s))]
    if clash:
        print("❌ 菌株号与其他菌种重复：" + "、".join(clash))
        print("   💡 株号应唯一；如确是不同菌株，请换一个菌株号")
        return 2
    print(f"将复制 {len(pairs)} 张到 {dest}/（菌株：{'、'.join(sorted(strains))}）")
    return _copy_plan(pairs, "micro", a.log, a.dry_run)


def cmd_import_colony(a):
    src = os.path.expanduser(a.src)
    pairs, bad = [], []
    for p in walk_imgs(src):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = COLONY_RE.match(stem)
        if not m:
            if "-C-" in stem.upper():
                continue
            bad.append(p)
            continue
        code = m.group(2).upper()
        pairs.append((p, os.path.join(a.colony_raw, code, os.path.basename(p))))
    if bad:
        print(f"❌ 菌落照片命名不规范：{len(bad)} 张，例如：")
        for p in bad[:8]:
            print(f"     {os.path.basename(p)}")
        print("   💡 改为 PUPH-FUM-001-B-O-1.jpg（PUPH-组代号-三位株号-B-O正面/R反面-序号）后重新加入")
        return 2
    if not pairs:
        print("❌ 没有找到菌落照片")
        return 2
    other = sorted({os.path.basename(os.path.dirname(d)) for _, d in pairs} - set(COLONY_GROUPS))
    if other:
        print(f"⚠️  组代号 {other} 不属于菌落模型的 5 个曲霉组（FUM/FLA/NIG/TER/NID），会复制但不参与菌落训练")
    by = Counter(os.path.basename(os.path.dirname(d)) for _, d in pairs)
    front = sum(1 for s, _ in pairs if COLONY_RE.match(os.path.splitext(os.path.basename(s))[0]).group(3).upper() == "O")
    print(f"将复制 {len(pairs)} 张（正面 {front}，反面 {len(pairs) - front}）到 {a.colony_raw}/：" +
          "、".join(f"{k} {v}" for k, v in sorted(by.items())))
    return _copy_plan(pairs, "colony", a.log, a.dry_run)


# =============================================================================
def cmd_audit(a):
    os.makedirs(a.out, exist_ok=True)
    tax = load_taxonomy(a.taxonomy)
    P = Problems()
    summary = []
    md5_index = defaultdict(list)   # md5 → [(菌种, 菌株, 路径)]

    # ---------- 1. 原图_重截 ----------
    summary.append("【1】显微原图（原图_重截）")
    if not os.path.isdir(a.recrop_root):
        P.add("dir_missing", a.recrop_root)
    else:
        strain_sp = defaultdict(set)
        tab = []
        for sp in sorted(os.listdir(a.recrop_root)):
            spd = os.path.join(a.recrop_root, sp)
            if not os.path.isdir(spd) or sp.startswith("."):
                continue
            imgs = list(walk_imgs(spd))
            if not imgs:
                continue
            if sp not in tax:
                P.add("species_unknown", spd)
            strains = set()
            for p in imgs:
                rel = os.path.relpath(p, spd).split(os.sep)
                if len(rel) == 1:
                    P.add("loose_images", p)
                    st = ""
                else:
                    st = rel[0]
                    strains.add(st)
                    strain_sp[st].add(sp)
                bad, w, h = check_image(p)
                if bad:
                    P.add(bad, p)
                    continue
                if min(w, h) < a.min_raw:
                    P.add("small_raw", p, f"{w}×{h}")
                md5_index[md5(p)].append((sp, st, p))
            tab.append(f"   {sp:<10s} {tax.get(sp, '（不在 taxonomy_map）'):<45s} 菌株 {len(strains):>3d}  原图 {len(imgs):>5d}")
        for st, sps in strain_sp.items():
            if len(sps) > 1:
                P.add("strain_two_species", st, "、".join(sorted(sps)))
        summary += tab or ["   （空）"]

    # ---------- 2. 补拍_2026 ----------
    summary.append("")
    summary.append("【2】菌落原始照片（补拍_2026）")
    if not os.path.isdir(a.colony_raw):
        P.add("dir_missing", a.colony_raw)
    else:
        views = defaultdict(set)
        cnt = Counter()
        for p in walk_imgs(a.colony_raw):
            if a.ann_dir and os.path.abspath(p).startswith(os.path.abspath(a.ann_dir) + os.sep):
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            m = COLONY_RE.match(stem)
            if not m:
                P.add("colony_has_micro" if "-C-" in stem.upper() else "colony_name", p)
                continue
            sid, code, view = m.group(1).upper(), m.group(2).upper(), m.group(3).upper()
            folder = os.path.relpath(os.path.dirname(p), a.colony_raw).split(os.sep)[0].upper()
            if folder != "." and folder in COLONY_GROUPS and folder != code:
                P.add("colony_folder", p, f"文件名 {code}，文件夹 {folder}")
            if code not in COLONY_GROUPS:
                P.add("colony_group", p, code)
            bad, w, h = check_image(p)
            if bad:
                P.add(bad, p)
                continue
            views[sid].add(view)
            cnt[(code, view)] += 1
            md5_index[md5(p)].append(("菌落", sid, p))
        for sid, vs in views.items():
            if "O" not in vs:
                P.add("colony_no_front", sid)
        codes = sorted({c for c, _ in cnt})
        for c in codes:
            ns = len({s for s in views if s.split("-")[1] == c})
            summary.append(f"   {c} {COLONY_GROUPS.get(c, '（非菌落模型组）'):<12s} 株 {ns:>3d}  正面 {cnt[(c, 'O')]:>4d}  反面 {cnt[(c, 'R')]:>4d}")
        if not codes:
            summary.append("   （没有符合命名的菌落照片）")

    # ---------- 3. 截图记录 ----------
    summary.append("")
    summary.append("【3】截图记录（crop_log.csv）")
    for name, root in (("crops_v2 规范截图", a.crops_v2), ("crops_new 网络补截", a.crops_new)):
        log = os.path.join(root, "crop_log.csv")
        if not os.path.exists(log):
            summary.append(f"   {name}：没有 {log}")
            continue
        rows = [r for r in csv.DictReader(open(log, encoding="utf-8-sig")) if r.get("status") == "crop"]
        miss = small = 0
        srcs = set()
        for r in rows:
            cf = r.get("crop_file", "")
            if not cf or not os.path.exists(cf):
                P.add("crop_missing", cf or "(空)", name)
                miss += 1
            try:
                if int(float(r.get("w") or 0)) < a.min_crop:
                    P.add("small_crop", cf, f"{r.get('w')} px")
                    small += 1
            except ValueError:
                pass
            srcs.add(r.get("source_image", ""))
        for s in srcs:
            if s and not os.path.exists(s):
                P.add("crop_src_missing", s, name)
        summary.append(f"   {name}：截图 {len(rows)} 张，来自 {len(srcs)} 张原图；文件缺失 {miss}，太小 {small}")

    # ---------- 4. 重复 ----------
    for h, its in md5_index.items():
        if len(its) < 2:
            continue
        sps = {s for s, _, _ in its}
        sts = {(s, t) for s, t, _ in its}
        paths_ = " ↔ ".join(p for _, _, p in its[:3])
        if len(sps) > 1:
            P.add("dup_cross_species", paths_)
        elif len(sts) > 1:
            P.add("dup_cross_strain", paths_)
        else:
            P.add("dup_same_strain", paths_)

    # ---------- 输出 ----------
    with open(os.path.join(a.out, "problems.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["等级", "类别", "说明", "路径", "细节", "建议"])
        for c, p, d in P.items:
            w.writerow([CATS[c][0], c, CATS[c][1], p, d, CATS[c][2]])
    n_e, n_w, n_i = P.count("❌"), P.count("⚠️"), P.count("ℹ️")
    lines = [f"训练图片审计  {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"pillow-heif：{'已安装' if HAVE_HEIF else '未安装（HEIC 照片无法检查）'}", ""] + summary + ["", "【问题】"]
    lines += P.report() or ["   没有发现问题"]
    lines += ["", f"合计：❌ {n_e}   ⚠️ {n_w}   ℹ️ {n_i}"]
    txt = "\n".join(lines)
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n报告：{a.out}/report.txt   逐条清单：{a.out}/problems.csv")
    return 2 if n_e else 0


# =============================================================================
def v2_done(v2_log):
    done, ncrop = set(), Counter()
    if os.path.exists(v2_log):
        for r in csv.DictReader(open(v2_log, encoding="utf-8-sig")):
            if r.get("status") in ("done", "skipped"):
                done.add(r.get("source_image", ""))
            elif r.get("status") == "crop":
                ncrop[r.get("source_image", "")] += 1
    return done, ncrop


def cmd_crop_status(a):
    tax = load_taxonomy(a.taxonomy)
    done, ncrop = v2_done(a.v2_log)
    rows = []
    if not os.path.isdir(a.recrop_root):
        print(f"❌ 找不到 {a.recrop_root}")
        return 1
    for sp in sorted(os.listdir(a.recrop_root)):
        spd = os.path.join(a.recrop_root, sp)
        if not os.path.isdir(spd) or sp.startswith("."):
            continue
        for st in sorted(os.listdir(spd)):
            std = os.path.join(spd, st)
            if not os.path.isdir(std) or st.startswith("."):
                continue
            imgs = list(walk_imgs(std))
            if not imgs:
                continue
            nd = sum(1 for p in imgs if p in done)
            nc = sum(ncrop[p] for p in imgs)
            rows.append(dict(species=sp, strain=st, path=std, n_img=len(imgs), n_done=nd, n_crop=nc,
                             known=sp in tax, prefix=default_prefix(sp, tax.get(sp, "")) if sp in tax else ""))
    pend = [r for r in rows if r["n_done"] < r["n_img"]]
    print(f"{'菌种':<10s}{'菌株':<16s}{'原图':>6s}{'已看':>6s}{'截图':>6s}  状态")
    for r in rows:
        stt = "✅ 已截完" if r["n_done"] >= r["n_img"] else ("🟡 截了一部分" if r["n_done"] else "⬜ 未截")
        if not r["known"]:
            stt += "  ❌ 菌种不在 taxonomy_map，不能截"
        print(f"{r['species']:<10s}{r['strain']:<16s}{r['n_img']:>6d}{r['n_done']:>6d}{r['n_crop']:>6d}  {stt}")
    print(f"\n菌株 {len(rows)} 个：已截完 {len(rows) - len(pend)}，未截完 {len(pend)}"
          f"（已看 = 截完按 n 或按 s 跳过的原图）")
    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8") as f:
            for r in pend:
                if r["known"]:
                    f.write("\t".join([r["species"], r["strain"], r["path"], str(r["n_img"]), str(r["n_done"]),
                                       str(r["n_crop"]), r["prefix"]]) + "\n")
    return 0


def colony_status(colony_raw, ann_dir):
    rows = defaultdict(Counter)
    for p in walk_imgs(colony_raw) if os.path.isdir(colony_raw) else []:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = COLONY_RE.match(stem)
        if not m:
            continue
        code, view = m.group(2).upper(), m.group(3).upper()
        r = rows[code]
        r[f"photo_{view}"] += 1
        if view != "O":
            continue
        rel = os.path.relpath(p, colony_raw)
        jp = os.path.join(ann_dir, os.path.dirname(rel), stem + ".json")
        if not os.path.exists(jp):
            r["need_ann"] += 1
            continue
        try:
            d = json.load(open(jp, encoding="utf-8"))
        except Exception:
            r["bad_json"] += 1
            continue
        if d.get("excluded"):
            r["excluded"] += 1
        elif "wb" not in d:
            r["need_wb"] += 1
        else:
            r["ready"] += 1
    return rows


def cmd_colony_status(a):
    rows = colony_status(a.colony_raw, a.ann_dir)
    print(f"{'组':<5s}{'':<12s}{'正面':>6s}{'反面':>6s}{'待标注':>8s}{'待白平衡':>9s}{'剔除':>6s}{'就绪':>6s}")
    tot = Counter()
    for code in sorted(rows):
        r = rows[code]
        tot.update(r)
        print(f"{code:<5s}{COLONY_GROUPS.get(code, '—'):<12s}{r['photo_O']:>6d}{r['photo_R']:>6d}{r['need_ann']:>8d}"
              f"{r['need_wb']:>9d}{r['excluded']:>6d}{r['ready']:>6d}")
    print(f"{'合计':<15s}{tot['photo_O']:>6d}{tot['photo_R']:>6d}{tot['need_ann']:>8d}{tot['need_wb']:>9d}"
          f"{tot['excluded']:>6d}{tot['ready']:>6d}")
    print("（只统计正面 O：待标注 = 还没有几何标注；待白平衡 = 已标注但没确认白平衡参照；剔除 = 标为污染/不合格）")
    if tot["bad_json"]:
        print(f"⚠️  {tot['bad_json']} 个标注文件读不出来")
    if a.kv:
        with open(a.kv, "w") as f:
            f.write(f"NEED_ANN={tot['need_ann']}\nNEED_WB={tot['need_wb']}\nREADY={tot['ready']}\nPHOTO_O={tot['photo_O']}\n")
    return 0


def cmd_fresh(a):
    if not os.path.exists(a.manifest):
        print("NEWER=-1")
        return 0
    t = os.path.getmtime(a.manifest)
    n = 0
    for root in (a.micro_root, a.colony_root):
        if os.path.isdir(root):
            n += sum(1 for p in walk_imgs(root) if os.path.getmtime(p) > t)
    print(f"NEWER={n}")
    print(f"MANIFEST_TIME={dt.datetime.fromtimestamp(t):%Y-%m-%d %H:%M}")
    return 0


def cmd_smoke_subset(a):
    import pandas as pd
    csvp = OP.find_trainset_csv(a.trainset)
    df = pd.read_csv(csvp, encoding="utf-8-sig")
    g = df.groupby("train_class")["cv_group"].nunique().sort_values(ascending=False)
    classes = list(g[g >= 4].index[:3])
    if len(classes) < 2:
        print("❌ 训练清单里每类分组都太少，抽不出自检子集")
        return 1
    parts = []
    for c in classes:
        d = df[df["train_class"] == c]
        firsts = d.drop_duplicates("cv_group").head(4)
        parts.append(firsts)
    sub = pd.concat(parts)
    os.makedirs(a.out, exist_ok=True)
    sub.to_csv(os.path.join(a.out, "trainset_micro.csv"), index=False, encoding="utf-8-sig")
    shutil.copy(os.path.join(os.path.dirname(csvp), "class_names.json"), os.path.join(a.out, "class_names.json"))
    print(f"自检子集：{len(sub)} 张，{len(classes)} 类（{'、'.join(classes)}），来自 {csvp}")
    return 0


# =============================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="train_*.zsh 的图片检查与盘点工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *keys):
        if "tax" in keys:
            p.add_argument("--taxonomy", default=os.path.join(here, "taxonomy_map.csv"))
        if "recrop" in keys:
            p.add_argument("--recrop_root", required=True)
        if "colony" in keys:
            p.add_argument("--colony_raw", required=True)
            p.add_argument("--ann_dir", default=os.path.join(OP.TRAINSETS, "colony_annotations"))
        if "log" in keys:
            p.add_argument("--log", default=os.path.join(OP.TRAIN_AUDIT, "import_log.csv"))

    common(sub.add_parser("species"), "tax")
    p = sub.add_parser("inspect"); p.add_argument("--src", required=True)
    p = sub.add_parser("import-micro"); common(p, "tax", "recrop", "log")
    p.add_argument("--src", required=True); p.add_argument("--species", required=True)
    p.add_argument("--strain", default=""); p.add_argument("--subfolders", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="只检查、显示将复制什么，不复制")
    p = sub.add_parser("import-colony"); common(p, "colony", "log"); p.add_argument("--src", required=True)
    p.add_argument("--dry_run", action="store_true", help="只检查、显示将复制什么，不复制")
    p = sub.add_parser("audit"); common(p, "tax", "recrop", "colony")
    p.add_argument("--crops_v2", required=True); p.add_argument("--crops_new", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_raw", type=int, default=300); p.add_argument("--min_crop", type=int, default=64)
    p = sub.add_parser("crop-status"); common(p, "tax", "recrop")
    p.add_argument("--v2_log", required=True); p.add_argument("--tsv", default="")
    p = sub.add_parser("colony-status"); common(p, "colony"); p.add_argument("--kv", default="")
    p = sub.add_parser("fresh"); p.add_argument("--manifest", default=OP.TRAIN_MANIFEST)
    p.add_argument("--micro_root", required=True); p.add_argument("--colony_root", required=True)
    p = sub.add_parser("smoke-subset"); p.add_argument("--trainset", required=True); p.add_argument("--out", required=True)

    a = ap.parse_args()
    fn = {"species": cmd_species, "inspect": cmd_inspect, "import-micro": cmd_import_micro,
          "import-colony": cmd_import_colony, "audit": cmd_audit, "crop-status": cmd_crop_status,
          "colony-status": cmd_colony_status, "fresh": cmd_fresh, "smoke-subset": cmd_smoke_subset}[a.cmd]
    sys.exit(fn(a) or 0)


if __name__ == "__main__":
    main()
