#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_make_recrop_dirs.py —— 建立"原图_重截"文件夹结构，并随时查看找图 / 截图进度

    python 01_make_recrop_dirs.py            # 建文件夹（已存在的不动），写说明和截图命令
    python 01_make_recrop_dirs.py --status   # 查看进度：每个菌种已放了几个菌株、多少原图、已截多少张

结构（每个菌株一个子文件夹，子文件夹名 = 菌株编号或标本号，如 CUR01、2021-0356）：
    原图_重截/
    ├── 00_说明.txt            总说明、优先顺序、目标数量
    ├── 截图命令.zsh           每个菌种一条 01_crop_batch_training.py 命令
    ├── 根霉/
    │   ├── 说明.txt
    │   ├── RHI01/  xxx.jpg …
    │   └── RHI02/  …
    └── …
01_build_trainset.py 会把同一菌株文件夹截出的所有截图分在同一组（菌株级交叉验证）。
"""
import os
import csv
import argparse
from collections import Counter

ROOT = "/Users/chunjiang/Documents/真菌AI/原图_重截"
V2_LOG = "/Users/chunjiang/Documents/真菌AI/CNN/crops_v2/crop_log.csv"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# 文件夹名（= --label，必须是 taxonomy_map.csv 的 species_cn）, 训练类, 截图编号前缀, 组, 现状, 目标
PLAN = [
    # A 组：旧截图基本不能用，全部重截
    ("根霉",        "Rhizopus",                     "PUPH-RHI", "A", "旧截图 579 张全为局部",   "≥10 株，200 张；孢子囊 + 孢囊梗，尽量带假根"),
    ("毛霉",        "Mucor",                        "PUPH-MUC", "A", "旧截图 89 张仅 1 张合格", "≥10 株，150 张；孢子囊 + 孢囊梗（无假根）"),
    ("弯孢霉",      "Curvularia",                   "PUPH-CUR", "A", "旧截图 313 张全为局部",   "≥10 株，150 张；孢子梗 + 成簇弯曲分生孢子"),
    ("拟青霉",      "Paecilomyces",                 "PUPH-PAE", "A", "旧截图 213 张全为局部",   "≥10 株，150 张；帚状枝 + 颈部细长的瓶梗 + 孢子链"),
    ("裂褶菌",      "Schizophyllum",                "PUPH-SCH", "A", "旧截图 221 张全不合格",   "≥8 株，120 张；锁状联合、菌丝侧刺（针状突起）"),
    ("灰绿曲霉",    "Aspergillus glaucus (sect. Aspergillus)", "PUPH-AGL", "A", "旧截图 152 张全为局部", "≥8 株，120 张；完整分生孢子头；有闭囊壳可单独截"),
    # B 组：剔除问题图后基本够用，用原图补截以替换
    ("链格孢",      "Alternaria",                   "PUPH-ALT", "B", "合格 237/325",            "≥10 株，150 张；砖格状孢子 + 孢子链"),
    ("离蠕孢",      "Bipolaris",                    "PUPH-BIP", "B", "合格 135/218",            "≥8 株，120 张；孢子梗膝状弯曲 + 孢子"),
    ("尖端赛多孢",  "Scedosporium",                 "PUPH-SCE", "B", "合格 248/332",            "≥10 株，150 张；单生环痕孢子 + 孢子梗；波氏赛多孢也放这里（另建菌株文件夹）"),
    ("组织胞浆菌",  "Histoplasma capsulatum",       "PUPH-HIS", "B", "合格 120/193",            "≥5 株，100 张；齿轮状大分生孢子"),
    ("申克孢子丝菌", "Sporothrix schenckii complex", "PUPH-SPO", "B", "合格 302/310",            "≥10 株，150 张；梗顶端花朵状孢子簇"),
    ("多育赛多孢",  "Lomentospora prolificans",     "PUPH-LOM", "B", "合格 117/132",            "≥5 株，100 张；基部膨大的环痕梗；不要和尖端赛多孢混放"),
    ("横梗霉",      "Lichtheimia",                  "PUPH-LIC", "B", "合格 84/100",             "≥8 株，120 张；梨形孢子囊 + 囊托 + 分枝孢囊梗"),
    ("小克银汉霉",  "Cunninghamella",               "PUPH-CUN", "B", "合格 44/75",              "≥8 株，120 张；顶囊 + 单孢小型孢子囊"),
    ("杂色曲霉",    "Aspergillus versicolor (sect. Versicolores)", "PUPH-AVE", "B", "合格 40/46", "≥10 株，120 张；小顶囊、双层小梗"),
    # C 组：截图质量没问题，菌株太少，补充菌株
    ("构巢曲霉",    "Aspergillus nidulans (sect. Nidulantes)", "PUPH-ANI", "C", "38 张（12 组）", "≥10 株，120 张；短柱状头、双层小梗；壳细胞可单独截"),
    ("黑曲霉",      "Aspergillus niger (sect. Nigri)",         "PUPH-ANG", "C", "19 张（15 组）", "≥10 株，120 张；球形大顶囊、双层小梗、黑色孢子"),
    ("土曲霉",      "Aspergillus terreus (sect. Terrei)",      "PUPH-ATE", "C", "48 张（13 组）", "≥10 株，120 张；柱状头、双层小梗"),
    ("马尔尼菲篮状菌", "Talaromyces marneffei",               "PUPH-TMA", "C", "51 张（15 组）", "≥8 株，120 张；25℃ 菌丝相双轮生帚状枝 + 孢子链"),
]
GROUP_NAME = {"A": "A 组 · 全部重截", "B": "B 组 · 补截替换问题图", "C": "C 组 · 补充菌株"}

COMMON = """截图规范（与 01_crop_batch_training.py 一致）
  · 一个框 = 一个完整的产孢结构；用 + / - 调框，使结构占框面积约 60–80%，看 224 预览窗口确认
  · 结构标注：1 完整（默认）、2 局部、3 多个结构；i 切换 成熟 / 未成熟
  · 每张原图截 3–8 个；没有可用结构按 s 跳过
原图整理
  · 每个菌株一个子文件夹（菌株编号或标本号），同一菌株不同日期、不同视野的原图都放进去
  · 分不清菌株的原图：按拍摄批次建一个子文件夹（如 未知_2021批次1），分组会更保守
  · 只放纯培养镜检图（C 类）；标本直接镜检（A 类）、菌落图不要放
  · 原图用复制，不要剪切，原来的文件夹保持不动
"""


def write_if_absent(path, text, force=False):
    if force or not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


def create(a):
    os.makedirs(a.root, exist_ok=True)
    made = []
    cmds = ["#!/bin/zsh", "# 每个菌种一条：放好原图后，在脚本所在文件夹（fungal 环境）运行对应的一行", ""]
    overview = [f"原图_重截 · 说明（由 01_make_recrop_dirs.py 生成）\n", COMMON,
                "优先顺序：A 组 → C 组 → B 组（B 组现有合格截图已基本够用）\n"]
    for g in "ABC":
        overview.append(f"\n【{GROUP_NAME[g]}】")
        cmds.append(f"# ===== {GROUP_NAME[g]} =====")
        for name, cls, prefix, grp, now, goal in [p for p in PLAN if p[3] == g]:
            d = os.path.join(a.root, name)
            if not os.path.isdir(d):
                os.makedirs(d)
                made.append(name)
            cmd = (f'python 01_crop_batch_training.py --folder "{d}" --label {name} --code_prefix {prefix}')
            write_if_absent(os.path.join(d, "说明.txt"),
                            f"{name} → 训练类 {cls}（{GROUP_NAME[grp]}）\n现状：{now}\n目标：{goal}\n\n"
                            f"截图命令：\n  {cmd}\n\n{COMMON}", force=a.overwrite_notes)
            overview.append(f"  {name:<8s} {cls:<45s} 现状：{now:<20s} 目标：{goal}")
            cmds.append(cmd)
        cmds.append("")
    overview.append("\n全部截完后重建训练集：\n"
                    "  python 01_build_trainset.py --replace_classes all_v2 --review review_images.csv\n"
                    "  （all_v2：某类只要在 crops_v2 里有新截图，该类旧截图全部替换 —— 一个菌种截完、数量够了再重建）\n"
                    "查看进度：python 01_make_recrop_dirs.py --status")
    write_if_absent(os.path.join(a.root, "00_说明.txt"), "\n".join(overview), force=a.overwrite_notes)
    write_if_absent(os.path.join(a.root, "截图命令.zsh"), "\n".join(cmds) + "\n", force=a.overwrite_notes)
    print(f"根目录：{a.root}")
    print(f"新建 {len(made)} 个菌种文件夹：{'、'.join(made) if made else '（都已存在，未改动）'}")
    print("每个文件夹里有 说明.txt；总说明 00_说明.txt；全部截图命令 截图命令.zsh")
    print("下一步：按菌株建子文件夹、复制原图进去；随时用 --status 查看进度")


def status(a):
    done = Counter()
    if os.path.exists(a.v2_log):
        with open(a.v2_log, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("status") == "crop" and r.get("source_image", "").startswith(a.root.rstrip("/") + "/"):
                    rel = os.path.relpath(r["source_image"], a.root).split(os.sep)
                    done[rel[0]] += 1
    print(f"{'菌种':<10s}{'组':<3s}{'菌株':>5s}{'原图':>6s}{'已截':>6s}   目标")
    tot = Counter()
    for name, cls, prefix, grp, now, goal in PLAN:
        d = os.path.join(a.root, name)
        strains, n_img, loose = set(), 0, 0
        if os.path.isdir(d):
            for dirpath, subs, files in os.walk(d):
                subs[:] = [s for s in subs if not s.startswith(".")]
                imgs = [f for f in files if f.lower().endswith(IMG_EXTS) and not f.startswith(".")]
                if not imgs:
                    continue
                rel = os.path.relpath(dirpath, d)
                if rel == ".":
                    loose += len(imgs)
                else:
                    strains.add(rel.split(os.sep)[0])
                n_img += len(imgs)
        warn = f"  ⚠ {loose} 张原图直接放在菌种文件夹里，请移入菌株子文件夹" if loose else ""
        print(f"{name:<10s}{grp:<3s}{len(strains):>5d}{n_img:>6d}{done[name]:>6d}   {goal.split('；')[0]}{warn}")
        tot["strain"] += len(strains); tot["img"] += n_img; tot["crop"] += done[name]
    print(f"{'合计':<13s}{tot['strain']:>5d}{tot['img']:>6d}{tot['crop']:>6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--status", action="store_true", help="只查看进度，不建文件夹")
    ap.add_argument("--v2_log", default=V2_LOG)
    ap.add_argument("--overwrite_notes", action="store_true", help="重写各说明文件（默认已存在就不覆盖）")
    a = ap.parse_args()
    status(a) if a.status else create(a)


if __name__ == "__main__":
    main()
