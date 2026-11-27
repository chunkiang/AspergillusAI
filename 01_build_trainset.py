#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_trainset.py —— 按 26 类方案整合开发集，生成训练清单

输入：
  train_audit/train_manifest.csv   00_audit_train.py 的输出（清理后重跑的那一份）
  taxonomy_map.csv                 菌名 → 训练标签（以赵老师确认的版本为准）
  CNN/crops_new/crop_log.csv       01_crop_batch.py 的截图记录
  CNN/crops_v2/crop_log.csv        01_crop_batch_training.py 的规范化截图记录
  CNN/roi_test/crop_log.csv        03_roi_blind.py 的外部图盲法 ROI（加 --add_roi 才纳入；答案从 ext_audit/manifest.csv 取）

输出（默认在 ./trainset/）：
  trainset_micro.csv     显微截图训练清单：path, train_class, class_idx, group_id, source, ...
  trainset_colony.csv    菌落图清单（次要分析用）
  class_names.json       类别顺序（训练、预测共用）
  summary.txt            每类张数、分组数、来源构成

用法：
  python 01_build_trainset.py
  python 01_build_trainset.py --min_crop 64      # 新截图中边长 < 64 像素的不用（默认 64）
  python 01_build_trainset.py --replace_classes all_v2 --add_roi --out trainset_roi   # 加入外部 ROI，另存一套清单
  python 01_build_trainset.py --replace_classes all_v2 --review review_images.csv    # 剔除"不合格/局部"，保留"合格"旧截图 + 加入新截图

  # 2026-10-21 起的主训练集：不用 ✅ 文件夹截图（dev_checked，外院投稿图、非盲截取）
  python 01_build_trainset.py --replace_classes all_v2 --review review_images.csv --exclude_sources dev_checked --out trainset_v4

source 列（截图来源）：dev_original 旧截图 / dev_checked ✅ 文件夹截图 / crops_new 网络补截 / crops_v2 规范截图 / roi_ext 外部 ROI
  --exclude_sources 按这一列整类剔除；--exclude_sites 按下面的 site 列剔除

site 列（数据来源 / 实验室）：本院开发集 = PUPH，其余按原图所在文件夹（谭盼、胡音音、网络图…），供以后按来源留一交叉验证
  ✅ 文件夹截图（dev_checked）按文件名里的投稿者/文件夹名标记（申克 → SK、蒋宇杰 → JYJ …；认不出的记为 CONTRIB），
  不再记为 PUPH（2026-10-21 之前的版本会把它们记成 PUPH）

分组（group_id，交叉验证时同组图片不拆开）：
  · 文件名能读出菌株/原图的（烟曲霉-02、tile_15、✅ 文件夹、W-B-14、crop_batch 的原图编号）→ 按菌株/原图分组
  · 其余追溯不到菌株的旧截图 → 每张图单独一组（内部交叉验证因此偏乐观，只用于选模型，论文主结论看外部验证）
  · 近似重复的图强制同组
"""
import os
import re
import sys
import json
import argparse
from collections import Counter

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

# 旧数据集类别编号 → 图库中的名称（与 labels.txt 一致；用于查 taxonomy_map）
OLD_LABELS = {
    "00": "土曲霉", "01": "链格孢", "02": "阿萨希毛孢子菌", "03": "诺卡革兰染色", "04": "布鲁菌革兰染色",
    "05": "黑曲霉", "06": "黄曲霉", "07": "尖端赛多孢", "08": "烟曲霉", "09": "微小根霉", "10": "白念珠菌",
    "11": "光滑念珠菌", "12": "近平滑念珠菌", "13": "克柔念珠菌", "14": "葡萄牙念珠菌", "15": "热带念珠菌",
    "16": "季也蒙念珠菌", "17": "阿沙丝孢酵母", "18": "产脘念珠菌", "19": "耳念珠菌", "20": "链状念珠菌",
    "21": "罗伦特隐球菌", "22": "酿酒酵母", "23": "溶脂念珠菌", "24": "土生念珠菌", "25": "希木龙念珠菌",
    "26": "褶皱念珠菌", "27": "根霉", "28": "构巢", "29": "马尔尼菲篮状菌", "30": "青霉", "31": "少根根霉",
    "32": "镰刀菌", "33": "多育赛多孢", "34": "申克孢子丝菌", "35": "小克银汉", "36": "帚霉", "37": "弯孢霉",
    "38": "拟青霉", "39": "无绿藻", "40": "杂色曲霉", "41": "毛霉", "42": "灰绿曲霉", "43": "着色霉",
    "44": "离蠕孢", "45": "组织胞浆菌", "46": "裂褶菌", "47": "波氏赛多孢", "48": "鬼伞菌", "49": "犁头霉",
    "50": "隐球菌",
}
MICRO_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/dataset"
COLONY_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/normal"
CROP_LOG = "/Users/chunjiang/Documents/真菌AI/CNN/crops_new/crop_log.csv"
V2_LOG = "/Users/chunjiang/Documents/真菌AI/CNN/crops_v2/crop_log.csv"
ROI_LOG = "/Users/chunjiang/Documents/真菌AI/CNN/roi_test/crop_log.csv"
RECROP_ROOT = "/Users/chunjiang/Documents/真菌AI/原图_重截"   # <菌种>/<菌株编号>/原图：crops_v2 按菌株分组
EXT_ROOT = "/Users/chunjiang/Desktop/丝状真菌"
SITE_TAG = {"网络真菌图片": "WEB1", "网络真菌图2": "WEB2", "谭盼": "TP", "胡音音": "HYY", "马琼琼": "MQQ",
            "李一凡": "LYF", "李艳": "LY", "田蕴琦": "TYQ", "麦老师": "MCW", "HSJ": "HSJ", "蒋宇杰": "JYJ",
            "申克": "SK", "已整理": "PUPH", "eXPORT": "PUPH"}


def site_of(path, ext_root=EXT_ROOT):
    """由原图路径得到来源（实验室）代码；不在 ext_root 下的视为本院"""
    rel = os.path.relpath(path, ext_root) if path.startswith(ext_root) else ""
    top = rel.split(os.sep)[0] if rel else ""
    for k, v in SITE_TAG.items():
        if k in top:
            return v
    return top or "PUPH"


def site_of_contrib(fname):
    """✅ 文件夹截图：文件名形如 "<外部文件夹名>_<N>pics<菌名>_<序号>"，按文件夹名里的投稿者认来源"""
    for k, v in SITE_TAG.items():
        if k in fname and v != "PUPH":
            return v
    return "CONTRIB"


def load_review(path):
    """04_review_train.py 导出的 review_images.csv → (按路径查标记, 按 类别|文件名 查标记)"""
    import unicodedata
    rv = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    if "path" not in rv or "tag" not in rv:
        sys.exit(f"{path} 里没有 path / tag 列，请用 04_review_train.py 导出的 review_images.csv")
    rv = rv[rv["tag"] != ""].copy()
    rv["_p"] = rv["path"].map(norm_path)
    src = rv["source"] if "source" in rv else ""
    rv["_k"] = rv["train_class"] + "|" + src + "|" + rv["path"].map(lambda x: os.path.basename(str(x).strip()))
    uniq = rv.drop_duplicates("_k", keep=False)
    return rv, dict(zip(rv["_p"], rv["tag"])), dict(zip(uniq["_k"], uniq["tag"]))


def norm_path(x):
    import unicodedata
    return unicodedata.normalize("NFC", os.path.abspath(os.path.expanduser(str(x).strip())))


def lookup_tags(df, by_path, by_name):
    t1 = df["path"].map(norm_path).map(by_path)
    src = df["source"].fillna("") if "source" in df else ""
    t2 = (df["train_class"] + "|" + src + "|" + df["path"].map(os.path.basename)).map(by_name)
    return t1.fillna(t2).fillna(""), t1, t2


def under(path, root):
    return bool(root) and os.path.abspath(path).startswith(os.path.abspath(root).rstrip("/") + "/")


def strain_group(path, code, root):
    """重截原图在 <root>/<菌种>/<菌株>/… 下：同一菌株文件夹的所有截图同组；否则按原图分组"""
    if under(path, root):
        rel = os.path.relpath(os.path.dirname(os.path.abspath(path)), os.path.abspath(root))
        return "v2strain|" + "/".join(rel.split(os.sep)[:2])      # 菌种/菌株（更深的子文件夹不再拆组）
    return "v2|" + code


def case_ids(df):
    """同一文件夹、同一菌种、编号连续（相差 ≤ 1）的图归为同一病例（与 03_eval_external.py 一致）"""
    num = df["rel_path"].map(lambda p: re.search(r"[-_](\d+)\D*$", os.path.splitext(os.path.basename(p))[0]))
    df = df.assign(_n=[int(m.group(1)) if m else -1 for m in num])
    ids = {}
    for (folder, sp), d in df.groupby(["folder", "species_cn"]):
        d = d.sort_values("_n")
        k, prev = 0, None
        for i, n in zip(d.index, d["_n"]):
            if n < 0 or prev is None or prev < 0 or n - prev > 1:
                k += 1
            ids[i] = f"{folder}|{sp}|{k}"
            prev = n
    return df.index.map(ids)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", default=OP.TRAIN_MANIFEST)
    ap.add_argument("--taxonomy", default=os.path.join(here, "taxonomy_map.csv"))
    ap.add_argument("--crop_log", default=CROP_LOG)
    ap.add_argument("--micro_root", default=MICRO_ROOT)
    ap.add_argument("--colony_root", default=COLONY_ROOT)
    ap.add_argument("--min_crop", type=int, default=64, help="新截图边长小于此值的不用")
    ap.add_argument("--out", default="trainset", help="只写名字时放进 output/01_trainsets/")
    ap.add_argument("--no_check", action="store_true", help="不检查文件是否存在（调试用）")
    ap.add_argument("--v2_log", default=V2_LOG, help="01_crop_batch_training.py 的规范化截图记录")
    ap.add_argument("--replace_classes", nargs="*", default=[],
                    help="这些类的旧截图（原有 + ✅）不再使用，只用 crops_v2 的规范化截图；写 all_v2 表示 crops_v2 里出现的所有类")
    ap.add_argument("--drop_web_crops", action="store_true", help="被替换的类同时不用 crops_new（网络补截）")
    ap.add_argument("--exclude_immature", action="store_true", help="不用标为“未成熟”的 crops_v2 截图")
    ap.add_argument("--recrop_root", default=RECROP_ROOT,
                    help="重截原图的总文件夹；其下 <菌种>/<菌株>/ 的原图截出的 crops_v2 按菌株分组")
    ap.add_argument("--review", default="", help="04_review_train.py 导出的 review_images.csv：按人工核对结果剔除截图")
    ap.add_argument("--strict_replace", action="store_true",
                    help="与 --review 合用时，被替换类别中标为'合格'的旧截图默认保留；加此项则一律替换")
    ap.add_argument("--drop_tags", nargs="*", default=["不合格", "局部"], help="剔除哪些标记（默认 不合格 局部）")
    ap.add_argument("--exclude_sites", nargs="*", default=[],
                    help="整个来源不进训练集（如 TP），用于按来源留一验证：之后只在该来源的 ROI 上测试")
    ap.add_argument("--exclude_sources", nargs="*", default=[],
                    choices=["dev_original", "dev_checked", "crops_new", "crops_v2", "roi_ext"],
                    help="按截图来源（source 列）整类剔除，如 dev_checked = ✅ 文件夹截图（外院投稿图，非盲截取）")
    ap.add_argument("--add_roi", action="store_true", help="加入 03_roi_blind.py 截取的外部 ROI（只取本期 26 类）")
    ap.add_argument("--roi_log", default=ROI_LOG)
    ap.add_argument("--manifest", default=OP.MANIFEST, help="外部图答案（00_build_manifest.py 输出）")
    ap.add_argument("--ext_root", default=EXT_ROOT)
    a = ap.parse_args()
    a.train_manifest = OP.find(a.train_manifest, OP.TRAIN_AUDIT, OP.AUDIT)
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    a.out = OP.place(a.out, OP.TRAINSETS)
    if a.review:
        a.review = OP.find(a.review, OP.AUDIT, OP.BASE)
    for p in (a.train_manifest, a.taxonomy):
        if not os.path.exists(p):
            sys.exit(f"找不到 {p}")
    os.makedirs(a.out, exist_ok=True)

    tax = pd.read_csv(a.taxonomy, encoding="utf-8-sig", dtype=str).fillna("")
    to_label = dict(zip(tax["species_cn"], tax["train_class"]))
    phase = dict(zip(tax["species_cn"], tax["note"]))
    class_names = sorted({c for c in tax["train_class"] if c})
    cidx = {c: i for i, c in enumerate(class_names)}

    # ---------- 1. 原有截图 + ✅ 截图（来自 train_manifest） ----------
    m = pd.read_csv(a.train_manifest, encoding="utf-8-sig", dtype=str).fillna("")
    m["cls"] = m["cls"].str.zfill(2)
    unknown = sorted(set(m["cls"]) - set(OLD_LABELS))
    if unknown:
        sys.exit(f"train_manifest 中有未知的类别编号：{unknown}")
    m["species_cn"] = m["cls"].map(OLD_LABELS)
    missing = sorted({s for s in m["species_cn"] if s not in to_label})
    if missing:
        sys.exit(f"这些名称不在 taxonomy_map.csv 中：{missing}")
    m["train_class"] = m["species_cn"].map(to_label)
    m["phase"] = m["species_cn"].map(phase)
    m["source"] = m["group_rule"].map(lambda r: "dev_checked" if r == "contrib_folder" else "dev_original")
    root = {"micro": a.micro_root, "colony": a.colony_root}
    m["path"] = [os.path.join(root[mo], rp) for mo, rp in zip(m["modality"], m["rel_path"])]
    # 追溯不到菌株的旧图：每张单独一组；能追溯的保留菌株/原图分组
    m["cv_group"] = [f"img|{p}" if rule == "date_mtime" else g
                     for p, rule, g in zip(m["rel_path"], m["group_rule"], m["group_id"])]
    # 近似重复强制同组
    near = m[m["near_dup_of"] != ""]
    if len(near):
        g_of = dict(zip(m["rel_path"], m["cv_group"]))
        for i, r in near.iterrows():
            m.at[i, "cv_group"] = g_of.get(r["near_dup_of"], r["cv_group"])
    m["crop_size"] = m["width"]
    excluded_old = m[m["train_class"] == ""]
    m = m[m["train_class"] != ""]

    # ---------- 2. crop_batch 新截图 ----------
    new = pd.DataFrame()
    if a.crop_log and os.path.exists(a.crop_log):
        cl = pd.read_csv(a.crop_log, encoding="utf-8-sig", dtype=str).fillna("")
        cl = cl[cl["status"] == "crop"].copy()
        cl["crop_size"] = cl["w"].astype(int)
        small = cl[cl["crop_size"] < a.min_crop]
        cl = cl[cl["crop_size"] >= a.min_crop]
        new = pd.DataFrame({
            "modality": "micro", "path": cl["crop_file"], "file": cl["crop_file"].map(os.path.basename),
            "species_cn": cl["species_cn"], "train_class": cl["train_class"],
            "phase": cl["species_cn"].map(phase), "source": "crops_new",
            "cv_group": "crop|" + cl["source_code"], "group_rule": "crop_source", "crop_size": cl["crop_size"],
            "angle": cl.get("angle", "0"),
            "site": cl["source_image"].map(lambda p: site_of(p, a.ext_root)),
        })
        bad = sorted(set(new["train_class"]) - set(class_names))
        if bad:
            sys.exit(f"crop_log 中的类别不在 taxonomy_map 的训练标签里：{bad}")
        print(f"新截图：采用 {len(new)} 张，舍弃边长 < {a.min_crop} 的 {len(small)} 张")
    else:
        print(f"未找到 {a.crop_log}，不加入新截图")

    # ---------- 3. crop_batch_training 规范化截图（crops_v2） ----------
    v2 = pd.DataFrame()
    if a.v2_log and os.path.exists(a.v2_log):
        vl = pd.read_csv(a.v2_log, encoding="utf-8-sig", dtype=str).fillna("")
        vl = vl[vl["status"] == "crop"].copy()
        if a.exclude_immature and "stage" in vl:
            n0 = len(vl)
            vl = vl[vl["stage"] != "未成熟"]
            print(f"crops_v2：排除未成熟截图 {n0 - len(vl)} 张")
        v2 = pd.DataFrame({
            "modality": "micro", "path": vl["crop_file"], "train_class": vl["train_class"],
            "species_cn": vl["species_cn"], "phase": vl["species_cn"].map(phase), "source": "crops_v2",
            "cv_group": [strain_group(p, c, a.recrop_root) for p, c in zip(vl["source_image"], vl["source_code"])],
            "group_rule": ["crop_v2_strain" if under(p, a.recrop_root) else "crop_v2_source" for p in vl["source_image"]],
            "crop_size": vl["w"].astype(int), "structure": vl.get("structure", ""), "stage": vl.get("stage", ""),
            "site": vl["source_image"].map(lambda p: site_of(p, a.ext_root)),
        })
        bad = sorted(set(v2["train_class"]) - set(class_names))
        if bad:
            sys.exit(f"crops_v2 日志中的类别不在 taxonomy_map 的训练标签里：{bad}")
        print(f"crops_v2：{len(v2)} 张规范化截图，来自 {vl['source_image'].nunique()} 张原图")
    replace = set(a.replace_classes)
    if "all_v2" in replace:
        replace = (replace - {"all_v2"}) | set(v2["train_class"]) if len(v2) else replace - {"all_v2"}
    if replace:
        unknown = replace - set(class_names)
        if unknown:
            sys.exit(f"--replace_classes 中有未知类别：{sorted(unknown)}")
        n0 = len(m)
        rm = m["train_class"].isin(replace) & (m["modality"] == "micro")      # 菌落图保留
        kept_ok = 0
        if a.review and not a.strict_replace:
            _, bp, bn = load_review(a.review)
            ok_old = lookup_tags(m, bp, bn)[0] == "合格"
            kept_ok = int((rm & ok_old).sum())
            rm = rm & ~ok_old
        m = m[~rm]
        print(f"替换类别 {len(replace)} 个：去掉旧截图 {n0 - len(m)} 张"
              + (f"；人工核对为'合格'的旧截图保留 {kept_ok} 张" if a.review and not a.strict_replace else ""))
        if a.drop_web_crops and len(new):
            n1 = len(new)
            new = new[~new["train_class"].isin(replace)]
            print(f"  同时去掉网络补截 {n1 - len(new)} 张")

    # ---------- 4. 外部图盲法 ROI（03_roi_blind.py） ----------
    roi = pd.DataFrame()
    if a.add_roi:
        if not os.path.exists(a.roi_log) or not os.path.exists(a.manifest):
            sys.exit(f"--add_roi 需要 {a.roi_log} 和 {a.manifest}")
        rl = pd.read_csv(a.roi_log, encoding="utf-8-sig", dtype=str).fillna("")
        rl = rl[rl["status"] == "crop"].copy()
        em = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
        em = em[(em["role"] == "external_micro") & (em["train_class"] != "")].copy()
        em["case_id"] = case_ids(em)
        root = a.ext_root.rstrip("/") + "/"
        rl["rel_path"] = [s_[len(root):] if s_.startswith(root) else s_ for s_ in rl["source_image"]]
        n_all = len(rl)
        rl = rl.merge(em[["rel_path", "folder", "species_cn", "train_class", "case_id"]]
                      .rename(columns={"species_cn": "sp", "train_class": "tc"}), on="rel_path", how="inner")
        rl["crop_size"] = rl["w"].astype(int)
        n_small = int((rl["crop_size"] < a.min_crop).sum())
        rl = rl[rl["crop_size"] >= a.min_crop]
        roi = pd.DataFrame({
            "modality": "micro", "path": rl["crop_file"], "train_class": rl["tc"], "species_cn": rl["sp"],
            "phase": rl["sp"].map(phase), "source": "roi_ext",
            "cv_group": "roi|" + rl["case_id"], "group_rule": "roi_case",      # 同一病例（菌株）的所有 ROI 同组
            "crop_size": rl["crop_size"], "structure": rl.get("structure", ""), "stage": rl.get("stage", ""),
            "site": rl["source_image"].map(lambda p: site_of(p, a.ext_root)),
        })
        bad = sorted(set(roi["train_class"]) - set(class_names))
        if bad:
            sys.exit(f"外部 ROI 的类别不在 taxonomy_map 的训练标签里：{bad}")
        print(f"外部 ROI：日志共 {n_all} 个；集内 26 类 {len(roi) + n_small} 个，采用 {len(roi)} 个"
              f"（舍弃边长 < {a.min_crop} 的 {n_small} 个；集外菌种的 ROI 不纳入）")
        print("  来自 " + "、".join(f"{k} {v}" for k, v in roi["site"].value_counts().items()))

    m["site"] = ["PUPH" if s != "dev_checked" else site_of_contrib(os.path.basename(p))
                 for s, p in zip(m["source"], m["path"])]
    dc = m[(m["source"] == "dev_checked") & (m["modality"] == "micro")]
    if len(dc):
        print("✅ 文件夹截图（dev_checked）来源：" + "、".join(f"{k} {v}" for k, v in dc["site"].value_counts().items()))
    # 旧截图里文件名带投稿者编号的（如 W-B-14_…），仍记为本院；列出来供人工确认是否其实是外院图
    cc = m[(m["source"] == "dev_original") & (m["group_rule"] == "contrib_code") & (m["modality"] == "micro")]
    if len(cc):
        print(f"⚠️  旧截图中有 {len(cc)} 张文件名带投稿者编号（记为 PUPH，请确认是否为本院图）："
              + "、".join(f"{k} {v}" for k, v in cc["cv_group"].str.split("-").str[0].value_counts().items()))
    cols = ["path", "modality", "train_class", "species_cn", "source", "site", "cv_group", "group_rule",
            "crop_size", "phase", "structure", "stage"]
    allm = pd.concat([m.reindex(columns=cols), new.reindex(columns=cols), v2.reindex(columns=cols),
                      roi.reindex(columns=cols)], ignore_index=True)
    if a.review:
        rv, bp, bn = load_review(a.review)
        allm["review_tag"], t1, t2 = lookup_tags(allm, bp, bn)
        n_full, n_name = int(t1.notna().sum()), int((t1.isna() & t2.notna()).sum())
        drop = allm["review_tag"].isin(a.drop_tags)
        print(f"人工核对：{a.review} 中有标记的 {len(rv)} 张；与训练清单对上 {n_full + n_name} 张"
              f"（完整路径 {n_full}，按类别+文件名 {n_name}）")
        print(f"  剔除标为 {a.drop_tags} 的截图 {int(drop.sum())} 张（核对表中共 {int(rv['tag'].isin(a.drop_tags).sum())} 张，"
              f"其余已被 --replace_classes 替换掉或文件已不存在）；新截图未核对，全部保留")
        if len(rv) and n_full + n_name == 0:
            print("  ⚠️  一张都没对上！核对表示例路径：", rv["path"].iloc[0])
            print("      训练清单示例路径：", allm["path"].iloc[0])
        if drop.any():
            print("  " + "、".join(f"{k} {v}" for k, v in allm[drop].groupby("train_class").size().items()))
        allm = allm[~drop]
    if a.exclude_sites:
        ex = allm["site"].isin(a.exclude_sites)
        print(f"按来源留出：去掉来源 {a.exclude_sites} 的全部截图 {int(ex.sum())} 张"
              f"（{'、'.join(f'{k} {v}' for k, v in allm[ex].groupby('source').size().items())}）")
        allm = allm[~ex]
    excluded_src = pd.DataFrame()
    if a.exclude_sources:
        ex = allm["source"].isin(a.exclude_sources)
        excluded_src = allm[ex & (allm["modality"] == "micro")]
        print(f"按截图来源剔除：{a.exclude_sources} 共 {int(ex.sum())} 张"
              + (f"（{'、'.join(f'{k} {v}' for k, v in excluded_src.groupby('site').size().items())}）"
                 if len(excluded_src) else ""))
        if len(excluded_src):
            print("  按类别：" + "、".join(f"{k} {v}" for k, v in excluded_src.groupby("train_class").size().items()))
        allm = allm[~ex]
    allm["class_idx"] = allm["train_class"].map(cidx).astype(int)
    allm["group_size"] = allm.groupby("cv_group")["path"].transform("size")
    if not a.no_check:
        exists = allm["path"].map(os.path.exists)
        if (~exists).any():
            print(f"⚠️  {int((~exists).sum())} 个文件不存在（已从清单中去掉），前 10 个：")
            for p in allm.loc[~exists, "path"].head(10):
                print("   ", p)
            allm = allm[exists]

    micro = allm[allm["modality"] == "micro"].reset_index(drop=True)
    colony = allm[allm["modality"] == "colony"].reset_index(drop=True)
    micro.to_csv(os.path.join(a.out, "trainset_micro.csv"), index=False, encoding="utf-8-sig")
    colony.to_csv(os.path.join(a.out, "trainset_colony.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # ---------- 汇总 ----------
    lines = ["命令: python 01_build_trainset.py " + " ".join(sys.argv[1:]),
             f"类别数: {len(class_names)}    显微截图: {len(micro)}    菌落图: {len(colony)}", ""]
    if len(excluded_src):
        lines.insert(1, f"按截图来源剔除 {a.exclude_sources}：显微截图 {len(excluded_src)} 张（"
                     + "、".join(f"{k} {v}" for k, v in excluded_src.groupby("site").size().items()) + "）")
    t = micro.groupby("train_class").agg(
        总数=("path", "size"),
        原有=("source", lambda s: int((s == "dev_original").sum())),
        打勾截图=("source", lambda s: int((s == "dev_checked").sum())),
        新截图=("source", lambda s: int((s == "crops_new").sum())),
        规范截图=("source", lambda s: int((s == "crops_v2").sum())),
        外部ROI=("source", lambda s: int((s == "roi_ext").sum())),
        分组数=("cv_group", "nunique"),
        可追溯分组=("group_rule", lambda s: int((s != "date_mtime").sum())),
    )
    t["菌落图"] = colony.groupby("train_class").size().reindex(t.index).fillna(0).astype(int)
    t["阶段"] = micro.groupby("train_class")["phase"].first().str[:1]
    t = t.reindex(class_names).fillna(0)
    lines.append(OP.table(t))
    lines.append("")
    lines.append("按来源（site）：" + "、".join(f"{k} {v}" for k, v in micro["site"].value_counts().items()))
    lines.append(f"最多/最少类张数比: {t['总数'].max() / max(t['总数'].min(), 1):.1f}")
    empty = [c for c in class_names if t.at[c, "总数"] == 0]
    few = [f"{c}({int(t.at[c, '分组数'])})" for c in class_names if 0 < t.at[c, "分组数"] < 5]
    if empty:
        lines.append("⚠️  没有任何显微截图的类别（5 折 CV 会出问题，需补图或从 taxonomy_map 中去掉）: " + "、".join(empty))
    if few:
        lines.append("⚠️  CV 分组数 < 5 的类别（StratifiedGroupKFold 无法每折都有）: " + "、".join(few))
    ex = Counter(excluded_old["species_cn"])
    lines.append("本期不纳入、已排除的旧类别（张数）: " + "、".join(f"{k} {v}" for k, v in ex.most_common()))
    txt = "\n".join(lines)
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    print(f"\n已写出 {a.out}/trainset_micro.csv, trainset_colony.csv, class_names.json, summary.txt")


if __name__ == "__main__":
    main()
