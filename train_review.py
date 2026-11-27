#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_review.py —— 训练原图 / 数据集 / 模型的汇总与查看（只读；write-* 和 summary 只写说明文件）

一般不直接运行，由 train_review.zsh、train_4_dataset.zsh、train_5_micro.zsh、train_6_colony.zsh 调用。

子命令
  sources              各来源训练原图有多少张、在哪里
  preview  --dir D     临时建出的数据集里各来源可用的张数（建数据集前给用户选）
  write-micro-info     写 DATASET_INFO.json（显微数据集）并打印摘要
  write-colony-info    写 DATASET_INFO.json（菌落数据集）并打印摘要
  datasets --kind K    列出 micro / colony 数据集（--tsv 写出编号 → 路径）
  dataset  --path P    一个数据集的详细信息
  models               列出全部模型（--tsv 写出编号 → 路径）
  model    --path P    一个模型的详细信息
  summary  --run P     把该模型的详细信息写成 TRAIN_SUMMARY.md（存在模型根目录）
  cvtable  --root P    同一数据集下各 backbone 的 CV 对比 → cv_汇总.txt
"""
import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402

SRC = [("dev_original", "O", "旧截图"), ("dev_checked", "C", "✅ 外院投稿截图"), ("crops_new", "N", "网络补截"),
       ("crops_v2", "V", "规范截图"), ("roi_ext", "R", "外部盲法 ROI")]
CODE = {s: c for s, c, _ in SRC}
NAME = {s: n for s, _, n in SRC}
METRICS = [("acc", "准确率"), ("macro_f1", "macro-F1"), ("balanced_acc", "平衡准确率"), ("top2", "top-2")]


def pd():
    import pandas
    return pandas


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def mtime(p):
    return dt.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M") if os.path.exists(p) else "—"


def jload(p, default=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return default


def rel(p):
    try:
        return os.path.relpath(p, OP.OUT)
    except ValueError:
        return p


def usable(name):
    return not (name.startswith("_") or name.startswith(".") or "_FAILED" in name or name.startswith("_preview"))


# =============================================================================
#  1. 原图
# =============================================================================
def count_log(root, status="crop"):
    log = os.path.join(root, "crop_log.csv")
    if not os.path.exists(log):
        return None, None
    rows = [r for r in csv.DictReader(open(log, encoding="utf-8-sig")) if r.get("status") == status]
    n_ok = sum(1 for r in rows if r.get("crop_file") and os.path.exists(r["crop_file"]))
    return len(rows), n_ok


def count_imgs(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
    n = 0
    if os.path.isdir(root):
        for d, subs, files in os.walk(root):
            subs[:] = [s for s in subs if not s.startswith(".") and s not in ("annotations", "preprocessed")]
            n += sum(1 for f in files if os.path.splitext(f)[1].lower() in exts and not f.startswith("."))
    return n


def cmd_sources(a):
    P = pd()
    rows = []
    man = OP.TRAIN_MANIFEST
    n_orig = n_chk = n_col = None
    if os.path.exists(man):
        m = P.read_csv(man, encoding="utf-8-sig", dtype=str).fillna("")
        mic = m[m["modality"] == "micro"] if "modality" in m.columns else m
        if "group_rule" in m:
            n_chk = int((mic["group_rule"] == "contrib_folder").sum())
            n_orig = int(len(mic) - n_chk)
        if "modality" in m:
            n_col = int((m["modality"] == "colony").sum())
    rows.append(["O dev_original", "旧截图", n_orig if n_orig is not None else count_imgs(a.micro_root), a.micro_root,
                 f"CNN/dataset 全部旧截图（含本期不训练的类），按 train_manifest.csv（{mtime(man)}）" if n_orig is not None else "train_manifest.csv 不存在，按文件数"])
    rows.append(["C dev_checked", "✅ 外院投稿截图", n_chk if n_chk is not None else "—", a.micro_root, "文件名含 XXpics 的 ✅ 文件夹截图"])
    for code, name, root in (("N crops_new", "网络补截", a.crops_new), ("V crops_v2", "规范截图", a.crops_v2),
                             ("R roi_ext", "外部盲法 ROI", a.roi_test)):
        n, ok = count_log(root)
        rows.append([code, name, "—" if n is None else ok, root,
                     "没有 crop_log.csv" if n is None else (f"记录 {n} 张，{n - ok} 张文件缺失" if n != ok else "记录与文件一致")])
    rows.append(["—", "旧菌落图", n_col if n_col is not None else count_imgs(a.colony_legacy), a.colony_legacy, "菌落模型不用"])
    print("【截图来源】（建显微数据集时选的就是这 5 个）")
    print(OP.table(P.DataFrame(rows, columns=["代号 来源", "说明", "张数", "位置", "备注"]), index=False))

    # 原图_重截
    print("\n【待截图的显微原图】原图_重截：" + a.recrop_root)
    t = []
    done = Counter()
    v2 = os.path.join(a.crops_v2, "crop_log.csv")
    if os.path.exists(v2):
        for r in csv.DictReader(open(v2, encoding="utf-8-sig")):
            if r.get("status") == "crop" and r.get("source_image", "").startswith(a.recrop_root.rstrip("/") + "/"):
                done[os.path.relpath(r["source_image"], a.recrop_root).split(os.sep)[0]] += 1
    if os.path.isdir(a.recrop_root):
        for sp in sorted(os.listdir(a.recrop_root)):
            d = os.path.join(a.recrop_root, sp)
            if os.path.isdir(d) and not sp.startswith("."):
                strains = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)) and count_imgs(os.path.join(d, s))]
                n = count_imgs(d)
                if n:
                    t.append([sp, len(strains), n, done[sp]])
    print(OP.table(P.DataFrame(t, columns=["菌种", "菌株", "原图", "已截"]), index=False) if t else "  （空）")

    # 补拍
    print("\n【菌落原始照片】补拍_2026：" + a.colony_raw)
    import importlib
    ck = importlib.import_module("00_train_check")
    st = ck.colony_status(a.colony_raw, a.ann_dir)
    t = [[c, ck.COLONY_GROUPS.get(c, "—"), r["photo_O"], r["photo_R"], r["need_ann"], r["need_wb"], r["excluded"], r["ready"]]
         for c, r in sorted(st.items())]
    print(OP.table(P.DataFrame(t, columns=["组", "", "正面", "反面", "待标注", "待白平衡", "剔除", "就绪"]), index=False)
          if t else "  （空）")
    print(f"  标注保存在 {a.ann_dir}")


# =============================================================================
#  2. 数据集
# =============================================================================
def micro_stats(d):
    P = pd()
    df = P.read_csv(os.path.join(d, "trainset_micro.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    by_src = df["source"].value_counts().to_dict() if "source" in df else {}
    return df, by_src


def src_codes(by_src):
    return "".join(c for s, c, _ in SRC if by_src.get(s, 0))


def list_micro():
    out = []
    for d in sorted(glob.glob(os.path.join(OP.TRAINSETS, "*"))):
        if not os.path.isfile(os.path.join(d, "trainset_micro.csv")) or not usable(os.path.basename(d)):
            continue
        info = jload(os.path.join(d, "DATASET_INFO.json"), {})
        created = info.get("created") or mtime(os.path.join(d, "trainset_micro.csv"))
        out.append((created, d, info))
    return sorted(out, key=lambda x: x[0], reverse=True)


def list_colony():
    out = []
    for d in sorted(glob.glob(os.path.join(OP.TRAINSETS, "*"))):
        if not os.path.isfile(os.path.join(d, "manifest.csv")) or not usable(os.path.basename(d)):
            continue
        info = jload(os.path.join(d, "DATASET_INFO.json"), {})
        cfg = jload(os.path.join(d, "preprocess_config.json"), {})
        created = info.get("created") or (cfg.get("built_at", "")[:16].replace("T", " ") or mtime(os.path.join(d, "manifest.csv")))
        out.append((created, d, info))
    return sorted(out, key=lambda x: x[0], reverse=True)


def colony_stats(d):
    P = pd()
    m = P.read_csv(os.path.join(d, "manifest.csv"), encoding="utf-8-sig")
    o = m[m["view"] == "O"] if "view" in m else m
    main = o[o["main_analysis"].astype(bool)] if "main_analysis" in o else o
    return m, o, main


def cmd_datasets(a):
    P = pd()
    paths_ = []
    kinds = ["micro", "colony"] if a.kind == "all" else [a.kind]
    for kind in kinds:
        rows = []
        if kind == "micro":
            for created, d, info in list_micro():
                df, by = micro_stats(d)
                paths_.append(d)
                rows.append([len(paths_), os.path.basename(d), created, len(df), df["train_class"].nunique(),
                             df["cv_group"].nunique()] + [by.get(s, 0) for s, _, _ in SRC] + ["" if info else "旧版"])
            cols = ["编号", "数据集", "创建时间", "张数", "类别", "分组"] + [c for _, c, _ in SRC] + ["备注"]
            title = "【显微数据集】"
        else:
            for created, d, info in list_colony():
                m, o, main = colony_stats(d)
                g = main.groupby("group")["strain_id"].nunique().to_dict() if len(main) else {}
                paths_.append(d)
                rows.append([len(paths_), os.path.basename(d), created, main["strain_id"].nunique() if len(main) else 0,
                             len(main), "、".join(f"{k} {v}" for k, v in g.items()), "" if info else "旧版"])
            cols = ["编号", "数据集", "创建时间", "株（主分析）", "正面照", "各组株数", "备注"]
            title = "【菌落数据集】"
        if len(kinds) > 1:
            print(title)
        if not rows:
            print(f"（output/01_trainsets 下没有{'显微' if kind == 'micro' else '菌落'}数据集）")
        else:
            print(OP.table(P.DataFrame(rows, columns=cols), index=False))
            if kind == "micro":
                print("来源代号：" + "  ".join(f"{c} = {s}（{n}）" for s, c, n in SRC))
        if len(kinds) > 1:
            print()
    if a.tsv:
        open(a.tsv, "w", encoding="utf-8").write("".join(p + "\n" for p in paths_))


def show_micro_dataset(d):
    P = pd()
    info = jload(os.path.join(d, "DATASET_INFO.json"), {})
    df, by = micro_stats(d)
    L = [f"数据集：{os.path.basename(d)}    （{d}）",
         f"创建时间：{info.get('created') or mtime(os.path.join(d, 'trainset_micro.csv'))}"
         + ("" if info else "    ⚠️ 旧版数据集（没有 DATASET_INFO.json，以下由清单现算）"),
         f"显微截图 {len(df)} 张；类别 {df['train_class'].nunique()}；CV 分组 {df['cv_group'].nunique()}"]
    if info:
        L.append("纳入来源：" + "、".join(f"{CODE.get(s, s)} {s}" for s in info.get("sources", [])))
        o = info.get("options", {})
        L.append("参数：" + "；".join(f"{k} = {v}" for k, v in o.items()))
        if info.get("sha256"):
            L.append(f"trainset_micro.csv sha256：{info['sha256'].get('trainset_micro.csv', '')[:16]}…")
    L.append("各来源张数：" + "、".join(f"{CODE.get(k, '?')} {k} {v}" for k, v in by.items()))
    if "site" in df:
        L.append("按实验室（site）：" + "、".join(f"{k} {v}" for k, v in df["site"].value_counts().items()))
    t = df.groupby("train_class").agg(张数=("path", "size"), 分组=("cv_group", "nunique"))
    for s, c, _ in SRC:
        if by.get(s):
            t[c] = df[df["source"] == s].groupby("train_class").size().reindex(t.index).fillna(0).astype(int)
    if "species_cn" in df:
        t["中文名"] = df.groupby("train_class")["species_cn"].agg(lambda x: "、".join(sorted(set(v for v in x if v))[:2]))
    L += ["", OP.table(t)]
    few = t[t["分组"] < 5].index.tolist()
    if few:
        L.append(f"⚠️  CV 分组 < 5 的类别（5 折时不能每折都有）：{'、'.join(few)}")
    L.append(f"最多 / 最少类张数比：{t['张数'].max() / max(t['张数'].min(), 1):.1f}")
    col = os.path.join(d, "trainset_colony.csv")
    if os.path.exists(col):
        L.append(f"（同时生成的旧菌落图清单 trainset_colony.csv：{len(P.read_csv(col, encoding='utf-8-sig'))} 张，菌落模型不用）")
    return L


def show_colony_dataset(d):
    info = jload(os.path.join(d, "DATASET_INFO.json"), {})
    cfg = jload(os.path.join(d, "preprocess_config.json"), {})
    m, o, main = colony_stats(d)
    L = [f"菌落数据集：{os.path.basename(d)}    （{d}）",
         f"创建时间：{info.get('created') or cfg.get('built_at', '')[:16].replace('T', ' ') or mtime(os.path.join(d, 'manifest.csv'))}"
         + ("" if info else "    ⚠️ 旧版数据集（没有 DATASET_INFO.json）"),
         f"预处理版本：{cfg.get('version', '—')}",
         f"照片 {len(m)} 张（正面 {len(o)}）；进入主分析的正面照 {len(main)} 张，来自 {main['strain_id'].nunique() if len(main) else 0} 株"]
    if info.get("source_root"):
        L.append(f"原始照片：{info['source_root']}；标注：{info.get('ann_dir', '')}")
    if len(o):
        t = o.groupby("group").agg(正面照=("strain_id", "size"), 株=("strain_id", "nunique"))
        t["主分析株"] = main.groupby("group")["strain_id"].nunique().reindex(t.index).fillna(0).astype(int)
        if "wb_mode" in o:
            t["白纸校正"] = o[o["wb_mode"] == "paper"].groupby("group").size().reindex(t.index).fillna(0).astype(int)
        L += ["", OP.table(t)]
    excl = o[~o["main_analysis"].astype(bool)] if "main_analysis" in o else o.iloc[0:0]
    if len(excl):
        fl = excl["flags"].fillna("").astype(str).str.split(";").explode().str.strip()
        fl = fl[fl != ""].value_counts()
        L.append(f"未进入主分析 {len(excl)} 张：" + ("、".join(f"{k} {v}" for k, v in fl.items()) or "（未注明原因）"))
    dup = os.path.join(d, "duplicates_check.csv")
    if os.path.exists(dup):
        import pandas
        dd = pandas.read_csv(dup)
        if len(dd) and "inliers" in dd:
            L.append(f"疑似同一平皿的照片对 {int((dd['inliers'] >= 300).sum())} 对（≥300 内点，训练时合并为同一 CV 组）")
    L.append(f"质控拼图：{os.path.join(d, 'qc_sheets')}")
    return L


def cmd_dataset(a):
    d = a.path.rstrip("/")
    L = show_micro_dataset(d) if os.path.exists(os.path.join(d, "trainset_micro.csv")) else show_colony_dataset(d)
    print("\n".join(L))


def cmd_preview(a):
    _, by = micro_stats(a.dir)
    lines = []
    for i, (s, c, n) in enumerate(SRC, 1):
        lines.append(f"{s}={by.get(s, 0)}")
    open(a.kv, "w").write("\n".join(lines) + "\n")


def cmd_write_micro(a):
    P = pd()
    d = a.dir
    df, by = micro_stats(d)
    opts = json.loads(a.options)
    info = {"kind": "micro", "name": os.path.basename(d), "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created_by": "train_4_dataset.zsh", "sources": a.sources, "source_codes": "".join(CODE[s] for s in a.sources),
            "excluded_sources": [s for s, _, _ in SRC if s not in a.sources], "options": opts, "build_command": a.command,
            "n_images": int(len(df)), "n_classes": int(df["train_class"].nunique()), "n_groups": int(df["cv_group"].nunique()),
            "by_source": {k: int(v) for k, v in by.items()},
            "by_site": {k: int(v) for k, v in df["site"].value_counts().items()} if "site" in df else {},
            "by_class": {c: {"n": int(len(g)), "groups": int(g["cv_group"].nunique())} for c, g in df.groupby("train_class")},
            "sha256": {f: sha256(os.path.join(d, f)) for f in ("trainset_micro.csv", "class_names.json") if os.path.exists(os.path.join(d, f))},
            "builder_sha256": sha256(os.path.join(OP.BASE, "01_build_trainset.py"))}
    json.dump(info, open(os.path.join(d, "DATASET_INFO.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(show_micro_dataset(d)))


def cmd_write_colony(a):
    d = a.dir
    m, o, main = colony_stats(d)
    info = {"kind": "colony", "name": os.path.basename(d), "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "created_by": "train_4_dataset.zsh", "source_root": a.root, "ann_dir": a.ann_dir, "build_command": a.command,
            "n_photos": int(len(m)), "n_front": int(len(o)), "n_main": int(len(main)),
            "n_strains_main": int(main["strain_id"].nunique()) if len(main) else 0,
            "strains_by_group": {k: int(v) for k, v in main.groupby("group")["strain_id"].nunique().items()} if len(main) else {},
            "sha256": {"manifest.csv": sha256(os.path.join(d, "manifest.csv"))}}
    json.dump(info, open(os.path.join(d, "DATASET_INFO.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(show_colony_dataset(d)))


# =============================================================================
#  3. 模型
# =============================================================================
def model_kind(d):
    if os.path.exists(os.path.join(d, "locked_config.json")):
        return "colony"
    if os.path.exists(os.path.join(d, "cv_summary.json")):
        return "cv"
    if os.path.exists(os.path.join(d, "final.pt")):
        return "full"
    return None


def model_dataset(d):
    args = jload(os.path.join(d, "args.json"), {})
    ts = args.get("trainset", "")
    if ts:
        return os.path.basename(os.path.dirname(ts)) if ts.endswith(".csv") else os.path.basename(ts.rstrip("/"))
    lc = jload(os.path.join(d, "locked_config.json"), {})
    pre = (lc.get("args") or {}).get("pre", "")
    return os.path.basename(str(pre).rstrip("/")) if pre else "—"


def model_time(d, k):
    if k == "colony":
        return (jload(os.path.join(d, "locked_config.json"), {}).get("locked_at", "")[:16].replace("T", " ")) or mtime(d)
    f = os.path.join(d, "cv_summary.json" if k == "cv" else "final.pt")
    return mtime(f)


def all_models():
    out = []
    if not os.path.isdir(OP.RUNS_ROOT):
        return out
    for d, subs, files in os.walk(OP.RUNS_ROOT):
        subs[:] = sorted(s for s in subs if usable(s) and not s.startswith("fold_") and s != "feature_cache")
        k = model_kind(d)
        if k:
            out.append((model_time(d, k), d, k))
            subs[:] = []
    return sorted(out, key=lambda x: x[0], reverse=True)


def headline(d, k):
    if k == "cv":
        s = jload(os.path.join(d, "cv_summary.json"), {})
        return f"macro-F1 {s.get('macro_f1_mean', float('nan')):.3f}±{s.get('macro_f1_sd', float('nan')):.3f}  acc {s.get('acc_mean', float('nan')):.3f}"
    if k == "full":
        sc = sanity_acc(d)
        return f"自检 acc {sc}" if sc else "（无自检记录）"
    lc = jload(os.path.join(d, "locked_config.json"), {})
    sel = lc.get("selected_method", "?")
    sm = (lc.get("summary") or {}).get(sel, {})
    return f"方案 {sel}  top1 {sm.get('top1', float('nan')):.3f}  NLL {sm.get('nll', float('nan')):.3f}"


def sanity_acc(d):
    p = os.path.join(d, "sanity_check.txt")
    if not os.path.exists(p):
        return ""
    import re
    m = re.findall(r"(?:准确率|acc(?:uracy)?)[^0-9]*([01]\.\d+)", open(p, encoding="utf-8", errors="ignore").read())
    return m[-1] if m else ""


def backbone_of(d, k):
    if k == "colony":
        lc = jload(os.path.join(d, "locked_config.json"), {})
        return "颜色+直径" if lc.get("selected_method") == "A" else (lc.get("args") or {}).get("backbone", "—")
    return jload(os.path.join(d, "args.json"), {}).get("backbone") or os.path.basename(d).split("_")[0]


def cmd_models(a):
    P = pd()
    rows, paths_ = [], []
    KN = {"cv": "显微 CV", "full": "显微 最终", "colony": "菌落"}
    for i, (t, d, k) in enumerate(all_models(), 1):
        legacy = "" if os.path.exists(os.path.join(d, "TRAIN_SUMMARY.md")) else "旧版"
        rows.append([i, KN[k], rel(d).replace("02_training_runs/", ""), backbone_of(d, k), model_dataset(d), t, headline(d, k), legacy])
        paths_.append(d)
    if rows:
        print(OP.table(P.DataFrame(rows, columns=["编号", "类型", "位置（02_training_runs/ 下）", "backbone", "数据集", "时间", "主要结果", "备注"]), index=False))
    else:
        print("（output/02_training_runs 下没有模型）")
    if a.tsv:
        open(a.tsv, "w", encoding="utf-8").write("".join(p + "\n" for p in paths_))


def show_model(d):
    P = pd()
    k = model_kind(d)
    L = []
    if k in ("cv", "full"):
        args = jload(os.path.join(d, "args.json"), {})
        L += [f"{'显微 交叉验证' if k == 'cv' else '显微 最终模型'}：{d}",
              f"backbone：{args.get('backbone', '—')}    数据集：{model_dataset(d)}（{args.get('trainset', '—')}）",
              f"时间：{model_time(d, k)}",
              "训练参数：" + "  ".join(f"{x}={args.get(x)}" for x in ("folds", "epochs", "patience", "batch", "lr",
                                                                      "weight_decay", "img_size", "seed") if x in args)]
        ds_info = jload(os.path.join(os.path.dirname(args.get("trainset", "")), "DATASET_INFO.json"), {})
        if ds_info.get("sha256"):
            L.append(f"数据集指纹（训练时的 trainset_micro.csv）：{ds_info['sha256'].get('trainset_micro.csv', '')[:16]}…")
    if k == "cv":
        s = jload(os.path.join(d, "cv_summary.json"), {})
        L.append(f"完成折数：{len(s.get('folds_done', []))}；各折最佳 epoch 中位数 {s.get('median_best_epoch', float('nan')):.0f}"
                 "（最终模型的 epoch 数）")
        t = P.DataFrame([[n, f"{s.get(c + '_mean', float('nan')):.3f} ± {s.get(c + '_sd', float('nan')):.3f}",
                          (s.get("pooled_oof") or {}).get(c, float("nan"))] for c, n in METRICS],
                        columns=["指标", "各折均值 ± SD", "合并 OOF"])
        L += ["", OP.table(t, index=False)]
        fc = os.path.join(d, "cv_folds.csv")
        if os.path.exists(fc):
            L += ["", "各折：", OP.table(P.read_csv(fc)[["fold", "n_val", "acc", "macro_f1", "top2", "best_epoch"]], index=False)]
        pc = os.path.join(d, "per_class.csv")
        if os.path.exists(pc):
            c = P.read_csv(pc, index_col=0, encoding="utf-8-sig")
            c = c[~c.index.isin(["accuracy", "macro avg", "weighted avg"])]
            c = c[c["support"] > 0].sort_values("f1-score")
            L += ["", "F1 最低的 5 类：", OP.table(c[["precision", "recall", "f1-score", "support"]].head(5))]
        for f, n in (("error_summary.txt", "错误分析"), ("errors.html", "错误图片网页"), ("confusion_matrix.png", "混淆矩阵图"),
                     ("oof_by_site.txt", "按来源拆分"), ("train_terminal.log", "训练输出")):
            if os.path.exists(os.path.join(d, f)):
                L.append(f"{n}：{os.path.join(d, f)}")
    elif k == "full":
        args = jload(os.path.join(d, "args.json"), {})
        L.append(f"训练 epoch：{args.get('epochs')}（取自同数据集 CV 的最佳 epoch 中位数）")
        shf = os.path.join(d, "model_sha256.txt")
        L.append(f"final.pt sha256：{open(shf).read().split()[0][:16] + '…' if os.path.exists(shf) else '（无 model_sha256.txt）'}")
        sc = os.path.join(d, "sanity_check.txt")
        if os.path.exists(sc):
            lines = [x for x in open(sc, encoding="utf-8", errors="ignore").read().splitlines() if x.strip()]
            i0 = next((i for i, x in enumerate(lines) if "训练截图抽样" in x), None)
            if i0 is not None:
                i1 = next((i for i, x in enumerate(lines[i0:], i0) if "预测类别分布" in x), len(lines))
                acc = sanity_acc(d)
                flag = "" if not acc else ("  ✅" if float(acc) >= 0.95 else "  ⚠️ 低于 0.95：模型载入 / 类别顺序 / 预处理可能有问题")
                L += ["", "自检：" + lines[i0].strip() + flag] + ["  " + x for x in lines[i0 + 1:i1]]
        cvd = os.path.join(os.path.dirname(d), f"{args.get('backbone', '')}_cv")
        if os.path.exists(os.path.join(cvd, "cv_summary.json")):
            s = jload(os.path.join(cvd, "cv_summary.json"), {})
            L.append(f"\n对应的交叉验证 {cvd}：macro-F1 {s['macro_f1_mean']:.3f} ± {s['macro_f1_sd']:.3f}，acc {s['acc_mean']:.3f} ± {s['acc_sd']:.3f}")
        L.append(f"\n临床验证引用写法（clin_config.zsh 的 MICRO_MODELS）：{rel(d).replace('02_training_runs/', '')}")
    elif k == "colony":
        lc = jload(os.path.join(d, "locked_config.json"), {})
        a_ = lc.get("args") or {}
        sel = lc.get("selected_method", "?")
        L += [f"菌落模型：{d}", f"锁定时间：{lc.get('locked_at', '—')}    版本：{lc.get('version', '—')}",
              f"数据集：{model_dataset(d)}（{a_.get('pre', '—')}）",
              f"类别：{'、'.join(lc.get('classes', []))}；{lc.get('n_strains', '—')} 株，{lc.get('n_cv_groups', '—')} 个 CV 组（留一组交叉验证）",
              f"选定方案 {sel}（规则：A、B 中株级 OOF NLL 最低者）；温度 T = {lc.get('temperature', float('nan')):.3f}",
              ("方案 B：未跑（--skip_b）" if a_.get("skip_b") else f"方案 B 骨干网络：{a_.get('backbone', '—')}")
              + f"；min_strains={a_.get('min_strains')}；dup_inliers={a_.get('dup_inliers')}"]
        rows = [[m, v.get("top1"), v.get("top2"), v.get("bal_acc"), v.get("nll"), v.get("ece")]
                for m, v in (lc.get("summary") or {}).items()]
        if rows:
            L += ["", OP.table(P.DataFrame(rows, columns=["方案", "top1", "top2", "平衡准确率", "NLL", "ECE"]), index=False)]
        sh = (lc.get("sha256") or {}).get("final_model.joblib", "")
        L.append(f"final_model.joblib sha256：{sh[:16]}…")
        L.append(f"完整报告：{os.path.join(d, 'metrics.txt')}")
        L.append(f"\n临床验证引用写法（clin_config.zsh 的 COLONY_MODEL）：{rel(d).replace('02_training_runs/', '')}")
    else:
        L.append(f"{d} 不是可识别的模型文件夹")
    return L


def cmd_model(a):
    print("\n".join(show_model(a.path.rstrip("/"))))


def cmd_summary(a):
    d = a.run.rstrip("/")
    L = show_model(d)
    txt = "\n".join(L)
    md = [f"# 训练总结：{os.path.basename(d)}", "",
          f"- 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M}（train_review.py summary）",
          f"- 运行记录：RUN_RECORD.md（本文件夹）", "", "```", txt, "```", ""]
    open(os.path.join(d, "TRAIN_SUMMARY.md"), "w", encoding="utf-8").write("\n".join(md))
    print(txt)
    print(f"\n已保存 {os.path.join(d, 'TRAIN_SUMMARY.md')}")


def cmd_cvtable(a):
    P = pd()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*_cv", "cv_summary.json"))):
        s = jload(f, {})
        rows.append([s.get("backbone", os.path.basename(os.path.dirname(f)))] + [len(s.get("folds_done", []))]
                    + [f"{s.get(c + '_mean', float('nan')):.3f} ± {s.get(c + '_sd', float('nan')):.3f}" for c, _ in METRICS]
                    + [s.get("median_best_epoch")])
    if not rows:
        print("（没有完成的交叉验证）")
        return
    t = P.DataFrame(rows, columns=["backbone", "折数"] + [n for _, n in METRICS] + ["最佳epoch中位数"])
    txt = (f"数据集 {os.path.basename(a.root)}：各 backbone 菌株级 {rows[0][1]} 折交叉验证（{dt.datetime.now():%Y-%m-%d %H:%M}）\n\n"
           + OP.table(t, index=False) + "\n\n各折均值 ± SD；StratifiedGroupKFold，按 cv_group 分组")
    open(os.path.join(a.root, "cv_汇总.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="训练原图 / 数据集 / 模型的汇总与查看")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sources")
    for k in ("micro_root", "colony_legacy", "crops_new", "crops_v2", "roi_test", "recrop_root", "colony_raw"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--ann_dir", default=os.path.join(OP.TRAINSETS, "colony_annotations"))
    p = sub.add_parser("preview"); p.add_argument("--dir", required=True); p.add_argument("--kv", required=True)
    p = sub.add_parser("write-micro-info"); p.add_argument("--dir", required=True)
    p.add_argument("--sources", nargs="+", required=True); p.add_argument("--options", required=True)
    p.add_argument("--command", default="")
    p = sub.add_parser("write-colony-info"); p.add_argument("--dir", required=True); p.add_argument("--root", required=True)
    p.add_argument("--ann_dir", default=""); p.add_argument("--command", default="")
    p = sub.add_parser("datasets"); p.add_argument("--kind", choices=["micro", "colony", "all"], required=True)
    p.add_argument("--tsv", default="")
    p = sub.add_parser("dataset"); p.add_argument("--path", required=True)
    p = sub.add_parser("models"); p.add_argument("--tsv", default="")
    p = sub.add_parser("model"); p.add_argument("--path", required=True)
    p = sub.add_parser("summary"); p.add_argument("--run", required=True)
    p = sub.add_parser("cvtable"); p.add_argument("--root", required=True)
    a = ap.parse_args()
    {"sources": cmd_sources, "preview": cmd_preview, "write-micro-info": cmd_write_micro,
     "write-colony-info": cmd_write_colony, "datasets": cmd_datasets, "dataset": cmd_dataset,
     "models": cmd_models, "model": cmd_model, "summary": cmd_summary, "cvtable": cmd_cvtable}[a.cmd](a)


if __name__ == "__main__":
    main()
