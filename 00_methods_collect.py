#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_methods_collect.py —— 为论文 Materials and Methods 收集需要核对的数字和参数（只读，不改任何原有文件）

一般不直接运行，由 methods_collect.zsh 调用（脚本和参数都写在 zsh 里，并记入总账）。

收集 6 项（对应 Methods 初稿末页的核对清单）：
  ① 开发集株数          显微数据集的分组数 / 可追溯菌株数、原图_重截 的菌株文件夹、菌落数据集株数；图片拍摄年份（EXIF，仅供参考）
  ② 显微数据集信息      DATASET_INFO.json（没有就现算），并注明来自哪个模型、哪个数据集
  ③ 显微训练参数        args.json + 02_train.py 中的优化器 / 学习率 / 损失函数 / 数据增强 / 早停相关代码行
  ④ 菌落模型            locked_config.json + metrics.txt（纳入的节、每节株数、选中的方案），并核对模型指纹
  ⑤ 软件版本与电脑型号  两个 python 环境的包版本、conda list、Mac 型号
  ⑥ 外部图像集来源      各投稿来源的图片数（医院名称无法由程序获得，生成待填表）

输出（--out 指定的文件夹）
  METHODS_INFO.md              全部结果的汇总（发给 Claude 的就是这个文件）
  methods_info.json            同样内容的机器可读版本
  table1_classes.csv           各类别的截图数、分组数（Table 1 草稿）
  external_sources_待填.csv    外部图像来源表，需要人工填写医院名称
  copies/                      原始文件的副本：args.json、02_train.py、DATASET_INFO.json、locked_config.json、metrics.txt、conda_list.txt …
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

VERSION = "00_methods_collect v1.0 (2026-11-10)"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
SRC_NAME = {"dev_original": "O 旧截图", "dev_checked": "C ✅外院投稿截图", "crops_new": "N 网络补截",
            "crops_v2": "V 规范截图", "roi_ext": "R 外部盲法ROI"}

# ---------------------------------------------------------------- 屏幕输出
COLOR = sys.stdout.isatty() or bool(os.environ.get("TRAIN_FORCE_COLOR"))


def c(code, s):
    return f"\033[{code}m{s}\033[0m" if COLOR else s


REPORT = []          # 写进 METHODS_INFO.md 的行
TODO = []            # 仍需人工处理的事项
STATUS = []          # (段落, 状态 emoji, 一句话)
DATA = {}            # methods_info.json


def out(s=""):
    print(s)
    REPORT.append(re.sub(r"\033\[[0-9;]*m", "", s))


def section(emoji, n, title):
    line = "═" * 68
    print()
    print(c("1;97;44", line))
    print(c("1;97;44", f"  {emoji}  第 {n} 段：{title}"))
    print(c("1;97;44", line))
    REPORT.extend(["", f"## {emoji} 第 {n} 段：{title}", ""])


def ok(s):
    out(c("1;32", f"✅ {s}"))


def info(s):
    out(c("1;36", f"ℹ️  {s}"))


def warn(s):
    out(c("1;33", f"⚠️  {s}"))


def err(s):
    out(c("1;31", f"❌ {s}"))


def hint(s):
    out(c("1;33", f"   💡 {s}"))


def need(s):
    """需要人工填写 / 确认"""
    out(c("1;35", f"✍️  需人工：{s}"))
    TODO.append(s)


def kv(k, v):
    out(f"   {k}：{v}")


def table(rows, header):
    if not rows:
        out("   （空）")
        return

    def w(s):
        return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))
    widths = [max(w(r[i]) for r in [header] + rows) for i in range(len(header))]

    def fmt(r):
        return "   " + "  ".join(str(x) + " " * (widths[i] - w(x)) for i, x in enumerate(r))
    out(fmt(header))
    out("   " + "  ".join("─" * x for x in widths))
    for r in rows:
        out(fmt(r))


# ---------------------------------------------------------------- 工具
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
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def keep_copy(src, name=None):
    if src and os.path.isfile(src):
        os.makedirs(COPIES, exist_ok=True)
        shutil.copy2(src, os.path.join(COPIES, name or os.path.basename(src)))
        return True
    return False


def resolve(sub, value):
    """短名字 → output/<sub>/<名字>；绝对路径或已存在的路径原样使用"""
    if not value:
        return ""
    p = os.path.expanduser(value)
    if os.path.isabs(p) or os.path.exists(p):
        return os.path.abspath(p)
    return os.path.join(A.out_root, sub, value)


def count_imgs(root):
    n = 0
    for d, subs, files in os.walk(root):
        subs[:] = [s for s in subs if not s.startswith(".")]
        n += sum(1 for f in files if os.path.splitext(f)[1].lower() in IMG_EXT and not f.startswith("."))
    return n


def run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def guarded(n, name, fn):
    try:
        fn()
        if not any(s[0] == n for s in STATUS):
            STATUS.append((n, "✅", name))
    except Exception as e:   # 一段出错不影响其他段
        err(f"这一段出错：{type(e).__name__}: {e}")
        hint("把这一段的屏幕输出发给 Claude")
        STATUS.append((n, "❌", f"{name}（出错：{type(e).__name__}）"))


def mark(n, emoji, text):
    STATUS[:] = [s for s in STATUS if s[0] != n]
    STATUS.append((n, emoji, text))


# ---------------------------------------------------------------- 定位：模型 → 数据集
def locate():
    L = {}
    L["micro_model"] = resolve("02_training_runs", A.micro_model)
    args = jload(os.path.join(L["micro_model"], "args.json"), {}) or {}
    L["micro_args"] = args
    ts = str(args.get("trainset", "") or "")
    cand = []
    if ts:
        d = os.path.dirname(ts) if ts.endswith(".csv") else ts.rstrip("/")
        cand += [d, os.path.join(A.out_root, "01_trainsets", os.path.basename(d))]
    if A.micro_trainset:
        cand.append(resolve("01_trainsets", A.micro_trainset))
    L["micro_dataset"] = next((d for d in cand if os.path.isfile(os.path.join(d, "trainset_micro.csv"))), "")
    L["micro_dataset_from"] = "模型的 args.json" if ts and L["micro_dataset"] and os.path.basename(L["micro_dataset"]) == os.path.basename(
        (os.path.dirname(ts) if ts.endswith(".csv") else ts).rstrip("/")) else "methods_collect.zsh 里的 MICRO_TRAINSET（args.json 里没有或找不到）"
    bb = args.get("backbone") or os.path.basename(L["micro_model"]).split("_")[0]
    L["backbone"] = bb
    parent = os.path.dirname(L["micro_model"])
    cvs = [os.path.join(parent, f"{bb}_cv")] + sorted(
        (os.path.join(parent, x) for x in (os.listdir(parent) if os.path.isdir(parent) else []) if x.startswith(f"{bb}_cv")), reverse=True)
    L["micro_cv"] = next((d for d in cvs if os.path.isfile(os.path.join(d, "cv_summary.json"))), "")
    L["colony_model"] = resolve("02_training_runs", A.colony_model)
    lock = jload(os.path.join(L["colony_model"], "locked_config.json"), {}) or {}
    L["colony_lock"] = lock
    pre = str((lock.get("args") or {}).get("pre", "") or "")
    cand = [pre, os.path.join(A.out_root, "01_trainsets", os.path.basename(pre.rstrip("/")))] if pre else []
    L["colony_dataset"] = next((d for d in cand if os.path.isfile(os.path.join(d, "manifest.csv"))), "")
    return L


# =============================================================================
#  ① 开发集株数
# =============================================================================
def exif_years(root, limit=4000):
    """读图片 EXIF 拍摄时间。返回 (有 EXIF 的张数, 检查的张数, 最早, 最晚, 各年张数)"""
    try:
        from PIL import Image
    except ImportError:
        return None
    n = have = 0
    dates = []
    for d, subs, files in os.walk(root):
        subs[:] = [s for s in subs if not s.startswith(".")]
        for f in files:
            if os.path.splitext(f)[1].lower() not in {".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif"} or f.startswith("."):
                continue
            if n >= limit:
                break
            n += 1
            try:
                with Image.open(os.path.join(d, f)) as im:
                    ex = im.getexif()
                    v = None
                    try:
                        v = ex.get_ifd(0x8769).get(0x9003)      # DateTimeOriginal
                    except Exception:
                        pass
                    v = v or ex.get(0x0132)                      # DateTime
                if v and re.match(r"\d{4}:\d{2}:\d{2}", str(v)) and not str(v).startswith("0000"):
                    have += 1
                    dates.append(str(v)[:10].replace(":", "-"))
            except Exception:
                pass
    if not dates:
        return have, n, None, None, {}
    return have, n, min(dates), max(dates), dict(sorted(Counter(x[:4] for x in dates).items()))


def sec1():
    section("🧫", 1, "开发集株数（清单第 2 项）")
    res = {}
    d = LOC["micro_dataset"]
    if not d:
        err("找不到显微数据集（trainset_micro.csv）")
        hint("检查 methods_collect.zsh 里的 MICRO_MODEL / MICRO_TRAINSET")
        mark(1, "❌", "开发集株数：找不到显微数据集")
        return
    rows = read_csv(os.path.join(d, "trainset_micro.csv"))
    cols = rows[0].keys() if rows else []
    info(f"显微数据集：{os.path.basename(d)}（来自{LOC['micro_dataset_from']}）")
    kv("位置", d)
    groups = {(r.get("train_class", ""), r.get("cv_group", "")) for r in rows}
    kv("截图总数", len(rows))
    kv("类别数", len({r.get('train_class') for r in rows}))
    kv("交叉验证分组数（cv_group）", len({r.get("cv_group") for r in rows}))
    res.update(n_crops=len(rows), n_groups=len({r.get("cv_group") for r in rows}))

    # 按截图来源拆分：哪些分组等于真实菌株
    out()
    info("按截图来源拆分（分组是否等于真实菌株，取决于来源）：")
    by = defaultdict(lambda: [0, set()])
    for r in rows:
        s = r.get("source", "?")
        by[s][0] += 1
        by[s][1].add(r.get("cv_group"))
    note = {"crops_v2": "按 原图_重截/<菌种>/<菌株>/ 截取 → 分组 = 菌株（可追溯）",
            "dev_original": "旧截图 → 分组多为推断（文件名 / 近似重复聚类），多数不能追溯到菌株",
            "crops_new": "网络图片 → 分组 = 原图，不是本院菌株",
            "dev_checked": "外院投稿截图（按约定不应进入训练集）", "roi_ext": "外部 ROI（按约定不应进入最终模型的训练集）"}
    table([[SRC_NAME.get(s, s), v[0], len(v[1]), note.get(s, "")] for s, v in sorted(by.items())],
          ["来源", "截图数", "分组数", "说明"])
    res["by_source"] = {s: {"crops": v[0], "groups": len(v[1])} for s, v in by.items()}
    for bad in ("dev_checked", "roi_ext"):
        if bad in by:
            warn(f"数据集里含 {SRC_NAME[bad]} {by[bad][0]} 张：与 Methods 初稿的写法（这部分不进训练集）不一致，请核对")
    if "group_rule" in cols:
        gr = defaultdict(set)
        for r in rows:
            gr[r.get("group_rule", "")].add(r.get("cv_group"))
        out()
        info("按分组规则（group_rule）拆分：")
        table([[k or "（空）", len(v)] for k, v in sorted(gr.items())], ["分组规则", "分组数"])
        res["by_group_rule"] = {k: len(v) for k, v in gr.items()}
    n_v2 = len(by["crops_v2"][1]) if "crops_v2" in by else 0
    n_all = res["n_groups"]
    out()
    kv("可追溯到菌株的分组（规范截图 V）", f"{n_v2} / {n_all}（{n_v2 / max(n_all, 1):.0%}）")
    res["traceable_groups"] = n_v2
    if n_v2 / max(n_all, 1) < 0.8:
        warn("可追溯菌株的分组不到 80%：论文里不宜直接写 isolate-level cross-validation，需写明其余按原图 / 近似重复分组")
    if "site" in cols:
        st = Counter(r.get("site", "") for r in rows)
        kv("按实验室（site）", "、".join(f"{k} {v}" for k, v in st.most_common()))
        res["by_site"] = dict(st)

    # 原图_重截：菌种 / 菌株文件夹
    out()
    info(f"原图_重截 文件夹（{A.recrop_root}）：")
    t, n_str = [], 0
    if os.path.isdir(A.recrop_root):
        for sp in sorted(os.listdir(A.recrop_root)):
            p = os.path.join(A.recrop_root, sp)
            if os.path.isdir(p) and not sp.startswith("."):
                strains = [s for s in sorted(os.listdir(p)) if os.path.isdir(os.path.join(p, s)) and not s.startswith(".")
                           and count_imgs(os.path.join(p, s))]
                if strains or count_imgs(p):
                    t.append([sp, len(strains), count_imgs(p)])
                    n_str += len(strains)
        table(t, ["菌种文件夹", "菌株数", "原图数"])
        kv("合计", f"{len(t)} 个菌种文件夹，{n_str} 株，{sum(x[2] for x in t)} 张原图")
        res["recrop"] = {"species_folders": len(t), "strains": n_str, "images": sum(x[2] for x in t)}
    else:
        warn("文件夹不存在，跳过")

    # 菌落数据集株数
    out()
    cd = LOC["colony_dataset"]
    if cd:
        m = [r for r in read_csv(os.path.join(cd, "manifest.csv")) if r.get("view", "O") == "O"]
        main = [r for r in m if str(r.get("main_analysis", "True")).lower() in ("true", "1")]
        g = defaultdict(set)
        for r in main:
            g[r.get("group", "?")].add(r.get("strain_id"))
        info(f"菌落数据集：{os.path.basename(cd)}（来自菌落模型的 locked_config.json）")
        table([[k, len(v)] for k, v in sorted(g.items())], ["节", "株数（进入主分析）"])
        n_col = len({r.get("strain_id") for r in main})
        kv("菌落株数合计", f"{n_col} 株（正面照 {len(main)} 张；另有 {len(m) - len(main)} 张未进入主分析）")
        res["colony_strains"] = {k: len(v) for k, v in g.items()}
        res["colony_strains_total"] = n_col
        micro_ids = {str(r.get("cv_group", "")) for r in rows} | {str(r.get("strain_id", "")) for r in rows if "strain_id" in cols}
        both = {r.get("strain_id") for r in main} & micro_ids
        kv("同时出现在显微分组名里的菌落株号", f"{len(both)} 个（按名字完全相同判断，命名不同会漏计）")
        res["paired_by_name"] = len(both)
    else:
        warn("找不到菌落数据集（manifest.csv），跳过菌落株数")

    # 拍摄年份（EXIF）
    out()
    info("图片拍摄时间（读 EXIF，仅供参考；显微相机常不写 EXIF，批量复制后的文件修改时间不可靠）：")
    ex = {}
    for label, root in (("显微原图（原图_重截）", A.recrop_root), ("菌落照片（补拍_2026）", A.colony_raw)):
        if not os.path.isdir(root):
            continue
        r = exif_years(root)
        if r is None:
            warn("当前 python 没有 pillow，跳过 EXIF")
            break
        have, n, lo, hi, years = r
        if have:
            kv(label, f"检查 {n} 张，{have} 张有拍摄时间：{lo} 至 {hi}；各年张数 {years}")
        else:
            kv(label, f"检查 {n} 张，都没有拍摄时间")
        ex[label] = {"checked": n, "with_exif": have, "earliest": lo, "latest": hi, "by_year": years}
    res["exif"] = ex
    out()
    need("开发集菌株的收集起止年份：程序只能给出 EXIF 拍摄时间作参考，菌株分离年份请查 LIS 或菌种库登记")
    need("保藏菌株的来源（本院菌种库名称）")
    if n_v2 / max(n_all, 1) < 0.8:
        need("不能追溯菌株的旧截图大约对应多少株（如能从登记本估计）；否则论文只报告“分组数”")
    DATA["1_isolates"] = res


# =============================================================================
#  ② 显微数据集信息
# =============================================================================
def sec2():
    section("🗂️", 2, "显微数据集 DATASET_INFO.json（清单第 5 项）")
    d = LOC["micro_dataset"]
    if not d:
        err("找不到显微数据集")
        mark(2, "❌", "显微数据集信息：找不到数据集")
        return
    info("来源链（请人工核对这条链是否就是论文要报告的那次训练）：")
    kv("显微最终模型", f"{LOC['micro_model']}（final.pt 修改时间 {mtime(os.path.join(LOC['micro_model'], 'final.pt'))}）")
    kv("它用的数据集", f"{d}（依据：{LOC['micro_dataset_from']}）")
    csvp = os.path.join(d, "trainset_micro.csv")
    cur = sha256(csvp)
    kv("trainset_micro.csv", f"修改时间 {mtime(csvp)}，sha256 {cur[:16]}…")
    rows = read_csv(csvp)
    ip = os.path.join(d, "DATASET_INFO.json")
    info_j = jload(ip)
    if info_j:
        ok(f"找到 DATASET_INFO.json（创建于 {info_j.get('created', '—')}，由 {info_j.get('created_by', '—')} 生成）")
        keep_copy(ip, "DATASET_INFO.json")
        old = (info_j.get("sha256") or {}).get("trainset_micro.csv", "")
        if old and old != cur:
            warn("trainset_micro.csv 的指纹与 DATASET_INFO.json 记录的不一致：数据集建好后清单被改动过，请核对")
        elif old:
            ok("清单指纹与 DATASET_INFO.json 记录一致")
        kv("纳入来源", "、".join(info_j.get("sources", [])) or "—")
        kv("排除来源", "、".join(info_j.get("excluded_sources", [])) or "—")
        kv("建数据集参数", json.dumps(info_j.get("options", {}), ensure_ascii=False))
        if info_j.get("build_command"):
            kv("建数据集命令", info_j["build_command"])
    else:
        warn("这个数据集没有 DATASET_INFO.json（旧版数据集，如 trainset_v4）。以下由清单现算，另存到输出文件夹，不写回数据集")
        info_j = {"kind": "micro", "name": os.path.basename(d), "created": mtime(csvp), "created_by": "00_methods_collect.py（现算）",
                  "n_images": len(rows), "n_classes": len({r.get("train_class") for r in rows}),
                  "n_groups": len({r.get("cv_group") for r in rows}),
                  "by_source": dict(Counter(r.get("source", "?") for r in rows)),
                  "by_site": dict(Counter(r.get("site", "?") for r in rows)) if rows and "site" in rows[0] else {},
                  "sha256": {"trainset_micro.csv": cur}}
        os.makedirs(COPIES, exist_ok=True)
        hint("旧版数据集没有记录建数据集时的参数（--replace_classes、--review 等）。可在 Audit/output/06_logs 里找当时 01_build_trainset 的日志")
        need(f"确认数据集 {os.path.basename(d)} 建立时用的参数（是否 --replace_classes all_v2 --review review_images.csv，纳入了哪些来源）")
    # 各类别
    by = defaultdict(lambda: {"n": 0, "g": set(), "src": Counter(), "cn": set()})
    for r in rows:
        b = by[r.get("train_class", "?")]
        b["n"] += 1
        b["g"].add(r.get("cv_group"))
        b["src"][r.get("source", "?")] += 1
        if r.get("species_cn"):
            b["cn"].add(r["species_cn"])
    srcs = [s for s in SRC_NAME if any(b["src"].get(s) for b in by.values())]
    out()
    info(f"各类别截图数与分组数（Table 1 草稿；共 {len(rows)} 张、{len(by)} 类、{info_j.get('n_groups', '—')} 个分组）：")
    t = [[k, "、".join(sorted(b["cn"])[:2]), b["n"], len(b["g"])] + [b["src"].get(s, 0) for s in srcs] for k, b in sorted(by.items())]
    hdr = ["类别", "中文名", "截图", "分组"] + [SRC_NAME[s].split()[0] for s in srcs]
    table(t, hdr)
    with open(os.path.join(A.out, "table1_classes.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_class", "中文名", "n_crops", "n_groups"] + srcs)
        w.writerows(t)
    few = [k for k, b in by.items() if len(b["g"]) < 5]
    if few:
        warn(f"分组少于 5 的类别（5 折交叉验证时不能每折都有）：{'、'.join(sorted(few))}")
    ns = [b["n"] for b in by.values()]
    kv("最多 / 最少类截图数比", f"{max(ns) / max(min(ns), 1):.1f}")
    if not jload(ip):
        info_j["by_class"] = {k: {"n": b["n"], "groups": len(b["g"])} for k, b in by.items()}
        with open(os.path.join(COPIES, "DATASET_INFO_computed.json"), "w", encoding="utf-8") as f:
            json.dump(info_j, f, ensure_ascii=False, indent=2)
    keep_copy(os.path.join(d, "RUN_RECORD.md"), "dataset_RUN_RECORD.md")
    keep_copy(os.path.join(d, "class_names.json"))
    DATA["2_micro_dataset"] = {"model": LOC["micro_model"], "dataset": d, "dataset_from": LOC["micro_dataset_from"],
                               "info": info_j, "sha256_now": cur}


# =============================================================================
#  ③ 显微训练参数
# =============================================================================
KEYWORDS = [
    ("优化器 / 学习率 / 权重衰减", r"optim\.|AdamW|\bAdam\b|\bSGD\b|weight_decay|[\"']--lr[\"']|\blr\s*="),
    ("学习率调度", r"lr_scheduler|Cosine|OneCycle|ReduceLROnPlateau|warmup"),
    ("损失函数", r"CrossEntropyLoss|label_smoothing|FocalLoss|criterion\s*="),
    ("类别不平衡", r"WeightedRandomSampler|class_weight|sample_weight|compute_class_weight"),
    ("数据增强", r"RandomResizedCrop|RandomRotation|RandomAffine|ColorJitter|GaussianBlur|Flip|JPEG|jpeg|RandAugment|Normalize\("),
    ("早停 / 选最佳 epoch", r"patience|best_(?:f1|acc|loss|metric|score|epoch)|early"),
    ("预训练权重 / 微调范围", r"Weights\.|pretrained|requires_grad|freeze"),
    ("随机种子 / 设备", r"manual_seed|\bseed\b.*=|mps|cuda"),
    ("交叉验证划分", r"StratifiedGroupKFold|GroupKFold|n_splits"),
]


def sec3():
    section("⚙️", 3, "显微模型训练参数（清单第 6 项）")
    res = {}
    md, cv = LOC["micro_model"], LOC["micro_cv"]
    args = LOC["micro_args"]
    info(f"最终模型：{md}")
    if args:
        ok("args.json（该模型训练时实际使用的参数）：")
        for k, v in args.items():
            kv(k, v)
        keep_copy(os.path.join(md, "args.json"), "micro_final_args.json")
        res["final_args"] = args
    else:
        err("最终模型文件夹里没有 args.json")
        hint("确认 methods_collect.zsh 里的 MICRO_MODEL 指向含 final.pt 的文件夹")
    for f in ("RUN_RECORD.md", "TRAIN_SUMMARY.md", "model_sha256.txt", "sanity_check.txt"):
        keep_copy(os.path.join(md, f), "micro_final_" + f)
    out()
    if cv:
        info(f"对应的交叉验证：{cv}")
        s = jload(os.path.join(cv, "cv_summary.json"), {}) or {}
        cva = jload(os.path.join(cv, "args.json"), {}) or {}
        for k in ("acc", "macro_f1", "balanced_acc", "top2"):
            if f"{k}_mean" in s:
                kv(k, f"{s[k + '_mean']:.3f} ± {s.get(k + '_sd', float('nan')):.3f}（各折均值 ± SD）")
        kv("完成折数", len(s.get("folds_done", [])) or "—")
        kv("各折最佳 epoch 中位数", s.get("median_best_epoch", "—"))
        if args.get("epochs") is not None and s.get("median_best_epoch") is not None:
            if float(args["epochs"]) == float(s["median_best_epoch"]):
                ok(f"最终模型的 epochs（{args['epochs']}）= 交叉验证最佳 epoch 中位数，与 Methods 写法一致")
            else:
                warn(f"最终模型的 epochs（{args['epochs']}）≠ 交叉验证最佳 epoch 中位数（{s['median_best_epoch']}），请核对")
        if cva.get("trainset") and args.get("trainset") and os.path.basename(str(cva["trainset"]).rstrip("/")) != os.path.basename(str(args["trainset"]).rstrip("/")):
            warn(f"交叉验证用的数据集（{cva['trainset']}）与最终模型（{args['trainset']}）不同，请核对")
        keep_copy(os.path.join(cv, "cv_summary.json"), "micro_cv_summary.json")
        keep_copy(os.path.join(cv, "args.json"), "micro_cv_args.json")
        keep_copy(os.path.join(cv, "per_class.csv"), "micro_cv_per_class.csv")
        keep_copy(os.path.join(cv, "cv_folds.csv"), "micro_cv_folds.csv")
        res["cv"] = {"dir": cv, "summary": s, "args": cva}
    else:
        warn(f"在 {os.path.dirname(md)} 下没找到 {LOC['backbone']} 的交叉验证结果（cv_summary.json）")
        need("指出最终模型对应的交叉验证文件夹（backbone 对比和 epoch 数都依赖它）")
    # 同一数据集下各 backbone 对比
    parent = os.path.dirname(md)
    comp = []
    for x in sorted(os.listdir(parent)) if os.path.isdir(parent) else []:
        s = jload(os.path.join(parent, x, "cv_summary.json"))
        if s and "_FAILED" not in x:
            comp.append([x, len(s.get("folds_done", [])), f"{s.get('acc_mean', float('nan')):.3f} ± {s.get('acc_sd', float('nan')):.3f}",
                         f"{s.get('macro_f1_mean', float('nan')):.3f} ± {s.get('macro_f1_sd', float('nan')):.3f}",
                         f"{s.get('top2_mean', float('nan')):.3f}", mtime(os.path.join(parent, x, "cv_summary.json"))])
    if comp:
        out()
        info(f"同一文件夹（{os.path.basename(parent)}）下已完成的交叉验证（Table 2 草稿）：")
        table(comp, ["文件夹", "折数", "准确率", "macro-F1", "top-2", "完成时间"])
        res["cv_compare"] = comp
        want = {"swin_t", "densenet121", "efficientnet_b0", "resnet50", "convnext_tiny", "vit_b_16"}
        got = {b for b in want if any(r[0].startswith(b) for r in comp)}
        if want - got:
            warn(f"Methods 写了 6 个 backbone 的比较，这里还缺：{'、'.join(sorted(want - got))}")
            need(f"补跑这些 backbone 的交叉验证，或告知 Claude 从 Methods 中删去：{'、'.join(sorted(want - got))}")
    keep_copy(os.path.join(parent, "cv_汇总.txt"))

    # 02_train.py 源码中的关键行
    out()
    tp = os.path.join(A.audit, "02_train.py")
    if os.path.isfile(tp):
        h = sha256(tp)
        info(f"02_train.py（sha256 {h[:16]}…，修改时间 {mtime(tp)}）中与训练设置有关的代码行：")
        rec = os.path.join(md, "RUN_RECORD.md")
        if os.path.isfile(rec):
            txt = open(rec, encoding="utf-8", errors="ignore").read()
            if h in txt or h[:16] in txt:
                ok("现在的 02_train.py 与训练该模型时的版本相同（指纹在 RUN_RECORD.md 中）")
            else:
                warn("RUN_RECORD.md 里没有现在这个 02_train.py 的指纹：脚本在训练后可能改过，下面的代码行不一定是当时的设置")
        else:
            warn("模型文件夹没有 RUN_RECORD.md（旧版流程训练），无法确认 02_train.py 是否与训练时相同")
        lines = open(tp, encoding="utf-8", errors="ignore").read().splitlines()
        found = {}
        for title, pat in KEYWORDS:
            hits = [(i + 1, ln.rstrip()) for i, ln in enumerate(lines)
                    if re.search(pat, ln) and not ln.lstrip().startswith("#") and len(ln.strip()) > 3]
            found[title] = hits
            out(c("1;35", f"   ▸ {title}") + ("" if hits else "：（没有匹配的代码行）"))
            for no, ln in hits[:14]:
                out(f"      {no:>4} │ {ln.strip()[:150]}")
            if len(hits) > 14:
                out(f"           … 另有 {len(hits) - 14} 行，见 copies/02_train.py")
        keep_copy(tp, "02_train.py")
        res["train_py"] = {"sha256": h, "hits": {k: [f"{n}: {x.strip()}" for n, x in v] for k, v in found.items()}}
        ok("02_train.py 已复制到 copies/，Claude 可据此补全 Methods 中的优化器、学习率、损失函数和增强幅度")
    else:
        err(f"找不到 {tp}")
    DATA["3_micro_training"] = res


# =============================================================================
#  ④ 菌落模型
# =============================================================================
def sec4():
    section("🎨", 4, "菌落模型（清单第 8 项）")
    md, lock = LOC["colony_model"], LOC["colony_lock"]
    info(f"菌落模型：{md}")
    if not lock:
        err("这个文件夹里没有 locked_config.json")
        hint("检查 methods_collect.zsh 里的 COLONY_MODEL（应与 clin_config.zsh 的 COLONY_MODEL 相同）")
        mark(4, "❌", "菌落模型：找不到 locked_config.json")
        return
    a = lock.get("args") or {}
    sel = lock.get("selected_method", "?")
    kv("锁定时间", lock.get("locked_at", "—"))
    kv("训练脚本版本", lock.get("version", "—"))
    kv("数据集", LOC["colony_dataset"] or a.get("pre", "—"))
    kv("纳入的节", "、".join(lock.get("classes", [])))
    kv("株数 / CV 组数", f"{lock.get('n_strains', '—')} 株 / {lock.get('n_cv_groups', '—')} 组（留一组交叉验证）")
    kv("选定方案", f"{sel}（规则：{lock.get('selection_rule', '—')}）")
    kv("温度 T", f"{lock.get('temperature', float('nan')):.3f}" + ("（触及下限 0.5）" if abs(lock.get("temperature", 9) - 0.5) < 1e-3 else ""))
    kv("正则强度 C（最终模型）", lock.get("C_final", "—"))
    kv("方案 B 骨干网络", "未跑（--skip_b）" if a.get("skip_b") else a.get("backbone", "—"))
    kv("方案 C（端到端微调）", "已跑（探索性）" if a.get("run_c") else "未跑")
    kv("min_strains / dup_inliers", f"{a.get('min_strains', '—')} / {a.get('dup_inliers', '—')}")
    summ = lock.get("summary") or {}
    if summ:
        out()
        info("各方案的株级交叉验证结果：")
        table([[k, *(f"{v.get(x, float('nan')):.3f}" for x in ("top1", "top2", "bal_acc", "nll", "ece"))] for k, v in summ.items()],
              ["方案", "top1", "top2", "平衡准确率", "NLL", "ECE"])
    # 每节株数
    oof = os.path.join(md, "oof_colony_selected_calibrated.csv")
    per = {}
    if os.path.isfile(oof):
        rr = read_csv(oof)
        per = dict(Counter(r.get("true") or r.get("group", "?") for r in rr))
        out()
        info("每节株数（来自 oof_colony_selected_calibrated.csv）：")
        table([[k, v] for k, v in sorted(per.items())], ["节", "株数"])
        keep_copy(oof)
    miss = [s for s in ("Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes") if s not in lock.get("classes", [])]
    if miss:
        warn(f"菌落模型不含：{'、'.join(miss)}（株数不足 {a.get('min_strains', 3)} 被剔除）。Methods 的融合规则已写明“镜检首位节不在菌落模型内时不融合”")
        need(f"决定是否补够 {'、'.join(miss)} 菌株后重训菌落模型；若不补，论文需写明菌落模型只覆盖 {len(lock.get('classes', []))} 个节")
    low = [f"{k}（{v} 株）" for k, v in per.items() if v < 10]
    if low:
        warn(f"株数少于 10 的节：{'、'.join(low)}")
    # 指纹
    mp = os.path.join(md, "final_model.joblib")
    want = (lock.get("sha256") or {}).get("final_model.joblib", "")
    if os.path.isfile(mp):
        cur = sha256(mp)
        if cur == want:
            ok(f"final_model.joblib 指纹与锁定记录一致（{cur[:16]}…）")
        else:
            err(f"final_model.joblib 指纹与 locked_config.json 不一致（锁定 {want[:16]}… / 现在 {cur[:16]}…）：模型文件被改动过")
            mark(4, "⚠️", "菌落模型：指纹不一致，需核对")
    else:
        warn("没有 final_model.joblib")
    # metrics.txt 的选择与校准一段
    mt = os.path.join(md, "metrics.txt")
    if os.path.isfile(mt):
        txt = open(mt, encoding="utf-8", errors="ignore").read().splitlines()
        i0 = next((i for i, x in enumerate(txt) if "选择与校准" in x), None)
        head = [x for x in txt[:12] if any(k in x for k in ("纳入类别", "不纳入", "合并为同一", "主分析排除", "正面照来自"))]
        out()
        info("metrics.txt 摘录：")
        for x in head + (txt[i0:i0 + 6] if i0 is not None else []):
            if x.strip():
                out("   │ " + x.strip())
        keep_copy(mt, "colony_metrics.txt")
    else:
        warn("没有 metrics.txt")
    keep_copy(os.path.join(md, "locked_config.json"), "colony_locked_config.json")
    for f in ("TRAIN_SUMMARY.md", "RUN_RECORD.md", "coef_A.csv"):
        keep_copy(os.path.join(md, f), "colony_" + f)
    if LOC["colony_dataset"]:
        keep_copy(os.path.join(LOC["colony_dataset"], "preprocess_config.json"), "colony_preprocess_config.json")
        keep_copy(os.path.join(LOC["colony_dataset"], "DATASET_INFO.json"), "colony_DATASET_INFO.json")
    # 与 clin_config.zsh 是否一致
    if A.clin_colony_model or A.clin_micro_model:
        out()
        for label, mine, clin in (("菌落模型", A.colony_model, A.clin_colony_model), ("显微模型", A.micro_model, A.clin_micro_model)):
            if not clin:
                continue
            if os.path.normpath(clin) == os.path.normpath(mine):
                ok(f"{label}与 clin_config.zsh（临床验证用）相同：{mine}")
            else:
                warn(f"{label}与临床验证设置不同：本次统计 {mine} / clin_config.zsh 是 {clin}。论文报告的应是临床验证锁定的那个")
    DATA["4_colony_model"] = {"dir": md, "locked_config": lock, "strains_per_section": per}


# =============================================================================
#  ⑤ 软件版本与电脑型号
# =============================================================================
PKGS = ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "scipy", "opencv-python", "opencv-python-headless",
        "pillow", "pillow-heif", "matplotlib", "joblib", "grad-cam"]
PROBE = r"""
import json, sys, platform
from importlib import metadata as M
out = {"python": platform.python_version(), "executable": sys.executable, "pkgs": {}}
for p in %r:
    try: out["pkgs"][p] = M.version(p)
    except Exception: pass
try:
    import torch
    out["mps"] = bool(torch.backends.mps.is_available()); out["cuda"] = bool(torch.cuda.is_available())
except Exception: pass
print(json.dumps(out))
""" % (PKGS,)


def probe(py):
    r = run([py, "-c", PROBE])
    try:
        return json.loads(r.splitlines()[-1])
    except Exception:
        return None


def sec5():
    section("💻", 5, "软件版本与电脑型号（清单第 10 项）")
    res = {}
    info("电脑：")
    hw = {}
    if platform.system() == "Darwin":
        sp = run(["system_profiler", "SPHardwareDataType"])
        for k_cn, k in (("型号", "Model Name"), ("型号标识", "Model Identifier"), ("芯片", "Chip"), ("处理器", "Processor Name"),
                        ("核心数", "Total Number of Cores"), ("内存", "Memory")):
            m = re.search(rf"{k}:\s*(.+)", sp)
            if m:
                hw[k_cn] = m.group(1).strip()
        hw["系统"] = f"macOS {run(['sw_vers', '-productVersion'])}"
    else:
        hw["系统"] = platform.platform()
        hw["处理器"] = platform.processor() or platform.machine()
    for k, v in hw.items():
        kv(k, v)
    res["hardware"] = hw
    envs = [("显微用 python（fungal）", sys.executable)]
    if A.colony_python and os.path.realpath(A.colony_python) != os.path.realpath(sys.executable):
        envs.append(("菌落用 python（COLONY_PYTHON）", A.colony_python))
    res["envs"] = {}
    for label, py in envs:
        out()
        p = probe(py)
        if not p:
            warn(f"{label} 无法运行：{py}")
            continue
        info(f"{label}：Python {p['python']}（{p['executable']}）")
        table([[k, v] for k, v in p["pkgs"].items()], ["包", "版本"])
        if "mps" in p:
            kv("计算设备", "Apple MPS 可用" if p["mps"] else ("CUDA 可用" if p.get("cuda") else "仅 CPU"))
        res["envs"][label] = p
        os.makedirs(COPIES, exist_ok=True)
        tag = "micro" if py == sys.executable else "colony"
        fz = run([py, "-m", "pip", "freeze"], timeout=120)
        if fz:
            open(os.path.join(COPIES, f"pip_freeze_{tag}.txt"), "w", encoding="utf-8").write(fz + "\n")
        prefix = os.path.dirname(os.path.dirname(py))
        conda = shutil.which("conda") or os.environ.get("CONDA_EXE", "")
        cl = run([conda, "list", "-p", prefix], timeout=120) if conda and os.path.isdir(os.path.join(prefix, "conda-meta")) else ""
        if cl:
            open(os.path.join(COPIES, f"conda_list_{tag}.txt"), "w", encoding="utf-8").write(cl + "\n")
            ok(f"conda list 已保存：copies/conda_list_{tag}.txt")
        elif fz:
            ok(f"pip freeze 已保存：copies/pip_freeze_{tag}.txt（没有找到 conda 或该环境不是 conda 环境）")
    lk = LOC["colony_lock"]
    if len(res["envs"]) == 2:
        v = [p["pkgs"].get("scikit-learn") for p in res["envs"].values()]
        if v[0] != v[1]:
            info(f"两个环境的 scikit-learn 版本不同（{v[0]} / {v[1]}）：论文里菌落模型报告 COLONY_PYTHON 环境的版本")
    DATA["5_software"] = res


# =============================================================================
#  ⑥ 外部图像集来源
# =============================================================================
def sec6():
    section("🏥", 6, "回顾性外部图像集的来源（清单第 11 项）")
    out("   这一项问的是：外部验证用的图片来自哪几家医院的实验室。")
    out("   意义：① 论文说“多来源 / 多中心外部验证”，审稿人要看到独立机构的数量，一家医院的 3 位老师只算 1 个来源；")
    out("         ② 留一来源验证（LOSO）以“来源”为单位，同一家医院的两个文件夹应合并成一个来源，否则高估独立性；")
    out("         ③ 投稿者需要写进致谢（或作者），期刊要求征得本人同意。")
    out()
    warn("医院名称不在任何文件里，程序无法获得。下面只能列出各来源的图片数，并生成一张待填表")
    res = {"sources": []}
    rows = []
    man = os.path.join(A.out_root, "04_audit_review", "ext_audit", "manifest.csv")
    if os.path.isfile(man):
        m = read_csv(man)
        cols = list(m[0].keys()) if m else []
        info(f"外部集总表：{man}（{len(m)} 行，修改时间 {mtime(man)}）")
        kv("列名", "、".join(cols))
        key = next((k for k in ("site", "contributor", "contributor_code", "source", "folder", "top_folder") if k in cols), None)
        if "role" in cols:
            rc = Counter(r.get("role", "") for r in m)
            out()
            info("按用途（role）：")
            table([[k or "（空）", v] for k, v in rc.most_common()], ["role", "图片数"])
            res["by_role"] = dict(rc)
        if key:
            agg = defaultdict(Counter)
            for r in m:
                agg[r.get(key, "") or "（空）"][r.get("role", "all")] += 1
            roles = [x for x in ("external_micro", "external_colony", "out_of_panel") if any(x in v for v in agg.values())]
            out()
            info(f"按来源（列 {key}）：")
            t = [[k, sum(v.values())] + [v.get(x, 0) for x in roles] for k, v in sorted(agg.items())]
            table(t, [key, "合计"] + roles)
            rows = [[k, sum(v.values()), v.get("external_micro", 0)] for k, v in sorted(agg.items())]
        else:
            warn("总表里没有能识别的“来源”列（site / contributor / folder），改按文件夹统计")
    else:
        warn(f"没有外部集总表（{man}），改按文件夹统计")
    if not rows and os.path.isdir(A.ext_root):
        info(f"外部图像文件夹：{A.ext_root}")
        for x in sorted(os.listdir(A.ext_root)):
            p = os.path.join(A.ext_root, x)
            if os.path.isdir(p) and not x.startswith("."):
                rows.append([x, count_imgs(p), ""])
        table(rows, ["文件夹", "图片数", ""])
    elif not rows:
        warn(f"外部图像文件夹也不存在：{A.ext_root}")
    loso = os.path.join(A.out_root, "03_external_eval", "ext_loso", "汇总.txt")
    if keep_copy(loso, "ext_loso_汇总.txt"):
        ok("已复制 LOSO 汇总（ext_loso/汇总.txt）")
    fp = os.path.join(A.out, "external_sources_待填.csv")
    with open(fp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["来源代号或文件夹", "图片数", "其中本期26类镜下图", "投稿人", "医院全称（请填）", "城市（请填）",
                    "与其他来源是否同一家医院（请填）", "是否网络图片（是/否）", "是否同意致谢署名（请填）"])
        for r in rows:
            w.writerow(list(r) + [""] * 6)
    res["sources"] = rows
    ok(f"待填表已生成：{fp}")
    need("填写 external_sources_待填.csv：每个来源对应的医院全称、城市、是否与其他来源同属一家医院、是否网络图片")
    mark(6, "✍️", "外部图像集来源：图片数已统计，医院名称需人工填写")
    DATA["6_external_sources"] = res


# =============================================================================
def main():
    global A, COPIES, LOC
    ap = argparse.ArgumentParser(description="为论文 Methods 收集数字和参数（只读）")
    ap.add_argument("--audit", required=True, help="Audit 文件夹")
    ap.add_argument("--out_root", required=True, help="Audit/output")
    ap.add_argument("--micro_model", required=True, help="显微最终模型；短名字 → output/02_training_runs/<名字>")
    ap.add_argument("--micro_trainset", default="", help="args.json 里找不到数据集时使用；短名字 → output/01_trainsets/<名字>")
    ap.add_argument("--colony_model", required=True, help="菌落模型；短名字 → output/02_training_runs/<名字>")
    ap.add_argument("--recrop_root", required=True)
    ap.add_argument("--colony_raw", required=True)
    ap.add_argument("--ext_root", required=True)
    ap.add_argument("--colony_python", default="")
    ap.add_argument("--clin_micro_model", default="", help="clin_config.zsh 里的 MICRO_MODELS（仅用于比对）")
    ap.add_argument("--clin_colony_model", default="", help="clin_config.zsh 里的 COLONY_MODEL（仅用于比对）")
    ap.add_argument("--out", required=True, help="本次输出文件夹")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    COPIES = os.path.join(A.out, "copies")
    t0 = dt.datetime.now()
    REPORT.extend([f"# Methods 资料收集结果（{t0:%Y-%m-%d %H:%M}）", "",
                   f"- 脚本：00_methods_collect.py（{VERSION}，sha256 {sha256(os.path.abspath(__file__))[:16]}…）",
                   f"- 完整命令：`{' '.join(sys.argv)}`", f"- 电脑：{platform.node()}", ""])
    print(c("1;36", f"ℹ️  {VERSION}；只读，不修改任何原有文件"))
    LOC = locate()
    guarded(1, "开发集株数", sec1)
    guarded(2, "显微数据集信息", sec2)
    guarded(3, "显微训练参数", sec3)
    guarded(4, "菌落模型", sec4)
    guarded(5, "软件版本与电脑型号", sec5)
    guarded(6, "外部图像集来源", sec6)

    # ---------------- 统一输出 ----------------
    line = "═" * 68
    print()
    print(c("1;97;45", line))
    print(c("1;97;45", "  📋  汇总"))
    print(c("1;97;45", line))
    REPORT.extend(["", "## 📋 汇总", ""])
    out("来源（请人工核对这就是论文要报告的模型）：")
    kv("🔬 显微最终模型", LOC["micro_model"])
    kv("🗂️ 显微数据集", LOC["micro_dataset"] or "（未找到）")
    kv("📊 显微交叉验证", LOC["micro_cv"] or "（未找到）")
    kv("🎨 菌落模型", LOC["colony_model"])
    kv("🧫 菌落数据集", LOC["colony_dataset"] or "（未找到）")
    out()
    out("各段完成情况：")
    for n, e, t in sorted(STATUS):
        out(f"   {e} 第 {n} 段  {t}")
    out()
    if TODO:
        out(c("1;35", f"✍️  仍需人工处理 {len(TODO)} 项（程序无法获得）："))
        for i, t in enumerate(dict.fromkeys(TODO), 1):
            out(f"   {i}. {t}")
    else:
        ok("没有需要人工处理的事项")
    DATA["sources"] = {k: LOC[k] for k in ("micro_model", "micro_dataset", "micro_cv", "colony_model", "colony_dataset")}
    DATA["todo"] = list(dict.fromkeys(TODO))
    DATA["status"] = [list(s) for s in sorted(STATUS)]
    DATA["generated"] = {"at": t0.isoformat(timespec="seconds"), "version": VERSION, "argv": sys.argv}
    with open(os.path.join(A.out, "methods_info.json"), "w", encoding="utf-8") as f:
        json.dump(DATA, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(A.out, "METHODS_INFO.md"), "w", encoding="utf-8") as f:
        body, in_tbl = [], False
        for ln in REPORT:      # 表格和缩进内容放进代码块，保持对齐
            pre = ln.startswith("   ")
            if pre and not in_tbl:
                body.append("```")
            if not pre and in_tbl:
                body.append("```")
            in_tbl = pre
            body.append(ln)
        if in_tbl:
            body.append("```")
        f.write("\n".join(body) + "\n")
    print()
    print(c("1;32", f"📦 输出文件夹：{A.out}"))
    print(c("1;32", "   METHODS_INFO.md  ← 把这个文件（或整个文件夹的压缩包）发给 Claude"))
    print(c("1;32", "   external_sources_待填.csv  ← 填好后一并发来"))
    return 1 if any(s[1] == "❌" for s in STATUS) else 0


if __name__ == "__main__":
    sys.exit(main())
