#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_fusion_sim_check.py —— 同节随机配对融合分析：运行前诊断（只读，不改任何文件）

做什么
  1. 菌落 OOF：在菌落模型文件夹里找株级 OOF 概率（优先 oof_colony_selected_calibrated.csv），
     列出类别、各组株数，检查概率每行之和是否为 1
  2. 镜检 OOF：在镜检训练目录（如 runs_v4）下递归查找交叉验证的 OOF 预测表，
     自动识别 概率列 / 真实类别列 / 菌株(组)列，打分排序，列出前几个候选
     ⚠ 不能用 swin_t_full_final 这类全量模型对训练图的预测：那是训练集上的结果，会严重高估
  3. 类别 → 曲霉节 映射：先查 taxonomy_map.csv，查不到的再按菌名关键词推断，
     写成 section_map_auto.csv（可以用 Excel 手工改，主脚本读这个文件）
  4. 顺带打印 clin_config.zsh 里和融合有关的设置，核对与临床验证是否一致
  5. 把选定的文件、列名写进 CHECK_RESULT.json，主脚本 03_fusion_simpair.py 直接读它

安装（清华源）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy

不要直接运行，用 zsh fsim_0_check.zsh
"""

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    sys.exit(f"❌ 缺少依赖：{e}\n   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy")

VERSION = "00_fusion_sim_check v1.0 (2026-11-13)"

ASP_SECTIONS = ["Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes", "Versicolores", "Aspergillus"]
# 菌名关键词 → 节（taxonomy_map 查不到时才用；小写匹配）
KEYWORDS = {
    "Fumigati": ["fumigati", "fumigatus", "lentulus", "fischeri", "udagawae", "烟曲霉", "迟缓曲霉"],
    "Flavi": ["flavi", "flavus", "oryzae", "parasiticus", "tamarii", "黄曲霉", "米曲霉"],
    "Nigri": ["nigri", "niger", "tubingensis", "welwitschiae", "awamori", "黑曲霉"],
    "Terrei": ["terrei", "terreus", "土曲霉"],
    "Nidulantes": ["nidulantes", "nidulans", "构巢曲霉"],
    "Versicolores": ["versicolores", "versicolor", "sydowii", "杂色曲霉", "聚多曲霉"],
    "Aspergillus": ["glaucus", "chevalieri", "灰绿曲霉", "sect_aspergillus", "section aspergillus"],
}
TRUE_COLS = ["true", "y_true", "true_class", "true_label", "label", "gt", "target", "class", "species", "truth"]
GROUP_COLS = ["isolate", "isolate_id", "strain", "strain_id", "cv_group", "group_id", "group", "case", "case_id"]
IMAGE_COLS = ["path", "file", "filepath", "image", "img", "image_path", "filename"]
FOLD_COLS = ["fold", "cv_fold", "fold_id"]
PROB_PREFIX = ["p_", "prob_", "proba_", "score_"]


def ok(m): print(f"  ✅ {m}")
def warn(m): print(f"  ⚠️  {m}")
def bad(m): print(f"  ❌ {m}")
def head(m): print(f"\n🔎 {m}")


def read_csv_any(p, nrows=None):
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, nrows=nrows, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ 菌落
def check_colony(colony_run: Path):
    head(f"菌落 OOF：{colony_run}")
    if not colony_run.is_dir():
        bad("文件夹不存在")
        return None
    cand = [colony_run / "oof_colony_selected_calibrated.csv"]
    lc = colony_run / "locked_config.json"
    sel = None
    if lc.exists():
        try:
            sel = json.loads(lc.read_text(encoding="utf-8")).get("selected_method")
            ok(f"locked_config.json：选定方案 {sel}")
        except Exception:
            warn("locked_config.json 读不了")
    if sel:
        cand.append(colony_run / f"oof_{sel}.csv")
    cand += [colony_run / "oof_B.csv", colony_run / "oof_A.csv"]
    f = next((c for c in cand if c.exists()), None)
    if f is None:
        bad("找不到 oof_colony_selected_calibrated.csv / oof_A.csv / oof_B.csv")
        return None
    if f.name != "oof_colony_selected_calibrated.csv":
        warn(f"没有温度校准后的 OOF，改用 {f.name}（未校准，概率可能偏自信）")
    df = read_csv_any(f)
    pcols = [c for c in df.columns if c.startswith("p_")]
    classes = [c[2:] for c in pcols]
    need = {"strain_id", "true"}
    if not need.issubset(df.columns) or not pcols:
        bad(f"{f.name} 缺少列，现有列：{list(df.columns)}")
        return None
    ok(f"使用 {f.name}：{len(df)} 株，类别 {classes}")
    s = df[pcols].sum(axis=1)
    if (s - 1).abs().max() > 0.02:
        warn(f"概率每行之和不为 1（{s.min():.3f}–{s.max():.3f}），主脚本会重新归一化")
    else:
        ok("概率每行之和 ≈ 1")
    cnt = df["true"].value_counts()
    print("     各组株数：" + "，".join(f"{k} {v}" for k, v in cnt.items()))
    for c in classes:
        if c not in ASP_SECTIONS:
            warn(f"菌落类别 {c} 不是标准节名，请核对")
    small = [k for k, v in cnt.items() if v < 3]
    if small:
        warn(f"株数 <3 的组：{small}；随机配对时可选的菌落株很少，区间会偏窄（低估不确定性）")
    return {"colony_oof": str(f), "colony_classes": classes, "colony_n": int(len(df))}


# ------------------------------------------------------------------ 镜检
def detect_cols(df, class_list=None):
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    # 概率列
    pcols = []
    for c in cols:
        cl = c.lower()
        if any(cl.startswith(p) for p in PROB_PREFIX) and pd.api.types.is_numeric_dtype(df[c]):
            pcols.append(c)
    if not pcols and class_list:
        pcols = [c for c in cols if c in class_list and pd.api.types.is_numeric_dtype(df[c])]
    def pick(names):
        for n in names:
            if n in low:
                return low[n]
        return None
    tcol = pick(TRUE_COLS)
    gcol = pick(GROUP_COLS)
    icol = pick(IMAGE_COLS)
    fcol = pick(FOLD_COLS)
    return pcols, tcol, gcol, icol, fcol


def strip_prefix(c):
    for p in PROB_PREFIX:
        if c.lower().startswith(p):
            return c[len(p):]
    return c


def score_candidate(p: Path, df, pcols, tcol, gcol, fcol):
    s = 0
    name = str(p).lower()
    if len(pcols) >= 5: s += 4
    if tcol: s += 3
    if gcol: s += 3
    if fcol: s += 1
    if "oof" in name: s += 3
    if "cv" in name or "fold" in name: s += 1
    if "swin" in name: s += 1
    if "full" in name or "final" in name: s -= 6     # 全量模型：不能用
    if "external" in name or "ext" in Path(name).name or "clinical" in name: s -= 4
    return s


def check_micro(micro_run: Path, backbone: str):
    head(f"镜检 OOF：在 {micro_run} 下查找")
    if not micro_run.is_dir():
        bad("文件夹不存在")
        return None
    class_list = None
    for cn in micro_run.rglob("class_names.json"):
        try:
            class_list = json.loads(cn.read_text(encoding="utf-8"))
            if isinstance(class_list, dict):
                class_list = [class_list[k] for k in sorted(class_list, key=lambda x: int(x))]
            ok(f"找到 class_names.json（{len(class_list)} 类）：{cn}")
            break
        except Exception:
            class_list = None
    csvs = [p for p in micro_run.rglob("*.csv") if p.stat().st_size > 0]
    print(f"     共 {len(csvs)} 个 csv")
    rows = []
    for p in csvs:
        df = read_csv_any(p, nrows=200)
        if df is None or df.empty:
            continue
        pcols, tcol, gcol, icol, fcol = detect_cols(df, class_list)
        if len(pcols) < 3:
            continue
        rows.append((score_candidate(p, df, pcols, tcol, gcol, fcol), p, pcols, tcol, gcol, icol, fcol))
    if not rows:
        bad("没有找到带概率列的 csv（列名以 p_ / prob_ 开头，或与 class_names.json 一致）")
        print("     如果 02_train.py 的 OOF 存成了 npz 或别的格式，把文件名和前几行发给 Claude")
        return None
    rows.sort(key=lambda r: (-r[0], len(str(r[1]))))
    if backbone:
        pref = [r for r in rows if backbone.lower() in str(r[1]).lower()]
        if pref and pref[0][0] >= rows[0][0] - 1:
            rows = pref + [r for r in rows if r not in pref]
    print("     候选（分数高的在前）：")
    for r in rows[:6]:
        sc, p, pcols, tcol, gcol, icol, fcol = r
        print(f"     [{sc:>3}] {p.relative_to(micro_run)}  概率列 {len(pcols)}，真实列 {tcol}，菌株列 {gcol}，折 {fcol}")
    sc, p, pcols, tcol, gcol, icol, fcol = rows[0]
    print(f"\n     → 选用：{p}")
    return inspect_micro_file(p, class_list)


def inspect_micro_file(p: Path, class_list=None):
    if "full" in str(p).lower() or "final" in str(p).lower():
        bad("文件路径含 full/final，像是全量模型对训练图的预测，不能用于本分析")
    df = read_csv_any(p)
    if df is None or df.empty:
        bad("读不了或为空")
        return None
    pcols, tcol, gcol, icol, fcol = detect_cols(df, class_list)
    if len(pcols) < 3:
        bad(f"没认出概率列；现有列：{list(df.columns)}")
        return None
    probs = df[pcols].astype(float)
    rs = probs.sum(axis=1)
    is_logit = bool((probs.min().min() < 0) or (rs - 1).abs().max() > 0.05)
    if is_logit:
        warn(f"概率列每行之和 {rs.min():.2f}–{rs.max():.2f}，像是 logits，主脚本会做 softmax")
    else:
        ok("概率每行之和 ≈ 1")
    if not tcol:
        bad(f"没认出真实类别列；现有列：{list(df.columns)}")
        return None
    if not gcol:
        bad(f"没认出菌株（组）列；株级分析必须有它。现有列：{list(df.columns)}")
        return None
    classes = [strip_prefix(c) for c in pcols]
    tv = df[tcol]
    if pd.api.types.is_integer_dtype(tv):
        warn(f"真实类别列 {tcol} 是整数，按概率列顺序解码为类名（请确认顺序一致）")
    else:
        miss = sorted(set(tv.astype(str)) - set(classes))
        if miss:
            bad(f"真实类别中有 {len(miss)} 个不在概率列类名里：{miss[:8]}")
            return None
    n_img, n_grp = len(df), df[gcol].nunique()
    ok(f"概率列 {len(pcols)}，真实列 {tcol}，菌株列 {gcol}；{n_img} 行、{n_grp} 个菌株（组）")
    if fcol:
        ok(f"折列 {fcol}：{sorted(df[fcol].unique().tolist())}")
    else:
        warn("没有折列；请确认这是交叉验证的 OOF（每张图只在它的验证折被预测），不是训练集预测")
    if icol and df[icol].duplicated().any():
        warn(f"{icol} 有 {int(df[icol].duplicated().sum())} 行重复，可能多折结果拼在一起；主脚本对同一菌株取平均")
    nlab = df.groupby(gcol)[tcol].nunique()
    if (nlab > 1).any():
        bad(f"{int((nlab > 1).sum())} 个菌株对应多个真实类别，菌株列可能选错了")
        return None
    return {"micro_oof": str(p), "prob_cols": pcols, "classes": classes, "true_col": tcol,
            "group_col": gcol, "image_col": icol, "fold_col": fcol, "needs_softmax": is_logit,
            "true_is_index": bool(pd.api.types.is_integer_dtype(tv)), "n_images": int(n_img), "n_groups": int(n_grp)}


# ------------------------------------------------------------------ 映射
def section_from_taxonomy(tax: Path, classes):
    """在 taxonomy_map.csv 里找 类名列 → 节列。
       节列要求至少出现 2 个不同的节名：只有 "Aspergillus" 一个值的列多半是"属"列，不是节列
       （灰绿曲霉节也叫 Aspergillus，容易混淆）"""
    if not tax or not tax.exists():
        return {}
    df = read_csv_any(tax)
    if df is None:
        return {}
    cls = set(classes)
    best, best_key = {}, (0, 0)
    for a in df.columns:
        va = df[a].astype(str).str.strip()
        if len(cls & set(va)) < max(2, len(cls) // 4):
            continue
        for b in df.columns:
            if a == b:
                continue
            vb = df[b].astype(str).str.strip()
            hits = vb[va.isin(cls) & vb.isin(ASP_SECTIONS)]
            n_distinct = hits.nunique()
            if n_distinct < 2:
                continue
            key = (n_distinct, len(hits))
            if key > best_key:
                best_key = key
                best = {x: (y if y in ASP_SECTIONS else None) for x, y in zip(va, vb) if x in cls}
                best["__cols__"] = (a, b)
    return best


def section_from_keywords(c):
    cl = c.lower().replace("-", "_")
    for sec, kws in KEYWORDS.items():
        if any(k in cl for k in kws):
            return sec
    # 类名本身就是节名（如 Aspergillus_Fumigati / Aspergillus sect. Flavi）
    for sec in ASP_SECTIONS:
        if sec != "Aspergillus" and re.search(sec.lower(), cl):
            return sec
    return None


def build_map(classes, tax: Path, out_csv: Path):
    head("类别 → 曲霉节 映射")
    tm = section_from_taxonomy(tax, classes)
    cols = tm.pop("__cols__", None)
    if cols:
        ok(f"taxonomy_map.csv：用列 {cols[0]} → {cols[1]}")
    elif tax and tax.exists():
        warn("taxonomy_map.csv 里没找到 类名 → 节 的对应列，改用关键词推断")
    else:
        warn("没有 taxonomy_map.csv，用关键词推断")
    rows = []
    for c in classes:
        if c in tm and tm[c]:
            sec, src = tm[c], "taxonomy_map"
        else:
            sec = section_from_keywords(c)
            src = "keyword" if sec else ""
            if not sec:
                sec = "NonAsp"
                src = "default(非曲霉)"
                if "aspergill" in c.lower() or "曲霉" in c:
                    sec, src = "Aspergillus_unknown", "❗曲霉但未识别节"
        rows.append({"class": c, "section": sec, "source": src})
    m = pd.DataFrame(rows)
    m.to_csv(out_csv, index=False, encoding="utf-8-sig")
    asp = m[m["section"].isin(ASP_SECTIONS)]
    print(f"     曲霉类别 {len(asp)} 个：")
    for _, r in asp.iterrows():
        print(f"       {r['class']:<36} → {r['section']:<13}（{r['source']}）")
    unk = m[m["section"] == "Aspergillus_unknown"]
    if len(unk):
        bad(f"{len(unk)} 个曲霉类别没识别出节：{unk['class'].tolist()}；请在 section_map_auto.csv 里手工填 section")
    print(f"     非曲霉类别 {int((m['section'] == 'NonAsp').sum())} 个（不参与本分析，但计入 P(曲霉) 的分流）")
    ok(f"映射表已写出：{out_csv}（可用 Excel 修改后另存为 CSV UTF-8）")
    return m


def show_clin_config(audit: Path):
    p = audit / "clin_config.zsh"
    head("clin_config.zsh 中与融合有关的设置（核对用）")
    if not p.exists():
        warn("Audit 下没有 clin_config.zsh")
        return []
    lines = [l.rstrip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines()
             if re.search(r"(FUSION|WEIGHT|ASP|GATE|RULE|融合|权重)", l, re.I) and not l.strip().startswith("#")]
    for l in lines:
        print("     " + l)
    if not lines:
        warn("没找到相关行")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--micro_run", required=True)
    ap.add_argument("--micro_oof", default="", help="直接指定镜检 OOF 文件（跳过自动查找）")
    ap.add_argument("--backbone", default="swin_t")
    ap.add_argument("--colony_run", required=True)
    ap.add_argument("--taxonomy", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"# {VERSION}\n# {' '.join(sys.argv)}")

    col = check_colony(Path(args.colony_run))
    if args.micro_oof:
        head(f"镜检 OOF（手工指定）：{args.micro_oof}")
        mic = inspect_micro_file(Path(args.micro_oof)) if Path(args.micro_oof).exists() else None
        if mic is None and not Path(args.micro_oof).exists():
            bad("文件不存在")
    else:
        mic = check_micro(Path(args.micro_run), args.backbone)
    smap = None
    if mic:
        smap = build_map(mic["classes"], Path(args.taxonomy) if args.taxonomy else None, out / "section_map_auto.csv")
    clin = show_clin_config(Path(args.audit))

    head("小结")
    problems = []
    if not col: problems.append("菌落 OOF")
    if not mic: problems.append("镜检 OOF")
    if smap is not None:
        if (smap["section"] == "Aspergillus_unknown").any(): problems.append("节映射")
        if col:
            cov = [c for c in col["colony_classes"] if c in set(smap["section"])]
            miss = [c for c in col["colony_classes"] if c not in set(smap["section"])]
            if miss:
                warn(f"菌落类别 {miss} 在镜检类别里找不到对应的节")
            ok(f"可以分析的节（镜检和菌落都有）：{cov}")
            if not cov: problems.append("没有共同的节")
    res = {"version": VERSION, "time": dt.datetime.now().isoformat(timespec="seconds"),
           "args": vars(args), "colony": col, "micro": mic,
           "section_map": str(out / "section_map_auto.csv") if smap is not None else None,
           "clin_config_lines": clin, "problems": problems}
    (out / "CHECK_RESULT.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    if problems:
        bad("需要处理：" + "、".join(problems))
        sys.exit(2)
    print("\n🎉 诊断通过。请看一眼上面的映射表和选用的文件，确认无误后运行 zsh fsim_1_run.zsh")


if __name__ == "__main__":
    main()
