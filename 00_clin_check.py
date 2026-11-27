#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_clin_check.py —— 临床验证素材检查 + 答案表维护（不改动任何照片）

不建议直接运行，请用 zsh clin_2_check.zsh（参数已写死在里面）。

做 4 件事：
  1. 检查镜检照片：<micro_dir>/<株号>/任意文件名.jpg ，株号必须符合 --id_regex（默认 PUPH-PCL-001）
  2. 检查菌落照片：<colony_dir>/<株号>-B-O-<n>.jpg（正面）/ -B-R-<n>（反面，可选）
  3. 交叉核对：有菌落没镜检、有镜检没菌落（曲霉株缺菌落会提示）
  4. 答案表：answers_master.csv 不存在就新建；有新株就追加空行（已填的内容绝不覆盖）；
     --split 时由它生成两张派生表：
        answers_clinical.csv  （03_roi_clinical.py 揭盲用：isolate_id, reference, train_class, method）
        answers_colony.csv    （03_colony_predict.py 用：strain_id, section）
     section 没填时，按 reference 自动推断（clin_sections.py 的对照表）

退出码：0 正常；2 有阻断性错误（命名错误、答案表类名不对等）；3 --require_answers 时答案表未填完
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from clin_sections import SECTIONS, is_aspergillus, read_csv_any, section_of  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".heic", ".heif"}
COLONY_RE = re.compile(r"^(PUPH-[A-Z]{3}-\d{3})-B-([OR])-(\d+)$", re.IGNORECASE)   # 与 01_colony_annotate.py 一致
MASTER_COLS = ["isolate_id", "reference", "train_class", "section", "method", "note"]
SUSPICIOUS = re.compile(r"(asp|fum|fla|nig|ter|nid|mucor|rhizo|曲霉|毛霉|根霉|镰刀|青霉)", re.IGNORECASE)


def resolve_model(p):
    p = os.path.expanduser(p)
    for c in (p, os.path.join(BASE, p), os.path.join(BASE, "output", "02_training_runs", p),
              os.path.join(BASE, "output", "02_training_runs", "runs", p)):
        if os.path.isdir(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro_dir", default="")
    ap.add_argument("--colony_dir", default="")
    ap.add_argument("--answers", default="", help="answers_master.csv 路径")
    ap.add_argument("--micro_model", default="", help="镜检模型文件夹（读 class_names.json 核对 train_class）")
    ap.add_argument("--id_regex", default=r"^PUPH-PCL-\d{3}$")
    ap.add_argument("--inventory", default="", help="写出每株素材清单 CSV")
    ap.add_argument("--split", action="store_true", help="生成 answers_clinical.csv / answers_colony.csv")
    ap.add_argument("--require_answers", action="store_true", help="答案表未填完则退出码 3")
    ap.add_argument("--show_mapping", action="store_true", help="只显示镜检模型类别 → 曲霉组 的对照后退出")
    a = ap.parse_args()

    if a.show_mapping:
        d = resolve_model(a.micro_model) if a.micro_model else None
        f = os.path.join(d, "class_names.json") if d else ""
        if not d or not os.path.exists(f):
            sys.exit(f"找不到镜检模型或其 class_names.json：{a.micro_model}")
        cls = json.load(open(f, encoding="utf-8"))
        print(f"镜检模型 {d}：{len(cls)} 类")
        for c in cls:
            s = section_of(c)
            print(f"   {c:<40s}{'→ 曲霉 ' + s if s else ''}")
        if not any(is_aspergillus(c) for c in cls):
            sys.exit("❌ 模型类别里没有曲霉，融合无法进行")
        return
    for k in ("micro_dir", "colony_dir", "answers"):
        if not getattr(a, k):
            sys.exit(f"缺少 --{k}")

    errs, warns = [], []
    idre = re.compile(a.id_regex)

    # ---------------------------------------------------------------- 1. 镜检
    micro = defaultdict(list)
    if not os.path.isdir(a.micro_dir):
        sys.exit(f"找不到镜检文件夹 {a.micro_dir}（先运行 zsh clin_1_init.zsh）")
    for name in sorted(os.listdir(a.micro_dir)):
        full = os.path.join(a.micro_dir, name)
        if name.startswith((".", "_")):
            continue
        if os.path.isfile(full):
            if os.path.splitext(name)[1].lower() in EXTS:
                errs.append(f"镜检图直接放在 micro/ 下：{name} → 请放进以株号命名的子文件夹")
            continue
        if not idre.match(name):
            errs.append(f"镜检子文件夹名不符合株号规则（{a.id_regex}）：{name}")
            continue
        for root, dirs, files in os.walk(full):
            dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
            for fn in files:
                if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in EXTS:
                    continue
                if SUSPICIOUS.search(fn):
                    warns.append(f"镜检文件名疑似含菌名（会破坏盲法）：{name}/{fn}")
                micro[name].append(fn)
        if not micro[name]:
            warns.append(f"镜检文件夹是空的：{name}")
            micro.pop(name, None)

    # ---------------------------------------------------------------- 2. 菌落
    colony = defaultdict(Counter)
    if not os.path.isdir(a.colony_dir):
        sys.exit(f"找不到菌落文件夹 {a.colony_dir}（先运行 zsh clin_1_init.zsh）")
    for root, dirs, files in os.walk(a.colony_dir):
        dirs[:] = [d for d in dirs if not d.startswith((".", "_"))]
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if fn.startswith(".") or ext.lower() not in EXTS:
                continue
            m = COLONY_RE.match(stem)
            if not m:
                errs.append(f"菌落照片命名不对：{os.path.relpath(os.path.join(root, fn), a.colony_dir)}"
                            "（应为 PUPH-PCL-001-B-O-1.jpg；O=正面，R=反面）")
                continue
            sid = m.group(1).upper()
            if not idre.match(sid):
                errs.append(f"菌落照片株号不符合规则：{fn}")
                continue
            colony[sid][m.group(2).upper()] += 1
    for sid, c in colony.items():
        if c["O"] == 0:
            warns.append(f"{sid} 只有反面照片，没有正面（菌落模型只用正面，该株菌落将无法评估）")
        if sid not in micro:
            warns.append(f"{sid} 有菌落照片但没有镜检文件夹（只能做菌落单独预测）")

    # ---------------------------------------------------------------- 3. 答案表
    ids = sorted(set(micro) | set(colony))
    if os.path.exists(a.answers):
        ans = read_csv_any(a.answers)
        for c in MASTER_COLS:
            if c not in ans.columns:
                ans[c] = ""
        ans = ans[MASTER_COLS + [c for c in ans.columns if c not in MASTER_COLS]]
        ans["isolate_id"] = ans["isolate_id"].str.strip().str.upper()
        dup = ans["isolate_id"][ans["isolate_id"].duplicated()].tolist()
        if dup:
            errs.append(f"答案表里株号重复：{', '.join(dup)}")
        new = [i for i in ids if i not in set(ans["isolate_id"])]
        if new:
            ans = pd.concat([ans, pd.DataFrame([{"isolate_id": i} for i in new])], ignore_index=True).fillna("")
            ans.to_csv(a.answers, index=False, encoding="utf-8-sig")
            print(f"📝 答案表追加了 {len(new)} 个新株：{', '.join(new[:10])}{' …' if len(new) > 10 else ''}")
        gone = [i for i in ans["isolate_id"] if i and i not in ids]
        if gone:
            warns.append(f"答案表里有 {len(gone)} 株找不到照片：{', '.join(gone[:10])}")
    else:
        ans = pd.DataFrame([{c: "" for c in MASTER_COLS} | {"isolate_id": i} for i in ids], columns=MASTER_COLS)
        os.makedirs(os.path.dirname(os.path.abspath(a.answers)), exist_ok=True)
        ans.to_csv(a.answers, index=False, encoding="utf-8-sig")
        print(f"📝 新建答案表 {a.answers}（{len(ids)} 株），请填写 reference / train_class / method")

    classes = []
    if a.micro_model:
        d = resolve_model(a.micro_model)
        if d and os.path.exists(os.path.join(d, "class_names.json")):
            classes = json.load(open(os.path.join(d, "class_names.json"), encoding="utf-8"))
        else:
            warns.append(f"读不到镜检模型的 class_names.json（{a.micro_model}），train_class 未核对")

    for c in ["reference", "train_class", "section", "method"]:
        ans[c] = ans[c].astype(str).str.strip()
    filled = ans[ans["isolate_id"].isin(ids)]
    todo = filled[filled["reference"] == ""]["isolate_id"].tolist()
    if classes:
        bad = filled[(filled["train_class"] != "") & ~filled["train_class"].isin(classes)]
        for _, r in bad.iterrows():
            errs.append(f"答案表 {r['isolate_id']} 的 train_class「{r['train_class']}」不是模型的类名（要照抄，含大小写和空格）")
    badsec = filled[(filled["section"] != "") & ~filled["section"].isin(SECTIONS + ["Other"])]
    for _, r in badsec.iterrows():
        errs.append(f"答案表 {r['isolate_id']} 的 section「{r['section']}」应为 {'/'.join(SECTIONS)}/Other 或留空")

    # 推断 section；曲霉缺菌落照片提示
    def sec(r):
        return r["section"] or section_of(r["reference"])
    filled = filled.assign(section_used=filled.apply(sec, axis=1))
    for _, r in filled.iterrows():
        if is_aspergillus(r["reference"]) and colony.get(r["isolate_id"], Counter())["O"] == 0:
            warns.append(f"{r['isolate_id']}（{r['reference']}）是曲霉但没有菌落正面照片 → 只能用镜检结果")

    # ---------------------------------------------------------------- 4. 素材清单
    inv = pd.DataFrame([{
        "isolate_id": i, "n_micro": len(micro.get(i, [])),
        "n_colony_O": colony.get(i, Counter())["O"], "n_colony_R": colony.get(i, Counter())["R"],
    } for i in ids])
    if a.inventory and len(inv):
        inv.to_csv(a.inventory, index=False, encoding="utf-8-sig")

    if a.split:
        out_dir = os.path.dirname(os.path.abspath(a.answers))
        f1 = filled[["isolate_id", "reference", "train_class", "method"]]
        f1.to_csv(os.path.join(out_dir, "answers_clinical.csv"), index=False, encoding="utf-8-sig")
        col = filled[filled["isolate_id"].isin([s for s, c in colony.items() if c["O"] > 0])]
        f2 = pd.DataFrame({"strain_id": col["isolate_id"],
                           "section": [s if s in SECTIONS else (ref or "未知") for s, ref in
                                       zip(col["section_used"], col["reference"])]})
        f2.to_csv(os.path.join(out_dir, "answers_colony.csv"), index=False, encoding="utf-8-sig")
        print(f"📄 已生成 answers_clinical.csv（{len(f1)} 株）、answers_colony.csv（{len(f2)} 株）")

    # ---------------------------------------------------------------- 报告
    n_asp = int(filled["reference"].map(is_aspergillus).sum())
    print("\n================ 素材盘点 ================")
    print(f"株数：{len(ids)}（有镜检 {len(micro)}，有菌落正面 {sum(1 for c in colony.values() if c['O'])}）")
    print(f"镜检图：{sum(len(v) for v in micro.values())} 张；菌落图：正面 {sum(c['O'] for c in colony.values())}，"
          f"反面 {sum(c['R'] for c in colony.values())}")
    print(f"答案表：已填 {len(filled) - len(todo)}/{len(filled)} 株（其中曲霉 {n_asp}）")
    if todo:
        print(f"   未填：{', '.join(todo[:15])}{' …' if len(todo) > 15 else ''}")
    if len(inv):
        print("\n每株素材（前 30 株）：")
        print(inv.head(30).to_string(index=False))
    for w in warns:
        print(f"⚠️  {w}")
    for e in errs:
        print(f"❌ {e}")
    if errs:
        sys.exit(2)
    if a.require_answers and todo:
        print(f"❌ 答案表还有 {len(todo)} 株没填 reference，评估需要先填完")
        sys.exit(3)


if __name__ == "__main__":
    main()
