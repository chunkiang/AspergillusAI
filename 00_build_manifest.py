#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_build_manifest.py —— 把审计结果 + 答案表 + 菌名对照表合成为外部验证 manifest

用法：
    python 00_build_manifest.py      # 默认读 ./ext_audit/inventory.csv，答案表和对照表与本脚本同目录

规则：
  - answers.csv: folder_key 是文件夹名里的关键字（如 "谭盼"），
    category 为 A/B/C 或 "*"（表示该文件夹 A/B/C 共用一套连续编号）
  - 没有编码、但文件名含中文菌名的图（马琼琼、已整理、蒋宇杰等），用文件名推断菌名，标记 label_source=filename
  - role 判定顺序：
      excluded_folder       文件夹名带 ❌（如王若帆：原始平板/涂片，不可用）
      dev_source            开发集截图的原始图（已整理、eXPORT，已确认），永不作外部测试
      dev_cropped           用 01_crop_batch.py 截过、已补充进训练集的外部原图
      dev_set_checked       ✅ 文件夹（截图已放入训练目录，归入开发集），或与训练集图片重复
      excluded_direct       A 类原始标本直接镜检
      excluded_duplicate    外部集内部重复（保留编号最小的一张）
      no_answer             还没有答案
      out_of_panel          菌种不在训练类别里（可用于"未知菌种能否被标记复核"的分析）
      external_micro        纯培养镜下图，菌种在训练类别里 —— 外部验证主分析
      external_colony       纯菌落图，菌种在训练类别里 —— 菌落融合次要分析
"""
import os
import re
import sys
import argparse
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

CODE_RE = re.compile(r"^([A-Za-z]+)-([ABC])-(\d+)", re.I)


def clean_folder(s):
    return re.sub(r"[✅❌\s]", "", str(s))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--inventory", default=OP.INVENTORY)
    ap.add_argument("--answers", default=os.path.join(here, "answers.csv"))
    ap.add_argument("--taxonomy", default=os.path.join(here, "taxonomy_map.csv"))
    ap.add_argument("--out", default=OP.MANIFEST)
    ap.add_argument("--dev_source_folders", nargs="*", default=["已整理", "eXPORT"],
                    help="开发集截图的原始图所在文件夹（赵老师已确认），永不作外部测试")
    ap.add_argument("--crop_log", default="/Users/chunjiang/Documents/真菌AI/CNN/crops_new/crop_log.csv",
                    help="01_crop_batch.py 的截图记录；截过的原图标记为 dev_cropped，不再作外部测试")
    a = ap.parse_args()
    a.inventory = OP.find(a.inventory, OP.EXT_AUDIT)
    a.out = OP.place(a.out, OP.EXT_AUDIT)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for p, what in ((a.inventory, "审计结果 inventory.csv（先运行 00_audit_external.py）"),
                    (a.answers, "答案表 answers.csv（和本脚本放在同一文件夹）"),
                    (a.taxonomy, "菌名对照表 taxonomy_map.csv（和本脚本放在同一文件夹）")):
        if not os.path.exists(p):
            sys.exit(f"找不到 {what}：{p}")

    inv = pd.read_csv(a.inventory, encoding="utf-8-sig", dtype=str).fillna("")
    cropped = set()
    if a.crop_log and os.path.exists(a.crop_log):
        cl = pd.read_csv(a.crop_log, encoding="utf-8-sig", dtype=str).fillna("")
        cropped = set(cl.loc[cl["status"] == "crop", "source_image"])
        print(f"截图记录：{len(cropped)} 张原图已用于补充训练，将从外部测试中排除")

    def was_cropped(rel):
        return any(p.endswith("/" + rel) or p == rel for p in cropped)
    ans = pd.read_csv(a.answers, encoding="utf-8-sig", dtype=str).fillna("")
    tax = pd.read_csv(a.taxonomy, encoding="utf-8-sig", dtype=str).fillna("")
    ans["number"] = ans["number"].astype(int)
    tax_d = tax.set_index("species_cn").to_dict("index")
    tax_names = sorted(tax["species_cn"], key=len, reverse=True)

    lookup = {}
    for _, r in ans.iterrows():
        lookup[(r.folder_key, r.prefix.upper(), r.category.upper(), r.number)] = r.species_cn
    # prefix="*"：同一文件夹内不同姓名缩写共用一套连续编号（如 L-C-01、W-B-16、WJ-A-28）

    def find_answer(row):
        folder = clean_folder(row["folder"])
        stem = row["rel_path"].split("/")[-1].rsplit(".", 1)[0].strip()
        m = CODE_RE.match(stem)
        if m:
            pre, cat, num = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
            for (fk, p, c, n), sp in lookup.items():
                if fk in folder and p in (pre, "*") and n == num and c in (cat, "*"):
                    return sp, "answer_doc"
            # 文件名里自带菌名（如 H-C-02_构巢曲霉）
        for name in tax_names:
            if name in stem:
                return name, "filename"
        if row.get("label_hint"):
            return row["label_hint"], "filename_unmapped"
        return "", ""

    out = []
    for _, r in inv.iterrows():
        sp, src = find_answer(r)
        t = tax_d.get(sp, {})
        cat = r.get("category", "")
        modality = r.get("modality_guess", "")
        in_train = (str(r.get("folder_in_train", "0")) not in ("", "0")) or bool(r.get("near_dup_in_train"))
        if "❌" in r["folder"]:
            role = "excluded_folder"
        elif any(k in r["folder"] for k in a.dev_source_folders):
            role = "dev_source"
        elif in_train:
            role = "dev_set_checked"
        elif was_cropped(r["rel_path"]):
            role = "dev_cropped"
        elif modality == "direct_specimen":
            role = "excluded_direct"
        elif not sp:
            role = "no_answer"
        elif not t.get("train_class"):
            role = "out_of_panel"
        elif modality == "colony":
            role = "external_colony"
        else:
            role = "external_micro"
        out.append({**r.to_dict(), "species_cn": sp, "label_source": src,
                    "latin": t.get("latin", ""), "train_class": t.get("train_class", ""),
                    "tax_note": t.get("note", ""), "role": role})
    df = pd.DataFrame(out)

    # 外部集内部重复：同组只保留第一张
    seen = set()
    for i, r in df.iterrows():
        if r["role"].startswith("external") or r["role"] == "out_of_panel":
            if r["rel_path"] in seen:
                df.at[i, "role"] = "excluded_duplicate"
                continue
            for d in filter(None, str(r.get("dup_within_ext", "")).split(";")):
                seen.add(d)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")

    print("按 role 统计:")
    print(OP.table(df["role"].value_counts()))
    ext = df[df["role"].isin(["external_micro", "external_colony"])]
    if len(ext):
        print("\n可用外部图（菌种 × 模态）:")
        print(OP.table(pd.crosstab(ext["train_class"], ext["role"])))
        print("\n可用外部图（来源 × 模态）:")
        print(OP.table(pd.crosstab(ext["folder"], ext["role"])))
    unm = df[df["label_source"] == "filename_unmapped"]["species_cn"].unique()
    if len(unm):
        print("\n文件名里的菌名不在 taxonomy_map.csv 中，请补充：", "、".join(unm))
    print(f"\n已写出: {a.out}")


if __name__ == "__main__":
    main()
