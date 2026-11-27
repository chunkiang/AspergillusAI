#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_colony_predict.py — 用锁定的菌落模型预测新菌株（前瞻验证 / 外部验证）

前提：新菌株的菌落照片已经走完同样的标注和预处理，并且用单独的名字，不与训练集混放：
  python 01_colony_annotate.py   --root "/…/前瞻菌落_2026" --ann_dir prospective_annotations
  python 01_colony_preprocess.py pick  --root "/…/前瞻菌落_2026" --ann_dir prospective_annotations
  python 01_colony_preprocess.py build --root "/…/前瞻菌落_2026" --ann_dir prospective_annotations --out prospective_preprocessed

盲法命名建议：前瞻株用不含菌种信息的代码，如 PUPH-PCL-001-B-O-1.jpg（PCL = 前瞻临床），
  序号与镜检的株号一致（如 P001 ↔ PUPH-PCL-001），后面融合时可自动对齐。

运行
  # 预测（揭盲前）
  python 03_colony_predict.py --model colony/20261020_2108 --pre prospective_preprocessed

  # 揭盲后计算指标：answers.csv 两列  strain_id, section
  #   section ∈ Fumigati / Flavi / Nigri / Terrei / Nidulantes / 其他菌种名（集外）
  python 03_colony_predict.py --model colony/20261020_2108 --pre prospective_preprocessed --answers answers_colony.csv

  # 自检：用训练集本身跑一遍，株级准确率应接近 100%（只检查流程，不能写进论文）
  python 03_colony_predict.py --model colony/20261020_2108 --pre colony_preprocessed --sanity

会拒绝运行的情况
  · final_model.joblib 的 sha256 与 locked_config.json 不一致（模型被改动）
  · 新数据的预处理参数与训练时不一致（白平衡目标、尺度、截图大小等）
  · 新数据中有株号或图片（sha1）与训练集重叠（--sanity 除外）

判读规则（与镜检临床验证一致，可改，改动会记录在 run_info.json）
  top1 < 0.70 或 top1 − top2 < 0.20 → 需复核；
  方案 A 另加：颜色/直径特征明显超出训练株范围（可能是集外菌种或拍摄异常）→ 需复核；
  需复核且 top1/top2 涉及 Fumigati、Terrei 或 Flavi → 建议 50 °C 生长试验

输出：Audit/output/03_external_eval/colony_pred_<预处理名>_<时间>/
  per_strain.csv   每株：各组概率（温度校准后）、top1/top2、是否复核、建议操作、颜色特征
  excluded.csv     无法评估的株（无白平衡参照、无菌落几何等）及原因
  metrics.txt      仅在提供 --answers 时生成
  run_info.json    模型哈希、预处理哈希、阈值、参数
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent


def _load_train_module():
    p = AUDIT_DIR / "02_colony_train.py"
    if not p.exists():
        sys.exit("找不到 02_colony_train.py：03_colony_predict.py 需要与它放在同一个 Audit 文件夹")
    spec = importlib.util.spec_from_file_location("colony_train", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T2 = _load_train_module()   # 复用：特征提取、路径约定、评估函数
import joblib  # noqa: E402

VERSION = "03_colony_predict v1.0 (2026-10-20)"
PRE_KEYS = ["PLATE_R", "CANVAS", "CROP", "WHITE_SRGB", "GRAY", "RING", "AGAR_AUTO", "EDGE_MAX_BAD", "MIN_REF_PX"]
FIFTY_C = {"Fumigati", "Terrei", "Flavi"}
OOD_Q = 0.999          # 方案 A：特征离训练分布过远（对角马氏距离 > χ² 的 99.9% 分位）→ 标记"超出训练范围"


def sha256(p):
    return T2.sha256(p)


def check_lock(model_dir: Path, force: bool):
    mpath, lpath = model_dir / "final_model.joblib", model_dir / "locked_config.json"
    if not mpath.exists() or not lpath.exists():
        sys.exit(f"{model_dir} 下缺少 final_model.joblib 或 locked_config.json")
    lock = json.loads(lpath.read_text(encoding="utf-8"))
    h = sha256(mpath)
    if h != lock["sha256"]["final_model.joblib"]:
        msg = f"模型文件哈希与锁定记录不一致：\n  锁定 {lock['sha256']['final_model.joblib']}\n  当前 {h}"
        if not force:
            sys.exit(msg + "\n拒绝运行（--force 可强制，但结果不能用于论文）")
        print("⚠ " + msg)
    return joblib.load(mpath), lock, h


def check_preprocess(bundle, pre: Path, force: bool):
    cfg_path = pre / "preprocess_config.json"
    if not cfg_path.exists():
        sys.exit(f"找不到 {cfg_path}，请先运行 01_colony_preprocess.py build")
    new = json.loads(cfg_path.read_text(encoding="utf-8"))
    old = bundle.get("preprocess_config") or {}
    diff = [k for k in PRE_KEYS if k in old and json.dumps(old.get(k)) != json.dumps(new.get(k))]
    notes = []
    if diff:
        msg = "预处理参数与训练时不一致：" + "，".join(f"{k} 训练 {old.get(k)} / 现在 {new.get(k)}" for k in diff)
        if not force:
            sys.exit(msg + "\n拒绝运行（--force 可强制，但结果不能用于论文）")
        notes.append("⚠ " + msg)
    if old.get("agar_target_srgb") and new.get("agar_target_srgb") and old["agar_target_srgb"] != new["agar_target_srgb"]:
        notes.append(f"提示：培养基标准色 训练 {[round(v * 255) for v in old['agar_target_srgb']]} / "
                     f"本批 {[round(v * 255) for v in new['agar_target_srgb']]}（只影响培养基模式校正的图）")
    return new, notes


def image_sha1s(ann_dir):
    out = {}
    if ann_dir and Path(ann_dir).is_dir():
        for jp in Path(ann_dir).rglob("*.json"):
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                if d.get("sha1"):
                    out[d["sha1"]] = d.get("strain_id")
            except Exception:
                pass
    return out


def check_overlap(model_dir, bundle, new_cfg, man, sanity):
    """株号重叠（训练 OOF 表）+ 图片 sha1 重叠（训练与本批的标注 JSON）"""
    train_ids = set()
    oof = model_dir / "oof_colony_selected_calibrated.csv"
    if oof.exists():
        train_ids = set(pd.read_csv(oof)["strain_id"].astype(str))
    ids_hit = sorted(set(man["strain_id"].astype(str)) & train_ids)
    old_cfg = bundle.get("preprocess_config") or {}
    tr_sha = image_sha1s(old_cfg.get("ann_dir"))
    new_sha = image_sha1s(new_cfg.get("ann_dir"))
    sha_hit = sorted({f"{new_sha[h]}（= 训练 {tr_sha[h]}）" for h in set(tr_sha) & set(new_sha)})
    if sanity:
        return [f"自检模式：与训练集重叠的株 {len(ids_hit)} 个（预期全部重叠）"]
    if ids_hit or sha_hit:
        lines = ["本批数据与训练集重叠，拒绝运行："]
        if ids_hit:
            lines.append(f"  株号重叠 {len(ids_hit)} 个：{', '.join(ids_hit[:10])}{' …' if len(ids_hit) > 10 else ''}")
        if sha_hit:
            lines.append(f"  图片完全相同 {len(sha_hit)} 张：{', '.join(sha_hit[:10])}")
        lines.append("  前瞻株请用新的株号（如 PUPH-PCL-001）；若确为自检请加 --sanity")
        sys.exit("\n".join(lines))
    if not tr_sha:
        return ["提示：未找到训练集标注，图片级重叠检查未执行（只检查了株号）"]
    return []


def features(bundle, pre, df, out_dir):
    if bundle["method"] == "A":
        X, keep, bad = [], [], []
        for i, (_, r) in enumerate(df.iterrows()):
            f = T2.features_A(pre, r)
            if f is None:
                bad.append(r["strain_id"])
            else:
                X.append(f)
                keep.append(i)
        return np.stack(X) if X else np.zeros((0, len(T2.FEAT_A))), keep, bad
    X = T2.extract_B(pre, df, bundle["backbone"], bundle.get("weights"), out_dir / "feature_cache")
    return X, list(range(len(df))), []


def main():
    ap = argparse.ArgumentParser(description="锁定菌落模型的预测与评估")
    ap.add_argument("--model", required=True, help="训练输出文件夹；短名字 → output/02_training_runs/<名字>，如 colony/20261020_2108")
    ap.add_argument("--pre", required=True, help="新数据的预处理文件夹；短名字 → output/01_trainsets/<名字>")
    ap.add_argument("--answers", default=None, help="揭盲答案 CSV：strain_id, section")
    ap.add_argument("--out", default=None, help="短名字 → output/03_external_eval/<名字>")
    ap.add_argument("--t1", type=float, default=0.70, help="top1 概率低于此值 → 复核")
    ap.add_argument("--t2", type=float, default=0.20, help="top1 − top2 低于此值 → 复核")
    ap.add_argument("--sanity", action="store_true", help="自检：允许与训练集重叠，结果不能用于论文")
    ap.add_argument("--force", action="store_true", help="忽略哈希/参数不一致（仅调试）")
    args = ap.parse_args()
    T2.start_log("03_colony_predict")

    model_dir = T2.resolve_in("02_training_runs", args.model)
    pre = T2.resolve_in("01_trainsets", args.pre)
    if not (pre / "manifest.csv").exists():
        sys.exit(f"找不到 {pre / 'manifest.csv'}")
    bundle, lock, mhash = check_lock(model_dir, args.force)
    new_cfg, notes = check_preprocess(bundle, pre, args.force)
    classes = bundle["classes"]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    name = args.out or f"colony_pred_{pre.name}{'_sanity' if args.sanity else ''}_{stamp}"
    out = T2.resolve_in("03_external_eval", name)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(pre / "manifest.csv")
    man = man[man["view"] == "O"].copy()
    notes += check_overlap(model_dir, bundle, new_cfg, man, args.sanity)

    # ---------- 无法评估的株 ----------
    ok = man["main_analysis"].astype(bool)
    excl_rows = []
    for sid, g in man.groupby("strain_id"):
        if not ok[g.index].any():
            excl_rows.append({"strain_id": sid, "n_images": len(g),
                              "reason": "; ".join(sorted(set(";".join(g["flags"].fillna("").astype(str)).split(";")) - {""}))})
    use = man[ok].reset_index(drop=True)

    # ---------- 特征 → 株级 → 概率 ----------
    X_img, keep, bad = features(bundle, pre, use, out)
    for sid in bad:
        if sid not in set(use.iloc[keep]["strain_id"]):
            excl_rows.append({"strain_id": sid, "n_images": int((use["strain_id"] == sid).sum()),
                              "reason": "菌落像素不足，无法计算颜色特征"})
    use = use.iloc[keep].reset_index(drop=True)
    if len(use) == 0:
        sys.exit("没有可评估的图片")
    use["cv_group"] = use["strain_id"]
    sids, Xs, _ = T2.strain_table(use, X_img)
    _, LG = T2.predict_lr(bundle["scaler"], bundle["model"], Xs, len(classes))
    P = T2.softmax_T(LG, bundle["T"])

    order = np.argsort(-P, 1)
    ood_d2 = None
    if bundle["method"] == "A":
        from scipy.stats import chi2
        z = bundle["scaler"].transform(Xs)
        ood_d2 = (z ** 2).sum(1)
        ood_thr = float(chi2.ppf(OOD_Q, Xs.shape[1]))
    rows = []
    for i, sid in enumerate(sids):
        p1, p2 = P[i, order[i, 0]], P[i, order[i, 1]]
        c1, c2 = classes[order[i, 0]], classes[order[i, 1]]
        ood = bool(ood_d2 is not None and ood_d2[i] > ood_thr)
        flag = bool(p1 < args.t1 or (p1 - p2) < args.t2 or ood)
        act = []
        if ood:
            act.append("特征超出训练范围（集外菌种？拍摄异常？）")
        if flag:
            act.append("人工复核 / MALDI-TOF / 测序")
            if {c1, c2} & FIFTY_C:
                act.append("50 °C 生长试验")
        r = {"strain_id": sid, "n_images": int((use["strain_id"] == sid).sum())}
        r.update({f"p_{c}": round(float(P[i, j]), 4) for j, c in enumerate(classes)})
        r.update({"top1": c1, "p_top1": round(float(p1), 4), "top2": c2, "p_top2": round(float(p2), 4),
                  "margin": round(float(p1 - p2), 4), "超出训练范围": ood,
                  "特征距离D2": None if ood_d2 is None else round(float(ood_d2[i]), 1),
                  "需复核": flag, "建议操作": "；".join(act)})
        if bundle["method"] == "A":
            r.update({k: round(float(v), 2) for k, v in zip(T2.FEAT_A, Xs[i])})
        rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_strain.csv", index=False, encoding="utf-8-sig")
    excl = pd.DataFrame(excl_rows, columns=["strain_id", "n_images", "reason"])
    excl.to_csv(out / "excluded.csv", index=False, encoding="utf-8-sig")

    head = [f"===== 菌落模型预测 =====",
            f"运行时间 {dt.datetime.now():%Y-%m-%d %H:%M}；{VERSION}",
            f"模型：{model_dir}（方案 {bundle['method']}，类别 {', '.join(classes)}，T = {bundle['T']:.3f}，锁定于 {lock.get('locked_at')}）",
            f"数据：{pre}",
            f"判读规则：top1 < {args.t1} 或 top1 − top2 < {args.t2}"
            + (f"，或特征距离 D² > {ood_thr:.1f}（χ² {OOD_Q} 分位，超出训练范围）" if ood_d2 is not None else "")
            + " → 需复核"] + notes
    head.append(f"可评估 {len(per)} 株；无法评估 {len(excl)} 株"
                + (f"（{', '.join(excl['strain_id'])}）" if len(excl) else ""))
    head.append(f"需复核 {int(per['需复核'].sum())}/{len(per)} 株")
    if args.sanity:
        head.append("【自检模式：数据与训练集重叠，结果只用于检查流程，不能写进论文】")

    # ---------- 揭盲后指标 ----------
    report = list(head)
    if args.answers:
        ans = pd.read_csv(T2.resolve_in("03_external_eval", args.answers) if not Path(args.answers).exists()
                          else Path(args.answers))
        ans.columns = [c.strip() for c in ans.columns]
        col = "section" if "section" in ans.columns else ("train_class" if "train_class" in ans.columns else None)
        if "strain_id" not in ans.columns or col is None:
            sys.exit("answers 文件需要两列：strain_id, section")
        amap = dict(zip(ans["strain_id"].astype(str), ans[col].astype(str).str.strip()))
        miss = [s for s in per["strain_id"] if s not in amap]
        if miss:
            sys.exit(f"以下株在 answers 中缺少答案：{', '.join(miss)}")
        per["reference"] = per["strain_id"].map(amap)
        per["in_panel"] = per["reference"].isin(classes)
        per["correct"] = per["top1"] == per["reference"]
        per.to_csv(out / "per_strain.csv", index=False, encoding="utf-8-sig")

        inp = per[per["in_panel"]]
        report.append("")
        if len(inp):
            y = inp["reference"].map({c: i for i, c in enumerate(classes)}).values
            Pin = inp[[f"p_{c}" for c in classes]].values
            lines, cm, sm = T2.evaluate(Pin, y, classes, "集内菌株（株级）")
            report += lines
            T2.plot_cm(cm, classes, "Colony model, locked", out / "confusion.png")
            fl = inp["需复核"].values
            cor = inp["correct"].values
            for lab, m in [("未标记（可直接报告）", ~fl), ("标记复核", fl)]:
                k, n = int(cor[m].sum()), int(m.sum())
                if n:
                    lo, hi = T2.cp_ci(k, n)
                    report.append(f"{lab}的准确率  {k / n:.3f}（{k}/{n}，95% CI {lo:.3f}–{hi:.3f}）")
            k, n = int(fl.sum()), len(fl)
            lo, hi = T2.cp_ci(k, n)
            report.append(f"复核标记比例        {k / n:.3f}（{k}/{n}，95% CI {lo:.3f}–{hi:.3f}）")
            wrong = inp[~inp["correct"]]
            if len(wrong):
                report.append("判错的株：")
                for _, r in wrong.iterrows():
                    report.append(f"  {r['strain_id']:<16}{r['reference']:<12}→ {r['top1']}（{r['p_top1']:.2f}） "
                                  f"第2 {r['top2']}（{r['p_top2']:.2f}）  {'已标记' if r['需复核'] else '未标记'}")
        outp = per[~per["in_panel"]]
        if len(outp):
            k = int(outp["需复核"].sum())
            report.append(f"\n集外菌株 {len(outp)} 株（模型只能在 {len(classes)} 个组中选，必然判错）："
                          f"被标记复核 {k}/{len(outp)}")
            for _, r in outp.iterrows():
                report.append(f"  {r['strain_id']:<16}{r['reference']:<20}→ {r['top1']}（{r['p_top1']:.2f}）"
                              f"  {'已标记' if r['需复核'] else '未标记'}")
        txt = "\n".join(report)
        (out / "metrics.txt").write_text(txt, encoding="utf-8")
    else:
        report.append("\n（未提供 --answers：只输出预测，未计算指标）")

    info = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "version": VERSION, "args": vars(args),
            "model_dir": str(model_dir), "model_sha256": mhash, "locked_sha256": lock["sha256"]["final_model.joblib"],
            "manifest_sha256": sha256(pre / "manifest.csv"), "classes": classes, "T": bundle["T"],
            "thresholds": {"t1": args.t1, "t2": args.t2}, "n_evaluable": len(per), "n_excluded": len(excl),
            "sanity": args.sanity, "forced": args.force}
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(report))
    print("\n每株预测：")
    cols = ["strain_id", "top1", "p_top1", "top2", "p_top2", "需复核"] + (["reference"] if "reference" in per else [])
    print(per[cols].to_string(index=False))
    print(f"\n输出目录：{out}")


if __name__ == "__main__":
    main()
