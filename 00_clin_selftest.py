#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_clin_selftest.py —— 临床验证流程自检用的辅助脚本（不建议直接运行，请用 zsh clin_0_selftest.zsh）

思路：不需要任何临床素材，用"训练数据"冒充临床菌株，把真实模型 + 真实评估脚本完整跑一遍。
训练数据模型都见过，所以准确率应该接近 100%；明显偏低说明模型载入、类别顺序或预处理有问题。

两个子命令：
  prepare  从镜检训练清单里每类抽 1 株（每株 N 张截图）做成"假临床菌株"，
           写出 roi_clinical_table.csv（03_eval_clinical.py 直接读）和 answers_master.csv；
           曲霉株借用菌落训练集里同组菌株的株号，这样融合步骤能把两边对上
  verify   读取三步的结果，按阈值判定 通过 / 警告 / 失败，退出码 0 / 0 / 1
"""
import argparse
import json
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from clin_sections import SECTIONS, is_aspergillus, section_of  # noqa: E402

try:
    import paths as OP
except Exception:
    OP = None


def find_model(p):
    for c in (p, os.path.join(BASE, p), os.path.join(BASE, "output", "02_training_runs", p),
              os.path.join(BASE, "output", "02_training_runs", "runs", p)):
        if os.path.isdir(c):
            return c
    sys.exit(f"找不到镜检模型 {p}")


def find_trainset(model_dir, fallback):
    """优先用模型 args.json 记录的训练清单；否则用 --trainset"""
    cands = []
    try:
        a = json.load(open(os.path.join(model_dir, "args.json"), encoding="utf-8"))
        if a.get("trainset"):
            cands.append(a["trainset"])
    except Exception:
        pass
    cands.append(fallback)
    for c in cands:
        for p in (c, os.path.join(BASE, c), os.path.join(BASE, "output", "01_trainsets", c)):
            if os.path.isdir(p):
                p = os.path.join(p, "trainset_micro.csv")
            if os.path.isfile(p):
                return p
    sys.exit(f"找不到镜检训练清单（试过 {cands}）；在 clin_config.zsh 里设置 MICRO_TRAINSET")


def img_path(p):
    p = str(p).strip()
    return p if os.path.isabs(p) else os.path.join(BASE, p)


def prepare(a):
    mdir = find_model(a.micro_model)
    names = json.load(open(os.path.join(mdir, "class_names.json"), encoding="utf-8"))
    ts = find_trainset(mdir, a.trainset)
    df = pd.read_csv(ts, encoding="utf-8-sig")
    df = df[df["train_class"].isin(names)].copy()
    df["abs"] = df["path"].map(img_path)
    lost = int((~df["abs"].map(os.path.exists)).sum())
    df = df[df["abs"].map(os.path.exists)]
    if df.empty:
        sys.exit(f"训练清单 {ts} 里的图片一张都找不到（路径是否已移动？）")
    print(f"镜检训练清单：{ts}（{len(df)} 张可用" + (f"，{lost} 张找不到" if lost else "") + "）")

    # 菌落训练集里每组挑一个可评估的株号
    col_ids = {}
    if a.colony_pred and os.path.exists(os.path.join(a.colony_pred, "per_strain.csv")):
        ps = pd.read_csv(os.path.join(a.colony_pred, "per_strain.csv"), dtype={"strain_id": str})
        man = pd.read_csv(os.path.join(a.colony_pre, "manifest.csv"), dtype={"strain_id": str})
        grp = man.drop_duplicates("strain_id").set_index("strain_id")["group"]
        for sid in sorted(ps["strain_id"]):
            g = grp.get(sid, "")
            if g in SECTIONS and g not in col_ids:
                col_ids[g] = sid
        print("借用菌落训练株号：" + "，".join(f"{g}={s}" for g, s in col_ids.items()))

    rows, ans, used, k = [], [], set(), 0
    gcol = "group_id" if "group_id" in df.columns else None
    for c in names:
        d = df[df["train_class"] == c]
        if d.empty:
            print(f"  ⚠️ 训练清单里没有 {c} 的图片，跳过")
            continue
        if gcol:                                    # 尽量取同一株的截图
            top = d[gcol].value_counts().index[0]
            d = d[d[gcol] == top] if (d[gcol] == top).sum() >= a.n_roi else d
        d = d.sample(min(len(d), a.n_roi), random_state=0)
        sec = section_of(c)
        if sec in col_ids and sec not in used:
            iso = col_ids[sec]
            used.add(sec)
        else:
            k += 1
            iso = f"SELF-{k:03d}"
        for j, (_, r) in enumerate(d.iterrows()):
            rows.append({"roi_file": f"{iso}__r{j:02d}{os.path.splitext(r['abs'])[1]}", "roi_path": r["abs"],
                         "image_id": f"{iso}-{j}", "isolate_id": iso, "src_image": r["abs"], "reference": c,
                         "train_class": c, "in_panel": 1, "method": "selftest", "structure": "complete",
                         "stage": "mature", "size": "", "angle": ""})
        ans.append({"isolate_id": iso, "reference": c, "train_class": c, "section": "", "method": "selftest", "note": ""})
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "roi_clinical_table.csv"), index=False, encoding="utf-8-sig")
    # 菌落里没被借用的株也写进答案表（融合时成为"仅菌落"株）
    if a.colony_pred and col_ids:
        ps = pd.read_csv(os.path.join(a.colony_pred, "per_strain.csv"), dtype={"strain_id": str})
        man = pd.read_csv(os.path.join(a.colony_pre, "manifest.csv"), dtype={"strain_id": str})
        grp = man.drop_duplicates("strain_id").set_index("strain_id")["group"]
        have = {r["isolate_id"] for r in ans}
        for sid in ps["strain_id"]:
            if sid not in have:
                g = grp.get(sid, "")
                ans.append({"isolate_id": sid, "reference": f"Aspergillus sect. {g}", "train_class": "",
                            "section": g if g in SECTIONS else "", "method": "selftest", "note": "仅菌落"})
    pd.DataFrame(ans).to_csv(os.path.join(a.out, "answers_master.csv"), index=False, encoding="utf-8-sig")
    print(f"假临床菌株 {len({r['isolate_id'] for r in rows})} 株、ROI {len(rows)} 个 → {a.out}")


def verify(a):
    res = []      # (项目, 结果字符串, 状态 ok/warn/fail)

    def judge(name, k, n, ok_at, fail_at):
        if n == 0:
            res.append((name, "没有数据", "warn"))
            return
        v = k / n
        st = "ok" if v >= ok_at else ("warn" if v >= fail_at else "fail")
        res.append((name, f"{v:.3f}（{k}/{n}）", st))

    mj = os.path.join(a.dir, "micro", "metrics.json")
    if os.path.exists(mj):
        m = json.load(open(mj, encoding="utf-8"))
        judge("镜检模型 株级 top1（训练图，应≈1）", m["top1"][0], m["top1"][1], 0.90, 0.60)
    else:
        res.append(("镜检模型", "没有结果", "fail"))

    ps = os.path.join(a.dir, "colony", "per_strain.csv")
    if a.colony_env_failed:
        res.append(("菌落运行环境", "scikit-learn 版本与训练时不兼容（见上方提示）", "fail"))
    elif os.path.exists(ps):
        p = pd.read_csv(ps, dtype={"strain_id": str})
        man = pd.read_csv(os.path.join(a.colony_pre, "manifest.csv"), dtype={"strain_id": str})
        grp = man.drop_duplicates("strain_id").set_index("strain_id")["group"]
        p["g"] = p["strain_id"].map(grp)
        ri = os.path.join(a.dir, "colony", "run_info.json")
        ccls = json.load(open(ri, encoding="utf-8"))["classes"] if os.path.exists(ri) else sorted(p["g"].dropna().unique())
        p = p[p["g"].isin(ccls)]                      # 菌落模型没有的组（如被剔除的 Nidulantes）不计
        judge("菌落模型 株级 top1（训练株，应≈1）", int((p["top1"] == p["g"]).sum()), len(p), 0.85, 0.50)
    else:
        res.append(("菌落模型", "没有结果（菌落部分失败？）", "warn"))

    pf = os.path.join(a.dir, "final", "per_isolate_final.csv")
    if os.path.exists(pf):
        f = pd.read_csv(pf, dtype={"isolate_id": str})
        f = f[~f["通路"].astype(str).str.startswith("仅菌落")]   # 自检里"仅菌落"的株是凑数的，不计
        ok = f["最终正确"].dropna().astype(str).str.lower().eq("true") if "最终正确" in f else pd.Series(dtype=bool)
        judge("融合后最终准确率（有镜检的株）", int(ok.sum()), len(ok), 0.90, 0.60)
        fu = f[f["通路"] == "曲霉：镜检+菌落融合"]
        res.append(("镜检+菌落 成功对上的曲霉株", f"{len(fu)} 株", "ok" if len(fu) else "warn"))
    else:
        res.append(("融合", "没有结果", "fail"))

    icon = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}
    lines = ["===== 自检结果 ====="] + [f"{icon[s]} {n:<28s}{v}" for n, v, s in res]
    worst = "fail" if any(s == "fail" for *_, s in res) else ("warn" if any(s == "warn" for *_, s in res) else "ok")
    lines.append({"ok": "🎉 自检全部通过：模型载入、类别顺序、预处理、融合都正常，可以开始临床验证",
                  "warn": "⚠️  自检基本通过，但有警告项，请看上面 ⚠️ 的说明",
                  "fail": "❌ 自检未通过：结果明显异常，先不要用于临床验证"}[worst])
    txt = "\n".join(lines)
    open(os.path.join(a.dir, "SELFTEST_RESULT.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    sys.exit(1 if worst == "fail" else 0)


def _conda_pythons():
    """当前 conda/miniforge 安装下所有环境的 python"""
    pre = os.path.realpath(sys.prefix)
    root = pre.split(os.sep + "envs" + os.sep)[0] if (os.sep + "envs" + os.sep) in pre else pre
    cands = [os.path.join(root, "bin", "python")]
    ed = os.path.join(root, "envs")
    if os.path.isdir(ed):
        cands += [os.path.join(ed, e, "bin", "python") for e in sorted(os.listdir(ed))]
    return [c for c in cands if os.path.exists(c)]


def envcheck(a):
    """菌落模型能否在当前 python 环境里正常预测；不能时扫描其他 conda 环境并推荐"""
    import subprocess
    import warnings
    import numpy as np
    print(f"菌落脚本使用的 python：{sys.executable}")
    try:
        import sklearn
        import joblib
        cur = sklearn.__version__
    except ImportError as e:
        print(f"❌ 当前环境缺少 {e.name}")
        cur = None
    trained, err = None, None
    if cur:
        try:
            mf = os.path.join(a.colony_model, "final_model.joblib")
            import re
            mm = re.search(rb"_sklearn_version.{1,6}?(\d+\.\d+(?:\.\d+)?(?:\.?(?:dev|rc|post)\d*)?)",
                           open(mf, "rb").read(), re.S)
            trained = mm.group(1).decode() if mm else None          # 模型文件里记录的训练时版本
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always")
                b = joblib.load(mf)
            for w in wl:                                             # 版本不同时 sklearn 会给出原版本
                ov = getattr(w.message, "original_sklearn_version", None)
                if ov:
                    trained = ov
            method = b.get("method")
            sc, m = b["scaler"], b["model"]
            X = np.zeros((2, int(getattr(sc, "n_features_in_", 0) or sc.mean_.shape[0])))
            m.predict_proba(sc.transform(X))          # 真正调用一次，和 03_colony_predict 同一条路径
            if method == "B":
                import torch  # noqa: F401           方案 B 要在这个环境里用 torch 提特征
            print(f"   scikit-learn：当前 {cur}；训练菌落模型时 {trained or '未记录'}；方案 {method}")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    if cur and not err and (trained is None or trained == cur):
        print("✅ 菌落模型可以在这个环境里正常预测，scikit-learn 版本与训练时一致")
        return
    if cur and not err:
        print(f"⚠️  能运行，但 scikit-learn 版本（{cur}）与训练时（{trained}）不同；为了结果可复现，建议换成训练时的环境")
    else:
        print(f"❌ 菌落模型在这个环境里无法预测：{err or '缺少依赖'}")
        if trained:
            print(f"   原因：模型是用 scikit-learn {trained} 训练保存的，当前环境是 {cur}，两者不兼容")
    # 扫描其他环境
    print("\n本机各 conda 环境的 scikit-learn 版本：")
    code = ("import sys\ntry:\n import sklearn,joblib,cv2,scipy,pandas;print(sklearn.__version__)\n"
            "except Exception as e:\n print('缺少 '+getattr(e,'name',str(e)))")
    hit = []
    for py in _conda_pythons():
        try:
            out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            out = "无法运行"
        mark = "  ← 与训练时一致" if trained and out == trained else ""
        if mark:
            hit.append(py)
        print(f"   {py}    {out}{mark}")
    print("\n解决办法（二选一）：")
    if hit:
        print(f"  A（推荐）在 clin_config.zsh 里写：COLONY_PYTHON=\"{hit[0]}\"")
        print("     菌落相关脚本就会用训练时的环境运行，模型文件不变，锁定指纹不变")
    else:
        print(f"  A 新建一个与训练时相同版本的环境（清华源）：")
        print(f"     conda create -n colony python=3.12 -y && conda activate colony")
        print(f"     pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn=={trained or '<训练时版本>'} "
              "joblib scipy pandas opencv-python pillow pillow-heif matplotlib")
        print("     然后 COLONY_PYTHON=\"$(conda run -n colony which python)\" 填进 clin_config.zsh")
    print(f"  B 在当前环境里重新训练菌落模型：python 02_colony_train.py  → 得到新的 colony/<时间> 文件夹，")
    print("     把 COLONY_MODEL 改成它（模型会变，需重新锁定；正式验证开始前这样做没有问题）")
    sys.exit(1 if (err or not cur) else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--micro_model", required=True)
    p.add_argument("--trainset", default="trainset_v4")
    p.add_argument("--colony_pred", default="")
    p.add_argument("--colony_pre", default="")
    p.add_argument("--n_roi", type=int, default=3)
    p.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    v.add_argument("--colony_pre", default="")
    v.add_argument("--colony_env_failed", action="store_true")
    e = sub.add_parser("envcheck")
    e.add_argument("--colony_model", required=True)
    a = ap.parse_args()
    {"prepare": prepare, "verify": verify, "envcheck": envcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
