#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_fusion_simpair.py —— 同节随机配对：估计 镜检 + 菌落 决策层融合的增益（内部数据，株级）

依据
  同一曲霉节的不同菌株，在相同培养条件下菌落和镜下形态高度一致，即在给定节的条件下，
  镜下图与菌落图近似条件独立（决策层融合本身的前提）。因此可以把同一节内、来自不同菌株的
  镜检预测和菌落预测随机配对，估计融合相对于只用镜检的增益。

输入（都是交叉验证的 OOF，即每个菌株只被\"没见过它\"的模型预测过）
  · 镜检：runs_v4 交叉验证的 OOF 概率（图片级 → 按菌株取平均）
  · 菌落：colony/<run>/oof_colony_selected_calibrated.csv（株级，温度校准后）
  · 类别 → 节 映射：00_fusion_sim_check.py 生成的 section_map_auto.csv

规则（与临床验证预设一致，默认）
  1. 分流：镜检 P(曲霉) = 所有曲霉类别概率之和；< --asp_gate（0.5）→ 判为非曲霉（节级算错）
  2. 镜检节概率 = 同节类别概率相加，在曲霉各节内归一化；镜检首位节 = 只用镜检的结果
  3. 镜检首位节不在菌落模型覆盖范围内 → 不融合，保持镜检结果
  4. 否则在菌落覆盖的节上：线性 f = (1 − w)·镜检 + w·菌落（w 默认 0.8）；--rule product 为乘积
  5. 评估对象：真实节属于菌落模型覆盖范围的曲霉菌株

输出（--out）
  report.txt                 中文结果，可直接用于论文
  per_strain.csv             每株：真实节、镜检结果、各权重下融合判对的比例
  iterations.csv             每次随机配对的准确率（主权重）
  weight_sensitivity.csv     权重 0–1 的敏感性分析
  confusion_micro.csv / confusion_fused_expected.csv
  weight_sensitivity.png     （有 matplotlib 时）
  RUN_INFO.json              脚本版本、参数、输入文件 sha256

安装（清华源）
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy scipy matplotlib

不要直接运行，用 zsh fsim_1_run.zsh
"""

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest
except ImportError as e:
    sys.exit(f"❌ 缺少依赖：{e}\n   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy scipy")

VERSION = "03_fusion_simpair v1.0 (2026-11-13)"
ASP_SECTIONS = ["Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes", "Versicolores", "Aspergillus"]
NONASP = "NonAsp"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def read_csv_any(p):
    for enc in ("utf-8-sig", "gbk"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit(f"❌ 读不了 {p}")


def softmax(x):
    z = x - x.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ====================================================================== 读数据
def load_micro(chk, smap):
    m = chk["micro"]
    df = read_csv_any(m["micro_oof"])
    P = df[m["prob_cols"]].astype(float).values
    if m["needs_softmax"]:
        P = softmax(P)
    else:
        P = P / P.sum(1, keepdims=True)
    classes = m["classes"]
    tv = df[m["true_col"]]
    true = [classes[int(i)] for i in tv] if m["true_is_index"] else tv.astype(str).tolist()
    g = df[m["group_col"]].astype(str).values
    # 图片 → 菌株：概率取平均
    gid, inv = np.unique(g, return_inverse=True)
    Ps = np.zeros((len(gid), P.shape[1]))
    np.add.at(Ps, inv, P)
    cnt = np.bincount(inv)
    Ps /= cnt[:, None]
    true_s = pd.Series(true).groupby(inv).first().values
    sec_of = dict(zip(smap["class"].astype(str), smap["section"].astype(str)))
    bad = [c for c in classes if c not in sec_of]
    if bad:
        sys.exit(f"❌ 映射表缺少类别：{bad[:5]}…；请重跑 fsim_0_check.zsh")
    csec = np.array([sec_of[c] for c in classes])
    if (csec == "Aspergillus_unknown").any():
        sys.exit("❌ 映射表里还有 Aspergillus_unknown，请先在 section_map_auto.csv 里填好节")
    secs = [s for s in ASP_SECTIONS if s in set(csec)]
    # 各节概率（未归一化）与 P(曲霉)
    S = np.stack([Ps[:, csec == s].sum(1) for s in secs], 1)
    p_asp = S.sum(1)
    true_sec = np.array([sec_of.get(t, NONASP) for t in true_s])
    return {"gid": gid, "n_img": cnt, "true_class": true_s, "true_sec": true_sec,
            "S": S, "p_asp": p_asp, "secs": secs, "n_images_total": int(len(df))}


def load_colony(chk):
    c = chk["colony"]
    df = read_csv_any(c["colony_oof"])
    pcols = [f"p_{k}" for k in c["colony_classes"]]
    P = df[pcols].astype(float).values
    P = P / P.sum(1, keepdims=True)
    return {"sid": df["strain_id"].astype(str).values, "true": df["true"].astype(str).values,
            "P": P, "classes": list(c["colony_classes"])}


# ====================================================================== 规则
def micro_pred(mic, gate):
    """只用镜检：返回 节名（或 NonAsp）"""
    secs = mic["secs"]
    out = np.array([secs[i] for i in mic["S"].argmax(1)], dtype=object)
    out[mic["p_asp"] < gate] = NONASP
    return out


def correctness_matrix(mic, col, idx_eval, w, rule, gate):
    """对每个被评估的镜检菌株 i、与其真实节相同的每个菌落株 j：融合后是否判对。
       返回 list[np.ndarray]（第 i 个元素长度 = 该节菌落株数），以及每株是否融合了"""
    secs = mic["secs"]
    ccls = col["classes"]
    cov = [s for s in ccls if s in secs]
    ci = [ccls.index(s) for s in cov]            # 菌落概率中覆盖节的列
    mi = [secs.index(s) for s in cov]            # 镜检节概率中覆盖节的列
    pool = {s: np.where(col["true"] == s)[0] for s in cov}
    mpred = micro_pred(mic, gate)
    res, fused_flag, mats = [], [], []
    for i in idx_eval:
        ts = mic["true_sec"][i]
        js = pool[ts]
        if mpred[i] == NONASP:                    # 分流到非曲霉：两种方法都错
            res.append(np.zeros(len(js), bool)); fused_flag.append("gated_nonasp"); mats.append(None)
            continue
        if mpred[i] not in cov:                  # 镜检首位节菌落模型不覆盖：不融合
            res.append(np.full(len(js), mpred[i] == ts)); fused_flag.append("not_covered"); mats.append(None)
            continue
        ms = mic["S"][i, mi]
        ms = ms / ms.sum() if ms.sum() > 0 else np.full(len(mi), 1 / len(mi))
        pc = col["P"][js][:, ci]
        pc = pc / pc.sum(1, keepdims=True)
        if rule == "linear":
            f = (1 - w) * ms[None, :] + w * pc
        else:
            f = ms[None, :] * pc
        pred = np.array(cov)[f.argmax(1)]
        res.append(pred == ts); fused_flag.append("fused"); mats.append(pred)
    return res, fused_flag, mats, cov, pool


def summarize(res, rng, n_iter, micro_ok, boot=True):
    """res[i]: 菌株 i 与各候选菌落株配对时融合是否判对。返回 期望准确率、配对区间、bootstrap 区间、McNemar"""
    n = len(res)
    exp_i = np.array([r.mean() for r in res])
    exp_acc = exp_i.mean()
    acc_it, b_it, c_it, p_it, diff_boot = [], [], [], [], []
    for _ in range(n_iter):
        pick = np.array([r[rng.integers(len(r))] for r in res])
        acc_it.append(pick.mean())
        b = int((micro_ok & ~pick).sum())       # 镜检对、融合错
        c = int((~micro_ok & pick).sum())       # 镜检错、融合对
        b_it.append(b); c_it.append(c)
        p_it.append(binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)
        if boot:
            # 株级 bootstrap：重抽镜检菌株，并在每株的同节菌落池里有放回抽取后再配对
            ii = rng.integers(n, size=n)
            fb = np.array([res[k][rng.integers(len(res[k]))] for k in ii])
            diff_boot.append(fb.mean() - micro_ok[ii].mean())
    acc_it = np.array(acc_it)
    out = {"exp_acc": float(exp_acc), "acc_lo": float(np.percentile(acc_it, 2.5)),
           "acc_hi": float(np.percentile(acc_it, 97.5)), "exp_i": exp_i, "acc_it": acc_it,
           "b_med": float(np.median(b_it)), "c_med": float(np.median(c_it)),
           "p_med": float(np.median(p_it)), "p_frac05": float(np.mean(np.array(p_it) < 0.05)),
           "b_it": b_it, "c_it": c_it, "p_it": p_it}
    d_pair = acc_it - micro_ok.mean()
    out.update({"diff": float(exp_acc - micro_ok.mean()),
                "diff_pair_lo": float(np.percentile(d_pair, 2.5)), "diff_pair_hi": float(np.percentile(d_pair, 97.5))})
    if boot:
        out.update({"diff_boot_lo": float(np.percentile(diff_boot, 2.5)),
                    "diff_boot_hi": float(np.percentile(diff_boot, 97.5))})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ====================================================================== 主程序
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="fsim_0_check 生成的 CHECK_RESULT.json")
    ap.add_argument("--section_map", default="", help="默认用 CHECK_RESULT.json 里记录的 section_map_auto.csv")
    ap.add_argument("--weight", type=float, default=0.8, help="菌落权重（线性融合）")
    ap.add_argument("--sens_weights", default="0,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--rule", default="linear", choices=["linear", "product"])
    ap.add_argument("--asp_gate", type=float, default=0.5)
    ap.add_argument("--n_iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"# {VERSION}\n# {' '.join(sys.argv)}")

    chk = json.loads(Path(args.check).read_text(encoding="utf-8"))
    if chk.get("problems"):
        sys.exit(f"❌ 诊断还有未解决的问题：{chk['problems']}；请先处理并重跑 fsim_0_check.zsh")
    smap_path = Path(args.section_map or chk["section_map"])
    smap = read_csv_any(smap_path)
    mic = load_micro(chk, smap)
    col = load_colony(chk)

    cov = [s for s in col["classes"] if s in mic["secs"]]
    idx_eval = np.where(np.isin(mic["true_sec"], cov))[0]
    if len(idx_eval) == 0:
        sys.exit("❌ 没有真实节属于菌落覆盖范围的镜检菌株")
    mpred = micro_pred(mic, args.asp_gate)
    micro_ok = (mpred[idx_eval] == mic["true_sec"][idx_eval])
    rng = np.random.default_rng(args.seed)

    # ---------- 主分析
    res, flag, _, cov, pool = correctness_matrix(mic, col, idx_eval, args.weight, args.rule, args.asp_gate)
    main_s = summarize(res, rng, args.n_iter, micro_ok, boot=True)

    # ---------- 敏感性分析
    sens = []
    ws = [float(x) for x in args.sens_weights.split(",") if x.strip()]
    per_w = {}
    for w in ws:
        r_w, _, _, _, _ = correctness_matrix(mic, col, idx_eval, w, "linear", args.asp_gate)
        s_w = summarize(r_w, np.random.default_rng(args.seed + 1), args.n_iter, micro_ok, boot=True)
        per_w[w] = s_w["exp_i"]
        sens.append({"colony_weight": w, "fused_acc_expected": round(s_w["exp_acc"], 4),
                     "fused_acc_lo": round(s_w["acc_lo"], 4), "fused_acc_hi": round(s_w["acc_hi"], 4),
                     "diff_vs_micro": round(s_w["diff"], 4),
                     "diff_boot_lo": round(s_w["diff_boot_lo"], 4), "diff_boot_hi": round(s_w["diff_boot_hi"], 4),
                     "mcnemar_p_median": round(s_w["p_med"], 4)})
    # 乘积融合作为另一种规则的对照
    r_p, _, _, _, _ = correctness_matrix(mic, col, idx_eval, 0, "product", args.asp_gate)
    s_p = summarize(r_p, np.random.default_rng(args.seed + 2), args.n_iter, micro_ok, boot=True)
    per_w["product"] = s_p["exp_i"]
    sens.append({"colony_weight": "product(乘积)", "fused_acc_expected": round(s_p["exp_acc"], 4),
                 "fused_acc_lo": round(s_p["acc_lo"], 4), "fused_acc_hi": round(s_p["acc_hi"], 4),
                 "diff_vs_micro": round(s_p["diff"], 4),
                 "diff_boot_lo": round(s_p["diff_boot_lo"], 4), "diff_boot_hi": round(s_p["diff_boot_hi"], 4),
                 "mcnemar_p_median": round(s_p["p_med"], 4)})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(out / "weight_sensitivity.csv", index=False, encoding="utf-8-sig")

    # ---------- 每株
    ps = pd.DataFrame({"strain": mic["gid"][idx_eval], "n_images": mic["n_img"][idx_eval],
                       "true_class": mic["true_class"][idx_eval], "true_section": mic["true_sec"][idx_eval],
                       "p_aspergillus": mic["p_asp"][idx_eval].round(4), "micro_pred": mpred[idx_eval],
                       "micro_correct": micro_ok.astype(int), "route": flag,
                       "n_colony_candidates": [len(pool[t]) for t in mic["true_sec"][idx_eval]]})
    for w, v in per_w.items():
        ps[f"fused_correct_rate_w{w}"] = np.round(v, 3)
    ps.to_csv(out / "per_strain.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"iter": range(1, args.n_iter + 1), "fused_acc": main_s["acc_it"],
                  "micro_right_fused_wrong": main_s["b_it"], "micro_wrong_fused_right": main_s["c_it"],
                  "mcnemar_p": main_s["p_it"]}).to_csv(out / "iterations.csv", index=False, encoding="utf-8-sig")

    # ---------- 混淆矩阵（镜检；融合取期望）
    labs = cov + [s for s in mic["secs"] if s not in cov] + [NONASP]
    ts = mic["true_sec"][idx_eval]
    cm_m = pd.DataFrame(0, index=cov, columns=labs, dtype=float)
    for t, p in zip(ts, mpred[idx_eval]):
        cm_m.loc[t, p] += 1
    cm_f = pd.DataFrame(0.0, index=cov, columns=labs)
    _, _, mats, _, _ = correctness_matrix(mic, col, idx_eval, args.weight, args.rule, args.asp_gate)
    for k, (t, mat) in enumerate(zip(ts, mats)):
        if mat is None:
            cm_f.loc[t, mpred[idx_eval][k]] += 1
        else:
            for lab in set(mat):
                cm_f.loc[t, lab] += float((mat == lab).mean())
    cm_m.astype(int).to_csv(out / "confusion_micro.csv", encoding="utf-8-sig")
    cm_f.round(2).to_csv(out / "confusion_fused_expected.csv", encoding="utf-8-sig")

    # ---------- 报告
    n = len(idx_eval)
    km = int(micro_ok.sum())
    lo, hi = wilson(km, n)
    L = [f"===== 同节随机配对：镜检 + 菌落 决策层融合（株级，内部 OOF） =====",
         f"运行时间 {dt.datetime.now():%Y-%m-%d %H:%M}；{VERSION}",
         f"镜检 OOF：{chk['micro']['micro_oof']}（{mic['n_images_total']} 张图，{len(mic['gid'])} 株）",
         f"菌落 OOF：{chk['colony']['colony_oof']}（{len(col['sid'])} 株）",
         f"节映射：{smap_path}",
         f"融合规则：{'线性' if args.rule == 'linear' else '乘积'}"
         + (f"，菌落权重 w = {args.weight}" if args.rule == "linear" else "")
         + f"；分流阈值 P(曲霉) ≥ {args.asp_gate}；镜检首位节不在菌落覆盖范围时不融合",
         f"随机配对次数 {args.n_iter}，随机种子 {args.seed}",
         "",
         f"评估的节（菌落模型覆盖）：{cov}",
         "评估菌株数（镜检）/ 菌落候选株数：" + "，".join(
             f"{s} {int((ts == s).sum())}/{len(pool[s])}" for s in cov),
         f"分流：融合 {flag.count('fused')} 株；镜检首位节不在覆盖范围、未融合 {flag.count('not_covered')} 株；"
         f"P(曲霉) < {args.asp_gate} 判为非曲霉 {flag.count('gated_nonasp')} 株",
         "",
         "----- 主要结果（节级 top-1 准确率） -----",
         f"只用镜检            {km / n:.3f}（{km}/{n}，95% CI {lo:.3f}–{hi:.3f}）",
         f"融合（期望值）      {main_s['exp_acc']:.3f}（随机配对 95% 区间 {main_s['acc_lo']:.3f}–{main_s['acc_hi']:.3f}）",
         f"差值                {main_s['diff']:+.3f}",
         f"  · 只考虑配对随机性的 95% 区间    {main_s['diff_pair_lo']:+.3f} 至 {main_s['diff_pair_hi']:+.3f}",
         f"  · 株级 bootstrap 95% CI（主报告） {main_s['diff_boot_lo']:+.3f} 至 {main_s['diff_boot_hi']:+.3f}",
         f"McNemar（每次配对各做一次）：镜检对融合错 中位数 {main_s['b_med']:.0f}，镜检错融合对 中位数 {main_s['c_med']:.0f}；"
         f"P 中位数 {main_s['p_med']:.4f}，P < 0.05 的比例 {main_s['p_frac05']:.1%}",
         "",
         "----- 各节 -----",
         f"  {'节':<12}{'株数':>5}{'镜检':>8}{'融合(期望)':>12}"]
    for s in cov:
        mk = ts == s
        L.append(f"  {s:<12}{int(mk.sum()):>5}{micro_ok[mk].mean():>8.3f}{main_s['exp_i'][mk].mean():>12.3f}")
    L += ["", "----- 混淆矩阵（行 = 真实节） -----", "只用镜检：", cm_m.astype(int).to_string(),
          "", "融合（期望计数）：", cm_f.round(1).to_string(), "",
          "----- 权重敏感性（线性；0 = 只用镜检的节概率，1 = 只用菌落） -----",
          sens_df.to_string(index=False), "",
          "----- 说明（写论文时注意） -----",
          "1. 镜检与菌落来自不同菌株，配对依据是\"给定节时两种形态条件独立\"。非典型菌株（产孢差、颜色变异）",
          "   在真实配对中两种模态可能同时出错，随机配对会把这种相关性平均掉，增益可能被高估；",
          "   真实配对的估计以中日友好医院前瞻队列为准。",
          "2. 主报告用株级 bootstrap CI（同时考虑菌株抽样和配对随机性）；只考虑配对随机性的区间偏窄。",
          "3. 菌落候选株少的节（见上表），随机配对的多样性有限，结果不确定性被低估。",
          "4. 权重敏感性只作补充材料；主要结果固定报告预设权重，不能按本表挑权重。",
          "5. 两边都是 OOF 预测：每个菌株只被训练时没见过它的模型预测过。"]
    txt = "\n".join(L)
    (out / "report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = sens_df[sens_df["colony_weight"].apply(lambda x: isinstance(x, float))]
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        x = d["colony_weight"].astype(float).values
        ax.plot(x, d["fused_acc_expected"], "o-", color="#2b6cb0", label="Fused (expected)")
        ax.fill_between(x, d["fused_acc_lo"], d["fused_acc_hi"], color="#2b6cb0", alpha=0.15)
        ax.axhline(km / n, color="#666", ls="--", label="Microscopy only")
        ax.axvline(args.weight, color="#c05621", ls=":", lw=1)
        ax.set_xlabel("Colony weight")
        ax.set_ylabel("Section-level top-1 accuracy")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "weight_sensitivity.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ 没画图：{e}")

    info = {"version": VERSION, "time": dt.datetime.now().isoformat(timespec="seconds"), "args": vars(args),
            "script_sha256": sha256(__file__),
            "inputs_sha256": {"micro_oof": sha256(chk["micro"]["micro_oof"]),
                              "colony_oof": sha256(chk["colony"]["colony_oof"]),
                              "section_map": sha256(smap_path)},
            "main": {k: main_s[k] for k in ("exp_acc", "acc_lo", "acc_hi", "diff", "diff_boot_lo",
                                            "diff_boot_hi", "p_med", "p_frac05")},
            "micro_acc": km / n, "n_eval": int(n)}
    (out / "RUN_INFO.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出目录：{out}")


if __name__ == "__main__":
    main()
