#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_clin_fusion.py —— 临床验证最后一步：曲霉"菌落为主、镜检为辅"融合，非曲霉用镜检，输出每株最终结果

不建议直接运行，请用 zsh clin_5_eval.zsh（参数已写死在里面）。

输入
  --micro   03_eval_clinical.py 的输出文件夹（读 per_isolate.csv，含 p::<26 类> 概率）
  --colony  03_colony_predict.py 的输出文件夹（读 per_strain.csv、run_info.json）；没有菌落时可省略
  --answers answers_master.csv（可省略；提供时计算指标）

预先设定的规则（全部写进 run_info.json；正式验证开始后不得再改）
  1. 分流：镜检概率中所有曲霉类之和 P_曲霉 ≥ --asp_min（默认 0.5）→ 曲霉通路；否则 → 非曲霉通路
  2. 曲霉通路：镜检"种"概率按组相加（Fumigati/Flavi/Nigri/Terrei/Nidulantes/Other），
     取最高者为"镜检首位组"。曲霉鉴定以菌落为必需证据：
     2a. 有可用菌落（菌落特征未超出训练范围）且镜检首位组属于菌落模型覆盖的组：
         只在菌落模型覆盖的 K 个组内融合，两边概率均在这 K 组内重新归一化，
         线性加权  s = (1 − α)·s_镜检 + α·s_菌落   （α = --alpha，默认 0.8，即菌落 80%）
         报告到"组"；组内镜检概率最高的种作为"种提示"
     2b. 有可用菌落，但镜检首位组不在菌落模型覆盖范围内（如 Nidulantes、Other）：
         不融合；报告"Aspergillus spp."，附镜检组提示与菌落结果，标记复核
     2c. 没有菌落，或菌落特征超出训练范围（D² 超限）：
         报告"Aspergillus spp."，附镜检组提示，标记复核并建议补拍菌落
  3. 非曲霉通路：直接用镜检结果（种）
  4. 需复核：报告到组/种时 top1 < --p_min 或 top1 − top2 < --gap_min；或 镜检首位组与菌落首位组不一致；
             或 菌落模型自身标记复核；或 镜检提示菌落模型未覆盖的组（概率 > 0.3）；2b、2c 一律复核
     需复核且涉及 Fumigati / Terrei / Flavi → 建议 50 °C 生长试验
  5. 敏感性分析（仅供讨论）：α = 0 / 0.5 / 0.7 / 0.8 / 0.9 / 1.0 时的组一级准确率

锁定（v1.3）
  --write_lock --lock clinical_lock.json --colony_model final_model.joblib ...
      在已有的 clinical_lock.json（03_eval_clinical.py 写的镜检部分）里加入 "fusion" 段：
      α、asp_min、p_min、gap_min、菌落模型 sha256、本脚本和 clin_sections.py 的 sha256
  --lock clinical_lock.json（正式评估）
      逐项核对上面的内容，任何一项不同就拒绝运行；PILOT 试运行不传 --lock

版本历史
  v1.1 (2026-10-21)：s ∝ s_镜检 × s_菌落^w，菌落未覆盖的组取中性值 1/K
  v1.2 (2026-10-22)：改为线性加权（菌落 α = 0.8）；取消中性值，未覆盖组不融合；缺菌落只报告到属
  v1.3 (2026-10-23)：α 等融合规则写进锁定文件并在正式评估时核对（--write_lock / --lock）
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from clin_sections import SECTIONS, cp_ci, fmt, is_aspergillus, read_csv_any, section_of  # noqa: E402,F401

VERSION = "03_clin_fusion v1.3 (2026-10-23)"
SEC_CN = {"Fumigati": "烟曲霉组", "Flavi": "黄曲霉组", "Nigri": "黑曲霉组", "Terrei": "土曲霉组",
          "Nidulantes": "构巢曲霉组", "Other": "其他曲霉组"}
FIFTY_C = {"Fumigati", "Terrei", "Flavi"}
ALPHA_GRID = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
UNCOVERED_HINT = 0.3          # 镜检在菌落未覆盖组上的概率超过此值 → 复核


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def truthy(v):
    return str(v).strip().lower() in {"true", "1", "yes"}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def norm(v):
    v = np.asarray(v, float)
    s = v.sum()
    return v / s if s > 0 else np.full(len(v), 1.0 / len(v))


def fuse(sm, sc, alpha):
    """线性加权：两边都已在菌落覆盖的 K 组内归一化。"""
    return (1 - alpha) * sm + alpha * sc


def fusion_lock_content(a):
    return {"alpha": a.alpha, "asp_min": a.asp_min, "p_min": a.p_min, "gap_min": a.gap_min,
            "uncovered_hint": UNCOVERED_HINT, "method": "linear pool within colony-covered sections",
            "script_sha256": sha256(os.path.abspath(__file__)),
            "clin_sections_sha256": sha256(os.path.join(BASE, "clin_sections.py"))}


def write_fusion_lock(a):
    if not a.lock or not os.path.exists(a.lock):
        sys.exit(f"找不到 {a.lock}：请先用 03_eval_clinical.py --write_lock 写出镜检部分（clin_lock.zsh 会按顺序做）")
    if not a.colony_model or not os.path.exists(a.colony_model):
        sys.exit(f"找不到菌落模型文件 {a.colony_model}")
    L = json.load(open(a.lock, encoding="utf-8"))
    if "fusion" in L:
        sys.exit(f"{a.lock} 里已经有 fusion 段，锁定只做一次；确需重新锁定请先把整个文件改名备份")
    c = fusion_lock_content(a)
    c["colony_model_sha256"] = sha256(a.colony_model)
    c["locked_at"] = dt.datetime.now().isoformat(timespec="seconds")
    L["fusion"] = c
    json.dump(L, open(a.lock, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"已把融合规则写进锁定文件 {a.lock}：α = {a.alpha}，ASP_MIN = {a.asp_min}，"
          f"p_min = {a.p_min}，gap_min = {a.gap_min}，菌落模型 sha256 {c['colony_model_sha256'][:16]}…")


def check_fusion_lock(a):
    if not os.path.exists(a.lock):
        sys.exit(f"找不到锁定文件 {a.lock}（正式模式必须先运行 zsh clin_lock.zsh）")
    L = json.load(open(a.lock, encoding="utf-8")).get("fusion")
    if L is None:
        sys.exit(f"{a.lock} 里没有 fusion 段：这是旧版 clin_lock.zsh 生成的锁定文件，α 等融合规则没有被锁定。\n"
                 "如果还没开始正式入组：把它改名备份，用新版 clin_lock.zsh 重新锁定。")
    now = fusion_lock_content(a)
    errs = []
    for k, lab in [("alpha", "α（ALPHA）"), ("asp_min", "ASP_MIN"), ("p_min", "P_MIN"), ("gap_min", "GAP_MIN"),
                   ("uncovered_hint", "UNCOVERED_HINT")]:
        if abs(float(L[k]) - float(now[k])) > 1e-9:
            errs.append(f"{lab} 与锁定时不同（锁定 {L[k]}，现在 {now[k]}）")
    if L["script_sha256"] != now["script_sha256"]:
        errs.append("03_clin_fusion.py 与锁定时不同（脚本被改过）")
    if L["clin_sections_sha256"] != now["clin_sections_sha256"]:
        errs.append("clin_sections.py 与锁定时不同（种→组对照表被改过）")
    if a.colony and os.path.isdir(a.colony):
        ri = json.load(open(os.path.join(a.colony, "run_info.json"), encoding="utf-8"))
        if ri.get("model_sha256") != L["colony_model_sha256"]:
            errs.append("菌落模型与锁定时不同（final_model.joblib 指纹不一致）")
    if errs:
        sys.exit("融合锁定核对未通过：\n  - " + "\n  - ".join(errs))
    print(f"融合锁定核对通过（α = {a.alpha}，锁定于 {L.get('locked_at', '?')}）")
    return L.get("locked_at", "")


def load_micro(d):
    p = os.path.join(d, "per_isolate.csv")
    if not os.path.exists(p):
        sys.exit(f"找不到镜检结果 {p}")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"isolate_id": str})
    df["isolate_id"] = df["isolate_id"].str.strip().str.upper()
    classes = [c[3:] for c in df.columns if c.startswith("p::")]
    return df.set_index("isolate_id"), classes, p


def load_colony(d):
    p = os.path.join(d, "per_strain.csv")
    ri = os.path.join(d, "run_info.json")
    if not os.path.exists(p) or not os.path.exists(ri):
        sys.exit(f"找不到菌落结果 {p} 或 {ri}")
    classes = json.load(open(ri, encoding="utf-8"))["classes"]
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"strain_id": str})
    df["strain_id"] = df["strain_id"].str.strip().str.upper()
    return df.set_index("strain_id"), classes, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", default="")
    ap.add_argument("--colony", default="")
    ap.add_argument("--answers", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--asp_min", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.8, help="菌落权重（线性加权），默认 0.8")
    ap.add_argument("--w_colony", type=float, default=None, help="v1.1 的参数，已停用（传入会被忽略）")
    ap.add_argument("--p_min", type=float, default=0.70)
    ap.add_argument("--gap_min", type=float, default=0.20)
    ap.add_argument("--lock", default="", help="clinical_lock.json；给出时核对 fusion 段（正式模式）")
    ap.add_argument("--write_lock", action="store_true", help="把融合规则写进 --lock 后退出（clin_lock.zsh 调用）")
    ap.add_argument("--colony_model", default="", help="--write_lock 用：菌落 final_model.joblib")
    a = ap.parse_args()
    if not 0.0 <= a.alpha <= 1.0:
        sys.exit(f"--alpha 必须在 0–1 之间（现在是 {a.alpha}）")
    if a.w_colony is not None:
        print(f"ℹ️  --w_colony {a.w_colony} 是 v1.1 的参数，v1.2 已改为线性加权 --alpha，本次忽略 --w_colony")
    if a.write_lock:
        write_fusion_lock(a)
        return
    if not a.out:
        sys.exit("缺少 --out")
    locked_at = check_fusion_lock(a) if a.lock else "未核对（试运行或自检）"
    os.makedirs(a.out, exist_ok=True)

    inputs = {}
    M, mcls = (None, [])
    if a.micro:
        M, mcls, f = load_micro(a.micro)
        inputs["micro_per_isolate"] = (f, sha256(f))
    C, ccls = (None, [])
    if a.colony and os.path.isdir(a.colony):
        C, ccls, f = load_colony(a.colony)
        inputs["colony_per_strain"] = (f, sha256(f))
    if M is None and C is None:
        sys.exit("镜检和菌落结果都没有，无法融合")
    unknown = [c for c in ccls if c not in SECTIONS]
    if unknown:
        print(f"⚠️  菌落模型含有对照表以外的组：{unknown}（照常参与融合）")

    asp_cols = [c for c in mcls if is_aspergillus(c)]
    all_secs = list(dict.fromkeys(SECTIONS + [c for c in ccls if c not in SECTIONS] + ["Other"]))
    uncovered = [s for s in all_secs if ccls and s not in ccls]
    if ccls:
        print(f"ℹ️  菌落模型覆盖 {', '.join(ccls)}；镜检首位组为 {', '.join(uncovered)} 时不融合，只报告到属并复核")
    sec_of_cls = {c: section_of(c) for c in asp_cols}

    ids = sorted(set(M.index if M is not None else []) | set(C.index if C is not None else []))
    rows, sens = [], {}          # sens：融合株的 (镜检 K 组概率, 菌落 K 组概率)，供 α 敏感性分析
    for iso in ids:
        r = {"isolate_id": iso}
        reasons, actions = [], []
        has_m = M is not None and iso in M.index
        has_c = C is not None and iso in C.index
        r["有镜检结果"], r["有菌落结果"] = has_m, has_c

        sec_raw, pm = {}, None
        if has_m:
            pm = M.loc[iso, [f"p::{c}" for c in mcls]].astype(float).values
            p_asp = float(sum(pm[mcls.index(c)] for c in asp_cols))
            o = np.argsort(-pm)
            r.update({"镜检_top1": mcls[o[0]], "镜检_p1": round(float(pm[o[0]]), 4),
                      "镜检_top2": mcls[o[1]], "镜检_p2": round(float(pm[o[1]]), 4),
                      "镜检_ROI数": M.loc[iso].get("n_roi", ""), "P_曲霉": round(p_asp, 4)})
            sec_raw = {s: 0.0 for s in all_secs}
            for c in asp_cols:
                sec_raw[sec_of_cls[c]] = sec_raw.get(sec_of_cls[c], 0.0) + float(pm[mcls.index(c)])
            msec_top = max(sec_raw, key=sec_raw.get) if p_asp > 0 else ""
            r["镜检_组"] = msec_top                      # 仅凭镜检的组（配对比较用）
            r["镜检_组_归一"] = msec_top                  # 与 v1.1 列名兼容
            r["镜检_组概率"] = round(sec_raw.get(msec_top, 0.0) / p_asp, 4) if p_asp > 0 else 0.0
        else:
            p_asp, msec_top = float("nan"), ""

        col_usable = False
        if has_c:
            pc = np.array([float(C.loc[iso, f"p_{c}"]) for c in ccls])
            o = np.argsort(-pc)
            r.update({"菌落_top1": ccls[o[0]], "菌落_p1": round(float(pc[o[0]]), 4),
                      "菌落_top2": ccls[o[1]], "菌落_p2": round(float(pc[o[1]]), 4),
                      "菌落_需复核": truthy(C.loc[iso].get("需复核", "")),
                      "菌落_超出训练范围": truthy(C.loc[iso].get("超出训练范围", ""))})
            col_usable = not r["菌落_超出训练范围"]

        def hint_in(sec):
            in_sec = [c for c in asp_cols if sec_of_cls[c] == sec]
            return max(in_sec, key=lambda c: pm[mcls.index(c)]) if in_sec and pm is not None else ""

        # ------------------------------------------------ 分流与判读
        check_thr = True          # 报告到组/种时才按 p_min / gap_min 判断
        if has_m and p_asp >= a.asp_min:
            if col_usable and msec_top in ccls:
                # 2a：在菌落覆盖的 K 组内线性加权融合
                route = "曲霉：菌落+镜检融合"
                sm = norm([sec_raw.get(s, 0.0) for s in ccls])
                sc = norm(pc)
                fz = fuse(sm, sc, a.alpha)
                sens[iso] = (sm, sc)
                o = np.argsort(-fz)
                top, p1, sec2, p2 = ccls[o[0]], float(fz[o[0]]), ccls[o[1]], float(fz[o[1]])
                for s, v in zip(ccls, fz):
                    r[f"融合_{s}"] = round(float(v), 4)
                if msec_top != r["菌落_top1"]:
                    reasons.append(f"镜检({msec_top})与菌落({r['菌落_top1']})不一致")
                if r["菌落_需复核"]:
                    reasons.append("菌落模型自身标记需复核")
                unc = sum(sec_raw.get(s, 0.0) for s in uncovered) / p_asp
                if unc > UNCOVERED_HINT:
                    reasons.append(f"镜检提示菌落模型未覆盖的组（{'/'.join(uncovered)} 合计 {unc:.2f}）")
                hint = hint_in(top)
                r.update({"最终_层级": "组", "最终_组": top, "最终_组提示": top, "最终_种提示": hint,
                          "最终_p1": round(p1, 4), "最终_第2": sec2, "最终_p2": round(p2, 4)})
                r["最终报告"] = (f"Aspergillus sect. {top}（{SEC_CN.get(top, top)}）{p1:.2f}"
                              + (f"；种提示 {hint}" if hint else ""))
                pair = {top, sec2}
            else:
                # 2b / 2c：不融合，只报告到属
                check_thr = False
                hint = hint_in(msec_top)
                p_sec = sec_raw.get(msec_top, 0.0) / p_asp if p_asp > 0 else 0.0
                tail = f"镜检提示 sect. {msec_top} {p_sec:.2f}"
                if col_usable:
                    route = "曲霉：镜检首位组菌落未覆盖"
                    tail += f"；菌落 {r['菌落_top1']} {r['菌落_p1']:.2f}"
                    reasons.append(f"镜检首位为 {msec_top} 组，菌落模型未覆盖该组，不融合")
                    pair = {msec_top, r["菌落_top1"]}
                elif has_c:
                    route = "曲霉：菌落超出训练范围"
                    reasons.append("菌落特征超出训练范围，菌落结果不采用")
                    actions.append("检查菌落照片（培养条件、拍摄）后重拍")
                    pair = {msec_top}
                else:
                    route = "曲霉：缺菌落"
                    reasons.append("曲霉缺菌落照片")
                    actions.append("补拍菌落（SDA 28 °C 第 3 天正面）")
                    pair = {msec_top}
                r.update({"最终_层级": "属", "最终_组": "", "最终_组提示": msec_top, "最终_种提示": hint,
                          "最终_p1": round(p_asp, 4), "最终_第2": "", "最终_p2": np.nan})
                r["最终报告"] = f"Aspergillus spp.（{tail}）"
                p1, p2 = p_asp, 0.0
        elif has_m:
            route = "非曲霉：镜检"
            r.update({"最终_层级": "种", "最终_组": "", "最终_组提示": "", "最终_种提示": r["镜检_top1"],
                      "最终_p1": r["镜检_p1"], "最终_第2": r["镜检_top2"], "最终_p2": r["镜检_p2"]})
            r["最终报告"] = f"{r['镜检_top1']} {r['镜检_p1']:.2f}"
            p1, p2, pair = r["镜检_p1"], r["镜检_p2"], set()
            if has_c:
                reasons.append(f"有菌落照片但镜检判为非曲霉（P_曲霉 {p_asp:.2f}），菌落结果 {r['菌落_top1']} 仅供参考")
        else:
            route = "仅菌落（无镜检 ROI）"
            top, p1, sec2, p2 = r["菌落_top1"], r["菌落_p1"], r["菌落_top2"], r["菌落_p2"]
            r.update({"最终_层级": "组", "最终_组": top, "最终_组提示": top, "最终_种提示": "",
                      "最终_p1": p1, "最终_第2": sec2, "最终_p2": p2})
            r["最终报告"] = f"Aspergillus sect. {top}（{SEC_CN.get(top, top)}）{p1:.2f}（仅菌落）"
            reasons.append("没有镜检 ROI")
            pair = {top, sec2}

        if check_thr:
            if p1 < a.p_min:
                reasons.append(f"top1 {p1:.2f} < {a.p_min}")
            if p1 - p2 < a.gap_min:
                reasons.append(f"top1−top2 {p1 - p2:.2f} < {a.gap_min}")
        flag = bool(reasons and not (len(reasons) == 1 and reasons[0].startswith("有菌落照片但")))
        if flag:
            actions.append("人工复核 / MALDI-TOF / 测序")
            if pair & FIFTY_C:
                actions.append("50 °C 生长试验")
        r.update({"通路": route, "需复核": flag, "复核原因": "；".join(reasons),
                  "建议操作": "；".join(dict.fromkeys(actions))})
        rows.append(r)

    D = pd.DataFrame(rows)
    front = ["isolate_id", "通路", "最终报告", "需复核", "复核原因", "建议操作", "最终_层级", "最终_组", "最终_组提示",
             "最终_种提示", "最终_p1", "最终_第2", "最终_p2", "P_曲霉", "镜检_top1", "镜检_p1", "镜检_top2", "镜检_p2",
             "镜检_组", "镜检_组概率", "菌落_top1", "菌落_p1", "菌落_top2", "菌落_p2", "菌落_需复核", "菌落_超出训练范围"]
    front = [c for c in front if c in D.columns]
    D = D[front + [c for c in D.columns if c not in front]]

    # ------------------------------------------------ 揭盲后指标
    rule = (f"规则：P_曲霉 ≥ {a.asp_min} 走曲霉通路；曲霉在菌落覆盖的 {len(ccls)} 组"
            f"（{'/'.join(ccls)}）内线性加权 s = {1 - a.alpha:.1f}·镜检 + {a.alpha:.1f}·菌落；"
            f"镜检首位组为 {'/'.join(uncovered)}、缺菌落或菌落超出训练范围 → 只报告到属并复核；"
            if ccls else f"规则：P_曲霉 ≥ {a.asp_min} 走曲霉通路；无菌落结果 → 曲霉只报告到属并复核；")
    L = ["===== 临床验证：最终结果（曲霉：菌落为主 + 镜检；非曲霉：镜检） =====",
         f"运行时间 {dt.datetime.now():%Y-%m-%d %H:%M}；{VERSION}；融合锁定 {locked_at}",
         rule + f"报告到组/种时 top1 < {a.p_min} 或 top1−top2 < {a.gap_min} → 需复核",
         f"株数 {len(D)}：" + "，".join(f"{k} {v}" for k, v in D["通路"].value_counts().items()),
         f"需复核 {int(D['需复核'].sum())}/{len(D)}"]
    if a.answers and os.path.exists(a.answers):
        ans = read_csv_any(a.answers)
        ans["isolate_id"] = ans["isolate_id"].str.strip().str.upper()
        ans = ans.set_index("isolate_id")
        D["参考鉴定"] = D["isolate_id"].map(ans["reference"]).fillna("")
        D["参考_train_class"] = D["isolate_id"].map(ans["train_class"]).fillna("")
        sec_col = (D["isolate_id"].map(ans["section"]).fillna("") if "section" in ans
                   else pd.Series("", index=D.index))
        D["参考_组"] = [s.strip() or section_of(ref) for s, ref in zip(sec_col, D["参考鉴定"])]
        miss = D[D["参考鉴定"].str.strip() == ""]["isolate_id"].tolist()
        if miss:
            L.append(f"⚠ 答案表缺 {len(miss)} 株，不计入指标：{', '.join(miss[:10])}")
        E = D[D["参考鉴定"].str.strip() != ""].copy()
        asp5 = E["参考_组"].isin(SECTIONS)
        ref_asp = E["参考鉴定"].map(is_aspergillus)

        def final_ok(r):
            # 曲霉：必须报告到组且组正确（只报告到属 = 未能鉴定到组，计为不正确）
            if r["参考_组"] in SECTIONS:
                return r["最终_层级"] == "组" and r["最终_组"] == r["参考_组"]
            if r["参考_train_class"]:
                return r["最终_种提示"] == r["参考_train_class"] and r["最终_层级"] == "种" \
                    or (r["参考_组"] == "Other" and r["最终_种提示"] == r["参考_train_class"])
            return np.nan
        E["最终正确"] = E.apply(final_ok, axis=1)
        inpanel = E[E["最终正确"].notna()]
        k = int(inpanel["最终正确"].astype(bool).sum())
        L.append("\n----- 主要结果（集内菌株；曲霉按组、其他按种；曲霉只报告到属计为不正确） -----")
        L.append(f"最终准确率           {fmt(k, len(inpanel))}")
        unf = inpanel[~inpanel["需复核"]]
        L.append(f"复核标记比例         {fmt(int(inpanel['需复核'].sum()), len(inpanel))}")
        L.append(f"未标记（可直接报告）准确率 {fmt(int(unf['最终正确'].astype(bool).sum()), len(unf))}")

        Ea = E[asp5]
        if len(Ea):
            to_grp = Ea[Ea["最终_层级"] == "组"]
            L.append("\n----- 曲霉（参考为 5 组之一） -----")
            L.append(f"报告到组的比例       {fmt(len(to_grp), len(Ea))}"
                     f"   （其余只报告到属：{'，'.join(f'{k} {v}' for k, v in Ea[Ea['最终_层级'] != '组']['通路'].value_counts().items()) or '无'}）")
            L.append(f"报告到组者的组准确率 {fmt(int((to_grp['最终_组'] == to_grp['参考_组']).sum()), len(to_grp))}")
            L.append(f"属一级（Aspergillus）正确 {fmt(int(Ea['通路'].str.startswith('曲霉').sum()), len(Ea))}")
            gen = Ea[Ea["最终_层级"] == "属"]
            if len(gen):
                L.append(f"只报告到属者，镜检组提示与参考一致 {fmt(int((gen['最终_组提示'] == gen['参考_组']).sum()), len(gen))}"
                         "（仅供参考，未作为报告结果）")

        # 分流
        routed = E["通路"].str.startswith("曲霉")
        hm = E["有镜检结果"].astype(bool)
        L.append("\n----- 分流（曲霉 vs 非曲霉；仅计有镜检结果的株） -----")
        L.append(f"曲霉株被分到曲霉通路   {fmt(int((ref_asp & routed & hm).sum()), int((ref_asp & hm).sum()))}")
        L.append(f"非曲霉株被分到非曲霉通路 {fmt(int((~ref_asp & ~routed & hm).sum()), int((~ref_asp & hm).sum()))}")

        # 曲霉：镜检 vs 菌落 vs 融合（同时有镜检和可用菌落的株，配对比较）
        both = E[asp5 & E["通路"].isin(["曲霉：菌落+镜检融合", "曲霉：镜检首位组菌落未覆盖"])]
        L.append(f"\n----- 曲霉 5 组：镜检 / 菌落 / 最终（同时有镜检和可用菌落的 {len(both)} 株，配对比较） -----")
        if len(both):
            okm = both["镜检_组"] == both["参考_组"]
            okc = both["菌落_top1"] == both["参考_组"]
            okf = (both["最终_层级"] == "组") & (both["最终_组"] == both["参考_组"])
            L.append(f"仅镜检（组）         {fmt(int(okm.sum()), len(both))}")
            L.append(f"仅菌落（组）         {fmt(int(okc.sum()), len(both))}")
            L.append(f"最终（组）           {fmt(int(okf.sum()), len(both))}")
            b, c = int((okf & ~okm).sum()), int((~okf & okm).sum())
            L.append(f"最终 vs 仅镜检：纠正 {b} 株，改错 {c} 株，McNemar 精确 P = "
                     + ("—（没有结果不一致的株）" if b + c == 0 else f"{mcnemar_exact(b, c):.3f}"))
            b, c = int((okf & ~okc).sum()), int((~okf & okc).sum())
            L.append(f"最终 vs 仅菌落：纠正 {b} 株，改错 {c} 株，McNemar 精确 P = "
                     + ("—（没有结果不一致的株）" if b + c == 0 else f"{mcnemar_exact(b, c):.3f}"))
            if uncovered:
                nm = both[both["参考_组"].isin(uncovered)]
                if len(nm):
                    L.append(f"注：其中 {len(nm)} 株参考为 {'/'.join(uncovered)}（菌落模型未覆盖），'仅菌落'对这些株必然判错")
            sp = both[(both["最终_层级"] == "组") & (both["最终_种提示"] != "")]
            ks = int((sp["最终_种提示"] == sp["参考_train_class"]).sum())
            L.append(f"种提示与参考一致     {fmt(ks, len(sp))}（仅供参考，曲霉正式报告到组）")

            # α 敏感性分析：只改变融合株的结果，其余株（只报告到属）不随 α 变化
            L.append("\n----- 菌落权重 α 的敏感性分析（组一级，上面同一批株；仅供讨论，主要结果以预设 α 为准） -----")
            for al in sorted(set(ALPHA_GRID + [a.alpha])):
                kk = 0
                for _, rr in both.iterrows():
                    if rr["isolate_id"] in sens:
                        sm, sc = sens[rr["isolate_id"]]
                        kk += int(ccls[int(np.argmax(fuse(sm, sc, al)))] == rr["参考_组"])
                lab = {0.0: "（=只看镜检，限菌落覆盖组）", 1.0: "（=只看菌落）"}.get(al, "")
                star = "  ← 预设" if abs(al - a.alpha) < 1e-9 else ""
                L.append(f"  α = {al:.1f}{lab:<18s}{fmt(kk, len(both))}{star}")

        nonasp = E[~ref_asp & (E["参考_train_class"] != "")]
        if len(nonasp):
            L.append(f"\n----- 非曲霉（集内，按种） -----")
            L.append(f"种一级准确率         {fmt(int((nonasp['最终_种提示'] == nonasp['参考_train_class']).sum()), len(nonasp))}")
        oos = E[E["最终正确"].isna()]
        if len(oos):
            L.append(f"\n----- 集外菌种 {len(oos)} 株（应被标记复核） -----")
            L.append(f"被标记复核           {fmt(int(oos['需复核'].sum()), len(oos))}")

        wrong = inpanel[~inpanel["最终正确"].astype(bool)]
        if len(wrong):
            L.append("\n----- 未正确鉴定的菌株（含只报告到属） -----")
            for _, r in wrong.iterrows():
                L.append(f"  {r['isolate_id']:<14}{r['参考鉴定']:<28}→ {r['最终报告']}"
                         f"  {'已标记' if r['需复核'] else '未标记 ⚠'}  [{r['通路']}]")
        D = D.merge(E[["isolate_id", "最终正确"]], on="isolate_id", how="left")
    else:
        L.append("\n（未提供答案表：只输出预测，未计算指标）")

    D.to_csv(os.path.join(a.out, "per_isolate_final.csv"), index=False, encoding="utf-8-sig")
    txt = "\n".join(L)
    open(os.path.join(a.out, "report.txt"), "w", encoding="utf-8").write(txt + "\n")
    info = {"run_at": dt.datetime.now().isoformat(timespec="seconds"), "version": VERSION,
            "script_sha256": sha256(os.path.abspath(__file__)),
            "clin_sections_sha256": sha256(os.path.join(BASE, "clin_sections.py")),
            "fusion": {"method": "linear pool within colony-covered sections", "alpha_colony": a.alpha,
                       "uncovered_sections": uncovered, "uncovered_rule": "no fusion; report genus; flag",
                       "missing_colony_rule": "report genus; flag"},
            "fusion_lock": locked_at, "args": vars(a), "inputs": {k: {"path": v[0], "sha256": v[1]} for k, v in inputs.items()},
            "micro_classes": mcls, "colony_classes": ccls, "aspergillus_class_to_section": sec_of_cls}
    if a.answers and os.path.exists(a.answers):
        info["answers_sha256"] = sha256(a.answers)
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(txt)
    print("\n每株最终结果（前 40 株）：")
    show = [c for c in ["isolate_id", "通路", "最终报告", "需复核", "最终正确"] if c in D.columns]
    print(D[show].head(40).to_string(index=False))
    print(f"\n已写出 {a.out}/per_isolate_final.csv、report.txt、run_info.json")


if __name__ == "__main__":
    main()
