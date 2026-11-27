#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_review_sites.py —— 把指定来源（默认 TP 谭盼、HYY 胡音音）的外部 ROI 挑出来，生成一个人工核对网页

每张外部原图一张卡片：
  · 原图缩略图，上面画出截取的 ROI 框（有交叉验证结果时：绿框 = 判对，红框 = 判错）
  · 各个 ROI 截图，附模型预测和置信度（需 --run）
  · 对照行 1：本院（PUPH）训练截图中【标注菌种】的随机样例
  · 对照行 2：本院训练截图中【模型最常预测的菌种】的随机样例（判错时才显示）
  · 核对选项：标签正确 / 标签错误 / 图片不合格 / 不确定，及备注（可填正确菌名）
网页右上角"导出 CSV"保存核对结果；勾选会自动记在浏览器里，关掉再开不会丢（同一浏览器）。

用法（在脚本所在文件夹、fungal 环境里）：
    python 04_review_sites.py --run runs/<含 ROI 的交叉验证目录>
    python 04_review_sites.py --sites TP HYY MQQ --run runs/<目录>       # 换来源
    python 04_review_sites.py                                             # 不带 --run：只看标注，不显示预测
输出：review_sites/index.html（双击用浏览器打开）和 review_sites/img/ 缩略图
"""
import os
import sys
import json
import html
import argparse
import hashlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

ROI_LOG = "/Users/chunjiang/Documents/真菌AI/CNN/roi_test/crop_log.csv"
SITE_NAME = {"TP": "谭盼", "HYY": "胡音音", "MQQ": "马琼琼", "LYF": "李一凡", "LY": "李艳", "TYQ": "田蕴琦",
             "WEB1": "网络图1", "WEB2": "网络图2", "MCW": "麦老师", "HSJ": "HSJ", "JYJ": "蒋宇杰", "PUPH": "本院"}


def short(c):
    """Aspergillus fumigatus (sect. Fumigati) → A. fumigatus"""
    if not isinstance(c, str) or not c:
        return ""
    c = c.split(" (")[0]
    w = c.split()
    return f"{w[0][0]}. {' '.join(w[1:])}" if len(w) > 1 and w[0] != "Sporothrix" else c


def corners(cx, cy, box, angle):
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    d = box / 2
    off = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
    return off @ R.T + np.array([cx, cy])


class Thumbs:
    def __init__(self, out):
        self.dir = os.path.join(out, "img")
        os.makedirs(self.dir, exist_ok=True)

    def name(self, key, size):
        return hashlib.md5(f"{key}|{size}".encode()).hexdigest()[:16] + ".jpg"

    def make(self, path, size):
        fn = self.name(path, size)
        dst = os.path.join(self.dir, fn)
        if not os.path.exists(dst):
            try:
                im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                return ""
            im.thumbnail((size, size))
            im.save(dst, quality=85)
        return "img/" + fn

    def original(self, path, boxes, size=720):
        """原图缩略图 + ROI 框；boxes = [(cx, cy, w, angle, color, label)]"""
        fn = self.name(path + json.dumps(boxes), size)
        dst = os.path.join(self.dir, fn)
        if not os.path.exists(dst):
            try:
                im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception:
                return ""
            sc = min(1.0, size / max(im.size))
            im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))))
            d = ImageDraw.Draw(im)
            lw = max(2, int(3 * size / 720))
            for cx, cy, w, ang, color, label in boxes:
                pts = [tuple(p) for p in (corners(cx, cy, w, ang) * sc)]
                d.line(pts + [pts[0]], fill=color, width=lw)
                x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
                d.rectangle([x0, y0, x0 + 16 + 7 * len(label), y0 + 16], fill=color)
                d.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255))
            im.save(dst, quality=85)
        return "img/" + fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=["TP", "HYY"])
    ap.add_argument("--trainset", default="trainset_roi")
    ap.add_argument("--run", default="", help="含 ROI 的交叉验证输出目录（读 oof_predictions.csv）")
    ap.add_argument("--roi_log", default=ROI_LOG)
    ap.add_argument("--n_ref", type=int, default=8, help="每行对照的本院训练截图张数")
    ap.add_argument("--out", default="review_sites", help="只写名字时放进 output/04_audit_review/")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.trainset = OP.find_trainset_csv(a.trainset)
    a.out = OP.place(a.out, OP.AUDIT)
    if a.run:
        a.run = OP.find_run(a.run)

    ts = pd.read_csv(a.trainset, encoding="utf-8-sig", dtype={"cv_group": str})
    if "site" not in ts:
        sys.exit("训练清单里没有 site 列，请用新版 01_build_trainset.py（--add_roi）重新生成")
    roi = ts[(ts["source"] == "roi_ext") & ts["site"].isin(a.sites)].copy()
    if roi.empty:
        sys.exit(f"训练清单里没有来源为 {a.sites} 的外部 ROI")
    lg = pd.read_csv(a.roi_log, encoding="utf-8-sig", dtype=str).fillna("")
    lg = lg[lg["status"] == "crop"].drop_duplicates("crop_file").set_index("crop_file")
    roi = roi[roi["path"].isin(lg.index)].copy()
    roi["source_image"] = roi["path"].map(lg["source_image"])
    for c in ("cx", "cy", "w", "angle"):
        roi[c] = pd.to_numeric(roi["path"].map(lg[c]), errors="coerce").fillna(0)

    has_pred = bool(a.run)
    if has_pred:
        oof = pd.read_csv(os.path.join(a.run, "oof_predictions.csv"), encoding="utf-8-sig")
        oof = oof.drop_duplicates("path").set_index("path")
        roi["pred"] = roi["path"].map(oof["pred_class"])
        roi["p"] = roi["path"].map(oof["p_top1"])
        miss = roi["pred"].isna().sum()
        if miss:
            print(f"注意：{miss} 个 ROI 在 oof_predictions.csv 里找不到（可能不是同一训练清单跑的）")
    else:
        roi["pred"], roi["p"] = np.nan, np.nan

    puph = ts[ts["site"] == "PUPH"]
    rng = np.random.RandomState(a.seed)

    _exist = {}

    def pool(df_, key):
        if key not in _exist:
            _exist[key] = df_[df_["path"].map(os.path.exists)]
        return _exist[key]

    def refs(cls):
        d = pool(puph[puph["train_class"] == cls], ("PUPH", cls))
        src = "本院"
        if d.empty:
            d = pool(ts[(ts["train_class"] == cls) & ~ts["site"].isin(a.sites)], ("other", cls))
            src = "其他来源"
        if d.empty:
            return [], src
        idx = rng.choice(len(d), min(a.n_ref, len(d)), replace=False)
        return list(d.iloc[idx]["path"]), src

    th = Thumbs(a.out)
    cards = []
    order = roi.groupby("source_image").agg(site=("site", "first"), cls=("train_class", "first")).reset_index()
    order = order.sort_values(["site", "cls", "source_image"])
    print(f"来源 {a.sites}：{len(order)} 张原图，{len(roi)} 个 ROI，正在生成缩略图…")
    for k, r in enumerate(order.itertuples(), 1):
        d = roi[roi["source_image"] == r.source_image].reset_index(drop=True)
        truth = d["train_class"].iloc[0]
        sp_cn = d["species_cn"].iloc[0]
        boxes, rois = [], []
        for j, x in d.iterrows():
            ok = (x["pred"] == truth) if has_pred and isinstance(x["pred"], str) else None
            color = (40, 160, 70) if ok else ((205, 50, 40) if ok is False else (230, 160, 0))
            boxes.append((float(x["cx"]), float(x["cy"]), float(x["w"]), float(x["angle"]), color, str(j + 1)))
            rois.append(dict(src=th.make(x["path"], 240), n=j + 1, ok=ok, box=int(x["w"]),
                             pred=short(x["pred"]) if isinstance(x["pred"], str) else "",
                             p=f"{x['p']:.2f}" if pd.notna(x["p"]) else ""))
        n_ok = sum(1 for z in rois if z["ok"])
        wrong = d.loc[d["pred"].notna() & (d["pred"] != truth), "pred"]
        top_wrong = wrong.value_counts().index[0] if len(wrong) else ""
        r1, s1 = refs(truth)
        r2, s2 = refs(top_wrong) if top_wrong else ([], "")
        rel = os.path.relpath(r.source_image, os.path.dirname(os.path.dirname(r.source_image)))
        cards.append(dict(
            id=hashlib.md5(r.source_image.encode()).hexdigest()[:10], site=r.site, rel=rel, full=r.source_image,
            sp_cn=sp_cn, truth=truth, n=len(d), n_ok=n_ok if has_pred else None,
            orig=th.original(r.source_image, boxes), rois=rois,
            ref1=[th.make(p, 200) for p in r1], ref1_src=s1,
            ref2=[th.make(p, 200) for p in r2], ref2_src=s2, ref2_cls=top_wrong,
        ))
        if k % 10 == 0:
            print(f"  {k}/{len(order)}")

    page = build_page(cards, a.sites, has_pred)
    path = os.path.join(a.out, "index.html")
    open(path, "w", encoding="utf-8").write(page)
    print(f"\n已生成 {path}（双击用浏览器打开）")


def build_page(cards, sites, has_pred):
    e = html.escape
    sec = []
    for c in cards:
        acc = f'<span class="acc {"bad" if c["n_ok"] == 0 else ""}">ROI 判对 {c["n_ok"]}/{c["n"]}</span>' \
            if c["n_ok"] is not None else f'<span class="acc">ROI {c["n"]} 个</span>'
        rois = "".join(
            f'<figure class="{"ok" if z["ok"] else ("no" if z["ok"] is False else "")}"><img loading="lazy" src="{z["src"]}">'
            f'<figcaption>#{z["n"]} · {z["box"]}px{("<br><i>" + e(z["pred"]) + "</i> " + z["p"]) if z["pred"] else ""}</figcaption></figure>'
            for z in c["rois"])
        ref1 = "".join(f'<img loading="lazy" src="{s}">' for s in c["ref1"]) or '<span class="muted">训练集中没有该类</span>'
        ref2 = ""
        if c["ref2_cls"]:
            ref2 = (f'<div class="refrow"><div class="lab">{e(c["ref2_src"])}训练截图：模型多判为 <i>{e(c["ref2_cls"])}</i></div>'
                    f'<div class="refs">{"".join(f"<img loading=lazy src={s}>" for s in c["ref2"])}</div></div>')
        sec.append(f"""
<section class="card" data-id="{c['id']}" data-site="{e(c['site'])}" data-wrong="{1 if (c['n_ok'] is not None and c['n_ok'] < c['n'] / 2) else 0}">
  <header><span class="site">{e(SITE_NAME.get(c['site'], c['site']))}</span>
    <span class="fn" title="{e(c['full'])}">{e(c['rel'])}</span>
    <span class="lbl">标注：{e(c['sp_cn'])} → <i>{e(c['truth'])}</i></span>{acc}</header>
  <div class="body">
    <a class="orig" href="file://{e(c['full'])}" target="_blank" title="点击打开原图"><img loading="lazy" src="{c['orig']}"></a>
    <div class="right">
      <div class="lab">外部 ROI（绿 = 判对，红 = 判错；数字对应原图上的框）</div>
      <div class="rois">{rois}</div>
      <div class="refrow"><div class="lab">{e(c['ref1_src'])}训练截图：标注菌种 <i>{e(c['truth'])}</i></div><div class="refs">{ref1}</div></div>
      {ref2}
      <div class="review">
        <label><input type="radio" name="d_{c['id']}" value="标签正确">标签正确</label>
        <label><input type="radio" name="d_{c['id']}" value="标签错误">标签错误</label>
        <label><input type="radio" name="d_{c['id']}" value="图片不合格">图片不合格（非纯培养/模糊/倍数异常）</label>
        <label><input type="radio" name="d_{c['id']}" value="不确定">不确定</label>
        <input type="text" class="note" placeholder="备注 / 正确菌名">
      </div>
    </div>
  </div>
</section>""")
    btns = "".join(f'<button class="f" data-site="{e(s)}">{e(SITE_NAME.get(s, s))}</button>' for s in sites)
    meta = json.dumps([{"id": c["id"], "site": c["site"], "file": c["rel"], "full": c["full"],
                        "label_cn": c["sp_cn"], "train_class": c["truth"], "n_roi": c["n"], "n_ok": c["n_ok"]}
                       for c in cards], ensure_ascii=False)
    return PAGE.replace("{{SECTIONS}}", "".join(sec)).replace("{{BTNS}}", btns) \
        .replace("{{META}}", meta).replace("{{N}}", str(len(cards))) \
        .replace("{{TITLE}}", "、".join(SITE_NAME.get(s, s) for s in sites)) \
        .replace("{{WRONGBTN}}", '<button class="f" data-wrong="1">只看多数判错</button>' if has_pred else "")


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>外部 ROI 核对</title>
<style>
:root{--bg:#F4F4F1;--panel:#fff;--line:#DDDFD9;--fg:#1D2220;--muted:#66706B;--ok:#2E9A4E;--no:#C8372D;--acc:#2A3F9E}
@media (prefers-color-scheme:dark){:root{--bg:#141716;--panel:#1C201F;--line:#2F3533;--fg:#E5E9E6;--muted:#98A29D;--ok:#5CC57A;--no:#F07D72;--acc:#93A6FF;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"PingFang SC",sans-serif}
.top{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.top h1{font-size:16px;margin:0 12px 0 0}.top .sp{flex:1}
button{font:13px inherit;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--fg);color:var(--bg)}
#prog{font-size:12px;color:var(--muted)}
main{max-width:1400px;margin:0 auto;padding:12px 16px 60px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:14px 0;overflow:hidden}
.card.done{border-left:4px solid var(--acc)}
.card header{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;padding:8px 12px;border-bottom:1px solid var(--line)}
.site{font-weight:700}.fn{font:12px ui-monospace,Menlo,monospace;color:var(--muted)}
.lbl i,.lab i,figcaption i{font-family:Georgia,serif}
.acc{margin-left:auto;font-size:12px;font-weight:600}.acc.bad{color:var(--no)}
.body{display:grid;grid-template-columns:minmax(260px,420px) 1fr;gap:12px;padding:12px}
@media (max-width:800px){.body{grid-template-columns:1fr}}
.orig img{width:100%;border-radius:4px;display:block}
.lab{font-size:12px;color:var(--muted);margin:2px 0 4px}
.rois{display:flex;flex-wrap:wrap;gap:6px}
figure{margin:0;width:150px;border:3px solid #E0A000;border-radius:4px;overflow:hidden;background:var(--bg)}
figure.ok{border-color:var(--ok)}figure.no{border-color:var(--no)}
figure img{width:100%;aspect-ratio:1;object-fit:cover;display:block}
figcaption{font-size:11px;padding:2px 4px;line-height:1.3}
.refrow{margin-top:10px}.refs{display:flex;flex-wrap:wrap;gap:4px}
.refs img{width:96px;height:96px;object-fit:cover;border-radius:3px;border:1px solid var(--line)}
.review{margin-top:12px;padding-top:8px;border-top:1px dashed var(--line);display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center}
.review label{cursor:pointer;font-size:13px}.note{flex:1 1 200px;font:13px inherit;padding:4px 8px;border:1px solid var(--line);border-radius:5px;background:var(--bg);color:var(--fg)}
.muted{color:var(--muted);font-size:12px}
</style></head><body>
<div class="top"><h1>外部 ROI 核对 · {{TITLE}}</h1>
<button class="f on" data-site="">全部</button>{{BTNS}}{{WRONGBTN}}<button class="f" data-todo="1">只看未核对</button>
<span class="sp"></span><span id="prog"></span><button id="exp">导出 CSV</button></div>
<main>{{SECTIONS}}</main>
<script>
const META={{META}}, KEY="review_sites_v1";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}; prog()};
function prog(){const n=Object.values(st).filter(v=>v.d).length;document.getElementById("prog").textContent=`已核对 ${n} / {{N}}`;
  document.querySelectorAll(".card").forEach(c=>c.classList.toggle("done",!!(st[c.dataset.id]||{}).d))}
document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.id,s=st[id]||{};
  c.querySelectorAll("input[type=radio]").forEach(r=>{if(r.value===s.d)r.checked=true;
    r.onchange=()=>{st[id]={...(st[id]||{}),d:r.value};save()}});
  const t=c.querySelector(".note");t.value=s.n||"";t.oninput=()=>{st[id]={...(st[id]||{}),n:t.value};save()}});
let F={};document.querySelectorAll("button.f").forEach(b=>b.onclick=()=>{
  if(b.dataset.site!==undefined){F.site=b.dataset.site;document.querySelectorAll("button.f[data-site]").forEach(x=>x.classList.toggle("on",x===b))}
  else if(b.dataset.wrong){F.wrong=!F.wrong;b.classList.toggle("on",F.wrong)}
  else{F.todo=!F.todo;b.classList.toggle("on",F.todo)}
  document.querySelectorAll(".card").forEach(c=>{const s=st[c.dataset.id]||{};
    c.hidden=(F.site&&c.dataset.site!==F.site)||(F.wrong&&c.dataset.wrong!=="1")||(F.todo&&s.d)})});
document.getElementById("exp").onclick=()=>{
  const q=v=>`"${String(v??"").replace(/"/g,'""')}"`;
  const rows=[["site","file","label_cn","train_class","n_roi","n_roi_correct","decision","note","full_path"]]
    .concat(META.map(m=>{const s=st[m.id]||{};return [m.site,m.file,m.label_cn,m.train_class,m.n_roi,m.n_ok??"",s.d||"",s.n||"",m.full]}));
  const blob=new Blob(["﻿"+rows.map(r=>r.map(q).join(",")).join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="review_result.csv";a.click()};
prog();
</script></body></html>"""


if __name__ == "__main__":
    main()
