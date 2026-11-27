#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_review_train.py —— 训练截图人工核对工具（生成本地网页，双击打开）

用途：逐类目视检查训练截图，标出"局部 / 不合格 / 标签可疑"的图，并给每一类下结论（是否需要重截），
最后导出 CSV，据此决定哪些类、哪些原图要用 01_crop_batch_training.py 重新规范截取。

数据来源（默认 = 当前训练集）：
    python 04_review_train.py                                        # trainset/trainset_micro.csv
    python 04_review_train.py --trainset trainset_roi/trainset_micro.csv
    python 04_review_train.py --sources dev_original dev_checked     # 只看旧截图（160×160 那批 + ✅ 文件夹）
    python 04_review_train.py --classes Curvularia Mucor Rhizopus    # 只看某几类
    python 04_review_train.py --sites PUPH                           # 只看本院
    python 04_review_train.py --colony                               # 同时看菌落图
    python 04_review_train.py --dir "/路径/某图片文件夹"               # 任意 <文件夹>/<类别>/图片 结构
输出：review_train/index.html 和缩略图 review_train/img/（再次运行会复用已生成的缩略图）

网页操作：
  · 左侧选类别；顶部选"标注工具"（或按键 1–4），再点图片即打上该标记；再点一次取消
      1 合格   2 局部（结构不完整）   3 不合格（模糊/非目标结构/杂质）   4 标签可疑
  · Shift+点击：把上一次点击到这一张之间的图片全部打上同一标记
  · 每组和每类都有"其余未标记 → 合格"按钮，适合先挑问题图、再一键把剩下的标为合格
  · 双击图片（或按放大镜）看原图大小；Esc 关闭
  · 每类右上角填【本类结论】：无需处理 / 部分重截 / 全部重截 / 补充菌株，以及备注
  · 标记自动保存在浏览器里；"导出"生成 2 个 CSV：每张图的标记、每类的结论；"导入"可恢复之前导出的结果
"""
import os
import sys
import json
import hashlib
import argparse

import pandas as pd
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SOURCE_NAME = {"dev_original": "旧截图(固定160)", "dev_checked": "✅文件夹截图", "crops_new": "补截(crop_batch)",
               "crops_v2": "规范截图(v2)", "roi_ext": "外部ROI", "dir": "文件夹"}


def thumb(path, out_dir, size):
    try:
        st = os.stat(path)
    except OSError:
        return "", 0, 0
    fn = hashlib.md5(f"{path}|{st.st_mtime}|{size}".encode()).hexdigest()[:16] + ".jpg"
    dst = os.path.join(out_dir, fn)
    meta = dst + ".wh"
    if os.path.exists(dst) and os.path.exists(meta):
        w, h = open(meta).read().split(",")
        return "img/" + fn, int(w), int(h)
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return "", 0, 0
    w, h = im.size
    im.thumbnail((size, size))
    im.save(dst, quality=82)
    open(meta, "w").write(f"{w},{h}")
    return "img/" + fn, w, h


def load(a):
    if a.dir:
        root = os.path.expanduser(a.dir)
        rows = []
        for c in sorted(os.listdir(root)):
            cd = os.path.join(root, c)
            if not os.path.isdir(cd) or c.startswith((".", "_")):
                continue
            for d, subs, files in os.walk(cd):
                subs[:] = sorted(s for s in subs if not s.startswith("."))
                for fn in sorted(files):
                    if fn.lower().endswith(IMG_EXTS) and not fn.startswith("."):
                        rel = os.path.relpath(d, cd)
                        rows.append(dict(path=os.path.join(d, fn), train_class=c, modality="micro", source="dir",
                                         site="", cv_group=rel if rel != "." else "", species_cn=""))
        return pd.DataFrame(rows)
    parts = [pd.read_csv(a.trainset, encoding="utf-8-sig", dtype=str).fillna("")]
    if a.colony:
        cp = os.path.join(os.path.dirname(a.trainset), "trainset_colony.csv")
        if os.path.exists(cp):
            parts.append(pd.read_csv(cp, encoding="utf-8-sig", dtype=str).fillna(""))
    df = pd.concat(parts, ignore_index=True)
    for c in ("site", "source", "cv_group", "species_cn", "modality"):
        if c not in df:
            df[c] = ""
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainset", default="trainset")
    ap.add_argument("--dir", default="", help="改为扫描任意 <文件夹>/<类别>/ 图片")
    ap.add_argument("--classes", nargs="*", default=[])
    ap.add_argument("--sources", nargs="*", default=[], help="dev_original dev_checked crops_new crops_v2 roi_ext")
    ap.add_argument("--sites", nargs="*", default=[])
    ap.add_argument("--colony", action="store_true", help="同时载入 trainset_colony.csv 的菌落图")
    ap.add_argument("--thumb", type=int, default=200)
    ap.add_argument("--out", default="review_train", help="只写名字时放进 output/04_audit_review/")
    a = ap.parse_args()
    if not a.dir:
        a.trainset = OP.find_trainset_csv(a.trainset)
    a.out = OP.place(a.out, OP.AUDIT)

    df = load(a)
    if a.classes:
        df = df[df["train_class"].isin(a.classes)]
    if a.sources:
        df = df[df["source"].isin(a.sources)]
    if a.sites:
        df = df[df["site"].isin(a.sites)]
    if df.empty:
        sys.exit("筛选后没有图片")
    df = df.sort_values(["train_class", "modality", "source", "cv_group", "path"]).reset_index(drop=True)

    img_dir = os.path.join(a.out, "img")
    os.makedirs(img_dir, exist_ok=True)
    print(f"共 {len(df)} 张，{df['train_class'].nunique()} 类；生成缩略图（第一次较慢，之后复用）…")
    items, miss = [], 0
    for i, r in df.iterrows():
        t, w, h = thumb(r["path"], img_dir, a.thumb)
        if not t:
            miss += 1
        items.append(dict(i=i, p=r["path"], t=t, c=r["train_class"], m=r.get("modality", "") or "micro",
                          s=r.get("source", ""), st=r.get("site", ""), g=r.get("cv_group", ""),
                          cn=r.get("species_cn", ""), w=w, h=h,
                          k=hashlib.md5(r["path"].encode()).hexdigest()[:12]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")
    if miss:
        print(f"⚠️  {miss} 张图片读不到（网页中显示为灰块）")
    label = os.path.abspath(a.dir) if a.dir else os.path.abspath(a.trainset)
    # 浏览器里保存标记用的键：只用训练集文件夹名，文件夹搬家后标记不丢
    stable_key = os.path.basename(label.rstrip("/")) if a.dir else os.path.basename(os.path.dirname(label))
    page = PAGE.replace("{{DATA}}", json.dumps(items, ensure_ascii=False)) \
               .replace("{{SRCNAME}}", json.dumps(SOURCE_NAME, ensure_ascii=False)) \
               .replace("{{LABEL}}", json.dumps(label, ensure_ascii=False)) \
               .replace("{{KEY}}", "rt_" + hashlib.md5(stable_key.encode()).hexdigest()[:8])
    path = os.path.join(a.out, "index.html")
    open(path, "w", encoding="utf-8").write(page)
    print(f"\n已生成 {path}\n用浏览器打开：open {path}")


PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>训练截图核对</title>
<style>
:root{--bg:#F3F3F0;--panel:#fff;--line:#DCDED8;--fg:#1D2220;--muted:#68716C;--acc:#2A3F9E;
 --t1:#2E9A4E;--t2:#D08A00;--t3:#C8372D;--t4:#7B3FB5}
@media (prefers-color-scheme:dark){:root{--bg:#131615;--panel:#1B1F1E;--line:#2E3432;--fg:#E5E9E6;--muted:#98A29D;--acc:#93A6FF;
 --t1:#5CC57A;--t2:#F0B13C;--t3:#F07D72;--t4:#B88AF0;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"PingFang SC",sans-serif}
.app{display:grid;grid-template-columns:250px 1fr;height:100vh}
nav{border-right:1px solid var(--line);overflow:auto;background:var(--panel)}
nav h1{font-size:14px;margin:0;padding:10px 12px;border-bottom:1px solid var(--line)}
nav button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--line);background:none;color:var(--fg);padding:7px 12px;cursor:pointer;font:13px inherit}
nav button.on{background:var(--bg);box-shadow:inset 3px 0 var(--acc)}
nav .cn{font-family:Georgia,serif;font-style:italic}
nav .meta{display:flex;gap:6px;font-size:11px;color:var(--muted);margin-top:2px;flex-wrap:wrap}
.bar{height:4px;display:flex;margin-top:4px;border-radius:2px;overflow:hidden;background:var(--line)}
.bar span{display:block;height:100%}
.verd{font-size:11px;font-weight:600;color:var(--acc)}
main{overflow:auto;display:flex;flex-direction:column}
.top{position:sticky;top:0;z-index:4;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tool{border:2px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;padding:3px 10px;cursor:pointer;font:13px inherit}
.tool.on{border-color:currentColor;font-weight:700}
.tool[data-t="ok"]{color:var(--t1)}.tool[data-t="part"]{color:var(--t2)}.tool[data-t="bad"]{color:var(--t3)}.tool[data-t="label"]{color:var(--t4)}
select,input[type=text],.btn{font:12.5px inherit;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:5px;padding:4px 8px}
.btn{cursor:pointer}.sp{flex:1}.muted{color:var(--muted);font-size:12px}
.head{padding:12px 14px 4px;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
.head h2{margin:0;font:italic 20px Georgia,serif}
.verdict{display:flex;gap:6px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:6px 10px}
.verdict input{width:280px}
.stats{font-size:12px}
.stats b{margin-right:8px}
.group{padding:4px 14px 10px}
.group h3{font-size:12.5px;font-weight:500;color:var(--muted);margin:10px 0 6px;display:flex;gap:10px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:3px}
.group h3 b{color:var(--fg);font-family:ui-monospace,Menlo,monospace;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--sz,130px),1fr));gap:6px}
.tile{position:relative;border:3px solid transparent;border-radius:5px;background:var(--panel);cursor:pointer;overflow:hidden;user-select:none}
.tile img{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#000}
.tile .cap{font:10px ui-monospace,Menlo,monospace;color:var(--muted);padding:1px 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .tg{position:absolute;top:3px;left:3px;font-size:10.5px;font-weight:700;color:#fff;padding:0 5px;border-radius:3px}
.tile .zoom{position:absolute;top:3px;right:3px;background:rgb(0 0 0/.55);color:#fff;border:0;border-radius:3px;font-size:11px;padding:0 5px;cursor:zoom-in}
.tile.ok{border-color:var(--t1)}.tile.ok .tg{background:var(--t1)}
.tile.part{border-color:var(--t2)}.tile.part .tg{background:var(--t2)}
.tile.bad{border-color:var(--t3)}.tile.bad .tg{background:var(--t3)}
.tile.label{border-color:var(--t4)}.tile.label .tg{background:var(--t4)}
.tile.dim{opacity:.35}
#lb{position:fixed;inset:0;background:rgb(0 0 0/.88);z-index:9;display:grid;place-items:center;padding:20px}
#lb[hidden]{display:none}#lb img{max-width:92vw;max-height:80vh;image-rendering:auto;background:#000}
#lb .info{color:#ddd;font:12px ui-monospace,Menlo,monospace;margin-top:8px;max-width:92vw;word-break:break-all;text-align:center}
kbd{font:11px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;padding:0 4px}
</style></head><body><div class="app">
<nav><h1>训练截图核对</h1><div id="classes"></div></nav>
<main>
<div class="top">
  <span class="muted">标注工具</span>
  <button class="tool on" data-t="part">2 局部</button><button class="tool" data-t="bad">3 不合格</button>
  <button class="tool" data-t="label">4 标签可疑</button><button class="tool" data-t="ok">1 合格</button>
  <select id="grp"><option value="s">按来源分组</option><option value="st">按 site 分组</option><option value="g">按分组(cv_group)</option><option value="sz">按原始尺寸分组</option><option value="">不分组</option></select>
  <select id="flt"><option value="">显示全部</option><option value="un">只看未标记</option><option value="prob">只看有问题</option></select>
  <label class="muted">大小 <input id="sz" type="range" min="80" max="260" value="130"></label>
  <span class="sp"></span><span class="muted" id="prog"></span>
  <button class="btn" id="exp">导出</button><label class="btn">导入<input id="imp" type="file" accept=".csv" hidden multiple></label>
</div>
<div id="view"></div>
</main></div>
<div id="lb" hidden><div style="text-align:center"><img id="lbimg"><div class="info" id="lbinfo"></div></div></div>
<script>
const D={{DATA}}, SN={{SRCNAME}}, LABEL={{LABEL}}, KEY="{{KEY}}";
const TAG={ok:"合格",part:"局部",bad:"不合格",label:"标签可疑"}, VERD=["","无需处理","部分重截","全部重截","补充菌株"];
let S={tags:{},verd:{}}; try{Object.assign(S,JSON.parse(localStorage.getItem(KEY)||"{}"))}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const byC={}; D.forEach(d=>(byC[d.c]=byC[d.c]||[]).push(d));
const classes=Object.keys(byC).sort();
let cur=classes[0], tool="part", last=null;
function cnt(c){const r={n:byC[c].length,ok:0,part:0,bad:0,label:0,un:0};byC[c].forEach(d=>{const t=S.tags[d.k];t?r[t]++:r.un++});return r}
function nav(){
  document.getElementById("classes").innerHTML=classes.map(c=>{const r=cnt(c),v=(S.verd[c]||{}).v||"";
    const seg=k=>`<span style="width:${100*r[k]/r.n}%;background:var(--t${({ok:1,part:2,bad:3,label:4})[k]})"></span>`;
    return `<button data-c="${esc(c)}" class="${c===cur?"on":""}"><span class="cn">${esc(c)}</span>
    <div class="meta"><span>${r.n} 张</span><span>未看 ${r.un}</span>${r.part?`<span style="color:var(--t2)">局部 ${r.part}</span>`:""}${r.bad?`<span style="color:var(--t3)">不合格 ${r.bad}</span>`:""}${r.label?`<span style="color:var(--t4)">可疑 ${r.label}</span>`:""}${v?`<span class="verd">${v}</span>`:""}</div>
    <div class="bar">${seg("ok")+seg("part")+seg("bad")+seg("label")}</div></button>`}).join("");
  document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{cur=b.dataset.c;last=null;render();nav();document.querySelector("main").scrollTop=0});
  const all=D.length,done=D.filter(d=>S.tags[d.k]).length,vd=classes.filter(c=>(S.verd[c]||{}).v).length;
  document.getElementById("prog").textContent=`已标记 ${done}/${all} 张 · 已下结论 ${vd}/${classes.length} 类`;
}
function szb(d){const m=Math.max(d.w,d.h);return m?(m<=170?"≤170 px（旧固定框）":m<=330?"171–330 px":m<=500?"331–500 px":">500 px"):"未知"}
function gkey(d,g){return g==="s"?(SN[d.s]||d.s||"—")+(d.m==="colony"?" · 菌落":""):g==="st"?(d.st||"—"):g==="g"?(d.g||"—"):g==="sz"?szb(d):"全部"}
function render(){
  const g=document.getElementById("grp").value,f=document.getElementById("flt").value,v=S.verd[cur]||{};
  const r=cnt(cur);
  let items=byC[cur].filter(d=>!f||(f==="un"?!S.tags[d.k]:["part","bad","label"].includes(S.tags[d.k])));
  const groups=new Map();items.forEach(d=>{const k=gkey(d,g);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(d)});
  const cn=[...new Set(byC[cur].map(d=>d.cn).filter(Boolean))].slice(0,6).join("、");
  document.getElementById("view").innerHTML=`<div class="head"><h2>${esc(cur)}</h2><span class="muted">${esc(cn)}</span>
   <span class="stats"><b>${r.n} 张</b>未看 ${r.un} · <span style="color:var(--t1)">合格 ${r.ok}</span> · <span style="color:var(--t2)">局部 ${r.part}</span> · <span style="color:var(--t3)">不合格 ${r.bad}</span> · <span style="color:var(--t4)">可疑 ${r.label}</span></span>
   <button class="btn" data-rest="__all">本类其余未标记 → 合格</button></div>
   <div class="head"><div class="verdict"><b>本类结论</b><select id="vsel">${VERD.map(x=>`<option ${x===(v.v||"")?"selected":""}>${x}</option>`).join("")}</select>
   <input type="text" id="vnote" placeholder="备注：例如 160 框全是局部，需按菌株重截；缺 xx 菌株" value="${esc(v.n||"")}"></div></div>
   ${[...groups].map(([k,ims])=>`<section class="group"><h3><b>${esc(k)}</b><span>${ims.length} 张</span>
     <button class="btn" data-rest="${esc(k)}">本组其余未标记 → 合格</button></h3>
     <div class="grid">${ims.map(tile).join("")}</div></section>`).join("")||'<p class="muted" style="padding:14px">没有符合筛选条件的图片</p>'}`;
  document.querySelectorAll(".tile").forEach(t=>{t.onclick=e=>click(t.dataset.k,e.shiftKey);t.ondblclick=()=>zoom(t.dataset.k)});
  document.querySelectorAll(".zoom").forEach(b=>b.onclick=e=>{e.stopPropagation();zoom(b.parentNode.dataset.k)});
  document.querySelectorAll("[data-rest]").forEach(b=>b.onclick=()=>{const k=b.dataset.rest;
    byC[cur].forEach(d=>{if(!S.tags[d.k]&&(k==="__all"||gkey(d,g)===k))S.tags[d.k]="ok"});save();render();nav()});
  document.getElementById("vsel").onchange=e=>{S.verd[cur]={...(S.verd[cur]||{}),v:e.target.value};save();nav()};
  document.getElementById("vnote").oninput=e=>{S.verd[cur]={...(S.verd[cur]||{}),n:e.target.value};save()};
}
const KM={};D.forEach(d=>KM[d.k]=d);
function tile(d){const t=S.tags[d.k]||"";
  return `<div class="tile ${t}" data-k="${d.k}" title="${esc(d.p)}">${d.t?`<img loading="lazy" src="${d.t}">`:`<div style="aspect-ratio:1;background:var(--line)"></div>`}
  ${t?`<span class="tg">${TAG[t]}</span>`:""}<button class="zoom">⤢</button><div class="cap">${d.w?d.w+"×"+d.h:""} ${esc(d.p.split("/").pop())}</div></div>`}
function click(k,shift){
  const vis=[...document.querySelectorAll(".tile")].map(t=>t.dataset.k);
  let keys=[k];
  if(shift&&last&&vis.includes(last)){const a=vis.indexOf(last),b=vis.indexOf(k);keys=vis.slice(Math.min(a,b),Math.max(a,b)+1)}
  const clear=!shift&&S.tags[k]===tool;
  keys.forEach(x=>{if(clear)delete S.tags[x];else S.tags[x]=tool});
  last=k;save();
  keys.forEach(x=>{const el=document.querySelector(`.tile[data-k="${x}"]`);if(el)el.outerHTML=tile(KM[x])});
  document.querySelectorAll(".tile").forEach(t=>{t.onclick=e=>click(t.dataset.k,e.shiftKey);t.ondblclick=()=>zoom(t.dataset.k)});
  document.querySelectorAll(".zoom").forEach(b=>b.onclick=e=>{e.stopPropagation();zoom(b.parentNode.dataset.k)});
  nav();
}
function zoom(k){const d=KM[k];const im=document.getElementById("lbimg");im.onerror=()=>{im.onerror=null;im.src=d.t};
  im.src="file://"+d.p.split("/").map(encodeURIComponent).join("/");
  document.getElementById("lbinfo").textContent=`${d.c} · ${SN[d.s]||d.s} · ${d.st||""} · ${d.w}×${d.h} · ${d.g}\n${d.p}`;
  document.getElementById("lb").hidden=false}
document.getElementById("lb").onclick=()=>document.getElementById("lb").hidden=true;
document.querySelectorAll(".tool").forEach(b=>b.onclick=()=>{tool=b.dataset.t;document.querySelectorAll(".tool").forEach(x=>x.classList.toggle("on",x===b))});
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
  if(e.key==="Escape")document.getElementById("lb").hidden=true;
  const m={"1":"ok","2":"part","3":"bad","4":"label"}[e.key];if(m)document.querySelector(`.tool[data-t="${m}"]`).click()});
document.getElementById("grp").onchange=render;document.getElementById("flt").onchange=render;
document.getElementById("sz").oninput=e=>document.documentElement.style.setProperty("--sz",e.target.value+"px");
const q=v=>`"${String(v??"").replace(/"/g,'""')}"`;
function dl(name,rows){const b=new Blob(["﻿"+rows.map(r=>r.map(q).join(",")).join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=name;document.body.appendChild(a);a.click();a.remove()}
document.getElementById("exp").onclick=()=>{
  dl("review_images.csv",[["path","train_class","modality","source","site","cv_group","width","height","tag"]]
    .concat(D.map(d=>[d.p,d.c,d.m,d.s,d.st,d.g,d.w,d.h,TAG[S.tags[d.k]]||""])));
  setTimeout(()=>dl("review_classes.csv",[["train_class","n","unreviewed","ok","partial","bad","label_doubt","verdict","note"]]
    .concat(classes.map(c=>{const r=cnt(c),v=S.verd[c]||{};return [c,r.n,r.un,r.ok,r.part,r.bad,r.label,v.v||"",v.n||""]}))),400)};
function parse(txt){const out=[];let row=[],f="",inq=false;txt=txt.replace(/^﻿/,"");
  for(let i=0;i<txt.length;i++){const ch=txt[i];
    if(inq){if(ch=='"'){if(txt[i+1]=='"'){f+='"';i++}else inq=false}else f+=ch}
    else if(ch=='"')inq=true;else if(ch==","){row.push(f);f=""}else if(ch=="\n"){row.push(f);out.push(row);row=[];f=""}else if(ch!="\r")f+=ch}
  if(f||row.length){row.push(f);out.push(row)}return out}
document.getElementById("imp").onchange=async e=>{const rev=Object.fromEntries(Object.entries(TAG).map(([k,v])=>[v,k]));
  const P={};D.forEach(d=>P[d.p]=d.k);let n=0;
  for(const file of e.target.files){const rows=parse(await file.text()),h=rows[0];
    if(h.includes("tag")){const ip=h.indexOf("path"),it=h.indexOf("tag");rows.slice(1).forEach(r=>{const k=P[r[ip]];if(k&&rev[r[it]]){S.tags[k]=rev[r[it]];n++}})}
    if(h.includes("verdict")){const ic=h.indexOf("train_class"),iv=h.indexOf("verdict"),inn=h.indexOf("note");rows.slice(1).forEach(r=>{if(r[iv]||r[inn]){S.verd[r[ic]]={v:r[iv],n:r[inn]};n++}})}}
  save();render();nav();alert(`已导入 ${n} 条`)};
nav();render();
</script></body></html>"""


if __name__ == "__main__":
    main()
