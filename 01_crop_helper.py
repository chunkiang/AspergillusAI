#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_crop_helper.py —— 半自动截取 160×160 产孢结构（只需点选，不用画框）

第一步：生成候选小块和点选网页
    python 01_crop_helper.py propose
    （默认处理：网络图片中 黄曲霉 / 黑曲霉 / 横梗霉 的 C 类镜下图）
    python 01_crop_helper.py propose --classes "Aspergillus flavus (sect. Flavi)" Lichtheimia --sources web clinical

    → 打开 crop_review/review.html，点选能看清产孢结构的小块（绿框 = 选中），
      做完点页面顶部"导出选择"，浏览器会下载 selected.json

第二步：把选中的小块保存为训练截图
    python 01_crop_helper.py export ~/Downloads/selected.json

    → 截图存到 CNN/dataset_supp/<训练标签>/，文件名含来源文件夹和图片编号，方便按来源分组、日后追溯
    → 同时把这些原图写入 crop_review/used_for_training.csv，验证脚本会自动把它们排除出外部测试集

依赖（清华源）：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow numpy pandas pillow-heif
"""
import os
import re
import sys
import json
import html
import argparse

import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as OP   # noqa: E402  输出文件夹约定（output/01_… 06_…）

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

EXT_ROOT = "/Users/chunjiang/Desktop/丝状真菌"
SUPP_ROOT = "/Users/chunjiang/Documents/真菌AI/CNN/dataset_supp"
DEFAULT_CLASSES = ["Aspergillus flavus (sect. Flavi)", "Aspergillus niger (sect. Nigri)", "Lichtheimia"]
TILE = 160
STRIDE = 80
TOP_K = 16          # 每张原图最多给出的候选数
MAX_SIDE = 2400     # 原图过大时先等比缩小到长边 2400（与你手工截图时看到的尺度接近；可按需调整）


def safe(s):
    return re.sub(r"[^\w一-鿿.-]+", "_", s)


def load(path):
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        r = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    return im


def tile_scores(im):
    """前景评分：结构越多（暗、边缘丰富）分越高；纯背景、过曝、全黑得分低"""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    bg = np.median(g)
    dark = (bg - g) > 25                          # 比背景明显更暗/更有颜色的像素
    gy, gx = np.gradient(g)
    edge = np.hypot(gx, gy) > 12
    H, W = g.shape
    out = []
    for y in range(0, max(1, H - TILE + 1), STRIDE):
        for x in range(0, max(1, W - TILE + 1), STRIDE):
            d = dark[y:y + TILE, x:x + TILE].mean()
            e = edge[y:y + TILE, x:x + TILE].mean()
            if d < 0.03 or d > 0.85:              # 几乎空白，或几乎全被遮住
                continue
            out.append((float(0.5 * d + e), x, y))
    out.sort(reverse=True)
    keep = []
    for s, x, y in out:                           # 简单非极大值抑制：避免候选高度重叠
        if all(abs(x - kx) >= TILE * 0.6 or abs(y - ky) >= TILE * 0.6 for _, kx, ky in keep):
            keep.append((s, x, y))
        if len(keep) >= TOP_K:
            break
    return keep


def propose(a):
    man = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("")
    src_ok = {"web": man.folder.str.contains("网络"), "clinical": ~man.folder.str.contains("网络")}
    mask = man.role.eq("external_micro") & man.train_class.isin(a.classes)
    mask &= np.logical_or.reduce([src_ok[s] for s in a.sources])
    sel = man[mask]
    if sel.empty:
        sys.exit("没有符合条件的图片：请检查 --classes 是否与 taxonomy_map.csv 的 train_class 完全一致")
    os.makedirs(os.path.join(a.out, "tiles"), exist_ok=True)
    cards, n_tiles = [], 0
    for _, r in sel.iterrows():
        p = os.path.join(a.ext_root, r.rel_path)
        try:
            im = load(p)
        except Exception as e:
            print("读取失败:", p, e)
            continue
        cands = tile_scores(im)
        tiles = []
        for s, x, y in cands:
            fn = f"{safe(r.rel_path)}__x{x}_y{y}.jpg"
            im.crop((x, y, x + TILE, y + TILE)).save(os.path.join(a.out, "tiles", fn), quality=92)
            tiles.append((fn, x, y))
        n_tiles += len(tiles)
        thumb = f"thumb__{safe(r.rel_path)}.jpg"
        t = im.copy()
        t.thumbnail((360, 360))
        t.save(os.path.join(a.out, "tiles", thumb), quality=85)
        cards.append((r, thumb, tiles, im.size))
    # 网页
    items = []
    for r, thumb, tiles, size in cards:
        tt = "".join(
            f'<img class="t" src="tiles/{html.escape(fn)}" data-k="{html.escape(r.rel_path)}|{x}|{y}|{size[0]}|{size[1]}" title="x={x}, y={y}">'
            for fn, x, y in tiles)
        items.append(
            f'<div class="card"><div class="meta"><img class="th" src="tiles/{html.escape(thumb)}">'
            f'<div><b>{html.escape(r.train_class)}</b><br>{html.escape(r.species_cn)}<br>'
            f'<small>{html.escape(r.rel_path)}</small></div></div><div class="grid">{tt or "<i>没有找到候选（图上可能几乎没有结构）</i>"}</div></div>')
    page = f"""<!doctype html><meta charset="utf-8"><title>截图点选</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;background:#f4f4f4}}
header{{position:sticky;top:0;background:#1f4e78;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:9}}
button{{font-size:14px;padding:6px 14px;border:0;border-radius:6px;cursor:pointer}}
.card{{background:#fff;margin:12px 16px;padding:10px;border-radius:8px}}
.meta{{display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;font-size:13px}}
.th{{max-width:220px;border-radius:4px}}
.grid{{display:flex;flex-wrap:wrap;gap:6px}}
.t{{width:160px;height:160px;border:4px solid transparent;border-radius:4px;cursor:pointer}}
.t.on{{border-color:#2ea44f}}
</style>
<header><b>点选能看清产孢结构的小块（再点一次取消）</b><span id="n">已选 0</span>
<button onclick="exp()">导出选择</button><button onclick="if(confirm('清空所有选择？')){{S={{}};save();draw()}}">清空</button></header>
{''.join(items)}
<script>
let S={{}};try{{S=JSON.parse(localStorage.getItem('crop_sel')||'{{}}')}}catch(e){{}}
function save(){{try{{localStorage.setItem('crop_sel',JSON.stringify(S))}}catch(e){{}}}}
function draw(){{document.querySelectorAll('.t').forEach(t=>t.classList.toggle('on',!!S[t.dataset.k]));
document.getElementById('n').textContent='已选 '+Object.keys(S).length}}
document.querySelectorAll('.t').forEach(t=>t.onclick=()=>{{const k=t.dataset.k;if(S[k])delete S[k];else S[k]=1;save();draw()}});
function exp(){{const b=new Blob([JSON.stringify(Object.keys(S),null,1)],{{type:'application/json'}});
const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='selected.json';a.click()}}
draw();
</script>"""
    with open(os.path.join(a.out, "review.html"), "w", encoding="utf-8") as f:
        f.write(page)
    sel.to_csv(os.path.join(a.out, "candidates_source.csv"), index=False, encoding="utf-8-sig")
    print(f"原图 {len(cards)} 张，候选小块 {n_tiles} 个 → 打开 {os.path.abspath(os.path.join(a.out, 'review.html'))}")


def export(a):
    keys = json.load(open(a.selected, encoding="utf-8"))
    man = pd.read_csv(a.manifest, encoding="utf-8-sig", dtype=str).fillna("").set_index("rel_path")
    used, n = set(), 0
    for k in keys:
        rel, x, y, w, h = k.split("|")
        x, y, w, h = int(x), int(y), int(w), int(h)
        r = man.loc[rel]
        im = load(os.path.join(a.ext_root, rel))
        if im.size != (w, h):
            sys.exit(f"图片尺寸与生成候选时不一致：{rel}（是否修改过 MAX_SIDE？）")
        d = os.path.join(a.supp_root, safe(r.train_class))
        os.makedirs(d, exist_ok=True)
        im.crop((x, y, x + TILE, y + TILE)).save(os.path.join(d, f"{safe(rel)}__x{x}_y{y}.png"))
        used.add(rel)
        n += 1
    out = os.path.join(a.out, "used_for_training.csv")
    old = set(pd.read_csv(out, encoding="utf-8-sig").rel_path) if os.path.exists(out) else set()
    pd.DataFrame({"rel_path": sorted(old | used)}).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"保存截图 {n} 张，来自原图 {len(used)} 张 → {a.supp_root}")
    print(f"这些原图已记入 {out}，之后不会进入外部测试集")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("propose", "export"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", default=OP.MANIFEST)
        p.add_argument("--ext_root", default=EXT_ROOT)
        p.add_argument("--out", default="crop_review")
        if name == "propose":
            p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
            p.add_argument("--sources", nargs="+", default=["web"], choices=["web", "clinical"])
        else:
            p.add_argument("selected")
            p.add_argument("--supp_root", default=SUPP_ROOT)
    a = ap.parse_args()
    a.manifest = OP.find(a.manifest, OP.EXT_AUDIT)
    a.out = OP.place(a.out, OP.AUDIT)
    propose(a) if a.cmd == "propose" else export(a)


if __name__ == "__main__":
    main()
