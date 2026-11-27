#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_gallery.py — 训练图片查看器
================================================================
在本机打开一个网页：先在网页上选择训练集（trainset、trainset_roi……，
即 build_trainset.py 生成的、含 trainset_micro.csv / trainset_colony.csv 的文件夹），
再输入菌名（如 Scedosporium、烟曲霉、fumigatus），就会读取该训练集的两个 CSV，
把这个类的全部显微截图和菌落图显示出来，可按来源/分组排列，点开看大图和元数据。

用法（在脚本所在文件夹、fungal 环境里）：
    python train_gallery.py                                   # 打开网页，训练集和菌名都在网页里选
    python train_gallery.py Scedosporium --trainset trainset_roi   # 打开时直接定位到某训练集和菌名
    python train_gallery.py --root /Users/chunjiang/Documents/真菌AI   # 到别的目录里找训练集

网页会列出 --root（默认当前目录）及其下两层里所有训练集文件夹；也可以在网页里
选“其他路径…”直接填写文件夹路径。若填的是按类别分文件夹的图片目录（没有 CSV），
会按 <目录>/<类别>/ 扫描图片。

只依赖 Python 标准库；装了 Pillow（fungal 环境里一般有）会生成缩略图，翻页更快，
并能在浏览器里显示 .tif 图片。服务只监听本机 127.0.0.1，按 Ctrl+C 退出。
"""
import argparse
import csv
import io
import json
import mimetypes
import os
import sys
import threading
import urllib.parse
import webbrowser
from collections import OrderedDict, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None
    HAS_PIL = True
except Exception:
    HAS_PIL = False

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
BROWSER_OK = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}

PATH_CANDS = ["path", "abs_path", "filepath", "file_path", "img_path", "image_path",
              "crop_path", "micro_path", "colony_path", "src", "file", "image", "filename"]
CLASS_CANDS = ["train_class", "class", "class_name", "label", "species", "latin"]
ALIAS_CANDS = ["species_cn", "label_cn", "cn", "latin", "label", "species", "genus"]
GROUP_CANDS = ["site", "cv_group", "source", "folder", "isolate_id", "strain", "case_id"]


# ------------------------------------------------------------------
# 读取数据
# ------------------------------------------------------------------
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def pick_col(cols, cands, rows=None, test=None):
    low = {c.lower().strip(): c for c in cols}
    for c in cands:
        if c in low:
            col = low[c]
            if test is None or (rows and test(rows, col)):
                return col
    return None


def looks_like_img(rows, col):
    vals = [str(r.get(col) or "") for r in rows[:200] if r.get(col)]
    if not vals:
        return False
    hits = sum(Path(v.strip()).suffix.lower() in IMG_EXT for v in vals)
    return hits / len(vals) > 0.6


def resolve(p, bases):
    p = p.strip().strip('"').strip("'")
    if not p:
        return None
    pp = Path(os.path.expanduser(p))
    if pp.is_absolute():
        return pp if pp.exists() else None
    for b in bases:
        q = (b / pp)
        if q.exists():
            return q.resolve()
    return None


def load_trainset(ts_dir, extra_bases):
    """读取 trainset_*.csv，返回图片记录列表。"""
    ts_dir = Path(ts_dir).expanduser().resolve()
    files = []
    for kind, name in [("micro", "trainset_micro.csv"), ("colony", "trainset_colony.csv")]:
        f = ts_dir / name
        if f.exists():
            files.append((kind, f))
    if not files:  # 兼容其他命名
        for f in sorted(ts_dir.glob("*.csv")):
            kind = "colony" if any(k in f.stem.lower() for k in ("colony", "normal", "macro")) else "micro"
            files.append((kind, f))
    if not files:
        raise FileNotFoundError(f"在 {ts_dir} 里没有找到 trainset_micro.csv / trainset_colony.csv")

    records, report = [], []
    for kind, f in files:
        rows = read_csv(f)
        if not rows:
            report.append(f"  {f.name}: 空文件")
            continue
        cols = list(rows[0].keys())
        pcol = pick_col(cols, PATH_CANDS, rows, looks_like_img) or next(
            (c for c in cols if looks_like_img(rows, c)), None)
        ccol = pick_col(cols, CLASS_CANDS)
        if not pcol or not ccol:
            report.append(f"  {f.name}: 未识别出路径列({pcol}) 或类别列({ccol})；列名 = {cols}")
            continue
        bases = [f.parent, f.parent.parent, f.parent.parent.parent, Path.cwd()] + extra_bases
        alias_cols = [c for c in cols if c.lower() in ALIAS_CANDS and c != ccol]
        ok = miss = 0
        for r in rows:
            cls = (r.get(ccol) or "").strip()
            if not cls:
                continue
            rp = resolve(r.get(pcol) or "", bases)
            if rp is None:
                miss += 1
            else:
                ok += 1
            meta = {k: v for k, v in r.items() if k not in (pcol,) and v not in (None, "")}
            records.append({
                "kind": kind, "cls": cls, "path": str(rp) if rp else None,
                "raw": r.get(pcol), "name": Path(str(r.get(pcol) or "")).name,
                "aliases": {(r.get(c) or "").strip() for c in alias_cols} - {""},
                "meta": meta,
            })
        report.append(f"  {f.name}: {len(rows)} 行（路径列 {pcol}，类别列 {ccol}）"
                      f" → 找到图片 {ok}，找不到 {miss}")
    return records, report


def is_trainset_dir(d):
    return d.is_dir() and any((d / n).exists() for n in ("trainset_micro.csv", "trainset_colony.csv"))


def discover_trainsets(roots):
    """在给定目录（及其下一层、两层）里找所有训练清单文件夹。"""
    found = OrderedDict()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        cands = [root] + [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        for d in list(cands[1:]):
            try:
                cands += [e for e in sorted(d.iterdir()) if e.is_dir() and not e.name.startswith(".")]
            except OSError:
                pass
        for d in cands:
            try:
                if is_trainset_dir(d):
                    found[str(d.resolve())] = 1
            except OSError:
                pass
    return list(found)


def scan_dir(root):
    """扫描 <root>/<类别>/**/图片；子路径里含 colony/normal/macro 的算菌落图。"""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在：{root}")
    records = []
    for cdir in sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))):
        for p in sorted(cdir.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or p.name.startswith("."):
                continue
            rel = p.relative_to(cdir)
            sub = "/".join(rel.parts[:-1])
            kind = "colony" if any(k in sub.lower() for k in ("colony", "normal", "macro")) else "micro"
            records.append({"kind": kind, "cls": cdir.name, "path": str(p), "raw": str(p.relative_to(root)),
                            "name": p.name, "aliases": set(),
                            "meta": {"subfolder": sub} if sub else {}})
    return records, [f"  扫描 {root}：{len(records)} 张图片"]


def load_aliases(files, classes):
    """可选：读取 taxonomy_map.csv，让中文名/拉丁名也能搜到训练类。"""
    out = defaultdict(set)
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        try:
            rows = read_csv(f)
        except Exception:
            continue
        if not rows:
            continue
        cols = list(rows[0].keys())
        ccol = pick_col(cols, ["train_class"])
        if not ccol:
            continue
        for r in rows:
            c = (r.get(ccol) or "").strip()
            if c in classes:
                for k, v in r.items():
                    if k != ccol and v and len(v) < 80:
                        out[c].add(v.strip())
    return out


# ------------------------------------------------------------------
# 服务
# ------------------------------------------------------------------
class Store:
    def __init__(self, records, alias_extra):
        self.records = records
        for i, r in enumerate(records):
            r["id"] = i
        self.by_cls = defaultdict(list)
        for r in records:
            self.by_cls[r["cls"]].append(r)
        self.aliases = defaultdict(set)
        for r in records:
            self.aliases[r["cls"]] |= r["aliases"]
        for c, a in alias_extra.items():
            self.aliases[c] |= a
        meta_keys = OrderedDict()
        for r in records:
            for k in r["meta"]:
                meta_keys[k] = 1
        self.group_cols = [k for k in GROUP_CANDS if k in meta_keys] + (["subfolder"] if "subfolder" in meta_keys else [])
        self.thumbs = OrderedDict()
        self.lock = threading.Lock()

    def classes(self):
        out = []
        for c, rs in sorted(self.by_cls.items(), key=lambda x: x[0].lower()):
            out.append({"name": c,
                        "micro": sum(r["kind"] == "micro" for r in rs),
                        "colony": sum(r["kind"] == "colony" for r in rs),
                        "missing": sum(r["path"] is None for r in rs),
                        "aliases": sorted(self.aliases[c])[:8]})
        return out

    def match(self, q):
        q = q.strip().lower()
        if not q:
            return []
        names = list(self.by_cls)
        exact = [c for c in names if c.lower() == q or q in {a.lower() for a in self.aliases[c]}]
        if exact:
            return exact
        return [c for c in names
                if q in c.lower() or any(q in a.lower() for a in self.aliases[c])]

    def images(self, classes):
        out = []
        for c in classes:
            for r in self.by_cls[c]:
                out.append({"id": r["id"], "kind": r["kind"], "cls": r["cls"], "name": r["name"],
                            "ok": r["path"] is not None, "path": r["path"] or r["raw"], "meta": r["meta"]})
        return out

    def image_bytes(self, i, thumb):
        r = self.records[i]
        p = r["path"]
        if not p:
            return None, None
        ext = Path(p).suffix.lower()
        if HAS_PIL and (thumb or ext not in BROWSER_OK):
            key = (i, thumb)
            with self.lock:
                if key in self.thumbs:
                    self.thumbs.move_to_end(key)
                    return self.thumbs[key], "image/jpeg"
            try:
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    if thumb:
                        im.thumbnail((420, 420))
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=85)
                    data = buf.getvalue()
            except Exception:
                return None, None
            with self.lock:
                self.thumbs[key] = data
                while len(self.thumbs) > 4000:
                    self.thumbs.popitem(last=False)
            return data, "image/jpeg"
        try:
            with open(p, "rb") as f:
                return f.read(), mimetypes.guess_type(p)[0] or "application/octet-stream"
        except OSError:
            return None, None


class Library:
    """按需加载训练集；同一路径只读一次，可强制重读。"""
    def __init__(self, roots, extra_bases, taxonomy):
        self.roots, self.extra, self.taxonomy = roots, extra_bases, taxonomy
        self.stores, self.keys = {}, []
        self.lock = threading.Lock()

    def resolve_ts(self, ts):
        p = Path(os.path.expanduser(ts.strip()))
        if not p.is_absolute():
            for r in self.roots:
                if (Path(r) / p).exists():
                    p = Path(r) / p
                    break
        return p.resolve()

    def get(self, ts, reload=False):
        path = self.resolve_ts(ts)
        key = str(path)
        with self.lock:
            if key in self.stores and not reload:
                return self.keys.index(key), self.stores[key]
        if is_trainset_dir(path):
            records, report = load_trainset(path, self.extra)
        elif path.is_dir():
            records, report = scan_dir(path)
        else:
            raise FileNotFoundError(f"找不到训练集文件夹：{path}")
        if not records:
            raise FileNotFoundError(f"{path} 里没有读到任何图片记录\n" + "\n".join(report))
        classes = {r["cls"] for r in records}
        alias = load_aliases([Path(r) / self.taxonomy for r in self.roots] + [path / "taxonomy_map.csv",
                              path.parent / "taxonomy_map.csv"], classes)
        store = Store(records, alias)
        store.report, store.path = report, key
        print(f"[载入] {key}\n" + "\n".join(report))
        with self.lock:
            self.stores[key] = store
            if key not in self.keys:
                self.keys.append(key)
            return self.keys.index(key), store

    def by_sid(self, sid):
        with self.lock:
            return self.stores[self.keys[sid]]


def make_handler(lib, default_ts):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj):
            self.send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(u.query)
            arg = lambda k, d="": (qs.get(k) or [d])[0]
            try:
                if u.path == "/":
                    self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif u.path == "/api/trainsets":
                    found = discover_trainsets(lib.roots)
                    rel = []
                    for f in found:
                        for r in lib.roots:
                            try:
                                rel.append(str(Path(f).relative_to(Path(r).resolve())) or ".")
                                break
                            except ValueError:
                                continue
                        else:
                            rel.append(f)
                    self.js({"trainsets": rel, "default": default_ts,
                             "roots": [str(Path(r).resolve()) for r in lib.roots]})
                elif u.path == "/api/load":
                    try:
                        sid, store = lib.get(arg("ts", default_ts), reload=arg("reload") == "1")
                    except (FileNotFoundError, OSError) as e:
                        self.js({"error": str(e)})
                        return
                    self.js({"sid": sid, "path": store.path, "report": store.report,
                             "classes": store.classes(), "group_cols": store.group_cols,
                             "pil": HAS_PIL, "total": len(store.records)})
                elif u.path == "/api/search":
                    store = lib.by_sid(int(arg("sid")))
                    q = arg("q")
                    cls = store.match(q)
                    self.js({"q": q, "classes": cls, "images": store.images(cls)})
                elif u.path.startswith("/img/"):
                    _, _, sid, i = u.path.split("/")[:4]
                    store, i = lib.by_sid(int(sid)), int(i)
                    if not 0 <= i < len(store.records):
                        raise ValueError
                    data, ct = store.image_bytes(i, (qs.get("t") or ["0"])[0] == "1")
                    if data is None:
                        self.send(404, b"not found", "text/plain")
                    else:
                        self.send(200, data, ct, cache=True)
                else:
                    self.send(404, b"not found", "text/plain")
            except (ValueError, IndexError):
                self.send(400, b"bad request", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


# ------------------------------------------------------------------
# 前端页面
# ------------------------------------------------------------------
PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>训练图片查看器</title>
<style>
:root{
  --bg:#F3F4F1; --panel:#FFFFFF; --line:#DCE0DA; --fg:#1C2220; --muted:#5E6863; --accent:#2A3F9E;
  --chip:#E9ECF5; --bad:#B23A2E; --bad-bg:#F8E3E0; --shade:rgb(10 14 13/.86);
  --f-body:"Noto Sans SC",system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  --f-sp:"Source Serif 4",Georgia,"Times New Roman",serif;
  --f-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#131716; --panel:#1B201F; --line:#2D3432; --fg:#E6EAE7; --muted:#97A19C; --accent:#8EA2FF;
  --chip:#232838; --bad:#F08074; --bad-bg:#3C1E1B; color-scheme:dark}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 var(--f-body)}
.wrap{max-width:1500px;margin:0 auto;padding-inline:16px;padding-block:0 48px}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin-inline:-16px;padding:12px 16px}
.row{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:center}
h1{font-size:16px;margin:0 auto 0 0;font-weight:700}
h1 small{font:400 12px var(--f-mono);color:var(--muted);margin-left:8px}
.search{position:relative;flex:1 1 320px;max-width:520px}
#q{width:100%;font:15px var(--f-body);padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg)}
#q:focus-visible,button:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#sugg{position:absolute;left:0;right:0;top:calc(100% + 4px);background:var(--panel);border:1px solid var(--line);border-radius:8px;max-height:60vh;overflow:auto;box-shadow:0 8px 24px rgb(0 0 0/.12)}
#sugg button{display:flex;justify-content:space-between;gap:12px;width:100%;text-align:left;border:0;background:none;color:var(--fg);padding:7px 12px;font:inherit;cursor:pointer}
#sugg button:hover,#sugg button.on{background:var(--chip)}
.sp{font-family:var(--f-sp);font-style:italic}
.cnt{font:12px var(--f-mono);color:var(--muted);white-space:nowrap}
.tools{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:10px}
.chip{font:12px var(--f-body);border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:999px;padding:3px 10px;cursor:pointer}
.chip[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
select,.btn{font:12px var(--f-body);border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:6px;padding:4px 8px;cursor:pointer}
label.sz{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}
.summary{margin:16px 0 4px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:baseline}
.summary .title{font-size:20px}
.note{color:var(--muted);font-size:12.5px}
.group{margin-top:18px}
.group h2{font-size:13px;font-weight:500;color:var(--muted);margin:0 0 8px;display:flex;gap:10px;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:4px}
.group h2 b{color:var(--fg);font-family:var(--f-mono);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--tile,150px),1fr));gap:8px}
.tile{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;cursor:zoom-in;padding:0;color:inherit;font:inherit;text-align:left}
.tile img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;background:var(--chip)}
.tile .cap{font:10.5px var(--f-mono);color:var(--muted);padding:3px 6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tile .k{position:absolute;top:4px;left:4px;font:600 10px var(--f-mono);background:var(--shade);color:#fff;padding:1px 5px;border-radius:3px}
.tile.bad{cursor:default}.tile.bad .ph{aspect-ratio:1/1;display:grid;place-items:center;background:var(--bad-bg);color:var(--bad);font-size:11px;padding:8px;text-align:center}
.empty{padding:60px 16px;text-align:center;color:var(--muted)}
.classlist{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:4px 16px;margin-top:12px}
.classlist button{display:flex;justify-content:space-between;gap:8px;border:0;border-bottom:1px dashed var(--line);background:none;color:var(--fg);padding:5px 2px;font:inherit;cursor:pointer;text-align:left}
.classlist button:hover{color:var(--accent)}
/* lightbox */
#lb{position:fixed;inset:0;background:var(--shade);z-index:20;display:grid;grid-template-columns:minmax(0,1fr) 320px}
#lb[hidden]{display:none}
#lb .stage{display:grid;place-items:center;padding:16px;min-width:0;position:relative}
#lb .stage img{max-width:100%;max-height:calc(100vh - 32px);object-fit:contain;border-radius:4px}
#lb aside{background:var(--panel);color:var(--fg);padding:16px;overflow:auto}
#lb aside h3{margin:0 0 4px;font-size:15px}
#lb dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:4px 10px;font-size:12.5px;margin:12px 0}
#lb dt{color:var(--muted);font-family:var(--f-mono)} #lb dd{margin:0;word-break:break-all}
.path{font:11.5px var(--f-mono);word-break:break-all;background:var(--bg);padding:6px 8px;border-radius:4px;user-select:all}
.navbtn{position:absolute;top:50%;transform:translateY(-50%);border:0;background:rgb(255 255 255/.15);color:#fff;font-size:24px;width:44px;height:64px;border-radius:6px;cursor:pointer}
.navbtn.l{left:10px}.navbtn.r{right:10px}
.x{position:absolute;top:10px;right:10px;border:0;background:rgb(255 255 255/.15);color:#fff;font-size:18px;width:36px;height:36px;border-radius:50%;cursor:pointer}
@media (max-width:760px){#lb{grid-template-columns:1fr;grid-template-rows:1fr auto}#lb aside{max-height:40vh}}
kbd{font:11px var(--f-mono);border:1px solid var(--line);border-bottom-width:2px;border-radius:3px;padding:0 4px}
.tsbox{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.tsbox label{font-size:12px;color:var(--muted)}
#ts{font:13px var(--f-mono);padding:6px 8px;max-width:260px}
#tsPath{font:12.5px var(--f-mono);padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--fg);width:22em;max-width:100%}
.status{margin-top:8px;font:12px var(--f-mono);color:var(--muted);white-space:pre-wrap;word-break:break-all}
.status.err{color:var(--bad)}
</style></head><body><div class="wrap">
<header>
  <div class="row">
    <h1>训练图片查看器</h1>
    <div class="tsbox">
      <label for="ts">训练集</label>
      <select id="ts" aria-label="训练集"></select>
      <input id="tsPath" type="text" spellcheck="false" placeholder="或填写训练集文件夹路径" hidden>
      <button class="btn" id="reload" title="重新读取 CSV（清单更新后用）">重新读取</button>
    </div>
    <div class="search">
      <input id="q" type="search" autocomplete="off" placeholder="输入菌名：Scedosporium / 烟曲霉 / fumigatus，回车查看" aria-label="菌名">
      <div id="sugg" hidden></div>
    </div>
  </div>
  <div class="tools" id="tools" hidden>
    <button class="chip" data-k="all" aria-pressed="true">全部</button>
    <button class="chip" data-k="micro" aria-pressed="false">显微</button>
    <button class="chip" data-k="colony" aria-pressed="false">菌落</button>
    <select id="grp" aria-label="分组方式"><option value="">不分组</option></select>
    <label class="sz">缩略图 <input id="size" type="range" min="90" max="320" value="150"></label>
    <span class="note">点图看大图，<kbd>←</kbd><kbd>→</kbd> 翻页，<kbd>Esc</kbd> 关闭</span>
  </div>
  <div class="status" id="src"></div>
</header>
<div id="main"></div>
</div>
<div id="lb" hidden role="dialog" aria-modal="true" aria-label="大图">
  <div class="stage"><img id="lbimg" alt=""><button class="navbtn l" id="lbp" aria-label="上一张">‹</button><button class="navbtn r" id="lbn" aria-label="下一张">›</button><button class="x" id="lbx" aria-label="关闭">✕</button></div>
  <aside id="lbinfo"></aside>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const sp=n=>/[一-鿿]/.test(n)?esc(n):`<span class="sp">${esc(n).replace(/\b(sect\.|complex)/g,'<span style="font-style:normal">$1</span>')}</span>`;
let INFO=null, data=null, kind="all", shown=[], lbi=-1, curTs="";
const OTHER="__other__";

/* 地址栏 #ts=…&q=… 记住当前训练集和菌名，刷新或收藏后可直接回到同一页面 */
function readHash(){ const h=new URLSearchParams(location.hash.slice(1)); return {ts:h.get("ts")||"", q:h.get("q")||""}; }
function writeHash(){ const h=new URLSearchParams(); if(curTs) h.set("ts",curTs); const q=$("#q").value.trim(); if(q) h.set("q",q); history.replaceState(null,"","#"+h.toString()); }

async function boot(){
  const r=await (await fetch("/api/trainsets")).json();
  const h=readHash();
  const sel=$("#ts");
  sel.innerHTML=r.trainsets.map(t=>`<option value="${esc(t)}">${esc(t)}</option>`).join("")+`<option value="${OTHER}">其他路径…</option>`;
  let want=h.ts||r.default||r.trainsets[0]||"";
  if(want && !r.trainsets.includes(want)){ sel.value=OTHER; $("#tsPath").hidden=false; $("#tsPath").value=want; }
  else sel.value=want;
  if(!r.trainsets.length && !want){
    sel.value=OTHER; $("#tsPath").hidden=false;
    status(`在 ${r.roots.join("、")} 下没有找到含 trainset_micro.csv 的文件夹。请在右侧填写训练集文件夹路径后回车。`,true);
    return;
  }
  $("#q").value=h.q;
  await loadTs(want);
}
function status(msg,err){ const s=$("#src"); s.textContent=msg; s.classList.toggle("err",!!err); }
async function loadTs(ts,reload){
  if(!ts) return;
  status(`正在读取 ${ts} …`);
  const r=await (await fetch(`/api/load?ts=${encodeURIComponent(ts)}${reload?"&reload=1":""}`)).json();
  if(r.error){ INFO=null; status(r.error,true); $("#main").innerHTML=""; $("#tools").hidden=true; return; }
  INFO=r; curTs=ts;
  status(`${r.path}　${r.classes.length} 类 · ${r.total} 张\n${r.report.join("\n")}`);
  const g=$("#grp"), prev=g.value;
  g.innerHTML='<option value="">不分组</option>'+r.group_cols.map(c=>`<option value="${esc(c)}">按 ${esc(c)} 分组</option>`).join("");
  g.value = r.group_cols.includes(prev) ? prev : (r.group_cols.includes("site")?"site":"");
  writeHash();
  const q=$("#q").value.trim();
  if(q) search(q); else home();
}
$("#ts").onchange=e=>{
  const v=e.target.value;
  $("#tsPath").hidden = v!==OTHER;
  if(v===OTHER){ $("#tsPath").focus(); return; }
  loadTs(v);
};
$("#tsPath").addEventListener("keydown",e=>{ if(e.key==="Enter"){ e.preventDefault(); loadTs(e.target.value.trim()); } });
$("#reload").onclick=()=>loadTs(curTs || ($("#ts").value===OTHER?$("#tsPath").value.trim():$("#ts").value), true);
function home(){
  data=null; $("#tools").hidden=true; if(!INFO) return;
  $("#main").innerHTML=`<div class="summary"><span class="note">全部训练类别（点击查看）</span></div>
   <div class="classlist">${INFO.classes.map(c=>`<button data-c="${esc(c.name)}"><span>${sp(c.name)}${c.aliases.length?` <span class="note">${esc(c.aliases.filter(a=>/[一-鿿]/.test(a)).slice(0,2).join(" / "))}</span>`:""}</span><span class="cnt">${c.micro}+${c.colony}</span></button>`).join("")}</div>
   <p class="note">数字为 显微+菌落 张数。</p>`;
  $("#main").querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>{ $("#q").value=b.dataset.c; search(b.dataset.c); });
}
async function search(q){
  $("#sugg").hidden=true;
  writeHash();
  if(!INFO){ status("请先选择训练集。",true); return; }
  if(!q.trim()){ home(); return; }
  data=await (await fetch(`/api/search?sid=${INFO.sid}&q=${encodeURIComponent(q)}`)).json();
  render();
}
function render(){
  if(!data) return;
  if(!data.classes.length){
    $("#tools").hidden=true;
    $("#main").innerHTML=`<div class="empty">训练集里没有匹配“${esc(data.q)}”的类别。<br>可以试试属名、种加词或中文名，或清空搜索框查看全部类别。</div>`; return;
  }
  $("#tools").hidden=false;
  shown=data.images.filter(im=>kind==="all"||im.kind===kind);
  const nm=data.images.filter(i=>i.kind==="micro").length, nc=data.images.length-nm, bad=data.images.filter(i=>!i.ok).length;
  const g=$("#grp").value;
  const groups=new Map();
  shown.forEach((im,i)=>{ im._i=i; const key=(data.classes.length>1?im.cls+" · ":"")+(g?(im.meta[g]??"（空）"):(im.kind==="micro"?"显微":"菌落")); if(!groups.has(key)) groups.set(key,[]); groups.get(key).push(im); });
  const nGroups=g?new Set(shown.map(i=>i.meta[g])).size:0;
  $("#main").innerHTML=`<div class="summary"><span class="title">${data.classes.map(sp).join("、")}</span><span class="cnt">训练集 ${esc(curTs)}</span>
    <span class="cnt">显微 ${nm} · 菌落 ${nc}${g?` · ${esc(g)} ${nGroups} 个`:""}${bad?` · <span style="color:var(--bad)">找不到文件 ${bad}</span>`:""}</span>
    ${data.classes.length>1?`<span class="note">匹配到多个类，输入完整类名可只看一个</span>`:""}</div>
    ${[...groups].map(([k,ims])=>`<section class="group"><h2><b>${esc(k)}</b><span>${ims.length} 张</span></h2><div class="grid">${ims.map(tile).join("")}</div></section>`).join("")}`;
  $("#main").querySelectorAll(".tile[data-i]").forEach(t=>t.onclick=()=>openLb(+t.dataset.i));
}
function tile(im){
  const k=im.kind==="micro"?"显微":"菌落";
  if(!im.ok) return `<div class="tile bad" title="${esc(im.path)}"><div class="ph">找不到文件<br>${esc(im.name)}</div><div class="cap">${esc(im.name)}</div></div>`;
  return `<button class="tile" data-i="${im._i}" title="${esc(im.name)}"><img loading="lazy" src="/img/${INFO.sid}/${im.id}?t=1" alt="${esc(im.name)}"><span class="k">${k}</span><div class="cap">${esc(im.name)}</div></button>`;
}
function openLb(i){
  lbi=i; const im=shown[i]; if(!im) return;
  $("#lb").hidden=false; $("#lbimg").src=`/img/${INFO.sid}/${im.id}`; $("#lbimg").alt=im.name;
  $("#lbinfo").innerHTML=`<h3>${sp(im.cls)}</h3><div class="cnt">${im.kind==="micro"?"显微":"菌落"} · 第 ${i+1} / ${shown.length} 张</div>
    <dl>${Object.entries(im.meta).map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
    <div class="note">文件路径</div><div class="path" id="lbpath">${esc(im.path)}</div>
    <p><button class="btn" id="cp">复制路径</button></p>`;
  $("#cp").onclick=()=>navigator.clipboard?.writeText(im.path).then(()=>$("#cp").textContent="已复制").catch(()=>getSelection().selectAllChildren($("#lbpath")));
}
const closeLb=()=>{ $("#lb").hidden=true; lbi=-1; };
const step=d=>{ if(lbi<0) return; let j=lbi+d; while(shown[j]&&!shown[j].ok) j+=d; if(shown[j]) openLb(j); };
$("#lbp").onclick=()=>step(-1); $("#lbn").onclick=()=>step(1); $("#lbx").onclick=closeLb;
$("#lb").addEventListener("click",e=>{ if(e.target.classList.contains("stage")) closeLb(); });
document.addEventListener("keydown",e=>{
  if(lbi>=0){ if(e.key==="Escape") closeLb(); else if(e.key==="ArrowLeft") step(-1); else if(e.key==="ArrowRight") step(1); return; }
  if(e.key==="/" && document.activeElement!==$("#q")){ e.preventDefault(); $("#q").focus(); }
});
document.querySelectorAll(".chip[data-k]").forEach(b=>b.onclick=()=>{ kind=b.dataset.k; document.querySelectorAll(".chip[data-k]").forEach(x=>x.setAttribute("aria-pressed",x===b)); render(); });
$("#grp").onchange=render;
$("#size").oninput=e=>document.documentElement.style.setProperty("--tile",e.target.value+"px");

/* 联想 */
let sel=-1;
function suggest(){
  const q=$("#q").value.trim().toLowerCase(); const box=$("#sugg");
  if(!q||!INFO){ box.hidden=true; return; }
  const hits=INFO.classes.filter(c=>c.name.toLowerCase().includes(q)||c.aliases.some(a=>a.toLowerCase().includes(q))).slice(0,12);
  if(!hits.length){ box.hidden=true; return; }
  sel=-1;
  box.innerHTML=hits.map(c=>`<button data-c="${esc(c.name)}"><span>${sp(c.name)}</span><span class="cnt">${c.micro}+${c.colony}</span></button>`).join("");
  box.hidden=false;
  box.querySelectorAll("button").forEach(b=>b.onmousedown=e=>{ e.preventDefault(); $("#q").value=b.dataset.c; search(b.dataset.c); });
}
$("#q").addEventListener("input",suggest);
$("#q").addEventListener("blur",()=>setTimeout(()=>$("#sugg").hidden=true,100));
$("#q").addEventListener("keydown",e=>{
  const bs=[...$("#sugg").querySelectorAll("button")];
  if(e.key==="ArrowDown"&&bs.length){ e.preventDefault(); sel=(sel+1)%bs.length; }
  else if(e.key==="ArrowUp"&&bs.length){ e.preventDefault(); sel=(sel-1+bs.length)%bs.length; }
  else if(e.key==="Enter"){ e.preventDefault(); const v=(!$("#sugg").hidden&&sel>=0)?bs[sel].dataset.c:$("#q").value; $("#q").value=v; search(v); return; }
  else if(e.key==="Escape"){ $("#sugg").hidden=true; return; }
  else return;
  bs.forEach((b,i)=>b.classList.toggle("on",i===sel));
});
boot();
</script></body></html>"""


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="按菌名查看训练图片（本机网页）。训练集和菌名都在网页里选择。")
    ap.add_argument("query", nargs="?", default="", help="（可选）打开时直接查看的菌名，如 Scedosporium")
    ap.add_argument("--trainset", default="", help="（可选）打开时默认选中的训练集，如 trainset_roi；网页里可随时切换")
    ap.add_argument("--root", action="append", default=[],
                    help="到哪些目录里找训练集文件夹，可重复；默认当前目录（脚本所在文件夹）")
    ap.add_argument("--base", action="append", default=[], help="清单里相对路径的额外基准目录，可重复")
    ap.add_argument("--taxonomy", default="taxonomy_map.csv", help="菌名对照表，用于中文名搜索（可选）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no_browser", action="store_true", help="不自动打开浏览器")
    a = ap.parse_args()

    roots = [str(Path(r).expanduser().resolve()) for r in (a.root or [os.getcwd()])]
    extra = [Path(b).expanduser().resolve() for b in a.base]
    lib = Library(roots, extra, a.taxonomy)

    found = discover_trainsets(roots)
    print("找到的训练集：" + ("、".join(Path(f).name for f in found) if found else "无（可在网页里填写路径）"))
    default_ts = a.trainset or ("trainset" if any(Path(f).name == "trainset" for f in found)
                                else (Path(found[0]).name if found else ""))
    if not HAS_PIL:
        print("提示：未安装 Pillow，缩略图用原图代替，.tif 图片在浏览器中可能无法显示。")

    port = a.port
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(lib, default_ts))
            break
        except OSError:
            port += 1
    else:
        sys.exit("[错误] 找不到可用端口。")
    h = urllib.parse.urlencode({k: v for k, v in (("ts", a.trainset), ("q", a.query)) if v})
    url = f"http://127.0.0.1:{port}/" + (("#" + h) if h else "")
    print(f"\n已启动：{url}\n在网页里选择训练集、输入菌名即可。按 Ctrl+C 退出。")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
