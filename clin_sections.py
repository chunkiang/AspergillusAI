# -*- coding: utf-8 -*-
"""
clin_sections.py —— 临床验证共用的小工具（00_clin_check.py 与 03_clin_fusion.py 都导入它）

· 曲霉 种 → 组（section）的对照：镜检模型给的是"种"，菌落模型给的是"组"，融合前要把镜检概率按组相加
· 读 CSV（兼容 Excel 存成 UTF-8 或 GBK 的情况）
· Clopper–Pearson 95% CI

对照表依据 Houbraken et al. 2020（Stud Mycol 95）；Versicolores 已并入 Nidulantes。
只列本期菌落模型覆盖的 5 个组；其他曲霉（如 A. glaucus，sect. Aspergillus）→ "Other"。
要增加菌种，直接在 SECTION_OF 里加一行（改动后运行记录里的 sha256 会变，可追溯）。
"""
import math
import re

import pandas as pd

SECTIONS = ["Fumigati", "Flavi", "Nigri", "Terrei", "Nidulantes"]

SECTION_OF = {
    # Fumigati
    "fumigatus": "Fumigati", "lentulus": "Fumigati", "udagawae": "Fumigati", "fumigatiaffinis": "Fumigati",
    "novofumigatus": "Fumigati", "hiratsukae": "Fumigati", "thermomutatus": "Fumigati", "fischeri": "Fumigati",
    # Flavi
    "flavus": "Flavi", "oryzae": "Flavi", "parasiticus": "Flavi", "tamarii": "Flavi", "nomius": "Flavi",
    "sojae": "Flavi", "alliaceus": "Flavi",
    # Nigri
    "niger": "Nigri", "tubingensis": "Nigri", "welwitschiae": "Nigri", "awamori": "Nigri",
    "brasiliensis": "Nigri", "japonicus": "Nigri", "aculeatus": "Nigri", "carbonarius": "Nigri",
    "uvarum": "Nigri", "costaricaensis": "Nigri",
    # Terrei
    "terreus": "Terrei", "citrinoterreus": "Terrei", "alabamensis": "Terrei", "hortai": "Terrei",
    "carneus": "Terrei", "niveus": "Terrei",
    # Nidulantes（含原 Versicolores）
    "nidulans": "Nidulantes", "versicolor": "Nidulantes", "sydowii": "Nidulantes", "creber": "Nidulantes",
    "protuberus": "Nidulantes", "unguis": "Nidulantes", "quadrilineatus": "Nidulantes",
    "latus": "Nidulantes", "jensenii": "Nidulantes", "tabacinus": "Nidulantes",
}


def is_aspergillus(name) -> bool:
    return str(name or "").strip().lower().replace("_", " ").startswith("aspergillus")


def section_of(name) -> str:
    """'Aspergillus fumigatus' / 'Aspergillus_fumigatus' / 'Aspergillus fumigatus complex' → 'Fumigati'；
    其他曲霉 → 'Other'；非曲霉 → ''"""
    s = str(name or "").strip().lower().replace("_", " ")
    if not s.startswith("aspergillus"):
        return ""
    toks = re.split(r"[\s.]+", s)
    for t in toks[1:]:
        if t in SECTION_OF:
            return SECTION_OF[t]
        if t.startswith("sect") or t in {"section", "complex", "sp", "spp", "species"}:
            continue
    # "Aspergillus section Fumigati" 这种直接写组名的
    for sec in SECTIONS:
        if sec.lower() in toks:
            return sec
    return "Other"


def read_csv_any(path):
    last = None
    for enc in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError as e:
            last = e
    raise last


def cp_ci(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return (lo, hi)
    except ImportError:
        z, p = 1.959964, k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    if n == 0:
        return "—（n=0）"
    lo, hi = cp_ci(k, n)
    return f"{k / n:.3f}（{k}/{n}，95% CI {lo:.3f}–{hi:.3f}）"
