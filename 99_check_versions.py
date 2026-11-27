#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
99_check_versions.py —— 检查 Audit 里的脚本是不是最新版（编号版，2026-10-16 表格对齐版）

    python 99_check_versions.py                       # 检查全部脚本
    python 99_check_versions.py 01_build_trainset.py  # 只查某几个

两种检查：
  1) 文件指纹（md5）与发布版一致 → "最新"
  2) 指纹不同时，再查关键功能是否都在 → 区分"旧版"和"你自己改过但功能齐全"
"""
import os
import sys
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = {
    "00_audit_external.py": "0e226589254bf0b81d612f7101c04e7b",
    "00_audit_train.py": "6d5454c7601c22e3638720f2a7efe648",
    "00_build_manifest.py": "af40b1760de56986bf68d7e7b07e4c6f",
    "00_clean_train.py": "6c25c57c4a27fb16e419799156a0a0d2",
    "01_build_trainset.py": "0f251858e712431eb160409072983062",
    "01_crop_batch.py": "b2f9d65036ad66b4ec1cc14f810d5298",
    "01_crop_batch_training.py": "0e7f868ba61900e06fc5d93628b0f62e",
    "01_crop_helper.py": "a1d5bcd54a988afebd85d5ae19e939c3",
    "01_make_recrop_dirs.py": "dfecb974cd33d1666b080ee4a57b050e",
    "02_run_all.sh": "5e989dc51e8516de3ace958e71390dab",
    "02_run_all_timestamp.sh": "4875647d83e5c5e9ec5181a0d7aa7f24",
    "02_train.py": "4d82f1c0b5b70245e1a0d0c775ba33d8",
    "02b_analyze_errors.py": "546064d18bb5b5233b645b955ab88197",
    "02b_analyze_errors_batch.sh": "eae45f0afbb01548c0c71a00111412f7",
    "02b_compare_runs.py": "9830ad96c66863deae2be8fe8a151330",
    "02b_oof_by_site.py": "718ab989ee67f04f34733cf21c464a1d",
    "02b_plot_confusion.py": "fec612de4038a7f5288ebcd045985acc",
    "03_eval_external.py": "77cca0b6a16a77f3dea5451120042571",
    "03_roi_blind.py": "c853e72a592258eedafce48c7c60785a",
    "03_run_final.zsh": "b2c63eececbae22a0f45ef348a9744b5",
    "03_run_loso.zsh": "f167481b894bb0f4d1395da37194ff93",
    "03_sanity_check.py": "a8a13c0d88b0b11cd57cfbe1f3d50115",
    "04_check_roi_overlap.py": "fd83c356c2c175a82a5f79ff7482fa5e",
    "04_review_sites.py": "7ac12b7fd41f767d1bd83fb2e76a621e",
    "04_review_train.py": "e7b0fd4142a53fe54d06b917efb72d18",
    "05_roi_analysis.py": "81236c4c27c143cbc2a3c337ee11fdb9",
    "99_organize_outputs.py": "b91d3a7b31985fc407ffe39922105fdd",
    "paths.py": "0e2e75469f2be1ecda351da10792ad52"
}

# 关键功能标记：脚本里必须出现的文字（缺任何一个就说明是旧版）
FEATURES = {
    "01_build_trainset.py": {
        "--exclude_sites（按来源留一）": "exclude_sites",
        "--review 剔除不合格/局部": "load_review",
        "合格旧截图保留": "strict_replace",
        "--add_roi 外部 ROI": "add_roi",
        "按菌株分组（原图_重截）": "strain_group",
        "site 来源列": "site_of",
        "output/ 文件夹约定": "import paths as OP",
    },
    "02_train.py": {"--trainset 可写文件夹名": "find_trainset_csv", "output/ 文件夹约定": "import paths as OP"},
    "03_eval_external.py": {"--roi_dir 盲法 ROI": "roi_dir", "调用 02_train": "02_train", "output/ 文件夹约定": "import paths as OP", "中文对齐表格": "OP.table("},
    "03_run_loso.zsh": {"调用编号版脚本": "01_build_trainset.py", "output/ 文件夹": "02_training_runs/runs_loso"},
    "01_crop_batch_training.py": {"递归读菌株子文件夹": "os.walk(root)"},
    "paths.py": {"find_run": "def find_run", "place": "def place", "中文对齐表格": "def table("},
}


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    names = sys.argv[1:] or sorted(RELEASE)
    w = max(len(n) for n in names) + 2
    bad = 0
    print(f"{'脚本':<{w}}状态")
    print("-" * (w + 40))
    for n in names:
        p = os.path.join(HERE, n)
        if not os.path.exists(p):
            old = n.split("_", 1)[1] if n[:2].isdigit() or n.startswith("02b") else n
            hint = f"（找到旧名 {old}，还没换成编号版）" if os.path.exists(os.path.join(HERE, old)) else ""
            print(f"{n:<{w}}✗ 缺少{hint}")
            bad += 1
            continue
        if n in RELEASE and md5(p) == RELEASE[n]:
            print(f"{n:<{w}}✓ 最新")
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        miss = [k for k, v in FEATURES.get(n, {}).items() if v not in txt]
        if miss:
            print(f"{n:<{w}}✗ 旧版，缺少：{'、'.join(miss)}")
            bad += 1
        elif n in RELEASE:
            print(f"{n:<{w}}△ 与发布版不同（可能自己改过），关键功能齐全")
        else:
            print(f"{n:<{w}}? 不在发布清单中")
    olds = [f for f in os.listdir(HERE) if f.endswith((".py", ".sh", ".zsh")) and f not in RELEASE
            and f != os.path.basename(__file__) and not f[:2].isdigit()]
    if olds:
        print(f"\n文件夹里还有未编号的旧脚本：{'、'.join(sorted(olds))}（可移到 _old_scripts/）")
    print("\n结论：" + ("全部是最新版，可以运行。" if not bad else f"有 {bad} 个需要更新，请用最新的 Audit脚本_编号版.zip 覆盖。"))


if __name__ == "__main__":
    main()
