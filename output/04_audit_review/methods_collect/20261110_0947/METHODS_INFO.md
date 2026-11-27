# Methods 资料收集结果（2026-11-10 09:47）

- 脚本：00_methods_collect.py（00_methods_collect v1.0 (2026-11-10)，sha256 6b673d22dc9d26ad…）
- 完整命令：`00_methods_collect.py --audit /Users/chunjiang/Documents/真菌AI/Audit --out_root /Users/chunjiang/Documents/真菌AI/Audit/output --micro_model runs_v4/swin_t_full_final --micro_trainset trainset_v4 --colony_model colony/20261020_2108 --recrop_root /Users/chunjiang/Documents/真菌AI/原图_重截 --colony_raw /Users/chunjiang/Documents/真菌AI/CNN/补拍_2026 --ext_root /Users/chunjiang/Desktop/丝状真菌 --colony_python /Users/chunjiang/miniforge3/envs/ai/bin/python --clin_micro_model runs_v4/swin_t_full_final --clin_colony_model colony/20261020_2108 --out /Users/chunjiang/Documents/真菌AI/Audit/output/04_audit_review/methods_collect/20261110_0947`
- 电脑：ChunkiangMBP-M3.local


## 🧫 第 1 段：开发集株数（清单第 2 项）

ℹ️  显微数据集：trainset_v4（来自模型的 args.json）
```
   位置：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/trainset_v4
   截图总数：3121
   类别数：26
   交叉验证分组数（cv_group）：1923
```

ℹ️  按截图来源拆分（分组是否等于真实菌株，取决于来源）：
```
   来源        截图数  分组数  说明                                                              
   ──────────  ──────  ──────  ──────────────────────────────────────────────────────────────────
   N 网络补截  125     31      网络图片 → 分组 = 原图，不是本院菌株                              
   V 规范截图  1237    239     按 原图_重截/<菌种>/<菌株>/ 截取 → 分组 = 菌株（可追溯）          
   O 旧截图    1759    1653    旧截图 → 分组多为推断（文件名 / 近似重复聚类），多数不能追溯到菌株
```

ℹ️  按分组规则（group_rule）拆分：
```
   分组规则        分组数
   ──────────────  ──────
   crop_source     31    
   crop_v2_source  106   
   crop_v2_strain  133   
   date_mtime      1639  
   tile_source     14    
```

```
   可追溯到菌株的分组（规范截图 V）：239 / 1923（12%）
```
⚠️  可追溯菌株的分组不到 80%：论文里不宜直接写 isolate-level cross-validation，需写明其余按原图 / 近似重复分组
```
   按实验室（site）：PUPH 2795、WEB2 118、MCW 114、WEB1 76、HSJ 15、JYJ 3
```

ℹ️  原图_重截 文件夹（/Users/chunjiang/Documents/真菌AI/原图_重截）：
```
   菌种文件夹  菌株数  原图数
   ──────────  ──────  ──────
   土曲霉      1       1     
   多育赛多孢  7       7     
   小克银汉霉  9       9     
   尖端赛多孢  5       6     
   弯孢霉      26      26    
   拟青霉      10      27    
   杂色曲霉    33      33    
   根霉        11      13    
   毛霉        11      11    
   灰绿曲霉    4       17    
   离蠕孢      3       15    
   组织胞浆菌  11      11    
   裂褶菌      3       20    
   链格孢      4       22    
   黄曲霉      1       2     
   黑曲霉      22      22    
   合计：16 个菌种文件夹，161 株，242 张原图
```

ℹ️  菌落数据集：colony_preprocessed（来自菌落模型的 locked_config.json）
```
   节          株数（进入主分析）
   ──────────  ──────────────────
   Flavi       11                
   Fumigati    9                 
   Nidulantes  1                 
   Nigri       10                
   Terrei      6                 
   菌落株数合计：37 株（正面照 37 张；另有 5 张未进入主分析）
   同时出现在显微分组名里的菌落株号：0 个（按名字完全相同判断，命名不同会漏计）
```

ℹ️  图片拍摄时间（读 EXIF，仅供参考；显微相机常不写 EXIF，批量复制后的文件修改时间不可靠）：
```
   显微原图（原图_重截）：检查 242 张，234 张有拍摄时间：2015-07-24 至 2026-10-22；各年张数 {'2015': 1, '2016': 63, '2017': 5, '2018': 33, '2019': 19, '2021': 50, '2022': 26, '2024': 12, '2025': 1, '2026': 24}
   菌落照片（补拍_2026）：检查 51 张，7 张有拍摄时间：2025-04-01 至 2025-04-01；各年张数 {'2025': 7}
```

✍️  需人工：开发集菌株的收集起止年份：程序只能给出 EXIF 拍摄时间作参考，菌株分离年份请查 LIS 或菌种库登记
✍️  需人工：保藏菌株的来源（本院菌种库名称）
✍️  需人工：不能追溯菌株的旧截图大约对应多少株（如能从登记本估计）；否则论文只报告“分组数”

## 🗂️ 第 2 段：显微数据集 DATASET_INFO.json（清单第 5 项）

ℹ️  来源链（请人工核对这条链是否就是论文要报告的那次训练）：
```
   显微最终模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_full_final（final.pt 修改时间 2026-10-21 15:13）
   它用的数据集：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/trainset_v4（依据：模型的 args.json）
   trainset_micro.csv：修改时间 2026-10-21 10:00，sha256 2350d441ac1bedc9…
```
⚠️  这个数据集没有 DATASET_INFO.json（旧版数据集，如 trainset_v4）。以下由清单现算，另存到输出文件夹，不写回数据集
```
   💡 旧版数据集没有记录建数据集时的参数（--replace_classes、--review 等）。可在 Audit/output/06_logs 里找当时 01_build_trainset 的日志
```
✍️  需人工：确认数据集 trainset_v4 建立时用的参数（是否 --replace_classes all_v2 --review review_images.csv，纳入了哪些来源）

ℹ️  各类别截图数与分组数（Table 1 草稿；共 3121 张、26 类、1923 个分组）：
```
   类别                                         中文名                        截图  分组  O    N   V  
   ───────────────────────────────────────────  ────────────────────────────  ────  ────  ───  ──  ───
   Alternaria                                   链格孢                        343   135   237  0   106
   Aspergillus flavus (sect. Flavi)             黄曲、黄曲霉                  85    22    0    52  33 
   Aspergillus fumigatus (sect. Fumigati)       烟曲、烟曲霉                  95    25    0    0   95 
   Aspergillus glaucus (sect. Aspergillus)      灰绿曲霉                      64    4     0    0   64 
   Aspergillus nidulans (sect. Nidulantes)      构巢、构巢曲霉                38    12    0    0   38 
   Aspergillus niger (sect. Nigri)              黑曲、黑曲霉                  44    36    0    13  31 
   Aspergillus terreus (sect. Terrei)           土曲霉                        80    14    0    0   80 
   Aspergillus versicolor (sect. Versicolores)  杂色曲霉、聚多曲霉            78    31    0    0   78 
   Bipolaris                                    离蠕孢                        208   138   135  0   73 
   Coprinellus/Coprinopsis                      鬼伞菌                        181   181   181  0   0  
   Cunninghamella                               小克银汉、小克银汉霉          76    52    44   0   32 
   Curvularia                                   弯孢霉                        47    19    0    0   47 
   Fonsecaea                                    着色霉                        165   165   165  0   0  
   Fusarium                                     双孢镰刀、镰刀菌              167   16    0    0   167
   Histoplasma capsulatum                       组织胞浆菌                    205   130   120  0   85 
   Lichtheimia                                  横梗霉、犁头霉                84    34    24   60  0  
   Lomentospora prolificans                     多育赛多孢                    141   112   106  0   35 
   Mucor                                        毛霉                          14    10    1    0   13 
   Paecilomyces                                 拟青霉                        50    7     0    0   50 
   Penicillium                                  青霉                          66    66    66   0   0  
   Rhizopus                                     根霉                          26    11    0    0   26 
   Scedosporium                                 尖端赛多孢、波氏赛多孢        273   253   248  0   25 
   Schizophyllum                                裂褶菌                        108   3     0    0   108
   Scopulariopsis                               帚霉                          187   187   187  0   0  
   Sporothrix schenckii complex                 申克孢子丝菌                  245   245   245  0   0  
   Talaromyces marneffei                        马尔尼菲篮状菌、马尔尼菲青霉  51    15    0    0   51 
```
⚠️  分组少于 5 的类别（5 折交叉验证时不能每折都有）：Aspergillus glaucus (sect. Aspergillus)、Schizophyllum
```
   最多 / 最少类截图数比：24.5
```

## ⚙️ 第 3 段：显微模型训练参数（清单第 6 项）

ℹ️  最终模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_full_final
✅ args.json（该模型训练时实际使用的参数）：
```
   backbone：swin_t
   mode：full
   trainset：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/trainset_v4/trainset_micro.csv
   class_names：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/trainset_v4/class_names.json
   out_root：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs
   folds：5
   only_fold：0
   epochs：16
   patience：8
   batch：32
   lr：0.0001
   weight_decay：0.05
   img_size：224
   workers：4
   seed：42
   no_pretrained：False
   resume_dir：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_full_final
```

ℹ️  对应的交叉验证：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_cv
```
   acc：0.928 ± 0.043（各折均值 ± SD）
   macro_f1：0.889 ± 0.041（各折均值 ± SD）
   balanced_acc：0.912 ± 0.025（各折均值 ± SD）
   top2：0.948 ± 0.042（各折均值 ± SD）
   完成折数：5
   各折最佳 epoch 中位数：16.0
```
✅ 最终模型的 epochs（16）= 交叉验证最佳 epoch 中位数，与 Methods 写法一致

ℹ️  同一文件夹（runs_v4）下已完成的交叉验证（Table 2 草稿）：
```
   文件夹              折数  准确率         macro-F1       top-2  完成时间        
   ──────────────────  ────  ─────────────  ─────────────  ─────  ────────────────
   convnext_tiny_cv    5     0.875 ± 0.041  0.812 ± 0.041  0.926  2026-10-22 21:34
   densenet121_cv      5     0.930 ± 0.037  0.887 ± 0.030  0.958  2026-10-22 10:23
   efficientnet_b0_cv  5     0.929 ± 0.032  0.867 ± 0.035  0.958  2026-10-22 11:08
   resnet50_cv         5     0.917 ± 0.036  0.856 ± 0.039  0.938  2026-10-22 12:20
   swin_t_cv           5     0.928 ± 0.043  0.889 ± 0.041  0.948  2026-10-21 12:07
   vit_b_16_cv         5     0.915 ± 0.053  0.868 ± 0.038  0.939  2026-10-23 01:12
```

ℹ️  02_train.py（sha256 b9396f55a65b53f1…，修改时间 2026-10-22 08:17）中与训练设置有关的代码行：
⚠️  模型文件夹没有 RUN_RECORD.md（旧版流程训练），无法确认 02_train.py 是否与训练时相同
```
   ▸ 优化器 / 学习率 / 权重衰减
       235 │ lr = base_lr * t / warmup if t < warmup else 0.5 * base_lr * (1 + math.cos(math.pi * (t - warmup) / max(epochs - warmup, 1e-6)))
       248 │ opt = torch.optim.AdamW([{"params": body, "lr_scale": 1.0}, {"params": head, "lr_scale": 10.0}],
       249 │ lr=a.lr, weight_decay=a.weight_decay)
       262 │ lr = cosine_lr(opt, a.lr, ep, step, len(dl_tr), a.epochs)
       270 │ row = dict(epoch=ep + 1, train_loss=tot / max(n, 1), lr=lr, seconds=round(time.time() - t0, 1))
       311 │ ap.add_argument("--lr", type=float, default=1e-4)
       312 │ ap.add_argument("--weight_decay", type=float, default=0.05)
   ▸ 学习率调度
       233 │ def cosine_lr(opt, base_lr, epoch, step, steps_per_epoch, epochs, warmup=1):
       235 │ lr = base_lr * t / warmup if t < warmup else 0.5 * base_lr * (1 + math.cos(math.pi * (t - warmup) / max(epochs - warmup, 1e-6)))
   ▸ 损失函数
       250 │ crit = nn.CrossEntropyLoss(label_smoothing=0.1)
   ▸ 类别不平衡
        42 │ from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
       201 │ def sample_weights(df, num_classes):
       252 │ sampler = WeightedRandomSampler(sample_weights(df_tr, num_classes), num_samples=len(df_tr), replacement=True)
   ▸ 数据增强
       141 │ class RandomJPEG:
       149 │ img.save(buf, format="JPEG", quality=random.randint(*self.q))
       161 │ return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
       165 │ norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
       171 │ transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.8, 1.25)),   # 多尺度：随机放大局部
       172 │ transforms.RandomHorizontalFlip(),
       173 │ transforms.RandomVerticalFlip(),
       174 │ transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
       177 │ RandomJPEG(p=0.25),
   ▸ 早停 / 选最佳 epoch
       287 │ if dl_va is not None and bad >= a.patience:
       288 │ print(f"{log_prefix}早停：{a.patience} 个 epoch 无提升")
       309 │ ap.add_argument("--patience", type=int, default=8)
       378 │ best_ep = json.load(open(os.path.join(fdir, "fold.json")))["best_epoch"]
       385 │ json.dump({"best_epoch": best_ep}, open(os.path.join(fdir, "fold.json"), "w"))
       389 │ m.update(fold=k, best_epoch=best_ep, n_val=len(va))
       402 │ "pooled_oof": pooled, "median_best_epoch": float(fr["best_epoch"].median())}
       418 │ print(f"  最佳 epoch 中位数 {summary['median_best_epoch']:.0f}（full 模式可用 --epochs 设为这个值）")
   ▸ 预训练权重 / 微调范围
        57 │ def build_model(name, num_classes, pretrained=True):
        58 │ W = "DEFAULT" if pretrained else None
       243 │ model = build_model(a.backbone, num_classes, pretrained=not a.no_pretrained).to(device)
       316 │ ap.add_argument("--no_pretrained", action="store_true", help="不加载 ImageNet 权重（调试用）")
   ▸ 随机种子 / 设备
        86 │ if torch.backends.mps.is_available():
        87 │ return torch.device("mps")
        88 │ if torch.cuda.is_available():
        89 │ return torch.device("cuda")
       315 │ ap.add_argument("--seed", type=int, default=42)
       322 │ torch.manual_seed(a.seed)
   ▸ 交叉验证划分
        45 │ from sklearn.model_selection import StratifiedGroupKFold
       364 │ sgkf = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
```
✅ 02_train.py 已复制到 copies/，Claude 可据此补全 Methods 中的优化器、学习率、损失函数和增强幅度

## 🎨 第 4 段：菌落模型（清单第 8 项）

ℹ️  菌落模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/colony/20261020_2108
```
   锁定时间：2026-10-20T21:08:58
   训练脚本版本：02_colony_train v1.1 (2026-10-20)
   数据集：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/colony_preprocessed
   纳入的节：Fumigati、Flavi、Nigri、Terrei
   株数 / CV 组数：36 株 / 33 组（留一组交叉验证）
   选定方案：A（规则：lowest OOF strain-level NLL among A and B）
   温度 T：1.150
   正则强度 C（最终模型）：3
   方案 B 骨干网络：swin_t
   方案 C（端到端微调）：未跑
   min_strains / dup_inliers：3 / 300
```

ℹ️  各方案的株级交叉验证结果：
```
   方案        top1   top2   平衡准确率  NLL    ECE  
   ──────────  ─────  ─────  ──────────  ─────  ─────
   A           0.944  1.000  0.950       0.268  0.061
   A_无直径    0.944  1.000  0.950       0.299  0.059
   A_仅中心块  0.944  1.000  0.933       0.185  0.028
   B           0.861  0.972  0.825       0.335  0.059
   B_仅中心块  0.861  0.944  0.822       0.461  0.052
```

ℹ️  每节株数（来自 oof_colony_selected_calibrated.csv）：
```
   节        株数
   ────────  ────
   Flavi     11  
   Fumigati  9   
   Nigri     10  
   Terrei    6   
```
⚠️  菌落模型不含：Nidulantes（株数不足 3 被剔除）。Methods 的融合规则已写明“镜检首位节不在菌落模型内时不融合”
✍️  需人工：决定是否补够 Nidulantes 菌株后重训菌落模型；若不补，论文需写明菌落模型只覆盖 4 个节
⚠️  株数少于 10 的节：Fumigati（9 株）、Terrei（6 株）
✅ final_model.joblib 指纹与锁定记录一致（a0bfde8db2feb302…）

ℹ️  metrics.txt 摘录：
```
   │ 正面照来自 42 株；主分析排除 5 株（PUPH-FUM-006, PUPH-NID-002, PUPH-NID-003, PUPH-NID-004, PUPH-TER-002）
   │ 株数不足、本轮不纳入的类别：Nidulantes（1 株）
   │ 疑似同一平皿，合并为同一 CV 组（不改株号）：PUPH-FLA-001+PUPH-FLA-005(1914)，PUPH-FLA-007+PUPH-FLA-009(1291)，PUPH-TER-003+PUPH-TER-004(569)
   │ 纳入类别与株数：Fumigati 9，Flavi 11，Nigri 10，Terrei 6；共 36 株、33 个 CV 组
   │ ===== 选择与校准 =====
   │ OOF NLL：A 0.268，B 0.335 → 选定方案 A
   │ 温度缩放 T = 1.150；校准前 NLL 0.268、ECE 0.061 → 校准后 NLL 0.264、ECE 0.047
   │ （T 在 OOF 预测上拟合；前瞻验证前随模型一并锁定）
   │ 最终模型：final_model.joblib（sha256 a0bfde8db2feb302…）
```

✅ 菌落模型与 clin_config.zsh（临床验证用）相同：colony/20261020_2108
✅ 显微模型与 clin_config.zsh（临床验证用）相同：runs_v4/swin_t_full_final

## 💻 第 5 段：软件版本与电脑型号（清单第 10 项）

ℹ️  电脑：
```
   型号：MacBook Pro
   型号标识：Mac15,6
   芯片：Apple M3 Pro
   核心数：12 (6 performance and 6 efficiency)
   内存：36 GB
   系统：macOS 15.6.1
```

ℹ️  显微用 python（fungal）：Python 3.10.20（/Users/chunjiang/miniforge3/envs/fungal/bin/python）
```
   包             版本     
   ─────────────  ─────────
   torch          2.2.2    
   torchvision    0.17.2   
   numpy          1.26.4   
   pandas         2.3.3    
   scikit-learn   1.7.2    
   scipy          1.15.3   
   opencv-python  4.13.0.92
   pillow         12.2.0   
   pillow-heif    1.8.0    
   matplotlib     3.10.8   
   joblib         1.5.3    
   计算设备：Apple MPS 可用
```
✅ conda list 已保存：copies/conda_list_micro.txt

ℹ️  菌落用 python（COLONY_PYTHON）：Python 3.11.14（/Users/chunjiang/miniforge3/envs/ai/bin/python）
```
   包                      版本     
   ──────────────────────  ─────────
   torch                   2.9.1    
   torchvision             0.24.1   
   numpy                   1.26.4   
   pandas                  3.0.1    
   scikit-learn            1.8.0    
   scipy                   1.16.3   
   opencv-python           4.11.0.86
   opencv-python-headless  4.10.0.84
   pillow                  12.1.1   
   pillow-heif             1.8.1    
   matplotlib              3.10.8   
   joblib                  1.5.3    
   计算设备：Apple MPS 可用
```
✅ conda list 已保存：copies/conda_list_colony.txt
ℹ️  两个环境的 scikit-learn 版本不同（1.7.2 / 1.8.0）：论文里菌落模型报告 COLONY_PYTHON 环境的版本

## 🏥 第 6 段：回顾性外部图像集的来源（清单第 11 项）

```
   这一项问的是：外部验证用的图片来自哪几家医院的实验室。
   意义：① 论文说“多来源 / 多中心外部验证”，审稿人要看到独立机构的数量，一家医院的 3 位老师只算 1 个来源；
         ② 留一来源验证（LOSO）以“来源”为单位，同一家医院的两个文件夹应合并成一个来源，否则高估独立性；
         ③ 投稿者需要写进致谢（或作者），期刊要求征得本人同意。
```

⚠️  医院名称不在任何文件里，程序无法获得。下面只能列出各来源的图片数，并生成一张待填表
ℹ️  外部集总表：/Users/chunjiang/Documents/真菌AI/Audit/output/04_audit_review/ext_audit/manifest.csv（1015 行，修改时间 2026-10-09 15:21）
```
   列名：folder、rel_path、code、category、modality_guess、label_hint、source_type、width、height、blur_var、mean_brightness、read_error、low_res、maybe_blurry、dup_within_ext、near_dup_in_train、code_seen_in_train、folder_in_train、species_ref、isolate_id、centre、ref_method、modality_confirmed、include、exclude_reason、species_cn、label_source、latin、train_class、tax_note、role
```

ℹ️  按用途（role）：
```
   role             图片数
   ───────────────  ──────
   external_micro   232   
   external_colony  177   
   out_of_panel     162   
   dev_source       122   
   no_answer        108   
   dev_set_checked  105   
   excluded_direct  66    
   dev_cropped      32    
   excluded_folder  11    
```

ℹ️  按来源（列 folder）：
```
   folder                                    合计  external_micro  external_colony  out_of_panel
   ────────────────────────────────────────  ────  ──────────────  ───────────────  ────────────
   HSJ-C_10pics✅                             12    0               0                0           
   eXPORT-2025-02-13                         39    0               0                0           
   已整理                                    83    0               0                0           
   李一凡王燕闫津津王静老师真菌图_41pics     41    8               11               0           
   李艳老师真菌图_27pics                     27    5               0                1           
   申克孢子丝_20pics✅                        20    0               0                0           
   真菌-广药附属第一医院王若帆_12pic❌        11    0               0                0           
   真菌AI----谭盼_100pics                    100   28              18               54          
   真菌图片-马琼琼_28pics                    28    17              7                4           
   真菌图片（田蕴琦）_15pics                 15    7               5                1           
   真菌图（麦老师、成老师、王老师）_56pics✅  55    0               0                0           
   细菌真菌图片-李珊_48pics                  48    0               0                0           
   网络真菌图2_234pics                       233   82              51               30          
   网络真菌图片_243pics                      242   72              69               66          
   胡音音-南阳市中心医院-真菌AI_44pics       43    13              16               6           
   蒋宇杰农垦医院_18pics✅                    18    0               0                0           
```
✅ 已复制 LOSO 汇总（ext_loso/汇总.txt）
✅ 待填表已生成：/Users/chunjiang/Documents/真菌AI/Audit/output/04_audit_review/methods_collect/20261110_0947/external_sources_待填.csv
✍️  需人工：填写 external_sources_待填.csv：每个来源对应的医院全称、城市、是否与其他来源同属一家医院、是否网络图片

## 📋 汇总

来源（请人工核对这就是论文要报告的模型）：
```
   🔬 显微最终模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_full_final
   🗂️ 显微数据集：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/trainset_v4
   📊 显微交叉验证：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/runs_v4/swin_t_cv
   🎨 菌落模型：/Users/chunjiang/Documents/真菌AI/Audit/output/02_training_runs/colony/20261020_2108
   🧫 菌落数据集：/Users/chunjiang/Documents/真菌AI/Audit/output/01_trainsets/colony_preprocessed
```

各段完成情况：
```
   ✅ 第 1 段  开发集株数
   ✅ 第 2 段  显微数据集信息
   ✅ 第 3 段  显微训练参数
   ✅ 第 4 段  菌落模型
   ✅ 第 5 段  软件版本与电脑型号
   ✍️ 第 6 段  外部图像集来源：图片数已统计，医院名称需人工填写
```

✍️  仍需人工处理 6 项（程序无法获得）：
```
   1. 开发集菌株的收集起止年份：程序只能给出 EXIF 拍摄时间作参考，菌株分离年份请查 LIS 或菌种库登记
   2. 保藏菌株的来源（本院菌种库名称）
   3. 不能追溯菌株的旧截图大约对应多少株（如能从登记本估计）；否则论文只报告“分组数”
   4. 确认数据集 trainset_v4 建立时用的参数（是否 --replace_classes all_v2 --review review_images.csv，纳入了哪些来源）
   5. 决定是否补够 Nidulantes 菌株后重训菌落模型；若不补，论文需写明菌落模型只覆盖 4 个节
   6. 填写 external_sources_待填.csv：每个来源对应的医院全称、城市、是否与其他来源同属一家医院、是否网络图片
```
