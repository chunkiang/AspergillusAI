# 🧫 AspergillusAI  
### Automatic identification of clinically important *Aspergillus* species by AI-based image recognition

---

## 📘 Introduction | 项目简介
**Background:** Rapid identification of clinically important microbial pathogens is essential for timely treatment. Conventional morphological examination of colony and microscopic images is accurate but labor-intensive. This study evaluates a dual-input convolutional neural network (CNN) architecture that integrates both image modalities for species-level classification.

**Methods:** Fifty-one classes of pathogens, including filamentous fungi and bacteria, were imaged under standardized conditions. Paired colony and microscopic images were processed by seven CNN backbones—EfficientNetB0, EfficientNetV2, ResNet50, DenseNet121, InceptionV3, VGG16, and MobileNetV2—embedded in an identical dual-input architecture. Models were trained with identical hyperparameters and data augmentation on the Google Colab Pro platform (NVIDIA T4 GPU, 16 GB VRAM; Python 3.11, TensorFlow 2.18) and evaluated using test accuracy, macro F1-score, confusion matrices, and ROC-AUC.

**Results:** EfficientNetV2 achieved the highest test accuracy (0.952) and macro F1-score (0.947) with ~14 minutes of training, followed by EfficientNetB0 and ResNet50. EfficientNet-based models converged rapidly (>85% peak validation accuracy within 10 epochs) and maintained stable loss curves. MobileNetV2 and VGG16 showed underfitting and high validation loss. ROC analysis demonstrated micro-average AUC values >0.95 for top models. Learning rate reductions preceded validation loss decreases, with model-specific lags.

**Conclusions:** The dual-input CNN effectively integrates multimodal morphological features for accurate, efficient classification of clinically important pathogens. EfficientNetV2 and EfficientNetB0 offer the best balance of performance and efficiency, supporting potential integration into automated diagnostic workflows.


---

## 🧠 Core Features | 核心特征
- 🔬 双输入卷积神经网络（Microscope + Colony）
- 🧩 可选主干网络：`EfficientNetB0`, `ResNet50`, `DenseNet121`, `InceptionV3` 等
- ⚙️ 自动数据集构建与平衡采样
- 📊 自动绘制混淆矩阵、分类报告、训练曲线
- 🧮 GPU 加速、混合精度训练（Mixed Precision）
- 💾 模型检查点、日志与训练记录自动保存

---

以下为本项目的文件与目录结构说明：

```
AspergillusAI/
├── 00_运行环境搭建.py        # 构建数据集目录结构（microscope + normal）
├── 01_主程序.py              # 主训练脚本，包含模型构建与评估
├── dataset/                  # 训练数据集（自动生成）
├── testdata/                 # 测试数据集（自动生成）
├── drive/MyDrive/models/     # 模型保存路径（Colab 环境下）
├── README.md
└── requirements.txt          # 可选：列出依赖包
```

---

✅ **说明：**
- `00_运行环境搭建.py`：自动构建训练与测试数据集的文件结构。
- `01_主程序.py`：用于训练与评估双输入卷积神经网络模型。
- `dataset/` 与 `testdata/`：运行脚本后自动生成。
- `drive/MyDrive/models/`：模型与日志在 Google Colab 环境下的保存位置。
- `requirements.txt`：可选，列出 Python 环境依赖包。

