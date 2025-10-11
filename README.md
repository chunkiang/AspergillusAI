# 🧫 AspergillusAI  
### Automatic identification of clinically important *Aspergillus* species by AI-based image recognition

---

## 📘 Introduction | 项目简介

This repository contains the implementation code for the research paper:  
> **Automatic identification of clinically important *Aspergillus* species by artificial intelligence-based image recognition: proof-of-concept study**  

本项目为论文《基于人工智能图像识别的临床重要 *Aspergillus* 种自动鉴定：概念验证研究》的开源实现。  
研究通过卷积神经网络（CNN）对菌落宏观照片与显微镜图像进行自动识别，实现临床常见 *Aspergillus* 种（如 *A. fumigatus*, *A. flavus*, *A. niger*, *A. terreus*）的自动分类。

---

## 🧠 Core Features | 核心特征
- 🔬 双输入卷积神经网络（Microscope + Colony）
- 🧩 可选主干网络：`EfficientNetB0`, `ResNet50`, `DenseNet121`, `InceptionV3` 等
- ⚙️ 自动数据集构建与平衡采样
- 📊 自动绘制混淆矩阵、分类报告、训练曲线
- 🧮 GPU 加速、混合精度训练（Mixed Precision）
- 💾 模型检查点、日志与训练记录自动保存

---

## 🏗️ Project Structure | 项目结构

AspergillusAI/
├── 00_运行环境搭建.py # 构建数据集目录结构（microscope + normal）
├── 01_主程序.py # 主训练脚本，包含模型构建与评估
├── dataset/ # 训练数据集（自动生成）
├── testdata/ # 测试数据集（自动生成）
├── drive/MyDrive/models/ # 模型保存路径（Colab 环境下）
├── README.md
└── requirements.txt # 可选：列出依赖包
