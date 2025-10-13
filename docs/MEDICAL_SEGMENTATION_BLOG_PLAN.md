# 医学影像分割网络系列博客规划

## 📋 总体方案

### 目标
撰写一系列高质量博客文章，系统介绍医学影像分割领域的经典网络，从基础到前沿，覆盖主要技术路线。

### 系列特色
- ✅ **时间脉络清晰**：按技术演进顺序组织
- ✅ **数学严谨**：包含完整的数学定义和公式推导
- ✅ **代码可复现**：提供官方代码库链接和关键实现
- ✅ **技术关联**：明确各网络之间的继承和创新关系
- ✅ **实用导向**：包含应用场景和性能对比

---

## 🗺️ 技术发展路线图

### 第一阶段：基础架构时代 (2015-2016)
**核心特征**：全卷积网络、编码器-解码器结构

```
FCN (2015) ────┐
               ├──→ 全卷积思想
UNet (2015) ───┤
               └──→ Skip Connection
                    
V-Net (2016) ──────→ 3D扩展
```

**技术关键词**：
- Fully Convolutional Networks
- Encoder-Decoder Architecture
- Skip Connections
- Upsampling (Transposed Convolution)

---

### 第二阶段：UNet改进时代 (2017-2020)
**核心特征**：注意力机制、密集连接、多尺度融合

```
UNet (2015)
    │
    ├──→ ResUNet (2017) ───────→ 残差连接
    │
    ├──→ Attention UNet (2018) ─→ 注意力门控
    │
    ├──→ UNet++ (2018) ─────────→ 嵌套Skip连接
    │
    └──→ UNet 3+ (2020) ────────→ 全尺度融合
```

**技术关键词**：
- Residual Connections
- Attention Gates
- Nested Skip Connections
- Deep Supervision
- Multi-scale Feature Fusion

---

### 第三阶段：Transformer革命 (2021-2022)
**核心特征**：自注意力、长程依赖、混合架构

```
Vision Transformer (2020)
    │
    ├──→ TransUNet (2021) ──→ Transformer作为Encoder
    │
    ├──→ Swin-UNet (2021) ──→ Shifted Window Attention
    │
    ├──→ UNETR (2021) ──────→ 纯Transformer Encoder
    │
    ├──→ nnFormer (2021) ───→ 3D医学图像
    │
    └──→ MedFormer (2022) ───→ 轻量化医学Transformer
```

**技术关键词**：
- Self-Attention Mechanism
- Patch Embedding
- Position Encoding
- Window-based Attention
- Hybrid CNN-Transformer

---

### 第四阶段：基础模型时代 (2023-至今)
**核心特征**：大规模预训练、提示学习、零样本/少样本

```
SAM (2023) ─────────────→ Segment Anything Model
    │
    ├──→ MedSAM (2023) ──────→ 医学领域微调
    │
    ├──→ SAM-Med2D (2023) ───→ 2D医学图像优化
    │
    ├──→ SAM-Med3D (2023) ───→ 3D医学图像扩展
    │
    └──→ MedicalSAM (2024) ──→ 多模态医学SAM
```

**技术关键词**：
- Foundation Models
- Prompt Engineering
- Zero-shot Learning
- Few-shot Learning
- Interactive Segmentation

---

### 特殊分支：自适应框架
```
nnU-Net (2018) ──────────→ 自动配置分割框架
    │
    └──→ nnU-Net v2 (2022) ─→ 改进版本
```

**技术关键词**：
- Automatic Configuration
- Self-adapting
- Best Practices

---

## 📚 博客系列规划

### 系列1：基础篇 - 医学分割的奠基之作

#### 博客1: FCN与UNet - 全卷积网络的诞生 (2015)
**文件名**: `2025-02-01-fcn-unet-foundation.md`
**发布日期**: 2025-02-01

**内容大纲**：
1. **引言**
   - 医学图像分割的挑战
   - 从分类到分割的演进

2. **FCN: Fully Convolutional Networks**
   - 核心思想：去除全连接层
   - 数学定义：
     - 转置卷积 (Transposed Convolution)
     - 上采样策略
   - 网络架构
   - 论文：[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
   - 代码：[官方实现](https://github.com/shelhamer/fcn.berkeleyvision.org)

3. **UNet: 医学图像分割的里程碑**
   - 核心创新：对称的U型结构 + Skip Connections
   - 数学定义：
     - 编码器-解码器公式
     - Skip Connection的数学表示
     - 损失函数（Dice Loss, Cross Entropy）
   - 网络架构详解
   - 为什么适合医学图像？
     - 小样本学习
     - 数据增强策略
   - 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
   - 代码：
     - [官方TensorFlow实现](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
     - [PyTorch实现](https://github.com/milesial/Pytorch-UNet)

4. **性能对比与应用**
   - 在医学数据集上的表现
   - 典型应用场景

5. **总结与展望**

---

#### 博客2: V-Net - 3D医学图像分割的突破 (2016)
**文件名**: `2025-02-05-vnet-3d-segmentation.md`
**发布日期**: 2025-02-05

**内容大纲**：
1. **从2D到3D的挑战**
2. **V-Net核心创新**
   - 3D卷积
   - Residual Connections
   - Dice Loss
3. **数学定义**
   - 3D卷积公式
   - Dice Loss推导
   - 残差块定义
4. **网络架构**
5. **论文与代码**
   - 论文：[V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
   - 代码：[PyTorch实现](https://github.com/mattmacy/vnet.pytorch)

---

### 系列2：进阶篇 - UNet的演进与改进

#### 博客3: Attention UNet - 注意力机制的引入 (2018)
**文件名**: `2025-02-10-attention-unet.md`
**发布日期**: 2025-02-10

**内容大纲**：
1. **注意力机制简介**
2. **Attention Gates**
   - 核心思想
   - 数学定义
   - 门控机制公式
3. **网络架构**
4. **与标准UNet对比**
5. **论文与代码**
   - 论文：[Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999)
   - 代码：[官方实现](https://github.com/ozan-octopus/attention-unet)

---

#### 博客4: UNet++ 和 UNet 3+ - 密集连接的力量
**文件名**: `2025-02-15-unet-plus-series.md`
**发布日期**: 2025-02-15

**内容大纲**：
1. **UNet++: 嵌套Skip连接**
   - 核心创新：Dense Skip Connections
   - 数学定义
   - Deep Supervision
   - 论文：[UNet++: A Nested U-Net Architecture](https://arxiv.org/abs/1807.10165)
   - 代码：[官方实现](https://github.com/MrGiovanni/UNetPlusPlus)

2. **UNet 3+: 全尺度特征融合**
   - 核心创新：Full-scale Skip Connections
   - 数学定义
   - 论文：[UNet 3+: A Full-Scale Connected UNet](https://arxiv.org/abs/2004.08790)
   - 代码：[PyTorch实现](https://github.com/ZJUGiveLab/UNet-Version)

3. **对比分析**

---

### 系列3：Transformer篇 - 自注意力的革命

#### 博客5: TransUNet - CNN与Transformer的融合 (2021)
**文件名**: `2025-02-20-transunet-hybrid-architecture.md`
**发布日期**: 2025-02-20

**内容大纲**：
1. **Transformer在视觉领域的应用**
2. **TransUNet架构**
   - CNN Encoder
   - Transformer作为Bottleneck
   - CNN Decoder
3. **数学定义**
   - Multi-Head Self-Attention
   - Position Encoding
   - 混合架构公式
4. **论文与代码**
   - 论文：[TransUNet: Transformers Make Strong Encoders](https://arxiv.org/abs/2102.04306)
   - 代码：[官方实现](https://github.com/Beckschen/TransUNet)

---

#### 博客6: Swin-UNet - 层级化视觉Transformer (2021)
**文件名**: `2025-02-25-swin-unet-hierarchical-transformer.md`
**发布日期**: 2025-02-25

**内容大纲**：
1. **Swin Transformer简介**
2. **Swin-UNet架构**
   - Shifted Window Attention
   - Patch Merging
   - 层级化特征
3. **数学定义**
   - Window-based Attention
   - Shifted Window机制
   - 相对位置编码
4. **论文与代码**
   - 论文：[Swin-UNet: Unet-like Pure Transformer](https://arxiv.org/abs/2105.05537)
   - 代码：[官方实现](https://github.com/HuCaoFighting/Swin-Unet)

---

#### 博客7: UNETR 和 nnFormer - 纯Transformer架构
**文件名**: `2025-03-01-unetr-nnformer-pure-transformer.md`
**发布日期**: 2025-03-01

**内容大纲**：
1. **UNETR: 纯Transformer Encoder**
   - 3D Patch Embedding
   - 论文：[UNETR: Transformers for 3D Medical Image Segmentation](https://arxiv.org/abs/2103.10504)
   - 代码：[MONAI实现](https://github.com/Project-MONAI/research-contributions)

2. **nnFormer: 3D医学图像的Transformer**
   - 论文：[nnFormer: Interleaved Transformer](https://arxiv.org/abs/2109.03201)
   - 代码：[官方实现](https://github.com/282857341/nnFormer)

---

### 系列4：基础模型篇 - SAM与医学应用

#### 博客8: SAM - Segment Anything Model (2023)
**文件名**: `2025-03-05-sam-segment-anything.md`
**发布日期**: 2025-03-05

**内容大纲**：
1. **基础模型的概念**
2. **SAM架构**
   - Image Encoder (ViT)
   - Prompt Encoder
   - Mask Decoder
3. **数学定义**
   - Prompt Learning
   - Multi-scale Features
4. **论文与代码**
   - 论文：[Segment Anything](https://arxiv.org/abs/2304.02643)
   - 代码：[官方实现](https://github.com/facebookresearch/segment-anything)

---

#### 博客9: MedSAM系列 - SAM的医学改进
**文件名**: `2025-03-10-medsam-medical-adaptation.md`
**发布日期**: 2025-03-10

**内容大纲**：
1. **SAM在医学领域的挑战**
2. **MedSAM**
   - 医学数据微调
   - 论文：[Segment Anything in Medical Images](https://arxiv.org/abs/2304.12306)
   - 代码：[官方实现](https://github.com/bowang-lab/MedSAM)

3. **SAM-Med2D**
   - 2D医学图像优化
   - 论文：[SAM-Med2D](https://arxiv.org/abs/2308.16184)
   - 代码：[官方实现](https://github.com/OpenGVLab/SAM-Med2D)

4. **性能对比与应用**

---

### 系列5：实用框架篇

#### 博客10: nnU-Net - 自适应医学分割框架 (2018-2022)
**文件名**: `2025-03-15-nnunet-self-configuring-framework.md`
**发布日期**: 2025-03-15

**内容大纲**：
1. **nnU-Net的哲学**
   - 自动配置
   - 最佳实践集合
2. **核心组件**
   - 数据预处理
   - 网络架构自适应
   - 训练策略
   - 后处理
3. **数学定义与实现**
4. **论文与代码**
   - 论文v1：[nnU-Net: Self-adapting Framework](https://arxiv.org/abs/1809.10486)
   - 论文v2：[nnU-Net Revisited](https://arxiv.org/abs/2106.06858)
   - 代码：[官方实现](https://github.com/MIC-DKFZ/nnUNet)

---

## 📊 网络对比总表

| 网络名称 | 年份 | 核心创新 | 维度 | Dice (示例) | 官方代码 |
|---------|------|---------|------|------------|---------|
| FCN | 2015 | 全卷积 | 2D | - | [链接](https://github.com/shelhamer/fcn.berkeleyvision.org) |
| UNet | 2015 | Skip连接 | 2D | 0.92 | [链接](https://github.com/milesial/Pytorch-UNet) |
| V-Net | 2016 | 3D+Dice Loss | 3D | 0.89 | [链接](https://github.com/mattmacy/vnet.pytorch) |
| Attention UNet | 2018 | 注意力门控 | 2D | 0.93 | [链接](https://github.com/ozan-octopus/attention-unet) |
| UNet++ | 2018 | 密集Skip | 2D/3D | 0.94 | [链接](https://github.com/MrGiovanni/UNetPlusPlus) |
| nnU-Net | 2018 | 自适应框架 | 2D/3D | 0.95+ | [链接](https://github.com/MIC-DKFZ/nnUNet) |
| UNet 3+ | 2020 | 全尺度融合 | 2D | 0.94 | [链接](https://github.com/ZJUGiveLab/UNet-Version) |
| TransUNet | 2021 | CNN+Transformer | 2D | 0.94 | [链接](https://github.com/Beckschen/TransUNet) |
| Swin-UNet | 2021 | Shifted Window | 2D | 0.95 | [链接](https://github.com/HuCaoFighting/Swin-Unet) |
| UNETR | 2021 | 纯Transformer | 3D | 0.93 | [链接](https://github.com/Project-MONAI/research-contributions) |
| nnFormer | 2021 | 3D Transformer | 3D | 0.94 | [链接](https://github.com/282857341/nnFormer) |
| SAM | 2023 | 基础模型 | 2D | - | [链接](https://github.com/facebookresearch/segment-anything) |
| MedSAM | 2023 | 医学SAM | 2D | 0.90 | [链接](https://github.com/bowang-lab/MedSAM) |
| SAM-Med2D | 2023 | 2D医学优化 | 2D | 0.92 | [链接](https://github.com/OpenGVLab/SAM-Med2D) |

---

## 🎨 博客统一模板

每篇博客将包含以下标准部分：

### 1. Front Matter
```yaml
---
layout: post
title: "网络名称 - 副标题"
date: YYYY-MM-DD
categories: [医学影像, 图像分割]
tags: [深度学习, UNet系列/Transformer/SAM, 医学AI]
excerpt: "简短摘要（2-3句话）"
author: YangCazz
math: true
---
```

### 2. 文章结构
1. **引言** - 背景与动机
2. **核心思想** - 主要创新点
3. **网络架构** - 详细结构
4. **数学定义** - 公式推导
5. **实现细节** - 代码解析
6. **实验结果** - 性能分析
7. **应用场景** - 实际案例
8. **总结** - 优缺点与展望
9. **参考资料** - 论文、代码、扩展阅读

### 3. 图片资源
- 网络架构图
- 关键模块示意图
- 实验结果可视化
- 性能对比图表

---

## 📅 发布时间表

| 博客编号 | 标题 | 计划发布日期 |
|---------|------|-------------|
| 1 | FCN与UNet - 基础 | 2025-02-01 |
| 2 | V-Net - 3D分割 | 2025-02-05 |
| 3 | Attention UNet | 2025-02-10 |
| 4 | UNet++/UNet 3+ | 2025-02-15 |
| 5 | TransUNet | 2025-02-20 |
| 6 | Swin-UNet | 2025-02-25 |
| 7 | UNETR/nnFormer | 2025-03-01 |
| 8 | SAM | 2025-03-05 |
| 9 | MedSAM系列 | 2025-03-10 |
| 10 | nnU-Net | 2025-03-15 |

---

## 🔧 技术准备

### 数学公式渲染
- ✅ 已配置MathJax 3
- ✅ 支持行内公式 `\( ... \)`
- ✅ 支持块级公式 `$$ ... $$`

### 代码高亮
- ✅ 已配置Rouge语法高亮
- ✅ 支持Python、YAML等
- ✅ 一键复制功能

### 图片管理
- 存放路径：`assets/images/medical-segmentation/`
- 子文件夹按网络分类

---

## 📖 参考资源

### 综述论文
1. [A survey on deep learning in medical image segmentation](https://arxiv.org/abs/2009.13120)
2. [Medical Image Segmentation: A Review](https://arxiv.org/abs/2102.09747)

### 数据集
1. ACDC (心脏分割)
2. Synapse (多器官分割)
3. BTCV (腹部器官)
4. BraTS (脑肿瘤)

### 评价指标
- Dice Coefficient
- IoU (Intersection over Union)
- Hausdorff Distance
- Surface Dice

---

**下一步**：开始创建第一篇博客 - FCN与UNet

