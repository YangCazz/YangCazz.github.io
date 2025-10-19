---
layout: post
title: "深度学习先驱：LeNet与AlexNet的开创之路"
date: 2021-09-10 10:00:00 +0800
categories: [深度学习, 经典网络]
tags: [CNN, 计算机视觉, PyTorch]
excerpt: "深入探讨计算机视觉领域的两大开山之作：LeNet和AlexNet。从手写数字识别到ImageNet挑战赛冠军，了解卷积神经网络如何改变了AI的历史进程。"
---

# 深度学习先驱：LeNet与AlexNet的开创之路

## 引言

计算机视觉领域中，有着几类经典应用派系（可以在[SOTA](https://paperswithcode.com/sota)网站查看各类别的模型架构）：

* 图像分类 - [Image Classification](https://paperswithcode.com/task/image-classification)
* 目标检测 - [Object Detection](https://paperswithcode.com/task/object-detection)
* 语义分割 - [Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)
* 图像生成 - [Image Generation](https://paperswithcode.com/task/image-generation)

本文将介绍图像分类任务的两个开创性工作：**LeNet** 和 **AlexNet**，它们为深度学习的发展奠定了基石。

## 1. LeNet (1998) - 深度学习的曙光

### 基本信息

* **作者**：Yann LeCun
* **论文**：[Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791)
* **DOI**：10.1109/5.726791
* **时间**：1998年

### 简介

LeNet是计算机视觉的开山之作之一，Yann LeCun最早采用了基于卷积+梯度优化的神经网络用于支票手写数字的识别。这一工作不仅推动了计算机在视觉任务上的应用，更在手写字识别上达到了令人满意的结果。

### 网络结构

![LeNet结构图](/assets/images/deep-learning/LeNet.png)

LeNet构建了**卷积-下采样(池化)-全连接**的卷积网络范式，这一架构成为了后续所有卷积神经网络的基础模板。

### 主要贡献

1. **确立了CNN的基本范式**：卷积层 → 池化层 → 全连接层的经典结构
2. **推动计算机视觉的发展**：证明了神经网络在视觉任务上的潜力
3. **实际应用价值**：在手写数字识别上取得了实用化的效果

### 模型复现

我已经在PyTorch平台上复现了LeNet模型：

* **平台**：PyTorch
* **主要库**：torchvision, torch, matplotlib
* **数据集**：CIFAR10
* **代码地址**：[GitHub - DeepLearning/model_classification/LeNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/LeNet)

## 2. AlexNet (2012) - 深度学习的复兴

### 基本信息

* **作者**：Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
* **论文**：[ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
* **DOI**：10.1145/3065386
* **时间**：2012年

### 简介

AlexNet是CV领域的另一部开山之作。时隔多年后，CNN在2012年ILSVRC（[ImageNet](https://image-net.org)大规模视觉识别挑战赛）上夺得冠军，延续了Yann LeCun的工作，展示了CNN在图像识别领域的优势，是CV领域承先启后的杰作。

### 网络结构

![AlexNet结构图](/assets/images/deep-learning/AlexNet.png)

AlexNet相比LeNet有了显著的深度增加，从5层增加到了8层，并引入了多项创新技术。

### 主要贡献

1. **证明了学习特征优于手工特征**
   
   **首次证明了学习到的特征可以超越手工设计的特征**，让更多人开始注意这个和黑匣子一样的"深度学习"，**掀起了深度学习的研究浪潮**。

2. **大规模数据训练**
   
   在大数据样本上做实验，取得更好的效果。但受限于硬件条件，提出了**多GPU训练模式**，这一模式至今仍在使用。

3. **ReLU激活函数**
   
   引入激活函数ReLU，让映射拟合增加非线性组件，解决了传统Sigmoid函数的梯度消失问题。

4. **Dropout正则化**
   
   引入Dropout随机失活操作，有效防止过拟合，逐渐成为CNN领域的核心组件。

5. **更深的网络**
   
   CNN开始向"深度"探索，LeNet为5层，AlexNet为8层，证明了深度对模型性能的重要性。

6. **端到端学习**
   
   实现**端到端**的模型定义，简化了模型设计和训练流程。

### 技术亮点

#### 1. ReLU激活函数

```python
# ReLU激活函数的简单实现
def relu(x):
    return max(0, x)
```

ReLU相比传统的Sigmoid和Tanh激活函数：
- 计算更简单，加速训练
- 缓解梯度消失问题
- 产生稀疏激活，提高模型效率

#### 2. Dropout正则化

```python
# Dropout在PyTorch中的使用
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # ... 卷积层定义 ...
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 50%的dropout率
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
```

### 模型复现

我在PyTorch平台上复现了AlexNet模型：

* **平台**：PyTorch
* **主要库**：torchvision, torch, matplotlib
* **数据集**：Oxford Flower102花分类数据集
* **代码地址**：[GitHub - DeepLearning/model_classification/AlexNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/Alexnet)

## 历史意义

### LeNet的开创性

LeNet在1998年就已经展示了卷积神经网络的强大能力，但由于当时的硬件限制和数据集规模，并没有引起广泛关注。然而，它确立的卷积-池化-全连接的架构成为了后续所有CNN的基础。

### AlexNet的革命性

AlexNet在2012年的ImageNet挑战赛上以巨大优势夺冠（top-5错误率15.3%，第二名26.2%），这一成就：

1. **重新点燃了深度学习研究的热情**
2. **证明了深度神经网络的实用价值**
3. **推动了GPU在深度学习中的应用**
4. **启发了后续一系列更深、更强的网络架构**

## 从LeNet到AlexNet的进化

| 特性 | LeNet (1998) | AlexNet (2012) |
|------|-------------|----------------|
| 层数 | 5层 | 8层 |
| 激活函数 | Tanh | ReLU |
| 正则化 | 无 | Dropout |
| 数据集 | MNIST (6万) | ImageNet (120万) |
| 训练设备 | CPU | 双GPU |
| 参数量 | ~6万 | ~6000万 |

## 实践经验

在复现这两个经典网络的过程中，我获得了以下体会：

1. **网络深度很重要**：AlexNet相比LeNet更深，性能显著提升
2. **激活函数的选择**：ReLU确实比Sigmoid/Tanh更好训练
3. **正则化的必要性**：Dropout对防止过拟合效果显著
4. **数据规模的影响**：更大的数据集能充分发挥深度网络的优势

## 总结

LeNet和AlexNet是深度学习历史上的两座里程碑：

* **LeNet**：开创了CNN范式，但限于时代，未能大放异彩
* **AlexNet**：在合适的时机（大数据+GPU算力），证明了深度学习的革命性价值

这两个网络虽然在今天看来已经相对简单，但它们确立的许多概念和技术至今仍在使用。理解这些经典网络，对于学习现代深度学习架构有着重要的意义。

## 参考资料

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition
2. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks
3. [我的GitHub代码仓库](https://github.com/YangCazz/DeepLearning)
4. [Papers with Code - Image Classification](https://paperswithcode.com/task/image-classification)

---

*这是深度学习经典网络系列的第一篇，后续将继续介绍VGG、GoogLeNet、ResNet等经典架构。欢迎关注！*

