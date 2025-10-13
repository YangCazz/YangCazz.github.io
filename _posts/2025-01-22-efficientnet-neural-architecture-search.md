---
layout: post
title: "EfficientNet：神经架构搜索的艺术"
date: 2025-01-22
categories: [深度学习, 经典网络]
tags: [CNN, EfficientNet, NAS, 复合缩放, PyTorch]
excerpt: "深入解析Google的EfficientNet系列（V1-V2）。探索网络深度、宽度、分辨率三维度的复合缩放策略，以及如何通过NAS找到最优网络架构。"
---

# EfficientNet：神经架构搜索的艺术

## 引言

在之前的手工设计网络中（AlexNet、VGG、ResNet等），经常有人问：
* 为什么输入图像分辨率要固定为224？
* 为什么卷积的个数要设置为这个值？
* 为什么网络的深度设为这么深？

如果你问设计者，估计回复就四个字——**工程经验**。

而EfficientNet则用**神经架构搜索（NAS）**技术来系统性地搜索网络的**图像输入分辨率r**、**网络深度depth**以及**通道宽度width**三个参数的合理化配置。

## 系列概览

### 论文列表

* **[2019] EfficientNet V1**：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* **[2021] EfficientNet V2**：[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)

## 1. EfficientNet V1 (2019)

### 核心问题

如何系统性地扩展网络规模以提升性能？

### 三个维度的影响

![EfficientNet V1](/assets/images/deep-learning/EfficientNet_v1_SE_Inverted_Residual.png)

#### 维度1：深度（Depth）

**增加网络深度**：

✅ **优点**：
* 获得更丰富、复杂的特征
* 更好的迁移性和鲁棒性
* 能够更好地应用到其它任务

❌ **缺点**：
* 过深会梯度消失
* 训练困难
* 容易过拟合

#### 维度2：宽度（Width）

**增加网络宽度**（通道数）：

✅ **优点**：
* 获得粒度更高的特征
* 更多信息量，更容易训练
* 捕获更细粒度的模式

❌ **缺点**：
* 宽度很大但深度过浅的网络难以学到更深层次的特征
* 性能提升很快饱和

#### 维度3：分辨率（Resolution）

**增加输入分辨率**：

✅ **优点**：
* 潜在获得更高粒度的Feature Maps
* 捕获更细节的模式
* 对小目标更友好

❌ **缺点**：
* 过高的分辨率，收益递减
* 计算量急剧增加

### 复合缩放策略

**核心思想**：**同时**优化深度、宽度和分辨率。

#### 数学表达

$$
\text{depth}: d = \alpha^{\phi}
$$

$$
\text{width}: w = \beta^{\phi}
$$

$$
\text{resolution}: r = \gamma^{\phi}
$$

约束条件：
$$
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

$$
\alpha \geq 1, \beta \geq 1, \gamma \geq 1
$$

其中：
* \(\phi\)：复合系数（用户指定）
* \(\alpha, \beta, \gamma\)：通过网格搜索得到

#### 为什么是这个约束？

* **深度加倍**：计算量约为2倍（\(\alpha\)）
* **宽度加倍**：计算量约为4倍（\(\beta^2\)）  
* **分辨率加倍**：计算量约为4倍（\(\gamma^2\)）

约束确保总计算量约为 \(2^{\phi}\) 倍。

### MBConv Block

![MBConv Block](/assets/images/deep-learning/EfficientNet_v1_SE_Inverted_Residual_2.png)

EfficientNet基于**MBConv**（Mobile Inverted Bottleneck Convolution）：

```python
class MBConvBlock(nn.Module):
    """MBConv模块 = 逆残差 + SE注意力"""
    def __init__(self, in_channels, out_channels, expand_ratio, 
                 kernel_size, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # 1. Expansion (升维)
        if expand_ratio != 1:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Swish()  # Swish激活函数
            ))
        
        # 2. Depthwise卷积
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Swish()
        ))
        
        # 3. SE注意力模块
        if se_ratio is not None:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            layers.append(SEBlock(hidden_dim, squeeze_channels))
        
        # 4. Projection (降维)
        layers.append(nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
            # 注意：没有激活函数（Linear Bottleneck）
        ))
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=0.2) if self.use_residual else None
    
    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        else:
            return self.conv(x)
```

### Swish激活函数

**定义**：
$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

```python
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
```

**特点**：
* 平滑、非单调
* 性能优于ReLU
* 计算稍复杂

### EfficientNet-B0到B7

| 模型 | \(\phi\) | 参数量 | FLOPs | Top-1准确率 |
|------|------|--------|-------|-----------|
| B0 | 0 | 5.3M | 0.39B | 77.1% |
| B1 | 0.5 | 7.8M | 0.70B | 79.1% |
| B2 | 1 | 9.2M | 1.0B | 80.1% |
| B3 | 2 | 12M | 1.8B | 81.6% |
| B4 | 3 | 19M | 4.2B | 82.9% |
| B5 | 4 | 30M | 9.9B | 83.6% |
| B6 | 5 | 43M | 19B | 84.0% |
| B7 | 6 | 66M | 37B | 84.3% |

### 性能对比

在相同准确率下：

| 模型 | 参数量 | FLOPs | Top-1准确率 |
|------|--------|-------|-----------|
| ResNet-152 | 60M | 11.3B | 77.8% |
| GPipe | 556M | 128B | 84.3% |
| **EfficientNet-B1** | **7.8M** | **0.70B** | **79.1%** |
| **EfficientNet-B7** | **66M** | **37B** | **84.3%** |

**EfficientNet-B7**：
* 参数量是GPipe的**1/8.4**
* 推理速度快**6.1倍**
* 达到相同精度！

### 模型复现

* **代码地址**：[GitHub - DeepLearning/model_classification/EfficientNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/EfficientNet)

## 2. EfficientNet V2 (2021)

### V1的问题

![EfficientNet V2](/assets/images/deep-learning/EfficientNet_v2.png)

1. **训练速度慢**：在大尺寸图像上训练很慢
2. **DW卷积速度慢**：浅层DW卷积无法利用硬件加速
3. **扩展性问题**：简单放大模型效果不佳

### 核心创新

#### 1. Fused-MBConv

![Fused-MBConv](/assets/images/deep-learning/EfficientNet_v2_Fused_MBConv.png)

**浅层使用Fused-MBConv，深层使用MBConv**

```python
class FusedMBConvBlock(nn.Module):
    """融合的MBConv：将DW+PW融合为标准卷积"""
    def __init__(self, in_channels, out_channels, expand_ratio, 
                 kernel_size, stride):
        super(FusedMBConvBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion + 常规卷积（代替DW+PW）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, 
                     stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )
        
        # Projection
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_residual:
            return x + out
        return out
```

**为什么浅层用Fused-MBConv？**
* 浅层特征简单，不需要复杂的分离卷积
* 标准卷积可以更好利用硬件加速
* 减少内存访问，提高速度

#### 2. Progressive Learning

**渐进式学习策略**：

**阶段1（Early）**：
* 使用较小的图像尺寸（如128×128）
* 使用较弱的数据增强
* 快速学习简单模式

**阶段2（Late）**：
* 逐渐增大图像尺寸（如224×224，甚至更大）
* 增强数据增强强度
* 学习更复杂的模式

```python
# 渐进式学习伪代码
for epoch in range(total_epochs):
    # 动态调整图像大小
    if epoch < total_epochs // 3:
        image_size = 128
        aug_strength = 'weak'
    elif epoch < 2 * total_epochs // 3:
        image_size = 224
        aug_strength = 'medium'
    else:
        image_size = 380
        aug_strength = 'strong'
    
    train_one_epoch(image_size, aug_strength)
```

**优势**：
* 早期快速收敛
* 后期精细调整
* 训练速度提升显著

### V1 vs V2

| 特性 | V1 | V2 |
|------|----|----|
| 浅层结构 | MBConv | Fused-MBConv |
| 深层结构 | MBConv | MBConv |
| 激活函数 | Swish | SiLU (Swish) |
| 训练策略 | 固定尺寸 | Progressive Learning |
| 训练速度 | 较慢 | 快3-9倍 |
| 参数效率 | 高 | 更高 |

### 性能对比

| 模型 | 参数量 | FLOPs | 训练速度 | Top-1准确率 |
|------|--------|-------|---------|-----------|
| EfficientNet-B7 | 66M | 37B | 1× | 84.3% |
| EfficientNet V2-M | 54M | 24B | **2.3×** | 85.1% |
| EfficientNet V2-L | 119M | 56B | **3.0×** | 85.7% |

**V2不仅更准，而且训练更快！**

## NAS（神经架构搜索）

### 什么是NAS？

**传统方法**：手工设计 → 实验 → 调整 → 再实验

**NAS方法**：定义搜索空间 → 自动搜索 → 找到最优架构

### 搜索空间

在EfficientNet中，NAS搜索的参数包括：
* 层数
* 每层的卷积核大小（3×3 or 5×5）
* 扩展比例（expand_ratio）
* SE模块的缩减比例

### NAS的成本

**巨大的计算量**！

以EfficientNet-B0为例：
* 搜索时间：**数千GPU小时**
* 成本：数万美元

**但是**：
* 搜索一次，受益无穷
* 找到的架构可以复用

## 实践经验

### 1. 选择合适的版本

```python
from torchvision.models import efficientnet_b0, efficientnet_b7, efficientnet_v2_s

# 移动端/边缘设备
model = efficientnet_b0(pretrained=True)

# 服务器/高性能场景
model = efficientnet_b7(pretrained=True)

# 平衡选择（V2）
model = efficientnet_v2_s(pretrained=True)
```

### 2. 迁移学习

```python
import torch.nn as nn
from torchvision import models

# 加载预训练模型
model = models.efficientnet_b0(pretrained=True)

# 修改分类器
num_classes = 10
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 冻结特征提取层
for param in model.features.parameters():
    param.requires_grad = False
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 4. 渐进式学习实现

```python
def get_progressive_image_size(epoch, total_epochs):
    """根据epoch返回图像大小"""
    progress = epoch / total_epochs
    
    if progress < 0.3:
        return 128
    elif progress < 0.6:
        return 192
    elif progress < 0.9:
        return 256
    else:
        return 300

# 在训练循环中使用
for epoch in range(num_epochs):
    image_size = get_progressive_image_size(epoch, num_epochs)
    # 更新数据加载器的transform...
```

## 设计哲学

### 参数少 ≠ 速度快

EfficientNet强调：
* 要关注**实际推理速度**
* 要关注**训练效率**
* 要关注**硬件适配性**

### 复合缩放的智慧

单独增加任一维度都会遇到瓶颈：
* 只增加深度 → 梯度消失
* 只增加宽度 → 学不到复杂特征
* 只增加分辨率 → 收益递减

**同时平衡三个维度** → 获得最佳性能！

## 总结

### EfficientNet V1的贡献

1. **复合缩放策略**：系统性地扩展网络
2. **NAS优化**：自动搜索最优架构
3. **MBConv Block**：高效的基础模块
4. **参数效率**：达到SOTA的同时大幅减少参数

### EfficientNet V2的贡献

1. **Fused-MBConv**：优化浅层结构
2. **Progressive Learning**：加速训练
3. **更好的性能**：更准、更快、更高效

### 关键启示

* **系统性设计很重要**：三个维度要平衡
* **NAS是未来趋势**：自动化优于手工
* **训练效率同样重要**：不只关注推理
* **实践出真知**：理论要结合实际

## 影响

EfficientNet系列：
* 📊 刷新了ImageNet准确率记录
* 🔧 成为工业界首选Backbone之一
* 🚀 广泛应用于各类视觉任务
* 🎓 启发了AutoML在网络设计中的应用

**EfficientNet证明了：好的网络设计需要理论、实验和自动化的完美结合！**

## 参考资料

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling
2. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training
3. [我的GitHub代码仓库](https://github.com/YangCazz/DeepLearning)
4. [EfficientNet论文解读](https://arxiv.org/abs/1905.11946)

---

*这是深度学习经典网络系列的第七篇，下一篇将介绍Attention机制。欢迎关注！*

