---
layout: post
title: "SAM与MedSAM：基础模型引领医学分割新范式"
date: 2022-10-01 10:00:00 +0800
categories: [医学影像, 图像分割]
tags: [UNet, 医学图像, Foundation Model]
excerpt: "深入探讨Meta AI的Segment Anything Model（SAM）如何通过prompt机制实现通用分割，以及MedSAM如何将其成功迁移到医学影像领域。"
author: YangCazz
math: true
---

## 📋 引言

在前面的文章中，我们学习了各种专门为医学图像分割设计的网络：从[UNet](/2025/02/01/fcn-unet-foundation/)的U型结构，到[Transformer](/2025/02/20/transunet-hybrid-architecture/)的全局建模。这些方法虽然有效，但都存在一个共同问题：

**需要针对每个任务单独训练**

```
传统方法的困境：

任务1：肝脏分割
→ 收集肝脏标注数据
→ 训练UNet/TransUNet
→ 仅能分割肝脏

任务2：肺部分割  
→ 重新收集肺部数据
→ 重新训练模型
→ 仅能分割肺部

问题：
✗ 每个任务需要大量标注
✗ 无法利用已学知识
✗ 泛化能力有限
```

**SAM（Segment Anything Model，2023）** 提出了革命性的想法：

> **一个模型，分割一切**

通过**Promptable Segmentation**（可提示分割），SAM实现：
- ✅ **Zero-shot**：无需训练即可分割新类别
- ✅ **交互式**：通过点击、框选、文本等方式指定目标
- ✅ **通用性**：一个模型处理所有分割任务

**MedSAM**则将SAM成功迁移到医学领域，成为医学图像分割的新范式。

---

## 🎯 SAM：核心思想

### 论文信息
- **标题**: Segment Anything
- **作者**: Alexander Kirillov, et al. (Meta AI Research)
- **发表**: ICCV 2023
- **论文链接**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **官方代码**: [GitHub](https://github.com/facebookresearch/segment-anything)
- **数据集**: SA-1B（11M图像，1.1B mask）

### 什么是Promptable Segmentation？

**传统分割**：输入图像 → 输出固定类别的mask

**Promptable分割**：输入图像 + **Prompt** → 输出对应的mask

**Prompt类型**：

1. **Point Prompt**（点提示）
   ```
   用户点击目标 → 分割该目标
   
   示例：点击心脏 → 分割心脏
         点击肿瘤 → 分割肿瘤
   ```

2. **Box Prompt**（框提示）
   ```
   用户框选区域 → 分割区域内目标
   
   示例：框选肝脏 → 精确分割肝脏边界
   ```

3. **Mask Prompt**（mask提示）
   ```
   用户提供粗糙mask → 精细化分割
   
   示例：涂鸦标注 → 精确分割
   ```

4. **Text Prompt**（文本提示，SAM不直接支持）
   ```
   用户输入"liver" → 分割肝脏
   ```

### SAM架构

SAM = **Image Encoder** + **Prompt Encoder** + **Mask Decoder**

```
图像输入 (1024×1024×3)
        ↓
┌──────────────────────┐
│  Image Encoder (ViT-H) │
│  - Vision Transformer  │
│  - 输出：256×64×64     │
└──────────────────────┘
        ↓
    图像嵌入
        ↓
Prompt输入  →  ┌────────────────────┐
(点/框/mask)   │  Prompt Encoder     │
               │  - 点：位置编码     │
               │  - 框：嵌入向量     │
               │  - Mask：卷积编码   │
               └────────────────────┘
                       ↓
                  Prompt嵌入
                       ↓
               ┌────────────────────┐
               │   Mask Decoder      │
               │   - Transformer     │
               │   - 交叉注意力      │
               │   - 输出多个mask    │
               └────────────────────┘
                       ↓
               Masks + IoU Scores
            (可能有多个候选)
```

### 关键组件

#### 1. Image Encoder

```python
# 使用ViT-H（Huge）作为图像编码器
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Vision Transformer (ViT-H/16)
        self.vit = VisionTransformer(
            img_size=1024,
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16
        )
    
    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        features = self.vit(x)  # (B, 256, 64, 64)
        return features
```

**特点**：
- 输入固定1024×1024（预处理时resize）
- 输出256通道的64×64特征图
- 参数量：约630M（占SAM总参数的99%）

#### 2. Prompt Encoder

```python
class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 点提示编码
        self.point_embeddings = nn.Embedding(2, embed_dim)  # 前景/背景点
        
        # 框提示编码
        self.box_embeddings = nn.Embedding(4, embed_dim)  # 左上、右下角
        
        # Mask提示编码
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, embed_dim, 3, padding=1)
        )
    
    def forward(self, points=None, boxes=None, masks=None):
        sparse_embeddings = []
        
        # 编码点
        if points is not None:
            point_embeddings = self.point_embeddings(points[:, :, 2])  # 前景=1,背景=0
            point_embeddings += self._get_positional_encoding(points[:, :, :2])
            sparse_embeddings.append(point_embeddings)
        
        # 编码框
        if boxes is not None:
            box_embeddings = self._encode_boxes(boxes)
            sparse_embeddings.append(box_embeddings)
        
        # 编码mask
        dense_embeddings = None
        if masks is not None:
            dense_embeddings = self.mask_encoder(masks)
        
        return sparse_embeddings, dense_embeddings
    
    def _get_positional_encoding(self, coords):
        """位置编码：将(x,y)坐标编码为高维向量"""
        # 使用正弦/余弦位置编码
        # ...
        return pos_encoding
```

**Prompt编码策略**：
- **稀疏Prompt**（点、框）：使用位置编码 + 学习嵌入
- **密集Prompt**（mask）：使用卷积网络编码

#### 3. Mask Decoder

```python
class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_mask_tokens=4):
        super().__init__()
        
        # Mask tokens（可学习的query）
        self.mask_tokens = nn.Embedding(num_mask_tokens, transformer_dim)
        
        # Transformer解码器
        self.transformer = nn.ModuleList([
            TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                num_heads=8
            )
        ])
        
        # 输出MLP
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, 2, 2),
            nn.LayerNorm(...),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, 2, 2),
            nn.GELU()
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(num_mask_tokens)
        ])
        
        # IoU预测头
        self.iou_prediction_head = MLP(transformer_dim, 256, num_mask_tokens, 3)
    
    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings):
        # image_embeddings: (B, 256, 64, 64)
        # sparse_prompt_embeddings: [(B, N, 256), ...]
        
        # 准备输出tokens
        output_tokens = self.mask_tokens.weight.unsqueeze(0).repeat(B, 1, 1)
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        
        # 将图像嵌入展平
        src = image_embeddings.flatten(2).permute(0, 2, 1)  # (B, 4096, 256)
        
        # Transformer解码
        hs, src = self.transformer[0](src, tokens)
        
        # 预测mask和IoU
        masks = []
        iou_pred = self.iou_prediction_head(hs[:, :self.num_mask_tokens, :])
        
        # 上采样特征
        src = src.transpose(1, 2).view(B, 256, 64, 64)
        upscaled_embedding = self.output_upscaling(src)  # (B, 32, 256, 256)
        
        # 为每个mask token生成mask
        for i in range(self.num_mask_tokens):
            masks.append(
                self.output_hypernetworks_mlps[i](hs[:, i, :]) @ upscaled_embedding.view(B, 32, -1)
            )
        
        masks = torch.stack(masks, dim=1).view(B, -1, 256, 256)
        
        return masks, iou_pred
```

**关键设计**：
- **多mask输出**：同时预测多个候选mask（通常3个）
- **IoU预测**：为每个mask预测质量分数
- **最优mask选择**：根据IoU分数选择最佳mask

### 训练策略

#### SA-1B数据集

**规模**：
- 图像数量：11M（1100万）
- Mask数量：1.1B（11亿）
- 平均每张图100个mask

**构建流程**（数据飞轮）：

```
阶段1：辅助标注（Assisted-manual）
→ 专业标注员使用SAM辅助标注
→ 收集4.3M mask（120K图像）

阶段2：半自动标注（Semi-automatic）
→ SAM自动建议mask
→ 标注员审核和修正
→ 收集5.9M mask（180K图像）

阶段3：全自动标注（Fully automatic）
→ SAM自动生成mask
→ 自动过滤低质量mask
→ 收集1.1B mask（11M图像）
```

#### 损失函数

$$
\mathcal{L} = \mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{IoU}}
$$

**Focal Loss**：处理前景/背景不平衡

$$
\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)
$$

**Dice Loss**：直接优化Dice系数

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2|P \cap G|}{|P| + |G|}
$$

**IoU Loss**：辅助IoU预测头

$$
\mathcal{L}_{\text{IoU}} = \text{MSE}(\text{IoU}_{\text{pred}}, \text{IoU}_{\text{true}})
$$

---

## 🏥 MedSAM：医学领域的SAM

### 论文信息
- **标题**: Segment Anything in Medical Images
- **作者**: Jun Ma, et al. (University of Toronto)
- **发表**: Nature Communications 2024
- **论文链接**: [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **官方代码**: [GitHub](https://github.com/bowang-lab/MedSAM)

### 为什么需要MedSAM？

**SAM在医学图像上的问题**：

```
测试SAM（零样本）在医学图像上：

数据集：Synapse Multi-organ CT
结果：
- 肝脏 Dice: 0.42（UNet: 0.94）
- 胰腺 Dice: 0.18（UNet: 0.70）
- 平均 Dice: 0.35（UNet: 0.85）

问题：
✗ SAM训练数据全是自然图像
✗ 医学图像特性（灰度、噪声、模态）完全不同
✗ Zero-shot泛化失败
```

**MedSAM的解决方案**：

使用**医学图像数据**fine-tune SAM

### MedSAM数据集

**规模**：
- 图像数量：1.57M
- Mask数量：约10M
- 模态：10种（CT、MRI、超声、X-ray、眼底、病理等）
- 解剖结构：30+ 类（器官、肿瘤、病灶）

**数据来源**：
- 公开数据集：NCI、TCIA、Medical Segmentation Decathlon等
- 合作医院：多中心数据

### MedSAM架构

**修改**：仅fine-tune SAM，架构不变

```python
# MedSAM = SAM + 医学图像fine-tuning
model = SAM(
    image_encoder='vit_h',  # 保持ViT-H
    prompt_encoder='default',  # 保持不变
    mask_decoder='default'  # 保持不变
)

# Fine-tuning策略
for name, param in model.named_parameters():
    if 'image_encoder' in name:
        param.requires_grad = True  # 解冻图像编码器
    else:
        param.requires_grad = False  # 冻结其他部分（初期）
```

### 训练策略

```python
# 阶段1：仅fine-tune Image Encoder
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

for epoch in range(10):
    for images, masks, boxes in train_loader:
        # 使用box prompt训练
        pred_masks, iou_pred = model(images, boxes=boxes)
        loss = focal_dice_loss(pred_masks, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 阶段2：fine-tune整个网络
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,  # 更小的学习率
    weight_decay=0.01
)

for epoch in range(10, 30):
    # ... 训练
```

### 性能对比

#### 多模态医学图像分割

| 模态 | 任务 | SAM (Zero-shot) | **MedSAM** | UNet |
|------|------|-----------------|-----------|------|
| CT | 肝脏 | 0.42 | **0.92** | 0.94 |
| CT | 胰腺 | 0.18 | **0.68** | 0.70 |
| MRI | 心脏 | 0.35 | **0.88** | 0.90 |
| 超声 | 甲状腺结节 | 0.25 | **0.75** | 0.78 |
| X-ray | 肺部 | 0.30 | **0.83** | 0.85 |
| 眼底 | 视盘 | 0.50 | **0.91** | 0.92 |
| 病理 | 细胞核 | 0.40 | **0.82** | 0.84 |

**关键发现**：
- ✅ MedSAM接近专用UNet的性能
- ✅ **一个模型处理所有模态**（vs. 每个任务训练一个UNet）
- ✅ 对新类别有良好泛化

#### Few-shot学习

```
场景：新任务（新器官/新模态），仅有少量标注

实验：使用1、5、10、50个标注样本fine-tune

结果（平均Dice）：
样本数 | SAM | MedSAM | UNet
  1    | 0.12 | 0.45  | 0.30
  5    | 0.25 | 0.62  | 0.55
 10    | 0.32 | 0.71  | 0.68
 50    | 0.40 | 0.80  | 0.82

观察：
- MedSAM在极少样本时优势巨大（+50% vs. SAM）
- 比UNet更高效（10样本达到UNet 50样本的性能）
```

---

## 💡 SAM/MedSAM的优势与局限

### ✅ 优势

#### 1. Zero/Few-shot能力

```
传统UNet：
任务A（肝脏） → 收集1000例 → 训练 → 模型A
任务B（肺）   → 收集1000例 → 训练 → 模型B

MedSAM：
预训练 → 模型
任务A → 5例fine-tune → 完成
任务B → 5例fine-tune → 完成
```

#### 2. 交互式分割

```python
# 用户交互流程
def interactive_segmentation(image, user_clicks):
    model.eval()
    
    # 初始点击
    points = user_clicks  # [(x1, y1, 1), ...]  1=前景,0=背景
    pred_mask, iou = model(image, points=points)
    
    # 显示结果给用户
    show_mask(pred_mask)
    
    # 用户修正：添加前景/背景点
    while True:
        new_point = get_user_click()
        if new_point is None:
            break
        
        points.append(new_point)
        pred_mask, iou = model(image, points=points)
        show_mask(pred_mask)
    
    return pred_mask
```

**应用场景**：
- 放射科医生快速标注
- 病理学家辅助诊断
- 研究人员数据准备

#### 3. 通用性

```
一个MedSAM模型支持：
- 10+ 医学图像模态
- 30+ 解剖结构
- 点/框/mask等多种prompt

vs.

传统方法需要50+ 个专用模型
```

### ❌ 局限

#### 1. 计算资源需求

```
MedSAM参数量：636M
推理时间：约2s/图（RTX 3090）
GPU内存：约16GB

vs.

UNet参数量：31M
推理时间：约50ms/图
GPU内存：约2GB

问题：
✗ 临床实时应用困难
✗ 边缘设备部署挑战
```

**解决方案**：
- MobileSAM（5.7M参数，60×加速）
- FastSAM（基于YOLO，实时推理）

#### 2. 精度仍有差距

```
复杂任务（如小器官、边界模糊）：
MedSAM Dice: 0.68-0.75
专用UNet Dice: 0.80-0.85

差距：约5-10%
```

#### 3. 需要Prompt

```
MedSAM不能：
- 输入图像 → 直接输出所有器官分割

需要：
- 手工点击/框选每个目标
- 或预先提供bounding box

自动化程度低于全自动分割
```

---

## 🎓 实用技巧

### 1. Prompt工程

```python
# 策略1：Box Prompt最稳定
def get_box_prompt(mask_gt):
    """从ground truth提取bounding box"""
    y, x = np.where(mask_gt > 0)
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    return np.array([x_min, y_min, x_max, y_max])

# 策略2：多点Prompt增强鲁棒性
def get_multi_point_prompt(mask_gt, num_points=5):
    """在目标区域内采样多个前景点"""
    y, x = np.where(mask_gt > 0)
    indices = np.random.choice(len(x), size=num_points, replace=False)
    points = np.stack([x[indices], y[indices], np.ones(num_points)], axis=1)
    return points

# 策略3：前景+背景点
def get_fg_bg_points(mask_gt):
    """结合前景和背景点"""
    # 前景点
    y_fg, x_fg = np.where(mask_gt > 0)
    fg_point = np.array([[x_fg[len(x_fg)//2], y_fg[len(y_fg)//2], 1]])
    
    # 背景点（在边界外）
    y_bg, x_bg = np.where(mask_gt == 0)
    bg_point = np.array([[x_bg[0], y_bg[0], 0]])
    
    return np.concatenate([fg_point, bg_point], axis=0)
```

### 2. Fine-tuning最佳实践

```python
# 针对特定模态/任务fine-tune

# 1. 数据准备
train_dataset = MedicalDataset(
    images=ct_images,
    masks=ct_masks,
    transform=augmentation
)

# 2. 学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=20,
    eta_min=1e-7
)

# 3. 早停策略
best_dice = 0
patience = 5
counter = 0

for epoch in range(50):
    train_dice = train_epoch(model, train_loader)
    val_dice = validate(model, val_loader)
    
    if val_dice > best_dice:
        best_dice = val_dice
        save_checkpoint(model)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
```

### 3. 后处理优化

```python
def refine_prediction(pred_mask):
    """后处理提升分割质量"""
    import cv2
    from scipy import ndimage
    
    # 1. 移除小连通域
    labeled, num = ndimage.label(pred_mask)
    sizes = ndimage.sum(pred_mask, labeled, range(num + 1))
    mask_size = sizes < 100  # 移除小于100像素的区域
    remove_pixel = mask_size[labeled]
    pred_mask[remove_pixel] = 0
    
    # 2. 形态学闭操作（填充小孔）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pred_mask = cv2.morphologyEx(pred_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # 3. 边界平滑
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smooth_mask = np.zeros_like(pred_mask)
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(smooth_mask, [approx], -1, 1, -1)
    
    return smooth_mask
```

---

## 📖 总结

### SAM的核心贡献

1. **Promptable Segmentation范式**
   - 通过prompt实现灵活交互
   - 一个模型处理多种任务

2. **SA-1B超大规模数据集**
   - 11亿mask，前所未有的规模
   - 数据飞轮：模型标注 → 改进模型

3. **Zero-shot泛化能力**
   - 无需训练即可分割新类别
   - 开启基础模型在视觉领域的应用

### MedSAM的贡献

1. **医学领域适配**
   - 157万医学图像fine-tune
   - 跨模态通用性

2. **Few-shot高效学习**
   - 5-10个样本即可适配新任务
   - 显著降低标注成本

3. **临床实用性**
   - 交互式分割辅助诊断
   - 加速数据标注流程

### 未来展望

**技术方向**：
- **MedSAM 2.0**：支持3D医学图像
- **文本Prompt**：结合CLIP实现"分割肝脏肿瘤"等自然语言指令
- **轻量化**：MobileMedSAM用于移动端

**应用前景**：
- 放射科：辅助阅片和测量
- 病理科：快速标注和诊断
- 外科：术前规划和导航
- 研究：高效数据集构建

**下一篇预告**：[nnU-Net：自适应医学分割框架](/2025/03/15/nnunet-self-configuring-framework/) - 探索如何通过自动化配置让UNet适配任何医学分割任务。

---

## 📚 参考资料

### 论文
1. [SAM] Kirillov, A., et al. (2023). Segment Anything. *ICCV*.
2. [MedSAM] Ma, J., et al. (2024). Segment Anything in Medical Images. *Nature Communications*.
3. [SAM-Med2D] Cheng, J., et al. (2023). SAM-Med2D. *arXiv*.

### 代码实现
- [SAM官方](https://github.com/facebookresearch/segment-anything) - Meta AI原始代码
- [MedSAM官方](https://github.com/bowang-lab/MedSAM) - 医学图像版本
- [SAM-Med2D](https://github.com/uni-medical/SAM-Med2D) - 另一医学SAM版本

### 数据集
- [SA-1B](https://ai.facebook.com/datasets/segment-anything/) - SAM训练数据集
- [MedSAM数据](https://github.com/bowang-lab/MedSAM#dataset) - 医学图像数据集链接

---

## 🔗 系列文章导航

1. [FCN与UNet：医学分割的奠基之作](/2025/02/01/fcn-unet-foundation/)
2. [V-Net：3D医学图像分割的突破](/2025/02/05/vnet-3d-segmentation/)
3. [Attention UNet：注意力机制的引入](/2025/02/10/attention-unet/)
4. [UNet++/UNet 3+：密集连接的力量](/2025/02/15/unet-plus-series/)
5. [TransUNet：CNN与Transformer的融合](/2025/02/20/transunet-hybrid-architecture/)
6. [Swin-UNet：层级化Transformer](/2025/02/25/swin-unet-hierarchical-transformer/)
7. 📍 **SAM与MedSAM：基础模型的医学应用**（本文）
8. [nnU-Net：自适应医学分割框架](/2025/03/15/nnunet-self-configuring-framework/)

---

*本文深入探讨了SAM如何通过Promptable Segmentation革新分割范式，以及MedSAM如何将其成功迁移到医学影像领域，开启few-shot学习和交互式分割的新时代。*

