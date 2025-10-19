---
layout: post
title: "GCN实战：PyTorch实现图卷积网络"
date: 2022-04-01 10:00:00 +0800
categories: [图神经网络, 深度学习, PyTorch]
tags: [GNN, PyTorch, 代码实现]
excerpt: "通过PyTorch和PyTorch Geometric实现图卷积网络，包含完整的代码示例和实际应用案例。"
---

# GCN实战：PyTorch实现图卷积网络

在前面的文章中，我们深入探讨了GCN的数学原理。现在让我们通过PyTorch和PyTorch Geometric来实现GCN，并展示如何在实际任务中使用它。

## 环境准备

### 安装依赖

```bash
pip install torch torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
```

### 导入必要的库

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
```

## PyTorch Geometric基础

### 图数据结构

PyTorch Geometric使用`Data`类来表示图：

```python
# 创建简单的图数据
edge_index = torch.tensor([[0, 1, 1, 2],
                          [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
# Data(edge_index=[2, 4], x=[3, 1])
```

### 图数据属性

- `data.x`: 节点特征矩阵 `[num_nodes, num_node_features]`
- `data.edge_index`: 边索引 `[2, num_edges]`
- `data.edge_attr`: 边特征矩阵 `[num_edges, num_edge_features]`
- `data.y`: 标签（节点级或图级）

## GCN的PyTorch实现

### 使用PyTorch Geometric的GCN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        
        # 第二层GCN
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
```

### 手动实现GCN层

为了更好地理解GCN的工作原理，让我们手动实现GCN层：

```python
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 添加自连接
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 线性变换
        x = self.linear(x)
        
        # 计算归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # 消息传递
        out = self.propagate(edge_index, x=x, norm=norm)
        
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        return aggr_out
```

## 完整训练示例

### 数据准备

```python
# 加载Cora数据集
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0]

print(f"节点数: {data.x.size(0)}")
print(f"边数: {data.edge_index.size(1)}")
print(f"特征维度: {data.x.size(1)}")
print(f"类别数: {dataset.num_classes}")
```

### 模型训练

```python
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
data = data.to(device)

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

# 训练函数
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数
def test():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
    return acc

# 训练循环
for epoch in range(200):
    loss = train()
    if epoch % 20 == 0:
        acc = test()
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
```

## 高级GCN实现

### 带残差连接的GCN

```python
class ResidualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(ResidualGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.layers.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            residual = x if i > 0 and x.size(1) == layer.out_channels else None
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            if residual is not None:
                x = x + residual
                
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 带注意力机制的GCN

```python
from torch_geometric.nn import GATConv

class AttentionGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super(AttentionGCN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training, p=0.6)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.6)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## 实际应用案例

### 节点分类任务

```python
# 在Cora数据集上的节点分类
def node_classification_example():
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    model = GCN(dataset.num_node_features, 16, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
                print(f'Epoch {epoch:03d}, Accuracy: {acc:.4f}')
```

### 图分类任务

```python
from torch_geometric.nn import global_mean_pool

class GraphGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # 图级别的池化
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
```

## 性能优化技巧

### 1. 批处理

```python
from torch_geometric.loader import DataLoader

# 创建数据加载器
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    # 处理批数据
```

### 2. 内存优化

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

class MemoryEfficientGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MemoryEfficientGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = checkpoint(self.conv1, x, edge_index)
        x = F.relu(x)
        x = checkpoint(self.conv2, x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_with_amp():
    model.train()
    optimizer.zero_grad()
    
    with autocast():
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 调试和可视化

### 1. 梯度检查

```python
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: {param.grad.norm():.4f}")
        else:
            print(f"{name}: No gradient")
```

### 2. 特征可视化

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_embeddings(model, data, epoch):
    model.eval()
    with torch.no_grad():
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
        
    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=data.y.cpu().numpy(), cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'Node Embeddings at Epoch {epoch}')
    plt.show()
```

## 常见问题和解决方案

### 1. 过平滑问题

```python
# 使用残差连接
class ResidualGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResidualGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = h2 + h1  # 残差连接
        h3 = self.conv3(h2, edge_index)
        return F.log_softmax(h3, dim=1)
```

### 2. 梯度消失

```python
# 使用层归一化
class LayerNormGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LayerNormGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        
    def forward(self, x, edge_index):
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.norm2(self.conv2(x, edge_index))
        return F.log_softmax(x, dim=1)
```

## 总结

通过本文的学习，我们掌握了：

1. **PyTorch Geometric基础**：图数据结构和基本操作
2. **GCN实现**：从简单到复杂的多种实现方式
3. **训练技巧**：批处理、内存优化、混合精度训练
4. **调试方法**：梯度检查、特征可视化
5. **问题解决**：过平滑、梯度消失等常见问题

GCN作为图神经网络的基础模型，为后续学习更复杂的图神经网络架构奠定了重要基础。在下一篇文章中，我们将探讨图神经网络在医学图像处理中的应用。

---

**参考文献**：
1. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
2. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric.
3. PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/
