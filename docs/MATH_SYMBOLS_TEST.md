# 数学符号渲染测试

## 常见符号测试

### 1. 希腊字母

**小写字母：**
- Alpha: \(\alpha\)
- Beta: \(\beta\)
- Gamma: \(\gamma\)
- Delta: \(\delta\)
- Epsilon: \(\epsilon\) 或 \(\varepsilon\)
- Theta: \(\theta\) 或 \(\vartheta\)
- Lambda: \(\lambda\)
- Mu: \(\mu\)
- Pi: \(\pi\)
- Sigma: \(\sigma\)
- Tau: \(\tau\)
- Phi: \(\phi\) 或 \(\varphi\)
- Omega: \(\omega\)

**大写字母：**
- Gamma: \(\Gamma\)
- Delta: \(\Delta\)
- Theta: \(\Theta\)
- Lambda: \(\Lambda\)
- Sigma: \(\Sigma\)
- Phi: \(\Phi\)
- Omega: \(\Omega\)

### 2. 运算符号

**基本运算：**
- 乘法: \(a \times b\)
- 点乘: \(a \cdot b\)
- 除法: \(a \div b\)
- 分数: \(\frac{a}{b}\)
- 正负: \(\pm a\)
- 负正: \(\mp a\)

**高级运算：**
- 求和: \(\sum_{i=1}^{n} x_i\)
- 求积: \(\prod_{i=1}^{n} x_i\)
- 积分: \(\int_0^{\infty} f(x) dx\)
- 偏导: \(\frac{\partial f}{\partial x}\)
- 梯度: \(\nabla f\)
- 极限: \(\lim_{x \to 0} f(x)\)

### 3. 关系符号

- 等于: \(a = b\)
- 不等于: \(a \neq b\)
- 小于: \(a < b\)
- 大于: \(a > b\)
- 小于等于: \(a \leq b\) 或 \(a \le b\)
- 大于等于: \(a \geq b\) 或 \(a \ge b\)
- 约等于: \(a \approx b\)
- 恒等于: \(a \equiv b\)
- 正比于: \(a \propto b\)

### 4. 集合符号

- 属于: \(x \in A\)
- 不属于: \(x \notin A\)
- 子集: \(A \subset B\) 或 \(A \subseteq B\)
- 真子集: \(A \subsetneq B\)
- 并集: \(A \cup B\)
- 交集: \(A \cap B\)
- 空集: \(\emptyset\) 或 \(\varnothing\)

### 5. 逻辑符号

- 与: \(\land\) 或 \(\wedge\)
- 或: \(\lor\) 或 \(\vee\)
- 非: \(\neg\) 或 \(\lnot\)
- 蕴含: \(\Rightarrow\) 或 \(\implies\)
- 等价: \(\Leftrightarrow\) 或 \(\iff\)
- 存在: \(\exists\)
- 任意: \(\forall\)

### 6. 箭头符号

- 右箭头: \(\rightarrow\) 或 \(\to\)
- 左箭头: \(\leftarrow\) 或 \(\gets\)
- 双向箭头: \(\leftrightarrow\)
- 粗右箭头: \(\Rightarrow\)
- 粗左箭头: \(\Leftarrow\)
- 粗双向箭头: \(\Leftrightarrow\)
- 长右箭头: \(\longrightarrow\)
- 映射箭头: \(\mapsto\)

### 7. 其他常用符号

- 无穷: \(\infty\)
- 角度: \(\angle\)
- 度数: \(90^\circ\)
- 因此: \(\therefore\)
- 因为: \(\because\)
- 省略号: \(\cdots\) (居中), \(\ldots\) (底部), \(\vdots\) (竖直), \(\ddots\) (斜向)

## 复杂公式测试

### 矩阵与向量

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}
$$

### 带符号的公式

注意力机制中的 Softmax 计算：

$$
\text{score}(h_i, h_j) = h_i^T W h_j
$$

$$
\alpha_{ij} = \frac{\exp(\text{score}(h_i, h_j))}{\sum_{k=1}^{n} \exp(\text{score}(h_i, h_k))}
$$

### 多种符号组合

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]
$$

$$
\nabla_{\theta} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta} = \sum_{i=1}^{N} (y_i - \sigma(z_i)) \cdot x_i
$$

## 常见问题与解决方案

### 问题1：反斜杠被吞掉

**错误示例：**
```markdown
行内公式 $\alpha$ 不显示
```

**正确示例：**
```markdown
行内公式 \(\alpha\) 正常显示
```

### 问题2：下划线冲突

**错误示例：**
```markdown
$$y_pred_i$$  <!-- Markdown会把 _pred_ 解释为斜体 -->
```

**正确示例：**
```markdown
$$y_{pred_i}$$  <!-- 使用大括号包裹 -->
```

或者使用转义：
```markdown
$$y\_pred\_i$$  <!-- 转义下划线 -->
```

### 问题3：Times符号不显示

**错误写法：**
```markdown
\times  <!-- 单独使用可能被Markdown处理 -->
```

**正确写法：**
```markdown
\(a \times b\)  <!-- 在公式环境中使用 -->
$$c = a \times b$$
```

### 问题4：希腊字母乱码

**确保使用正确的命令：**
- ✅ `\alpha` → \(\alpha\)
- ✅ `\sigma` → \(\sigma\)
- ✅ `\Sigma` → \(\Sigma\)
- ❌ `α` (直接Unicode) → 可能不一致
- ❌ `&alpha;` (HTML实体) → 不起作用

## 完整测试公式

以下公式包含了多种常见符号：

$$
\begin{align}
f(x) &= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) dx \\
&= \sum_{i=1}^{n} \alpha_i \times x_i + \beta \\
&\approx \prod_{j=1}^{m} (1 + \lambda_j) \\
&\Rightarrow y \in [0, 1] \subset \mathbb{R}
\end{align}
$$

如果上述公式正确显示，说明数学公式渲染功能正常！

## 调试检查清单

如果符号显示有问题，请检查：

- [ ] 浏览器控制台是否显示 "MathJax 渲染完成"
- [ ] 是否在公式环境中（`\(...\)` 或 `$$...$$`）
- [ ] 反斜杠是否被正确保留（不被Markdown转义）
- [ ] 下划线是否被大括号包裹（避免Markdown斜体）
- [ ] 是否清除了浏览器缓存
- [ ] MathJax脚本是否成功加载

## 配置信息

当前配置：
- **MathJax版本**: 3.x
- **Markdown处理器**: Kramdown
- **数学引擎**: MathJax
- **CDN**: jsDelivr

---

*测试日期：2025年10月*

