# 数学公式渲染配置指南

## 🎯 功能概述

博客现已支持**LaTeX数学公式渲染**，使用 **MathJax 3** 引擎。

## ✨ 支持的公式格式

### 1. 行内公式

使用 `\(...\)` 包裹：

```markdown
这是一个行内公式 \(E = mc^2\)，它会嵌入在文本中。
```

**渲染效果**：这是一个行内公式 \(E = mc^2\)，它会嵌入在文本中。

### 2. 块级公式（居中显示）

使用 `$$...$$` 或 `\[...\]` 包裹：

```markdown
$$
\frac{\partial L}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) \cdot x_i
$$
```

**渲染效果**：

$$
\frac{\partial L}{\partial w} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) \cdot x_i
$$

## 📝 常用公式示例

### 1. 基础数学

#### 分数
```latex
$$
\frac{a}{b} \quad \frac{\partial f}{\partial x}
$$
```

#### 上下标
```latex
$$
x^2 + y^2 = z^2 \quad a_1, a_2, ..., a_n
$$
```

#### 根号
```latex
$$
\sqrt{2} \quad \sqrt[n]{x}
$$
```

#### 求和与积分
```latex
$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2} \quad \int_0^{\infty} e^{-x} dx
$$
```

### 2. 矩阵与向量

```latex
$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix}
$$
```

```latex
$$
\mathbf{x} = \begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{pmatrix}
$$
```

### 3. 希腊字母

```latex
行内：\(\alpha, \beta, \gamma, \delta, \epsilon, \theta, \lambda, \mu, \sigma, \omega\)

块级：
$$
\Alpha, \Beta, \Gamma, \Delta, \Theta, \Lambda, \Sigma, \Omega
$$
```

### 4. 机器学习/深度学习常用公式

#### Softmax
```latex
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$
```

#### 交叉熵损失
```latex
$$
L = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$
```

#### 梯度下降
```latex
$$
w_{t+1} = w_t - \eta \nabla_w L(w_t)
$$
```

#### 注意力机制
```latex
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

#### 卷积操作
```latex
$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
$$
```

### 5. 概率统计

#### 正态分布
```latex
$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
```

#### 贝叶斯公式
```latex
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
```

### 6. 多行公式

使用 `align` 环境：

```latex
$$
\begin{align}
f(x) &= (x+1)^2 \\
     &= x^2 + 2x + 1
\end{align}
$$
```

使用 `cases` 环境（分段函数）：

```latex
$$
f(x) = \begin{cases}
x^2 & \text{if } x \geq 0 \\
-x^2 & \text{if } x < 0
\end{cases}
$$
```

## 🎨 公式样式

### 自动应用的样式

1. **块级公式**：
   - 居中显示
   - 上下边距：1.5rem
   - 支持横向滚动（公式过长时）
   - 紫色滚动条

2. **行内公式**：
   - 与文本对齐
   - 左右边距：0.2em
   - 颜色：深灰色 `#2d3748`

3. **可选择性**：
   - 所有公式文本可选中
   - 方便复制公式内容

### 响应式设计

- **移动端**：块级公式字体缩小至 90%，避免溢出

## 🔧 MathJax 配置详情

### 支持的语法

```javascript
MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)']],           // 行内公式
    displayMath: [['$$', '$$'], ['\\[', '\\]']],  // 块级公式
    processEscapes: true,                   // 处理转义字符
    processEnvironments: true,               // 处理环境（如 align）
    tags: 'ams',                            // AMS 数学包
    packages: {'[+]': ['ams', 'newcommand', 'configmacros']}
  }
}
```

### 支持的LaTeX包

- ✅ **ams** - 美国数学学会扩展
- ✅ **newcommand** - 自定义命令
- ✅ **configmacros** - 配置宏

### CDN源

使用 **jsDelivr CDN**：

```html
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
```

**备选CDN**（如果jsDelivr不可用）：
- CDNJS: `https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js`
- unpkg: `https://unpkg.com/mathjax@3/es5/tex-svg.js`

## 📚 高级用法

### 1. 自定义宏

在文章开头定义，全文可用：

```latex
$$
\newcommand{\argmax}{\mathop{\arg\max}}
\newcommand{\argmin}{\mathop{\arg\min}}
$$

后续使用：
$$
\theta^* = \argmax_\theta P(D|\theta)
$$
```

### 2. 颜色

```latex
$$
\textcolor{red}{红色文字} \quad \textcolor{blue}{蓝色文字}
$$
```

### 3. 加粗与斜体

```latex
$$
\mathbf{粗体} \quad \mathit{斜体} \quad \mathbb{R}^n
$$
```

### 4. 特殊符号

```latex
$$
\infty, \partial, \nabla, \in, \subset, \cup, \cap, \rightarrow, \Rightarrow
$$
```

### 5. 括号自适应

```latex
$$
\left( \frac{a}{b} \right) \quad \left[ \frac{a}{b} \right] \quad \left\{ \frac{a}{b} \right\}
$$
```

## 🐛 常见问题

### 问题1：数学符号不显示（如 \times, \sigma 等）

**可能原因**：
1. Kramdown在MathJax之前处理了反斜杠
2. 符号不在公式环境中
3. 下划线与Markdown语法冲突

**解决方案**：

#### ✅ 正确用法

```markdown
# 行内公式 - 必须在 \(...\) 中
这是乘法符号 \(a \times b\) 和希腊字母 \(\sigma\)

# 块级公式 - 必须在 $$...$$ 中
$$
\alpha + \beta \times \gamma = \delta \cdot \sigma
$$
```

#### ❌ 错误用法

```markdown
# 错误1：不在公式环境中
这是 \times 符号  <!-- 会被Markdown处理 -->

# 错误2：使用了错误的分隔符
这是 $\sigma$  <!-- 应该用 \(...\) -->

# 错误3：下划线冲突
$$y_pred_i$$  <!-- _pred_ 会被解释为斜体 -->
```

#### 🔧 配置修复

确保 `_config.yml` 中有以下配置：

```yaml
kramdown:
  math_engine: mathjax
  math_engine_opts:
    preview: true
    preview_as_code: false
```

### 问题2：公式不显示

**可能原因**：
1. MathJax脚本未加载
2. 公式语法错误
3. 被代码块或其他元素包裹

**解决方案**：
```markdown
# ❌ 错误 - 在代码块中
```
$$公式$$
```

# ✅ 正确 - 直接在Markdown中
$$
公式
$$
```

### 问题2：行内公式显示为块级

**原因**：使用了 `$$...$$` 而不是 `\(...\)`

```markdown
# ❌ 错误
这是 $$E=mc^2$$ 公式

# ✅ 正确
这是 \(E=mc^2\) 公式
```

### 问题3：下划线问题（重要！）

**原因**：Markdown会将 `_..._` 解释为斜体，干扰LaTeX公式

**常见错误：**
```markdown
# ❌ 错误 - 下划线会被Markdown处理
$$y_pred_i = w_1 \times x_1$$
<!-- _pred_ 被解释为斜体，_1 也可能有问题 -->

# ❌ 错误 - 变量名带下划线
\(\theta_max\) 和 \(\theta_min\)
```

**正确写法：**
```markdown
# ✅ 方法1：使用大括号
$$y_{pred_i} = w_1 \times x_1$$

# ✅ 方法2：使用大括号（推荐）
$$y_{pred\_i} = w_{1} \times x_{1}$$

# ✅ 方法3：对于简单下标，确保在公式环境中
\(\theta_{max}\) 和 \(\theta_{min}\)
```

**最佳实践：**
- 所有多字符下标都用大括号：`a_{bc}` 而不是 `a_bc`
- 所有包含下划线的表达式都放在公式环境中
- 避免在公式外使用下划线

### 问题4：特殊字符转义

**需要转义的字符**：`\`, `{`, `}`, `$`

```markdown
# 显示 $ 符号
使用 \\$ 而不是 $

# 显示反斜杠
使用 \\\\ 或在公式中用 \backslash
```

### 问题5：公式过长溢出

**解决方案**：
1. 使用 `align` 拆分为多行
2. 使用缩写或简化符号
3. 依赖自动横向滚动

```latex
# 拆分长公式
$$
\begin{align}
\text{result} &= \text{very long expression part 1} \\
              &\quad + \text{very long expression part 2} \\
              &\quad + \text{very long expression part 3}
\end{align}
$$
```

## 🎯 最佳实践

### 1. 公式编号

使用 `\tag{}`：

```latex
$$
E = mc^2 \tag{1}
$$
```

### 2. 对齐等号

```latex
$$
\begin{align}
x &= a + b \\
  &= c + d \\
  &= e
\end{align}
$$
```

### 3. 公式注释

```latex
$$
\underbrace{x^2 + y^2}_{\text{勾股定理}} = z^2
$$
```

### 4. 矩阵省略

```latex
$$
\begin{bmatrix}
a_{11} & \cdots & a_{1n} \\
\vdots & \ddots & \vdots \\
a_{m1} & \cdots & a_{mn}
\end{bmatrix}
$$
```

## 🔍 调试技巧

### 查看MathJax状态

打开浏览器控制台（F12），应该看到：
```
MathJax 渲染完成
```

### 检查公式错误

如果公式显示为红色，右键点击公式 → "Show Math As" → "TeX Commands" 查看源码

### 性能优化

- ✅ 避免过多的小公式（合并为一个）
- ✅ 复杂公式使用图片代替（如果不需要复制）
- ✅ 使用 `\text{}` 包裹文字，避免误解释

## 📖 参考资料

### 官方文档
- [MathJax 文档](https://docs.mathjax.org/en/latest/)
- [LaTeX数学符号](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)
- [AMS-LaTeX指南](https://www.ams.org/publications/authors/tex/amslatex)

### 在线工具
- [LaTeX公式编辑器](https://latexeditor.lagrida.com/)
- [MathJax在线测试](https://www.mathjax.org/#demo)
- [Detexify](http://detexify.kirelabs.org/classify.html) - 手写识别LaTeX符号

### 速查表
- [Overleaf数学符号表](https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols)
- [LaTeX Wiki](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

## ✅ 快速检查清单

创建数学公式前的检查：

- [ ] 确定是行内 `\(...\)` 还是块级 `$$...$$`
- [ ] 检查括号、大括号是否配对
- [ ] 特殊字符是否正确转义
- [ ] 下标上标是否使用大括号（多字符时）
- [ ] 公式是否在代码块外
- [ ] 复杂公式是否拆分为多行

## 🎨 示例文章

查看以下博客文章了解实际应用：
- [注意力机制详解](/_posts/2025-01-24-attention-mechanism-explained.md)
- [Vision Transformer与Swin Transformer](/_posts/2025-01-28-vision-transformer-swin-transformer.md)
- [ResNet与ResNeXt](/_posts/2025-01-16-resnet-resnext-residual-revolution.md)

---

*最后更新：2025年10月*
*MathJax版本：3.x*

