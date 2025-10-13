# 博客样式优化指南

## 🎨 已优化的内容

### 1. 代码块优化 ⭐ 新增语法高亮！

#### ✨ 行内代码
- **渐变背景**：紫色渐变背景，白色文字
- **专业字体**：Fira Code, Consolas, Monaco
- **圆角阴影**：4px圆角 + 柔和阴影
- **示例**：`const example = 'code'`

#### 📦 代码块（现已支持语法高亮！）
- **深色主题**：#1e1e1e 背景（VS Code Dark+ 主题）
- **圆角边框**：12px圆角
- **立体阴影**：多层阴影效果
- **自定义滚动条**：半透明滚动条，悬停高亮
- **完整语法高亮**：支持多种编程语言
  - ✅ Python
  - ✅ JavaScript/TypeScript
  - ✅ HTML/CSS/SCSS
  - ✅ Bash/Shell
  - ✅ YAML/JSON
  - ✅ Markdown
  - ✅ 更多...
- **⭐ 一键复制按钮**：
  - 鼠标悬停代码块时自动显示
  - 点击复制所有代码到剪贴板
  - 复制成功显示绿色勾选提示
  - 移动端始终显示
  - 支持新旧浏览器

### 2. 图片优化

#### 🖼️ 图片显示（简洁融入式设计）
- **居中显示**：自动居中对齐
- **相框效果**：白色背景 + 0.5rem内边距
- **细边框**：1px浅色边框，更像文档插图
- **小圆角**：4px圆角，自然不突兀
- **轻量阴影**：0 2px 8px，微妙不抢眼
- **悬停效果**：
  - 阴影轻微加深（4px → 16px）
  - 边框变为紫色调
  - 无位移、无缩放，保持稳定
- **响应式**：max-width: 100%，自适应
- **移动端优化**：内边距减小为0.4rem

### 3. 表格优化

#### 📊 表格样式
- **玻璃态效果**：半透明背景 + 毛玻璃模糊
- **渐变表头**：紫色渐变
- **悬停行**：鼠标悬停时高亮 + 轻微放大
- **条纹背景**：偶数行浅灰背景
- **圆角边框**：12px圆角

### 4. 引用块优化

#### 💬 引用样式
- **左侧边框**：紫色渐变边框
- **引号装饰**：大号引号作为背景装饰
- **渐变背景**：淡紫色渐变背景
- **圆角边框**：右侧圆角
- **柔和阴影**：紫色调阴影

### 5. 列表优化

#### 📝 列表样式
- **彩色标记**：紫色列表标记
- **加粗标记**：font-weight: 600
- **增大行距**：line-height: 1.8
- **适当间距**：每项0.8rem间距

### 6. 数学公式 ⭐ 新增！

#### 🔢 LaTeX公式渲染（MathJax 3）
- **行内公式**：`\(...\)` 格式
  - 与文本对齐
  - 左右边距：0.2em
  - 示例：`\(E = mc^2\)`
  
- **块级公式**：`$$...$$` 格式
  - 居中显示
  - 上下边距：1.5rem
  - 支持横向滚动
  - 示例：`$$\sum_{i=1}^{n} i$$`
  
- **样式特性**：
  - 可选择并复制
  - 深灰色文字（#2d3748）
  - 紫色主题滚动条
  - 移动端字体缩放（90%）
  
- **支持功能**：
  - AMS数学包
  - 矩阵、向量、分数
  - 多行公式（align环境）
  - 分段函数（cases环境）
  - 希腊字母、特殊符号
  - 自定义宏命令

### 7. 其他元素

#### 分割线
- **渐变线条**：中间紫色，两端透明
- **2px高度**：优雅的水平线

#### 响应式设计
- **移动端优化**：
  - 更小的代码字体（0.85rem）
  - 公式字体缩放（0.9em）
  - 适配的表格间距
  - 调整的图片内边距

## 🎯 使用方法

### 在Markdown中使用

#### 代码块（指定语言以启用语法高亮）

**Python示例：**
\`\`\`python
def fibonacci(n):
    """计算斐波那契数列"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 调用函数
result = fibonacci(10)
print(f"结果：{result}")
\`\`\`

**JavaScript示例：**
\`\`\`javascript
const fetchData = async () => {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
};
\`\`\`

**Bash示例：**
\`\`\`bash
#!/bin/bash
echo "开始部署..."
npm install
npm run build
echo "部署完成！"
\`\`\`

**YAML示例：**
\`\`\`yaml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: npm run build
\`\`\`

#### 行内代码
使用 \`代码\` 来高亮显示行内代码。例如：使用 \`const\` 声明常量。

#### 图片
![图片说明](/assets/images/example.png)

#### 数学公式

**行内公式：**
```markdown
这是一个行内公式 \(E = mc^2\)，它会嵌入在文本中。
计算梯度：\(\nabla_\theta L = \frac{\partial L}{\partial \theta}\)
```

**块级公式：**
```markdown
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

注意力机制：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
```

**多行公式：**
```markdown
$$
\begin{align}
\mathbf{h}_t &= \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}) \\
\mathbf{y}_t &= \text{softmax}(\mathbf{W}\mathbf{h}_t + \mathbf{b})
\end{align}
$$
```

**矩阵：**
```markdown
$$
\mathbf{W} = \begin{bmatrix}
w_{11} & w_{12} & w_{13} \\
w_{21} & w_{22} & w_{23} \\
w_{31} & w_{32} & w_{33}
\end{bmatrix}
$$
```

#### 表格
| 列1 | 列2 | 列3 |
|-----|-----|-----|
| 数据1 | 数据2 | 数据3 |

#### 引用
> 这是一段引用文字

#### 列表
- 项目1
- 项目2
  - 子项目2.1
  - 子项目2.2

## 🎨 配色方案

### 主要颜色
- **主紫色**：#667eea
- **副紫色**：#764ba2
- **深色背景**：#1e1e1e
- **浅色背景**：rgba(255, 255, 255, 0.9)

### 语法高亮配色（VS Code Dark+ 主题）

#### 代码主题基础色
- **背景**：#1e1e1e
- **默认文字**：#d4d4d4
- **滚动条**：rgba(255, 255, 255, 0.2)

#### 语法元素配色
- **关键字** (def, class, import)：#569cd6 蓝色
- **字符串** ("hello")：#ce9178 橙色
- **数字** (123, 3.14)：#b5cea8 浅绿色
- **注释** (# comment)：#6a9955 绿色（斜体）
- **函数名**：#dcdcaa 黄色
- **类名**：#4ec9b0 青色
- **变量名**：#9cdcfe 浅蓝色
- **装饰器** (@decorator)：#dcdcaa 黄色
- **内置函数** (print, len)：#4ec9b0 青色
- **布尔值/None** (True, False, None)：#569cd6 蓝色
- **正则表达式**：#d16969 红色
- **转义字符** (\n, \t)：#d7ba7d 金色

#### HTML/XML
- **标签名** (<div>)：#569cd6 蓝色
- **属性名** (class=)：#9cdcfe 浅蓝色
- **属性值** ("value")：#ce9178 橙色

#### CSS/SCSS
- **选择器**：#d7ba7d 金色
- **属性名**：#569cd6 蓝色
- **属性值**：#ce9178 橙色
- **数值单位**：#b5cea8 浅绿色

## 📱 浏览器兼容性

### 支持的浏览器
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### 使用的特性
- CSS渐变
- backdrop-filter（毛玻璃效果）
- transform（变换动画）
- box-shadow（阴影效果）
- 自定义滚动条（-webkit-）

## 🚀 性能优化

### 已实施的优化
1. **硬件加速**：使用transform而非top/left
2. **过渡效果**：cubic-bezier缓动函数
3. **响应式图片**：max-width而非固定宽度
4. **打印样式**：单独的打印媒体查询

## 📝 注意事项

### 图片优化建议
- 使用适当的图片格式（WebP > PNG > JPG）
- 压缩图片以提高加载速度
- 提供alt文本以提高可访问性

### 代码块建议
- 指定语言以启用语法高亮
- 避免过长的单行代码
- 使用有意义的变量名

## 🔧 自定义

如需修改样式，请编辑 `assets/styles.scss` 文件的以下部分：

```scss
/* ========================================
   博客文章增强样式 - 代码块和图片优化
   ======================================== */
```

### 修改建议
- **代码块背景色**：修改 `pre { background: #1e1e1e; }`
- **行内代码颜色**：修改 `code { background: ... }`
- **表格主题色**：修改 `thead { background: ... }`
- **图片圆角**：修改 `img { border-radius: 12px; }`

## 📚 参考资料

- [Markdown语法指南](https://www.markdownguide.org/)
- [CSS渐变生成器](https://cssgradient.io/)
- [阴影生成器](https://shadows.brumm.af/)

---

*样式最后更新：2025年10月*

