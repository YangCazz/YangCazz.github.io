# MathJax 故障排除指南

## 问题描述
数学公式显示为 `$$` 包含的英文字符，而不是渲染的数学公式。

## 可能的原因

### 1. MathJax 脚本加载失败
**症状**：公式显示为原始文本
**原因**：
- 网络连接问题
- CDN 不可用
- 脚本路径错误

**解决方案**：
```html
<!-- 检查脚本是否正确加载 -->
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
```

### 2. MathJax 配置错误
**症状**：公式不渲染
**原因**：
- 配置语法错误
- 包加载失败
- 宏定义冲突

**解决方案**：
```javascript
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  },
  svg: {
    fontCache: 'global'
  }
};
```

### 3. CSS 样式冲突
**症状**：公式渲染但显示异常
**原因**：
- CSS 样式覆盖
- 字体冲突
- 布局问题

**解决方案**：
```css
/* 确保 MathJax 样式不被覆盖 */
mjx-container {
  display: inline-block;
  margin: 0;
  padding: 0;
}
```

### 4. 浏览器兼容性问题
**症状**：某些浏览器不显示公式
**原因**：
- 浏览器不支持
- JavaScript 被禁用
- 安全策略限制

**解决方案**：
- 使用现代浏览器
- 启用 JavaScript
- 检查安全策略

## 诊断步骤

### 步骤 1：检查网络连接
```bash
# 检查 CDN 是否可访问
curl -I https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js
```

### 步骤 2：检查浏览器控制台
1. 打开浏览器开发者工具 (F12)
2. 查看 Console 标签
3. 查找错误信息

### 步骤 3：检查 MathJax 加载状态
```javascript
// 在浏览器控制台中运行
console.log('MathJax loaded:', typeof MathJax !== 'undefined');
console.log('MathJax version:', MathJax?.version);
```

### 步骤 4：检查公式语法
```html
<!-- 正确的公式语法 -->
<p>行内公式：$x^2 + y^2 = z^2$</p>
<p>块级公式：$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})$$</p>
```

## 常见错误及解决方案

### 错误 1：MathJax is not defined
**原因**：MathJax 脚本未加载
**解决方案**：
- 检查网络连接
- 使用本地 MathJax 文件
- 更换 CDN

### 错误 2：公式不渲染
**原因**：配置错误
**解决方案**：
- 简化配置
- 检查语法
- 使用默认配置

### 错误 3：公式显示异常
**原因**：CSS 冲突
**解决方案**：
- 检查 CSS 样式
- 添加 MathJax 样式
- 修复布局问题

### 错误 4：性能问题
**原因**：配置过于复杂
**解决方案**：
- 简化配置
- 使用字体缓存
- 优化渲染选项

## 测试方法

### 方法 1：使用测试页面
1. 打开 `simple-math-test.html`
2. 检查公式是否正确渲染
3. 查看控制台错误信息

### 方法 2：检查网络请求
1. 打开浏览器开发者工具
2. 查看 Network 标签
3. 检查 MathJax 脚本是否加载成功

### 方法 3：验证配置
```javascript
// 在浏览器控制台中运行
MathJax.startup.document.state(0);
```

## 解决方案

### 方案 1：简化配置
```javascript
MathJax = {
  tex: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$']],
    processEscapes: true
  }
};
```

### 方案 2：使用本地文件
```html
<script src="/assets/js/mathjax/tex-svg.js"></script>
```

### 方案 3：更换 CDN
```html
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
```

### 方案 4：添加错误处理
```javascript
MathJax = {
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        console.log('MathJax 渲染完成');
      }).catch((error) => {
        console.error('MathJax 渲染失败:', error);
      });
    }
  }
};
```

## 预防措施

### 1. 使用稳定的 CDN
- 选择可靠的 CDN 服务
- 配置备用 CDN
- 监控 CDN 状态

### 2. 简化配置
- 避免过于复杂的配置
- 使用默认配置
- 逐步添加功能

### 3. 测试兼容性
- 测试不同浏览器
- 检查移动设备
- 验证打印效果

### 4. 监控性能
- 检查加载时间
- 监控内存使用
- 优化渲染速度

## 总结

MathJax 公式不渲染的问题通常由以下原因造成：
1. 脚本加载失败
2. 配置错误
3. CSS 冲突
4. 浏览器兼容性问题

通过系统性的诊断和测试，可以快速定位和解决问题。
