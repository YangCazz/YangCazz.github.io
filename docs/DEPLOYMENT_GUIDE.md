# 🚀 GitHub Pages 部署问题解决指南

## 问题：本地正常，GitHub Pages 上格式丢失

### 常见原因

1. **浏览器缓存**：浏览器缓存了旧版本的CSS
2. **GitHub Pages编译延迟**：需要等待几分钟
3. **CSS文件路径问题**：相对路径和绝对路径混用
4. **Jekyll编译配置**：SASS编译设置不正确

### 解决步骤

#### 1. 提交最新更改

```bash
git add .
git commit -m "优化Jekyll SASS编译配置"
git push origin main
```

#### 2. 等待GitHub Pages重新部署

- 访问 https://github.com/YangCazz/YangCazz.github.io/actions
- 等待"pages build and deployment"工作流完成（通常需要1-3分钟）
- 看到绿色的✓表示部署成功

#### 3. 强制刷新浏览器缓存

**Windows:**
```
Ctrl + Shift + R
```

**Mac:**
```
Cmd + Shift + R
```

**或者使用开发者工具：**
1. 按 `F12` 打开开发者工具
2. 右键点击刷新按钮
3. 选择"清空缓存并硬性重新加载"

#### 4. 验证CSS文件

访问以下URL检查CSS是否正确生成：
```
https://yangcazz.github.io/assets/styles.css
```

应该能看到包含以下样式的内容：
- `.apps-header-section`
- `.apps-main-title`
- `.app-card`
- 等等

#### 5. 检查控制台错误

1. 按 `F12` 打开开发者工具
2. 切换到 "Console" 标签
3. 刷新页面
4. 查看是否有404错误或其他错误信息

### 配置优化说明

我在 `_config.yml` 中添加了以下配置：

```yaml
sass:
  style: compressed
```

**作用：**
- `compressed`: 压缩CSS，减小文件大小，加快加载速度
- 确保Jekyll正确处理SASS文件

### 其他可能的解决方案

#### 方案A：清除GitHub Pages缓存

1. 进入项目设置：https://github.com/YangCazz/YangCazz.github.io/settings/pages
2. 临时禁用GitHub Pages
3. 等待1分钟
4. 重新启用GitHub Pages
5. 等待重新部署

#### 方案B：分离CSS文件

如果问题持续存在，可以将应用页面的样式分离到单独的文件：

1. 创建 `assets/apps.scss`
2. 在应用页面的 front matter 中添加：
```yaml
---
layout: default
title: YangCazz - 应用与项目
custom_css: apps
---
```

#### 方案C：使用版本号

在CSS链接中添加版本参数，强制刷新：

```html
<link href="{{ '/assets/styles.css?v=1.0' | relative_url }}" rel="stylesheet">
```

### 检查清单

- [ ] 提交并推送所有更改到GitHub
- [ ] 等待GitHub Actions完成部署
- [ ] 强制刷新浏览器（Ctrl+Shift+R）
- [ ] 检查CSS文件是否能访问
- [ ] 检查浏览器控制台是否有错误
- [ ] 清除浏览器所有缓存（可选）

### 调试技巧

#### 查看实际加载的CSS

在浏览器开发者工具中：
1. 切换到 "Network" 标签
2. 刷新页面
3. 找到 `styles.css` 文件
4. 点击查看完整内容
5. 搜索 `.apps-header-section` 确认样式存在

#### 临时解决方案

如果急需修复，可以在 `apps.html` 页面末尾添加内联样式：

```html
<style>
.apps-header-section {
    padding: 4.5rem 2rem 2rem;
    text-align: center;
}
/* 其他必要样式... */
</style>
```

### 常见问题

**Q: 推送后要等多久才能看到更新？**
A: 通常1-3分钟，可以在 Actions 页面查看进度。

**Q: 为什么本地正常但线上不行？**
A: 本地Jekyll实时编译，线上有缓存和CDN延迟。

**Q: 如何确认是否是缓存问题？**
A: 使用无痕模式或其他浏览器访问，如果正常则是缓存问题。

**Q: CSS文件404怎么办？**
A: 检查 `assets/styles.scss` 文件开头是否有 `---` 分隔符。

---

**最后更新**: 2025-01-13

