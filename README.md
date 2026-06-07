<div align="center">

# YangCazz.github.io

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-deployed-blue?style=flat-square&logo=github)](https://yangcazz.github.io)
[![Jekyll](https://img.shields.io/badge/Jekyll-4.x-cc0000?style=flat-square&logo=jekyll)](https://jekyllrb.com)
[![Ruby](https://img.shields.io/badge/Ruby-3.0+-cc342d?style=flat-square&logo=ruby)](https://www.ruby-lang.org)
[![Posts](https://img.shields.io/badge/posts-72-3b82f6?style=flat-square)](https://yangcazz.github.io/blog)
[![Tags](https://img.shields.io/badge/tags-64-8b5cf6?style=flat-square)](https://yangcazz.github.io/tag-network-v2)
[![License](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)

个人技术博客 &middot; 医疗AI · 深度学习 · 开源工程

[在线预览](https://yangcazz.github.io) &nbsp;|&nbsp; [博客](https://yangcazz.github.io/blog) &nbsp;|&nbsp; [关键词网络](https://yangcazz.github.io/tag-network-v2)

</div>

---

<table align="center"><tr>
<td align="center"><b>72</b><br><sub>Posts</sub></td>
<td align="center"><b>64</b><br><sub>Tags</sub></td>
<td align="center"><b>8</b><br><sub>JS Modules</sub></td>
<td align="center"><b>17</b><br><sub>SCSS Partials</sub></td>
</tr></table>

---

## ▸ 功能特性

<table>
<tr><td width="50%">

**粒子背景**<br><sub>动态交互粒子系统，响应鼠标移动</sub>

**关键词网络**<br><sub>力导向图可视化 · 规范标签注册 · domain/method 双轨分类</sub>

**模块化样式**<br><sub>17 个 SCSS partial · accent-color 驱动暗色主题</sub>

</td><td width="50%">

**博客系统**<br><sub>Markdown + Rouge 语法高亮 · 标签网络 · 日历视图 · 目录生成</sub>

**PWA + SEO**<br><sub>manifest.json · JSON-LD 结构化数据 · OG/Twitter 卡片 · sitemap</sub>

**一键部署**<br><sub>Push to main → GitHub Pages 自动构建</sub>

</td></tr>
</table>

---

## ▸ 技术栈

| 类别 | 技术 |
|------|------|
| 静态站点 | Jekyll, GitHub Pages |
| 样式 | SCSS (17 partials), CSS Variables |
| 脚本 | Vanilla ES6+ (8 modules) |
| 图标 | Octicons v19.8.0 |
| 排版 | MathJax 3, Mermaid, Rouge |
| 插件 | jekyll-feed, jekyll-sitemap, jemoji |

---

## ▸ 快速开始

```bash
git clone https://github.com/YangCazz/YangCazz.github.io.git
cd YangCazz.github.io
bundle install
bundle exec jekyll serve
```

打开 `http://localhost:4000`

---

## ▸ 项目结构

```
YangCazz.github.io/
├── _config.yml                   # Jekyll 配置
├── _data/                        # 数据文件
│   ├── tags.yml                  #   标签注册表 (64 规范标签)
│   ├── navigation.yml            #   全局导航配置
│   └── applications.yml          #   应用列表数据
├── _layouts/                     # 布局模板
│   ├── default.html              #   默认布局 (粒子背景)
│   └── post.html                 #   文章布局 (MathJax + SEO)
├── _includes/                    # 可复用组件
├── _posts/                       # 博客文章 (72 篇)
├── _sass/                        # SCSS 模块 (17 partials)
├── assets/
│   ├── js/                       # JavaScript 模块 (8 files)
│   └── images/                   # 图片资源
├── _plugins/                     # Jekyll 插件
│   └── tag_validator.rb          #   标签校验 (domain/method 双轨)
├── apps/                         # 子应用
│   └── PixelKnit/                #   PixelKnit 像素织梦
├── blog.html                     # 博客列表 + 关键词网络
├── resume.html                   # 在线简历
├── apps.html                     # 应用展示
└── showcase.html                 # 项目展演
```

---

## ▸ 内容管理

### 创建文章

在 `_posts/` 下创建 `YYYY-MM-DD-slug.md`：

```yaml
---
layout: post
title: "文章标题"
date: 2026-06-07 10:00:00 +0800
categories: [深度学习]
tags: [PyTorch, CNN, 计算机视觉]
excerpt: "一句话摘要"
image: /assets/images/covers/ai-dev-tools.jpg
---
```

### 标签规范

所有标签必须在 `_data/tags.yml` 注册表中注册，每条标签包含类型 (`domain` / `method`)、别名映射和参考文献链接。每篇文章上限 8 个标签，至少包含 1 个领域标签和 1 个方法标签。

构建时自动校验：`bundle exec jekyll build`

---

## ▸ 作者

**YangCazz / 杨钱俊** — 医疗机器人算法工程师

[GitHub](https://github.com/YangCazz) &nbsp;·&nbsp; [Email](mailto:yangcazz@qq.com) &nbsp;·&nbsp; [Blog](https://yangcazz.github.io)
