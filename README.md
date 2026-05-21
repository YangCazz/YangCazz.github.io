# YangCazz.github.io

<div align="center">

![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-v232-blue?style=flat-square)
![Ruby](https://img.shields.io/badge/Ruby-3.0+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

基于 Jekyll + GitHub Pages 的个人技术网站，涵盖博客、简历、项目展示、应用集等模块。

[在线预览](https://yangcazz.github.io)

</div>

---

## 功能特性

- **粒子背景** — 动态交互式粒子系统，响应鼠标移动
- **深色主题** — 全局暗色设计，accent-color 驱动的蓝色点缀体系
- **Octicon 图标** — 全站使用 GitHub Octicons SVG 矢量图标
- **博客系统** — Markdown 写作、Rouge 语法高亮、标签分类、关键词网络可视化、日历浏览
- **简历页面** — 工作经历时间线、技能进度条、教育背景卡片
- **项目展示** — 全屏滚动展演（scroll-snap），支持图片/视频/iframe 媒体嵌入
- **应用集** — 专业项目、开源项目、实用工具、研究项目分类展示与筛选
- **PWA 支持** — manifest.json + favicon
- **SEO 优化** — JSON-LD 结构化数据、OG/Twitter 卡片、jekyll-feed、jekyll-sitemap
- **响应式布局** — 适配桌面、平板、手机

---

## 技术栈

- **静态站点**: Jekyll (GitHub Pages)
- **样式**: SCSS 模块化 (17 个 partial，`_sass/` 目录管理)
- **JavaScript**: 原生 ES6+ (8 个独立模块，`assets/js/`)
- **图标**: [jekyll-octicons](https://github.com/primer/octicons) v19.8.0
- **插件**: jekyll-feed, jekyll-sitemap, jemoji
- **部署**: GitHub Pages

---

## 本地开发

### 前置要求

- Ruby 3.0+
- Bundler

### 安装与运行

```bash
git clone https://github.com/YangCazz/YangCazz.github.io.git
cd YangCazz.github.io
bundle install
bundle exec jekyll serve
```

访问 http://localhost:4000

### 构建

```bash
bundle exec jekyll build   # 输出到 _site/
bundle exec jekyll clean   # 清理缓存
```

---

## 项目结构

```
YangCazz.github.io/
├── _config.yml              # Jekyll 配置
├── Gemfile                  # Ruby 依赖
│
├── _data/                   # 数据文件
│   ├── navigation.yml           # 全局导航
│   ├── home_navigation.yml      # 首页导航
│   ├── blog_navigation.yml      # 博客页导航
│   ├── resume_navigation.yml    # 简历页导航
│   ├── apps_navigation.yml      # 应用页导航
│   ├── showcase_navigation.yml  # 展示页导航
│   ├── applications.yml         # 应用列表
│   ├── github_projects.yml      # GitHub 开源仓库
│   └── showcase_media.yml       # 展示页媒体配置
│
├── _includes/
│   ├── app-card.html            # 应用卡片组件
│   └── github-project-card.html # GitHub 仓库卡片
│
├── _layouts/
│   ├── default.html             # 默认布局（导航、页脚、粒子背景）
│   └── post.html                # 博客文章布局（MathJax、结构化数据）
│
├── _posts/                  # 博客文章 (Markdown)
│
├── _sass/                   # SCSS 模块化样式
│   ├── _variables.scss          # CSS 变量 & 全局动画
│   ├── _mixins.scss             # 可复用 mixin
│   ├── _layout.scss             # 主布局
│   ├── _navigation.scss         # 顶部导航 & 侧边栏
│   ├── _footer.scss             # 页脚
│   ├── _components.scss         # 通用组件（按钮、头像、日历、TOC）
│   ├── _home.scss               # 首页
│   ├── _blog-list.scss          # 博客列表
│   ├── _blog-post.scss          # 博客文章
│   ├── _tag-network.scss        # 关键词网络图
│   ├── _resume.scss             # 简历页
│   ├── _apps.scss               # 应用展示页
│   ├── _about.scss              # 展示页内容区
│   ├── _showcase.scss           # 展示页滚动框架
│   ├── _enhancements.scss       # 代码复制、公式样式
│   └── _highlight-syntax.scss   # 代码高亮
│
├── assets/
│   ├── styles.scss              # SCSS 入口 (仅 @import)
│   ├── js/                      # JavaScript 模块
│   │   ├── particle-system.js   #   粒子背景
│   │   ├── navigation.js        #   导航系统
│   │   ├── tag-network.js       #   关键词力导向图
│   │   ├── calendar.js          #   博客日历面板
│   │   ├── blog-toc.js          #   文章目录生成
│   │   ├── code-copy.js         #   代码块复制按钮
│   │   ├── click-effect.js      #   鼠标波纹效果
│   │   └── back-to-top.js       #   回到顶部
│   └── images/                  # 图片资源
│
├── _scripts/                # 辅助脚本
│   ├── create-blog.js           # 博客文章创建
│   ├── blog-helper.bat          # Windows 博客管理
│   └── blog-helper.sh           # Linux/Mac 博客管理
│
├── apps/                    # 子应用
│   └── PixelKnit/               # PixelKnit 像素织梦
│
├── index.html               # 首页
├── blog.html                # 博客列表
├── resume.html              # 简历
├── apps.html                # 应用展示
├── showcase.html            # 项目展示
├── manifest.json            # PWA 配置
└── favicon.ico              # 网站图标
```

---

## 内容管理

### 创建博客文章

在 `_posts/` 下创建 `YYYY-MM-DD-title.md`：

```yaml
---
layout: post
title: "文章标题"
date: 2024-01-15 10:00:00 +0800
categories: [深度学习]
tags: [PyTorch, CNN]
---
```

或使用脚本：`node _scripts/create-blog.js "标题" "分类" "标签"`

### 配置导航

编辑 `_data/xxx_navigation.yml`：

```yaml
- id: about
  name: 自我总结
  icon: person
```

可用的 octicon 名称参考 [octicons v19.8.0](https://github.com/primer/octicons/tree/v19.8.0)。

### 展示页媒体

编辑 `_data/showcase_media.yml`，取消注释即可配置图片/视频/在线 PPT：

```yaml
showcase-surgery-nav:
  type: image
  src: /assets/images/showcase/surgery-nav.png
  caption: "手术导航系统界面"
```

---

## 部署

推送到 `main` 分支，GitHub Pages 自动构建部署。

---

## 作者

**YangCazz / 杨钱俊** — 医疗机器人算法工程师

- GitHub: [@YangCazz](https://github.com/YangCazz)
- Email: yangcazz@qq.com
