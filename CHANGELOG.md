# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 规范。

## [Unreleased]

### Added
- 关键词网络 V2 (`/tag-network-v2`) — 力导向图可视化，支持搜索/拖拽/右键固定
- 标签注册体系 (`_data/tags.yml`) — 64 个规范标签，domain/method 双轨分类
- Jekyll 标签校验插件 (`_plugins/tag_validator.rb`) — 构建时自动校验 post frontmatter
- 博客动画机制更新 — GSAP 驱动的时间线动画系统

### Changed
- 全站 72 篇文章标签迁移至规范注册表，精简 60%（160 → 64 标签）
- README 全面改版
- 博客页样式与交互优化

### Fixed
- 移动端导航适配
- 代码块复制按钮样式修复

## [2026-05]

### Added
- Claude Code 实战指南系列 (12 篇)
- AI Agent 从零构建系列 (8 篇)
- Agent Skills 架构与设计模式系列
- Mamba 状态空间模型深度解析
- DICOM/PACS 医疗信息化全面指南
- 拓扑结构学习与血管分割
- WiseSurgery 手术智能识别系统
- CazzSegmentator 医学图像分割框架
- CazzAI 流式语音助手架构

### Changed
- 博客写作规范标准化 (CLAUDE.md)
- 移动端响应式全面优化
- 代码高亮主题更新
- 粒子背景性能优化

### Fixed
- 关键词网络移动端触控
- 代码块内嵌公式渲染
- Mermaid 流程图中文兼容

## [2024-2025]

### Added
- PixelKnit 像素织梦 (v1.0 → v2.0)
- 项目展演页 (scroll-snap 全屏滚动)
- 应用展示页 (分类筛选)
- 博客日历视图
- Octicon 图标系统集成

### Changed
- SCSS 模块化重构 (17 partials)
- JavaScript 模块化拆分 (8 modules)
- PWA 支持 (manifest.json)
- SEO 结构化数据 (JSON-LD, OG, Twitter Cards)

## [2021-2023]

### Added
- CNN 经典架构系列 (LeNet → EfficientNet)
- YOLO 目标检测系列 (v1 → v8)
- U-Net 医学图像分割系列
- GNN 图神经网络系列
- Transformer / Vision Transformer 解析
- ResNet / GoogleNet / MobileNet / ShuffleNet 论文精读

### Changed
- 从 Jekyll 默认主题迁移至自定义暗色主题
- Rouge 语法高亮配置
- MathJax 数学公式渲染
