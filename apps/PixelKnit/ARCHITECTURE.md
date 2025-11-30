# PixelKnit 架构设计文档

## 📐 架构概述

本项目采用模块化架构设计，将原有的单体应用拆分为多个独立的模块，提高代码的可维护性和可扩展性。

## 🏗️ 目录结构

```
apps/PixelKnit/
├── index.html                  # 主入口 HTML
├── README.md                   # 项目说明
├── ARCHITECTURE.md             # 架构设计文档（本文件）
├── assets/
│   ├── css/
│   │   └── main.css           # 主样式文件
│   └── js/
│       ├── main.js            # 主入口文件（模块加载器）
│       ├── app.js             # 应用主逻辑（临时，待迁移）
│       ├── core/              # 核心业务模块
│       │   ├── MapManager.js      # 地图管理
│       │   ├── GridRenderer.js    # 网格渲染
│       │   ├── Editor.js          # 编辑器
│       │   ├── PathCalculator.js  # 路径计算
│       │   └── ProgressTracker.js # 进度跟踪
│       ├── ui/                # UI 组件
│       │   ├── Navigation.js      # 导航栏
│       │   ├── MapList.js         # 地图列表
│       │   ├── Toolbar.js         # 工具栏
│       │   ├── ControlPanel.js    # 控制面板
│       │   ├── Magnifier.js       # 放大镜
│       │   └── ColorPalette.js    # 调色板
│       ├── utils/             # 工具类
│       │   ├── Algorithms.js      # 算法工具
│       │   ├── Helpers.js         # 辅助函数
│       │   ├── Storage.js         # 数据持久化
│       │   └── History.js         # 编辑历史
│       └── effects/           # 特效模块
│           └── ParticleSystem.js # 粒子系统
└── data/
    └── json/                  # 数据文件
        ├── maps_list.json
        └── pixel_map_data.json
```

## 📦 模块说明

### 工具类模块 (utils/)

#### Algorithms.js
- **功能**: 路径计算算法
- **方法**:
  - `calculatePath(p1, p2)`: Bresenham 直线算法
  - `calculateAllDiagonalPaths(gridData, colorMap)`: 对角线路径计算

#### Helpers.js
- **功能**: 通用辅助函数
- **方法**:
  - `generateMapId(nextId)`: 生成地图ID
  - `hexToRgb(hex)`: 十六进制转RGB
  - `rgbToString(rgb)`: RGB数组转字符串
  - `coordToKey(row, col)`: 坐标转键
  - `keyToCoord(key)`: 键转坐标
  - `deepClone(obj)`: 深拷贝
  - `throttle(func, delay)`: 节流函数
  - `debounce(func, delay)`: 防抖函数

#### Storage.js
- **功能**: 数据持久化（localStorage 封装）
- **方法**:
  - `saveMapData(mapName, mapData)`: 保存地图数据
  - `loadMapData(mapName)`: 加载地图数据
  - `deleteMapData(mapName)`: 删除地图数据
  - `saveUserMapsList(mapsList)`: 保存用户地图列表
  - `loadUserMapsList()`: 加载用户地图列表
  - `saveProgress(mapName, progressState)`: 保存进度
  - `loadProgress(mapName)`: 加载进度
  - `deleteProgress(mapName)`: 删除进度
  - `saveNextMapId(nextId)`: 保存下一个地图ID
  - `loadNextMapId()`: 加载下一个地图ID

#### History.js
- **功能**: 编辑历史管理（撤销/重做）
- **方法**:
  - `add(gridData, colorMap)`: 添加历史记录
  - `undo()`: 撤销
  - `redo()`: 重做
  - `canUndo()`: 是否可以撤销
  - `canRedo()`: 是否可以重做
  - `clear()`: 清空历史记录
  - `getCurrent()`: 获取当前历史记录

### 特效模块 (effects/)

#### ParticleSystem.js
- **功能**: 粒子背景系统
- **方法**:
  - `init()`: 初始化粒子系统
  - `resizeCanvas()`: 调整画布大小
  - `createParticles()`: 创建粒子
  - `animate()`: 动画循环

### 核心模块 (core/) - 待实现

#### MapManager.js
- **功能**: 地图管理
- **职责**:
  - 地图加载和保存
  - 地图列表管理
  - 地图切换
  - 地图复制和删除

#### GridRenderer.js
- **功能**: 网格渲染
- **职责**:
  - 主网格渲染
  - 小地图渲染
  - 放大镜渲染
  - 状态覆盖层渲染

#### Editor.js
- **功能**: 像素地图编辑器
- **职责**:
  - 编辑模式管理
  - 像素绘制
  - 调色板管理
  - 撤销/重做

#### PathCalculator.js
- **功能**: 路径计算
- **职责**:
  - 手动路径选择
  - 对角线路径计算
  - 路径高亮

#### ProgressTracker.js
- **功能**: 进度跟踪
- **职责**:
  - 完成状态管理
  - 进度计算
  - 进度保存和加载

### UI组件 (ui/) - 待实现

#### Navigation.js
- **功能**: 顶部导航栏
- **职责**: 导航菜单和响应式处理

#### MapList.js
- **功能**: 左侧地图列表
- **职责**: 地图列表显示和交互

#### Toolbar.js
- **功能**: 编辑工具栏
- **职责**: 编辑工具按钮和操作

#### ControlPanel.js
- **功能**: 右侧控制面板
- **职责**: 编织模式控制和进度显示

#### Magnifier.js
- **功能**: 放大镜
- **职责**: 鼠标悬停放大显示

#### ColorPalette.js
- **功能**: 调色板
- **职责**: 颜色选择和显示

## 🔄 模块间通信

### 当前架构
- 工具类通过 `window.PixelKnit` 全局对象暴露
- 各模块通过 ES6 import/export 进行依赖注入
- 事件驱动：通过 DOM 事件进行模块间通信

### 未来优化方向
- 考虑引入事件总线（EventBus）进行模块间通信
- 使用状态管理（State Management）统一管理应用状态
- 考虑使用依赖注入容器（DI Container）

## 📊 重构进度

### ✅ 已完成
- [x] 项目结构设计
- [x] 工具类模块实现
- [x] 粒子系统模块实现
- [x] 基础 HTML 结构
- [x] 样式文件迁移
- [x] 数据文件迁移

### 🚧 进行中
- [ ] 核心模块拆分和实现
- [ ] UI组件拆分和实现
- [ ] 主入口文件完善

### 📋 待完成
- [ ] 完整功能迁移
- [ ] 模块间通信优化
- [ ] 性能优化
- [ ] 单元测试
- [ ] 文档完善

## 🎯 设计原则

1. **单一职责原则**: 每个模块只负责一个明确的功能
2. **依赖注入**: 通过参数传递依赖，避免全局变量
3. **接口隔离**: 模块间通过明确的接口进行通信
4. **开闭原则**: 对扩展开放，对修改关闭
5. **可测试性**: 模块设计便于单元测试

## 🔧 技术选型

- **模块化**: ES6 Modules
- **构建工具**: 暂无（未来可考虑 Webpack/Vite）
- **测试框架**: 待定（未来可考虑 Jest）
- **代码规范**: ESLint（待配置）

## 📝 使用说明

### 开发环境
1. 使用支持 ES6 Modules 的现代浏览器
2. 推荐使用本地服务器（避免 CORS 问题）
3. 使用支持 ES6 的代码编辑器

### 添加新模块
1. 在对应的目录下创建新文件
2. 使用 ES6 export 导出功能
3. 在 main.js 中导入并注册
4. 更新本文档

### 修改现有模块
1. 保持模块接口稳定
2. 更新相关文档
3. 确保向后兼容

## 🚀 未来规划

1. **完整模块化**: 将所有功能迁移到模块化结构
2. **构建系统**: 引入 Webpack 或 Vite 进行构建
3. **类型系统**: 考虑引入 TypeScript
4. **测试覆盖**: 添加单元测试和集成测试
5. **性能优化**: 代码分割、懒加载等
6. **文档完善**: API 文档、使用指南等

---

**最后更新**: 2025-11-30

