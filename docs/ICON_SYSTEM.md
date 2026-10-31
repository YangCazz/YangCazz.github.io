# 图标系统：Outline / 24

核查日期：2026-10-30。基线：`71090bd7020861829a82873a1b8170e35b22d791`。

## 本轮交付

- **67 枚原创 SVG 功能图标**：24 × 24 坐标网格，1.75 px 圆端/圆连接描边，`currentColor`；无图标字体、CDN、emoji 或栅格生成依赖。
- **3 枚独立品牌标志**：GitHub、LinkedIn、Twitter 的既有页脚路径原样归档，不将商标重画成原创，不暗示官方背书。网站 favicon、头像、PixelKnit logo 也是品牌/内容资产，不混入功能图标库。
- 一个源生成 standalone SVG、sprite、动态 JS helper、明暗目录和带 SHA-256 的版本化资产清单。
- 统一顶部/侧栏导航、页脚、首页、博客列表、简历、应用卡、文章目录/回顶/系列导航、日历/标签弹窗、复制代码按钮及两版 PixelKnit 控件。
- 同步替换 CSS 伪元素中的箭头/对勾，使用本地 SVG mask。普通段落的引号、列表圆点、数学符号不属于交互图标，保留。

没有将研究正文中的历史 emoji、LaTeX、论文结构图或用户像素画当作 UI 图标批量改写。动画汉堡菜单的三条 CSS 线是现有本地结构绘制，不是字符或字体图标；保留交互动画。

## 审计：原先是否都在本地库？

**不是统一资产库。** 基线源文件中发现 79 个 Octicon 调用、26 个内联 SVG，以及 HTML/JS/CSS/导航数据中的字符图标。完整逐文件候选清单见 `icon-audit-before-20261030.json`；计数是源码出现次数，不是浏览器展开后的实例数。Unicode 字段也包含注释候选，不能当作待替换数量。

| 原来源 | 是否本地 | 处理 |
| --- | --- | --- |
| `jekyll-octicons` Liquid 调用 | 构建依赖已安装后本地，图形源不在仓库统一库 | UI 调用全部移除；普通图标改原创，GitHub 标志改本地既有品牌路径 |
| HTML / JS 内联 SVG | 本地、重复且风格分散 | 收敛至统一源；品牌另外管理 |
| 独立应用 emoji / Unicode 导航和按钮 | 由操作系统字体提供，不是受控图片资产 | 改本地 sprite / 原创动态 SVG，补按钮 accessible name |
| CSS `content` 箭头/对勾 | 依赖系统字符形状 | 改本地 SVG mask |
| 网站 favicon、头像、PixelKnit logo | 本地现有文件 | 保留品牌识别，不伪造重新授权或替换原照片 |
| Font Awesome / 图标字体 CDN | 审计范围未发现 | 不引入 |
| 两版 PixelKnit Google Fonts 引用 | 仍有外部 **正文文字字体**，不是图标字体 | 本轮不把“UI 图标已本地化”误称为“全站所有资源离线化” |

历史文章仍可能有 jemoji 渲染或外部素材；访问计数徽章是外部动态服务。这些不属于本轮原创 UI 图标资产，不作全站无外链承诺。Gemfile/Gemfile.lock 中已安装的 Octicons 依赖本轮未更改，避免无关依赖升级；当前 UI 不再调用它。

## 文件结构

```text
_data/icons.json                   # 原创名称、中文用途、SVG path（唯一绘图源）
_data/brand_icons.json              # 既有品牌路径，严格单独管理
_includes/icon.html                # Jekyll inline SVG（继承颜色，无外部请求）
_includes/brand-icon.html          # 品牌标志 include
assets/icons/*.svg                 # 原创独立图标，可用于CSS mask /编辑器
assets/icons/sprite.svg            # 静态HTML使用；不要加隐藏display:none到symbol
assets/icons/brands/*.svg          # 本地品牌标志
assets/icons/catalog.html          # 67 枚，24/16 px 明暗对照预览
assets/icons/icons.css             # 布局、描边及对旧filled样式的兼容
assets/js/icons.js                 # 生成的动态SVG helper
_data/visual_assets/icons-20261030.json # 70条资产记录、用途/源/日期/权利/hash
_scripts/build-icons.py            # 只用Python标准库生成
_scripts/test-icons.mjs            # 图标回归
```

保留 `.octicon` CSS class 仅为旧版间距/颜色兼容，**不代表继续使用 Octicons 图形**。`.site-icon` 禁止 fill，品牌 `.brand-icon` 则保留实心填充。

## 使用

### Jekyll 页面/文章

```liquid
{% include icon.html name="research" size=20 %} 研究笔记
{% include icon.html name="medical-image" size=24 label="医学影像" %}
<button aria-label="关闭目录">{% include icon.html name="close" size=18 %}</button>
{% include brand-icon.html name="github" size=20 %} GitHub
```

紧邻可见文字的图标不写 `label`，默认 `aria-hidden="true"` + `focusable="false"`，避免读屏重复朗读。独立有含义的图像写 `label`（输出 `role="img"` 和转义后的 `aria-label`）。纯图标按钮的名称放在 **button**，不要依赖 `title` 或 SVG 内容。

### 独立 HTML

```html
<link rel="stylesheet" href="../../assets/icons/icons.css">
<button aria-label="下一项">
  <svg class="site-icon" width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <use href="../../assets/icons/sprite.svg#chevron-right"></use>
  </svg>
</button>
```

sprite 必须同源 HTTP(S)；直接双击 `file://` 可能被浏览器跨文件策略拦截。应用按现有 Jekyll/HTTP 方式运行。独立 SVG 用 `<img>` 时继承不到父级 `currentColor`；需要主题适配请用 include、sprite 或 CSS mask。

### JS 动态按钮

独立 app 先加载 `assets/js/icons.js`；共享 ES module 中使用 `import './icons.js'`：

```js
button.innerHTML = SiteIcons.render('copy');
button.setAttribute('aria-label', '复制地图');
```

渲染器只接受资产名单内的 name，并清洗 class 参数。不要往 SVG 注入用户提供的 path 或 HTML。`currentColor` 使用既有界面主题，适配深浅色；图标和按钮背景仍须保持足够对比度。

## 新文章/页面选型

- 导航/操作：`home article showcase apps menu search filter close arrow-* chevron-* external-link link list`
- 内容/研究：`book archive research layers network medical-image robot-arm database chart image`
- 编程/工程：`code repository branch cpu desktop server broadcast tools settings key shield`
- 编辑：`copy check edit trash undo redo plus minus play pause upload download knit`
- 元信息：`person mail phone location briefcase education idea globe info eye star calendar clock warning`

先复用现有语义，不为每篇文章添加一个近似图标。复杂架构应画独立示意图，不把内容硬塞进 24 px 图标。16 px 更适合紧凑操作；标题/导航用 18–24 px。图标小不代表点击区域也小。

## 修改、验证与回退

1. 只编辑 `_data/icons.json`（品牌修改必须单独核查来源/权利）。
2. `python3 _scripts/build-icons.py`；不要手改生成产物。
3. `npm test`、`python3 _scripts/check-visual-assets.py`、常规 Jekyll build。
4. 在 `/assets/icons/catalog.html` 检查 24/16 px 明暗显示，并在受影响页面实测。新变更请创建新的资产批次/记录核查日期，不伪称旧日期下已核验未来内容。
5. 发布前备份现有 Git commit；本轮无数据库迁移。回退本轮提交即可恢复旧模板/资产；不能只删 SVG 而保留 include/JS 引用。

2026-10-30 本地测试：7 个新图标回归通过（XML、源一致性、引用解析、UI 字符残留、按钮标签、品牌隔离）；包含其他维护的 `npm test` 当前 29 个 Node + 13 个 Python 通过。Jekyll 独立构建通过；94 个 HTML / 3210 个本地引用检查无错误，Ruby 内容验证 17 runs / 481 assertions 通过，metadata 检查通过。实际云浏览器已渲染目录：67 枚 × 明暗 × 24/16 px = 268 个 SVG，1180 px 视口无横向溢出；目视确认无填充黑块、线条一致。截图保存于共享验收目录 `blog-visual-review-20261030/icon-catalog.jpg`。两版应用动态按钮已用本地图标；PixelKnit 复制→保存→编辑实操通过。另对工具条拥挤时中文竖排作局部修复：图标与文案并列、按钮不压缩、行内自然换行、字段标签成对；窄栏复验通过：1180 px 视口、内容 1165 px、左右栏 260/240 px 同时显示时，工具条宽 487 px，6 个按钮都有 1 个 SVG、文字 nowrap 且不超出工具条边界。截图 `blog-visual-review-20261030/editor-narrow-both-sidebars.jpg`。**390 px 手机视口未实测**：云浏览器 DevTools 被组织策略禁用，停止尝试；不将 CSS 响应式规则或桌面截图冒充手机验收。
