---
layout: post
title: "拆解 nature-skills：从一篇论文到 Agent 技能设计的工程化范式"
date: 2026-06-04 10:00:00 +0800
categories: [AI, 软件工程]
tags: [Agent, Skill, Claude Code, nature-skills, 设计模式, SKILL.md, 学术写作, Agent 工程]
excerpt: "以 Agent 工程师的视角深度拆解 nature-skills 项目——不仅是学术写作工具的科普，更是一次对 Agent Skill 设计模式、SKILL.md 规范、渐进披露架构和可复用工作流的系统性工程分析。"
image: /assets/images/covers/ai-dev-tools.jpg
---

## 引言

2026 年 4 月，上海交通大学博士生开源了一个叫 **nature-skills** 的项目<cite>[1]</cite>。一个月内拿到 8,400+ GitHub stars，登上多个技术社区热榜。

如果你只看标题——「AI 帮你写出 Nature 级论文」——它不过是又一个 AI 写作工具。但如果你以 Agent 工程师的视角打开它的源码，你会发现另一件事：**这不是一个写作工具，这是一份 Agent Skill 设计的工程规范**。

nature-skills 的真正价值不在于它的 7 个学术写作技能本身，而在于它展示了一套系统化的方法论：**如何将隐性领域知识转化为 Agent 可执行的规则系统，如何用文件系统的树状结构自然建模任务复杂度，以及如何在不修改一行 Agent 代码的前提下持续扩展能力边界**。

这篇文章以 Agent 工程师的视角，从架构设计、SKILL.md 规范、渐进披露机制、设计模式四个层面，拆解 nature-skills 背后的工程思想。

---

## nature-skills 是什么

nature-skills 是一套为 Claude Code 设计的 Skill 集合<cite>[1]</cite>，目标是让 AI Agent 产出符合 *Nature* 期刊学术规范的论文文本、图表、引用和审稿回复。作者是上交大博士生 Yuan Yizhe。

**七个核心技能：**

| 技能 | 功能 | 关键约束 |
|---|---|---|
| `nature-polishing` | 学术英文润色（中→英） | 每句 ≤30 词，时态审查，overclaim 检测 |
| `nature-writing` | 从零起草论文 | 不编造数据/机制/参考文献 |
| `nature-figure` | 论文级科研绘图 | Arial 字体，SVG 输出，信息层级制 |
| `nature-citation` | 文献引用检索 | 限 CNS 期刊范围，宁可漏引不滥引 |
| `nature-reader` | 论文全文阅读 | 中英对照 Markdown，溯源 ID 系统 |
| `nature-response` | 审稿意见回复 | 结构化逐点回复草稿 |
| `nature-paper2ppt` | 论文转汇报 PPT | 7 个科学问题驱动演讲逻辑 |

**五个共享设计原则<cite>[1]</cite>：**

1. **一手资料优先** — 所有规则源自 Nature 官方指南、已发表论文、结构化写作课程。不是「我感觉应该这么写」，而是「Nature 2026 年第 3 期第 5 页第 3 段这样写了」
2. **显式优于隐式** — 每个规则有明确来源；每个输出有结构、边界标注
3. **章节感知** — 技能理解 Abstract / Results / Discussion 的不同职责，不同段落应用不同时态和语气
4. **输出优先** — 每个技能产出立即可用的成品（.svg 图表、Markdown 文本、.pptx 文件），而非中间分析和评论
5. **可扩展设计** — 每个 Skill 自包含在独立目录中，添加新 Skill 不需要修改现有 Skill

这五个原则，后面你会发现，恰好映射到了 Agent Skill 设计的核心工程问题。

---

## Skill 的目录结构：文件系统即能力模型

nature-skills 的仓库结构非常简洁<cite>[1]</cite>：

```
nature-skills/
├── skills/
│   ├── nature-polishing/
│   │   ├── SKILL.md              # 核心规则 + 12步工作流
│   │   └── README.md
│   ├── nature-figure/
│   │   ├── SKILL.md
│   │   └── references/           # 模块化参考文件
│   │       ├── api.md
│   │       ├── design-theory.md
│   │       └── tutorials.md
│   ├── nature-citation/
│   │   ├── SKILL.md
│   │   └── scripts/
│   │       └── nature_citation.py
│   └── ...
```

这个结构看起来平平无奇——但正是这种「平平无奇」揭示了 Agent Skill 设计的核心原则：**文件系统就是能力模型**。

### 每个 Skill 三要素

| 文件 | 角色 | 类比 |
|---|---|---|
| `SKILL.md` | 核心指令：规则 + 工作流 | API 接口定义 |
| `references/` | 模块化参考：API 文档、设计理论、样式指南 | API 实现细节 |
| `scripts/` | 可执行辅助：自动化脚本 | SDK / 工具函数 |

这种结构的精妙之处在于它天然支持**按需加载**——Agent 不需要一次性读入所有 7 个 Skill 的全部内容。事实上它只需要知道每个 Skill 的 `name` 和 `description`（~100 tokens），直到用户触发某个特定场景时才加载对应的 `SKILL.md`（~3000 tokens），再按需深入 `references/` 子文件<cite>[2]</cite>。

这就是**渐进披露（Progressive Disclosure）**——Agent Skill 架构的心脏。

### 三层渐进披露机制

| 层级 | 加载时机 | 内容 | 上下文成本 |
|---|---|---|---|
| **Tier 1** | Host 启动 | `(name, description)` | ~100 tokens / Skill |
| **Tier 2** | 模型决定激活 | `SKILL.md` 正文 | ~3000 tokens / Skill |
| **Tier 3** | 指令引用 | `references/` 子文件 | 按需递增 |

这个设计在工程上有两个关键后果：

**第一，Skill 的 `description` 字段是召回命门。** 如果 description 写得太宽泛（如「帮助写论文」），模型不知道该何时激活；写得太窄（如「仅在润色 Nature 材料学论文第三段时使用」），永远无法被触发。nature-skills 使用了事实标准的**双句结构**：WHAT 句（动词开头）+ WHEN 句（`Use when...` 开头），兼顾覆盖面与精确度<cite>[2]</cite>。

**第二，Tier 1→Tier 2→Tier 3 的加载链天然形成了一道「复杂度阶梯」。** 简单任务只需要 Tier 1-2，复杂任务才深入 Tier 3。用户的上下文窗口不会被一次性塞满——这在多 Skill 协作场景中至关重要。

---

## SKILL.md 的设计规范

翻开 nature-skills 的任意一个 `SKILL.md`，你会发现它不是一篇散文，而是一份**操作手册**——有编号的工作流步骤、有明确的输入输出契约、有强制的默认行为<cite>[2]</cite>。

### Frontmatter 模板

```yaml
---
name: nature-polishing
description: >-
  Polish academic text to Nature journal style.
  Use when user asks to polish, edit, or improve academic writing,
  wants Nature-level prose quality, or needs Chinese→English translation.
version: 5.0.2
author: Derived from Nature Portfolio author guidelines
trigger: polish / edit / improve academic writing / Nature style
---
```

### 九条编写规律

从 nature-skills 的源码中可以提取九条 SKILL.md 编写规律<cite>[2]</cite>，它们构成了 Agent Skill 设计的实操手册：

| # | 规律 | 为什么重要 |
|---|---|---|
| 1 | **触发时机明确** | description 是 Skill 召回的唯一入口，必须中英文关键词详尽 |
| 2 | **主轻子重** | 核心规则放 `SKILL.md`，场景细节放 `references/`。控制主文件在 ~3000 tokens |
| 3 | **工作流编号** | 显式写「第一步、第二步」，防止 AI 跳过关键步骤 |
| 4 | **默认行为写死** | 无特殊指令时的标准动作必须硬编码，不给模型「即兴发挥」的空间 |
| 5 | **输出格式模板化** | 将输出视为 API 接口——固定字段、固定顺序、固定格式 |
| 6 | **规则溯源** | 每条规则有出处（官方文档/已发表论文/权威课程），而非「我感觉」 |
| 7 | **体量路线灵活** | 简单 Skill = 单文件；复杂 Skill = 主文件 + 子文件 |
| 8 | **示例与测试** | 提供优质输出示范 + 测试用例，作为质量的 baseline |
| 9 | **中文处理专章** | 中文输入/术语转换是高频场景，必须专设章节处理 |

其中第 4 条（默认行为写死）和第 6 条（规则溯源）是 nature-skills 与传统 Prompt Engineering 最根本的区别。

**Prompt Engineering 的范式：** 「请帮我写出 Nature 风格的论文」——把责任推给模型的判断力。

**Agent Skill 的范式：** 「这是 Nature 的 25 条规则，这是 12 步工作流，每一步的输出格式如下，现在开始执行第 1 步」——把规则显式编码，模型只需要执行，不需要判断。

这解释了为什么 nature-skills 的润色质量显著优于通用 Prompt：**不是模型变聪明了，而是模型不再需要猜测什么是「Nature 风格」了**<cite>[2]</cite>。

---

## Agent Skill 的五种设计模式

Google Cloud Tech（2026.03）发布的《5 Agent Skill Design Patterns》归纳了 SKILL.md 的五种设计模式<cite>[3]</cite>。nature-skills 几乎覆盖了全部五种，且多种模式可以组合使用。

### 模式一：Tool Wrapper（工具包装器）

```
问题：Agent 对特定库/框架不熟悉，每次都要在 Prompt 里重复解释 API 用法
方案：将 API 规范打包成 Skill，从 references/ 动态加载
```

nature-skills 的 `nature-figure` 是典型的 Tool Wrapper——它在 `references/api.md` 中定义了 matplotlib 的强制 rcParams（Arial 字体、SVG fonttype='none'等），在 `references/design-theory.md` 中定义了图表信息层级（overview → deviation → relationship）。Agent 不需要「知道」matplotlib 的所有用法，只需要知道该 Skill 定义的约束子集<cite>[2]</cite>。

### 模式二：Generator（生成器）

```
问题：每次输出结构不一致，A 次和 B 次的格式毫无关联
方案：用 assets/ 放模板，references/ 放样式指南，强制填空流程
```

`nature-polishing` 是最典型的 Generator——它的 12 步工作流本质上是一个精密的状态机：时态审查 → 句长检查 → 动词强度校准 → overclaim 检测 → hedging 校准 → British English 转换……每步有明确的输入、处理规则和输出格式。输出就像填表一样精准，而不是靠「感觉」组织段落<cite>[2]</cite>。

### 模式三：Reviewer（审查器）

```
问题：审查标准混乱，每次都靠临时发挥
方案：将「检查什么」与「如何检查」分离，评分规则存在 review-checklist.md
```

`nature-citation` 的引用校验流程是一个内置的 Reviewer：它不生成引用，而是检查现有引用的完整性——是否来自 CNS 期刊、DOI 是否有效、支持力度是否匹配（强支持 vs 背景引用）<cite>[2]</cite>。

### 模式四：Inversion（反转提问）

```
问题：Agent 天生「猜测并立即生成」，经常答非所问
方案：Agent 充当采访者，按阶段提问，门控指令约束执行节奏
```

`nature-writing` 的部分流程采用了 Inversion 模式——它在起草前必须确认论文类型、目标期刊、数据可用性、作者列表等硬事实，然后才进入生成阶段。门控指令禁止 Agent 在信息不足时编造内容<cite>[2]</cite>。

### 模式五：Pipeline（流水线）

```
问题：复杂任务跳步、忽略指令、执行顺序混乱
方案：指令即工作流定义，通过硬关卡强制顺序执行
```

nature-skills 的终极形态本质上是一个跨 Skill 的 Pipeline<cite>[3]</cite>：

```mermaid
graph LR
    A[nature-reader<br/>文献阅读与溯源] --> B[nature-citation<br/>引用检索与校验]
    B --> C[nature-writing<br/>初稿撰写]
    C --> D[nature-polishing<br/>语言润色]
    D --> E[nature-figure<br/>图表生成]
    E --> F[nature-response<br/>审稿回复]
    F --> G[nature-paper2ppt<br/>汇报呈现]
    style A fill:#1a237e,stroke:#4299e1,color:#e8edf5
    style D fill:#1a2a1a,stroke:#48bb78,color:#e8edf5
    style E fill:#2a1a2e,stroke:#ed64a6,color:#e8edf5
    style G fill:#1a1f3a,stroke:#7c3aed,color:#e8edf5
```

这个 Pipeline 不是硬连线的——每个 Skill 独立可用，也可以按需组合。但当串联使用时，前置 Skill 的输出（如 `nature-reader` 的溯源 ID 标注）会自然流入后置 Skill（如 `nature-citation` 的引用校验），形成一套完整的学术生产流水线。

---

## 三种 Agent Skill 范式的对比

从 nature-skills 的成功中，我们可以抽象出 Agent Skill 设计的三层价值模型：

| 层次 | 解决的问题 | 典型手段 | 对比传统 Prompt |
|---|---|---|---|
| **规则编码** | AI 不知道什么是「好的」 | 25 条写作规则、12 步工作流 | Prompt 靠暗示，Skill 靠显式约束 |
| **流程编排** | AI 跳步、顺序混乱、忽略前置条件 | 编号步骤 + 门控指令 + 硬关卡 | Prompt 靠祈使句，Skill 靠状态机 |
| **能力组合** | 单个 Skill 不够，但多 Skill 上下文爆炸 | 渐进披露 + 独立目录 + 按需加载 | 长 Prompt 一次性塞入所有规则 |

**nature-skills 的工程启示：**

<cite>[2]</cite>

> 「不是让 AI 更会聊天，而是让 AI 少一点临场发挥。」

这句话抓住了 Agent Skill 设计的本质。Prompt Engineering 假设模型需要在更大的自由度中找到最优解；Agent Skill 设计假设模型需要的不是自由度，而是约束——越精确的约束，越可靠的输出。

---

## 从 nature-skills 看 Agent Skill 的未来

nature-skills 已经展示了 Skill 作为能力分发单元的巨大潜力。但从 Agent 工程的视角，这条路线还有几个关键问题需要回答<cite>[3]</cite>：

**Skill 的版本管理与依赖。** 当前每个 Skill 独立演进，但 `nature-polishing` 的输出版本 v5.0.2 和 `nature-figure` 的图表规范 v1.0 之间存在隐式的格式依赖。当 Pipeline 变长，这种隐含依赖会成为故障点。类似 npm 的显式依赖声明 (`skill-dependencies: [nature-reader@^1.0]`) 可能是解决方案。

**Skill 的测试与质量保障。** nature-skills 的测试目前主要依赖人工 review 输出质量。但作为一个「规则引擎」，它天然适合基于断言的自动化测试——给定输入文本，断言输出是否满足 25 条规则中的每一条。这不是传统软件的单元测试，而是 **rule compliance verification**。

**Skill 的发现与市场。** nature-skills 通过 GitHub 分发，但 GitHub 不是为 Agent Skill 优化的平台——无法按领域搜索、无法看「好评率」、无法追溯 Skill 的真实效果。一个类似 VS Code Marketplace 的 Skill Registry 正在被需要。这也是为什么 Google 在推动 A2A（Agent-to-Agent）协议和 Skill 标准化<cite>[3]</cite>。

**Skill 组合的自动化。** 目前的 Pipeline 编排是手动的——你需要知道先读文献、再写初稿、再润色。但理想的未来是：Agent 理解用户意图后，自动选择合适的 Skill 组合并编排执行顺序。这需要在 Skill 的 description 层之上增加一层语义理解——类似 Function Calling 但粒度更粗、领域更深。

---

## 总结

nature-skills 是一个「麻雀虽小，五脏俱全」的 Agent Skill 系统工程样本。它用 7 个学术写作技能展示了：

1. **SKILL.md 不是 README，而是 API 接口**——它需要有明确的输入契约、输出格式和错误处理
2. **文件系统是能力模型的天然载体**——目录树的结构映射了领域知识的层级，Tier 1/2/3 的加载链映射了人类「先概览再深入」的认知模式
3. **五种设计模式（Tool Wrapper / Generator / Reviewer / Inversion / Pipeline）覆盖了绝大多数 Skill 场景**——且可以组合使用
4. **Agent Skill 的核心工程价值是把「隐性规则」变成「显性约束」**——不是让模型更聪明，而是让模型不用猜

对于 Agent 工程师来说，nature-skills 的价值不在于它的润色规则有多精准，而在于它证明了一件事：**用工程化的方法管理 Agent 的领域知识，比用更大的 Prompt 更可靠**。

这也正是 Agent Skills 作为「Agent 时代的 package.json」的根本意义——当我们不再依赖单次 Prompt 的即兴发挥，而是构建可复用、可测试、可组合的 Skill 生态时，AI Agent 才真正从「demo」走向「工程」。

---

## 参考文献

1. *nature-skills: 符合 Nature 论文学术表达和科研绘图的 Skill.* Yuan Yizhe, SJTU. GitHub, 2026.  
   <https://github.com/Yuan1z0825/nature-skills>
2. *Skills 规范、构建与设计模式 —— 从 SKILL.md 到生产落地的完整拆解.* Google Cloud Tech, 2026.  
   <https://cloud.tencent.cn/developer/article/2668502>
3. *5 Agent Skill Design Patterns.* Google Cloud Tech Blog, 2026.03.18.  
4. *Claude Code Skills Documentation.* Anthropic, 2026.  
   <https://docs.anthropic.com/en/docs/claude-code/skills>
5. *科研 AI 标准化技能库构建：基于 nature-skills 项目的可复用工作流实践.* CSDN, 2026.  
6. *我终于明白，科研 AI 最缺的不是提示词，而是规矩.* CSDN, 2026.  
   <https://blog.csdn.net/a13730374634/article/details/161213728>
{: .references }
