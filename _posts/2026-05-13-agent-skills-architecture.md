---
layout: post
title: "AI Agent Skills 深度解析：设计哲学、架构模式与形成逻辑"
date: 2026-05-13 10:00:00 +0800
categories: [AI, 软件工程]
tags: [Agent Skills, Claude Code, AI架构, 渐进式披露, AI Agent]
excerpt: "从 Anthropic 官方设计哲学、agentskills.io 开放标准到学术论文的系统分析，深入探讨 AI Agent Skills 的核心设计模式：渐进式披露、结构化认知约束、Plan-Execute-Verify 循环，以及 Skills 在 AI 智能体生态中的架构定位。"
image: /assets/images/covers/ai-dev-tools.jpg
---

## 引言

2025 年 12 月，Anthropic 将 Claude Code 中的 Skills 系统提炼为 **agentskills.io** 开放标准并正式发布 <cite>[1]</cite>。此后不到半年，该标准被 Claude Code、OpenAI Codex、GitHub Copilot、VS Code、Cursor 等 26+ 个平台采纳 <cite>[2]</cite>。与此同时，学术界也开始系统性审视 Skills 的安全性和架构设计——2026 年初的两篇安全研究论文发现，公开 Skills 仓库中超过 26% 存在安全漏洞，157 个 Skills 被确认为恶意 <cite>[3][4]</cite>。

Skills 到底是什么？它为什么能成为 AI Agent 生态中最快速的标准化运动之一？它的设计背后遵循怎样的工程哲学？

本文基于 Anthropic 官方技术博客、agentskills.io 规范文档、学术论文和社区实践，系统性地分析 Skills 的核心设计模式、形成逻辑与架构定位。

> **信息来源筛选规则**：本文优先采用一手来源（Anthropic 官方博客、agentskills.io 规范、arXiv 学术论文），其次参考经过同行验证的行业分析。AI 生成内容聚合网站（如 skywork.ai 等）的信息已在交叉验证后排除或标注。

---

## 一、什么是 Agent Skill

### 1.1 定义

Agent Skill 是一个 **文件系统级别的能力包**——它是一个包含 `SKILL.md` 文件（含 YAML 前置元数据和 Markdown 指令正文）的目录，可选附带脚本、参考文档和资源文件 <cite>[1]</cite>。

```
my-skill/
├── SKILL.md          # 必需：YAML 元数据 + Markdown 指令
├── scripts/          # 可选：可执行脚本
├── references/       # 可选：补充文档
└── assets/           # 可选：模板、图片
```

**SKILL.md 最小示例** <cite>[1]</cite>：

```yaml
---
name: pdf-processing
description: Extract PDF text, fill forms, merge files. Use when handling PDFs.
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---

# PDF Processing

## Step 1: Analyze the PDF
...
```

这和传统的 "prompt 模板" 有本质区别。Skills 不是一段文本，而是一个**结构化的认知约束系统**——它定义了 AI Agent 在面对特定任务时应该遵循的步骤、约束条件和决策标准。

### 1.2 Skills 解决了什么根本问题

在 Skills 出现之前，让 AI Agent 获得特定领域能力的途径主要有三种：

| 途径 | 问题 |
|------|------|
| 把所有指令写入 System Prompt | 上下文窗口爆炸，不同指令相互干扰 |
| 使用 RAG 检索相关文档 | 检索精度不稳定，Agent 不主动"知道该找什么" |
| 在对话中手动提供指令 | 每次重复，无法复用 |

Skills 解决的核心问题是：**如何让 Agent 按需获取能力，而不污染持续上下文**。

---

## 二、核心设计哲学：渐进式披露

### 2.1 三层加载架构

Skills 最核心的设计模式是 **渐进式披露（Progressive Disclosure）**，它定义了三个上下文加载层级 <cite>[5][6]</cite>：

| 层级 | 加载内容 | Token 预算 | 触发时机 |
|------|---------|-----------|---------|
| **发现层（Discovery）** | `name` + `description` | ~50-100 tokens/skill | 会话启动时全部加载 |
| **激活层（Activation）** | 完整 `SKILL.md` 正文 | < 5,000 tokens 推荐 | Agent 判断任务匹配描述时 |
| **执行层（Execution）** | `scripts/`、`references/` | 按需加载 | 指令中明确引用时 |

这意味着你可以安装 **100 个 Skills** 而不会影响会话启动性能——只有 name 和 description（各约一行）会常驻上下文 <cite>[5]</cite>。仅当 Agent 判断某个 Skill 与当前任务相关时，才会加载其完整指令。

### 2.2 设计来源：Anthropic 的工程实践

渐进式披露并非凭空产生，而是 Anthropic 团队在构建 Claude Code 过程中反复迭代的结果。在 2026 年 4 月的技术博客 *"Seeing like an agent: how we design tools in Claude Code"* 中，Claude Code 工程师 Thariq Shihipar 详细阐述了这一设计哲学的来源 <cite>[7]</cite>：

> "我们目前只有约 20 个工具，添加新工具的门槛极高——每增加一个工具，模型就多一个需要思考的决策点。"

渐进式披露解决了这个矛盾：它让 Agent 在**不增加核心工具数量**的前提下获得了**可扩展的专业知识能力**。Shihipar 以 "Claude Code Guide" 子智能体为例说明了这一点：当用户询问 Claude Code 自身的使用方法时，系统不会将全部文档塞入上下文，而是调用一个专门的子智能体去检索文档并返回精炼答案 <cite>[7]</cite>。

### 2.3 与 RAG 的关键区别

渐进式披露与传统的 RAG（检索增强生成）有本质不同：

| 维度 | RAG | 渐进式披露 |
|------|-----|-----------|
| 检索主体 | 外部检索器（向量搜索） | Agent 自身（意图匹配） |
| 触发方式 | 基于语义相似度 | 基于任务理解和自主决策 |
| 上下文控制 | 检索器决定返回什么 | Skill 作者预定义披露层级 |
| 信息结构 | 扁平化的文本片段 | 结构化的指令 + 资源引用 |

本质上，渐进式披露让 Agent 从 "被动投喂者" 变为 "主动知识管理者"。

---

## 三、Skills 的结构设计

### 3.1 SKILL.md 的前置元数据规范

agentskills.io 规范对 `SKILL.md` 的 YAML 前置元数据定义了严格的约束 <cite>[1]</cite>：

**必需字段**：

| 字段 | 约束 |
|------|------|
| `name` | 最长 64 字符，仅小写字母/数字/连字符，不得以连字符开头或结尾，不得连续使用连字符，**必须与父目录名一致** |
| `description` | 最长 1024 字符，非空。需描述**做什么 + 何时使用**，包含触发关键词 |

**可选字段**：

| 字段 | 约束 |
|------|------|
| `license` | 许可证名称或引用 |
| `compatibility` | 最长 500 字符，环境要求 |
| `metadata` | 任意键值对（如 `author`、`version`） |
| `allowed-tools` | 预批准的工具列表（实验性） |

### 3.2 Description 的设计标准

Description 是整个 Skill 中**最关键但最容易被低估**的字段。它直接决定了 Agent 何时激活该 Skill。规范和实践总结的最佳实践包括 <cite>[5][8]</cite>：

- **结构公式**：`[做什么] + [何时使用] + [关键能力]`
- **包含真实的触发短语**：用户可能输入的自然语言表达
- **不要总结工作流**：如果 description 中包含了步骤摘要，Agent 可能跳过正文直接 "走捷径"
- **模糊描述是反模式**：`"Helps with projects"` 永远不会被触发

**好的示例**：

```yaml
description: Summarizes uncommitted changes and flags risky patterns. Use when user asks what changed, wants a commit message, or asks to review their diff before committing.
```

### 3.3 正文编写原则

Trail of Bits（知名安全研究机构）在其 Skills 仓库中提出了一套社区认可的质量标准 <cite>[8]</cite>：

- **正文控制在 500 行以内**，详细参考资料移至 `references/`
- **用项目符号和编号列表**替代散文段落
- **关键指令置顶**，使用 `## Important` 或 `## Critical` 标记
- **解释 WHY 而非仅 WHAT**：包含权衡、决策标准、判断边界
- **一个优秀示例胜于五个平庸示例**
- **只引用一层深度的文件**：SKILL.md → references/something.md，不要在 references 中再链向更深的文件（链式引用会降低 Agent 性能）

**反模式** <cite>[5]</cite>：

| 错误做法 | 问题 |
|---------|------|
| 模糊描述（"Helps with projects"） | 永远不会被触发 |
| Description 中总结工作流 | Agent 跳过正文，失去关键约束 |
| XML 尖括号出现在前置元数据中 | 安全限制，可能导致解析错误 |
| 重复 CLAUDE.md 中已有的内容 | 浪费上下文，且可能产生冲突 |
| 多层文件引用链 | Agent 迷失在引用层级中 |

---

## 四、Skills 的形成模式：从认知约束到执行闭环

### 4.1 Plan-Execute-Verify 循环

Anthropic 在 Claude Code Power User Tips 中明确指出 <cite>[9]</cite>：

> "本指南中唯一最有影响力的建议是**验证（verification）**——给 Claude 一种检查自己输出的方法。如果你只采纳一条实践，就采纳这一条。"

这一原则催生了 Skills 设计中的 **Plan → Execute → Verify** 标准循环：

```
Assess（评估需求）
  → Plan（制定计划）
    → Execute（执行步骤）
      → Verify（验证结果）
```

这不是一个硬编码的工作流，而是每个 Skill 应该在指令中**结构化引导 Agent 遵循**的认知模式。多个高质量的社区 Skill 库（如 LUNARTECH Superpowers）将这一模式固化为可分拆的独立 Skills：`brainstorming` → `writing-plans` → `executing-plans` → `verification-before-completion` <cite>[10]</cite>。

### 4.2 认知约束设计

从工程心理学角度看，Skills 的本质是**为 LLM 构建结构化的认知约束（Structured Cognitive Constraints）**。Anthropic 在 *"Building effective agents"* 一文中提出的三项核心原则对此做了精确表述 <cite>[11]</cite>：

**原则一：保持简洁性（Simplicity）**

> "在 LLM 领域取得成功不在于构建最复杂的系统，而在于构建适合你需求的系统。从简单的提示开始，通过全面的评估进行优化，仅在简单方案不足时才添加多步骤 Agent 系统。" <cite>[11]</cite>

这解释了为什么 Skills 采用文件系统级别的简单抽象，而非复杂的插件框架。一个 Skill 本质上只是一个带 YAML 头部的 Markdown 文件——理解成本趋近于零。

**原则二：保证透明性（Transparency）**

> "明确展示 Agent 的规划步骤，让用户能看到推理过程和任务分解结果。" <cite>[11]</cite>

Skills 中的 `## Step 1`, `## Step 2` 等结构化步骤正是这一原则的体现——它们让 Agent 的行为**可预测、可审计、可调试**。

**原则三：精心设计 Agent-Computer Interface (ACI)**

> "通过全面的工具文档和测试来精心打造 Agent 与计算机之间的接口，像重视人机交互 (HCI) 一样重视 ACI 设计。" <cite>[11]</cite>

`allowed-tools` 字段、`context: fork` 隔离模式、工具调用的预批准机制——这些都是 ACI 设计在 Skills 层面的具体体现。

### 4.3 从 Prompt 模板到 Skills：一个演化视角

2025 年的 Prompt 工程已经演化为一门系统学科 <cite>[12]</cite>。Skills 位于这条演化路径的最新节点：

```
静态 Prompt（2022）
  → 动态 Prompt 链（2023）
    → 结构化 Prompt 蓝图 / PDL（2024）
      → Skills 系统（2025）
        → 自优化 Agent 系统（2026+）
```

学术研究证实了这一趋势。2026 年 4 月，Liu 等人的论文 *"Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems"* 对 Claude Code v2.1.88（约 1900 个 TypeScript 文件、512K 行代码）进行了源代码级别的架构分析 <cite>[13]</cite>。他们发现：

> "Agent 循环本身只是一个简单的 while 循环（调用模型 → 运行工具 → 重复）。**98.4% 的代码是基础设施**：权限门控、上下文管理、工具路由和恢复逻辑。"

Skills 正是这 98.4% 基础设施的关键组成部分。它不属于核心 Agent 循环，而是作为**认知支持层**提供领域知识和行为约束。

---

## 五、Skills 在 AI Agent 架构中的定位

### 5.1 Claude Code 的四层扩展机制

Liu 等人的论文识别出 Claude Code 拥有四种扩展机制 <cite>[13]</cite>：

| 机制 | 用途 | 激活方式 |
|------|------|---------|
| **MCP** | 外部工具连接（数据库、API、浏览器） | 会话级连接 |
| **Plugins** | Skills + Hooks + Agents 的打包分发 | 安装到项目/用户目录 |
| **Skills** | 按需加载的专业知识和流程 | 意图匹配后自动激活 |
| **Hooks** | 确定性自动化（27 个生命周期事件） | 事件触发 |

它们的分工关系可以这样理解：

> **CLAUDE.md** 是"永远在线的规则"（代码风格、禁止事项）  
> **Skills** 是"按需加载的知识"（领域流程、操作手册）  
> **MCP**  是"按需连接的工具"（数据库、外部 API）  
> **Hooks** 是"自动执行的触发器"（lint 检查、通知）

### 5.2 Skills vs Plugins

这两个概念容易混淆，但职责边界清晰 <cite>[5][14]</cite>：

一个 **Plugin** 是一个打包分发单元，它可以包含多个 Skills、自定义子智能体、Hooks 和 MCP 配置。Plugin 安装后，其中的 Skills 自动注册到 Agent 的发现层。

Skills 是 Plugin 的**内容**，Plugin 是 Skills 的**分发载体**。

### 5.3 Skills 与 MCP 的协作模式

Skills 和 MCP 最常见的协作模式是 <cite>[14]</cite>：

- **MCP** 连接你的数据库
- **Skill** 记录数据库 schema 和查询模式，告诉 Agent **如何**正确查询

这种分工避免了将 schema 信息硬塞入 System Prompt，同时确保 Agent 在需要时能自动获取正确语境。

---

## 六、Skills 安全：攻击面与防护

### 6.1 已知威胁

2026 年初的学术研究系统性地揭示了 Agent Skills 的安全风险 <cite>[3][4]</cite>：

- **26%+ 的公开 Skills 包含至少一个安全漏洞**
- **157 个 Skills 被确认为恶意**（2026 年 2 月，通过行为测试验证）
- **ClawHavoc 攻击行动**（2026 年 1 月）：单一攻击者在 3 天内向注册表灌入 **341 个恶意 Skills**
- **单个攻击者贡献了 54% 的已确认恶意 Skills**

主要攻击向量包括 <cite>[3]</cite>：

| 攻击类型 | 手段 |
|---------|------|
| **提示注入** | 通过 HTML 注释、不可见 Unicode 字符在 SKILL.md 中嵌入劫持指令 |
| **凭证窃取** | Skill 指令中隐藏环境变量读取和外部发送逻辑 |
| **工具滥用** | 利用 `allowed-tools` 的宽松配置执行危险命令 |

### 6.2 当前防护缺口

Security researchers identified several current gaps <cite>[4]</cite>:

- **无版本锁定/锁文件机制**：无法固定 Skill 版本，自动更新可能引入恶意变更
- **无标准审查工具**：NVIDIA 的 NemoClaw 是早期尝试，但尚未普及
- **注册表无强制安全审查**：大多数 Skill 注册表没有提交前安全审计
- **组织内 Skills 不可见**：无法获知团队中安装了哪些 Skills

### 6.3 Trail of Bits 的安全实践

Trail of Bits 在其 Skills 仓库中提出了一套防御性编写原则 <cite>[8]</cite>：

- **`## When NOT to Use`** 是必需章节——明确定义边界条件
- **`## Rationalizations to Reject`** 是安全 Skills 的强制条款——列出 Agent 常见的 "合理化借口" 并教育其拒绝
- **验证指令必须具体**：不说 "确保一切正常"，而说 "运行 `npm test` 并确认 0 failures"

---

## 七、Skills 的生态现状

### 7.1 标准化进程

agentskills.io 开放标准（2025 年 12 月 18 日发布）是 Skills 生态的基石 <cite>[1]</cite>。该标准可能随 MCP 协议一道，被移交给即将成立的 **Agentic AI Foundation (AAIF)**（隶属于 Linux 基金会），与 MCP 并列成为 AI Agent 基础设施的双支柱 <cite>[2]</cite>。

### 7.2 跨平台采纳

截至 2026 年中，Skills 标准已被 26+ 个平台采纳 <cite>[2]</cite>：

- **AI 编程助手**：Claude Code、OpenAI Codex CLI、GitHub Copilot、Cursor
- **IDE**：VS Code、JetBrains（通过插件）
- **Agent 框架**：Goose（Block）、Amp、Letta、OpenCode
- **安全工具**：Trail of Bits 的 Skills 仓库

### 7.3 社区生态

Skill 注册表和市场的出现标志着从 "个人工具" 到 "生态基础设施" 的转变 <cite>[4]</cite>：

- **ClawHub**：社区 Skill 注册表
- **skillsmp.com**：Skill 市场
- **skills.sh**：命令行 Skill 发现工具

这些平台让 Skills 具备了类似于 npm/PyPI 的网络效应，但也带来了前述的安全挑战。

---

## 八、总结与展望

### 8.1 核心洞见

Skills 的设计揭示了 AI Agent 系统工程的几个深层洞见：

**1. 认知约束是比 Prompt 质量更根本的问题**

Skills 不是 "写得更好的 Prompt"，而是为 LLM 构建结构化的认知框架——告诉它**什么时候该做什么、怎么做、为什么这样做、何时不该做**。

**2. 渐进式披露是 AI 原生设计**

传统软件工程讲究 "关注点分离"，Skills 的渐进式披露将这一原则延伸到了上下文层面：**不在上下文中放不必要的信息**。这比 RAG 的 "检索式" 方法更精确、更可预测。

**3. Plan-Execute-Verify 是 Agent 的工程闭环**

从 Anthropic 的官方建议到开源社区的最佳实践，Plan → Execute → Verify 循环已被证明是 Agent Skills 设计的核心模式。它的本质是：**给 Agent 可控的自主权，但强制验证输出**。

**4. 安全是 Skill 设计的一等公民**

26% 的公开 Skills 存在漏洞这一发现，提醒我们 Skills 安全的紧迫性。`## When NOT to Use` 和 `## Rationalizations to Reject` 等防御性章节应该成为每个 Skill 的标配。

### 8.2 未来方向

Liu 等人的论文指出了六个开放方向 <cite>[13]</cite>，其中两个尤为关键：

- **基于使用反馈的 Skills 自适应机制**：Skills 不应是静态的。随着 Agent 在实际使用中积累经验，Skills 的指令应该自动优化——类似于 A/B 测试驱动的 Prompt 优化
- **从单步安全分类到边界级访问控制**：当前 Skills 的 `allowed-tools` 是粗略的，未来需要更精细的权限模型

### 8.3 实践建议

如果你正在构建或使用 Agent Skills：

1. **从最简单的形式开始**：一个 `SKILL.md` 文件，50 行正文，一个清晰的使用场景
2. **description 投入最多精力**：它是决定 Skill 能否被正确触发的唯一入口
3. **验证优于一切**：每个 Skill 必须包含一个显式的验证步骤
4. **定义边界**：`## When NOT to Use` 不是可选的
5. **保持一层引用深度**：Skill → references，不要再 deeper

---

## 参考文献

<ol class="references">
<li>Anthropic. <em>Agent Skills Specification</em>. agentskills.io, December 18, 2025.<br>
<a href="https://agentskills.io/specification">agentskills.io/specification</a> · <a href="https://github.com/agentskills/agentskills">GitHub</a></li>

<li>Simon Willison. <em>Agent Skills</em>. simonwillison.net, December 19, 2025.<br>
<a href="https://simonwillison.net/2025/Dec/19/agent-skills/">simonwillison.net/2025/Dec/19/agent-skills/</a></li>

<li>Yi Liu et al. <em>Security Vulnerabilities in Agent Skills</em>. arXiv:2601.10338, January 2026.</li>

<li>Yi Liu et al. <em>Behavioral Testing of Malicious Agent Skills</em>. arXiv:2602.06547, February 2026.</li>

<li>Anthropic. <em>Claude Code Features Overview</em>.<br>
<a href="https://code.claude.com/docs/en/features-overview">code.claude.com/docs</a></li>

<li>Anthropic. <em>Plugins Reference — Claude Code Docs</em>.<br>
<a href="https://code.claude.com/docs/en/plugins-reference">code.claude.com/docs</a></li>

<li>Thariq Shihipar. <em>Seeing like an agent: how we design tools in Claude Code</em>.<br>
Anthropic Engineering Blog, April 10, 2026.<br>
<a href="https://claude.com/blog/seeing-like-an-agent">claude.com/blog/seeing-like-an-agent</a></li>

<li>Trail of Bits. <em>Skills Repository — Quality Standards</em>.<br>
<a href="https://github.com/trailofbits/skills">github.com/trailofbits/skills</a></li>

<li>Anthropic. <em>Claude Code Power User Tips</em>. Claude Help Center.<br>
<a href="https://support.claude.com/en/articles/14554000-claude-code-power-user-tips">support.claude.com</a></li>

<li>LUNARTECH. <em>Superpowers: The Complete Claude AI Skills Library</em>. lunartech.ai, 2026.</li>

<li>Erik Schluntz &amp; Barry Zhang. <em>Building Effective Agents</em>.<br>
Anthropic Engineering Blog, December 19, 2024.<br>
<a href="https://www.anthropic.com/engineering/building-effective-agents">anthropic.com/engineering/building-effective-agents</a></li>

<li>Devoteam. <em>Universal Prompt Blueprint</em>. devoteam.com, 2025.</li>

<li>Jiacheng Liu, Xiaohan Zhao, Xinyi Shang, Zhiqiang Shen.<br>
<em>Dive into Claude Code: The Design Space of Today's and Future AI Agent Systems</em>.<br>
arXiv:2604.14228, April 14, 2026.<br>
<a href="https://arxiv.org/abs/2604.14228">arxiv.org/abs/2604.14228</a> · <a href="https://github.com/VILA-Lab/Dive-into-Claude-Code">GitHub</a></li>

<li>Anthropic. <em>Equipping agents for the real world with Agent Skills</em>.<br>
Anthropic Blog, October 16, 2025.</li>
</ol>

---

*本文中所有技术断言均基于上述一手来源，经交叉验证后采纳。AI 内容聚合网站的信息已在交叉验证后排除或标注为间接参考。*
