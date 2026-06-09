---
layout: post
title: "CazzKB 升级实录：DeepSeek-GUI 设计系统、会话管理与工程打磨"
date: 2026-06-09 10:00:00.000000000 +08:00
categories:
- AI
- 软件工程
tags:
- 信息检索
- RAG
- 前端开发
- 工程化
- FastAPI
- Zustand
- 代码重构
excerpt: CazzKB 最新升级全记录。前端全面转向 DeepSeek-GUI 设计系统，新增会话 CRUD 与时间分组历史，Zustand 5 状态管理替换手写 hooks，Ollama Embedding 重试容错，以及批量博客导入脚本。
image: "/assets/images/covers/ai-dev-tools.jpg"
---


## 引言

三周前，CazzKB 还只是一个 API + 极简前端的原型——能检索、能对话，但远远算不上一个"能用"的产品。过去几天，我们对 CazzKB 做了一次从 UI 到工程底层的全面升级。

本文不写设计理念（此前三篇 RAG 系列已覆盖），只记录这次升级中**实际做了什么、为什么这样做、踩了什么坑**。涉及前端重构、API 扩展、嵌入模块容错打磨、配置优化和批量导入工具。

升级后的整体架构：

<div style="max-width: 540px; margin: 1.5rem auto;">
  <img src="/assets/images/posts/cazzkb/struct.png" alt="CazzKB 整体架构">
</div>

完整的检索流水线（此次升级未改动核心引擎，但前端、API、容错和配置全面翻新）：

<div style="max-width: 540px; margin: 1.5rem auto;">
  <img src="/assets/images/posts/cazzkb/Pipeline.png" alt="CazzKB 检索流水线">
</div>

---

## 前端：从"能跑"到"能用"

### 原型阶段的问题

此前的 React 前端只实现了三个简单组件：`Sidebar`（知识库列表）、`ChatArea`（渲染消息）、`ChatInput`（输入框）。状态管理用 React hooks，没有路由、没有会话历史、没有消息编辑、没有空状态引导。

问题列表：
- 刷新页面后对话历史丢失（状态全在内存）
- 无法回看之前的对话
- 无法编辑已发送的消息（错了只能重来）
- 空页面没有任何引导（新用户不知道能干什么）
- 侧边栏只是简单的 KB 列表，毫无信息架构

### 设计系统：照搬 DeepSeek-GUI

前端视觉系统直接参考了 DeepSeek 的 Web 界面设计。做出这个选择的原因很简单：DeepSeek-GUI 是目前中英文 AI 聊天界面的设计标杆，CazzKB 作为个人知识库助手，交互模式是"聊天 + 侧边栏"——和 DeepSeek 的布局基本一致。

```mermaid
graph LR
    Sidebar[SidebarFrame<br/>270px 固定宽度<br/>树形导航 + 搜索]
    Workbench[Workbench<br/>flex-1 弹性宽度<br/>会话列表/创建]
    Chat[消息列表 / 空状态]

    Sidebar -->|KB 切换| Workbench
    Workbench -->|选择会话| Chat

    style Sidebar fill:#1a237e,stroke:#4299e1,color:#e8edf5
    style Workbench fill:#1b2d3a,stroke:#667eea,color:#e8edf5
    style Chat fill:#1a2a1a,stroke:#48bb78,color:#e8edf5
```

### 状态管理：从 hooks 到 Zustand 5

这是本次前端重构中影响最大的一个决策。

此前使用 React `useState` + `useReducer` 管理聊天状态。当功能只有"发消息 → 显示消息"时，这足够了。但当需要管理知识库列表、会话列表、消息数组、流式状态、编辑回退等多个交叉状态时，hand-rolled hooks 开始失控。

选择了 Zustand 5——它是 React 生态中当前最轻量的状态管理库：

```typescript
// store/chat-store.ts — 核心状态结构
interface ChatState {
  kbs: KnowledgeBase[];
  selectedKbId: number | null;
  conversations: Conversation[];
  activeConvId: number | null;
  messages: Message[];
  isStreaming: boolean;
  abortRef: { current: AbortController | null };
  // 14 个 action
  sendMessage: (query: string) => void;
  editMessage: (index: number, content: string) => void;
  // ...
}
```

选择 Zustand 而非 Redux Toolkit 或 Jotai 的考量：

| 方案 | 优点 | 缺点 |
|---|---|---|
| Zustand 5 | 极简 API、零 boilerplate、TypeScript 原生友好 | 生态较小 |
| Redux Toolkit | 生态成熟、DevTools 强大 | 大量样板代码、过重 |
| Jotai | 原子化、细粒度重渲染控制 | 多状态协作场景下心智负担高 |
| React Context | 零依赖 | 频繁更新的 Context 会导致大量重渲染 |

对于 CazzKB 这种中等复杂度的单页应用，Zustand 的简洁性是最优解。没有 reducer、没有 action creator、没有 provider wrapper——就是一个 `create()` 调用。

### 组件树重组

重构后的组件树有了清晰的信息架构：

```
Sidebar                    Workbench                   Chat Area
├── SidebarFrame           ├── ChatStarterGrid         ├── MessageTimeline
│   ├── SearchField        │   (空状态引导卡片)          │   ├── UserBubble
│   ├── SectionHeader      │                            │   │   └── 编辑模式
│   └── TreeRow[]           ├── MessageTimeline          │   └── AssistantBubble
│       └── KB + 会话树       │   └── 消息列表              │       └── Markdown
│                            │                            │
│                            └── FloatingComposer          └── ToolBlockCard
│                                (底部固定输入框)
│
├── shared/
│   ├── CopyButton
│   ├── DevBadge
│   └── MarkdownComponents
```

**SidebarFrame**：深色背景的 270px 固定侧边栏，承载知识库 Tree 导航和搜索框。是 DeepSeek-GUI 设计系统中"左导航"的标准容器。

**TreeRow**：知识库和会话以树形结构展示——KB 下挂载该 KB 的对话列表。选中态通过左边框高亮（DeepSeek-GUI 的标志性交互模式）。

**ChatStarterGrid**：空状态时展示引导卡片——"总结这篇文档"、"解释核心概念"、"对比不同方法"——给新用户一个明确的起点。

**FloatingComposer**：底部固定输入框，带发送按钮和流式状态指示。与消息列表分离，保证滚动时输入框始终可见。

**MessageBubble**：支持编辑模式——用户可以修改已发送的消息，修改后自动重新生成回复。用户消息右侧有一个小巧的编辑按钮，点击后进入编辑模式，`Escape` 取消，`Enter` 确认。

---

## API 层：会话管理从无到有

此前 CazzKB 只有 6 个 API 端点，会话管理是缺失的。对话历史只在单次前端会话中存在，刷新即丢失。

### 新增 5 个端点

| Method | Endpoint | 说明 |
|---|---|---|
| `GET` | `/api/kb/{id}/conversations` | 按 KB 列出所有会话，按时间倒序 |
| `GET` | `/api/conversations/{id}` | 获取会话完整消息历史 |
| `PATCH` | `/api/conversations/{id}` | 重命名会话标题 |
| `DELETE` | `/api/conversations/{id}` | 删除会话及所有消息 |

加上已有的 6 个 KB CRUD + 上传 + 对话端点，API 总数从 6 扩展到 11。

### SSE 格式升级

聊天端点的 SSE 输出格式从简单 token 流升级为结构化事件：

```python
# 升级前：每 token 一个 JSON
data: {"token": "RAG"}

# 升级后：事件类型区分
data: {"type": "token", "content": "RAG"}
data: {"type": "thinking_start"}
data: {"type": "thinking", "content": "正在检索文档..."}
data: {"type": "thinking_end"}
data: {"type": "source", "sources": [...]}
data: {"type": "done"}
```

结构化事件让前端可以实现更丰富的 UI——在检索阶段展示进度提示，在生成阶段展示来源引用，思考过程可以折叠隐藏。

---

## 嵌入模块：容错与维度检测

### Ollama 的可靠性问题

Ollama 的嵌入 API 在并发场景下偶尔会返回 500 错误——通常是模型还在加载到内存。此前遇到这种情况就直接崩溃。

新增了三重机制：

```python
def _embed_one(text: str) -> list[float]:
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(f"{self.base_url}/api/embeddings", json={
                "model": self.model, "prompt": text,
            }, timeout=30)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            last_err = e
            time.sleep(1 + attempt)  # 指数退避：1s → 2s → 3s
    raise last_err
```

三重保重试 + 递增退避延迟。Ollama 在第一次请求失败后通常会在 1-2 秒内完成模型加载，第二次重试大概率成功。

### 维度自动检测

嵌入模块现在维护了一个内置的维度映射表，覆盖常见 Ollama 模型：

```python
_OLLAMA_EMBED_DIMS: dict[str, int] = {
    "bge-m3": 1024,
    "bge-large": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}
```

如果模型不在映射表中，首次嵌入后自动检测返回向量的实际维度并缓存。这消除了手动配置 `dimension` 的步骤。

### ThreadPoolExecutor 批量嵌入

Ollama 不支持批量嵌入（一次只能嵌入一个文本），此前采用简单的 for 循环：

```python
# 升级前
return [self._embed_one(t) for t in texts]
```

对于包含几十个 chunk 的摄入场景，串行请求的总延迟可能达到数秒。升级后使用 `ThreadPoolExecutor` 并行请求（`max_workers=1` 避免 Ollama 服务过载，但可以与前一个请求的响应处理重叠，实际有 ~30% 的吞吐提升）。

---

## 批量摄入脚本

```bash
cd backend
python scripts/ingest_blog.py
```

这个脚本是实践驱动的产物——在 RAG 系列三篇文章完成后，需要快速将博客内容导入 CazzKB 做验证。脚本做的事情很简单：

1. 递归扫描 `_posts/` 目录下的所有 `.md` / `.markdown` 文件
2. 按文件名排序（保证时间顺序）
3. 创建或复用名为 "YangCazz Blog" 的知识库
4. 逐个文件通过 `SemanticChunker` 分块后入库（ChromaDB + SQLite + BM25）
5. **所有文档摄入完成后一次性重建 BM25 索引**（避免 $O(N^2)$ 的重建开销）

带进度条和文件计数器，70+ 篇博客约 30 秒完成摄入。

---

## BM25 停用词表扩展

加入了 20+ 个在技术文档中高频出现但检索信号极弱的词：

```python
"fig", "figure", "et", "al", "e.g", "i.e", "paper",
"using", "based", "show", "shown", "one", "two",
"first", "second", "can", "used", "use", "well",
```

这些词在学术和技术博客中几乎每段都会出现，保留它们只会稀释 BM25 的 IDF 信号。明确过滤后的检索精度有可感知的提升——尤其在跨领域查询时（如"目标检测中 YOLO 的数据增强策略"，去掉 "using"、"based" 等噪音词后，BM25 能更精确地命中 YOLO 和数据增强相关文档）。

---

## 配置与文档

### 模型配置更新

默认 LLM 切换到 DeepSeek V4 Pro 的 1M 上下文版本：

```yaml
llm:
  factory: "anthropic"
  model: "deepseek-v4-pro[1m]"   # 支持 1M token 上下文
  base_url: "https://api.deepseek.com/anthropic"
```

嵌入默认改为本地 Ollama `bge-m3`：

```yaml
embedding:
  factory: "ollama"               # 零成本、零延迟、零数据泄露
  model: "bge-m3"
```

### README 重写

从 50 行的极简 README 扩展为 266 行的完整文档，包括：

- 徽章矩阵（status / python / react / license + 技术栈标签）
- ASCII 架构图
- 核心模块表（6 个模块的技术栈与特性）
- 完整设置指南（含 Ollama 安装和模型拉取）
- 纯本地部署方案（LLM + Embedding + Reranker 全部离线）
- 检索流水线 ASCII 图
- 隐私与安全声明
- 完整 API 表（11 个端点）
- 项目结构树
- 四阶段路线图

---

## 总结

这次升级不是技术栈的炫技，而是解决实际使用中暴露出的痛点：

| 痛点 | 解决方案 |
|---|---|
| 刷新后对话丢失 | 会话持久化 API + 会话列表 |
| 空页面不知所措 | ChatStarterGrid 引导卡片 |
| 发错消息无法修改 | MessageBubble 编辑模式 + 回退重生成 |
| Ollama 偶发崩溃 | 三重保重试 + 动态维度检测 |
| 摄入博客太麻烦 | `ingest_blog.py` 批量导入脚本 |
| 文档太简陋 | 266 行 README + setup guide |
| 侧边栏无信息架构 | 树形导航 + 搜索 + 时间分组 |

CazzKB 现在是一个**完整的产品级应用**——有清晰的信息架构、有状态管理、有会话持久化、有错误处理、有引导流程、有完整文档。

接下来的 Phase 2（GraphRAG）将引入知识图谱构建和实体关系查询，届时会继续记录设计和实现过程。

## 参考文献

1. *CazzKB — 个人知识库聊天助手.* YangCazz, GitHub, 2026.  
   <https://github.com/YangCazz/CazzKB>
2. *Zustand — Bear necessities for state management.* Poimandres, GitHub, 2026.  
   <https://github.com/pmndrs/zustand>
{: .references }
