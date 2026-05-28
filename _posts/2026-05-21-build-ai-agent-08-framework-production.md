---
layout: post
title: "构建自己的 AI Agent · 第八篇：框架选型 & 生产部署 — 从手写代码到工业级服务"
date: 2026-05-21 10:00:00 +0800
categories: [AI, 软件工程, AI Agent 系列]
tags: [AI Agent, LangChain, CrewAI, AutoGen, FastAPI, Docker, 生产部署, 框架选型, 从零构建]
excerpt: "AI Agent 系列终篇。对比 5 大 Agent 框架的本质差异——它们只是对前 6 篇手写代码的封装。将 8 篇累积代码部署为 FastAPI 生产服务，含 Docker、日志监控和医疗机器人场景案例。"
image: /assets/images/covers/ai-dev-tools.jpg
---

## 引言

八篇文章前，我们从 30 行 Python 开始了 Agent 之旅。现在回头看，我们已经手写了：
- ReAct 循环引擎
- 类型驱动的工具系统
- Token 感知的对话管理器
- 四层记忆系统（工作/情节/语义/程序）
- 多 Agent 编排（Generator-Verifier + Orchestrator-Workers）
- MCP 协议集成
- 安全护栏 + 评测体系

有了这些底层理解，我们再来看流行的 Agent 框架，会发现它们**本质上就是对上述组件的标准化封装**。

本篇是系列的终章，做两件事：
1. **框架选型**：理解 5 大框架的本质差异
2. **生产部署**：将我们的 Agent 用 FastAPI + Docker 部署为生产服务

---

## 理解框架：不过是标准化的封装

### 框架 = 组件 + 约定 + 生态

```mermaid
graph LR
    subgraph 我们的代码[我们手写的 6 篇代码]
        R[ReAct 循环]
        T[工具系统]
        C[对话管理]
        M[记忆系统]
        MA[多Agent]
        P[MCP 协议]
    end

    subgraph 框架做的事[框架 = 标准化封装]
        R2[Agent 基类]
        T2[Tool 装饰器]
        C2[Memory 抽象]
        M2[VectorStore 接口]
        MA2[Orchestrator 模式]
        P2[Protocol 适配器]
    end

    R --> R2
    T --> T2
    C --> C2
    M --> M2
    MA --> MA2
    P --> P2

    style 我们的代码 fill:#2b2d42,color:#fff
    style 框架做的事 fill:#667eea,color:#fff
```

### 五大框架对比

| 维度 | LangChain/LangGraph | CrewAI | AutoGen (Microsoft) | OpenAI Agents SDK | Dify / n8n |
|------|---------------------|--------|---------------------|-------------------|------------|
| **定位** | 通用框架 + 图编排 | 多 Agent 角色扮演 | 对话驱动多 Agent | 轻量官方 SDK | 低代码/可视化 |
| **核心理念** | Chain → Graph | Crew + Task | ConversableAgent | Agent + Handoff | 拖拽式编排 |
| **对应本系列** | 第1-4篇的标准化 | 第5篇的AgentTeam | 第5篇的对话式 | 第1-2篇的精简版 | 无代码封装 |
| **学习曲线** | 陡峭 (v0.1→0.3 破坏性变更) | 平缓 | 中等 | 平缓 | 极低 |
| **适用场景** | 复杂工作流、RAG | 角色扮演协作 | 研发/调试/对话 | OpenAI 生态 | 快速原型 |
| **灵活性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **生产成熟度** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 框架的本质解构

### LangChain/LangGraph：标准化了我们的第 1-4 篇

```python
# 我们的代码 (第 1-4 篇)
class ConversationalAgent:
    def __init__(self):
        self.registry = ToolRegistry()       # → LangChain Tool
        self.conv = ConversationManager()     # → LangChain Memory
        self.memory = AgentMemory("./mem")    # → LangChain VectorStore
        self.max_iter = 10

    def run(self, query: str) -> str:
        for _ in range(self.max_iter):        # → LangGraph: AgentState loop
            response = self.client.chat.completions.create(...)
            if msg.tool_calls:                 # → LangGraph: conditional edge
                result = self.registry.execute(...)
            else:
                return msg.content


# LangChain/LangGraph 的等价写法
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)       # → 我们的 while 循环体
graph.add_node("tools", ToolNode(tools))  # → 我们的 registry.execute()
graph.add_edge("agent", "tools")          # → 我们 while 循环中的分支
graph.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
```

**核心区别**：LangGraph 用**显式状态图**替代了我们的 `while + if` 隐式控制流。对于复杂工作流（如递归审查、多分支路由），图结构更清晰；但对于简单任务，while 循环更直观。

### CrewAI：标准化了我们的 AgentTeam（第 5 篇）

```python
# 我们的 AgentTeam (第 5 篇)
team = AgentTeam(members=[
    {"name": "研究员", "role": "收集信息"},
    {"name": "写作者", "role": "撰写报告"}
])
result = team.discuss("分析 AI 趋势", rounds=3)


# CrewAI 等价写法
from crewai import Agent, Task, Crew

researcher = Agent(role="研究员", goal="收集最新 AI 趋势信息")
writer = Agent(role="写作者", goal="撰写专业报告")

task1 = Task(description="搜索 AI 趋势", agent=researcher)
task2 = Task(description="撰写报告", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

CrewAI 的核心价值在于**角色定义的标准化**（`role` + `goal` + `backstory`），比我们手写的字典方式更结构化。

### AutoGen：对话驱动的多 Agent

```python
# AutoGen 核心设计：ConversableAgent
from autogen import ConversableAgent, initiate_chats

assistant = ConversableAgent(
    name="assistant",
    system_message="你是助手",
    llm_config={"model": "gpt-4o"}
)

critic = ConversableAgent(
    name="critic",
    system_message="你是评论者，找出回答中的问题",
    llm_config={"model": "gpt-4o"}
)

# 两个 Agent 之间自动对话
initiate_chats([
    {"sender": assistant, "recipient": critic,
     "message": "写一个排序算法", "max_turns": 3}
])
```

AutoGen 的创新在于**对话本身作为编排机制**——Generator-Verifier 循环变成了两个 Agent 间的对话，更贴近人类协作方式。

### OpenAI Agents SDK：最精简的封装

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    return f"{city}: 晴, 22°C"

agent = Agent(
    name="助手",
    instructions="你是生活助手",
    tools=[get_weather]
)

result = Runner.run_sync(agent, "北京天气怎么样？")
print(result.final_output)
```

这本质上就是我们第一篇 30 行代码的**官方版本**。OpenAI Agents SDK 的优势在于与 OpenAI 生态深度集成（Handoff、Guardrails、Tracing 开箱即用）。

### 框架选型决策树

```mermaid
graph LR
    Start{你的需求?} -->|"简单工具调用<br/>OpenAI 生态"| OpenAI[OpenAI Agents SDK]
    Start -->|"复杂工作流<br/>需要图编排"| LangGraph[LangGraph]
    Start -->|"多角色协作<br/>脑暴/辩论"| CrewAI[CrewAI]
    Start -->|"研发调试<br/>对话式协作"| AutoGen[AutoGen]
    Start -->|"无代码<br/>快速原型"| Dify[Dify / n8n]
    Start -->|"理解原理<br/>从零学习"| Ours[本系列代码 ✨]

    style Ours fill:#2b2d42,color:#fff
    style LangGraph fill:#667eea,color:#fff
    style CrewAI fill:#8B5CF6,color:#fff
    style AutoGen fill:#4299e1,color:#fff
    style OpenAI fill:#764ba2,color:#fff
    style Dify fill:#10b981,color:#fff
```

---

## 生产部署：FastAPI + Docker

### 架构总览

```mermaid
graph LR
    Client[客户端] --> Nginx[Nginx 反向代理]
    Nginx --> API[FastAPI Agent 服务]
    API --> Redis[Redis<br/>会话缓存]
    API --> VectorDB[(ChromaDB<br/>向量记忆)]
    API --> SQL[(PostgreSQL<br/>情节记忆)]
    API --> LLM[LLM API<br/>gpt-4o / claude]

    API --> Prometheus[Prometheus<br/>指标收集]
    Prometheus --> Grafana[Grafana<br/>监控面板]

    style API fill:#667eea,color:#fff
    style LLM fill:#8B5CF6,color:#fff
```

### FastAPI 服务实现

```python
# agent_api.py — 生产级 Agent API 服务
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager

# 会话存储 (生产环境用 Redis)
sessions: dict[str, dict] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载 Agent
    app.state.agent = SecureAgent(
        base_agent=MemoryAwareAgent(
            system_prompt="你是一个专业的 AI 助手。",
            registry=create_default_registry(),
            persist_dir="/data/agent_memory"
        ),
        sandbox=True
    )
    logger.info("Agent 服务已启动")
    yield
    # 关闭时清理
    logger.info("Agent 服务正在关闭")

app = FastAPI(title="AI Agent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── 请求/响应模型 ──
class AgentRequest(BaseModel):
    query: str = Field(..., description="用户问题", max_length=10000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = Field(default="auto", description="模型选择: auto/gpt-4o/gpt-4o-mini/o3-mini")
    stream: bool = Field(default=False)

class AgentResponse(BaseModel):
    session_id: str
    answer: str
    iterations: int
    tokens_used: int
    latency_ms: float
    model_used: str

# ── API 端点 ──
@app.post("/agent/chat", response_model=AgentResponse)
async def chat(request: AgentRequest):
    """Agent 对话端点"""
    start_time = time.time()
    logger.info(f"[{request.session_id}] 收到请求: {request.query[:100]}")

    # 获取或创建会话
    if request.session_id not in sessions:
        sessions[request.session_id] = {
            "created_at": time.time(),
            "messages": [],
            "token_total": 0
        }

    # 选择模型
    if request.model == "auto":
        tier = CostController().select_model(request.query)
        model_name = tier["model"]
    else:
        model_name = request.model

    # 执行 Agent
    agent = app.state.agent
    agent.agent.model = model_name

    try:
        answer = await asyncio.to_thread(agent.run, request.query)
    except Exception as e:
        logger.error(f"[{request.session_id}] Agent 执行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start_time) * 1000

    # 更新会话统计
    session = sessions[request.session_id]
    session["messages"].append({"role": "user", "content": request.query})
    session["messages"].append({"role": "assistant", "content": answer})

    logger.info(f"[{request.session_id}] 完成: {latency:.0f}ms")

    return AgentResponse(
        session_id=request.session_id,
        answer=answer,
        iterations=getattr(agent, '_last_iterations', 0),
        tokens_used=getattr(agent, '_last_tokens', 0),
        latency_ms=latency,
        model_used=model_name
    )

@app.get("/agent/sessions/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    return sessions[session_id]

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "uptime": time.time() - start_time
    }

@app.get("/metrics")
async def metrics():
    """Prometheus 指标端点"""
    return {
        "agent_requests_total": total_requests,
        "agent_latency_p50": latency_histogram.get("p50", 0),
        "agent_latency_p95": latency_histogram.get("p95", 0),
        "agent_success_rate": success_rate,
        "agent_active_sessions": len(sessions),
    }


# ── 启动 ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制源码
COPY agent_api.py .
COPY agent_core/ ./agent_core/
COPY configs/ ./configs/

# 创建数据目录
RUN mkdir -p /data/agent_memory

# 非 root 用户运行
RUN useradd -m agentuser && chown -R agentuser:agentuser /app /data
USER agentuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### docker-compose.yml

```yaml
# docker-compose.yml
version: "3.9"

services:
  agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://agent:agent@postgres:5432/agent
    volumes:
      - agent_data:/data/agent_memory
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent
      POSTGRES_DB: agent
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent"]
    volumes:
      - pg_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - agent-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    restart: unless-stopped

volumes:
  agent_data:
  redis_data:
  pg_data:
```

---

## 医疗机器人场景案例

作为本系列的结束，让我们看一个与你（医疗机器人算法工程师）场景直接相关的 Agent 应用：

### 医学图像分析工作流 Agent

```python
# 医学图像分析 Agent 配置
medical_agent = MemoryAwareAgent(
    system_prompt="""你是医学图像分析助手，专精于：

1. **分割模型推荐**：根据图像模态（CT/MRI/超声/X光）和目标器官推荐合适的模型
   - nnU-Net：通用分割基线（所有模态）
   - V-Net：3D 医学图像（CT/MRI 体积数据）
   - Attention U-Net：需要关注特定区域时
   - TransUNet：需要结合全局上下文时
   - SAM/MedSAM：零样本分割需求

2. **预处理建议**：
   - CT：窗宽窗位调整、HU 值归一化
   - MRI：偏置场校正（N4ITK）、多序列配准
   - 超声：散斑降噪、时间增益补偿

3. **训练建议**：
   - 损失函数选择（Dice/CrossEntropy/Focal/组合）
   - 数据增强策略（弹性形变对医学图像特别重要）
   - 小样本策略（迁移学习、少样本学习）""",

    registry=create_medical_registry(),
    persist_dir="./medical_agent_memory"
)

# 注册医学专用工具
@medical_agent.registry.register
def recommend_segmentation_model(
    modality: Literal["CT", "MRI", "Ultrasound", "X-ray"],
    organ: str,
    data_size: Literal["small", "medium", "large"] = "medium"
) -> dict:
    """根据图像模态和目标器官推荐分割模型"""
    recommendations = {
        ("CT", "liver"): ["nnU-Net", "V-Net"],
        ("MRI", "brain"): ["nnU-Net", "Attention U-Net", "TransUNet"],
        ("CT", "lung"): ["nnU-Net", "U-Net++"],
        ("Ultrasound", "heart"): ["Attention U-Net", "DeepLabV3+"],
    }
    key = (modality, organ.lower())

    if key in recommendations:
        models = recommendations[key]
        if data_size == "small":
            models = [m for m in models if m != "TransUNet"]
            models.append("nnU-Net (with transfer learning)")
        return {"recommended": models, "baseline": "nnU-Net"}

    return {"recommended": ["nnU-Net"], "rationale": "nnU-Net 作为通用基线适用于未见过的模态/器官组合"}


# 使用示例
result = medical_agent.run(
    "我有一组腹部 CT 数据，需要分割肝脏和肿瘤，数据量约 50 例。"
    "推荐合适的模型和预处理策略。"
)
```

---

## 八篇之旅：架构演进全景

```mermaid
graph LR
    W1[第1篇<br/>30行最小Agent] --> W2[第2篇<br/>类型安全工具系统]
    W2 --> W3[第3篇<br/>Token感知对话管理]
    W3 --> W4[第4篇<br/>4层记忆系统]
    W4 --> W5[第5篇<br/>多Agent协作]
    W5 --> W6[第6篇<br/>MCP/A2A协议]
    W6 --> W7[第7篇<br/>评测+安全护栏]
    W7 --> W8[第8篇<br/>框架选型+部署]

    W1 -.->|30行| C1["ReAct 循环<br/>LLM + 工具 + While"]
    W2 -.->|350行| C2["@tool 装饰器<br/>Python→JSON Schema"]
    W3 -.->|200行| C3["Token 计数<br/>滑动窗口+摘要"]
    W4 -.->|400行| C4["ChromaDB<br/>HNSW向量检索"]
    W5 -.->|300行| C5["Generator-Verifier<br/>Orchestrator-Workers"]
    W6 -.->|350行| C6["MCP Client/Server<br/>A2A Agent Card"]
    W7 -.->|250行| C7["LLM-as-Judge<br/>3层安全护栏"]
    W8 -.->|200行| C8["FastAPI+Docker<br/>生产级服务"]

    style W8 fill:#2b2d42,color:#fff
    style C8 fill:#4299e1,color:#fff
```

从 30 行到生产服务，我们的代码累积增长到了约 2000 行——但这 2000 行定义了理解任何 Agent 框架所需的全部核心概念。

---

## 系列总结

### 核心收获

通过这八篇文章，你不仅构建了一个完整的 AI Agent，更获得了：

1. **底层原理的直觉**：理解了 ReAct 循环的 MDP 建模、工具选择的多分类本质、记忆检索的 ANN 算法、多 Agent 协作的博弈论基础
2. **工程实践的技能**：从类型系统到 Schema 映射，从 Token 计数到 BPE 原理，从向量检索到 HNSW 索引
3. **框架选型的能力**：不再被框架文档牵着走，而是能够理解每个框架"在封装什么"

### 推荐学习路径

```
入门：本系列 8 篇 → 理解 Agent 底层原理
  │
  ├─→ 想深入原理？→ 读 ReAct/ MemGPT/ Anthropic 设计指南 论文
  │
  ├─→ 想用框架？  → 选 LangGraph (复杂流程) 或 CrewAI (多角色)
  │
  ├─→ 想自己造？  → 基于本系列代码扩展（MCP/A2A集成）
  │
  └─→ 想生产部署？→ 本文 Docker Compose 一键启动
```

### 关于我

作者杨钱俊 (YangCazz)，医疗机器人算法工程师，专注深度学习、医学图像处理和 AI Agent 技术。本博客（YangCazz.github.io）持续分享技术探索，欢迎关注。

**联系方式**：
- GitHub: [YangCazz](https://github.com/YangCazz)
- Email: yangcazz@qq.com

---

## 参考文献

<ol class="references">
<li><em>Anthropic. "Building Effective Agents."</em> Anthropic Research Blog, Dec 2024.<br><a href="https://www.anthropic.com/research/building-effective-agents">https://www.anthropic.com/research/building-effective-agents</a></li>
<li><em>LangChain. "LangGraph Documentation."</em> LangChain, 2024.<br><a href="https://langchain-ai.github.io/langgraph/">https://langchain-ai.github.io/langgraph/</a></li>
<li><em>CrewAI. "CrewAI Documentation."</em> CrewAI, 2024.<br><a href="https://docs.crewai.com/">https://docs.crewai.com/</a></li>
<li><em>Microsoft. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation."</em> GitHub, 2024.<br><a href="https://github.com/microsoft/autogen">https://github.com/microsoft/autogen</a></li>
<li><em>OpenAI. "Agents SDK."</em> OpenAI, 2025.<br><a href="https://github.com/openai/openai-agents-python">https://github.com/openai/openai-agents-python</a></li>
<li><em>Dify. "Dify — LLM Application Development Platform."</em> GitHub, 2024.<br><a href="https://github.com/langgenius/dify">https://github.com/langgenius/dify</a></li>
<li><em>FastAPI. "FastAPI Documentation."</em> tiangolo, 2024.<br><a href="https://fastapi.tiangolo.com/">https://fastapi.tiangolo.com/</a></li>
<li><em>Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."</em> Nature Methods 2021.<br><a href="https://www.nature.com/articles/s41592-020-01008-z">https://www.nature.com/articles/s41592-020-01008-z</a></li>
</ol>
