---
layout: post
title: "Super Agent 架构解读：从单次会话走向持续在线的协作者"
date: 2026-10-31 21:30:00 +0800
categories: [AI, 软件工程]
tags: [AI Agent, Harness Engineering, Tool Calling, Subagent, 安全]
excerpt: "DeepSeek 的 Super-agent 设计探索，把问题从单任务的模型能力转向多任务中的人的注意力。本文拆解 Router/Worker、事件直通、链式召回、异步交互与持久运行，并核查原型和初步实验的适用边界。"
image: "/assets/images/research/20261031/super-agent-routing-v1.png"
cover_image: false
---

## 当瓶颈从模型转向人的注意力

一个 Agent 能修好一个问题，不代表它能同时跟进代码评审、群聊反馈、CI 失败和新的需求变更。如果每项工作都对应一个独立 Session，用户就得自己记住：哪段讨论发生在哪个窗口，哪项任务仍在等待，哪个决定已经被后来消息推翻。

DeepSeek 的《Super-agent 设计探索：迈向更自主的 Agent 系统》把这个问题放在了中心：单任务的上限与模型能力密切相关，多任务协作的上限还受到人的注意力约束。原文所说的 Super-agent，是一种持续在线、跨会话协调工作的系统形态，而不是发布一个新模型，也不是介绍 DeepSeek Harness 已有产品功能。<cite>[1]</cite>

这篇设计探索最有价值的地方，在于它把 **Session 从用户必须维护的工作单位，重新放回系统的实现层**。用户仍可以在 IM、GitHub 或邮件中推进事情；系统负责把分散的交互连成任务，而不要求人充当会话之间的同步器。

本文沿着这一问题拆解架构，并区分三类证据：原文提出和报告的设计，公开原型实际承诺的范围，以及本文延伸的工程判断。相关图为依据原文关系重新组织的示意，不是官方架构原图。

## Router 与 Worker：拆开连续性和执行吞吐

把所有消息塞进一个 Session，容易保持上下文连续，却会把耗时执行和新消息处理挤在一起。为每条消息开新 Session 可以提高并发，但跨任务决定容易割裂，对外回复也会混乱。

原文采用双层职责划分：

| 层次 | 应负责什么 | 有意避免什么 |
| --- | --- | --- |
| Router | 人类交互、全局关联、任务分派、汇总关键结果 | 亲自执行耗时的具体工作 |
| Worker | 在隔离上下文中推进具体任务，调用执行工具 | 各自向外部信道输出过程，争夺用户注意力 |

Router 是逻辑上的全局协调点，不意味着它必须规划每个任务的全部细节。Worker 内部仍可使用 Subagent 或 Agent Team，执行复杂任务的内部协作对 Router 保持透明。原文的实现选择单个常驻 Router Session，但“双层”本身并不限定每层只有一个会话。<cite>[1]</cite>

这与常见的单任务多代理分工有一处重要差异：后者通常让负责人深度介入同一任务的拆解；这里优先解决的是**多个任务之间的连续交互**。因此，是否采用多个角色不是核心，职责和信息流怎样分开才是核心。

此前的 [Subagent 架构文章]({% post_url 2026-05-22-subagent-architecture-isolation %})讨论了委派与隔离。这里进一步要解决的，是任务结束之后仍持续到来的事件，以及用户跨信道改变方向时的全局一致性。

不过，中心化协调也带来权衡。原文认为较轻量模型可胜任其当前 Router 工作负载，并观察到批处理缓解了负载；这不等于任意任务规模下都可以使用弱模型。Router 一旦误判任务归属或漏掉全局约束，多个 Worker 可能高效地沿着错误方向推进。水平切分可以增加吞吐，却也会重新引入分片之间的记忆与协调问题。

## 事件进入系统后，不应每次都重新“思考路由”

原文把群聊消息、代码评论、CI 结果等半结构化输入称为 **Trigger**；同一个 PR、IM 话题或邮件串下的连续事件组成 **Thread**。这两个对象把 Agent 的驱动方式从“等待一条提示词”转为“持续处理环境事件”。<cite>[1]</cite>


<figure class="research-figure research-figure--scientific">
  <a href="{{ '/assets/images/research/20261031/super-agent-routing-v1.svg' | relative_url }}"><img src="{{ '/assets/images/research/20261031/super-agent-routing-v1.png' | relative_url }}" alt="环境事件经过程序与语义策略过滤，新任务由 Router 分派，已订阅事件直通隔离 Worker；对外回复仍经 Router，历史上下文沿 ID 引用链召回。" loading="lazy" decoding="async" width="1536" height="1024" style="width:100%;height:auto;"></a>
  <figcaption><strong>AI 生成科研示意图（非论文原图）</strong>。本图为依据原文第 2—3 节制作的原创矢量解释图：橙色表示当前事件与直通路径，青绿色表示 Router/Worker 计算角色，蓝色表示按需召回的历史对象。Router 保留全局观察，但不必逐条转发已订阅事件；下方重复展示的是同一 Router 的输出职责。省略部分创建、取消订阅及异常控制路径，Thread X/Y 仅为示例，无实测数值。依据<a href="https://mp.weixin.qq.com/s/zoS7GlReWPhfCjENGxXCdg">原文设计探索</a>；<a href="{{ '/assets/images/research/20261031/super-agent-routing-v1.svg' | relative_url }}">打开矢量图放大阅读</a>。</figcaption>
</figure>

### 两种 Fast Path，不要混在一起

第一种是**事件过滤**。明确的来源、群组和准入规则先由程序检查；需要理解语义的候选事件再交给模型。它减少的是不必要的模型判断。

第二种是**订阅直通**。新任务由 Router 决定归属并创建 Worker；Worker 订阅相关 Thread 后，后续事件可以由程序直接送达 Worker，不必让 Router 每次再调用工具转发。它减少的是已经确定归属后的重复路由。

这里有两个容易在简化架构图中画错的细节：

- **直通不等于 Router 失明。** 原文仍允许 Router 观测全局事件；只是已订阅事件的转发不再依赖它逐次决策。
- **订阅不是一对一绑定。** 一个 Worker 可以订阅多个 Thread，一个 Thread 也可以通知多个 Worker。关键结论仍回到 Router，由它统一面向用户回复。

这种结构与事件驱动系统的发布/订阅思路相通：确定性的归属交给程序，语义不确定性留给模型。共享 Thread 能让 Worker 有机会发现彼此的操作，但**收到相同通知不等于已经具备并发一致性保证**。修改同一分支、同一文档或同一外部对象时，仍可能需要版本校验、冲突检测与合并责任。这一补充属于工程推论，不能视为原文已经解决的能力。

## Inline & Spill：把上下文当成可管理的工作集

持续事件流很快会超过任何有限的上下文窗口。把全部历史都喂给模型，会挤占窗口并混入噪音；只给一个 ID，又可能迫使模型不断读取工具，增加往返。

原文用 **Inline & Spill** 表达这个权衡：近期很可能需要的信息直接内联；长内容或低频细节保留在窗口之外，窗口内留下摘要、ID 和续读入口。<cite>[1]</cite> 它借用了 Cache 和局部性的思想，但不是说上下文窗口真的具备硬件缓存的命中、一致性或替换保证。

例如，一条代码评审事件可以内联当前评论、相邻评论摘要和 PR 标识，而不是把整个仓库与完整讨论一并塞入。若需要详细背景，再沿入口取回。

当 Trigger、Thread、Worker Session 都有唯一 ID，并在返回对象中附带关联对象的 ID，就形成 **Chain of Recall（链式召回）**：

```text
当前消息 → 所属 Thread → 关联 Worker Session
         → Worker 关联的另一个评审 Thread → 所需历史细节
```

这是沿引用图逐跳展开上下文，不应把它误读为另一个 Chain of Thought，也不等同于完整的长期记忆系统。原文明确把跨月乃至更长周期的记忆列为开放问题。

Anthropic 的上下文工程实践也讨论了按需加载、压缩、结构化笔记和子代理隔离。<cite>[2]</cite> 两者的共同启示是：上下文质量取决于选择与组织，而不只是窗口容量。链式召回的特点则在于对象之间显式可追踪的关联。

进一步落地时，应问几个更细的问题：摘要是否过时？关联对象是否已被删除？引用跨越了哪个权限边界？召回到的决定是否已经被后来版本取代？**能找到历史，不等于找到仍然有效的历史。** 这些需要版本、时效和访问控制配合，不能交给“模型自己记住”来兜底。

## 异步交互与投机执行：减少等待，也要控制浪费

ReAct 提供了推理、行动与观察交错的基本范式。<cite>[3]</cite> 原文针对的是常见的同步工具运行方式：调用耗时的 `run()` 后，会话等待工具结束，期间难以及时处理新输入。

其异步改造把长任务拆成 `start()` 与 `stop()`：启动后立即返回 Started；任务在后台运行，最终结果以新事件回到会话；模型可在需要时发出停止请求。新输入在当前推理步结束后即可被处理，无须等待后台工具完成。<cite>[1]</cite>

因此，“全双工”描述的是前台交互与后台执行解耦，**并不意味着正在生成的每一个模型 token 都可被瞬时抢占**。Started 也只表示已经启动，不能当作成功完成。若用户中途改变目标，系统还需要识别旧任务的晚到结果，避免把过时产物继续交付。

投机执行则更进一步：在尚不确定结果是否会被采用时，先在后台复现问题、搜集材料或准备草稿，以额外计算换取更少的人类等待。原文报告了内部群聊中的主动问题处理实践，但这不是对所有公开消息自动行动的通用授权。

这里值得借鉴计算机系统的思想，也需要看到类比的边界：CPU 可以丢弃投机结果，现实系统未必能撤回一封邮件或一次公开发布。本文建议把投机工作优先限制在隔离、可丢弃的草稿和检查中；跨过外部副作用边界之前，再检查授权、目标版本和验收条件。

输出也应适配信道。群聊适合简短结果与需要决策的问题；PR、邮件和文档承载完整证据。统一回复出口的意义，不只是少发消息，更是让人不必从多个 Worker 的过程日志中重新拼出一个结论。

## 持久在线，需要拆开计算、状态和执行环境

原文的另一条主线，是将三件常被捆在本机上的东西分开：推进 Agent Loop 的计算、需要保存的会话状态，以及带有代码、工具链和服务依赖的执行环境。<cite>[1]</cite>


<figure class="research-figure research-figure--scientific">
  <a href="{{ '/assets/images/research/20261031/super-agent-persistence-v1.svg' | relative_url }}"><img src="{{ '/assets/images/research/20261031/super-agent-persistence-v1.png' | relative_url }}" alt="无状态 Agent Loop 将会话持久化到外部存储，通过只执行命令的 daemon 接入真实环境，外部 Watchdog 同时观察系统健康与任务进展。" loading="lazy" decoding="async" width="1536" height="1024" style="width:100%;height:auto;"></a>
  <figcaption><strong>AI 生成科研示意图（非论文原图）</strong>。本图为依据原文第 5—6 节制作的原创矢量解释图，展示架构设想而非当前开源原型的完整部署清单。青绿色表示计算与执行组件，蓝色表示状态及结果回流，橙色表示工具请求，灰色虚线表示独立 Watchdog 控制回路；只画出一条代表性的结果返回路径，其余环境同理。Loop 数量仅作示意，无实测数据；信任边界、凭据隔离与权限控制仍需另行设计，Watchdog 不等于安全保证。依据<a href="https://mp.weixin.qq.com/s/zoS7GlReWPhfCjENGxXCdg">原文设计探索</a>；<a href="{{ '/assets/images/research/20261031/super-agent-persistence-v1.svg' | relative_url }}">打开矢量图放大阅读</a>。</figcaption>
</figure>

第一步是计算与状态分离。Agent Loop 可以运行在可替换的云端计算容器中，Session 等状态保存在持久存储里。容器重启不必意味着对话历史和任务关系丢失。

第二步是计算与执行环境分离。真实环境未必容易搬到云端：未提交修改、内网依赖、特定硬件和工具链，都可能使同步不完整。原文提出让环境留在原地，以轻量 daemon 接收命令并返回结果；云端 Agent 可以连接个人机器，也可以使用托管环境。

原文还把这一路线与 Anthropic 的 *Scaling Managed Agents: Decoupling the brain from the hands* 联系起来。后者同样分离 Harness、会话日志和执行接口，但主要面向托管环境；这里特别关心带有本地状态、可能间歇在线的个人设备。共同的分离原则，不意味着两者部署假设或实现已经相同。<cite>[7]</cite>

这个设计让“Agent 持续在线”与“某台执行设备暂时离线”成为不同状态。它并不让离线机器上的任务凭空继续执行；需要该环境的工作仍须等待环境恢复，或明确切换到另一个满足条件的环境。

### Watchdog 监控的不只是进程活着

原文在主系统之外放置独立 Watchdog Session，用定时事件驱动外部控制回路。它既关注接口错误、恢复失败，也尝试判断 Worker 有没有实质进展、是否偏离任务目标。<cite>[1]</cite>

这一思路抓住了 Agent 的特殊故障：进程和 API 都正常，任务却可能不断重复无效动作。但模型判断进展同样会出错，Watchdog 也不是万能的监督者。本文建议把“最近新增了什么可验证产物”“同一失败是否重复”“是否超过预算”等可观察信号与语义判断结合，并约束自动修复的权限。否则监督层可能变成另一个有完整工具、但缺少制约的行动者。

跨设备连接也扩大了风险范围。环境事件可能携带提示词注入，端侧又可能保存私有代码与凭据。原文已明确承认安全和隐私问题尚未解决。系统应将事件内容视为数据，而不是新的授权来源；凭据尽量通过受控机制使用，不直接进入模型上下文；日志还应限制敏感内容、访问范围与保留期限。

## 初步实验支持什么，又没有支持什么

原文给出两组实验，但必须把结论放回实验对象。端到端消融使用约 300 条群聊与 GitHub CI 消息，其中约 20 条是工作请求；**编码工作被只消耗时间的工具替代**，每种方案重复 5 次。消息间隔与工作时长按四分之一缩放，模型调用耗时保持真实值。完成时间指从工作请求到结果消息的耗时；仅有 5 次重复，且原文未报告统计检验，本文不据此断言统计显著性或生产稳定性。<cite>[1]</cite>

| 原文报告 | 可以说明什么 | 不能推出什么 |
| --- | --- | --- |
| Super-agent 完成耗时 p50 为 12.4 分钟、p95 为 15.7 分钟 | 在该合成负载与配置下的调度表现 | 真实编码任务的质量、通用速度或生产 SLA |
| 中性提示的单 Session 基线，5 次均未调用 Subagent | 在该设置中，临场委派受提示和模型行为影响 | 所有单 Session 系统都不会正确委派 |
| 去掉直通后，耗时基本一致，Router 决定数约增 30%，读缓存 token 从 2323 万增至 3177 万 | 直通在该负载下节省路由与 token 开销 | 直通已经提高了任务质量或完成速度 |

这组实验中，系统约束 Router 不亲自执行，比只用提示词劝主会话委派更稳定；同时，多会话方案付出了更多读缓存 token。合理的结论是一个**调度时延与读缓存 token 开销之间的权衡**，不是“多 Agent 全面胜出”。

第二组压力测试使用真实 Router 模型与 Mock Worker，观察到事件增多时积压事件自动形成批处理，单个决定分摊的推理开销下降。它支持作者在当时团队规模下暂不急于切分 Router 的选择，但不能证明无限吞吐。引用这些实验数字时，也应明确它们是原文作者报告，本文未独立复现。

MAST 对多代理故障的研究，将问题归纳为系统设计、代理间失配和任务验证三类；它提醒我们调度更快之外仍有协作失败。<cite>[4]</cite> τ-bench 则关注最终数据库状态及重复试验的一致成功，而不仅是助手是否回复“已完成”。<cite>[5]</cite>

由此可以提出后续评测：保留合成事件负载，同时加入有真实验收条件的任务，固定模型、预算与工具权限，记录成果正确性、重复可靠性、尾延迟、token、人工介入和权限违规；再注入重复事件、乱序、设备断线与用户叫停。这里是本文的评测建议，不是原文已经完成的实验。

## 开源原型与完整愿景之间仍有距离

公开的 `deepseek-ai/dsh-superagent-prototype` 仓库采用 MIT 协议，但 README 明确说明：它是研究原型，不是 DeepSeek 的正式产品，也不是生产就绪的软件。公开代码经过适配，与内部系统及文章描述不完全一致，文章中的部分设计尚未实现。<cite>[6]</cite>

其可用范围聚焦于单个 GitHub 仓库的主动编码协作，可选接入飞书，并提供 Worker 会话/目录与需要人介入的 Web 面板。下面固定到提交 `6337fc8` 核对设计与实现，避免把可变仓库首页当成永久能力说明：

| 关注点 | 原文设计 | 公开原型文档显示的边界 |
| --- | --- | --- |
| Router 工具 | 可不配 Bash，专注轻量协调 | 主会话实际拥有 Bash、read 和 job 工具；避免亲自读代码主要依赖提示规则，而非撤销工具权限 <cite>[8]</cite> |
| 事件直通 | 已订阅 Thread 的事件直接进入 Worker | 文档明确：直通提供上下文，不授予行动权限；事件准入不等于操作授权 <cite>[9]</cite> |
| 持久状态 | 计算、状态与执行环境分离 | 文档给出 SQLite 状态存储、目录锁与每 Ticket 的会话/目录；不能据此宣称完整云端 BYOM 架构已实现 <cite>[8]</cite> |
| 外部恢复 | Watchdog 判断故障和任务进展 | 部署文档提供由操作者设置的独立 profile 定时恢复检查；不应把它写成完整语义监督已自动启用 <cite>[10]</cite> |

尤其需要注意，部署文档说明两类预设使用 `danger-full-access`，以操作者的操作系统身份运行。凭据包装脚本提供默认选择，不构成文件或进程隔离；Worker 仍可能访问该身份可访问的主目录和路径。文档建议使用专用操作系统账户。<cite>[10]</cite> 因而本文不把默认启动方式作为安全部署建议，也不提供一键接入真实凭据的命令。

阅读这个原型，更适合追踪“哪些对象、事件和调度机制已落到代码”，而不是依据架构图假设完整的跨设备个人助手已经开箱可用。

此前的 [Harness Engineering 文章]({% post_url 2026-05-22-harness-engineering-agent-kernel %})讨论执行基础设施；这次设计探索进一步把系统问题推向了持续事件流和人的注意力。其贡献主要是组织已有工程思想：协调与执行分层、确定性快路径、按需召回、异步任务、持久状态和外部监督。

我更看重的判断是：**减少人的会话管理负担，不能靠隐藏更多执行细节来完成，而要靠更清楚的任务状态与结果证据。** Router 让交互连续，Worker 让执行并行；订阅与召回控制信息流；异步机制减少阻塞；持久状态和 Watchdog 支持长期运行。它们相互配合，才有机会把一个聊天窗口背后的调用循环，变成能在真实工作环境中持续协作的系统。

下一步真正值得观察的，不只是它能同时启动多少工作，而是面对冲突、过时信息、模糊授权和环境故障时，是否仍知道该继续、该等待、该停止，还是该把问题交还给人。

## 参考文献

1. *Super-agent 设计探索：迈向更自主的 Agent 系统.* Yujia Qiao；DeepSeek Harness 团队实践。依据提供的原文全文与配图核查，核查日期：2026-10-31。<https://mp.weixin.qq.com/s/zoS7GlReWPhfCjENGxXCdg>
2. *Effective context engineering for AI agents.* Anthropic Applied AI. 2025-09-29。<https://www.anthropic.com/research/effective-context-engineering-for-ai-agents>
3. *ReAct: Synergizing Reasoning and Acting in Language Models.* Yao S, et al. ICLR, 2023。<https://arxiv.org/abs/2210.03629v3>
4. *Why Do Multi-Agent LLM Systems Fail?* Cemri M, et al. 2025，arXiv v3（2025-10-26）。<https://arxiv.org/abs/2503.13657v3>
5. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.* Yao S, et al. 2024。<https://arxiv.org/abs/2406.12045>
6. *DSH Superagent Prototype — README.* DeepSeek Harness Team，研究原型，提交 `6337fc823ce20251c8de8085948165c81aaa701e`，核查日期：2026-10-31。<https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/README.md>
7. *Scaling Managed Agents: Decoupling the brain from the hands.* Anthropic. 2026-04-08。<https://www.anthropic.com/research/managed-agents>
8. *DSH Superagent Prototype — Architecture.* 同上固定提交，核查日期：2026-10-31。<https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/007-architecture.md>
9. *DSH Superagent Prototype — Authorization.* 同上固定提交，核查日期：2026-10-31。<https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/002-authorization.md>
10. *DSH Superagent Prototype — Deployment.* 同上固定提交，核查日期：2026-10-31。<https://github.com/deepseek-ai/dsh-superagent-prototype/blob/6337fc823ce20251c8de8085948165c81aaa701e/docs/006-deployment.md>
{: .references }
