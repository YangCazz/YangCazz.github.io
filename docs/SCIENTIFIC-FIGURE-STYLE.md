# 科研配图范式：iCaRL 锚点 / SF-01

## 用户批准的唯一视觉锚点

2026-10-30 用户明确选择「第二张图」的风格，要求其他图沿用并形成范式。对应本地文件：

`assets/images/research/20261030/icarl-training-classification-v3.png`

SHA-256：`69b318ffd725e979fef5ff6ce6fc08622d4f2104784f95f547db441a0e70e792`

**不得覆盖或重新生成这张已批准原件。** 修改其他图时将它作为风格参考，必须另存新版本；它不是其他算法的内容模板。用户拒绝的是 v2 纸艺/风景/玩具隐喻，不是要求增加艺术装饰。“鲜活”指科研图的结构层次、数据原语和信息组织不僵硬。

## 可复用设计参数

下表是为后续可重复制作制定的标准 token，不声称生成 PNG 的每个抗锯齿像素或原始字体都严格等于这些值。原图字体没有可读取的字体元数据；视觉上为 Times 类衬线体。

| 用途 | 标准 |
| --- | --- |
| 画布 | 横向 1536 × 1024；白底 `#FFFFFF`；外边距至少 24 px |
| 正文/主箭头 | 深海军蓝 `#0B2045` |
| 历史信息、回放样本 | 蓝 `#1767B4`；浅填充 `#EDF4FC` |
| 当前数据、动作 | 橙 `#F06A22`；浅填充 `#FFF2E8` |
| 当前模型、特征 | 青绿 `#087E83`；浅填充 `#EDF8F7` |
| 分区线/次注释 | 灰蓝 `#8091AA`；细实线或有语义的短虚线 |
| 字体 | 可控矢量用 `Times New Roman, Liberation Serif, serif`；中文解释放 HTML 图注，用站点中文字体 |
| 字级 | 标题 40–46 px；泳道标题 30–34 px；模块标签 22–26 px；注释 18–22 px；图内最小 17 px |
| 笔画 | 主要箭头/模块边 2–3 px；辅助线 1–2 px；同一图保持统一 |
| 箭头 | 实心小三角、直线或直角折线；无装饰弧形箭头海；避免穿字、穿模块 |
| 模块 | 小圆角约 6–10 px；浅填充、留白；不能统一成僵硬等宽卡片链 |
| 层堆叠 | 只对样本栈、特征图/张量薄片使用轻透视；偏移约 8–18 px；无投影场景或外部光晕 |

## 布局语法

1. 先确认原文的科学关系，再布局。一般从左至右呈现主数据流；训练与推理、真实经验与想象学习分泳道。
2. 泳道用细水平分隔线与小海军蓝编号圆标，不用巨大粗边框整块包围。编号表示阅读区，不得暗示原算法没有的严格执行顺序。
3. 主要计算模块大小由内容决定；损失、预算、采样与更新等支持信息用旁路、括号或小插图。适度错列与内部张量细节提供层次，不靠装饰“活泼”。
4. 数据卡、样本栈、特征向量、薄层模块是可复用原语。不同颜色用于来源或功能，同一语义整图保持一致；不要把颜色同时解释成数据来源与类别标签。
5. 箭头必须能回答“谁向谁传什么”。把数据流、梯度、参数更新、控制关系分开说明；省略的线路在图注注明，不允许用近似图形冒充完整计算图。
6. 示例几何样本均为占位示意。不能画虚构准确率、损失曲线或柱状测量；有真实数据才画定量图，并记录数据来源和单位。
7. 其他算法继承**视觉语法**，不继承 iCaRL 的损失、均值分类器、内存预算等语义。

## 图注与资产数据

图注首句：**AI 生成科研示意图（非论文原图）**。随后中文说明颜色/数据流、简化与省略、示例是否非实测；链接到核查过的原论文。图片有准确中文 alt，说明关键关系而不是“某某插图”。

稳定文件名带版本；每次替换新增清单项，登记 source URL/title/version/date、核查日期、页码/算法编号、版权边界、完整生成与修正提示词、参考图片、最终 SHA256、尺寸、used_by。旧文件保留，废弃项 `status: archived`、`used_by: []`，填写原因。不声称未暴露的生成模型版本；不得将生成图叫作论文截图、架构原图或实验照片。

精确图不作为标题背景裁切：`cover_image: false`。正文完整比例展示，允许放大阅读；小屏图注保留关键信息，避免靠缩小文字代替响应式设计。

## 提示词模板

```text
Reference image 1 is the approved SF-01 iCaRL style anchor.
Reference image 2 is the target figure (if editing).
Output only the target method, preserving its verified scientific semantics.
Match the anchor: Times-like serif hierarchy, navy thin arrows, white canvas,
small-radius pale modules, blue stored data, orange current data/actions,
teal feature-map slabs, fine lane separators, numbered badges, support insets.
Do not copy iCaRL semantics into this method.
Title: [exact short title]
Verified nodes: [node list]
Directed edges: [source -> target; carried quantity; phase]
Support tracks/insets: [memory, loss, budget, or evaluation; exact meaning]
Text labels: [exact labels]
Schematic primitives: [samples/tensors; no measured results]
Omitted details to state in caption: [list]
Invariants: [labels, data colors, flow, approved anchor bytes]
Avoid: giant dashboard bands, heavy sans-serif headlines, scenery, paper art,
plants, toys, glow, invented scores, illegible text, decorative arrows.
```

## 出图验收清单

- [ ] 原文版本/方法关系已核查；图中无没有依据的事实或数据。
- [ ] 已打开实际锚点及目标图；采用 image_gen 参考图编辑，不手工篡改栅格内容。
- [ ] 标题与标签逐字检查；数据来源颜色不漂移，混合样本来源正确。
- [ ] 每个箭头方向/语义正确；训练/推理、价值/奖励、数据/梯度不混淆。
- [ ] 泳道、字级、薄层原语与锚点一致；不改成文艺化隐喻。
- [ ] 无数字实验伪装；占位样本和省略线路在图注说明。
- [ ] 原始批准 iCaRL 哈希不变；新图版本与manifest、prompt、hash匹配。
- [ ] 正文/缩略图/移动端不裁关键标签；提供完整原图或可放大入口。
- [ ] 构建、资产校验和相关测试通过；未测试的浏览器项目单列。

适用范围：当前三张与未来新增科研图；不因本范式而批量重绘全部历史文章、图标或展示页。
