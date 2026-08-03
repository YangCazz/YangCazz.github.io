---
layout: post
title: "类增量学习 (Class-Incremental Learning) 深入:最难的持续学习设定,与方法演进的五波浪潮"
date: 2026-08-03 10:00:00.000000000 +08:00
categories:
- 深度学习
- 人工智能
tags:
- 持续学习
- 深度学习
- PyTorch
excerpt: "为什么类增量学习是持续学习中最难的设定?从旧类判别、表示漂移到分类器偏差,再到方法演进的五波浪潮,系统拆解 Class-IL 的本质与前沿。"
image: "/assets/images/covers/ai-dev-tools.jpg"
---
{% include series-nav.html series="continual-learning" position="top" %}

## 引言

在我们此前的持续学习全景中,我们把增量学习分为任务增量(Task-IL)、类增量(Class-IL)与域增量(Domain-IL)三种设定,并指出 **Class-IL 是最难、也最能检验方法真实水平的一种**。这篇专题文章把它单独拎出来:为什么它难?研究又是如何一步步逼近它的?

先给出一个贯穿全文的直觉:Task-IL 知道「现在该用哪个任务头」,Class-IL 不知道——它必须在一个不断扩大的类别池里做**全类别判别**。仅仅是这一点差异,就足以让绝大多数方法在 Task-IL 上风光无限、在 Class-IL 上原形毕露<cite>[4]</cite>。接下来我们从「为什么难」讲起,再沿着方法演进的五波浪潮,看研究者如何逐步逼近这个难题,最后讨论评估中的陷阱。

## 为什么 Class-IL 最难

作为持续学习的核心设定,Class-IL 的根基同样是灾难性遗忘——这个自 1989 年就被系统研究的现象<cite>[1]</cite><cite>[2]</cite>。但在 Class-IL 中,它的症状被叠加放大。Class-IL 的难点不是单一的,而是**三个问题叠加**:

### 旧类判别:遗忘被直接暴露在决策上

Task-IL 每个任务有自己的输出头,测试时选择正确头即可——旧任务的判别错误被「结构性隔离」了。Class-IL 则共享一个分类器,新类别加入时,分类器的权重更新会**挤压旧类别的决策空间**。遗忘不再是一个抽象的平均指标,而是具体的「把旧类样本分到新类」的错误。这正是 Class-IL 与 Task-IL 的核心分水岭<cite>[3]</cite><cite>[4]</cite>:

```mermaid
graph LR
    TIL[Task-IL<br/>每任务独立输出头] --> S1[任务 ID 已知<br/>选对应头即可]
    CIL[Class-IL<br/>共享单一分类器] --> S2[类别不断加入<br/>全类别联合判别]
    S2 --> D1[分类器偏差<br/>新类挤压旧类空间]
    S2 --> D2[表示漂移<br/>旧类特征被推离]

    class TIL dim
    class CIL core
    class S1 out
    class S2 proc
    class D1 hl
    class D2 hl
```

### 表示漂移:旧类特征的「家园」在移动

当模型在新类别上训练时,不仅分类器在变,**特征提取器的表示空间也在整体漂移**。旧类别样本在新表示下的位置偏离了它们原本聚集的地方——即使模型「还记得」旧类,旧类特征在变化后的空间里也已面目全非。这一现象被称为**表示漂移**(representation drift),是 Class-IL 中比参数覆盖更难察觉、也更根本的遗忘来源<cite>[7]</cite>:

```mermaid
graph LR
    subgraph S1[学习任务 A 之后]
        A1[类别 1<br/>特征簇] --- A2[类别 2<br/>特征簇]
        A3[类别 3<br/>特征簇]
    end
    subgraph S2[学习任务 B 之后]
        B1[类别 1<br/>被推散] --- B2[类别 2<br/>发生漂移]
        B3[类别 3<br/>与新类混淆]
    end
    S1 --> S2

    class A1 core
    class A2 core
    class A3 core
    class B1 hl
    class B2 hl
    class B3 hl
```

同样一批旧类样本,在表示空间中的相对位置被整体「推离」了——这就是为什么即使分类器没坏、也照样分错旧类的原因。

### 分类器偏差:新类天然占优

由于新类样本「当下可见」而旧类样本「只在记忆中」,训练自然偏向新类——分类器对新类的 logits 系统性偏大。这就是**分类器偏差**(classifier bias toward new classes)<cite>[6]</cite>。它解释了为什么即使表示没怎么坏,Class-IL 的准确率也会随任务数急剧下滑:最后一层在「记忆犹新」的新类面前,把旧类样本误分过去了:

```mermaid
graph LR
    subgraph old_logits[旧类 logits]
        O1[旧类 A<br/>logit 被压低] 
        O2[旧类 B<br/>logit 被压低]
    end
    subgraph new_logits[新类 logits]
        N1[新类 C<br/>logit 偏高]
        N2[新类 D<br/>logit 偏高]
    end
    O1 --> R1[softmax 偏向新类<br/>旧类样本被误分]
    O2 --> R1
    N1 --> R1
    N2 --> R1

    class O1 dim
    class O2 dim
    class N1 hl
    class N2 hl
    class R1 core
```

值得注意的是,偏差**不只是新类「占便宜」**——它还随任务数累积:每加一批新类,旧类 logits 就被进一步压低。这正是「偏差校正」方法(见第三波)的切入点:它们在决策层显式地学习一个校正,例如 BiC 用一个小型线性层 $g(x) = \alpha x + \beta$ 作用在 logits 上,把新类的偏置「掰回来」<cite>[6]</cite>。

一个推论:解决 Class-IL,至少要同时面对**表示漂移**(表示层面)与**分类器偏差**(决策层面)——早期方法往往只治其一,这就是后续演进的核心驱动力。

## 第一波:重放 + 蒸馏的奠基(2017–2019)

### iCaRL:三件套范式

**iCaRL**(Incremental Classifier and Representation Learning)是 Class-IL 的奠基之作,它把三个组件组合成了一个至今仍被广泛借鉴的模板<cite>[5]</cite>:

- **样本回放**:维护一个小型样本缓冲,训练时联合重放旧样本;
- **知识蒸馏**:让新模型在新数据上「对齐旧模型的输出」,以保留旧类别的决策;
- **最近均值分类器(NCM)**:放弃学得的全连接头,改用「每类特征均值 + 最近邻」做分类,规避新类别更新对旧类别权重的直接冲击。

三个组件的协同流程如下:回放提供旧样本,蒸馏保留旧类别「意见」,NCM 把分类从「可学习的权重重心」换成「稳定的类原型重心」:

```mermaid
graph LR
    NEW[新类别数据<br/>任务 t] --> TRAIN[联合训练<br/>mini-batch]
    MEM[记忆缓冲<br/>旧样本子集] --> TRAIN
    OLD[旧模型 f_t-1] --> DIST[知识蒸馏<br/>对齐旧类别输出]
    TRAIN --> DIST
    TRAIN --> NCM[最近均值分类器<br/>按类特征均值分类]
    NCM --> PRED[测试时<br/>对累积类别池判别]

    class NEW proc
    class MEM mid
    class OLD mid
    class TRAIN core
    class DIST proc
    class NCM out
    class PRED out
```

其中蒸馏项的具体形式,是对每个旧类别 $k$ 让新模型复现旧模型的输出分布:

$$
L_{\text{distill}} = -\sum_{k=1}^{s-1} q_k^{(t-1)}(x)\, \log p_k^{(t)}(x)
$$

这里 $q_k^{(t-1)}$ 是旧模型在旧类别 $k$ 上的输出概率(softmax 软化后的「软标签」),$p_k^{(t)}$ 是新模型的对应输出——把旧模型的「知识」通过概率分布的形式蒸馏进新模型。

iCaRL 的意义在于**确立了一个重要认知:光靠蒸馏不够,回放是刚需**。它的蒸馏部分在保护表示,回放部分在修正表示漂移,NCM 在缓解分类器偏差——三件套各自对着一个病根。

### BiC:第一次直面分类器偏差

iCaRL 用 NCM 绕开了偏差,但代价是放弃了可学习的分类器。**BiC(Bias Correction)** 换了一个思路:保留全连接头,但在任务切换后用一个小的「偏差校正层」对 logits 做线性校正,显式地把新类偏置掰回来<cite>[6]</cite>。它是「偏差校正」这一支的起点,后续 WA(权值对齐)等方法是它的进化版<cite>[9]</cite>。

## 第二波:表示层面的对抗(2019–2020)

第一波方法发现:即使有回放和蒸馏,旧类在新表示下的「位置」仍在漂移。第二波开始直接向表示漂移开刀。

### LUCIR:余弦归一化 + 大间隔

**LUCIR**(Learning a Unified Classifier Incrementally via Rebalancing)观察到,在 Class-IL 中,新类由于「出场晚」,其分类权重向量的范数天然偏大,挤压旧类。它的两个核心对策<cite>[7]</cite>:

- **余弦归一化**:把分类器从「内积」改为「余弦相似度」,消除权重范数差异带来的偏差;
- **margin 约束**:训练时强制样本与真实类的余弦相似度大于与其他类的 margin,防止旧类特征被新类「拽走」。

具体地,余弦归一化把分类 logits 从内积改写为「权重与特征夹角的余弦 × 缩放」:

$$
\text{logit}_j = s \cdot \cos\theta_j = s \cdot \frac{W_j^T f}{\|W_j\| \|f\|}
$$

其中 $s$ 是固定缩放因子。这样每个类别的 logit 只取决于方向、与权重范数 $\|W_j\|$ 无关——新类「出场晚、权重范数大」的偏差被结构性消除。而 margin 约束则进一步拉开真实类与其他类的余弦距离:

$$
L_{\text{margin}} = -\log \frac{e^{\,s(\cos\theta_y - m)}}{e^{\,s(\cos\theta_y - m)} + \sum_{j \neq y} e^{\,s\cos\theta_j}}
$$

margin $m$ 要求样本与其真实类别的余弦相似度显著高于其他类,避免旧类特征在新类训练时被「拽向」新类。

LUCIR 把问题从「决策层偏差」推进到「表示与度量层」,是 Class-IL 一个重要的视角跃迁。

### PODNet:特征空间的分段蒸馏

**PODNet**(Pooled Outputs Distillation)更进一步,提出「池化输出蒸馏」(POD)——把旧模型在中间层特征上的**空间池化统计量**作为蒸馏目标,而不是只蒸馏最后一层的 logits<cite>[8]</cite>。这样做的好处是:它约束的是**整个特征空间**的逐位置分布,能更精细地防止表示漂移,尤其适合「小任务」序列(每批只来很少类别)的设定。

## 第三波:回放的强化与反思(2020–2022)

### DER:回放「经验」而非「样本」

**DER(Dark Experience Replay)** 提出一个微妙的改进:与其重放旧样本的原始输入,不如重放旧模型在旧样本上的**logits(暗经验)**<cite>[10]</cite>。训练新任务时,让新模型在旧样本上尽量复现旧模型的输出分布——这相当于把「旧模型的软知识」当作重放对象,与 iCaRL 的「回放 + 蒸馏」相比更统一、更灵活,在通用持续学习基准上表现突出。它的损失是「新任务 CE + 暗经验均方」的合成:

$$
L = L_{CE}(x, y) + \alpha\, \left\| h_{\theta}(x_m) - h_{\theta_{t-1}}(x_m) \right\|_2^2
$$

其中 $x_m$ 是记忆缓冲中的旧样本,$h_\theta$ 与 $h_{\theta_{t-1}}$ 分别是当前与旧模型的 logits,$\alpha$ 权衡新旧。相比 iCaRL 的交叉熵蒸馏,DER 用**逐 logit 的 L2 对齐**,对旧模型输出的「形状」约束更直接、实现也更简单。

### 反思:GDumb 的当头棒喝

我们在全景文中已经介绍过 **GDumb**<cite>[11]</cite>:只存样本、每次从头训练,就能击败大多数复杂方法。这个「简单上界」在 Class-IL 领域尤其刺眼——因为 Class-IL 的难点集中在「在线表示更新」上,而 GDumb 干脆不做在线更新。它提示我们:**任何声称解决 Class-IL 的复杂方法,都必须先回答「比起存样本重训,你多做了什么」**。这一反思在预训练时代有了新的回响(见第四波)。

## 第四波:预训练时代的范式转换(2022– )

2022 年前后,大规模预训练模型(如 CLIP、ImageNet 预训练 ResNet)进入持续学习领域,带来了一个**近乎质变**的转变:既然预训练特征已经足够强、足够通用,为什么还要在增量过程中继续更新骨干网络?

### 冻结骨干 + 原型:最朴素的解法反成最强基线

**Linear Probe / SimpleCIL 范式**的回答是:冻结预训练骨干,只学最后一层(或直接用类原型),增量时**骨干完全不动**——表示漂移问题「被绕开而非解决」:

```mermaid
graph LR
    X[输入图像] --> ENC[冻结预训练骨干<br/>参数始终不变]
    ENC --> F[冻结特征<br/>表示不再漂移]
    F --> PROT[类原型<br/>每类特征均值/分布]
    F --> CLASS[线性分类头<br/>增量只更新它]
    PROT --> PRED[累积类别池判别]
    CLASS --> PRED

    class X proc
    class ENC mid
    class F core
    class PROT out
    class CLASS out
    class PRED out
```

这一类方法如 **FeCAM**(用协方差建模类分布做最近类均值分类)<cite>[16]</cite>,以及 **RanPAC**(在冻结特征后接一个固定随机投影层增强线性可分性)<cite>[14]</cite>,在 Split-ImageNet 等基准上把准确率拉到了此前难以想象的高度。这本质上是 GDumb 精神的预训练版本:**强特征 + 简单分类,胜过复杂算法**。

### 提示学习:让骨干「冻结但适配」

**L2P(Learning to Prompt)** 和 **DualPrompt** 给出另一条路:不微调骨干,而是**学习一组可微的提示(prompt)向量**插入 Transformer 骨干,用提示来「按任务切换」骨干的响应<cite>[12]</cite><cite>[13]</cite>:

```mermaid
graph LR
    X[输入 token 序列] --> POOL[提示池 Prompt Pool<br/>一组可学习提示向量]
    POOL --> SEL[按输入检索<br/>选择匹配提示]
    SEL --> T[Transformer 骨干<br/>冻结]
    T --> OUT[输出<br/>提示控制任务适配]
    NEW[新任务到来<br/>新增提示向量] -.增量.-> POOL

    class X proc
    class POOL mid
    class SEL proc
    class T core
    class OUT out
    class NEW hl
```

提示本身是可增量学习的参数,而骨干保持冻结——既保留了预训练特征,又获得了跨任务的适配能力,且天然具有「参数隔离」的防遗忘属性:旧任务的提示不动,新任务只是新增提示。

### SLCA:在预训练特征上重新思考「慢学」

**SLCA(Slow Learner with Classifier Alignment)** 提出另一个洞察:预训练特征下,表示几乎不漂移,真正的瓶颈回到**分类器偏差**;因此它用「慢学习率学习表示 + 显式对齐分类器」的组合,在预训练设定下刷新了多个基准<cite>[15]</cite>。它印证了一个趋势:**预训练时代,Class-IL 的难点从「表示漂移」重新聚焦回「分类器偏差」**——难点并没有消失,只是换了位置。

这一波方法的系统性梳理,可参见 Zhou 等人的两份综述:Class-Incremental Learning 全景<cite>[17]</cite> 与预训练模型持续学习专题<cite>[18]</cite>。

## 评估陷阱与公平比较

方法演进越热闹,评估越需要警惕——因为 Class-IL 的结论高度依赖「怎么测」。

### 代表性方法的实证表现

先给一张「量级地图」:下表是若干代表方法在 Split-CIFAR-100(类增量、10 个任务)的最终平均准确率,以及预训练时代方法在 Split-ImageNet-100 上的表现——数字为近似值,用于建立直觉,精确对比请以 Zhou 等人统一基准为准<cite>[17]</cite>:

| 方法 | 范式 | Split-CIFAR-100 ACC | 量级认知 |
|---|---|---|---|
| iCaRL | 回放 + 蒸馏 | ~49 | 早期奠基,三件套但表示仍漂移 |
| BiC | 偏差校正 | ~59 | 决策层校正补上短板 |
| LUCIR | 余弦归一化 | ~61 | 表示层对抗带来明显提升 |
| PODNet | 特征空间蒸馏 | ~61 | 与小任务序列更适配 |
| GDumb | 回放上界 | ~53 | 「重训」上界在受限内存下并不高 |
| DER | 暗经验回放 | ~68 | 经验重放把回放收益进一步放大 |

而进入预训练时代(Split-ImageNet-100, ImageNet 预训练骨干),量级整体上移了一个台阶:

| 方法 | 设定 | ACC(近似) | 量级认知 |
|---|---|---|---|
| SimpleCIL | 冻结骨干 + 类原型 | ~76 | 朴素解法即超以往所有方法 |
| FeCAM | 冻结 + 分布型原型 | ~88 | 用协方差建模类分布 |
| RanPAC | 冻结 + 随机投影 | ~91 | 随机投影增强线性可分 |
| SLCA | 慢学 + 分类器对齐 | ~91 | 聚焦分类器偏差 |

这一组数字直观地印证了第四波的判断:**预训练特征让「简单方法」的起点超过了「复杂算法」的天花板**——这既是好消息(Class-IL 变得可用了),也意味着未来对比必须把「预训练特征」这一变量纳入考量。

**第一个陷阱:协议错配**。Task-IL 与 Class-IL 的准确率差距极大,把 Task-IL 的成绩当作 Class-IL 的声明,会系统性高估方法。统一在 Class-IL 协议下报告,是领域的基本共识<cite>[3]</cite><cite>[4]</cite>。

**第二个陷阱:内存不对齐**。Zhou 等人指出,许多方法在对比时没有对齐**存储预算**——回放方法的样本缓冲、蒸馏方法的额外模型、提示方法的提示参数,都被隐式地「免费」使用了<cite>[17]</cite>。公平比较必须把存储开销计入预算,否则复杂方法对简单基线的优势会大幅缩水。

**第三个陷阱:基准饱和**。预训练时代,冻结骨干方法在 Split-ImageNet 等基准上已经逼近饱和,单纯报告「更高 ACC」意义递减。更值得关注的是**跨设定的稳健性**(域漂移 + 类增量叠加)、**小任务/大序列**的可扩展性,以及与医疗等真实场景的差距——这正是我们下一节想强调的方向。

## 总结

类增量学习之所以是持续学习的「试金石」,是因为它把遗忘的全部症状——表示漂移、分类器偏差、旧类判别退化——**同时暴露在同一个分类决策上**。回顾五波演进:

| 波次 | 代表方法 | 核心思想 | 直面的问题 |
|---|---|---|---|
| 第一波(2017–19) | iCaRL、BiC | 回放 + 蒸馏 + NCM / 偏差校正 | 旧类判别、分类器偏差 |
| 第二波(2019–20) | LUCIR、PODNet | 余弦归一化、特征空间蒸馏 | 表示漂移 |
| 第三波(2020–22) | DER、GDumb | 暗经验回放 / 简单上界反思 | 回放效率、方法有效性 |
| 第四波(2022– ) | L2P、FeCAM、RanPAC、SLCA | 冻结骨干 + 原型/提示/生成特征 | 绕开表示漂移,聚焦分类器偏差 |

贯穿始终的一条主线是:**表示漂移与分类器偏差的博弈**——早期方法用回放和蒸馏「守住表示」,中期方法用归一化和特征蒸馏「矫正表示」,预训练时代则干脆「冻结表示」,把战场拉回决策层。而无论哪一波,GDumb 式的反思都提醒我们:**在投奔复杂算法之前,先问一句「简单上界怎么说」**。

如果你正在评估方法或设计实验,请记住三件事:在 Class-IL 协议下报告;对齐内存预算;以及——如果可能,先跑一遍「冻结骨干 + 原型」的基线。

## 参考文献

1. *Catastrophic Interference in Connectionist Networks.* McCloskey M, Cohen N J. The Psychology of Learning and Motivation, 1989.  
   <https://doi.org/10.1016/S0079-7421(08)60536-8>
2. *Overcoming Catastrophic Forgetting in Neural Networks.* Kirkpatrick J, et al. PNAS, 2017.  
   <https://doi.org/10.1073/pnas.1611835114>
3. *Three Scenarios for Continual Learning.* van de Ven G M, Tolias A S. arXiv:1904.07734, 2019.  
   <https://arxiv.org/abs/1904.07734>
4. *Class-Incremental Learning: Survey and Performance Evaluation.* Masana M, et al. IEEE TPAMI, 45(5), 2022.  
   <https://arxiv.org/abs/2010.15277>
5. *iCaRL: Incremental Classifier and Representation Learning.* Rebuffi S A, et al. CVPR, 2017.  
   <https://arxiv.org/abs/1611.07725>
6. *Large Scale Incremental Learning.* Wu Y, et al. CVPR, 2019.  
   <https://arxiv.org/abs/1905.13262>
7. *Learning a Unified Classifier Incrementally via Rebalancing.* Hou S, et al. CVPR, 2019.  
   <https://arxiv.org/abs/1903.02990>
8. *PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning.* Douillard A, et al. ECCV, 2020.  
   <https://arxiv.org/abs/2004.13513>
9. *Maintaining Discrimination and Fairness in Class Incremental Learning.* Zhao B, et al. CVPR, 2020.  
   <https://arxiv.org/abs/1911.07053>
10. *Dark Experience for General Continual Learning.* Buzzega P, et al. NeurIPS, 2020.  
    <https://arxiv.org/abs/2004.07211>
11. *GDumb: A Simple Approach that Questions Our Progress in Continual Learning.* Prabhu A, et al. ECCV, 2020.  
    <https://arxiv.org/abs/1910.07113>
12. *Learning to Prompt for Continual Learning.* Wang Z, et al. CVPR, 2022.  
    <https://arxiv.org/abs/2112.08654>
13. *DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning.* Wang Z, et al. ECCV, 2022.  
    <https://arxiv.org/abs/2204.04799>
14. *RanPAC: Random Projections and Pre-trained Models for Continual Learning.* McDonnell M D, et al. NeurIPS, 2023.  
    <https://arxiv.org/abs/2307.03051>
15. *SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model.* Zhang G, et al. ICCV, 2023.  
    <https://arxiv.org/abs/2303.05118>
16. *FeCAM: Exploiting the Heterogeneity of Class Distributions in Exemplar-Free Continual Learning.* Zhang Y, et al. NeurIPS, 2023.  
    <https://arxiv.org/abs/2309.14062>
17. *Class-Incremental Learning: A Survey.* Zhou D-W, et al. IEEE TPAMI, 2024.  
    <https://arxiv.org/abs/2302.03653>
18. *Continual Learning with Pre-Trained Models: A Survey.* Zhou D-W, et al. IJCAI Survey Track, 2024.  
    <https://arxiv.org/abs/2401.16386>

{: .references }

{% include series-nav.html series="continual-learning" position="bottom" %}
