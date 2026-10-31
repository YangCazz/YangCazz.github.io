# 医学方法配图第二批（2026-10-30）

范围仅两篇现有文章：拓扑分割、Mamba综述中的U-Mamba段。原因：正文已有流程Mermaid，但缺少能说明骨架/掩码关系、混合模块与跳连的具体方法图。不是对所有无图文章机械填图。

## 图与证据

- `cldice-bidirectional-coverage-v1.png`：SF-01风格；人工Y形掩码/断裂示意，双向覆盖比例与调和均值；不显示虚构分数。依据 [clDice v7](https://arxiv.org/abs/2003.07311v7)，修订日2022-07-15；§2式(1)–(2)，PDF第3页；§3 Corollary 1.1，第4页；Algorithms 1–2，第5页。
- `u-mamba-hybrid-architecture-v1.png`：残差卷积→展平归一化→Mamba→恢复形状，以及Enc概念编码器/解码器/跳连。依据 [U-Mamba v1](https://arxiv.org/abs/2401.04722v1)，2024-01-09；[HTML §2.2 / Figure 1](https://arxiv.org/html/2401.04722v1)。层数、图形样本均为示意。

核查日2026-10-30。两图均原创生成、不是论文原图或患者影像；不复制受限图形。U-Mamba原文页面标明CC BY-NC-ND，本批仅依据事实独立组织概念图，不以其许可为由复制或改绘原图。生成使用原始批准iCaRL作为风格参考，该PNG hash不变；全部prompt/局部修正/hash在 `research-medical-20261030-batch2.json`。

## 配图直接关联的内容修复

1. **clDice定义**：旧文写成 `2 |Vpred ∩ Sgt| / (|Vpred| + |Sgt|)`，这是单向掩码/骨架Dice，不是论文clDice。改为两个骨架-对侧掩码覆盖比例 `Tprec`、`Tsens` 的调和均值；Mermaid同步更正。
2. **理论边界**：去除无前提的“使用clDice保证任意二值分割同伦等价”表述。写明原文骨架假设以及前景/背景都达到1的条件；不把训练损失或高分当作临床保证。
3. **示例骨架代码**：旧`soft_skel`用`torch.where(x == max_pool, ...)`筛选、还带未使用的Top-K，不符合论文迭代软骨架算法。没有从记忆重写算法；改为导入作者官方`SoftSkeletonize`的教学封装，正文明确设置官方`cldice_loss/pytorch`模块搜索路径。官方代码 [soft_skeleton.py](https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py) / [MIT许可证](https://github.com/jocpae/clDice/blob/master/LICENSE)，作者Johannes C. Paetzold与Suprosanna Shit；没有在仓库内复制其实现。原文Algorithm1–2与官方代码已核对。
4. **U-Mamba结构**：旧表“卷积块替换为Mamba”改为保留两段残差卷积后接Mamba，并区分Enc/Bot。无需引入性能承诺。

## 测试与局限

- `python tests/test_research_batch2.py -v`：6项通过（双向公式、三个人工集合例子、Python AST/官方骨架调用关系、U-Mamba表述）；1项PyTorch数值/梯度测试明确跳过，因当前环境没有PyTorch及作者模块。
- 集合例子检验的是公式关系，不是骨架提取器、真实图像或临床结果。
- `node --test _scripts/test-visual-assets.mjs` 与资产清单校验；最终Jekyll、数学公式/Mermaid和移动端截图交整合批次检查。
- 未声明运行过训练或验证过精度、速度。两篇其余性能表、SOTA断言和全部参考文献未在本批完成审计；特别是Mamba篇中Swin-UMamba实时速度等需要后续独立核查，不能从本图推断得到支持。
- 文章旧标题中的SOTA语气和更广泛引用问题列入后续内容审计，本次不把新图包装成全篇真实性背书。
