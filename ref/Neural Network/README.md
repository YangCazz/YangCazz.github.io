# 学习笔记-深度学习DeepLearning
# 前言
计算机视觉领域中，一共有以下几类经典应用派系(可以在[SOTA](https://paperswithcode.com/sota)网站查看各类别的模型架构)：
* 图像分类-[Image Classification](https://paperswithcode.com/task/image-classification)
* 目标检测-[Object Detection](https://paperswithcode.com/task/object-detection)
* 语义分割-[Semantic Segmentation](https://paperswithcode.com/task/semantic-segmentation)
* 图像生成-[Image Generation](https://paperswithcode.com/task/image-generation)

研究生入学一年以来，一直在追逐前人的工作，也跟着网络教程([吴恩达](https://www.bilibili.com/video/BV1FT4y1E74V?from=search&seid=2765695004787511604&spm_id_from=333.337.0.0&vd_source=7310cba54f2c72f9e81d1689195a6e63)、[李沐](https://space.bilibili.com/1567748478?from=search&seid=13202777208076360823&spm_id_from=333.337.0.0)、[李弘毅](https://www.bilibili.com/video/BV1Wv411h7kN?from=search&seid=15312611128680858730&vd_source=7310cba54f2c72f9e81d1689195a6e63))完成了对深度学习先零散后系统化的学习。随着自己对研究方向的探索及推进，这里做一些关于各传统模型的简单笔记。

# 1. 图像分类任务 Image Classification
* 定义：图像分类是计算机视觉的一项基本任务，也是自然界最直观的可解释的基本视觉。分类任务试图从整体上理解图像，将特定标签域中的标签分配给图像。通常，分类任务指的是单标签任务。
 
## 1.1 LeNet-1998-Yann LeCun
**论文**：[LeNet：Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) 
**DOI**：10.1109/5.726791. 
**简介**：CV开山作之一，Yann LeCun最早采用了基于卷积+梯度优化的神经网络用于支票手写数字的识别 

**主要贡献**：
* 构建了：**卷积-下采样(池化)-全连接**的卷积网络范式
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/LeNet.png)

* 推动计算机应用于视觉任务上来
* 在手写字识别上达到不错的结果

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib
* 数据集：CIFAR10
* 复现代码[GitHub]：[DeepLearning/model_classification/LeNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/LeNet)

## 1.2 AlexNet-2012-Hinton和他的学生Alex
**论文**：[AlexNet: ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
**DOI**：10.1145/3065386. 
**简介**：CV开山作之二，时隔多年后CNN在2012 ILSVRC ([ImageNet](https://image-net.org)大规模视觉识别挑战赛)冠军，延续了Yang LeCun的工作，展示了CNN在图像识别领域的优势，是CV领域承先启后的作品。
 
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/AlexNet.png)

**主要贡献**：
* **首次证明了学习到的特征可以超越手工设计的特征**，让更多的人开始注意这个和黑匣子一样的“深度学习”，**掀起深度学习的研究浪潮**
* 在大数据样本上做实验，取得更好的效果；但受限于硬件条件，提出**多GPU训练模式**
* 引入激活函数ReLU，让映射拟合增加非线性组件
* 引入DropOut随机失活操作，逐渐成为CNN领域的核心组件
* CNN开始向“深度”探索，LeNet为5层，AlexNet为8层
* 实现**端到端**的模型定义

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/Alexnet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/Alexnet)

## 1.3 VGG-2014-牛津大学Visial Geometry Group(VGG)
**论文**：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) 
**DOI**：arXiv:1409.1556 
**简介**：2014年的ImageNet中定位任务的第一名，分类任务的第二名。随着硬件技术的进步，研究人员们有能力构建足够深足够大的卷积神经网络来做分类任务。**事实证明，卷积神经网络在深度上的提升，带来分类效果的极大改善**。 

  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/VGG.png)

**主要贡献**：
* 第一次深入研究了**网络的深度**对模型效果的影响，分别对11层、13层、16层和19层的模型进行训练，后来人们常用的是**VGG16**和**VGG19**
* 将卷积神经网络模块化定义为不同的**Stage**，提出了可以通过重复使⽤简单的基础块来构建深度模型的思路
* 讨论了模型的**感受野问题**，两层3x3的卷积核在感受野上可以等价于一个5x5的卷积核，三层3x3的卷积核在感受野上可以等价于7x7的卷积核，并且拥有更小的计算量和更好的鲁棒性，这一结论深刻影响了后续研究者的探索。

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]： [DeepLearning/model_classification/VGG](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/VGG)

## 1.4 GoogLeNet-Google
GoogLeNet，命名中L为大写是为了致敬Yang LeCun提出的LeNet，是第一个超过100层的卷积神经网络。因为提出了Inception结构，所以也被叫做InceptionNet，历经多个版本的更迭。**有一点不好的是，GoogLeNet是比较难复现的**。和VGG在深度Depth上的探索相比，GoogLeNet更像是深度学习网络在广度上的探索。

**论文**：
[2014]**InceptionNet V1**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)  
[2015]**InceptionNet V2**:[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://proceedings.mlr.press/v37/ioffe15.html)    
[2015]**InceptionNet V3**:[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)   
[2017]**InceptionNet V4**:[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) 


### 1.4.1 InceptionNet V1-2014
**简介**：在和VGG提出的同年，谷歌团队提出了GoogLeNet一起参加了当年的ImageNet比赛，以其卓越的表现在当年的ImageNet中分类任务中取得第一名
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_v1.png)
  

**主要贡献**：
* 提出**Inception的卷积组结构**(也叫非对称卷积结构)，在一个卷积模块中同时采用不同大小的卷积，可以同时对图像多个尺度的特征进行学习，并在channel维度进行拼接，融合不同尺度的特征信息；将Inception结构与原有的卷积结构进行结构，构建出一个深层的多尺度特征提取模型
  
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_Inception_v1.png)

* **引入辅助输出层**，在模型训练时的几个Stage分别构建输出层，将这些输出层(2个辅助输出层和一个主输出层)一起作用于模型的损失计算和优化

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_shortcut.png)
* 使用**1x1卷积**进行数据降维(展平)，其效果等同于全连接层，**从信息学的角度上来说信息损失极小**，却极大地减小了的计算量和计算复杂度
* 丢弃全连接层，使用平均池化层，极大地减少模型的参数，其参数是VGG的1/20

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]： [DeepLearning/model_classification/GoogleNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/GoogleNet)

### 1.4.2 InceptionNet V2-2015
**简介**：谷歌团队认为-**网络训练过程中参数不断改变导致后续每一层输入的分布也发生变化，而学习的过程又要使每一层适应输入的分布，因此我们不得不降低学习率、小心地初始化**。这样的分布变化被称为**internal covariate shift**(**内部协变量位移问题**)。谷歌团队在V1的基础上以批正规化(BN)实现激活值的稳定分布，将层级结构的输出值映射到一个正态的域上来，实现结构的稳定可加性。

**主要贡献**：
* 构建小批量统计的归一化方法，加速了模型的收敛速度，称为**Batch Normalization**，这一应用以及后来pytorch框架对于sync Batch Norm的应用造就了分布式训练方式
* **BN成为深度学习的一个经典归一化结构**
* BN使得模型可以使用较大的学习率而**不用特别关心诸如梯度爆炸或消失等优化**；降低了模型效果**对初始权重的依赖**；可以**加速收敛**，一定程度上可以不使用Dropout这种降低收敛速度的方法，但却起到了正则化作用**提高了模型泛化性**；即使不使用ReLU也能**缓解激活函数饱和**问题；能够学习到从当前层到下一层的分布缩放( scaling (方差)，shift (期望))系数。

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[同上](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/GoogleNet)，增加**BN模块**

### 1.4.3 InceptionNet V3-2015(未复现)
**简介**：作者认为随意增大Inception的复杂度，后果就是Inception的错误率很容易飙升，还会成倍的增加计算量，所以必须按照一套合理的规则来优化Inception结构。V3对Inception结构和网络整体的设计进行了新的探索，并总结出**通用设计准则**，奠定了Inception系列最常用的模型。

**设计准则**：
*  **规则1**：**要防止出现特征瓶颈(representational bottleneck)**。所谓特征瓶颈是指**中间某层对特征在空间维度进行较大比例的压缩**（比如使用pooling时），导致很多特征丢失。虽然Pooling是CNN结构中必须的功能，但我们可以通过一些优化方法来减少Pooling造成的损失。
* **规则2**：**特征的数目越多收敛的越快**。相互独立的特征越多，输入的信息就被分解的越彻底，分解的子特征间相关性低，**子特征内部相关性高，把相关性强的聚集在了一起会更容易收敛**。
规则2和规则1可以组合在一起理解，特征越多能加快收敛速度，但是无法弥补Pooling造成的特征损失，Pooling造成的特征瓶颈要靠其他方法来解决。
* **规则3**：**可以压缩特征维度数，来减少计算量**。inception-v1中提出的用1x1卷积先降维再作特征提取就是利用这点。不同维度的信息有相关性，**降维可以理解成一种无损或低损压缩，即使维度降低了，仍然可以利用相关性恢复出原有的信息**。
* **规则4**：**整个网络结构的深度和宽度（特征维度数）要做到平衡**。只有等比例的增大深度和维度才能最大限度的提升网络的性能。

**扩展概念**：
* 神经网络的**表征瓶颈**：任何神经网络往往**容易建模极简单交互效应**和**极复杂的交互效应**，但是**不容易建模中等复杂度的交互效应**。这一数学证明来自2021年文章：[Discovering and Explaining the Representation Bottleneck of DNNs](https://arxiv.org/abs/2111.06236)

**主要贡献**：
* 提出几个网络设计准则；引入卷积分解提高效率；引入高效的feature map降维
* 相关炼丹经验成为后续深度学习研究推进的重要参考
* 修改了Inception模块，将大的卷积拆解为小的卷积。替换5x5为多个3x3，1x7和7x1的堆叠，将3x3替换为1x3和3x1，下图中的小图展示了Inception结构的变化。
  
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_Inception_v3.png)

**模型复现**：未复现

### 1.4.4 InceptionNet V4-2016(未复现)
**简介**：V4的提出是在2015何凯明提出残差结构之后，V4将残差模块融入Inception中，研究残差连接对模型训练的影响。残差模块的产生来自于对深度模型退化的思考，而GoogLeNet实际上是一个非常深的神经网络，所以**应该考虑使用残差模块来改进Inception结构中的Concatenation结构**。论文给出经验性的结论-使用残差连接可以很显著地加速模型的收敛，但是同样得出结论表明，一些新构建的Inception结构同样要比之前的结构要好。

**主要贡献**：
* 在InceptionV3的基础之上改进，提出InceptionV4模块；借鉴ResNet提出Inception-ResNet，这两者在ImageNet上的表现很相似。
* 进一步对网络结构做划分，分解为Stem层(即数据预前处理层，图1)、Stage层(即模型层)、Reduction层(即特征放缩放缩层，图2)和后处理层(包括Pooling池化、DropOut随机失活和SoftMax聚合)

  ![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_Inception_v4.png)

  ![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_Inception_v4_Reduction.png)

* V3的Inception结构中的Pooling模块也被改成了AvgPooling结构，3x3的卷积结构也被拆解为1xN和Nx1的卷积的组合，以下是InceptionNet V4的总览

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_v4.png)

* Inception-ResNet模块，从某种程度上来说，残差结构可以等价于Inception中的1x1卷积的分支

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/GoogLeNet_Inception_v4_Reduction.png)
**模型复现**：未复现


## 1.5 ResNet-2015-MicroSoft Asian


**论文**：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
**DOI**：arXiv:1512.03385  

**简介**：2015年的ImageNet中的目标检测任务第一名，分类任务第一名；2015年CoCo数据集目标检测任务的第一名，图像分割任务的第一名。自2012年AlexNet的诞生，到VGG和GoogLeNet对深度学习网络Deepth的探索，逐渐形成一个观念“**网络越深效果越好**”。通过实验，ResNet随着网络层不断的加深，模型的准确率先是不断的提高，达到最大值（准确率饱和），然后随着网络深度的继续增加，模型准确率毫无征兆的出现大幅度的降低。这说明，网络并不是越深越好，过深的网络会导致准确率的“**退化”(Degradation)**。2015年，何凯明团队提出的**残差结构，就是应对模型的退化而来的**，这一优秀的想法造就了ResNet，这一优秀的网络直到现在还能够得到各领域的广泛应用，可见其强大和有效性。

**主要贡献**：
* 提出Residual残差模块，连接不同层之间的输出，防止模型退化。左边的模型是ResNet[18/34]的残差模块，右边的是ResNet[50/101/152]的残差模块。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ResNet_Residual.png)
* 超深的网络结构，突破200层，几乎是传统网络结构的究极体
![[ResNet.png]]
![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ResNet.png)
* 采用BatchNormalization加速，丢弃DropOut结构
* 残差神经网络ResNet-34的计算量是VGG-19的18%左右，但是准确率却远高于后者

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/ResNet_ResNeXt](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/ResNet_ResNeXt)

## 1.6 ResNeXt-2016-MicroSoft Asian
**论文**：[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)  
**DOI**：arXiv:1611.05431

**简介**：是ResNet的升级，融入了GoogLeNet的Inception结构，改造原有的残差模块。同样是由谷歌何凯明一行人在原有研究基础之上的探索，借助Split-Transform-Merge思想发展出了新的残差结构。

**主要贡献**：
* 将Inception结构概念进行抽取，形成Group Conv的概念-**Split-Transform-Merge**实际上这样的思想符合神经网络的基础定义   $\mathcal{F}=\sum_{i=1}^C \mathcal{T}_i(\mathbf{x})$

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ResNeXt.png)
在ResNeXt中加上残差部分，就构成了  $\mathcal{Y}=X + \sum_{i=1}^C \mathcal{T}_i(\mathbf{x})$
![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ResNeXt_Residual.png)
* GoogLeNet中吸取了ResNet的残差结构写出了InceptionNet V4，而ResNet吸收Inception写成了ResNeXt，两则在思路上很是相似，只不过ResNeXt中的分支拓扑结构是一样的，而InceptionNet V4是手工设计的。

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/ResNet_ResNeXt](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/ResNet_ResNeXt)

## 1.7 MobileNet-Google
**论文**：
[2017]**MobileNet V1**:[ MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  
[2018]**MobileNet V2**:[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
[2019]**MobileNet V3**:[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)  

### 1.7.1  MobileNet V1-2017
**简介**：经历了GoogLeNet多年多个版本递进式的研究，**深度学习各模型之间的竞争大多集中在大规模计算**，**集中在对硬件和算力的竞争上**，2017年谷歌团队转而**将目光转到深度学习在小规模计算集群的部署上来**。也就是说，让一些算力较低的设备-如手机和小型电脑等也能够完成深度学习的任务。**MobileNet，正如其名意为可以在移动设备上部署的深度学习网络**，Google团队将原始的卷积替换成了深度可分离卷积(Deepwise separable Conv，DW卷积)，V1的基本构成就是在VGG的架构上将卷积改造成DW卷积，其理论卷积计算量是VGG的1/8左右。巧合的是，同一时期谷歌的另一个团队也提出了基于相同架构的Xception网络。

**主要贡献**：
* 提出深度可分离卷积(**Deepwise separable Conv**，DW卷积+PW卷积)，中心思想其实还是来源于Inception结构的**Split-Transform-Merge**架构。标准卷积其卷积核是用在所有的输入通道上（input channels），而depthwise convolution针对每个输入通道采用不同的卷积核，就是说一个卷积核对应一个输入通道(和前面画的图相类似)

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v1_dw_pw.png)
* 在DW卷积的定义下，提出PointwiseConv(PW卷积)，也就是卷积核大小为1x1的卷积，一般DW和PW是配合使用的
* DW卷积的理论计算量是传统卷积的1/9，符号定义： $M$ 为图像channel数量， $D_F$ 为图像的原始尺寸， $N$ 为使用的卷积核个数， $D_K$ 为卷积核的大小。从参数角度上来看，这些参数都可以被分解为点态的参数。在 $D_K=3$ 时，这个比值极限为 $\frac{1}{9}$。
$S_1=M*D_F^2*N*D_K^2$   $S_2=M*D_F^2*D_K^2+M*N*D_F^2$ 
$$
	\frac{S_1}{S_2}=\frac{M*D_F^2*D_K^2+M*N*D_F^2}{M*D_F^2*N*D_K^2}=\frac{1}{N}+\frac{1}{D_K^2}
$$
* 网络中的DW卷积组件为

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v1.png)


**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/MobileNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/MobileNet)

### 1.7.2  MobileNet V2-2018
**简介**：在V1的基础之上，探讨如何优化整个模型，从数据流的角度**讨论了模型不同层对数据的处理和损失效应**，提出Mainfold of Interest(兴趣流)的概念，并讨论了ReLU对特征的损失效果，提出ReLU6函数；单独讨论了残差结构，借鉴ResNet和DenseNet并结合DW卷积的特点提出了Inverted Residual倒残差结构。

**主要贡献**： 
* 讨论了数据流的损失效应。在使用DW卷积的情况下，被卷积压缩后的Feature Maps经过压缩，经过非线性的ReLU的激活会损失掉特征空间中的负值特征，这使得数据中的感兴趣信息损失太多。引入**ReLU6非线性激活函数**，在一定程度上保留负值空间的特征。 $y = ReLU6(x)=min(max(x,0),6)$  相比于ReLU，在 $y=6$ 时做了一个数据截断。ReLU激活函数对于低维的信息可能会造成比较大的瞬损失，而对于高维的特征信息造成的损失很小，导致数据两头偏差较大。
  ![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v2_ReLU6.png)
* 讨论了残差结构，将原有的两头大中间小的结构改为两头小中间大的**Inverted Residual逆残差结构**，传统卷积核逆残差从模块上来说是一样的(Conv1x1+**DWConv**3x3+Conv1x1)，只是在维度方向上是不一样的，传统卷积的路线为(**Input->降维->处理->升维->Output**)，逆残差结构的路线为(**Input->升维->处理->降维->Output**)，具体可见如下示意图。

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v2_Inverted_Residual.png)
* 逆残差结构的引入，**进一步降低了参数计算量**，其基本效果依旧保持较好的水平
  ![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v2_Bottleneck.png)
**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/MobileNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/MobileNet)

### 1.7.3 MobileNet V3-2019
**简介**：在V2的基础之上引入了通道注意力机制，采用暴力美学的NAS(自动网络架构搜索)来搜索最优参数，并且重新设计了耗时多的层，改用了**H-Swish激活函数**。此外，将通道注意力机制(Squeeze-and-excitation，SE模块)引入进来。

**扩展概念**：
* **NAS-网络结构搜索**。之前的VGG、ResNet、GoogLeNet、MobileNet V1 V2都是由手动设计得到的，其中网络Layer层数，卷积核Kernel的大小，步长等参数都需要手动设置，采用NAS方法则是计算机通过优化算法寻优计算得到的，也就是采用了进化算法等优化理论的思想。但是这样的方法需要极大的算力支持，简直暴力美学。

**主要贡献**：
* Hard-Swish激活函数来取代原有的激活函数，来自于谷歌大脑2017的论文：[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)中的swish函数 $f(x)=x \cdot \operatorname{sigmoid}(\beta x)$ 。由于**sigmoid函数计算复杂**，所以V3改用其近似函数来逼近swish函数，作者认为使用ReLU6作为这个近似函数理由为：(1)在几乎所有的软件和硬件框架上都可以使用ReLU6的优化实现，(2)ReLU6能在特定模式下消除由于近似sigmoid的不同实现而带来的潜在的**数值精度损失**。
  
$$
Hard—Swish(x)=x\frac{ReLU6(x+3)}{6}=\begin{cases}0, & \text { if } x \leq-3 \\ x, & \text { if } x \geq 3 \\ \frac{x(x+3)}{6}, & \text { otherwise }\end{cases}
$$

* 提出通道注意力机制(Squeeze-and-excitation，SE模块)在网络计算中的利用

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/MobileNet_v3_SE.png)
* 重新设计原有耗时过多的层，替换了V2中NAS搜索得到的层

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/MobileNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/MobileNet)

## 1.8 ShuffleNet-旷视科技
**论文**：
[2018]ShuffleNet V1:[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083v1)  
[2018]ShuffleNet V2:[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)  

### 1.8.1 ShuffleNet V1-2017
**简介**：ShuffleNet是AI四小龙之一旷视科技的代表之作，算是一个里程碑式的成果，第一作者张祥雨也是ResNet的作者之一。ShuffleNet和谷歌的MobileNet一样，都是轻量级模型的代表作，也正如其名Shuffle模型融入了随机洗牌的机制，其目的在于**解决组卷积中的组内关联性低的问题**。同时，作者还讨论了模型的评价指标，常用的FLOPs是一种间接的评价指标并不能我们完全在意的内容，需要用更直接的指标来进行评价如每秒训练速度(Batches/sec)和每秒推理速度(Images/sec)。

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ShuffleNet_v1.png)
**主要贡献**：
* 讨论了应该如何**设计更合适的评价指标**，采用最直观的(能够感受得到的)评价指标来
* 提出**组卷积GroupConv**的概念，和DW卷积几乎是一致的。在DW卷积中，图像的每一个Channel都有对应的卷积核来进行计算，而**GroupConv则是多个Channels作为一组**，每一组使用一个卷积核进行计算。MobileNet中的1x1卷积使得模型的计算量增大很多，于是ShuffleNet作者同样提出了对应的1x1组卷积。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ShuffleNet_v1_GroupConv.png)

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ShuffleNet_v1_Shuffle.png)
* 结合ResNeXt和MobileNet的旁支结构设计**ShuffleNet单元**

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ShuffleNet_v1_ShuffleBlock.png)
* 以上的卷积模组不禁告诉研究者，**深度分离卷积几乎是构造轻量高效模型的必用结构**

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/ShuffleNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/ShuffleNet)

### 1.8.1 ShuffleNet V2-2018
**简介**：作者团队再次思考了衡量模型的具象化指标，作为衡量计算复杂度的指标，FLOPs实际并不等同于速度。LOPs相似的网络，其速度却有较大的差别，只用FLOPs作为衡量计算复杂度的指标是不够的，还要考虑内存访问消耗(**内存访问成本**，MAC)以及GPU并行。作者籍此提出设计轻量级网络的几个准则

**设计轻量级网络的指南**：
* **相同维度的通道数将最小化内存访问成本**(MAC)，即当input channles = output channels时，模型每秒处理的照片数量越多(Images/sec)
* **过多的分组卷积会加大内存访问成本**，导致图像处理速度大幅度下降
* **碎片化的网络会降低并行度**，采用多路结构提高网络精度，但多路结构会造成网络的碎片化，使得网络速度变慢
* **元素级操作不能忽视**，对于ReLU、TensorAdd、BiasAdd等元素级操作，它们的FLOPs较少，但MAC较大。经过作者实验证明，将残差网络的残差单元中的ReLU和短接移除，速度会有20%的提升

**设计准则**：
-   **使用“平衡”卷积层，即输入与输出通道相同**
-   **谨慎使用分组卷积并注意分组数**
-   **减少碎片化的操作**
-   **减少元素级的操作**

**主要贡献**：
* 在设计准则的指导下进行模型设计，改造了原来的模块。将ADD换成Concat，将Gconv1x1改回了Conv1x1，所以真的和炼丹一样，把原来改的模块全搞回去了。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/ShuffleNet_v2_ShuffleBlock.png)
**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/ShuffleNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/ShuffleNet)


## 1.9 EfficientNet-Google
**论文**：
[2019]**EfficientNet V1**:[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)  
[2021]**EfficientNet V2**:[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)  

### 1.9.1 EfficientNet V1-2019
**简介**：在之前的一些手工设计网络中(AlexNet，VGG，ResNet等等)经常有人问，为什么输入图像分辨率要固定为224，为什么卷积的个数要设置为这个值，为什么网络的深度设为这么深？这些问题你要问设计作者的话，估计回复就四个字——工程经验。而这篇论文主要是用NAS（Neural Architecture Search）技术来搜索网络的**图像输入分辨率r**，**网络的深度depth**以及通**道的宽度width**三个参数的合理化配置。作者也籍此讨论了各个指标对模型的影响，实际上**参数数量越少并不意味着推理的速度越快**，也有可能是以空间换时间的以增大系统IO的成本来提高模型的准确度。同此可以发现，直接用暴力求优法真的能找到相对较好的模型架构，说白了**写文章就是说故事**。

**工程经验**：
* 增加网络的深度Depth能够得到**更加丰富、复杂的特征**；可以使得模型具有**更好的迁移性和鲁棒性，能够更加好地应用到其它任务中**；但是**过深的网络会面临梯度消失、训练困难的问题**
* 增加**网络的宽度Width能够获得粒度更高的特征**；更多的信息量**使得模型更容易训练**；但是**Width很大且Depth过深的网络很难学到更深层次的特征**
* 增加网络输入图像的分辨率r能够**潜在获得更高粒度的Feature Maps**；但是**对于过高的r，收益会减小**

**主要贡献**：
* 探索三个基础参数对模型效率之间的关联，**找到最优的r，width，depth配比**
* **重新设计了逆残差模块**(来源于MobileNet)，命名为**MBConv Block**，区别在于MBConv中的激活函数改为Swish(原为ReLU)，增加DropOut随机失活层。实际上并没有多大差别，纯属增加一个DropOut，如下两图展示。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/EfficientNet_v1_SE_Inverted_Residual.png)

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/EfficientNet_v1_SE_Inverted_Residual_2.png)

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/EfficientNet](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/EfficientNet)

### 1.9.1 EfficientNet V2-2021
**简介**：时隔两年，作者团队在V1产生的诸多问题上得出总结，并提出解决办法。同样，在原有的评价指标上又考虑了模型的训练速度。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/EfficientNet_v2.png)

**EfficientNet V1的问题**：
* 在训练的图像尺寸很大时，模型的训练速度很慢
* 模型的浅层使用DW卷积会降低训练速度，无法利用现有的底层加速器

**主要贡献**：
* 修改V1的**浅层卷积模块**，将DW卷积和SE模块退回不带注意力的传统卷积，**避免浅层过度关注不重要的信息**，这样的浅层模块叫做**Fused-MBConv**。所以V2同时使用了两种结构，**即浅层采用Fused-MBConv而深层采用MBConv**。仔细想一想，这是不是在炼丹呢？确实是在炼丹。

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/EfficientNet_v2_Fused_MBConv.png)
* 设计了渐进式学习策略**Progressive Learning**。其基本原则在于：(1)训练**早期使用较小的训练尺寸**以及**较弱的正则方法**做Weak Regulation，**这样网络可以快速学到一些简单的表达**；(2)之后**逐渐提升图像的尺寸**，同时**增强图像的正则化方法**，做Strong Regulation

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/EfficientNetV2](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/EfficientNetV2)


## 1.10 Transformer家族
### 1.10.1 Attention机制-2014-Google DeepMind
**论文**：
[2014]**视觉任务**：[Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247)
DOI：arXiv:1406.6247  
[2014]**机器翻译**：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473v7)  
DOI：arXiv:1409.0473

**简介**：纵观深度学习发展历史，谷歌的贡献是无与伦比的，几乎全是谷歌的作品。Attention机制最早出现在视觉领域，之后应用在自然语言处理领域，前一篇文章开创了Attention先河，后一篇文章首先将Attention应用在机器翻译领域。在此之后，Attention被广泛应用在基于RNN/CNN的各种视觉和翻译任务中。

**扩展概念**：Attention机制的形成
* Attention机制，其实正如它的名字一样-**注意力机制**，**注意力表示在观察一个事物时对其不同部分的侧重**。例如拍摄的一张照片里有各种动物，有的人特别注意里面的猫咪，有的人特别注意其中的狗子，还有的人则只在意里边的小青蛙。这种对于首要感兴趣内容的特别在意，或者说对同批数据中不同数据单元具有不同的侧重，就是一种注意力。
* 在NLP机器翻译领域中，通常要对一个序列化的数据进行处理，一个Sequence通常包含多个连续且具有强相关性的数据单元。对Sequence的处理通常会使用RNN(Recurrent NN，循环神经网络)，也会使用CNN来对输入做相关性处理来达到理解上下文的作用，但是**很难处理并行的情况**，此时若要输出 $b_4$ 就必须先后学习 $a_1, a_2, a_3, a_4$  。这样的问题极大限制了需要考虑上下文联系性的机器翻译技术的发展，不过RNN中输出的每一个向量都能学习到原来每一个输入的信息，这一点**启发了之后Self-Attention技术的发展**。同样，相邻的词语其影响作用是不一样的，如下图CNN和RNN的表示中，**同批次数据之间的关联都是等价的**，这实际上是不对的，所以**要计算不同个体之间的关联性的差异**，才能更准确地确定当前数据个体在语义空间的表示。

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_CNN_RNN.png)
* 实际上，在机器翻译领域还有很多嵌入式的表达方式(Embedding)。将**单词库表示成一组正则化的向量库，一段文字就对应了一个向量组，将相邻相关的文字对应的向量做一个相关性嵌入聚合**，这样这段文字对应的向量组就能够在一定程度上包含上下文的信息，能够**利用语境更好地理解多意词语的语境涵义**(例如上图中的两个粉丝，前者代表英译的fans即对某事某人某物的爱好者，后者则表示用淀粉等做成的丝状食品)。

**Attention机制**详解：
* **命名方式**：**借鉴了人类的选择性视觉注意力机制**，人类视觉会快速扫描全局，获得需要关注的目标区域，随后对关注区域投入更多的注意力资源，获取足够多的目标细节，同时**抑制其它区域的无用信息**
* **Encoder-Decoder框架**：是机器翻译的基础框架，最早在2014年提出-[Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)。输入的句子Source经过编码器Encoder转化为句子的语义编码C，之后由解码器转化为目标输出Target。但是，很容易就看出来**在公式 $y_i=G(C,y_1,...,y_{i-1})$ 中$y_{i-1}$是地位相同的**，也就是说要生成 $y_i$ 输出时， $y_1$ — $y_{i-1}$ 是等价的。这一点显然是不符合实际情况的，于是将语义编码改成了**动态语义编码** $C_i=\sum_{j}a_{ij}f(x_i)$ ，其中 $a_ij$ 是对于当前的 $y_i$ 的概率分布，由之前的输出和输入产生而来。Encoder-Decoder也被叫做Seq2Seq模型，中文称为序列到序列的学习，论文于2014年- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)。

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_Encoder_Decoder.png)

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_index.png)
* **计算注意力分布系数** $a_{ij}$ ：以RNN的模型为例，将输入 $x_i$ 与之前的输出状态一起用于计算注意力分布系数

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_RNN_index.png)
* **Attention机制的本质思想**：若将输入**Source**中的元素看成是由一系列的(**Key, Value**)数据对构成，对于**某个元素Query**欲计算其输出**Target**，则(1)**先计算Query和各个Key的相似/相关性，得到每个Key对于Value的权重系数W**，(2)**然后对Value进行加权求和，得到最终的Attention数值**。

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_path.png)

* Attention也可以看作是一种**软寻址**方式，输入Source可以看作存储器内存储的内容(**地址Key,值 Value**)。对一个查询Query，执行(1)查询Key=Query，将每个Key地址都与Query进行相似性比较，取出权重系数(2)将系数与原Value进行加权求和得到最后的Attention数值。以下是Attention机制总览，(1)首先计算 $Query$ 与 $Key_i$ 的相似性Similarity，可以采用点积、Cosine相似性或者MLP，(2)归一化，突出重要元素，(3)依据权重系数计算Attention数值

![image|400](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Attention_path_real.png)
**总结**：
* 实际上在我个人看来，人工智能**有多少人工才有多少智能**。Attention过程，会给原始数据做初步计算，计算出数据与数据之间的关联性，增加一个结构化的标签，从熵的角度来说，一开始就把信息熵降低了，把杂乱的数据部分有序化了，自然在某些情况下会提高模型的效果。
* 从本质上理解，**Attention是从大量信息中有筛选出少量重要信息，并聚焦到这些重要信息上**，忽略大多不重要的信息。**权重越大越聚焦于其对应的Value值上**，即权重代表了信息的重要性，而Value是其对应的信息。

### 1.10.2 Self-Attention / Transformer-2017-Google
**论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
**DOI**：arXiv:1706.03762  
**简介**：Self-Attention机制正如其名，叫做自注意力机制，也就是在原有数据之上先计算其内在的关联性，构建自我关联的关系模型，再将建模好的数据用来做后面一系列的处理。  

![image|300](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Self_Attention.png)

**Self-Attention机制详解**：
* **产生原因**：CNN不能直接用于处理变长的序列样本，但可以实现并行计算；完全基于CNN的Seq2Seq模型虽然可以并行实现，但非常占内存，很难在大数据量的参数进行调整。RNN则难以处理长序列的句子，计算时面临对齐问题，每个时间步的输出需要依赖于前面时间步的输出，这使得模型没有办法并行无法实现并行。**所以才考虑先计算全局的关联权重，用来避免输入距离的限制，降低模型的并行计算成本**。
* **Attention系数**：欲要计算输入向量之间的关联，可以采用加权点乘(可以再加一个非线性激活)的方法进行计算。这里引入几个加权矩阵 $Q-Query$ 搜索系数，表示 与其它输入的连接， $K-Key$ 关联值，表示被其它输入的连接， $V-Value$ 用于提取信息。  $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_K}})V$ ，其中的 $\sqrt{d_k}$ 用于对数值趋势进行强化，使得SoftMax操作更为明显。

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Self_Attention_Caculation.png)
* 多头注意力机制**Multi-Head Attention**-同时计算多组关联权重，多组可以表示不同的关系侧重
$W_q=[{W_{q_1},...,W_{q_m}}],W_k=[{W_{k_1},...,W_{k_m}}],W_v=[{W_{v_1},...,W_{v_m}}]$


![image|300](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Self_Attention_MultiHead.png)
* 总结Self-attention和多头注意力的**计算范式**：
  
$Multi-Head(Q,K,V)=Concat(head_1,...,head_h)W^o=ZW^o$
$head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Self_Attention_Path.png)
* **前馈神经网络**：实际上就是两层全连接层的层叠， $FFN(x)=max(0,xW_1+b_1)W2+b_2$ ，第一层带一个ReLU
* **Transformer模型**：**实际上是一个Encoder-Decoder结构(论文中是6个Encoder和6个Decoder组成)**，不过将传统的RNN换成了Self-Attention。其中编码器**Encoder由Self-Attention层和Position-wise Feed Forward Network(前馈网络，缩写为 FFN)组成**，**Decoder则由Self-Attention层、Encoder-Decoder Attention和FFN组成**。这个架构中，使用了残差连接和**Layer Normalization**结构，LN的论文(2016)如下：[Layer Normalization](https://arxiv.org/abs/1607.06450)，这一归一化过程是针对NLP领域提出的。

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Transformer.png)

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/Transformer_path.png)

* 位置编码**Position Encoding**：NLP算法中需要对输入序列做顺序编码，Transformer采用了一种独特的位置编码，使其维度与词向量Embedding长度一致。
$PE(pos,2i)=sin(\frac{pos}{10000^{\frac{2i}{d_model}}})$， $PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d_model}}})$ ，其中pos是指当前词在句子中的位置，$i$是指向量中每个值的位置下标。所以**在奇数位使用正弦编码，在偶数位使用余弦编码**。

**总结**：
* Self-Attention与CNN的区别在于CNN限制了感受野大小，CNN只计算了感受野(Receptive field)范围内的相似度，而**Self-Attention考虑了整个图像的相似度**。确切来说Self-Attention是CNN的扩展版本，其感受野由特征矩阵QKV设置(或者自动学习)
* 在同样的条件下，**Self-Attention需要更多的数据来完成权重的训练**，而CNN需要的数据就相对较少
* Self-Attention与RNN相比，RNN存在长期记忆遗忘的问题而且是串行输出的，**Self-Attention具有更好的并行性能**，在完成模型训练的前提下其计算效率更高
* 在NLP的翻译任务中需要解决三个任务：(1)原句内部词向量之间的关系(2)目标句内部的关系(3)原句与目标句之间的关系，在NLP的几大模型中**Seq2Seq只关注了(3)**，它对(1)和(2)的关注依旧采用的是RNN，对远距离信息的捕捉能力很差，而且训练慢，顺序执行并行度太低；**Transformer采用Self-attention和多头机制对(1)(2)(3)都进行了学习**，Position Encoding机制增大了模型的并行性

### 1.10.3 BERT-2018-Google(未复现)
**论文**：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)   
**DOI**：arXiv:1810.04805   
**简介**：Transformer在NLP大火之后，在Transformer之上的应用更多了。BERT的本质是一个自编码语言模型(Autoencoder LM)，它由多个Transformer叠合而成的，能够实现在多项NLP任务上的应用。  

**主要贡献**：(暂时不做太多展开)
* 是NLP领域的集大成者，在各类别NLP任务上都有很好的效果
* 堆叠Transformer的encoder构成深层模型，Base参数达到1.1亿，Large参数达到3.4亿
* 采用无监督训练，其效果竟然超i过有监督的模型
* 借鉴Transformer的Embedding，同时采用了Token Embedding(词向量)，Segment Embedding(语句标签)，Position Embedding(位置编码)

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/BERT.png)

### 1.10.4 Vision Transformer-2020-Google
**论文**：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)   
**DOI**：arXiv:2010.11929   
**简介**：ViT并不是率先将Transformer应用于CV的模型，但是其模型简单、效果好且可扩展性强，**成为transformer在CV领域应用的里程碑**，掀起CV领域里的Transformer的浪潮。**ViT论文的核心在于：当拥有足够多的数据进行预训练的时候，ViT的表现会超过CNN**；通常在数据集不够大的时候，ViT的表现通常比同等大小的ResNet要差一些，因为Transformer缺少**归纳偏置**。   

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/VisionTransformer.png)
**扩展概念**：
* **归纳偏置(Inductive Bias)**：(1)通俗一点，归纳偏置可以理解为，从现实生活中观察到的现象中归纳出一定的规则，然后对模型做一定的约束，从而可以起到 “模型选择” 的作用，类似贝叶斯学习中的 “先验”。(2)广义上，归纳偏置让学习算法优先考虑具有某些属性的解。从以上定义来看，**归纳偏置的意义或作用是使得学习器具有了泛化的能力**。
个人理解，如果拿一些数据给模型去进行学习，模型确实能够学习到数据中的特点，**但是这个过程可能会非常的长而且不一定能够达到预期的效果**。所以，我们可以在自然经验的基础之上，为整个模型加上某些**引导**，**使得模型能够更快更好地达到合适想象的效果**。

**主要贡献**：
* **将视觉任务转化为序列编码的NLP问题**，利用Transformer进行学习，以事实依据证明了Transformer的能力
* 掀起CV领域的Transformer浪潮 

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/VisionTransformer](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/VisionTransformer)

**题外话**：
* **论文**：[MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)，在CNN和Transformer大行其道的背景下，舍弃了卷积和注意力机制，提出了MLP-Mixer，一个完全基于MLPs的结构(**channel-mixing MLPs**融合通道信息，**token-mixing MLPs**融合空间信息)，依然达到SOTA。实验结果证明了convolution和attention不是必要操作，如果将其替换为简单的MLP，模型依然可以完美运行。
* **论文**：[When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)，**讨论了ViTs和MLP-Mixers对海量数据的依赖性**，使用SAM提高收敛模型的平滑度，大大提升了ViTs和MLPs在多个任务(监督、对抗、对比、迁移学习)上的准确度和鲁棒性。
* **论文**：[How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)，使用系统性的实证研究方法探究训练数据量、AugReg、模型大小、计算成本的相互影响。作者还对ViT迁移学习进行了深入分析。结论是**即使下游数据似乎与预训练数据只有微弱的关联，迁移学习仍然是最佳选择。**


### 1.10.5 Swin Tranformer-2021-MircoSoft Asian
**论文**：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)  
**DOI**：arXiv:1409.1556  
**简介**：又是微软接过了谷歌的视觉工作，**Swin-Transformer(Shifted Windows Transformer)结合CNN中的滑动窗口机制**计算**局部注意力**，**解决了ViT中全局注意力导致的计算量过大的问题**，其在诸多下游任务上的卓越表现使其成为2021年ICCV的最佳论文，此项工作进一步证明了Transformer在视觉领域是可以得到广泛应用的。  

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/SwinTransformer.png)
**主要贡献**：
* 结合CNN提出包含**滑动窗口**机制的**分级局部注意力计算机制**，构建类似CNN的**层次化构建方法**(Hierarchical feature maps)，进行图样的下4倍、8倍和16倍采样，构建有助于检测、分割任务的BackBone，不同于ViT只有16倍
* 构建Windows Multi-Head Self-Attention(**窗口多头注意力机制**, W-MSA)，将特征图划分为多个不相交的Window，只在每个Window内做Self-Attention，不同于ViT对全局进行多头注意力计算。采用**窗口多头注意力机制可以有效减少计算量，但是缺隔绝了不同窗口之间的信息传递**，由此作者又在增加了滑动窗口的思想，即Shifted Windows Multi-Head Self-Attention(**滑动窗口多头注意力机制**, SW-MSA)，让相邻窗口之间能够进行信息传递

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/SwinTransformer_Feature_Maps.png)
* 构建非常巧巧妙的**循环移位掩码机制**

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/SwinTransformer_CyclicShift.png)
**模块详解**：
* **窗口多头注意力机制**, W-MSA：将全局注意力**降解为局部注意力计算** 
$Attention(Q,K,V)=SoftMax(\frac{QK^T}{\sqrt{d}}V+B)$ ，其中 $B$ 为Relative Position Bias**相对位置索引**  
**MSA中计算步骤**为:
	(1)计算$QKV$矩阵， $A^{hw \times C} \cdot [W_{q}^{C \times C}, W_{k}^{C \times C},W_{v}^{C \times C}]=[Q^{{hw} \times C},K^{hw \times C},W_{v}^{C \times C}]$   
	(2)计算 $QK^T$ ， $Q^{{hw} \times C} \cdot K^{T(C \times hw)}=X^{hw \times hw}$  
	(3)计算中间结果乘上V， $\Lambda^{hw \times hw} \cdot V^{{hw} \times C}=B^{{hw} \times C}$  
	(4)计算多头的加权， $B^{hw \times C} \cdot W_O^{C \times C}=O^{hw \times C}$  
**总计复杂度为**：  $3 hwC^2+(hw)^2 C+(hw)^2 C + hwC^2=4 hw C^2+2(hw)^2C$  
**对应与W-MSA**，其窗口大小为$M*M$将原图划分成 $\frac{h}{M}\cdot\frac{w}{M}$ 个窗口，相当于MSA中的 $h=M,w=M$ ，带回原式，得到
 $$
 \frac{h}{M}\cdot\frac{w}{M}\cdot [3 {M \cdot M} \cdot {C}^2+2(M \cdot M)^2 \cdot C]=4hwC^2+2hwM^2C
 $$ 
**得到MSA与W-MSA的复杂度差距为**： $2(hw)^2C-2hwM^2C$ 
* **滑动窗口多头注意力机制**, SW-MSA：以滑动窗口打通局部注意力之间的关联计算，**重构全局注意力**

![image](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/SwinTransformer_SW_MSA.png)

* **模块迁移机制**，Patch Merging：

![image|500](https://github.com/YangCazz/papers/blob/main/Neural%20Network/pics/SwinTransformer_PatchMerging.png)

**模型复现**：
* 平台：Pytorch
* 主用库：torchvision, torch, matplotlib, tqdm
* 数据集：Oxford Flower102花分类数据集
* 代码[GitHub]：[DeepLearning/model_classification/SwinTransformer](https://github.com/YangCazz/DeepLearning/tree/master/model_classification/SwinTransformer)
