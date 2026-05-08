# Wukong

特征交叉模型，核心创新在于通过堆叠多层因子分解机，大幅提升了传统因子分解机的表征能力，能够有效捕捉高阶特征交互并且满足缩放定律，作者关注于特征交互层的扩容而不是嵌入表的，即便模型复杂度跨越两个数量级，在现有模型表现失效的区间内，该模型仍能保持效果优势。

## 动机

目前的推荐模型并未呈现出与大语言模型领域类似的缩放规律，当下推荐系统的扩容方式为spare scaling，即扩大嵌入表规模或维度，这种方法无法增强对海量特征间复杂交互的捕捉能力，并且嵌入表查找操作无法利用当下的GPU的算力优势。

这种使得推荐模型难以适配复杂的大型的现实数据，因此作者提出了Wukong旨在为推荐领域建立缩放定律。

## 模型结构

模型中所有的特征先经过EMbedding layer转换为稠密的向量，随后送入到堆叠的交互层，交互层由FMB和LCB组成，最终进入MLP。

![image-20260507153817990](assets/image-20260507153817990.png)

### Embedding Layer

Embedding Layer并非简单的查表，对于特征中较为重要的会生成多个嵌入，次要的则会生成较少的嵌入，特征最后会拼接起来经过MLP映射到统一的维度。

文章中将Embedding Layer输出的每一个嵌入向量视为独立的整体，表示为$$\boldsymbol{X}_0 \in \mathbb{R}^{n\times d}$$，而非一维形式$$\boldsymbol{X}_0 \in \mathbb{R}^{nd}$$。

### Interaction Stack

交互层包含两个并行模块：因子分解机模块（FMB）与线性压缩模块（LCB）。因子分解机模块计算该层输入嵌入之间的特征交互，线性压缩模块则对该层输入嵌入进行线性压缩并直接传递输出。两个模块的输出随后进行拼接。



FMB的计算可以总结为以下：
$$
\text{FMB}(X_i) = \text{reshape}\left( \text{MLP}\left( \text{LN}\left( \text{flatten}\left( \text{FM}(X_i) \right) \right) \right) \right)
$$


- 输入$\boldsymbol{X}_i \in \mathbb{R}^{n\times d}$先经过$\text{FM}(\boldsymbol{X})=\boldsymbol{X}\boldsymbol{X}^\text{T}$，得到$\boldsymbol{X}_i \in \mathbb{R}^{n\times n}$即每个token之间的interaction score
- 接着展平再经过一个归一化层稳定训练，$\boldsymbol{X}_i \in \mathbb{R}^{nn}$ 
- 随后送入MLP中把信息重新编码回embedding space，也就是$\boldsymbol{X}_i \in \mathbb{R}^{n_fd}$，这里$n_f$是新生成的Embedding数量
- 最后reshape展开，$\boldsymbol{X}_i \in \mathbb{R}^{n_f\times d}$



因式分解机的计算与存储复杂度会随嵌入数量呈平方级增长，在包含数千个特征的真实数据集上开销会很快达到难以承受的程度。

为此作者利用了逐对内积矩阵的低秩特性来降低复杂度——当嵌入维度d<=嵌入数量的时候，内积交互矩阵$\boldsymbol{X}\boldsymbol{X}^\text{T}$ 为低秩矩阵，大规模数据集一般都满足这一条件。因此引入形状为$n\times k$的可学习投影矩阵Y，与$\boldsymbol{X}\boldsymbol{X}^\text{T}$相乘得到$\boldsymbol{X}\boldsymbol{X}^\text{T}Y$，可以在理论无损的前提下将输出矩阵尺寸从$n \times n$降至$n \times k$大幅降低交互矩阵的存储开销。

利用矩阵乘法结合律，先计算$\boldsymbol{X}^\text{T}Y$，可以将计算复杂度从$O(n^2d)$降至$O(nkd)$



LCB是简单的线性层
$$
 \text{LCB}(X_i) = W_L X_i 
$$
$W_j \in \mathbb{R}^{n_L \times n_i}$ 为权重矩阵，$n_L$ 是表示压缩后嵌入数量的超参数，$n_i$ 为第 $i$ 层的输入嵌入数量。

## Scaling

模型扩容的超参数如下

+ $l$：交互堆叠层的层数
+ $n_F$：FMB生成的嵌入向量数量
+ $n_L$：LCB生成的嵌入向量数量
+ k：优化因子分解机中的压缩嵌入向量数量
+ $MLP$：网络的深度和维度



## 工程优化

在大规模数据集上应用模型需要做工程上的优化

+ 进行分布式训练——多机多卡
+ 对Embedding table 进行column-wise分片，即[100000000, 128]->GPU0: [100000000, 32]，GPU1: [100000000, 32]，GPU2: [100000000, 32]，GPU3: [100000000, 32]。相比于row-wise，该方法的通信效率更高。
+ 使用FSDP策略，将模型的参数划分到多卡上，前向传播反向传递时聚合，能降低显存占用
+ 精度混合

## 结果

模型在六个公开数据集上优于以往模型

![image-20260508101642713](assets/image-20260508101642713.png)

由消融实验可知，模型中各个部分

![image-20260508102321447](assets/image-20260508102321447.png)



文中分析了悟空模型中各超参数单独扩容对模型效果的贡献。横坐标表示一个样本所需的计算量，纵坐标是loss相对于baseline（DLRM）的减少。

![image-20260508103739760](assets/image-20260508103739760.png)

实验发现：增加悟空模型层数l可显著提升模型效果，原因是能够捕捉更高阶特征交互；扩大 MLP 规模也能带来可观性能增益；增大k与$n_F$同样有正向收益，而$n_L$在基础配置下已趋于性能饱和。值得注意的是，**联合扩容**k、$n_F$、$n_L$带来的效果提升，明显优于各参数单独扩容。

## 不足

将模型扩容至高复杂度级别会给在线实时推理带来显著挑战

由于算力开销巨大，本文对模型的理论上限还未研究彻底

尽管模型在多项测评中表现优异，但是对底层原理人缺乏万倍的理论解析



