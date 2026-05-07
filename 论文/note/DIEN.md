# 模型

相比于Din模型的只能对历史物品用目标物品加权的特点，DIEN还能建模用户的历史兴趣是如何关于目标物品演变为最终的兴趣向量。

## 动机

现有多种用于兴趣建模的点击率预测方法，但多数方法直接将行为等同于用户兴趣，缺乏对具体行为背后潜在兴趣的针对性建模，同时鲜有研究考虑兴趣的变化趋势。

## 模型结构

DIEN的结构与DIN相似，首先所有的特征过一个embeding layer，随后用户序列特征进入兴趣提取模块处理，最后将各个embedding拼接后送入到mlp中。

![image-20260428145647499](assets/image-20260428145647499.png)

兴趣提取模块由两部分组成：Interest Extractor Layer从用户行为中提取兴趣序列；Interest Evolving Layer根据目标物品建模兴趣的进化。

### Interest Extractor Layer

作者认为历史行为不能表示兴趣，所以需要从行为中提取兴趣序列，Interest Extractor Layer使用GRU（Gated Recurrent Unit）从用户行为序列中提取一系列兴趣状态，其输入会按照时间排好序。

GRU，门控循环单元，循环神经网络RNN的一种改进结构，用于建模序列数据，能解决传统RNN在长序列中出现的梯度消失问题，且运算速度比LSTM更快。

GRU 的计算公式如下，有T个物品就会生成T个兴趣状态，每个兴趣状态都是从上一个兴趣状态与当前物品融合得到的：
$$
u_t = \sigma(W_u i_t + U_u h_{t-1} + b_u) \\
r_t = \sigma(W_r i_t + U_r h_{t-1} + b_r) \\
\tilde{h}_t = \tanh(W_h i_t + r_t \circ U_h h_{t-1} + b_h) \\
h_t = (1 - u_t) \circ h_{t-1} + u_t \circ \tilde{h}_t \\
$$
其中：

- $ \sigma $ 表示 sigmoid 激活函数
- $ i_t $ 表示 GRU 在时刻 $t $ 的输入，其中 $i_t = e_b[t] $，表示用户在第 $t $ 次的行为 embedding
- $\circ $ 表示逐元素乘
- $ W_u, W_r, W_h \in \mathbb{R}^{n_H \times n_I} $
- $ U_u, U_r, U_h \in \mathbb{R}^{n_H \times n_H} $
- $ n_H $ 为隐状态维度，$n_I $ 为输入维度

- $ h_t $ 表示第 $ t $ 个隐状态
- $u_t$ 更新门，控制保留多少旧状态，引入多少新信息
- $r_t$ 重置门，控制在计算候选状态时，是否忽略过去



作者认为仅捕捉行为依赖关系的隐藏状态 ${h}_t$ 无法有效表征用户兴趣。目标商品的点击行为由最终兴趣驱动，目标损失仅监督最终兴趣的预测真值，无法为历史时刻隐藏状态提供有效监督，考虑到每一时刻的用户兴趣会直接引导下一连续行为的产生，因此作者引入了辅助任务：利用下一时刻的行为，监督当前兴趣状态的学习。

![image-20260428170905160](assets/image-20260428170905160.png)

模型将真实的下一行为作为正样本，同时从全商品集合中采样非点击商品作为负样本（排除正样本），计算
$$
L_{aux} = -\frac{1}{N} \left( \sum_{i=1}^{N} \sum_{t} \log \sigma(\mathbf{h}_t^i, \mathbf{e}_b^i[t+1]) + \log\left(1 - \sigma(\mathbf{h}_t^i, \hat{\mathbf{e}}_b^i[t+1])\right) \right)
$$

+ ${h}^i_t$为第 i 个用户在第 t 步的 GRU 隐藏状态
+ $e_b^i$是用户点击序列，$\hat{\mathbf{e}_b^i}$为负采样序列
+ $ \sigma(\mathbf{x}_1, \mathbf{x}_2) = \frac{1}{1 + \exp\left(-\left[\mathbf{x}_1, \mathbf{x}_2\right]\right)} $



点击率预测的最终损失函数为
$$
L = L_{target} + \alpha * L_{aux}
$$


拼接全部 T 个兴趣状态 $[\boldsymbol{h}_1,\boldsymbol{h}_2,\dots,\boldsymbol{h}_T]$，即可生成具备强表达能力的兴趣序列

### Interest Evolving Layer

受外部环境与内在认知的共同影响，用户各类兴趣会随时间不断演化。以服饰类兴趣为例，随着流行趋势与个人审美变化，用户的服饰偏好会持续迭代，而这一兴趣演化过程将直接决定服饰候选商品的点击率预测结果。

用户的兴趣存在两个关键特性：（1）兴趣漂移，用户的兴趣是多样的，某一阶段用户可能集中关注书籍类商品，后续又会产生服饰类消费需求；（2）不同兴趣间虽存在潜在关联，但各自拥有独立演化路径。



Interest Evolving Layer融合注意力机制的局部激活能力与门控循环单元的序列学习能力完成兴趣演化建模——在门控循环单元的每一步运算中引入局部激活，可强化相关兴趣的作用权重、削弱兴趣漂移带来的干扰。

具体表现为候选物品的向量与每一步兴趣状态计算attention激活对应的兴趣，attention score的得分在AUGRU中控制兴趣状态的转移，如果某一行为不相关，则不会更新，即$h_t=h_t-1$，最终得到的是一条和候选物品相符的路径。Din模型虽然能弱化不相关行为，但是做不到消除。

注意力权重的计算如下
$$
a_t = \frac{\exp(h_t W e_a)}{\sum_{j=1}^{T} \exp(h_j W e_a)}
$$

- $h_t$：Interest Extractor 输出
- $i'_t$：Interest Evolving 输入
- $W \in \mathbb{R}^{n_H \times n_A}$，$n_A$是目标物品的维度
- $e_a$ 目标物品的 embedding



作者提出了三种结合注意力机制与GRU的结构

首先是AIGRU，用注意力分数调控演化层的输入
$$
i'_t = h_t \cdot a_t
$$
式中，$\boldsymbol{h}_t$为兴趣抽取层 GRU 第t步隐藏状态，$\boldsymbol{i}'_t$ 为第二层演化 GRU 的输入.

该结构存在明显缺陷，即使输入置0，仍会改变GRU的隐藏状态（$h_t^`$），导致弱相关兴趣依旧干扰兴趣演化的学习过程。



随后是 AGRU，用注意力得分直接替换GRU的更新门
$$
h'_t = (1 - a_t), h'_{t-1} + a_t, \tilde{h}'_t
$$

AGRU 虽可通过注意力得分直接约束隐藏状态更新，但使用单一标量注意力得分替换向量形式的更新门，忽略了更新门不同维度间的重要性差异。



最后是AUGRU，用attention score对更新门加权
$$
\tilde{u}'_t = a_t \cdot u'_t \\
h'_t = (1 - \tilde{u}'_t) \circ h'_{t-1} + \tilde{u}'_t \circ \tilde{h}'_t
$$

- $u'_t$：GRU 原始 update gate

AUGRU 完整保留更新门的维度特征，并利用注意力得分对更新门全局维度进行缩放。



## 结果

在公开数据集上测试，其中，BaseModel使用与DIEN一致的embedding layer和MLR，但是对行为的建模只用了sum pooling。

![image-20260429103513488](assets/image-20260429103513488.png)

![image-20260429103004587](assets/image-20260429103004587.png)

可以观察到，DIEN有不错的提升，且第二层的注意力门控循环单元选择AUGRU效果更优。



在真实数据集上进行A/B测试，该模型带来了巨大的提升

![image-20260429103421390](assets/image-20260429103421390.png)
