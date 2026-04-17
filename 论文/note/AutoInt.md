# AutoInt

提出了一种高效实用的算法——自动学习输入特征的高阶特征交互，显式地建模不同阶数的特征组合，通用性强，同时适用于数值型与类别型输入特征，可解释性强。

## 动机

推荐场景下，输入特征通常稀疏且高纬，精准的预测依赖于告诫特征组合，这类特征由人工构建耗时严重，且无法穷举。因此，研究者一直致力于为稀疏高维的原始特征及其有效组合构建低维表征。

FM在各类任务中被验证具备有效性，但是受限于多项式拟合的时间复杂度，该方法仅适用于低阶特征交互建模。

大量基于MLP的模型在学习乘法类特征交互时效率较低，这些模型隐式地学习特征交互，可解释性弱。

## 模型结构

本文基于多头自注意力机制提出了这样一种方法。该方法能够学习稀疏高维输入特征的有效低维表示，同时兼容类别型与数值型输入特征。具体而言，首先将类别型特征与数值型特征嵌入至低维空间，以此降低输入特征维度，并使不同类型的特征可通过向量运算（如求和、内积）实现交互。随后，本文设计一种全新的交互层，用以强化不同特征间的关联。在每一层交互层中，各特征能够与其余所有特征进行交互，并借助多头注意力机制 [36] 自动筛选关联特征，生成具有实际意义的高阶特征。此外，多头机制可将单个特征映射至多个子空间，从而在不同子空间中捕获多样化的特征交互模式。单层交互层仅能建模特征间的一阶关联，通过堆叠多层交互层，即可实现不同阶数特征交互的建模。实际应用中，本文为交互层引入残差连接 ，以融合多阶特征组合信息。

模型结构如下所示

![image-20260417153019539](assets/image-20260417153019539.png)

模型首先经过嵌入层，转换为统一维度的嵌入表示。

### 嵌入层

输入特征分为三类：单值类别特征，多值类别特征，数值特征。

单值类别特征就是简单的映射，多值类别特征映射后取平均，数值特征做$e_m=v_m\times_m$，其中$v_m$是一个可学习的向量。

最后嵌入层的输出为各个类别的拼接。

### 交互层

作者使用注意力机制捕捉具备有效意义的特征组合。

以第$m$个特征为例，定义注意力头$h$下，特征$m$与特征$k$的关联度： 
$$
 \alpha_{m,k}^{(h)} &= \frac{\exp\left(\psi^{(h)}(\boldsymbol{e}_m,\boldsymbol{e}_k)\right)}{\sum_{l=1}^{M}\exp\left(\psi^{(h)}(\boldsymbol{e}_m,\boldsymbol{e}_l)\right)},\\ \psi^{(h)}(\boldsymbol{e}_m,\boldsymbol{e}_k) &= \left\langle \boldsymbol{W}_{\text{Query}}^{(h)}\boldsymbol{e}_m,\boldsymbol{W}_{\text{Key}}^{(h)}\boldsymbol{e}_k\right\rangle   
$$
其中，$\psi^{(h)}(\cdot,\cdot)$为注意力函数，用于度量特征$m$与特征$k$的相似度，可采用神经网络或内积$\langle\cdot,\cdot\rangle$实现。文中选用内积运算，$\boldsymbol{W}_{\text{Query}}^{(h)}$、$\boldsymbol{W}_{\text{Key}}^{(h)} \in \mathbb{R}^{d'\times d}$为变换矩阵，用于将原始嵌入空间$\mathbb{R}^d$映射至新的特征空间$\mathbb{R}^{d'}$。

得到的注意力分数可用于解释特征之间的交互。

随后，结合注意力系数$\alpha_{m,k}^{(h)}$聚合所有相关特征，对子空间$h$中特征$m$的表征进行更新：
$$
\tilde{\boldsymbol{e}}_m^{(h)}=\sum_{k=1}^{M}\alpha_{m,k}^{(h)}\left(\boldsymbol{W}_{\text{Value}}^{(h)}\boldsymbol{e}_k\right) 
$$
其中$\boldsymbol{W}_{\text{Value}}^{(h)} \in \mathbb{R}^{d'\times d}$。 $\tilde{\boldsymbol{e}}_m^{(h)}$融合了当前注意力头$h$下特征$m$及其关联特征，代表模型学习得到的全新组合特征。单一特征可参与多种特征组合，多头机制可构建不同子空间，独立学习差异化的特征交互模式，以此实现多类型组合特征的挖掘。

整合所有子空间学习到的组合特征，公式如下： 
$$
\tilde{\boldsymbol{e}}_m = \tilde{\boldsymbol{e}}_m^{(1)} \oplus \tilde{\boldsymbol{e}}_m^{(2)} \oplus \dots \oplus \tilde{\boldsymbol{e}}_m^{(H)} \tag{7}
$$
 其中$\oplus$为拼接操作，$H$为注意力头总数。为保留已学习的组合特征（包括原始一阶独立特征），网络引入标准残差连接。定义如下：
$$
 \boldsymbol{e}_m^{\text{Res}} = \text{ReLU}\big(\tilde{\boldsymbol{e}}_m + \boldsymbol{W}_{\text{Res}}\boldsymbol{e}_m\big)  
$$
$e_m$是交叉层的输入，$\boldsymbol{W}_{\text{Res}} \in \mathbb{R}^{d'H\times d}$ 为维度匹配投影矩阵。 通过该交互层，每个特征$\boldsymbol{e}_m$会被更新为高阶特征表征$\boldsymbol{e}_m^{\text{Res}}$。堆叠多层交互层，将上一层输出作为下一层输入，即可实现任意阶数组合特征的建模。

### 输出层

交互层的输出为一组特征向量 $\{\boldsymbol{e}_m^{\text{Res}}\}_{m=1}^M$，m是层数，将所有特征向量拼接后进行非线性映射： $$ \hat y = \sigma\Big(\boldsymbol{w}^\text{T}\big(\boldsymbol{e}_1^{\text{Res}} \oplus \boldsymbol{e}_2^{\text{Res}} \oplus \dots \oplus \boldsymbol{e}_M^{\text{Res}}\big) + b\Big) \tag{9} $$ 其中，$\boldsymbol{w} \in \mathbb{R}^{d'HM}$ 为投影列向量，$\sigma(x)$ 为sigmoid，输出得到最终的预测点击率。

## 结果

在四个真实数据集中，AutoInt 于三个数据集上取得最优性能。在 Avazu 数据集上，CIN 的 AUC 指标略高，但本文的模型对数损失更低。

![image-20260417161853473](assets/image-20260417161853473.png)

与竞争力最强的基线模型 CIN 相比，本文的模型参数量更少，在线推理阶段的运行效率更高。

![image-20260417162237189](assets/image-20260417162237189.png)

去除残差连接后，模型在所有数据集上的性能均出现下降

<img src="assets/image-20260417162613377.png" alt="image-20260417162613377" style="zoom:50%;" />

随着交互层数继续增加，模型可捕获更高阶的特征组合，性能持续优化；当层数达到三层后，性能趋于稳定，表明过度高阶的特征组合对预测增益有限。

<img src="assets/image-20260417162721398.png" alt="image-20260417162721398" style="zoom:50%;" />

文中将两层前馈神经网络与原模型并行训练，输出融合，探索隐式特征对模型性能的提升，经过对比（其他模型也同时联合训练），证明引入隐式特征交互能够增强模型预测能力。但从后两列数据可以看出，相比其他模型，本文模型的性能提升幅度较小，这也说明原生 AutoInt 模型本身已具备极强的特征建模能力。

![image-20260417163155370](assets/image-20260417163155370.png)





