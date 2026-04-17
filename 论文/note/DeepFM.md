# DeepFM

特征交叉的CTR模型

## 动机

现有方法往往对低阶或高阶交互存在严重偏向，而非兼顾二者；特征交叉依赖专业的特征工程，而不能通过常理推断，甚至于只能通过机器学习挖掘出。

文章证明了可以构建一种端到端学习模型，该模型同时兼顾低阶与高阶特征交互，且除原始特征外无需任何特征工程。

## 模型结构

模型由两部分组成FM侧和Deep侧，二者共享底层Embedding.



FM侧如下

![image-20260415104352223](assets/image-20260415104352223.png)

图的表达比较迷惑，可以认为就是普通的FM结构——$$y_{FM} = \langle w, x \rangle + \sum_{i=1}^d \sum_{j=i+1}^d \langle V_i, V_j \rangle x_i \cdot x_j$$



DeeP侧如下，

![image-20260415104811865](assets/image-20260415104811865.png)

可以认为是MLP



最后两部分的结果相加—— $$\hat{y} = sigmoid(y_{FM} + y_{DNN})$$



## 结果

![image-20260415105542633](assets/image-20260415105542633.png)

FM&DNN是wide&Deep中将wide替换为FM，但是没有共享底层Embedding，结果证明共享嵌入层对指标有所提升。