# 简介
基于 PyTorch 实现的序列推荐模型 SASRec，支持 MovieLens-1M 数据集训练与评估,
实现参考 https://github.com/seanswyi/sasrec-pytorch/tree/main    最初自行实现，但是指标达不到论文的中效果k,recall和ndcg只有一半，怀疑是数据流、验证流水线没对齐，因此仿照这个版本实现。

注意，模型初始化非常重要，如果没有下面的初始化，指标只能达到一半。
    for param in model.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
.
├── data
│   └── movie-lens_1m.txt     # 原始数据
├── logs
│   └── sasrec_train.log      # 训练日志
├── outputs
│   └── best_sasrec.pt        # 最优模型
├── readme.md
├── setup.py
└── src
    ├── dataset.py            # 数据处理，截断，填充，划分...
    ├── model
    │   ├── embeding_layer.py # embedding层
    │   ├── pffn.py           # 前馈网络
    │   ├── sasrec.py         # 主模型
    │   └── self_attn_block.py# 自注意力块（多头自注意力机制+PFFNN）
    ├── train.py              # 训练入口
    └── utils
        └── utils.py          # 工具函数（采样/参数等）

这里的数据划分使用了留一法，train=dta[:-2] 对train中每个从头开始的子序列向右移动一位作为正样本，负样本是随机抽样组成的负样本；训练中，使用valid=data[:-2]，label=data[-2]验证；最后，使用test=data[:-1]，label=data[-1]测试。

