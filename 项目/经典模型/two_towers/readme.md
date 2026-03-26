# 简介
基础的双塔模型，使用movie-1m数据集进行训练，并实现全量召回评估（Recall@K / NDCG@K）。

.
├── Data
│   ├── Movielens.py          # 数据集定义（特征映射，截断补充）
│   ├── build.py              # 原始 ml-1m 数据构建 train/test parquet（duckdb操作），按照时间对用户历史点击排序，取出最后n个用于测试
│   ├── build_id2idx.py       # user/movie/genre 等 ID 映射构建
│   ├── build_vocab.py        # title 词表构建（word2idx, word2emb）
│   └── review.ipynb          # 数据分析与调试 notebook
│
├── ml-1m
│   ├── ratings.dat           # 原始评分数据
│   ├── users.dat             # 原始用户数据
│   ├── movies.dat            # 原始电影数据
│   ├── movies_utf8.dat       # 转 utf8 后的电影数据
│
│   ├── train.parquet         # 训练集（正样本）
│   ├── test.parquet          # 测试集（leave-last-n）
│   ├── movies.parquet        # 全量物品库（用于召回）
│   ├── users.parquet         # 用户表
│
│   ├── uid2idx.pkl           # userId → uid
│   ├── mid2idx.pkl           # movieId → mid
│   ├── genre2idx.pkl         # genre → idx
│   ├── zip2idx.pkl           # zipcode → idx
│   ├── word2idx.pkl          # 词表（title）
│
│   ├── idx2uid.pkl           # uid → userId
│   ├── idx2mid.pkl           # mid → movieId
│   ├── idx2genre.pkl         # idx → genre
│   ├── idx2word.pkl          # idx → word
│
│   ├── word2emb.npy          # 词向量矩阵（用于 title embedding）
│
├── model
│   └── two_towers.py         # 双塔模型（user tower + item tower）
│
├── train.py                  # 训练 + 全量召回评估（Recall@K / NDCG@K）
├── config.py                 # 路径与超参数配置
├── utils.py                  # 工具函数（如年龄分桶等）
└── README.md


结果
epoch=1 train_loss=6.839180 recall@50=0.040545 ndcg@50=0.017496
epoch=11 train_loss=6.415965 recall@50=0.090284 ndcg@50=0.040023
epoch=21 train_loss=6.372253 recall@50=0.094629 ndcg@50=0.041206
epoch=31 train_loss=6.349845 recall@50=0.092797 ndcg@50=0.040291
epoch=41 train_loss=6.336064 recall@50=0.095907 ndcg@50=0.041544
epoch=51 train_loss=6.327689 recall@50=0.097722 ndcg@50=0.042740
epoch=61 train_loss=6.321719 recall@50=0.095167 ndcg@50=0.041142
epoch=71 train_loss=6.315387 recall@50=0.097932 ndcg@50=0.042245
epoch=81 train_loss=6.311047 recall@50=0.096873 ndcg@50=0.041427
epoch=91 train_loss=6.306791 recall@50=0.096504 ndcg@50=0.041205
best_recall@50=0.100370 

多次实验结果有所波动，大概在0.95附近