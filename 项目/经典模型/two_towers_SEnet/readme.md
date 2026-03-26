# 简介
双塔模型+SENET，使用movie-1m数据集进行训练，并实现全量召回评估（Recall@K / NDCG@K）。

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
epoch=1 train_loss=6.844913 recall@50=0.030610 ndcg@50=0.013179
epoch=11 train_loss=6.415206 recall@50=0.093310 ndcg@50=0.041983
epoch=21 train_loss=6.369711 recall@50=0.094150 ndcg@50=0.041017
epoch=31 train_loss=6.350050 recall@50=0.095428 ndcg@50=0.041523
epoch=41 train_loss=6.337281 recall@50=0.095739 ndcg@50=0.041648
epoch=51 train_loss=6.327601 recall@50=0.092789 ndcg@50=0.040483
epoch=61 train_loss=6.319650 recall@50=0.095285 ndcg@50=0.041048
epoch=71 train_loss=6.313009 recall@50=0.095571 ndcg@50=0.041244
epoch=81 train_loss=6.306838 recall@50=0.095520 ndcg@50=0.041103
epoch=91 train_loss=6.302124 recall@50=0.096462 ndcg@50=0.041232
best_recall@50=0.098378

与基本的双塔模型相比，并没有明显的改进，可能还降低了，也许是因为特征太少了，senet学习的权重没有带来明显的作用
