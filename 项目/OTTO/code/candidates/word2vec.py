# %%
import numpy as np
import polars as pl # 高速DataFrame库，用于数据处理与分析，速度更快，内存占用更低
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd
# %%
train=pl.read_parquet('/home/mingyu/Recommand-System/项目/OTTO/data/processData/train.parquet')
test=pl.read_parquet('/home/mingyu/Recommand-System/项目/OTTO/data/processData/test.parquet')
# %%
min(train['aid'])
# %%
# pl.concat纵向合并，groupby按照session分组，agg聚合函数(不保留原维度，合并session)，pl.col选择列，alias重命名
# 为每个session合并aid组合成句子
sentences_df=pl.concat([train,test]).group_by('session').agg(pl.col('aid').alias('sentence'))
# %%
sentences_df
# %%
sentences=sentences_df['sentence'].to_list()
# %%
sentences
# %%
import os
# 做embedding
model_path='/home/mingyu/Recommand-System/项目/OTTO/model/w2vec.model'
if os.path.exists(model_path):
    w2vec=Word2Vec.load(model_path)
else:
    # sentences训练语料，vector_size词向量维度，min_count此至少出现一次才会被训练，workers训练时使用的CPU线程数
    w2vec=Word2Vec(sentences=sentences,vector_size=32,min_count=1,workers=4)
    w2vec.save(model_path)
# %%
aid_list=w2vec.wv.index_to_key # 按照出现频率从高到低返回词的列表
aid_list
# %%
aid2idx={aid:i for i,aid in enumerate(aids)}

# w2vec.wv[aid]可以检索到向量
embeddings = np.array([w2vec.wv[aid] for aid in w2vec.wv.index_to_key], dtype=np.float32)# 词的向量
d=w2vec.wv.vectors.shape[1]
# %%
embeddings
# %%
from cuml.neighbors import NearestNeighbors
knn=NearestNeighbors(n_neighbors=21,metric='euclidean') # 最近邻查找，每个样本返回21个（包括自己）
knn.fit(embeddings) # 接受（n_samples,n_features）训练模型
# %%
_, aid_nns = knn.kneighbors(embeddings)
# %%
aid_nns = aid_nns[:, 1:] #排除自身
aid_nns
# %%
top_aids = [[aid_list[i] for i in row] for row in aid_nns]
top_aids
# %%
sub=[]
for aid_x in range(aid_nns.shape[0]):
    for aid_y in top_aids[aid_x]:
        sub.append([aid_x,aid_y])
        sub = pd.DataFrame(sub, columns=['aid_x','aid_y'])
sub.to_parquet('/home/mingyu/Recommand-System/项目/OTTO/save/co_visitation_result/word2vec.parquet', index=False)