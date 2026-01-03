from collections import defaultdict
from heapq import heapreplace,heappush,heappushpop
from math import sqrt,log
import polars as pl
from tqdm import tqdm
from gensim.models import Word2Vec
import numpy as np
from cuml.neighbors import NearestNeighbors
import os

DAY = 86400
def build_itemcf(transactions,topk=50):
    item_cnt = defaultdict(int)
    cooc = defaultdict(float)

    for _, g in tqdm(transactions.group_by("customer_id"),total=transactions['customer_id'].unique().shape[0]):
        g = g.sort("time", descending=True).head(25) # 只能最近的25个防止计算超出内存限制
        items = g.select(["article_id", "time"]).unique().to_numpy()
        for i, ti in items:
            item_cnt[i] += 1
            for j, tj in items:
                if i == j:
                    continue
                dt = abs(ti - tj) # 购买两个物品的间隔时间
                time_w = 1 / (1 + dt / (7 * 86400)) # 约束，在相邻时间内购买的物品，应有较大的权重
                cooc[(i, j)] += time_w

    # 用小根堆堆每个物品保留最相似的20个
    item_sim=defaultdict(list)
    for (i, j), cij in cooc.items():
        # 对热门物品打压
        weight=cij / (
            sqrt(item_cnt[i] * item_cnt[j]) * log(item_cnt[i] + 10)
        )

        if len(item_sim[i]) < topk:
            heappush(item_sim[i], (weight, j))
        else:
            if weight> item_sim[i][0][0]:
                heapreplace(item_sim[i], (weight, j))

    return item_sim
def recall_itemcf(data, item_sim, topk=50):
    dfs=[]
    DAY=86400

    for cid ,g in tqdm(data.group_by('customer_id'),total=data['customer_id'].unique().shape[0]):
        g=g.sort('time',descending=True)
        cid=cid[0]
        scores=defaultdict(float)

        max_time=g['time'].max()
        hist_items = set(g["article_id"])

        # 不能截断物品序列，因为之前交互过的物品中可能有相似度更大的临近物品
        for aid,t in zip(g['article_id'],g['time']):
            dt=(max_time-t)/DAY
            # 对用户的物品序列做权重，操作时间越近权重越大
            w_t=1/(1+dt)
            for w_sim,sim_item in item_sim[aid]:
                if sim_item in hist_items:
                    continue
                scores[sim_item]+=w_t*w_sim

        res=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:topk]

        dfs.append(
            pl.DataFrame({
                "customer_id": [cid] * len(res),
                "article_id": [aid for aid, _ in res],
                "score": [score for _, score in res]
            })
        )

    return pl.concat(dfs)

def build_popular_items(transactions,window_days=7):
    max_time=transactions.select(pl.col('time').max()).item()
    start_time=max_time-window_days*DAY
    df=(
        transactions
        .filter(pl.col('time')>=start_time)
        .with_columns(
            weight=1/(1+(max_time-pl.col('time'))/DAY)
        )
        .group_by('article_id')
        .agg(
            score=pl.sum('weight'),
            cnt=pl.count()
        )
        .sort('score',descending=True)
    )
    return df
def recall_popularity(data,window_days,topk=50):
    df_pop=build_popular_items(data,window_days)
    top_item=df_pop.select('article_id').head(topk)['article_id'].to_list()

    users=data['customer_id'].unique()

    customer_ids = []
    article_ids = []
    ranks = []
    for cid in tqdm(users):
        for rank,aid in enumerate(top_item):
            customer_ids.append(cid)
            article_ids.append(aid)
            ranks.append(rank)
    del top_item,df_pop

    df = pl.DataFrame({
        "customer_id": customer_ids,
        "article_id": article_ids,
        "rank": ranks,
    }).with_columns(
        pl.col("rank").cast(pl.UInt8)
    )

    return df

def recall_repurchase_decay(data,topk=50,hist_len=100):
    DAY=86400

    dfs=[]

    for cid ,g in tqdm(data.group_by('customer_id'),total=data['customer_id'].unique().shape[0]):
        g=g.sort('time',descending=True).head(hist_len)
        cid=str(cid[0])
        scores=defaultdict(float)
        max_time=g['time'].max()

        for aid,t in zip(g['article_id'],g['time']):
            dt=(max_time-t)/DAY
            scores[aid]+=1/(1+dt)

        res=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:topk]

        dfs.append(
            pl.DataFrame({
                "customer_id": [cid] * len(res),
                "article_id": [aid for aid, _ in res],
                "rank": list(range(len(res)))
            })
        )

    return pl.concat(dfs).with_columns(
        pl.col("rank").cast(pl.UInt8)
    )


from sklearn.metrics.pairwise import cosine_similarity
def recall_item(model,topk):
    embeddings = np.array([model.wv[aid] for aid in model.wv.index_to_key], dtype=np.float32)# 词的向量
    knn=NearestNeighbors(n_neighbors=topk+1,metric='cosine') # 最近邻查找，每个样本返回21个（包括自己）
    knn.fit(embeddings)

    _,aid_nns=knn.kneighbors(embeddings)
    aid_nns=aid_nns[:,1:]

    idx2aid=model.wv.index_to_key

    item2item={}

    for idx, row in enumerate(aid_nns):
        src_vec = embeddings[idx].reshape(1, -1)
        nbr_vecs = embeddings[row]

        cos = cosine_similarity(src_vec, nbr_vecs)[0] # 返回[1,k]的矩阵[[cos_1,cos_2,...,cos_k]]

        item2item[idx2aid[idx]] = [
            (idx2aid[i], float(c))
            for i, c in zip(row, cos)
        ]
    return item2item

def train_w2vec(data,is_valid=False,model_path='../save/model/item2vec.model'):
    sentences=(
        data.sort(['customer_id','time'])
        .group_by('customer_id', maintain_order=True)
        .agg(pl.col('article_id').tail(30))
        ['article_id']
        .to_list()
    )
    if os.path.exists(model_path) and not is_valid:model=Word2Vec.load(model_path)
    else:
        model=Word2Vec(
            sentences=sentences,
            vector_size=64,
            window=5,
            min_count=5,
            workers=5,
            sg=1,
            epochs=5,
        )
    if not is_valid:model.save('../save/model/item2vec.model')

    return model

def recall_w2vec(data,topk=50,hist_len=10,is_valid=False,model_path='../save/model/item2vec.model'):
    rows=[]
    model=train_w2vec(data,is_valid,model_path)
    item2item=recall_item(model,topk)

    customer_ids=[]
    article_aids=[]
    scores=[]

    for cid,g in tqdm(data.group_by('customer_id'),total=data['customer_id'].unique().shape[0]):
        hist=(
            g.sort('time',descending=True)
            .select('article_id')
            .head(hist_len)['article_id']
            .to_list()
        )
        score = defaultdict(float)

        for i, aid in enumerate(hist):
            if aid not in item2item:
                continue
            w = 1.0 / (i + 1)

            for j,(rec_aid, cos) in enumerate(item2item[aid]):
                if rec_aid in hist:
                    continue
                score[rec_aid] += w/(j+1) # 越相似的物品权重越大

        res=sorted(score.items(),key=lambda x:x[1],reverse=True)[:topk]



        for aid, score in res:
            customer_ids.append(cid[0])
            article_aids.append(aid)
            scores.append(score)   # 这里已不再是 rank
    df = pl.DataFrame({
        "customer_id": customer_ids,
        "article_id": article_aids,
        "score": scores,
    })

    return df