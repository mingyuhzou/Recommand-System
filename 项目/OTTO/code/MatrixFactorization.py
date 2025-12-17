# %%
from os import access

import polars as pl
# %%
train=pl.read_parquet('../data/processData/train.parquet')
test=pl.read_parquet('../data/processData/test.parquet')
# %%
train_pairs=pl.concat([train,test]).select(['session','aid'])
train_pairs.head(),train_pairs.shape
# %%
# 构建aid_next，agg聚合函数，收集组内的aid返回一个列表，shift(-1)上移一位
train_pairs=train_pairs.group_by('session').agg([pl.col('aid'),pl.col('aid').shift(-1).alias('aid_next')])
# explode 将aid和aid_next展开
train_pairs=train_pairs.explode(['aid','aid_next']).drop_nulls()[['aid','aid_next']]
# %%
train_pairs.head(),train_pairs.shape
# %%
cardinality_aids=max(train_pairs['aid'].max(),train_pairs['aid_next'].max())
cardinality_aids
# %%
train_pairs[:-10_000_000].to_pandas().to_parquet('../save/train_pairs.parquet')
train_pairs[-10_000_000:].to_pandas().to_parquet('../save/valid_pairs.parquet')
# %%
from torch.utils.data import Dataset
import pandas as pd

class AIDPairDataset(Dataset):
    def __init__(self,path):
        self.df=pd.read_parquet(path).reset_index(drop=True)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        return {'aid':torch.tensor(row.aid,dtype=torch.long),'aid_next':torch.tensor(row.aid_next,dtype=torch.long)}
    def __len__(self):
        return len(self.df)
# %%
from torch.utils.data import DataLoader

train=AIDPairDataset('../save/train_pairs.parquet')
valid=AIDPairDataset('../save/valid_pairs.parquet')

train_loader=DataLoader(dataset=train,batch_size=65536,shuffle=True)
valid_loader=DataLoader(dataset=valid,batch_size=65536,shuffle=True)
# %%
import torch
from torch import nn

class MatrixFactorization(nn.Module):
    def __init__(self,n_aids,n_factors):
        super(MatrixFactorization,self).__init__()
        self.aid_factors=nn.Embedding(n_aids,n_factors,sparse=True)
    def forward(self,aid1,aid2):
        aid1=self.aid_factors(aid1)
        aid2=self.aid_factors(aid2)
        return (aid1*aid2).sum(dim=1)
class AverageMeter(object):
    def __init__(self,name,fmt=':f'):
        self.name=name
        self.fmt=fmt
        self.reset()
    def reset(self):
        self.val=self.avg=self.sum=self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count
    def __str__(self):
        fmtstr='{name} {val'+self.fmt+'} ({avg' +self.fmt +'})'
        return fmtstr.format(**self.__dict__)
# %%
from torch.optim import SparseAdam

num_epochs=1
lr=0.1

model=MatrixFactorization(cardinality_aids+1,32)
optimizer=SparseAdam(model.parameters(),lr=lr)
BCE=nn.BCEWithLogitsLoss()
# %%
model.to('cuda')
for epoch in range(num_epochs):
    for batch in train_loader:
        model.train()
        shower=AverageMeter('loss',':.4e')

        aid1,aid2=batch['aid'],batch['aid_next']
        aid1=aid1.to('cuda')
        aid2=aid2.to('cuda')

        # 矩阵分解是二分类问题，要构造正样本和负样本
        output_pos=model(aid1,aid2)
        output_neg=model(aid1,aid2[torch.randperm(aid2.shape[0])])

        # 构造二分类损失
        output=torch.cat([output_pos,output_neg])
        targets=torch.cat([torch.ones_like(output_pos),torch.zeros_like(output_neg)])

        loss=BCE(output,targets)
        shower.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

    with torch.no_grad():
        accuracy=AverageMeter('accuracy')
        for batch in valid_loader:
            aid1,aid2=batch['aid'],batch['aid_next']
            output_pos=model(aid1,aid2)
            output_neg=model(aid1,aid2[torch.randperm(aid2.shape[0])])
            accuracy_batch=torch.cat([output_pos.sigmod()>0.5,output_neg.sigmod()<0.5]).float().mean()
            accuracy.update(accuracy_batch,aid1.shape[0])
    print(f'{epoch+1:02d}: * TrainLoss {shower.avg:.3f}  * Accuracy {accuracy.avg:.3f}')
# %%
# detach从计算图中分离出来，将数据一道cpu上才能转换为numpy数组
embeddings=model.aid_factors.weight.detach().cpu.numpy()
# %%
import faiss
import numpy as np

vecs=embeddings.astype(np.float32)
d=vecs.shape[0]
k=21

index=faiss.IndexFlatL2(d)

index.add(vecs)


# %%
from collections import defaultdict
session_types=['clicks','carts','orders']

sample_sub=pd.read_csv('../data/rawData/sample_submission.csv')
# 聚合session
test_session_AIDs=test.reset_index(drop=True).groupby('session')['aid'].apply(list)
test_session_types=test.reset_index(drop=True).groupby('session')['type'].apply(list)

labels=[]

type_weight_multipliers={0:1,1:6,2:3}
for AIDs,types in zip(test_session_AIDs,test_session_types):
    # 当session的操作数>=20，对操作加权，权重由位置和类型决定
    if len(AIDs)>=20:
        # base底数，endpoint是否包含右边界
        weights=np.logspace(0.1,1,len(AIDs),base=2,endpoint=True)-1
        aids_temp=defaultdict(lambda: 0)
        for aid,w,t in zip(AIDs,weights,types):
            aids_temp[aid]+=w*type_weight_multipliers[t]
        sorted_aids=[k for k,v in sorted(aids_temp.items(),key=lambda item:-item[1])]
        labels.append(sorted_aids[:20]) # 取前20个
    else:
        # 对短序列用共现矩阵补全
        AIDs=list(dict.fromkeys(AIDs[::-1])) # 去重,倒序保证能保留最近的操作

        most_recent_aid=AIDs[0]

        nns=index.search(most_recent_aid,k)
        # 为每个商品找到出现次数最多的20个
        for AID in AIDs:
            if AID in next_AIDs:
                # most_common返回出现次数最多的元素及其频次
                candidates+=[aid for aid,count in next_AIDs[AID].most_common(20)]
        # 保证不重复添加到最终结果中
        AIDs+=[AID for AID ,cnt in Counter(candidates).most_common(40) if AID not in AIDs]

        labels.append(AIDs[:20])
# %%
labels_as_strings=[' '.join([str(l) for l in lls]) for lls in labels] # 将预测的结果按转换为用空格分开的字符串

predictions=pd.DataFrame(data={'session_type':test_session_AIDs.index,'labels':labels_as_strings})
prediction_dfs=[]

for st in session_types:
    modified_predictions=predictions.copy()
    modified_predictions.session_type=modified_predictions.session_type.astype('str')+f'_{st}'
    prediction_dfs.append(modified_predictions)

submission=pd.concat(prediction_dfs).reset_index(drop=True)
submission.to_csv('../save/mf_submission.csv',index=False)