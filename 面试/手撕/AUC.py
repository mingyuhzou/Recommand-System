import numpy as np
from sklearn.metrics import roc_auc_score

#给定的真实y 和 预测pred
y    = np.array([1,   0,   0,   0,   1,    0,   1,    0,    0,   1  ])
pred = np.array([0.9, 0.4, 0.3, 0.1, 0.35, 0.6, 0.65, 0.32, 0.8, 0.7])

numerator = 0    #分子
denominator = 0  #分母

# for i in range(0, len(y)-1):
#     for j in range(i, len(y)):
#         # 只讨论一正一负样本对
#         if y[i] != y[j]:
#             denominator += 1
#             #统计所有正负样本对中，模型把相对位置排序正确的数量
#             if(y[i]>y[j] and pred[i]>pred[j]) or (y[i]<y[j] and pred[i]<pred[j]):
#                 numerator += 1

for i in range(0,len(y)-1):
    for j in range(i,len(y)):
        if y[i]!=y[j]:
            denominator+=1
            if (y[i]<y[j] and pred[i]<pred[j]) or (y[i]>y[j] and pred[i]>pred[j]):
                numerator+=1
print("AUC =" , numerator/denominator)
print(roc_auc_score(y, pred))

def auc(y, pred):
    """nlogn写法"""
    order=pred.argsort()

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    sorted_pred = pred[order]
    sorted_y = y[order]
    neg_count=count_pair=0
    i=0

    while i<len(y):
        j=i+1
        while j<len(y) and sorted_pred[j]==sorted_pred[i]:
            j+=1

        group_y=sorted_y[i:j]
        group_pos=sum(group_y==1)
        group_neg=sum(group_y==0)

        count_pair+=0.5*(group_pos*group_neg)
        count_pair+=neg_count*group_pos

        neg_count+=group_neg
        i=j
    return count_pair/(n_pos*n_neg)
print(auc(y, pred))