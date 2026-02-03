import numpy as np

y    = np.array([1,0,1,0,0])
pred = np.array([1,1,0,0,0])

def nDCG(y, y_pred,k=10):
    k=min(k,len(y))
    ideal =sorted(y)[::-1][:k]

    order=np.argsort(y_pred)[::-1] # 按数值排序但是显示坐标
    y_sorted=y[order][:k]

    denominator=np.log2(np.arange(2,k+2))

    iDCG=np.sum(ideal/denominator)
    DCG=np.sum(y_sorted/denominator)

    return DCG/iDCG


print(nDCG(y, pred))