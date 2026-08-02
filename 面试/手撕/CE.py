import numpy as np

# 真实标签
y = np.array([1, 0, 1])

# 模型预测概率
y_pred = np.array([0.9, 0.2, 0.8])

# eps=1e-8
# def BCEloss(y,y_pred):
#     y_pred=np.clip(y_pred,eps,1-eps) # 截断
#     loss=y*np.log(y_pred)+(1-y)*np.log(1-y_pred)
#     loss=np.sum(loss)
#     return -loss/y.shape[0]
#
# def CE(y,y_pred):
#     logits=y_pred-np.max(y_pred,axis=1,keepdims=True) # 防止数值太大溢出
#     probs=np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims=True) # [N,C]/[N,C] keepdim保留维度做广播，否则[N,C]/[N,]会报错
#     log_probs=-np.log(probs[np.arange(len(y)),y]+eps)
#     return np.mean(log_probs)

eps=1e-8

def BCE(y,y_pred):
    y_pred=np.clip(y_pred,eps,1-eps)
    return -np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))/y.shape[0]
print(BCE(y,y_pred))


def CE(y,y_pred):
    y_max=np.max(y_pred,axis=1,keepdims=True)
    y_pred-=y_max
    y_exp=np.exp(y_pred)
    probs=y_exp/np.sum(y_exp,axis=1,keepdims=True)
    logits=np.log(probs[np.arange(y_pred.shape[0]),y]+eps)
    return -np.mean(logits)

y=[0,2]
y_pred=[
 [3.0,1.0,0.2],
 [0.1,0.2,4.0]
]

print(CE(y,y_pred))
