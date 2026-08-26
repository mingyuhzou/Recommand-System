import numpy as np

# 真实标签
y = np.array([1, 0, 1])

# 模型预测概率
y_pred = np.array([0.9, 0.2, 0.8])

eps=1e-8

# def BCE(y,y_pred):
#     y_pred=np.clip(y_pred,eps,1-eps) # 防止出现0
#     return -1/len(y)*np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
#
# def CE(y,y_pred):
#     y_max=np.max(y_pred,axis=-1,keepdims=True) # 防止指数溢出，softmax中所有logits减去同一个常数不改变结果
#     y_pred-=y_max
#     y_exp=np.exp(y_pred)
#     probs=y_exp/np.sum(y_exp,axis=-1,keepdims=True)
#     logits=np.log(probs[np.arange(len(y)),y]+eps)
#     return -np.mean(logits)

def BCE(y,y_pred):
    y_pred=np.clip(y_pred,eps,1-eps)
    return -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))

print(BCE(y,y_pred))


def CE(y,y_pred):
    y_pred=y_pred-np.max(y_pred,axis=-1,keepdims=True)
    y_exp=np.exp(y_pred)
    probs=y_exp/np.sum(y_exp,axis=-1,keepdims=True)
    logits=np.log(probs[np.arange(len(y)),y])
    return -np.mean(logits)

y=np.array([0,2])
y_pred=np.array([
 [3.0,1.0,0.2],
 [0.1,0.2,4.0]
])

print(CE(y,y_pred))
