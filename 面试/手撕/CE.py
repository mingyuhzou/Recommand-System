import numpy as np

eps=1e-8
def BCEloss(y,y_pred):
    y_pred=np.clip(y_pred,eps,1-eps) # 截断
    loss=y*np.log(y_pred)+(1-y)*np.log(1-y_pred)
    loss=np.sum(loss)
    return -loss/y.shape[0]

def CE(y,y_pred):
    logits=y_pred-np.max(y_pred,axis=1,keepdims=True) # 防止数值太大溢出
    probs=np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims=True) # [N,C]/[N,C] keepdim保留维度做广播，否则[N,C]/[N,]会报错
    log_probs=-np.log(probs[np.arange(len(y)),y]+eps)
    return np.mean(log_probs)


