import torch

def DotProductAttention(q,k,v,W_q,W_k) :
    Q=torch.matmul(q,W_q)
    K=torch.matmul(k,W_k)

    weight=torch.matmul(Q,K.transpose(1,2))/torch.sqrt(q.shape[-1])
    weight=torch.softmax(weight,dim=-1)
    return torch.matmul(weight,v)