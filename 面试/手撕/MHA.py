import torch
import torch.nn as nn
import numpy as np


# class ScaleDotProductAttention(nn.Module):
#     def  __init__(self,scale):
#         super().__init__()
#         self.scale=scale
#         self.softmax = nn.Softmax(dim=2)
#     def forward(self,q,k,v,mask=None):
#         u=torch.bmm(q,k.transpose(1,2))
#         u=u/self.scale
#
#         if mask is not None:
#             u=u.masked_fill(mask,-np.inf)
#
#         attn=self.softmax(u)
#         output=torch.bmm(attn,v)
#         return attn,output
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self,n_head,d_k_,d_v_,d_k,d_v,d_o):
#         """
#         :param n_head: 头数
#         :param d_k_: 输入K的维度
#         :param d_v_: 输入V的维度
#         :param d_k: K映射后的维度
#         :param d_v: V映射后的维度
#         :param d_o: 最后全连接层输出的维度
#         """
#         super().__init__()
#         self.n_head=n_head
#         self.d_k=d_k
#         self.d_v=d_v
#
#         # 将所有head的参数拼在一起计算
#         self.fc_q=nn.Linear(d_k_,n_head*d_k)
#         self.fc_k=nn.Linear(d_k_,n_head*d_k)
#         self.fc_v=nn.Linear(d_v_,n_head*d_v)
#
#         self.attention=ScaleDotProductAttention(scale=np.power(d_k,0.5))
#
#         self.fc_o=nn.Linear(n_head*d_v,d_o)
#     def forward(self,q,k,v,mask=None):
#         '''
#         q: (batch, n_q, d_k_)
#         k: (batch, n_k, d_k_)
#         v: (batch, n_v, d_v_)
#         '''
#         n_head,d_q,d_k,d_v=self.n_head,self.d_k,self.d_k,self.d_v
#
#         batch,n_q,d_q_=q.size()
#         batch,n_k,d_k_=k.size()
#         batch,n_v,d_v_=v.size()
#
#         '''
#         q: (batch, n_q, n_head*d_k)
#         k: (batch, n_k, n_head*d_k)
#         v: (batch, n_v, n_head*d_v)
#         '''
#         q=self.fc_q(q)
#         k=self.fc_k(k)
#         v=self.fc_v(v)
#
#         # view拆为(batch, n_q, n_head，d_k)，permute后变为(batch,head,n_q,d_k),view->(head*batch,n_q,d_k)可以用attention计算所有的head
#         # view不改变数据的排列顺序，只是重新解释一块内存，要求tensor必须是contiguous
#         q=q.view(batch,n_q,n_head,d_q).permute(0,2,1,3).contiguous().view(-1,n_q,d_q)
#         k=k.view(batch,n_k,n_head,d_k).permute(0,2,1,3).contiguous().view(-1,n_k,d_k)
#         v=v.view(batch,n_v,n_head,d_v).permute(0,2,1,3).contiguous().view(-1,n_v,d_v)
#
#         # mask (batch,n_q,d_k)->(head*batch,n_q,d_k)
#         if mask is not None:
#             mask=mask.repeat_interleave(n_head,dim=0)
#         attn,output=self.attention(q,k,v,mask=mask)
#
#         # 拼回来
#         output=output.view(batch,self.n_head,n_q,d_v).permute(0,2,1,3).contiguous().view(batch,n_q,-1)
#         output=self.fc_o(output)
#         return attn,output
#
# def softmax(x,dim):
#     x_max=torch.max(x,dim=dim,keepdim=True).values
#     x_exp=torch.exp(x-x_max) # 减去最大值保证数值稳定
#     return x_exp/torch.sum(x_exp,dim=dim,keepdim=True)
#
#
# n_q,n_k,n_v=2,4,4
# d_q_,d_k_,d_v_=128,128,64
# batch=3
#
# q=torch.randn(batch,n_q,d_q_)
# k=torch.randn(batch,n_k,d_k_)
# v=torch.randn(batch,n_v,d_v_)
# mask=torch.zeros(batch,n_q,n_v).bool()
# mha=MultiHeadAttention(n_head=8,d_k_=128,d_v_=64,d_k=256,d_v=128,d_o=128)
#
# attn,output=mha(q,k,v,mask)
# print(attn.size())
# print(output.size())


def DotScaleAttention(q,k,mask):
    weight=torch.matmul(q,k.transpose(1,2))/torch.sqrt(torch.tensor(k.shape[-1]))
    if mask is not None:
        weight.masked_fill_(mask,-np.inf)
    return torch.softmax(weight,-1)

class MHA(nn.Module):
    def __init__(self,q_size,k_size,v_size,q_dim,k_dim,v_dim,o_dim,n_heads):
        super(MHA,self).__init__()

        self.n_heads=n_heads
        self.q_size=q_size
        self.k_size=k_size
        self.v_size=v_size
        self.q_dim=q_dim
        self.k_dim=k_dim
        self.v_dim=v_dim

        self.W_q=nn.Linear(q_size,q_dim*n_heads)
        self.W_k=nn.Linear(k_size,k_dim*n_heads)
        self.W_v=nn.Linear(v_size,v_dim*n_heads)

        self.o=nn.Linear(v_dim*n_heads,o_dim)
    def forward(self,q,k,v,mask):
        Q,K,V=self.W_q(q),self.W_k(k),self.W_v(v)

        q_len,k_len,v_len=Q.shape[1],K.shape[1],V.shape[1]

        B=Q.shape[0]
        n_heads=self.n_heads
        q_dim=self.q_dim
        k_dim=self.k_dim
        v_dim=self.v_dim

        Q=Q.view(B,q_len,n_heads,q_dim).permute(0,2,1,3).contiguous().view(-1,q_len,q_dim)
        K=K.view(B,k_len,n_heads,k_dim).permute(0,2,1,3).contiguous().view(-1,k_len,k_dim)
        V=V.view(B,v_len,n_heads,v_dim).permute(0,2,1,3).contiguous().view(-1,v_len,v_dim)

        if mask is not None:
            mask=mask.repeat_interleave(n_heads,dim=0)
        attn=DotScaleAttention(Q,K,mask)

        h=torch.matmul(attn,V)
        h=h.view(B,n_heads,q_len,v_dim).permute(0,2,1,3).view(B,q_len,-1)
        o=self.o(h)
        return o
def softmax(x):
    x_max=torch.max(x,dim=-1,keepdim=True)[0]
    x-=x_max
    x_exp=torch.exp(x)
    return x_exp/torch.sum(x_exp,dim=-1,keepdim=True)