import np
import torch
import torch.nn as nn
from mha import MultiHeadAttention

class SelfAttention(nn.Module):
    def __init__(self, n_head,d_k,d_v,d_x,d_o):
        self.wq=nn.Parameter(torch.Tensor(d_x,d_k))
        self.wk=nn.Parameter(torch.Tensor(d_x,d_k))
        self.wv=nn.Parameter(torch.Tensor(d_x,d_v))

        self.mha=MultiHeadAttention(n_head,d_k_=d_k,d_v_=d_v,d_k=d_k,d_v=d_v,d_o=d_o)
        self.init_weights()
    def init_weights(self):
        for param in self.parameters():
            stdv=1./np.power(param.size(-1),0.5)
            param.data.uniform_(-stdv,stdv)
    def forward(self,x,mask=None):
        q,k,v=torch.matmul(x,self.wq),torch.matmul(x,self.wk),torch.matmul(x,self.wv)

        attn,output=self.mha(q,k,v,mask)
        return attn,output
