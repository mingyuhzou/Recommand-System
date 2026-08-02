import torch

def additiveAttention(q,k,v,W_q,W_k,w_v):
    '''

    :param q: (B,seq_len,q_size)
    :param k: (B,seq_len,k_size)
    :param v: (B,seq_len,v_size)
    :param W_q: (q_size,hidden_size)
    :param W_k: (k_size,hidden_size)
    :param w_v: (hidden_size,1)
    :return:
    '''
    q_=torch.matmul(q, W_q)
    k_=torch.matmul(k, W_k)
    h=torch.tanh(q_.unsqueeze(2)+k_.unsqueeze(1)) # (B,seq_len,seq_len,hidden_size)

    weight=torch.matmul(h,w_v).squeeze(-1) # (B,seq_len,seq_len)

    weight=torch.softmax(weight,dim=-1)
    return weight@v # (B,seq_len,v_size)
