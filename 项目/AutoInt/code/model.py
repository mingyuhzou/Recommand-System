import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        embed_dim
    ):
        super().__init__()

        # 离散特征 embedding（padding_idx=0 预留）
        self.user_emb = nn.Embedding(2**18 + 1, embed_dim, padding_idx=0)
        self.movie_emb = nn.Embedding(2**18 + 1, embed_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(2 + 1, embed_dim, padding_idx=0)
        self.age_emb = nn.Embedding(5 + 1, embed_dim, padding_idx=0)
        self.occ_emb = nn.Embedding(10 + 1, embed_dim, padding_idx=0)
        self.zip_emb = nn.Embedding(25 + 1, embed_dim, padding_idx=0)
        self.year_emb = nn.Embedding(25 + 1, embed_dim, padding_idx=0)

        # 多值特征
        self.genre_emb = nn.Embedding(25 + 1, embed_dim, padding_idx=0)

        # 数值特征（timestamp）
        self.timestamp_emb = nn.Parameter(torch.randn(embed_dim))

    def forward(self, inputs):
        """
        inputs:
            user_id: [B]
            movie_id: [B]
            gender: [B]
            age: [B]
            occupation: [B]
            zipcode: [B]
            year: [B]
            genre: [B, 6]
            timestamp: [B]
        """

        # 单值特征
        e_user = self.user_emb(inputs["user_id"])
        e_movie = self.movie_emb(inputs["movie_id"])
        e_gender = self.gender_emb(inputs["gender"])
        e_age = self.age_emb(inputs["age"])
        e_occ = self.occ_emb(inputs["occupation"])
        e_zip = self.zip_emb(inputs["zipcode"])
        e_year = self.year_emb(inputs["year"])

        # 多值特征（mean pooling，忽略padding=0）
        genre_ids = inputs["genre"]               # [B, 6]
        genre_emb = self.genre_emb(genre_ids)     # [B, 6, D]

        mask = (genre_ids != 0).unsqueeze(-1)     # [B, 6, 1]
        genre_sum = (genre_emb * mask).sum(dim=1) # [B, D]
        denom = mask.sum(dim=1).clamp(min=1)      # 防止除0
        e_genre = genre_sum / denom

        # 数值特征
        e_time = inputs["timestamp"].unsqueeze(-1) * self.timestamp_emb  # [B, D]

        # 拼接
        out = torch.cat([
            e_user,
            e_movie,
            e_gender,
            e_age,
            e_occ,
            e_zip,
            e_year,
            e_genre,
            e_time
        ], dim=1)

        return out

class InteractionLayer(nn.Module):
    def __init__(self, embed_dim,attn_dim):
        super().__init__()

        self.Wq = nn.Linear(embed_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, attn_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, attn_dim, bias=False)

    def forward(self,x):
        # x: [B, M, D]

        q=self.Wq(x)
        k=self.Wk(x)
        v=self.Wv(x)

        attn_score=torch.bmm(q,k.transpose(1,2))
        alpha = F.softmax(attn_score, dim=-1)             # [B, M, M]

        # 聚合
        out = torch.matmul(alpha, v)                # [B, M, D']

        return out

class MultiHeadInteraction(nn.Module):
    def __init__(self, embed_dim, attn_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            InteractionLayer(embed_dim, attn_dim)
            for _ in range(num_heads)
        ])

        # 残差投影（解决维度不一致）
        out_dim = attn_dim * num_heads
        if out_dim != embed_dim:
            self.W_res = nn.Linear(embed_dim, out_dim, bias=False)
        else:
            self.W_res = nn.Identity()

    def forward(self, E):
        # E: [B, M, D]

        # 多头
        outs = [head(E) for head in self.heads]   # list of [B, M, D']

        # concat
        out = torch.cat(outs, dim=-1)             # [B, M, H*D']

        # 残差
        res = self.W_res(E)                       # [B, M, H*D']

        out = F.relu(out + res)

        return out

class AutoInt(nn.Module):
    def __init__(
        self,
        embed_dim,
        attn_dim,
        num_heads,
        hidden_units,
        layers,
        dropout,
    ):
        super().__init__()
        self.embed_layer = EmbeddingLayer(embed_dim)
        self.inter_layer = nn.ModuleList(MultiHeadInteraction(embed_dim, attn_dim, num_heads) for _ in range(layers))




