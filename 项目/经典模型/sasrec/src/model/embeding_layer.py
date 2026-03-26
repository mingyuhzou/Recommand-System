# import math
# import torch
# import torch.nn as nn
#
#
# class EmbeddingLayer(nn.Module):
#     def __init__(self, num_items, embed_dim, max_seq_len):
#         super().__init__()
#
#         self.embed_dim = embed_dim
#
#         # padding_idx=0 很关键
#         self.item_embeddings = nn.Embedding(
#             num_items + 1,
#             embed_dim,
#             padding_idx=0
#         )
#
#         self.positional_emb = nn.Embedding(max_seq_len, embed_dim)
#
#     def forward(self, x):
#         """
#         x: [B, L]
#         """
#         device = x.device
#         batch_size, seq_len = x.shape
#
#         # item embedding
#         x = self.item_embeddings(x)                     # [B, L, D]
#         x = x * math.sqrt(self.embed_dim)
#
#         # position embedding（标准写法）
#         positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
#         positions = positions.expand(batch_size, seq_len)              # [B, L]
#
#         positional_embs = self.positional_emb(positions)               # [B, L, D]
#
#         x = x + positional_embs
#
#         return x

import math

import torch
from torch import nn


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.item_emb_matrix = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=hidden_dim,
        )
        self.positional_emb = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=hidden_dim,
        )

    def forward(self, x):
        x = self.item_emb_matrix(x)
        x *= math.sqrt(self.hidden_dim)

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device.type

        seq_len_range = torch.tensor(range(seq_len))
        positions = torch.tile(input=seq_len_range, dims=(batch_size, 1))
        positions = positions.to(device)

        positional_embs = self.positional_emb(positions)
        x += positional_embs

        return x