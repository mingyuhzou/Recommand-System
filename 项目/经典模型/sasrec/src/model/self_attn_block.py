# import torch
# import torch.nn as nn
# from src.model.pffn import Pffn
#
#
# class SelfAttnBlock(nn.Module):
#     def __init__(self, max_seq_len, embed_dim, dropout):
#         super().__init__()
#
#         self.max_seq_len = max_seq_len
#         self.embed_dim = embed_dim
#
#         self.attn_norm = nn.LayerNorm(embed_dim)
#         self.ffn_norm = nn.LayerNorm(embed_dim)
#
#         self.attn_dropout = nn.Dropout(dropout)
#         self.ffn_dropout = nn.Dropout(dropout)
#
#         self.attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=1,
#             dropout=dropout,
#             batch_first=True
#         )
#
#         self.pffn = Pffn(embed_dim)
#
#     def forward(self, x, padding_mask=None):
#         """
#         x: [B, L, D]
#         padding_mask: [B, L]
#                       True 表示该位置是 padding，需要被 mask
#         """
#         batch_size, seq_len, _ = x.shape
#         device = x.device
#
#         # causal mask: True 表示不允许注意
#         causal_mask = torch.triu(
#             torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
#             diagonal=1
#         )  # [L, L]
#
#         # ===== Self-Attention 子层 =====
#         residual = x
#         x_norm = self.attn_norm(x)
#
#         attn_output, _ = self.attn(
#             query=x_norm,
#             key=x_norm,
#             value=x_norm,
#             attn_mask=causal_mask,
#             key_padding_mask=padding_mask,
#             need_weights=False
#         )  # [B, L, D]
#
#         x = residual + self.attn_dropout(attn_output)
#
#         # ===== FFN 子层 =====
#         residual = x
#         x_norm = self.ffn_norm(x)
#
#         ffn_output = self.pffn(x_norm)  # [B, L, D]
#         x = residual + self.ffn_dropout(ffn_output)
#
#         return x


import torch
from torch import nn

from src.model.pffn import PointWiseFFNN


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
        dropout_p: float,
        device: str,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout_p,
            batch_first=True,
        )
        self.ffnn = PointWiseFFNN(hidden_dim=hidden_dim)

    def dropout_layernorm(self, x: torch.Tensor) -> torch.Tensor:
        layer_norm_output = self.layer_norm(x)
        dropout_output = self.dropout(layer_norm_output)

        return dropout_output

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        attention_mask = ~torch.tril(
            torch.ones(size=(seq_len, seq_len), dtype=torch.bool)
        )
        device = x.device.type
        attention_mask = attention_mask.to(device)

        x_attn, _ = self.self_attn(
            key=self.layer_norm(x),
            query=x,
            value=x,
            attn_mask=attention_mask,
        )
        x_attn_output = x + self.dropout_layernorm(x_attn)

        x_ffnn = self.ffnn(x_attn_output)
        x_ffnn_output = x_attn_output + self.dropout_layernorm(x_ffnn)

        output = x_ffnn_output * padding_mask.unsqueeze(-1)
        return output