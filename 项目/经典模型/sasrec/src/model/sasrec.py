# import torch
# from torch import nn
# from src.model.embeding_layer import EmbeddingLayer
# from src.model.self_attn_block import SelfAttnBlock
#
#
# class SASRec(nn.Module):
#     def __init__(self, num_items, num_block, hidden_dim, max_seq_len, dropout, device):
#         super().__init__()
#
#         self.device = device
#         self.max_seq_len = max_seq_len
#         self.hidden_dim = hidden_dim
#
#         self.embedding_layer = EmbeddingLayer(
#             num_items=num_items,
#             embed_dim=hidden_dim,
#             max_seq_len=max_seq_len
#         )
#
#         self.self_attn_blocks = nn.ModuleList([
#             SelfAttnBlock(
#                 max_seq_len=max_seq_len,
#                 embed_dim=hidden_dim,
#                 dropout=dropout
#             )
#             for _ in range(num_block)
#         ])
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(hidden_dim)
#
#     def get_padding_mask(self, seqs):
#         return (seqs != 0)
#
#     def forward(
#         self,
#         input_seqs: torch.Tensor,
#         item_idxs: torch.Tensor = None,
#         positive_seqs: torch.Tensor = None,
#         negative_seqs: torch.Tensor = None,
#     ):
#         padding_mask = self.get_padding_mask(input_seqs).to(input_seqs.device)   # [B, L]
#
#         input_embs = self.dropout(self.embedding_layer(input_seqs))               # [B, L, D]
#         input_embs = input_embs * padding_mask.unsqueeze(-1)
#
#         attn_output = input_embs
#         for block in self.self_attn_blocks:
#             attn_output = block(x=attn_output, padding_mask=~padding_mask)
#
#         attn_output = self.layer_norm(attn_output)                                # [B, L, D]
#
#         if item_idxs is not None:
#             # item_idxs: [B, C]
#             item_embs = self.embedding_layer.item_embeddings(item_idxs)           # [B, C, D]
#             logits = torch.matmul(attn_output, item_embs.transpose(1, 2))         # [B, L, C]
#             logits = logits[:, -1, :]                                             # [B, C]
#             return logits
#
#         if positive_seqs is not None and negative_seqs is not None:
#             positive_embs = self.dropout(self.embedding_layer.item_embeddings(positive_seqs))  # [B, L, D]
#             negative_embs = self.dropout(self.embedding_layer.item_embeddings(negative_seqs))  # [B, L, D]
#
#             positive_logits = (attn_output * positive_embs).sum(dim=-1)           # [B, L]
#             negative_logits = (attn_output * negative_embs).sum(dim=-1)           # [B, L]
#
#             return positive_logits, negative_logits
#
#         raise ValueError("item_idxs 或 positive_seqs/negative_seqs 必须提供一组")

import torch
from torch import nn

from src.model.embeding_layer import EmbeddingLayer
from src.model.self_attn_block import SelfAttnBlock



class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_blocks: int,
        hidden_dim: int,
        max_seq_len: int,
        dropout_p: float,
        share_item_emb: bool,
        device: str,
    ) -> None:
        super().__init__()

        self.device = device

        self.embedding_layer = EmbeddingLayer(
            num_items=num_items,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )
        self_attn_blocks = [
            SelfAttnBlock(
                max_seq_len=max_seq_len,
                hidden_dim=hidden_dim,
                dropout_p=dropout_p,
                device=device,
            )
            for _ in range(num_blocks)
        ]
        self.self_attn_blocks = nn.Sequential(*self_attn_blocks)

        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)

    def get_padding_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        is_padding = torch.tensor(seqs == 0, dtype=torch.bool)
        padding_mask = ~is_padding

        return padding_mask

    def forward(
        self,
        input_seqs: torch.Tensor,
        item_idxs: torch.Tensor = None,
        positive_seqs: torch.Tensor = None,
        negative_seqs: torch.Tensor = None,
    ) -> torch.Tensor:
        padding_mask = self.get_padding_mask(seqs=input_seqs).to(self.device)

        input_embs = self.dropout(self.embedding_layer(input_seqs))
        input_embs *= padding_mask.unsqueeze(-1)

        # For loop because nn.Sequential can't handle multiple inputs.
        attn_output = input_embs
        for block in self.self_attn_blocks:
            attn_output = block(x=attn_output, padding_mask=padding_mask)
        attn_output = self.layer_norm(attn_output)

        if item_idxs is not None:  # Inference.
            item_embs = self.embedding_layer.item_emb_matrix(item_idxs)
            logits = attn_output @ item_embs.transpose(2, 1)
            logits = logits[:, -1, :]
            outputs = (logits,)
        elif (positive_seqs is not None) and (negative_seqs is not None):  # Training.
            positive_embs = self.dropout(self.embedding_layer(positive_seqs))
            negative_embs = self.dropout(self.embedding_layer(negative_seqs))

            positive_logits = (attn_output * positive_embs).sum(dim=-1)
            negative_logits = (attn_output * negative_embs).sum(dim=-1)

            outputs = (positive_logits,)
            outputs += (negative_logits,)

        return outputs