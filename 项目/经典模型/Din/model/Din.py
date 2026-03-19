import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Dice(nn.Module):
    """
    Dice激活函数，Prelu的泛化
    """
    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim, eps=eps, affine=False)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] or [B, T, D]
        original_shape = x.shape
        if x.dim() == 3:
            b, t, d = x.shape
            x_2d = x.reshape(-1, d)
            norm_x = self.bn(x_2d).reshape(b, t, d)
        else:
            norm_x = self.bn(x)

        p = torch.sigmoid(norm_x)
        return p * x + (1 - p) * self.alpha * x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        use_dice: bool = True,
        output_dim: int | None = None,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_dice:
                layers.append(Dice(h))
            else:
                layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DINAttention(nn.Module):
    """
    DIN的局部激活单元：
    对历史序列中的每个item embedding，与target item embedding做交互，
    学出每个历史item对当前target的重要性。
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.0,
        use_dice: bool = True,
    ):
        super().__init__()
        # [hist, target, hist-target, hist*target]
        input_dim = embed_dim * 4
        self.att_mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_dice=use_dice,
            output_dim=1,
        )

    def forward(
        self,
        query: torch.Tensor,      # [B, D] 候选物品
        keys: torch.Tensor,       # [B, T, D] 用户的点击历史
        mask: torch.Tensor,       # [B, T]  1有效 0 padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = keys.shape

        query_expand = query.unsqueeze(1).expand(-1, t, -1)  # [B, T, D]

        att_input = torch.cat(
            [
                keys,
                query_expand,
                keys - query_expand,
                keys * query_expand,
            ],
            dim=-1,
        )  # [B, T, 4D]

        att_score = self.att_mlp(att_input).squeeze(-1)  # [B, T]

        # padding位置置成极小
        att_score = att_score.masked_fill(mask == 0, -1e9)

        att_weight = torch.softmax(att_score, dim=-1)  # [B, T]
        att_weight = att_weight * mask.float()

        # 防止全0 mask时 nan
        denom = att_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        att_weight = att_weight / denom

        user_interest = torch.bmm(att_weight.unsqueeze(1), keys).squeeze(1)  # [B, D]

        return user_interest, att_weight


class DIN(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.1,
        use_dice: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # 1. 定义 Embedding 层
        self.item_emb = nn.Embedding(cfg["item_count"], embed_dim, padding_idx=0)
        self.cate_emb = nn.Embedding(cfg["cate_count"], embed_dim, padding_idx=0)
        self.user_emb = nn.Embedding(cfg["user_count"], embed_dim, padding_idx=0)

        # item表征 = item_emb + cate_pool_emb -> 2D
        self.attention = DINAttention(
            embed_dim=embed_dim * 2,
            hidden_dims=[128, 64],
            dropout=dropout,
            use_dice=use_dice,
        )

        # user_emb[D] + user_interest[2D] + target_item[2D] = 5D
        final_input_dim = embed_dim + embed_dim * 2 + embed_dim * 2

        self.final_mlp = MLP(
            input_dim=final_input_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout,
            use_dice=use_dice,
            output_dim=1,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
    def _pool_cate(self, cate_ids: torch.Tensor) -> torch.Tensor:
        """
        cate_ids:
            [B, C] or [B, T, C]
        return:
            [B, D] or [B, T, D]
        """
        cate_emb = self.cate_emb(cate_ids)

        mask = (cate_ids != self.padding_idx).unsqueeze(-1).float()
        cate_sum = (cate_emb * mask).sum(dim=-2)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        cate_pool = cate_sum / denom
        return cate_pool

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ):
        uid = inputs["userId"]  # [B]
        movie_id = inputs["movie_id"]  # [B]
        movie_cate = inputs["movie_cate_id_list"]  # [B, C]
        hist_movie = inputs["hist_movie_id_list"]  # [B, T]
        hist_cate = inputs["hist_movie_cate_id_list"]  # [B, T, C]

        user_emb = self.user_emb(uid)  # [B, D]

        movie_id_emb = self.item_emb(movie_id)  # [B, D]
        movie_cate_emb = self._pool_cate(movie_cate)  # [B, D]
        target_emb = torch.cat([movie_id_emb, movie_cate_emb], dim=-1)  # [B, 2D]

        hist_movie_emb = self.item_emb(hist_movie)  # [B, T, D]
        hist_cate_emb = self._pool_cate(hist_cate)  # [B, T, D]
        hist_item_emb = torch.cat([hist_movie_emb, hist_cate_emb], dim=-1)  # [B, T, 2D]

        hist_mask = (hist_movie != self.padding_idx)  # [B, T]

        user_interest, att_weight = self.attention(
            query=target_emb,
            keys=hist_item_emb,
            mask=hist_mask,
        )

        final_input = torch.cat([user_emb, user_interest, target_emb], dim=-1)  # [B, 5D]
        logit = self.final_mlp(final_input).squeeze(-1)  # [B]

        return {
            "logit": logit,
            "att_weight": att_weight,
        }