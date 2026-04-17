import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class Embedding_layer(nn.Module):
    def __init__(
        self,
        book_vocab_size: int,
        word_vocab_size: int,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.uid_embeddings = nn.Embedding(2 ** 18,embed_dim, padding_idx=0)
        self.book_embeddings = nn.Embedding(book_vocab_size, embed_dim, padding_idx=0)
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim, padding_idx=0)

        self.register_bucket_embed = nn.Embedding(5, 16, padding_idx=0)

        self.cat_embeddings = nn.Linear(2 * embed_dim, embed_dim)

    @staticmethod
    def masked_mean(x, ids):
        mask = (ids != 0).float().unsqueeze(-1)
        x = x * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return x.sum(dim=1) / denom

    def forward(self, inputs, flag="impression"):
        emb_dict = {}

        if "uid" in inputs:
            uid = inputs["uid"]
            if uid.dim() == 2:
                uid = uid.squeeze(1)
            emb_dict["uid"] = self.uid_embeddings(uid)

        if "likemarks" in inputs:
            word_emb = self.word_embeddings(inputs["likemarks"])
            emb_dict["likemarks"] = self.masked_mean(word_emb, inputs["likemarks"])

        if "searchwords" in inputs:
            word_emb = self.word_embeddings(inputs["searchwords"])
            emb_dict["searchwords"] = self.masked_mean(word_emb, inputs["searchwords"])

        if "read_books" in inputs:
            book_emb = self.book_embeddings(inputs["read_books"])
            emb_dict["read_books"] = self.masked_mean(book_emb, inputs["read_books"])

        if "register_time" in inputs:
            reg_emb = self.register_bucket_embed(inputs["register_time"])
            if reg_emb.dim() == 3:
                reg_emb = reg_emb.squeeze(1)
            emb_dict["register_time"] = reg_emb

        if "read_search_keywords_books_day_30" in inputs:
            seq = inputs["read_search_keywords_books_day_30"]
            book_ids = seq["book"]
            word_ids = seq["word"]

            book_emb = self.book_embeddings(book_ids)
            word_emb = self.word_embeddings(word_ids)

            x = torch.cat([book_emb, word_emb], dim=-1)
            x = self.cat_embeddings(x)

            valid = ((book_ids != 0) | (word_ids != 0)).long()
            emb_dict["read_search_keywords_books_day_30"] = self.masked_mean(x, valid)

        # 统一 item 特征名，不再把 impression_* / negative_* 直接往后传
        item_feature_map = {
            "bookid": f"{flag}_bookid",
            "bookmarks": f"{flag}_bookmarks",
            "bookname": f"{flag}_bookname",
            "tag": f"{flag}_tag",
            "bookwords": f"{flag}_bookwords",
            "bookinfo": f"{flag}_bookinfo",
        }

        for std_name, input_name in item_feature_map.items():
            if input_name not in inputs:
                continue

            if std_name == "bookid":
                x = self.book_embeddings(inputs[input_name])
                if x.dim() == 3:
                    x = x.squeeze(1)
                emb_dict[std_name] = x
            else:
                ids = inputs[input_name]
                x = self.word_embeddings(ids)
                emb_dict[std_name] = self.masked_mean(x, ids)

        return emb_dict

    def load_pretrained_weights(
        self,
        word_weights: Optional[torch.Tensor] = None,
        book_weights: Optional[torch.Tensor] = None,
    ):
        with torch.no_grad():
            if word_weights is not None:
                n = min(self.word_embeddings.weight.size(0), word_weights.size(0))
                d = min(self.word_embeddings.weight.size(1), word_weights.size(1))
                self.word_embeddings.weight[:n, :d].copy_(word_weights[:n, :d])

            if book_weights is not None:
                n = min(self.book_embeddings.weight.size(0), book_weights.size(0))
                d = min(self.book_embeddings.weight.size(1), book_weights.size(1))
                self.book_embeddings.weight[:n, :d].copy_(book_weights[:n, :d])

            if self.word_embeddings.padding_idx is not None:
                self.word_embeddings.weight[self.word_embeddings.padding_idx].zero_()
            if self.book_embeddings.padding_idx is not None:
                self.book_embeddings.weight[self.book_embeddings.padding_idx].zero_()


class SemanticTokenizer(nn.Module):
    """
    严格按论文思路：
    语义分组 -> 组内拼接 -> 组间concat成 e_input -> 切成T段 -> shared Proj
    """
    def __init__(
        self,
        input_dims: Dict[str, int],
        groups: List[List[str]],
        token_dim: int,
        num_tokens: int,
    ):
        """
        input_dims = {
            "uid":  embed_dim,
            "register_time": 16,
            "likemarks": embed_dim,
            "searchwords": embed_dim,
            "read_books": embed_dim,
            "read_search_keywords_books_day_30": embed_dim,

            "impression_bookid": embed_dim,
            "impression_bookmarks": embed_dim,
            "impression_bookname": embed_dim,
            "impression_tag": embed_dim,
            "impression_bookwords": embed_dim,
            "impression_bookinfo": embed_dim,

            "negative_bookid": embed_dim,
            "negative_bookmarks": embed_dim,
            "negative_bookname": embed_dim,
            "negative_tag": embed_dim,
            "negative_bookwords": embed_dim,
            "negative_bookinfo": embed_dim,
        }

        groups
                ["uid", "register_time"], 用户特征
                ["likemarks", "searchwords"],序列特征
                ["read_books", "read_search_keywords_books_day_30"],行为特征
                ["impression_bookid", "impression_bookmarks"], # 书籍特征
                ["impression_bookname", "impression_tag", "impression_bookwords", "impression_bookinfo"]

        """
        super().__init__()

        self.groups = groups
        self.input_dims = input_dims
        self.token_dim = token_dim
        self.num_tokens = num_tokens

        self.group_dims = [sum(input_dims[f] for f in g) for g in groups]
        self.e_input_dim = sum(self.group_dims) # 所有特征维度的总和

        self.slice_dim = math.ceil(self.e_input_dim / self.num_tokens) # 按照slice_dim切分
        self.target_dim = self.slice_dim * self.num_tokens # 切分后的各个token维度总和

        self.proj = nn.Linear(self.slice_dim, self.token_dim) # token投影到统一的维度

    def forward(self, emb_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        first_tensor = next(iter(emb_dict.values()))
        B = first_tensor.size(0)
        device = first_tensor.device
        dtype = first_tensor.dtype

        group_embs = []
        for g in self.groups:
            xs = []
            for name in g:
                feat_dim = self.input_dims[name]
                if name in emb_dict:
                    xs.append(emb_dict[name])
                else:
                    xs.append(torch.zeros(B, feat_dim, device=device, dtype=dtype))
            group_embs.append(torch.cat(xs, dim=-1)) # 按语义分组拼接

        e_input = torch.cat(group_embs, dim=-1)  # [B, e_input_dim] 组间拼接

        # embedding不能刚好切分，这里补齐
        if e_input.size(1) < self.target_dim:
            pad = torch.zeros(B, self.target_dim - e_input.size(1), device=device, dtype=dtype)
            e_input = torch.cat([e_input, pad], dim=-1)

        x = e_input.view(B, self.num_tokens, self.slice_dim)  # [B, T, d] 切分
        x = self.proj(x)  # [B, T, D]
        return x


class TokenMixing(nn.Module):
    """
    Parameter-free token mixing
    严格对齐你给的 TensorFlow 版本：
    - 无可学习参数
    - 要求 num_heads == num_tokens
    - 通过 head 维和 token 维交换实现 mixing
    """

    def __init__(self, num_tokens: int, token_dim: int, num_heads: int = None, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = float(dropout)

        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")
        if self.token_dim % self.num_heads != 0:
            raise ValueError(
                f"token_dim must be divisible by num_heads, got token_dim={self.token_dim}, num_heads={self.num_heads}"
            )

        self.head_dim = self.token_dim // self.num_heads
        self.drop = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, D], but got shape={tuple(x.shape)}")

        B, T, D = x.shape
        if T != self.num_tokens:
            raise ValueError(f"token count mismatch: expected {self.num_tokens}, got {T}")
        if D != self.token_dim:
            raise ValueError(f"token dim mismatch: expected {self.token_dim}, got {D}")

        H = self.num_heads
        d = self.head_dim

        # [B, T, D] -> [B, T, H, d]
        x = x.view(B, T, H, d)

        # [B, T, H, d] -> [B, H, T, d]
        x = x.transpose(1, 2).contiguous()

        # [B, H, T, d] -> [B, H, T*d]
        # 每个 head 聚合所有 token 的对应子空间，形成一个新的 mixed token
        x = x.view(B, H, T * d)

        # [B, H, T*d] -> [B, T, D]
        x = x.view(B, T, D)

        x = self.drop(x)
        return x

class PerTokenFFN(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.hidden_dim = int(token_dim * hidden_ratio)
        self.dropout = dropout

        self.W1 = nn.Parameter(torch.empty(num_tokens, token_dim, self.hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, self.hidden_dim))
        self.W2 = nn.Parameter(torch.empty(num_tokens, self.hidden_dim, token_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, token_dim))

        nn.init.kaiming_uniform_(self.W1, a=0, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, a=0, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = torch.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        y = torch.einsum("bth,thd->btd", h, self.W2) + self.b2
        y = F.dropout(y, p=self.dropout, training=self.training)
        return y


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        hidden_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_mixing = TokenMixing(num_tokens=num_tokens, token_dim=token_dim,num_heads=num_heads)
        self.norm1 = nn.LayerNorm(token_dim)

        self.pffn = PerTokenFFN(
            num_tokens=num_tokens,
            token_dim=token_dim,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.token_mixing(x))
        x = self.norm2(x + self.pffn(x))
        return x


class RankMixer(nn.Module):
    def __init__(
        self,
        book_vocab_size: int,
        word_vocab_size: int,
        embed_dim: int = 64,
        token_dim: int = 128,
        num_tokens: int = 8,
        num_layers: int = 2,
        hidden_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_layer = Embedding_layer(
            book_vocab_size=book_vocab_size,
            word_vocab_size=word_vocab_size,
            embed_dim=embed_dim,
        )

        self.input_dims = {
            "uid": embed_dim,
            "register_time": 16,
            "likemarks": embed_dim,
            "searchwords": embed_dim,
            "read_books": embed_dim,
            "read_search_keywords_books_day_30": embed_dim,

            "bookid": embed_dim,
            "bookmarks": embed_dim,
            "bookname": embed_dim,
            "tag": embed_dim,
            "bookwords": embed_dim,
            "bookinfo": embed_dim,
        }

        self.groups = [
            ["uid", "register_time"],
            ["likemarks", "searchwords"],
            ["read_books", "read_search_keywords_books_day_30"],
            ["bookid", "bookmarks"],
            ["bookname", "tag", "bookwords", "bookinfo"],
        ]

        self.tokenizer = SemanticTokenizer(
            input_dims=self.input_dims,
            groups=self.groups,
            token_dim=token_dim,
            num_tokens=num_tokens,
        )

        self.blocks = nn.ModuleList([
            RankMixerBlock(
                num_tokens=num_tokens,
                token_dim=token_dim,
                num_heads=num_tokens,
                hidden_ratio=hidden_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    def load_pretrained_weights(
        self,
        word_weights: Optional[torch.Tensor] = None,
        book_weights: Optional[torch.Tensor] = None,
    ):
        self.embedding_layer.load_pretrained_weights(
            word_weights=word_weights,
            book_weights=book_weights,
        )

    def forward(self, inputs: Dict[str, torch.Tensor], flag: str = "impression") -> torch.Tensor:
        if flag not in {"impression", "negative"}:
            raise ValueError(f"不支持的 flag: {flag}")

        emb_dict = self.embedding_layer(inputs, flag=flag)
        x = self.tokenizer(emb_dict)

        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=1)
        pooled = self.out_norm(pooled)
        score = self.head(pooled).squeeze(-1)
        return score


    @torch.no_grad()
    def predict(self, inputs: Dict[str, torch.Tensor], flag: str = "impression") -> Dict[str, torch.Tensor]:
        score = self.forward(inputs, flag=flag)
        return {
            "books_ranking": score,
            "probs": torch.sigmoid(score),
        }