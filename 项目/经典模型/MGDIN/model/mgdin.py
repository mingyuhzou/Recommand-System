from typing import Dict, List, Any, Optional, Tuple
import logging

log = logging.getLogger(__name__)


import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PFFN(nn.Module):
    """
    Per-token Feed-Forward Network
    输入:  [B, M, D]
    输出:  [B, M, D]
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DeferredInteractionLayer(nn.Module):
    """
    单层 Information-Aware Deferred Interaction
    """
    def __init__(self, dim: int, attn_dim: Optional[int] = None, ffn_hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        attn_dim = attn_dim or dim

        self.q = nn.Linear(dim, attn_dim, bias=False)
        self.k = nn.Linear(dim, attn_dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.pffn = PFFN(dim, hidden_dim=ffn_hidden_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def build_topk_mask_from_a0(
        a0: torch.Tensor,
        k: int,
        include_self: bool = True,
    ) -> torch.Tensor:
        """
        a0: [B, M, M] 或 [M, M]
        返回 mask: 同形状, float tensor, 0/1

        这里按论文思想:
        用初始 A0 决定哪些 pair 在当前层允许交互
        """
        if a0.dim() == 2:
            a0 = a0.unsqueeze(0)  # [1, M, M]

        bsz, m, _ = a0.shape
        total = m * m

        k = max(1, min(k, total))

        scores = a0.clone()

        if not include_self:
            eye = torch.eye(m, device=scores.device, dtype=torch.bool).unsqueeze(0)  # [1, M, M]
            scores = scores.masked_fill(eye, float("-inf"))

        flat_scores = scores.view(bsz, -1)  # [B, M*M]
        topk_idx = torch.topk(flat_scores, k=k, dim=-1).indices  # [B, k]

        mask = torch.zeros_like(flat_scores, dtype=torch.float32)
        mask.scatter_(dim=-1, index=topk_idx, value=1.0)
        mask = mask.view(bsz, m, m)

        return mask

    def forward(
        self,
        x: torch.Tensor,
        a0: torch.Tensor,
        k_active: int,
        scale: bool = True,
    ) -> torch.Tensor:
        """
        x:         [B, M, D]
        a0:        [B, M, M] 或 [M, M]
        k_active:  当前层激活多少个 pair

        return:    [B, M, D]
        """
        bsz, m, d = x.shape

        q = self.q(x)  # [B, M, A]
        k = self.k(x)  # [B, M, A]
        v = self.v(x)  # [B, M, D]

        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, M, M]
        if scale:
            scores = scores / math.sqrt(q.size(-1))

        mask = self.build_topk_mask_from_a0(a0, k=k_active, include_self=True)  # [B, M, M]

        # 这里按论文是 score ⊙ mask 再与 V 作用
        # 为了训练更稳定，实际工程里通常会转成 masked softmax
        # 下面保留“论文思想 + 工程稳定性”的实现
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(masked_scores, dim=-1)  # [B, M, M]
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        z = torch.matmul(attn, v)  # [B, M, D]
        z = self.dropout(z)

        # Eq.(6)
        z_hat = self.norm1(z + x)

        # Eq.(7)
        out = self.pffn(z_hat)
        out = self.norm2(out + z_hat)

        return out


class WindowBranch(nn.Module):
    """
    一个 window / granularity 分支:
    1. group features
    2. initial A0
    3. L 层 deferred interaction
    """
    def __init__(
        self,
        num_fields: int,
        field_dim: int,
        group_size: int,
        num_layers: int,
        attn_dim: Optional[int] = None,
        branch_dim: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_fields % group_size != 0:
            raise ValueError(f"num_fields={num_fields} 必须能被 group_size={group_size} 整除")

        self.num_fields = num_fields
        self.field_dim = field_dim
        self.group_size = group_size
        self.num_groups = num_fields // group_size

        in_dim = field_dim * group_size # 每个组别输出的维度大小
        branch_dim = branch_dim or in_dim

        # group concat 后可能再投影到统一维度
        self.group_proj = nn.Linear(in_dim, branch_dim) if branch_dim != in_dim else nn.Identity()

        # A0 用第一层输入计算
        a0_attn_dim = attn_dim or branch_dim
        self.a0_q = nn.Linear(branch_dim, a0_attn_dim, bias=False)
        self.a0_k = nn.Linear(branch_dim, a0_attn_dim, bias=False)

        self.layers = nn.ModuleList([
            DeferredInteractionLayer(
                dim=branch_dim,
                attn_dim=attn_dim or branch_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.branch_dim = branch_dim

    def group_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        将每 group_size 个 field 拼接成一个 group
        输出: [B, M, group_size*D]
        """
        bsz, n, d = x.shape
        assert n == self.num_fields
        assert d == self.field_dim

        x = x.view(bsz, self.num_groups, self.group_size, d)   # [B, M, g, D]
        x = x.reshape(bsz, self.num_groups, self.group_size * d)  # [B, M, gD]
        return x

    def compute_a0(self, x0: torch.Tensor) -> torch.Tensor:
        """
        x0: [B, M, D']
        A0 = (X0 Q0)(X0 K0)^T
        输出: [B, M, M]
        """
        q0 = self.a0_q(x0)
        k0 = self.a0_k(x0)
        a0 = torch.matmul(q0, k0.transpose(-1, -2))
        a0 = a0 / math.sqrt(q0.size(-1))
        return a0

    def get_k_active(self, layer_idx: int) -> int:
        """
        论文公式(5):
            k_l = floor(l/L * n^2)
        实现时更合理的是按当前 branch 的 pair 数:
            k_l = floor(l/L * M^2)
        """
        l = layer_idx + 1
        total_pairs = self.num_groups * self.num_groups
        k_active = int((l / self.num_layers) * total_pairs)
        k_active = max(1, min(k_active, total_pairs))
        return k_active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D]
        return: [B, M, D_branch]
        """
        x = self.group_features(x)     # [B, M, gD]
        x = self.group_proj(x)         # [B, M, D_branch]
        a0 = self.compute_a0(x)        # [B, M, M]

        for layer_idx, layer in enumerate(self.layers):
            k_active = self.get_k_active(layer_idx)
            x = layer(x, a0=a0, k_active=k_active)

        return x

class EmbeddingLayer(nn.Module):

    def __init__(
            self,
            book_vocab_size: int,
            word_vocab_size: int,
            embed_dim: int = 64,
            hidden_units: List[int] = [128, 64],
            feature_nums: int = 12,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.feature_nums = feature_nums

        self.uid_embeddings = nn.Embedding(2 ** 18, embed_dim)

        # ========== 嵌入层 ==========
        # 对应TensorFlow版本的SharedEmbeddingLayerNonereduce
        self.book_embeddings = nn.Embedding(
            book_vocab_size, embed_dim
        )
        self.word_embeddings = nn.Embedding(
            word_vocab_size, embed_dim
        )

        self.register_bucket_embed = nn.Embedding(
            num_embeddings=5,  # 桶数量
            embedding_dim=embed_dim,  # 嵌入维度（建议与其他离散特征维度一致）
        )

        self.cat_embeddings = nn.Linear(2 * hidden_units[-1], hidden_units[-1])

        # 随机初始化
        self._init_random_weights()


    def _init_random_weights(self):
        """随机初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def load_pretrained_weights(self, word_weights: Optional[torch.Tensor] = None,
                                book_weights: Optional[torch.Tensor] = None):
        """
        从外部加载预训练权重

        Args:
            word_weights: 词汇预训练权重 [word_vocab_size, word_embed_dim]
            book_weights: 书籍预训练权重 [book_vocab_size, book_embed_dim]
        """
        if word_weights is not None:
            expected_shape = (self.word_embeddings.num_embeddings, self.word_embeddings.embedding_dim)
            if word_weights.shape == expected_shape:
                self.word_embeddings.weight.data.copy_(word_weights)
                log.info(f"✓ 成功加载词汇预训练权重: {word_weights.shape}")
            else:
                log.warning(f"词汇权重维度不匹配: 期望{expected_shape}, 实际{word_weights.shape}")

        if book_weights is not None:
            expected_shape = (self.book_embeddings.num_embeddings, self.book_embeddings.embedding_dim)
            if book_weights.shape == expected_shape:
                self.book_embeddings.weight.data.copy_(book_weights)
                log.info(f"✓ 成功加载书籍预训练权重: {book_weights.shape}")
            else:
                log.warning(f"书籍权重维度不匹配: 期望{expected_shape}, 实际{book_weights.shape}")

    def _do_call(
            self,
            inputs: Dict[str, torch.Tensor],
            flag: str,
    ) ->torch.Tensor:
        """
        前向传播核心逻辑 - 对应TensorFlow版本的_do_call

        Args:
            inputs: 输入字典
            flag: 标记（"impression", "neg_impression", "neg_impression_rand"）

        Returns:
            预测得分 [B, 1]
        """
        # ========== 收集特征 ==========
        emb_tensor = []

        # 用户特征（对应TensorFlow版本的user_input_name）
        user_features = ["uid", "likemarks", "searchwords", "read_books", "read_search_keywords_books_day_30", "register_time"]

        for name in user_features:
            if name == "uid":
                # UID哈希特征
                uid_emb = self.uid_embeddings(inputs["uid"])
                emb_tensor.append(uid_emb.squeeze(1))
            elif name in ["likemarks", "searchwords"]:
                word_emb = self.word_embeddings(inputs[name])
                emb_tensor.append(torch.mean(word_emb, dim=1))
            elif name in ["read_books"]:
                book_emb = self.book_embeddings(inputs[name])
                emb_tensor.append(torch.mean(book_emb, dim=1))
            elif name in ["register_time"]:
                register_bucket_embed = self.register_bucket_embed(inputs[name])
                emb_tensor.append(register_bucket_embed)

        # read_search_keywords_books_day_30
        if "read_search_keywords_books_day_30" in inputs:
            book_emb = self.book_embeddings(inputs["read_search_keywords_books_day_30"]["book"])
            word_emb = self.word_embeddings(inputs["read_search_keywords_books_day_30"]["word"])
            book_word_emb = torch.cat([
                book_emb,
                word_emb
            ], dim=-1)
            book_word_emb = self.cat_embeddings(book_word_emb)
            emb_tensor.append(torch.mean(book_word_emb, dim=1))

        # 书籍特征（根据flag选择）
        prefix = flag  # "impression", "neg_impression", "neg_impression_rand"
        book_features = [f"{prefix}_bookid", f"{prefix}_bookmarks", f"{prefix}_bookname",f"{prefix}_tag",f"{prefix}_bookwords",f"{prefix}_bookinfo"]
                         # f"{prefix}_tag"]

        for name in book_features:
            if "bookid" in name:
                book_emb = self.book_embeddings(inputs[name])
                emb_tensor.append(book_emb)
            else:
                word_emb = self.word_embeddings(inputs[name])
                emb_tensor.append(torch.mean(word_emb, dim=1))

        return torch.stack(emb_tensor, dim=1)

    def forward(
            self,
            inputs: Dict[str, torch.Tensor],
            flag: str = "impression",
    ) -> list:
        """
        前向传播 - 对应TensorFlow版本的call

        Args:
            inputs: 输入字典
            flag: 标记

        Returns:
            预测得分 [B, 1]
        """
        return self._do_call(inputs, flag)

class MGDIN(nn.Module):
    """
    Multi-Granularity Information-Aware Deferred Interaction Network

    输入:
        field_embeddings: [B, N, D]
    输出:
        logits: [B]
        probs:  [B]
    """
    def __init__(
        self,

        num_fields: int, # 特征个数
        field_dim: int, # 特征维度
        group_sizes: List[int], # 组别大小 列表

        book_vocab_size: int, # 书表的大小
        word_vocab_size: int, # 词表的大小
        hidden_units: List[int] = [128, 64], # embedding层拼接书、章节、词所用的线性层维度

        num_layers: int = 3, # 层数
        attn_dim: Optional[int] = None,
        branch_dim: Optional[int] = None,
        ffn_hidden_dim: Optional[int] = None,
        mlp_hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        pooling: str = "mean", # 最终各个组别的池化方法
    ):

        super().__init__()
        self.num_fields = num_fields
        self.field_dim = field_dim
        self.group_sizes = group_sizes
        self.pooling = pooling

        # 数据处理同DCN模型
        self.embedding_layer= EmbeddingLayer(
            book_vocab_size=book_vocab_size,
            word_vocab_size=word_vocab_size,
            embed_dim=field_dim,
            hidden_units=hidden_units,
            feature_nums=num_fields,
        )

        self.branches = nn.ModuleList([
            WindowBranch(
                num_fields=num_fields,
                field_dim=field_dim,
                group_size=g,
                num_layers=num_layers,
                attn_dim=attn_dim,
                branch_dim=branch_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
            )
            for g in group_sizes
        ])

        if branch_dim is None:
            # 不同 group_size 时 branch 原始维度会不同，所以最好指定统一 branch_dim
            raise ValueError("建议显式传入统一的 branch_dim，否则多分支 concat 维度不一致")

        out_dim = len(group_sizes) * branch_dim

        mlp_layers = []
        in_dim = out_dim
        for h in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def pool_branch_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, M, D]
        返回: [B, D]
        """
        if self.pooling == "mean":
            return x.mean(dim=1)
        elif self.pooling == "sum":
            return x.sum(dim=1)
        elif self.pooling == "max":
            return x.max(dim=1).values
        else:
            raise ValueError(f"不支持的 pooling: {self.pooling}")

    def load_pretrained_weights(self, word_weights=None, book_weights=None):
        self.embedding_layer.load_pretrained_weights(
            word_weights=word_weights,
            book_weights=book_weights,
        )

    def forward(self, inputs,flag) -> torch.Tensor:
        """
        field_embeddings: [B, N, D]
        """

        field_embeddings=self.embedding_layer(inputs,flag)

        branch_outputs = []
        for branch in self.branches:
            out = branch(field_embeddings)     # [B, M_h, D_branch]
            pooled = self.pool_branch_output(out)  # [B, D_branch]
            branch_outputs.append(pooled)

        fused = torch.cat(branch_outputs, dim=-1)  # [B, K*D_branch]
        logits = self.mlp(fused).squeeze(-1)       # [B]
        probs = torch.sigmoid(logits)              # [B]
        return logits
    def predict(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        预测函数，返回用户向量 - 对应TensorFlow版本的predict

        Args:
            inputs: 输入字典

        Returns:
            包含用户向量的字典
        """
        self.eval()

        with torch.no_grad():
            # 直接使用forward计算分数
            return {"books_ranking": self.forward(inputs, flag="impression")}