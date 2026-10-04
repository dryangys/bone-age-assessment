"""Age-aware Attention：自注意力 + age-conditioned query。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AgeAwareAttention(nn.Module):
    """在 49 个 visual tokens 上做：

    1) 标准自注意力 T_att = Softmax(QK^T/√d) V + residual + LN + FFN
    2) age-conditioned query z_visual = Softmax(q_age K^T/√d) V

    q_age：可学习 age query（无 chronological age 时），
    或 MLP(CA)（数据存在 chronological age 且 use_age_embedding=True 时）。
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        use_ca_condition: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

        # age query：可学习参数 z_age → q_age = f(z_age)
        self.age_query = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.age_query, std=0.02)
        self.use_ca_condition = use_ca_condition
        if use_ca_condition:
            self.ca_proj = nn.Sequential(
                nn.LayerNorm(1),
                nn.Linear(1, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        ca: torch.Tensor | None = None,
        ca_mask: torch.Tensor | None = None,
    ):
        """tokens [B,49,768] → z_visual [B,768]。

        ca      : [B] chronological age（月）
        ca_mask : [B] bool，True = CA 有效。
                  混合 batch（RSNA 无 CA / RHPE 有 CA）中，
                  invalid 样本退化为可学习 age query，
                  -1 绝不进入 ca_proj。
        """
        b, n, d = tokens.shape
        assert n == 49 and d == 768, f"tokens 须为 [B,49,768]，得到 {tuple(tokens.shape)}"

        q = self._split(self.q(tokens))    # [B,H,49,dh]
        k = self._split(self.k(tokens))
        v = self._split(self.v(tokens))

        # 1) 标准自注意力 + residual + LN
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        t_att = self.attn_drop(torch.softmax(att, dim=-1)) @ v
        t_att = self.out(t_att.transpose(1, 2).reshape(b, n, d))
        tokens_p = self.norm1(tokens + t_att)

        # 2) FFN + residual + LN
        tokens_pp = self.norm2(tokens_p + self.ffn(tokens_p))

        # 3) age-conditioned query 聚合
        q_learned = self.age_query.expand(b, -1)                   # [B,768]

        if self.use_ca_condition and ca is not None:

            ca_safe = ca.view(-1, 1).float().clamp(min=0.0)        # invalid 占位，会被 mask 替换
            q_ca = self.ca_proj(ca_safe)                           # [B,768]

            if ca_mask is not None:

                mask = ca_mask.view(-1, 1).to(q_ca.dtype)

                q_age = q_learned * (1.0 - mask) + q_ca * mask

            else:

                q_age = q_ca

        else:

            q_age = q_learned

        q_age = self._split(q_age.unsqueeze(1))                    # [B,H,1,dh]
        a_age = torch.softmax((q_age @ k.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1)
        z_visual = (a_age @ v).transpose(1, 2).reshape(b, d)       # [B,768]
        return z_visual, tokens_pp
