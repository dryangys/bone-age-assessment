"""Visual Tokens：多尺度投影 + 融合 → 49×768。"""
from __future__ import annotations

import torch
import torch.nn as nn


class VisualTokenizer(nn.Module):
    """F2/F3/F4 → 统一 768 维 → 7×7=49 tokens。

    每个尺度：AdaptiveAvgPool(7×7)（Token Reduction，防显存爆炸）
    → 1×1 Conv 投影到 768 → 融合（逐元素相加 + LayerNorm）。
    use_multiscale=False 时仅用 F4（消融 Exp1/Exp7 基线）。
    """

    def __init__(
        self,
        dim: int = 768,
        grid: int = 7,
        use_multiscale: bool = True,
        c2: int = 192,
        c3: int = 384,
        c4: int = 768,
    ):
        super().__init__()
        self.grid = grid
        self.use_multiscale = use_multiscale
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj4 = nn.Conv2d(c4, dim, 1, bias=True) if c4 != dim else nn.Identity()
        if use_multiscale:
            self.proj2 = nn.Conv2d(c2, dim, 1, bias=True)
            self.proj3 = nn.Conv2d(c3, dim, 1, bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor):
        t4 = self.proj4(self.pool(f4))
        if self.use_multiscale:
            t2 = self.proj2(self.pool(f2))
            t3 = self.proj3(self.pool(f3))
            fused = t2 + t3 + t4
        else:
            fused = t4

        b, c, h, w = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)  # [B, HW, C]

        # 必须 assert，不允许 silent shape mismatch
        assert tokens.shape[1] == self.grid * self.grid, (
            f"token 数应为 {self.grid ** 2}，得到 {tokens.shape[1]}"
        )
        assert tokens.shape[2] == 768, f"token 维度须为 768，得到 {tokens.shape[2]}"
        tokens = self.norm(tokens)
        assert tokens.shape[1] == 49 and tokens.shape[2] == 768, (
            f"Visual Tokens 必须为 [B,49,768]，得到 {tuple(tokens.shape)}"
        )
        return tokens
