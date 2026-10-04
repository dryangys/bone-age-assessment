"""Swin-T Backbone 包装：只允许 Swin-T，返回 F2/F3/F4。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t


class SwinTBackbone(nn.Module):
    """Swin-Tiny，输出三个尺度特征（torchvision 输出 [B, C, H, W]）。

    输入 512×512 时（stride 4/8/16/32）：
      F2: [B, 192, 64, 64]   （对应 224 输入下的 28×28×192）
      F3: [B, 384, 32, 32]   （14×14×384）
      F4: [B, 768, 16, 16]   （7×7×768 → token grid 自适应池化）
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.swin = swin_t(weights=weights)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.shape[1] == 3, f"输入须为 [B,3,H,W]，得到 {x.shape}"
        feats = self.swin.features
        f2 = f3 = f4 = x
        for i, layer in enumerate(feats):
            x = layer(x)
            if i == 3:
                f2 = x
            elif i == 5:
                f3 = x
            elif i == 7:
                f4 = x
        f2 = self._to_bchw(f2, 192)
        f3 = self._to_bchw(f3, 384)
        f4 = self._to_bchw(f4, 768)
        return f2, f3, f4

    @staticmethod
    def _to_bchw(f: torch.Tensor, ch: int) -> torch.Tensor:
        """自动适配 [B,C,H,W] / [B,H,W,C]。"""
        if f.dim() != 4:
            raise AssertionError(f"特征须为 4D，得到 {tuple(f.shape)}")
        if f.shape[1] == ch:
            return f
        if f.shape[-1] == ch:
            return f.permute(0, 3, 1, 2).contiguous()
        raise AssertionError(f"期望 {ch} 通道，得到 {tuple(f.shape)}")
