"""预测头：Regression / Ordinal / Distribution。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    """LayerNorm → Linear → GELU → Dropout → Linear → Bone Age。"""

    def __init__(self, in_dim: int = 768, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # [B]


class OrdinalHead(nn.Module):
    """骨龄序结构：K 个阈值，输出 P(y > τ_k)。"""

    def __init__(self, in_dim: int = 768, hidden: int = 256,
                 step: int = 12, max_age: int = 216, dropout: float = 0.1):
        super().__init__()
        self.register_buffer(
            "thresholds",
            torch.arange(step, max_age + 1, step, dtype=torch.float32),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(self.thresholds)),
        )

    @property
    def num_thresholds(self) -> int:
        return len(self.thresholds)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits [B, K]

    @torch.no_grad()
    def predict(self, logits: torch.Tensor, max_bone_age: float) -> torch.Tensor:
        """由 P(y>τ_k) 恢复年龄：E[y] ≈ Σ_k P(y>τ_k)·step（左闭右开阶梯积分）。"""
        p = torch.sigmoid(logits)
        y = p.sum(dim=-1) * float(self.thresholds[0])
        return y.clamp(0.0, max_bone_age)


class DistributionHead(nn.Module):
    """离散骨龄分布：0..A_max 每个 age-bin 一个概率，Gaussian soft label。"""

    def __init__(self, in_dim: int = 768, hidden: int = 256,
                 max_age: int = 228, dropout: float = 0.1):
        super().__init__()
        self.max_age = max_age
        self.register_buffer(
            "bin_centers",
            torch.arange(0, max_age + 1, dtype=torch.float32),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max_age + 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits [B, A_max+1]

    def soft_target(self, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """真实骨龄附近 Gaussian label distribution q_i。"""
        centers = self.bin_centers.unsqueeze(0)            # [1, A+1]
        yv = y.view(-1, 1)                                  # [B,1]
        logits = -((centers - yv) ** 2) / (2.0 * sigma ** 2)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """期望 ŷ_dist = Σ_i i·p_i，并 clamp 到 [0, A_max]。"""
        p = torch.softmax(logits, dim=-1)
        y = (p * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        return y.clamp(0.0, float(self.max_age))
