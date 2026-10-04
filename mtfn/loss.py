"""多任务损失：

L_total = λ_reg·L_reg + λ_ord·L_ordinal + λ_dist·L_dist
        + λ_fused·L_fused + λ_cons·L_cons

全部权重配置化，禁止隐藏硬编码。
Regression 是主要监督来源，Ordinal/Distribution 为辅助任务
。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import DistributionHead, OrdinalHead


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg, ordinal_head: OrdinalHead | None,
                 dist_head: DistributionHead | None):
        super().__init__()
        self.cfg = cfg
        self.ordinal_head = ordinal_head
        self.dist_head = dist_head

    def reg_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss_mode == "mae":
            return F.l1_loss(pred, target)
        return F.smooth_l1_loss(pred, target, beta=self.cfg.smoothl1_beta)

    def ordinal_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE：z_ik = I(y_i > τ_k)，单调性由阈值排序天然保证。"""
        thresholds = self.ordinal_head.thresholds            # [K]
        z = (target.view(-1, 1) > thresholds.view(1, -1)).float()
        return F.binary_cross_entropy_with_logits(logits, z)

    def dist_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """soft target cross entropy：L = -Σ q_i log p_i。"""
        q = self.dist_head.soft_target(target, self.cfg.dist_sigma)
        log_p = F.log_softmax(logits, dim=-1)
        return -(q * log_p).sum(dim=-1).mean()

    def consistency_loss(self, y_orig: torch.Tensor, y_aug: torch.Tensor) -> torch.Tensor:
        """默认 |ŷ_orig − ŷ_aug|。"""
        return (y_orig - y_aug).abs().mean()

    def forward(self, outputs: dict, target: torch.Tensor,
                y_aug: torch.Tensor | None = None) -> tuple:
        losses = {}
        total = torch.zeros((), device=target.device)

        if self.cfg.regression_weight != 0:
            l_reg = self.reg_loss(outputs["reg"], target)
            losses["reg"] = l_reg
            total = total + self.cfg.regression_weight * l_reg

        # L_fused = |y_hat_fused - y|（learnable fusion 的梯度来源）
        if self.cfg.fused_weight != 0 and "fused" in outputs:
            l_fused = self.reg_loss(outputs["fused"], target)
            losses["fused"] = l_fused
            total = total + self.cfg.fused_weight * l_fused

        if self.ordinal_head is not None and self.cfg.ordinal_weight != 0:
            l_ord = self.ordinal_loss(outputs["ordinal"], target)
            losses["ord"] = l_ord
            total = total + self.cfg.ordinal_weight * l_ord

        if self.dist_head is not None and self.cfg.distribution_weight != 0:
            l_dist = self.dist_loss(outputs["dist"], target)
            losses["dist"] = l_dist
            total = total + self.cfg.distribution_weight * l_dist

        if y_aug is not None and self.cfg.consistency_weight != 0:
            l_cons = self.consistency_loss(outputs["fused"], y_aug)
            losses["cons"] = l_cons
            total = total + self.cfg.consistency_weight * l_cons

        losses["total"] = total
        return total, losses
