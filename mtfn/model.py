"""MTFN
Bone Age Prediction Model（RSNA + RHPE 联合训练，ONE SHARED MTFN）

X
↓
ROI Crop
↓
Swin-T
↓
Multi-scale
↓
49 Visual Tokens
↓
Age-aware Attention
↓
Sex / Anatomical Landmark Fusion（ROI validity gating）
↓
Regression + Ordinal + Distribution
↓
Learnable Prediction Fusion
↓
Bone Age
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AgeAwareAttention
from .backbone import SwinTBackbone
from .embeddings import AgeEmbedding, SexEmbedding
from .heads import DistributionHead, OrdinalHead, RegressionHead
from .roi_encoder import KP_DIM, ROIEncoder
from .visual_tokens import VisualTokenizer


class MTFN(nn.Module):
    """
    MTFN

    Main pipeline:

        Image
          ↓
        Swin-T
          ↓
        Multi-scale features
          ↓
        VisualTokenizer
          ↓
        49 visual tokens
          ↓
        Age-aware Attention
          ↓
        Visual representation
          ↓
        + Sex embedding
        + Anatomical landmark embedding（ROI gating）
          ↓
        Fusion MLP
          ↓
        ┌───────────────┬───────────────┬───────────────┐
        ↓               ↓               ↓
      Reg Head       Ordinal Head    Distribution Head
        ↓               ↓               ↓
        └───────────────┴───────────────┴───────────────┘
                          ↓
                  Learnable Fusion
                          ↓
                     Bone Age

    RSNA / RHPE 完全共享全部模块，
    不引入 dataset embedding。
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # ============================================================
        # Basic dimension
        # ============================================================

        d = int(cfg.visual_token_dim)

        # ============================================================
        # 1. Swin-T Backbone
        # ============================================================

        self.backbone = SwinTBackbone(
            pretrained=bool(cfg.pretrained)
        )

        # ============================================================
        # 2. Multi-scale → Visual Tokens
        # ============================================================

        self.tokenizer = VisualTokenizer(
            dim=d,
            grid=int(cfg.token_grid),
            use_multiscale=bool(cfg.use_multiscale),
        )

        # ============================================================
        # 3. Age-aware Attention
        # ============================================================

        self.use_age_attention = bool(
            getattr(cfg, "use_age_attention", False)
        )

        self.use_ca = bool(
            getattr(cfg, "use_age_embedding", False)
        )

        if self.use_age_attention:

            self.attention = AgeAwareAttention(
                dim=d,
                num_heads=8,
                ffn_dim=3072,
                dropout=0.1,
                use_ca_condition=self.use_ca,
            )

        # ============================================================
        # 4. Sex Embedding
        # ============================================================

        self.use_sex = bool(
            getattr(cfg, "use_sex_embedding", False)
        )

        if self.use_sex:
            self.sex_embed = SexEmbedding(dim=16)

        # ============================================================
        # 5. Chronological Age Embedding
        #
        # Important:
        # RSNA 标准数据通常没有 chronological_age。
        # invalid CA（-1 / NaN / 负值）绝不进入 embedding。
        # ============================================================

        if self.use_ca:
            self.age_embed = AgeEmbedding(out_dim=d)

        # ============================================================
        # 6. Anatomical Landmark / ROI branch
        # ============================================================

        self.use_roi_branch = bool(
            getattr(cfg, "use_roi_branch", False)
        )

        if self.use_roi_branch:

            self.roi_encoder = ROIEncoder(
                out_dim=int(cfg.roi_out_dim)
            )

        # ROI validity gating
        self.use_roi_gating = bool(
            getattr(cfg, "use_roi_gating", True)
        )

        # ============================================================
        # 7. Feature Fusion
        # ============================================================

        fuse_in = d

        if self.use_sex:
            fuse_in += 16

        if self.use_roi_branch:
            fuse_in += int(cfg.roi_out_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fuse_in, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
        )

        # ============================================================
        # 8. Prediction Heads
        # ============================================================

        self.use_reg = bool(
            getattr(cfg, "use_regression_head", True)
        )

        self.use_ord = bool(
            getattr(cfg, "use_ordinal_head", True)
        )

        self.use_dist = bool(
            getattr(cfg, "use_distribution_head", True)
        )

        hidden_dim = int(cfg.regression_hidden_dim)
        dropout = float(cfg.head_dropout)

        if self.use_reg:

            self.reg_head = RegressionHead(
                d,
                hidden_dim,
                dropout,
            )

        if self.use_ord:

            self.ordinal_head = OrdinalHead(
                d,
                hidden_dim,
                cfg.ordinal_step,
                cfg.ordinal_max,
                dropout,
            )

        if self.use_dist:

            self.dist_head = DistributionHead(
                d,
                hidden_dim,
                cfg.dist_max,
                dropout,
            )

        # ============================================================
        # 9. Dynamic Prediction Fusion
        # ============================================================

        self.head_names = []

        if self.use_reg:
            self.head_names.append("reg")

        if self.use_ord:
            self.head_names.append("ord")

        if self.use_dist:
            self.head_names.append("dist")

        if len(self.head_names) == 0:
            raise ValueError(
                "MTFN 至少需要启用一个 prediction head."
            )

        # Learnable logits
        #
        # Initialize regression slightly stronger.
        #
        # If:
        #   reg + ord + dist
        #
        # initial weights:
        #   softmax([2, 0, 0])
        #
        init_logits = []

        for name in self.head_names:

            if name == "reg":
                init_logits.append(2.0)
            else:
                init_logits.append(0.0)

        self.fusion_logits = nn.Parameter(
            torch.tensor(
                init_logits,
                dtype=torch.float32,
            )
        )

    # ==================================================================
    # Helper: chronological age validity
    # ==================================================================

    @staticmethod
    def _ca_mask(ca, batch_size: int, device):
        """
        返回 (ca_tensor[B], mask[B] bool)。

        - ca 为 None / 长度不匹配 → (None, None)，CA 分支整体退化
        - per-sample: finite 且 >= 0 才 valid
        - invalid 位置的值会被安全替换，绝不进入 AgeEmbedding / ca_proj
        """

        if ca is None:
            return None, None

        if not torch.is_tensor(ca):
            ca = torch.as_tensor(ca)

        ca = ca.to(device=device, dtype=torch.float32).view(-1)

        if ca.numel() != batch_size:
            return None, None

        mask = torch.isfinite(ca) & (ca >= 0)

        if not bool(mask.any().item()):
            return None, None  # 全 invalid → 完全退化为普通 attention

        return ca, mask

    # ==================================================================
    # Encode
    # ==================================================================

    def encode(
        self,
        x: torch.Tensor,
        sex=None,
        ca=None,
        kp=None,
        roi_valid=None,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # 1. Backbone
        # --------------------------------------------------------------

        f2, f3, f4 = self.backbone(x)

        # --------------------------------------------------------------
        # 2. Multi-scale → 49 tokens
        # --------------------------------------------------------------

        tokens = self.tokenizer(
            f2,
            f3,
            f4,
        )

        # Expected:
        #
        # [B, 49, 768]
        #
        # or corresponding configured dimensions.

        # --------------------------------------------------------------
        # 3. Age-aware Attention
        # --------------------------------------------------------------

        ca_input, ca_mask = self._ca_mask(
            ca,
            x.shape[0],
            x.device,
        )

        if self.use_age_attention:

            # IMPORTANT:
            # CA invalid 的样本退化为可学习 age query（普通 attention），
            # 绝不把 -1 送入 ca_proj / AgeEmbedding。
            if self.use_ca and ca_input is not None:

                z_visual, _ = self.attention(
                    tokens,
                    ca_input,
                    ca_mask=ca_mask,
                )

            else:

                z_visual, _ = self.attention(
                    tokens,
                    None,
                )

        else:

            # Simple token pooling
            z_visual = tokens.mean(dim=1)

        # --------------------------------------------------------------
        # 4. Feature collection
        # --------------------------------------------------------------

        feats = [z_visual]

        batch_size = x.shape[0]

        # --------------------------------------------------------------
        # 5. Sex
        # --------------------------------------------------------------

        if self.use_sex:

            if sex is None:

                sex_input = torch.zeros(
                    batch_size,
                    device=x.device,
                    dtype=torch.long,
                )

            else:

                sex_input = sex.to(
                    device=x.device,
                    dtype=torch.long,
                )

            sex_input = sex_input.clamp(0, 1)

            sex_feat = self.sex_embed(
                sex_input
            )

            feats.append(sex_feat)

        # --------------------------------------------------------------
        # 6. Anatomical landmarks + ROI gating
        # --------------------------------------------------------------

        if self.use_roi_branch:

            if kp is None:

                kp_input = torch.zeros(
                    batch_size,
                    KP_DIM,
                    device=x.device,
                    dtype=torch.float32,
                )

            else:

                kp_input = kp.to(
                    device=x.device,
                    dtype=torch.float32,
                )

            roi_feat = self.roi_encoder(
                kp_input
            )

            # ROI gating: 无 ROI 样本不产生虚假解剖特征
            if self.use_roi_gating and roi_valid is not None:

                gate = roi_valid.to(
                    device=x.device,
                    dtype=roi_feat.dtype,
                ).view(-1, 1)

                roi_feat = roi_feat * gate

            feats.append(roi_feat)

        # --------------------------------------------------------------
        # 7. Fusion
        # --------------------------------------------------------------

        z = torch.cat(
            feats,
            dim=-1,
        )

        z_fusion = self.fusion_mlp(z)

        # --------------------------------------------------------------
        # 8. Chronological Age embedding
        #
        # ONLY valid CA samples contribute。
        # --------------------------------------------------------------

        if self.use_ca and ca_input is not None:

            ca_safe = ca_input.clamp(min=0.0)

            age_feat = self.age_embed(ca_safe)

            z_fusion = (
                z_fusion
                + age_feat
                * ca_mask.view(-1, 1).to(age_feat.dtype)
            )

        return z_fusion

    # ==================================================================
    # Forward
    # ==================================================================

    def forward(
        self,
        x: torch.Tensor,
        sex=None,
        ca=None,
        kp=None,
        roi_valid=None,
    ) -> dict:

        assert x.dim() == 4, (
            f"输入必须是 [B,C,H,W]，得到 {x.shape}"
        )

        assert x.shape[1] == 3, (
            f"输入必须为 3 通道，得到 {x.shape}"
        )

        # --------------------------------------------------------------
        # Encode
        # --------------------------------------------------------------

        z = self.encode(
            x=x,
            sex=sex,
            ca=ca,
            kp=kp,
            roi_valid=roi_valid,
        )

        outputs = {
            "z": z
        }

        # --------------------------------------------------------------
        # Regression
        # --------------------------------------------------------------

        if self.use_reg:

            outputs["reg"] = self.reg_head(z)

        # --------------------------------------------------------------
        # Ordinal
        # --------------------------------------------------------------

        if self.use_ord:

            outputs["ordinal"] = self.ordinal_head(z)

            # IMPORTANT:
            # Do NOT detach during training.
            #
            # We keep the prediction differentiable so that
            # fused loss can propagate into the ordinal head.

            outputs["ord"] = self.ordinal_head.predict(
                outputs["ordinal"],
                float(self.cfg.dist_max),
            )

        # --------------------------------------------------------------
        # Distribution
        # --------------------------------------------------------------

        if self.use_dist:

            outputs["dist"] = self.dist_head(z)

            outputs["dist_pred"] = self.dist_head.predict(
                outputs["dist"]
            )

        # --------------------------------------------------------------
        # Prediction Fusion
        # --------------------------------------------------------------

        candidates = []

        if self.use_reg:

            candidates.append(
                outputs["reg"]
            )

        if self.use_ord:

            candidates.append(
                outputs["ord"]
            )

        if self.use_dist:

            candidates.append(
                outputs["dist_pred"]
            )

        # [N_HEADS]
        w = torch.softmax(
            self.fusion_logits,
            dim=0,
        )

        outputs["w"] = w

        # --------------------------------------------------------------
        # Weighted prediction
        # --------------------------------------------------------------

        fused = torch.zeros_like(
            candidates[0]
        )

        for i, pred in enumerate(candidates):

            fused = fused + w[i] * pred

        outputs["fused"] = fused

        return outputs

    # ==================================================================
    # Prediction
    # ==================================================================

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        sex=None,
        ca=None,
        kp=None,
        roi_valid=None,
        mode: str = "learned",
    ) -> torch.Tensor:

        out = self.forward(
            x=x,
            sex=sex,
            ca=ca,
            kp=kp,
            roi_valid=roi_valid,
        )

        # --------------------------------------------------------------
        # Individual heads
        # --------------------------------------------------------------

        if mode == "reg":

            if not self.use_reg:
                raise RuntimeError(
                    "Regression head 未启用."
                )

            return out["reg"]

        if mode == "ord":

            if not self.use_ord:
                raise RuntimeError(
                    "Ordinal head 未启用."
                )

            return out["ord"]

        if mode == "dist":

            if not self.use_dist:
                raise RuntimeError(
                    "Distribution head 未启用."
                )

            return out["dist_pred"]

        # --------------------------------------------------------------
        # Average
        # --------------------------------------------------------------

        if mode == "average":

            cand = []

            if self.use_reg:
                cand.append(out["reg"])

            if self.use_ord:
                cand.append(out["ord"])

            if self.use_dist:
                cand.append(out["dist_pred"])

            return torch.stack(
                cand,
                dim=0,
            ).mean(dim=0)

        # --------------------------------------------------------------
        # Learned fusion
        # --------------------------------------------------------------

        return out["fused"]

    # ==================================================================
    # Head predictions
    # ==================================================================

    def head_predictions(
        self,
        out: dict,
    ) -> dict:

        cand = {}

        if "reg" in out:
            cand["reg"] = out["reg"]

        if "ord" in out:
            cand["ord"] = out["ord"]

        if "dist_pred" in out:
            cand["dist"] = out["dist_pred"]

        cand["fused"] = out["fused"]

        return cand

    # ==================================================================
    # Fusion weights
    # ==================================================================

    @torch.no_grad()
    def fusion_weights(self) -> dict:

        w = torch.softmax(
            self.fusion_logits,
            dim=0,
        )

        return {
            name: float(w[i].item())
            for i, name in enumerate(
                self.head_names
            )
        }
