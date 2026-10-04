"""MTFN: Multi-scale Token Fusion Network for Pediatric Bone Age Assessment.

架构要点：
- ROI 裁剪 + 保持纵横比 letterbox 预处理
- 保守数据增强（关键点坐标与图像同步变换）
- Age-Binned Sampling（平滑逆频率加权，alpha 可调）
- Swin Transformer backbone（torchvision ImageNet 预训练权重）
- 多尺度 F2/F3/F4 特征 -> 49 x 768 Visual Tokens
- 可退化 ROI 分支（COCO 17 关键点，无标注时自动置零）
- Age-aware Attention / Sex Embedding / 可选 Chronological Age
- Regression + Ordinal + Distribution 三预测头
- Consistency Loss + 多任务损失
- Learnable 预测融合（验证 MAE 锁定最优融合）
- 差分学习率 + Warmup/Cosine + AMP + 分阶段冻结
- 全量指标 + 年龄/性别分层验证 + 一次性 TEST 评估
- 全套消融开关（multi-scale / age-attention / sex / ROI / 各预测头 / 采样）
"""
from .config import MTFNConfig
from .model import MTFN

__all__ = ["MTFNConfig", "MTFN"]
