"""MTFN 全局配置（RSNA + RHPE 联合训练框架）。"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


# 默认年龄分桶（月）：0-24-48-...-192-300
DEFAULT_AGE_BINS: list = [0, 24, 48, 72, 96, 120, 144, 168, 192, 300]

# dataset_id 约定
RSNA_ID = 0
RHPE_ID = 1


@dataclass
class MTFNConfig:
    # ---------------- RSNA 数据 ----------------
    image_size: int = 512
    in_channels: int = 3
    use_roi_crop: bool = True          # 自动手部 ROI 检测
    roi_margin: float = 0.05           # ROI 边距 5%~15%
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    train_image_dir: str = ""
    val_image_dir: str = ""
    test_image_dir: str = ""

    # ---------------- RHPE 数据----------------
    # 未提供 rhpe_csv 时自动退化为 RSNA-only 模式
    rhpe_csv: str = ""                 # RHPE 训练 CSV（空 = RSNA only）
    rhpe_image_dir: str = ""           # RHPE 训练图像目录（递归扫描）
    rhpe_roi_annotation: str = ""      # RHPE ROI/关键点标注 JSON（可选）
    rhpe_val_csv: str = ""             # RHPE 验证 CSV（可选，仅泛化评估）
    rhpe_val_image_dir: str = ""
    rhpe_age_unit: str = ""            # "months" / "years"；提供 RHPE 时必须显式指定（禁止猜测）

    # ---------------- Backbone----------------
    backbone: str = "swin_t"
    pretrained: bool = True
    freeze_backbone: bool = False

    # ---------------- Visual Tokens ----------------
    token_grid: int = 7                # 7×7 = 49 tokens
    visual_token_dim: int = 768

    # ---------------- 模块开关（消融实验）----------------
    use_multiscale: bool = True        # F2/F3/F4 多尺度
    use_age_attention: bool = True     # Age-aware Attention
    use_sex_embedding: bool = True     # Sex Embedding (d_s=16)
    use_age_embedding: bool = False    # chronological age 条件（数据存在时才生效）
    use_roi_branch: bool = True        # ROI/关键点分支（无标注自动退化）
    roi_annotation: str = ""           # RSNA COCO 关键点标注 json（train 版）
    use_annotated_roi: bool = True     # 标注 bbox 优先裁剪（缺失回退 Otsu）
    roi_out_dim: int = 128             # 关键点编码维度
    use_roi_gating: bool = True        # ROI validity gating
    # Reviewer 9 w/o Anatomy 消融：True 时训练/验证中关键点输入严格置零
    # （kp=None + roi_valid=0），架构占位保留，仅屏蔽解剖信息输入。
    zero_anatomy: bool = False

    # ---------------- Chronological Age----------------
    use_chronological_age: bool = True # RHPE 存在真实 CA 时启用；RSNA-only 自动失效

    # ---------------- Heads ----------------
    use_regression_head: bool = True
    use_ordinal_head: bool = True
    use_distribution_head: bool = True
    use_consistency_loss: bool = True
    regression_hidden_dim: int = 256
    head_dropout: float = 0.1
    ordinal_step: int = 12
    ordinal_max: int = 216             # 阈值 12,24,...,216 → K=18
    dist_max: int = 228
    dist_sigma: float = 2.0            # Gaussian soft label σ（月）

    # ---------------- 损失权重----------------
    regression_weight: float = 1.0
    ordinal_weight: float = 0.2
    distribution_weight: float = 0.2
    fused_weight: float = 0.5          # L_fused = |y_hat_fused - y|
    consistency_weight: float = 0.05
    loss_mode: str = "smoothl1"        # smoothl1 | mae
    smoothl1_beta: float = 1.0

    # ---------------- 预测融合----------------
    fusion_mode: str = "learned"       # learned | reg | ord | dist | average

    # ---------------- Sampling----------------
    use_age_sampling: bool = True
    sampler_alpha: float = 0.5         # age_sampling_alpha
    age_bins: list = field(default_factory=lambda: list(DEFAULT_AGE_BINS))
    age_bin_overflow: str = "clip"     # clip | error（超界年龄处理，默认 clip 并打印统计）
    # dataset-aware sampling 比例：从 sampling probability 推导权重
    dataset_ratio_rsna: float = 0.7    # RSNA-only 模式下自动忽略
    dataset_ratio_rhpe: float = 0.3
    # 训练中 batch dataset ratio 长期偏离设定值超过此容差时 warning
    dataset_ratio_warn_tol: float = 0.10
    sampler_debug_simulate: int = 0    # >0 时初始化采样 N 次验证 ratio（仅初始化时运行）

    # ---------------- Normalization----------------
    normalization_mode: str = "shared"  # shared（RSNA train + RHPE train 联合统计）| dataset_specific
    stats_sample_size: int = 1000       # 计算归一化统计的抽样数

    # ---------------- 泄漏检查----------------
    cross_dataset_hash_check: bool = True   # 跨数据集 SHA256 重复图像检查
    cross_dataset_hash_max: int = 20000     # hash 检查的最大图像数（防止过慢）

    # ---------------- 优化器 / 训练策略 ----------------
    backbone_lr: float = 1e-5          # 差分学习率：Swin-T
    head_lr: float = 1e-4              # 差分学习率：新增模块
    weight_decay: float = 0.05
    epochs: int = 150
    warmup_epochs: int = 10
    batch_size: int = 14
    num_workers: int = 8
    max_grad_norm: float = 1.0
    amp: bool = True
    early_stopping_patience: int = 20
    seed: int = 42
    freeze_backbone_epochs: int = 0
    stage_b_epochs: int = 0

    # ---------------- 数据增强----------------
    use_horizontal_flip: bool = False  # 默认关闭（需与 KP 同步变换）
    aug_rotation: float = 6.0          # ±6°
    aug_scale_min: float = 0.9
    aug_scale_max: float = 1.1
    aug_translate: float = 0.05
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2

    # ---------------- 输出 ----------------
    output_dir: str = "outputs/mtfn"
    # 图像归一化统计
    norm_mean: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    norm_std: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # RHPE 数据集专用归一化（normalization_mode=dataset_specific 时使用）
    rhpe_norm_mean: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    rhpe_norm_std: list = field(default_factory=lambda: [0.5, 0.5, 0.5])

    # ---------------- 运行时 ----------------
    device: str = "auto"
    limit_train: int = 0               # 调试用：截断样本数（0=全量）
    limit_val: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MTFNConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
