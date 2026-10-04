"""MTFN 训练入口（RSNA + RHPE 联合训练）。

要点：
- 未提供 --rhpe_csv 时自动退化为 RSNA-only 模式
- 提供 RHPE 时必须显式指定 --rhpe_age_unit（禁止猜测单位）
- 归一化统计仅用 RSNA train + RHPE train（带签名缓存防错用旧统计）
- split 内（check_leakage）+ 跨数据集（ID + SHA256）泄漏检查
- CA availability 差异提示 + use_chronological_age 消融开关
- 启动时打印 Dataset Summary
- 输出 dataset_summary.json / sampler_statistics.json /
  normalization.json / config.json / train.log 等全部文件
- fusion 策略仅用 RSNA validation 锁定（写入 fusion_selection.json）
- RSNA test 绝不参与任何训练/选择逻辑（train.py 只做 ID 泄漏检查）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtfn.config import MTFNConfig, RHPE_ID, RSNA_ID
from mtfn.dataset import (
    BoneAgeDataset,
    MultiDatasetBoneAgeDataset,
    check_leakage,
    cross_dataset_leakage_check,
    load_csv_meta,
    print_dataset_summary,
)
from mtfn.loss import MultiTaskLoss
from mtfn.model import MTFN
from mtfn.preprocessing import compute_train_stats
from mtfn.roi_encoder import derive_roi_json
from mtfn.sampler import (
    build_multi_dataset_age_sampler,
    simulate_sampling,
)
from mtfn.trainer import Trainer, save_config_yaml


# ================================================================
# CLI
# ================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MTFN 训练（RSNA + RHPE 联合训练；"
                    "不提供 --rhpe_csv 时自动退化为 RSNA-only）")

    # ---------------- RSNA 数据 ----------------
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--val_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default="",
                   help="可选：仅用于泄漏检查，训练不使用")
    p.add_argument("--train_image_dir", type=str, required=True)
    p.add_argument("--val_image_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/mtfn")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--limit_train", type=int, help="调试：截断训练样本")
    p.add_argument("--limit_val", type=int, help="调试：截断验证样本")

    # ---------------- RHPE 数据----------------
    p.add_argument("--rhpe_csv", type=str, default="",
                   help="RHPE 训练 CSV（空 = RSNA-only 模式）")
    p.add_argument("--rhpe_image_dir", type=str, default="",
                   help="RHPE 训练图像目录（递归扫描）")
    p.add_argument("--rhpe_roi_annotation", type=str, default="",
                   help="RHPE ROI/关键点标注 JSON（可选）")
    p.add_argument("--rhpe_val_csv", type=str, default="",
                   help="可选：RHPE 验证 CSV（仅泛化参考，不影响模型选择）")
    p.add_argument("--rhpe_val_image_dir", type=str, default="")
    p.add_argument("--rhpe_age_unit", type=str, default="",
                   choices=["", "months", "years"],
                   help="RHPE 骨龄单位；提供 --rhpe_csv 时必须显式指定（禁止猜测）")

    # ---------------- Sampling----------------
    p.add_argument("--dataset_ratio_rsna", type=float,
                   help="RSNA sampling ratio（默认 0.7）")
    p.add_argument("--dataset_ratio_rhpe", type=float,
                   help="RHPE sampling ratio（默认 0.3）")
    p.add_argument("--age_sampling_alpha", type=float,
                   help="年龄平衡指数 α（等价于 --sampler_alpha，默认 0.5）")
    p.add_argument("--age_bins", type=str,
                   help="逗号分隔月龄边界，如 0,24,48,72,96,120,144,168,192,300")
    p.add_argument("--age_bin_overflow", type=str,
                   choices=["clip", "error"])
    p.add_argument("--sampler_debug_simulate", type=int,
                   help=">0 时初始化阶段模拟采样 N 次验证 ratio（仅 debug）")

    # ---------------- Normalization / 泄漏检查 ----------------
    p.add_argument("--normalization_mode", type=str,
                   choices=["shared", "dataset_specific"])
    p.add_argument("--cross_dataset_hash_check", type=str, default=None,
                   choices=["true", "false"],
                   help="跨数据集 SHA256 图像重复检查（默认开启）")

    # ---------------- Chronological Age----------------
    p.add_argument("--use_chronological_age", type=str, default=None,
                   choices=["true", "false"],
                   help="RHPE 存在真实 CA 时是否启用 CA 条件（消融 A/B/C）")
    p.add_argument("--use_age_embedding", type=str, default=None,
                   choices=["true", "false"],
                   help="强制覆盖 CA embedding（默认由数据 CA 可用性自动决定）")

    # ---------------- 优化器（差分 lr / warmup / cosine）----------------
    p.add_argument("--epochs", type=int)
    p.add_argument("--warmup_epochs", type=int)
    p.add_argument("--backbone_lr", type=float)
    p.add_argument("--head_lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--max_grad_norm", type=float)
    p.add_argument("--early_stopping_patience", type=int)
    p.add_argument("--amp", type=str, default=None, choices=["true", "false"])

    # ---------------- 分阶段训练 ----------------
    p.add_argument("--freeze_backbone_epochs", type=int)
    p.add_argument("--stage_b_epochs", type=int)

    # ---------------- 模块开关（消融实验）----------------
    p.add_argument("--use_age_sampling", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--sampler_alpha", type=float)
    p.add_argument("--use_multiscale", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_age_attention", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_sex_embedding", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_ordinal_head", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_distribution_head", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_consistency_loss", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_roi_branch", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_roi_crop", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_annotated_roi", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_roi_gating", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--use_horizontal_flip", type=str, default=None,
                   choices=["true", "false"])
    p.add_argument("--roi_annotation", type=str, default="",
                   help="RSNA COCO 关键点标注 json（train 版，val/test 自动推导）")

    # ---------------- 损失权重----------------
    p.add_argument("--regression_weight", type=float)
    p.add_argument("--ordinal_weight", type=float)
    p.add_argument("--distribution_weight", type=float)
    p.add_argument("--fused_weight", type=float)
    p.add_argument("--consistency_weight", type=float)
    p.add_argument("--loss_mode", type=str, choices=["smoothl1", "mae"])
    p.add_argument("--dist_sigma", type=float)
    p.add_argument("--ordinal_step", type=int,
                   help="ordinal 二分类阈值间距（月）；P2 敏感性实验用"
                        "（默认 12，见 config.ordinal_step）")

    # ---------------- 融合 ----------------
    p.add_argument("--fusion_mode", type=str,
                   choices=["learned", "reg", "ord", "dist", "average"])
    # ---------------- Reviewer 9 消融 ----------------
    p.add_argument("--zero_anatomy", action="store_true",
                   help="Reviewer 9 w/o Anatomy：关键点输入严格置零"
                        "（kp=None + roi_valid=0），架构占位保留，"
                        "数据管线/裁剪完全不变")
    return p


def str2opt(v):
    if v is None:
        return None
    return v == "true"


def parse_age_bins(s: str) -> list:
    try:
        bins = [float(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"[age_bins] 解析失败: {s!r}（应为逗号分隔数字）")
    if len(bins) < 2 or any(b2 <= b1 for b1, b2 in zip(bins, bins[1:])):
        raise SystemExit(f"[age_bins] 必须严格递增且至少两个边界: {bins}")
    return bins


def apply_overrides(cfg: MTFNConfig, args) -> MTFNConfig:
    """命令行覆盖 config。"""
    for field_name in ("epochs", "warmup_epochs", "backbone_lr", "head_lr",
                       "weight_decay", "max_grad_norm",
                       "early_stopping_patience", "freeze_backbone_epochs",
                       "stage_b_epochs", "sampler_alpha",
                       "regression_weight", "ordinal_weight",
                       "distribution_weight", "fused_weight",
                       "consistency_weight", "loss_mode", "dist_sigma",
                       "ordinal_step",
                       "fusion_mode", "roi_annotation", "batch_size",
                       "num_workers", "seed", "limit_train", "limit_val",
                       "rhpe_csv", "rhpe_image_dir", "rhpe_roi_annotation",
                       "rhpe_val_csv", "rhpe_val_image_dir", "rhpe_age_unit",
                       "dataset_ratio_rsna", "dataset_ratio_rhpe",
                       "age_bin_overflow", "sampler_debug_simulate",
                       "normalization_mode", "zero_anatomy"):
        v = getattr(args, field_name, None)
        if v is not None:
            setattr(cfg, field_name, v)

    # --age_sampling_alpha 与 --sampler_alpha 等价，显式给出者优先
    if args.age_sampling_alpha is not None:
        cfg.sampler_alpha = args.age_sampling_alpha

    if args.age_bins:
        cfg.age_bins = parse_age_bins(args.age_bins)

    for flag in ("use_age_sampling", "use_multiscale", "use_age_attention",
                 "use_sex_embedding", "use_ordinal_head",
                 "use_distribution_head", "use_consistency_loss",
                 "use_roi_branch", "use_roi_crop", "use_annotated_roi",
                 "use_roi_gating", "use_horizontal_flip",
                 "use_chronological_age", "use_age_embedding",
                 "cross_dataset_hash_check"):
        v = str2opt(getattr(args, flag, None))
        if v is not None:
            setattr(cfg, flag, v)
    if args.amp is not None:
        cfg.amp = args.amp == "true"

    cfg.train_csv = args.train_csv
    cfg.val_csv = args.val_csv
    cfg.test_csv = args.test_csv
    cfg.train_image_dir = args.train_image_dir
    cfg.val_image_dir = args.val_image_dir
    cfg.output_dir = args.output_dir
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================================================================
# stdout → 终端 + train.log
# ================================================================

class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ================================================================
# Epoch 日志
# ================================================================

def epoch_writer(row: dict, va: dict, improved: bool,
                 va_rhpe: dict | None = None) -> None:
    print(f"[epoch {row['epoch']:>3}] stage={row['stage']:<10} "
          f"loss={row['train_loss']:.4f} "
          f"(reg={row.get('loss_reg', 0.0):.3f} "
          f"ord={row.get('loss_ord', 0.0):.3f} "
          f"dist={row.get('loss_dist', 0.0):.3f} "
          f"fused={row.get('loss_fused', 0.0):.3f}) "
          f"b_lr={row['backbone_lr']:.2e} h_lr={row['head_lr']:.2e} "
          f"train_mae={row['train_mae']:.3f}", flush=True)
    print(f"           val-RSNA: MAE={row['val_mae_rsna']:.4f} "
          f"RMSE={row['val_rmse_rsna']:.4f} "
          f"MedAE={row['val_medae_rsna']:.3f} "
          f"±6={row['val_acc_6'] * 100:.1f}% "
          f"±12={row['val_acc_12'] * 100:.1f}%", flush=True)
    if va_rhpe is not None:
        print(f"           val-RHPE: MAE={row.get('val_mae_rhpe', float('nan')):.4f} "
              f"RMSE={row.get('val_rmse_rhpe', float('nan')):.4f}"
              f"（仅泛化参考，不影响模型选择）", flush=True)
    print(f"           heads: reg={row.get('mae_reg', float('nan')):.3f} "
          f"ord={row.get('mae_ord', float('nan')):.3f} "
          f"dist={row.get('mae_dist', float('nan')):.3f} "
          f"fused={row.get('mae_fused', float('nan')):.3f} | "
          f"fusion w: reg={row['w_reg']:.3f} ord={row['w_ord']:.3f} "
          f"dist={row['w_dist']:.3f} | "
          f"seen: RSNA={row['rsna_seen']} RHPE={row['rhpe_seen']}"
          + ("  ← best" if improved else ""), flush=True)


# ================================================================
# Normalization statistics
# ================================================================

def norm_signature(cfg: MTFNConfig, use_rhpe: bool) -> str:
    keys = [
        cfg.train_csv, cfg.train_image_dir,
        cfg.rhpe_csv if use_rhpe else "",
        cfg.rhpe_image_dir if use_rhpe else "",
        cfg.rhpe_age_unit if use_rhpe else "",
        cfg.normalization_mode,
        str(cfg.image_size), str(bool(cfg.use_roi_crop)),
        str(float(cfg.roi_margin)), str(bool(cfg.use_annotated_roi)),
        str(int(cfg.stats_sample_size)), str(int(cfg.seed)),
        str(int(cfg.limit_train)),
    ]
    return hashlib.md5("|".join(keys).encode()).hexdigest()


def compute_normalization(cfg: MTFNConfig, out: Path,
                          rsna_paths: list,
                          rhpe_paths: list | None) -> dict:
    """计算（或复用缓存）归一化统计。

    shared           : RSNA train + RHPE train 联合统计
    dataset_specific : RSNA / RHPE 各自统计

    缓存带 signature（数据集 + 预处理配置），防止修改训练数据后
    错误复用旧 mean/std。
    """
    sig = norm_signature(cfg, rhpe_paths is not None)
    stats_path = out / "normalization.json"

    if stats_path.exists():
        cached = json.loads(stats_path.read_text())
        if cached.get("signature") == sig:
            print(f"[stats] 复用 {stats_path}（signature 匹配）: "
                  f"mean={cached['mean']}, std={cached['std']}")
            return cached
        print("[stats] 缓存 signature 不匹配（训练数据/配置已变化）"
              "→ 重新计算 normalization 统计")

    n_r = len(rsna_paths)
    n_h = len(rhpe_paths) if rhpe_paths else 0

    if cfg.normalization_mode == "shared":
        paths = list(rsna_paths) + (list(rhpe_paths) if rhpe_paths else [])
        print(f"[stats] shared 模式：RSNA train({n_r}) + RHPE train({n_h}) "
              f"联合计算 ...")
        mean, std = compute_train_stats(
            paths, cfg.image_size, cfg.use_roi_crop, cfg.roi_margin,
            cfg.stats_sample_size, cfg.seed,
        )
        stats = {
            "signature": sig,
            "mode": "shared",
            "mean": [float(m) for m in mean],
            "std": [float(s) for s in std],
            "rhpe_mean": [float(m) for m in mean],
            "rhpe_std": [float(s) for s in std],
            "computed_from": {"RSNA_train": n_r, "RHPE_train": n_h},
        }
    else:
        print(f"[stats] dataset_specific 模式：分别计算 "
              f"RSNA train({n_r}) / RHPE train({n_h}) ...")
        r_mean, r_std = compute_train_stats(
            rsna_paths, cfg.image_size, cfg.use_roi_crop, cfg.roi_margin,
            cfg.stats_sample_size, cfg.seed,
        )
        stats = {
            "signature": sig,
            "mode": "dataset_specific",
            "mean": [float(m) for m in r_mean],
            "std": [float(s) for s in r_std],
            "rhpe_mean": [float(m) for m in r_mean],
            "rhpe_std": [float(s) for s in r_std],
            "computed_from": {"RSNA_train": n_r, "RHPE_train": n_h},
        }
        if rhpe_paths:
            h_mean, h_std = compute_train_stats(
                rhpe_paths, cfg.image_size, cfg.use_roi_crop, cfg.roi_margin,
                cfg.stats_sample_size, cfg.seed,
            )
            stats["rhpe_mean"] = [float(m) for m in h_mean]
            stats["rhpe_std"] = [float(s) for s in h_std]

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[stats] RSNA mean={stats['mean']}, std={stats['std']}")
    if rhpe_paths:
        print(f"[stats] RHPE mean={stats['rhpe_mean']}, "
              f"std={stats['rhpe_std']}")
    return stats


# ================================================================
# Fusion strategy selection
# ================================================================

def predict_val_loader(model: MTFN, loader: DataLoader,
                       device: torch.device, mode: str,
                       amp_enabled: bool) -> tuple:
    """以指定 fusion mode 在 RSNA validation 上推理（8 元组 batch）。"""
    preds, ys = [], []
    for x, y, sex, ca, kp, roi_valid, _, _ in loader:
        kwargs = {}
        if getattr(model, "use_ca", False):
            kwargs["ca"] = ca.to(device).float()
        if getattr(model, "use_roi_branch", False):
            kwargs["roi_valid"] = roi_valid.to(device)
        with torch.autocast(device.type, enabled=amp_enabled):
            p = model.predict(x.to(device), sex.to(device),
                              kp=kp.to(device), mode=mode, **kwargs)
        preds.append(p.float().cpu())
        ys.append(y)
    return (torch.cat(preds).numpy(), torch.cat(ys).numpy())


# ================================================================
# Main
# ================================================================

def main() -> None:
    args = build_parser().parse_args()
    cfg = apply_overrides(MTFNConfig(), args)

    out = Path(cfg.output_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # stdout → 终端 + train.log
    sys.stdout = Tee(out / "train.log")

    set_seed(cfg.seed)

    use_cuda = (
        (cfg.device == "auto" and torch.cuda.is_available())
        or (cfg.device == "cuda" and torch.cuda.is_available())
    )
    device = torch.device("cuda" if use_cuda else "cpu")

    # ================================================================
    # 模式判定：RHPE 提供与否
    # ================================================================

    use_rhpe = bool(cfg.rhpe_csv)

    if use_rhpe and not cfg.rhpe_age_unit:
        raise SystemExit(
            "[RHPE] 提供了 --rhpe_csv 但未指定 --rhpe_age_unit "
            "(months/years)。禁止自动猜测骨龄单位。"
        )
    if cfg.rhpe_age_unit and not use_rhpe:
        print("[warning] --rhpe_age_unit 已设置但未提供 --rhpe_csv，忽略")

    if use_rhpe:
        ratio_sum = cfg.dataset_ratio_rsna + cfg.dataset_ratio_rhpe
        if abs(ratio_sum - 1.0) > 1e-6:
            raise SystemExit(
                f"[sampler] dataset sampling ratio 之和必须为 1，"
                f"得到 {ratio_sum:.4f}"
                f"（RSNA={cfg.dataset_ratio_rsna}, "
                f"RHPE={cfg.dataset_ratio_rhpe}）"
            )
        if cfg.dataset_ratio_rsna < 0 or cfg.dataset_ratio_rhpe < 0:
            raise SystemExit("[sampler] dataset ratio 不能为负")

    mode_label = "RSNA + RHPE Joint" if use_rhpe else "RSNA-only"
    print(f"[mode] {mode_label}")
    if cfg.zero_anatomy:
        print("[mode] zero_anatomy=True → 解剖关键点输入严格置零"
              "（Reviewer 9 w/o Anatomy；kp=None + roi_valid=0，"
              "bbox 裁剪等数据管线不变）")

    # ================================================================
    # 数据集构建（一次构造；归一化统计计算后回填 normalize）
    # ================================================================

    train_rsna = BoneAgeDataset(
        cfg.train_csv, cfg.train_image_dir, cfg,
        train=True, roi_json=cfg.roi_annotation or None,
        dataset_id=RSNA_ID, age_unit="months",
    )
    val_roi_json = derive_roi_json(cfg.roi_annotation, "val")
    val_rsna = BoneAgeDataset(
        cfg.val_csv, cfg.val_image_dir, cfg,
        train=False, roi_json=val_roi_json,
        dataset_id=RSNA_ID, age_unit="months",
    )

    test_ids = None
    if cfg.test_csv:
        test_df, _ = load_csv_meta(cfg.test_csv)
        test_ids = set(test_df["image_id"].astype(str))

    train_rhpe = None
    val_rhpe = None
    if use_rhpe:
        train_rhpe = BoneAgeDataset(
            cfg.rhpe_csv, cfg.rhpe_image_dir, cfg,
            train=True, roi_json=cfg.rhpe_roi_annotation or None,
            dataset_id=RHPE_ID, age_unit=cfg.rhpe_age_unit,
        )
        if cfg.rhpe_val_csv:
            rhpe_val_dir = cfg.rhpe_val_image_dir or cfg.rhpe_image_dir
            val_rhpe = BoneAgeDataset(
                cfg.rhpe_val_csv, rhpe_val_dir, cfg,
                train=False,
                roi_json=derive_roi_json(cfg.rhpe_roi_annotation, "val"),
                dataset_id=RHPE_ID, age_unit=cfg.rhpe_age_unit,
            )

    # ================================================================
    # 泄漏检查
    # ================================================================

    print("[leakage-check] RSNA train/val/test ID 交集 ...")
    check_leakage(train_rsna.id_set, val_rsna.id_set, test_ids, label="RSNA")
    print("[leakage-check] RSNA 通过：无重复 ID")

    if use_rhpe:
        rhpe_val_ids = val_rhpe.id_set if val_rhpe is not None else set()
        print("[leakage-check] RHPE train/val ID 交集 ...")
        check_leakage(train_rhpe.id_set, rhpe_val_ids, None, label="RHPE")
        print("[leakage-check] RHPE 通过：无重复 ID")

        # 跨数据集：ID 交集 + SHA256 图像内容检查
        id_sets = {
            "RSNA-train": train_rsna.id_set,
            "RSNA-val": val_rsna.id_set,
            "RHPE-train": train_rhpe.id_set,
        }
        image_lists = {
            "RSNA-train": train_rsna.image_paths(),
            "RSNA-val": val_rsna.image_paths(),
            "RHPE-train": train_rhpe.image_paths(),
        }
        if val_rhpe is not None:
            id_sets["RHPE-val"] = val_rhpe.id_set
            image_lists["RHPE-val"] = val_rhpe.image_paths()

        print("[leakage-check] 跨数据集 ID + SHA256 图像检查 ...")
        cross_dataset_leakage_check(
            id_sets, image_lists,
            hash_check=cfg.cross_dataset_hash_check,
            hash_max=cfg.cross_dataset_hash_max,
        )
    else:
        # RSNA-only：仍检查 train/val 图像内容重复（防同数据集重复图像）
        if cfg.cross_dataset_hash_check:
            print("[leakage-check] RSNA train/val SHA256 图像检查 ...")
            cross_dataset_leakage_check(
                {"RSNA-train": train_rsna.id_set,
                 "RSNA-val": val_rsna.id_set},
                {"RSNA-train": train_rsna.image_paths(),
                 "RSNA-val": val_rsna.image_paths()},
                hash_check=True,
                hash_max=cfg.cross_dataset_hash_max,
            )

    print("[leakage-check] 全部通过")

    # ================================================================
    # Chronological Age 可用性
    # ================================================================

    ca_available = use_rhpe and train_rhpe.has_ca
    cfg.use_age_embedding = bool(cfg.use_chronological_age and ca_available)

    if use_rhpe:
        print(f"[CA] availability: RSNA={train_rsna.ca_valid_rate:.1%} "
              f"RHPE={train_rhpe.ca_valid_rate:.1%}")
        if abs(train_rsna.ca_valid_rate - train_rhpe.ca_valid_rate) > 0.5:
            print("[CA] warning: CA availability 差异大 → "
                  "存在 dataset-specific shortcut 风险；"
                  "可用 --use_chronological_age false 做无 CA 消融")
    print(f"[CA] use_chronological_age={cfg.use_chronological_age} "
          f"→ use_age_embedding={cfg.use_age_embedding}")

    # ================================================================
    # 归一化统计
    # ================================================================

    stats = compute_normalization(
        cfg, out,
        train_rsna.image_paths(),
        train_rhpe.image_paths() if use_rhpe else None,
    )
    cfg.norm_mean = [float(m) for m in stats["mean"]]
    cfg.norm_std = [float(s) for s in stats["std"]]
    cfg.rhpe_norm_mean = [float(m) for m in stats["rhpe_mean"]]
    cfg.rhpe_norm_std = [float(s) for s in stats["rhpe_std"]]

    # 回填各 dataset 的 normalize（构造时 cfg 中统计尚未就绪）
    train_rsna.normalize = T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std)
    val_rsna.normalize = T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std)
    if use_rhpe:
        rhpe_mean = (cfg.rhpe_norm_mean
                     if cfg.normalization_mode == "dataset_specific"
                     else cfg.norm_mean)
        rhpe_std = (cfg.rhpe_norm_std
                    if cfg.normalization_mode == "dataset_specific"
                    else cfg.norm_std)
        train_rhpe.normalize = T.Normalize(mean=rhpe_mean, std=rhpe_std)
        if val_rhpe is not None:
            val_rhpe.normalize = T.Normalize(mean=rhpe_mean, std=rhpe_std)

    # ================================================================
    # Multi-Dataset + Sampler
    # ================================================================

    datasets = [train_rsna]
    if use_rhpe:
        datasets.append(train_rhpe)
    train_multi = MultiDatasetBoneAgeDataset(datasets)

    sampler = None
    sampler_stats: dict = {}

    if cfg.use_age_sampling:
        if use_rhpe:
            ratios = {
                RSNA_ID: float(cfg.dataset_ratio_rsna),
                RHPE_ID: float(cfg.dataset_ratio_rhpe),
            }
        else:
            ratios = {RSNA_ID: 1.0}

        sampler, sampler_stats = build_multi_dataset_age_sampler(
            train_multi.ages,
            train_multi.dataset_ids,
            cfg.age_bins,
            alpha=cfg.sampler_alpha,
            dataset_ratios=ratios,
            overflow=cfg.age_bin_overflow,
            verbose=True,
        )
        print(f"[sampler] Dataset-aware + Age-Binned α={cfg.sampler_alpha} "
              f"已启用")

        if cfg.sampler_debug_simulate > 0:
            sim = simulate_sampling(
                sampler, train_multi.dataset_ids, train_multi.ages,
                cfg.age_bins,
                num_samples=int(cfg.sampler_debug_simulate),
                seed=cfg.seed,
            )
            sampler_stats["simulation"] = sim
    else:
        print("[sampler] Age sampling 关闭 → uniform shuffle")

    (out / "sampler_statistics.json").write_text(
        json.dumps(sampler_stats, indent=2))

    train_loader = DataLoader(
        train_multi, batch_size=cfg.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_multi) > cfg.batch_size,
    )
    val_loader = DataLoader(
        val_rsna, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )
    rhpe_val_loader = None
    if val_rhpe is not None:
        rhpe_val_loader = DataLoader(
            val_rhpe, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
        )

    # ================================================================
    # Dataset Summary
    # ================================================================

    summary = {
        "mode": "joint" if use_rhpe else "rsna_only",
        "datasets": {
            "RSNA": {
                "train": len(train_rsna),
                "val": len(val_rsna),
                "test": len(test_ids) if test_ids else 0,
                "roi_rate": train_rsna.roi_valid_rate,
                "ca_rate": train_rsna.ca_valid_rate,
                "sex_rate": train_rsna.sex_valid_rate,
            },
        },
        "normalization": {
            "mode": cfg.normalization_mode,
            "mean": cfg.norm_mean,
            "std": cfg.norm_std,
            "rhpe_mean": cfg.rhpe_norm_mean if use_rhpe else None,
            "rhpe_std": cfg.rhpe_norm_std if use_rhpe else None,
        },
    }
    if use_rhpe:
        summary["datasets"]["RHPE"] = {
            "train": len(train_rhpe),
            "val": len(val_rhpe) if val_rhpe else 0,
            "test": 0,
            "roi_rate": train_rhpe.roi_valid_rate,
            "ca_rate": train_rhpe.ca_valid_rate,
            "sex_rate": train_rhpe.sex_valid_rate,
        }
    if cfg.use_age_sampling:
        ratio_map = {"RSNA": cfg.dataset_ratio_rsna}
        if use_rhpe:
            ratio_map["RHPE"] = cfg.dataset_ratio_rhpe
        summary["sampler"] = {
            "dataset_ratio": ratio_map,
            "alpha": cfg.sampler_alpha,
            "age_bins": list(cfg.age_bins),
        }

    print_dataset_summary(summary)
    (out / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # ================================================================
    # 模型 / 损失
    # ================================================================

    model = MTFN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MTFN 参数量 {n_params / 1e6:.2f}M, device={device}")
    print(f"[model] 开关: multiscale={cfg.use_multiscale} "
          f"age_attn={cfg.use_age_attention} sex={cfg.use_sex_embedding} "
          f"ca_embed={cfg.use_age_embedding} "
          f"ord={cfg.use_ordinal_head} dist={cfg.use_distribution_head} "
          f"cons={cfg.use_consistency_loss} "
          f"roi_branch={model.use_roi_branch} "
          f"roi_gating={model.use_roi_gating}")

    loss_fn = MultiTaskLoss(
        cfg,
        model.ordinal_head if cfg.use_ordinal_head else None,
        model.dist_head if cfg.use_distribution_head else None,
    )

    # config 输出：config.json+ config.yaml（可读）
    (out / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    save_config_yaml(out / "config.yaml", cfg.to_dict())

    # ================================================================
    # 训练
    # ================================================================

    def writer(row, va, improved, va_rhpe=None):
        epoch_writer(row, va, improved, va_rhpe=va_rhpe)
        if row["epoch"] == 0:
            # crop 记录
            try:
                train_multi.save_crop_records(out / "crop_records.csv")
                n_records = sum(len(ds.crop_records) for ds in datasets)
                if n_records == 0 and cfg.num_workers > 0:
                    print("[warning] num_workers>0 时 crop_records 由 worker "
                          "进程记录，主进程为空 → crop_records.csv 可能不完整",
                          flush=True)
            except OSError as e:
                print(f"[warning] crop_records.csv 写入失败: {e}", flush=True)

    trainer = Trainer(cfg, model, loss_fn, device)
    summary_res = trainer.fit(
        train_loader, val_loader, ckpt_dir, writer,
        rhpe_val_loader=rhpe_val_loader,
    )

    # ================================================================
    # RHPE validation 预测输出
    # ================================================================

    if rhpe_val_loader is not None:
        va_rhpe_final = trainer.evaluate(rhpe_val_loader,
                                         desc="         [val-RHPE final]")
        Trainer._write_predictions_csv(out / "predictions_rhpe_val.csv",
                                       va_rhpe_final)
        print(f"[RHPE-val] MAE={va_rhpe_final['basic']['mae']:.4f} "
              f"RMSE={va_rhpe_final['basic']['rmse']:.4f} "
              f"→ predictions_rhpe_val.csv（泛化参考）")

    # ================================================================
    # 融合策略锁定：仅用 RSNA VAL
    # ================================================================

    best_ckpt = torch.load(ckpt_dir / "best_rsna_model.pth",
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    mode_scores = {}
    with torch.no_grad():
        for mode in ("reg", "ord", "dist", "average", "learned"):
            p, y = predict_val_loader(model, val_loader, device, mode,
                                      trainer.amp_enabled)
            mode_scores[mode] = float(np.abs(p - y).mean())

    selected = min(mode_scores, key=mode_scores.get)
    (out / "fusion_selection.json").write_text(json.dumps(
        {"selected": selected, "val_mae_by_mode": mode_scores}, indent=2))

    print("\n[best] epoch={} RSNA val_mae={:.4f}".format(
        summary_res["best_epoch"], summary_res["best_val_mae"]))
    print("[fusion] 各策略 RSNA val MAE: " + "  ".join(
        f"{k}={v:.4f}" for k, v in mode_scores.items()))
    print(f"[fusion] 锁定策略 = {selected}（写入 fusion_selection.json，"
          f"test 评估使用，绝不用 test 选策略）")
    print("[done] 训练完成。最终 RSNA test 评估请运行 evaluate.py "
          "（训练与模型选择已全部完成，test 仅评估一次）")


if __name__ == "__main__":
    main()
