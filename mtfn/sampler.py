"""MTFN Dataset-aware + Age-aware Sampling

数学定义：

    w_age(i)     = (N / N_k)^alpha          # 年龄平衡，k = age bin of i
    d_R          = p_R / (N_R / N)          # 从 sampling probability 推导权重
    d_H          = p_H / (N_H / N)
    w_i          = d_dataset(i) × w_age(i)
    w_i          = w_i / mean(w_i)          # 整体归一化

要求：
- replacement=True
- num_samples=len(dataset)
- dtype=torch.double
- weights finite / > 0

Sampler 只允许用于 TRAIN。
"""

from __future__ import annotations

import numpy as np
import torch

from torch.utils.data import (
    WeightedRandomSampler,
)

from .config import RHPE_ID, RSNA_ID


# ================================================================
# Age bin 统计
# ================================================================

def compute_age_bins(
    ages: np.ndarray,
    bins: list,
    overflow: str = "clip",
) -> tuple:
    """返回 (bin_idx, n_bins, overflow_count)。

    超界年龄（age < bins[0] 或 age >= bins[-1]）：
        clip  → 归入边界 bin（digitize 自然行为），打印统计
        error → 直接报错
    """

    ages = np.asarray(ages, dtype=np.float64)
    bins_arr = np.asarray(bins, dtype=np.float64)

    if not np.isfinite(ages).all():
        raise ValueError("ages 中存在 NaN / Inf")

    # 超界统计（clip 模式下 digitize 自动归入边界 bin，但必须打印）
    n_below = int((ages < bins_arr[0]).sum())
    n_above = int((ages >= bins_arr[-1]).sum())

    if (n_below > 0 or n_above > 0) and overflow == "error":
        raise SystemExit(
            f"[sampler] 年龄超出 bins 范围 "
            f"[{bins_arr[0]}, {bins_arr[-1]})："
            f"below={n_below}, above={n_above}。"
            f"当前模式 error → 停止（可改 age_bin_overflow=clip）"
        )

    if n_below > 0 or n_above > 0:

        print(
            f"[sampler] [warning] 年龄超界："
            f"<{bins_arr[0]:.0f} 月 {n_below} 例，"
            f">={bins_arr[-1]:.0f} 月 {n_above} 例 "
            f"→ clip 到边界 bin（模式={overflow}）"
        )

    # digitize: ages < bins[0] → 0；ages >= bins[-1] → n_bins-1（右边界收尾）
    bin_idx = np.digitize(
        ages,
        bins_arr[1:-1],
        right=False,
    )

    # ages >= bins[-1] → digitize 返回 n_bins-1 已正确（最后一个 bin 半开区间收尾）
    bin_idx = np.clip(bin_idx, 0, len(bins_arr) - 2)

    return bin_idx, len(bins_arr) - 1, n_below + n_above


# ================================================================
# Multi-Dataset Age-Balanced Sampler
# ================================================================

def build_multi_dataset_age_sampler(
    ages: np.ndarray,
    dataset_ids: np.ndarray,
    bins: list,
    alpha: float = 0.5,
    dataset_ratios: dict | None = None,
    overflow: str = "clip",
    verbose: bool = True,
) -> tuple:
    """构建 Dataset-aware + Age-balanced WeightedRandomSampler。

    参数：
        ages          : [N] bone age（months）
        dataset_ids   : [N] 0=RSNA, 1=RHPE
        bins          : 年龄分桶边界（月）
        alpha         : 年龄平衡指数
        dataset_ratios: {RSNA_ID: p_R, RHPE_ID: p_H}，默认 0.7/0.3
        overflow      : "clip" | "error"

    返回：(sampler, statistics_dict)
    """

    # ============================================================
    # 输入校验
    # ============================================================

    ages = np.asarray(ages, dtype=np.float64)
    dataset_ids = np.asarray(dataset_ids, dtype=np.int64)

    if ages.ndim != 1 or dataset_ids.ndim != 1:
        raise ValueError("ages / dataset_ids 必须是一维数组")

    if len(ages) == 0:
        raise ValueError("样本为空，无法构建 sampler")

    if len(ages) != len(dataset_ids):
        raise ValueError(
            f"ages({len(ages)}) 与 dataset_ids({len(dataset_ids)}) 长度不一致"
        )

    bins_arr = np.asarray(bins, dtype=np.float64)

    if bins_arr.ndim != 1 or len(bins_arr) < 2:
        raise ValueError("bins 必须是一维且至少两个边界")

    if not np.all(np.diff(bins_arr) > 0):
        raise ValueError(f"bins 必须严格递增: {bins_arr.tolist()}")

    if alpha < 0:
        raise ValueError(f"alpha 必须 >= 0，得到 {alpha}")

    if dataset_ratios is None:
        dataset_ratios = {RSNA_ID: 0.7, RHPE_ID: 0.3}

    # dataset ratio 合法性
    present_ids = sorted(set(dataset_ids.tolist()))
    ratios_used = {}

    for did in present_ids:

        if did not in dataset_ratios:
            raise SystemExit(
                f"[sampler] dataset_id={did} 无对应 sampling ratio，"
                f"现有 ratios={dataset_ratios}"
            )

        ratios_used[did] = float(dataset_ratios[did])

    ratio_sum = sum(ratios_used.values())

    if len(ratios_used) > 1 and abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(
            f"[sampler] dataset sampling ratio 之和必须为 1，"
            f"得到 {ratio_sum:.4f}（{ratios_used}）"
        )

    if any(r < 0 for r in ratios_used.values()):
        raise SystemExit(
            f"[sampler] dataset ratio 不能为负: {ratios_used}"
        )

    # ============================================================
    # Age bin 分配（含超界处理）
    # ============================================================

    bin_idx, n_bins, _ = compute_age_bins(
        ages,
        bins,
        overflow=overflow,
    )

    # ============================================================
    # 1) 年龄权重 w_age(i) = (N / N_k)^alpha
    #    （全局 bin 统计，RSNA + RHPE 合并）
    # ============================================================

    counts = np.bincount(
        bin_idx,
        minlength=n_bins,
    ).astype(np.float64)

    # 空 bin 权重安全处理（空 bin 无样本，不会分配任何权重）
    safe_counts = np.maximum(counts, 1.0)

    n_total = float(len(ages))

    w_bin = (n_total / safe_counts) ** float(alpha)

    w_age = w_bin[bin_idx]  # [N]

    # ============================================================
    # 2) Dataset 权重 d_R = p_R / (N_R / N)
    # ============================================================

    dataset_counts = {
        did: float((dataset_ids == did).sum())
        for did in present_ids
    }

    dataset_weight = {}

    for did in present_ids:

        n_d = dataset_counts[did]

        if n_d <= 0:
            raise SystemExit(
                f"[sampler] dataset {did} 样本数为 0 但出现在 dataset_ids"
            )

        share = n_d / n_total  # N_dataset / N

        dataset_weight[did] = ratios_used[did] / share

    w_dataset = np.asarray(
        [dataset_weight[d] for d in dataset_ids],
        dtype=np.float64,
    )  # [N]

    # ============================================================
    # 3) 最终权重 + 整体归一化
    # ============================================================

    w_sample = w_dataset * w_age

    if not np.isfinite(w_sample).all() or (w_sample <= 0).any():
        raise SystemExit(
            "[sampler] 权重存在非 finite 或非正值，拒绝构建"
        )

    w_sample = w_sample / np.mean(w_sample)

    # ============================================================
    # 统计输出
    # ============================================================

    stats = {
        "total_samples": int(n_total),
        "dataset_counts": {
            str(d): int(dataset_counts[d]) for d in present_ids
        },
        "dataset_ratios": {
            str(d): ratios_used[d] for d in present_ids
        },
        "dataset_weights": {
            str(d): dataset_weight[d] for d in present_ids
        },
        "alpha": float(alpha),
        "bins": [float(b) for b in bins_arr],
        "n_bins": int(n_bins),
        "bin_counts": {
            f"{bins_arr[i]:.0f}-{bins_arr[i + 1]:.0f}": int(counts[i])
            for i in range(n_bins)
        },
        "bin_age_weights": {
            f"{bins_arr[i]:.0f}-{bins_arr[i + 1]:.0f}": float(w_bin[i])
            for i in range(n_bins)
        },
        "per_bin_dataset_counts": {},
        "expected_sampling_prob": {
            str(d): float(
                (w_sample[dataset_ids == d]).sum()
                / w_sample.sum()
            )
            for d in present_ids
        },
    }

    for i in range(n_bins):

        label = f"{bins_arr[i]:.0f}-{bins_arr[i + 1]:.0f}"
        mask = bin_idx == i

        stats["per_bin_dataset_counts"][label] = {
            str(d): int((mask & (dataset_ids == d)).sum())
            for d in present_ids
        }

    if verbose:

        names = {RSNA_ID: "RSNA", RHPE_ID: "RHPE"}

        print(
            "\n[Multi-Dataset Age-Balanced Sampler]"
        )

        print(f"  Total samples: {int(n_total)}")

        for d in present_ids:

            print(
                f"  {names.get(d, f'DS{d}')}: "
                f"{int(dataset_counts[d])} "
                f"(ratio={ratios_used[d]:.2f}, "
                f"d_weight={dataset_weight[d]:.4f})"
            )

        print(f"  Age bins: {[float(b) for b in bins_arr]}")
        print(f"  Alpha: {alpha}")

        print("\n  Bin statistics (dataset count / age weight / final weight):")

        for i in range(n_bins):

            label = (
                f"{bins_arr[i]:.0f}-"
                f"{bins_arr[i + 1]:.0f}"
            )

            mask = bin_idx == i

            w_mean = float(w_sample[mask].mean()) if mask.any() else 0.0

            parts = " ".join(
                f"{names.get(d, f'DS{d}')}="
                f"{int((mask & (dataset_ids == d)).sum())}"
                for d in present_ids
            )

            print(
                f"    {label:<10} "
                f"count={int(counts[i]):>6} "
                f"[{parts}] "
                f"w_age={w_bin[i]:.3f} "
                f"w_final={w_mean:.3f}"
            )

        print("\n  Expected sampling probability:")

        for d in present_ids:

            p = stats["expected_sampling_prob"][str(d)]

            print(
                f"    {names.get(d, f'DS{d}')} = "
                f"{p:.4f} (target {ratios_used[d]:.2f})"
            )

        print()

    # ============================================================
    # WeightedRandomSampler
    # ============================================================

    weights = torch.as_tensor(
        w_sample,
        dtype=torch.double,
    )

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(ages),
        replacement=True,
    )

    return sampler, stats


# ================================================================
# 旧接口兼容
# ================================================================

def build_age_binned_sampler(
    ages: np.ndarray,
    bins: list,
    alpha: float = 0.5,
) -> WeightedRandomSampler:
    """RSNA-only age-binned sampler。"""

    dataset_ids = np.zeros(
        len(ages),
        dtype=np.int64,
    )

    sampler, _ = build_multi_dataset_age_sampler(
        ages,
        dataset_ids,
        bins,
        alpha=alpha,
        dataset_ratios={RSNA_ID: 1.0},
        verbose=True,
    )

    return sampler


# ================================================================
# Sampler 验证
# ================================================================

def simulate_sampling(
    sampler: WeightedRandomSampler,
    dataset_ids: np.ndarray,
    ages: np.ndarray,
    bins: list,
    num_samples: int = 100000,
    seed: int = 42,
) -> dict:
    """模拟采样验证 dataset ratio / age-bin ratio 是否生效。

    仅在初始化时调用，每个 epoch 不得运行。
    """

    torch.manual_seed(seed)

    idx = torch.multinomial(
        sampler.weights,
        num_samples=num_samples,
        replacement=True,
    ).numpy()

    ds_sim = dataset_ids[idx]
    ages_sim = np.asarray(ages, dtype=np.float64)[idx]

    names = {RSNA_ID: "RSNA", RHPE_ID: "RHPE"}

    result = {}

    print(f"\n[sampler-simulate] 模拟 {num_samples} 次采样：")

    for did in sorted(set(dataset_ids.tolist())):

        ratio = float((ds_sim == did).mean())

        result[f"dataset_{did}"] = ratio

        print(
            f"  {names.get(did, f'DS{did}')} observed ratio = "
            f"{ratio:.4f}"
        )

    bin_idx, n_bins, _ = compute_age_bins(
        ages_sim,
        bins,
        overflow="clip",
    )

    bins_arr = np.asarray(bins, dtype=np.float64)

    print("  Age-bin observed ratio:")

    for i in range(n_bins):

        label = (
            f"{bins_arr[i]:.0f}-"
            f"{bins_arr[i + 1]:.0f}"
        )

        ratio = float((bin_idx == i).mean())

        result[f"bin_{label}"] = ratio

        print(f"    {label:<10} {ratio:.4f}")

    print()

    return result
