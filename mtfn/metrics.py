"""评估指标：总体 + 年龄分层 + 性别分层。"""
from __future__ import annotations

import numpy as np
from scipy import stats


def compute_basic(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    abs_err = np.abs(err)
    m = {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "medae": float(np.median(abs_err)),
        "acc_3": float((abs_err <= 3).mean()),
        "acc_6": float((abs_err <= 6).mean()),
        "acc_12": float((abs_err <= 12).mean()),
        # 相关系数需要 n≥2 且方差>0，小样本桶中置为 nan
        "pearson": float(stats.pearsonr(y_true, y_pred)[0])
        if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan"),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0])
        if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan"),
        "n": int(len(y_true)),
    }
    return m


def age_bin_index(y: np.ndarray, bins: list) -> np.ndarray:
    return np.digitize(y, bins[1:-1], right=False)


def stratified_by_age(y_true, y_pred, bins: list) -> dict:
    """每个年龄桶：age_bin, support, MAE, RMSE, ±6, ±12。"""
    idx = age_bin_index(y_true, bins)
    out = {}
    for k in range(len(bins) - 1):
        mask = idx == k
        if mask.sum() == 0:
            continue
        out[f"{bins[k]}-{bins[k + 1]}"] = compute_basic(
            np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        )
    return out


def stratified_by_sex(y_true, y_pred, male: np.ndarray) -> dict:
    """Male/Female 分层。"""
    out = {}
    for label, mask in (("male", male == 1), ("female", male == 0)):
        if mask.sum() == 0:
            continue
        out[label] = compute_basic(
            np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
        )
    return out


def format_basic(m: dict) -> str:
    return (
        f"MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  MedAE={m['medae']:.4f}  "
        f"±3={m['acc_3'] * 100:.2f}%  ±6={m['acc_6'] * 100:.2f}%  "
        f"±12={m['acc_12'] * 100:.2f}%  r={m['pearson']:.4f}"
    )
