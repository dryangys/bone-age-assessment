"""TEST 评估：模型锁定后一次性评估。

- batch 为统一 8 元组（x, y, sex, ca, kp, roi_valid, img_id, dataset_id）
- fusion 策略默认读取训练侧 fusion_selection.json（val 锁定，绝不用 test 选策略）
- test CSV 无真值时仅输出 predictions_rsna_test.csv（image_id, predicted_bone_age），
  绝不伪造 test MAE
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtfn.config import MTFNConfig
from mtfn.dataset import BoneAgeDataset, load_csv_meta
from mtfn.metrics import (compute_basic, format_basic, stratified_by_age,
                          stratified_by_sex)
from mtfn.model import MTFN
from mtfn.roi_encoder import derive_roi_json


def age_bin_label(y: float, bins: list) -> str:
    for k in range(len(bins) - 1):
        if bins[k] <= y < bins[k + 1]:
            return f"{bins[k]}-{bins[k + 1]}"
    return f"{bins[-2]}+"


def main() -> None:
    p = argparse.ArgumentParser(description="MTFN 测试集评估（模型锁定后使用）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--test_image_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="test_results")
    p.add_argument("--fusion_mode", type=str, default=None,
                   choices=["learned", "reg", "ord", "dist", "average"],
                   help="默认读取训练侧 fusion_selection.json 锁定的策略")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MTFNConfig.from_dict(state["config"])
    # 评估阶段禁止沿用训练调试的截断参数，避免静默漏评
    cfg.limit_train = 0
    cfg.limit_val = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = args.fusion_mode
    sel_file = Path(args.ckpt).parent.parent / "fusion_selection.json"
    if mode is None:
        if sel_file.exists():
            mode = json.loads(sel_file.read_text())["selected"]
        else:
            mode = "learned"
    print(f"[eval] ckpt={args.ckpt} (epoch={state.get('epoch')}, "
          f"val_mae={state.get('val_mae')})")
    print(f"[eval] 融合策略 = {mode}"
          + ("（来自 fusion_selection.json，val 锁定）" if args.fusion_mode is None else ""))

    model = MTFN(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    test_roi_json = derive_roi_json(cfg.roi_annotation, "test")
    ds = BoneAgeDataset(args.test_csv, args.test_image_dir, cfg, train=False,
                        roi_json=test_roi_json, dataset_id=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")

    # 真值存在性
    meta_df, _ = load_csv_meta(args.test_csv)
    has_label = bool(meta_df["bone_age"].notna().any())

    preds, ys, sexes, ids = [], [], [], []
    with torch.no_grad():
        for x, y, sex, ca, kp, roi_valid, img_id, _ in loader:
            kwargs = {}
            if getattr(model, "use_ca", False):
                kwargs["ca"] = ca.to(device).float()
            if getattr(model, "use_roi_branch", False):
                kwargs["roi_valid"] = roi_valid.to(device)
            with torch.autocast(device.type, enabled=cfg.amp and device.type == "cuda"):
                out = model.predict(x.to(device), sex.to(device),
                                    kp=kp.to(device), mode=mode, **kwargs)
            preds.append(out.float().cpu())
            ys.append(y)
            sexes.append(sex)
            ids.extend(img_id)
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(ys).numpy()
    male = torch.cat(sexes).numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_label:
        # 有真值：完整预测记录
        pred_csv = out_dir / "predictions_rsna_test.csv"
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "true_bone_age", "pred_bone_age", "error",
                        "absolute_error", "male", "age_bin"])
            for i, img_id in enumerate(ids):
                e = float(y_pred[i] - y_true[i])
                w.writerow([img_id, round(float(y_true[i]), 1),
                            round(float(y_pred[i]), 3), round(e, 3),
                            round(abs(e), 3), int(male[i]),
                            age_bin_label(float(y_true[i]), cfg.age_bins)])
        print(f"[eval] 预测已保存: {pred_csv} ({len(ids)} 例)")
    else:
        # 无真值：仅 image_id + predicted_bone_age
        pred_csv = out_dir / "predictions_rsna_test.csv"
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "predicted_bone_age"])
            for i, img_id in enumerate(ids):
                w.writerow([img_id, round(float(y_pred[i]), 3)])
        print(f"[eval] test CSV 无 bone_age 真值 → 仅输出预测: {pred_csv} "
              f"({len(ids)} 例)，不计算 test MAE（禁止伪造）")
        print("[eval] 诚实性能以 RSNA validation MAE 为准（best checkpoint 内 val_mae）")
        return

    basic = compute_basic(y_true, y_pred)
    by_age = stratified_by_age(y_true, y_pred, cfg.age_bins)
    by_sex = stratified_by_sex(y_true, y_pred, male)

    print("\n===== RSNA TEST 总体 =====")
    print(format_basic(basic))
    print("\n===== 年龄分层 =====")
    print(f"{'age_bin':<12}{'n':>6}{'MAE':>9}{'RMSE':>9}{'±6':>8}{'±12':>8}")
    for name, m in by_age.items():
        print(f"{name:<12}{m['n']:>6}{m['mae']:>9.3f}{m['rmse']:>9.3f}"
              f"{m['acc_6'] * 100:>7.1f}%{m['acc_12'] * 100:>7.1f}%")
    print("\n===== 性别分层 =====")
    for name, m in by_sex.items():
        print(f"{name:<8} n={m['n']:>5}  {format_basic(m)}")

    summary = {
        "ckpt": args.ckpt, "fusion_mode": mode,
        "test_mae": basic["mae"], "test_rmse": basic["rmse"],
        "test_medae": basic["medae"], "test_acc_3": basic["acc_3"],
        "test_acc_6": basic["acc_6"], "test_acc_12": basic["acc_12"],
        "test_pearson": basic["pearson"], "test_spearman": basic["spearman"],
        "male_mae": by_sex.get("male", {}).get("mae"),
        "female_mae": by_sex.get("female", {}).get("mae"),
        "age_bin_mae": {k: v["mae"] for k, v in by_age.items()},
    }
    (out_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[eval] 汇总已保存: {out_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
