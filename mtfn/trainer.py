"""MTFN 训练器

关键原则：
- Primary metric = RSNA validation MAE（best checkpoint / early stopping 唯一依据）
- RHPE validation（若存在）仅作为泛化参考记录，绝不影响模型选择
- 保存 best_rsna_model.pth（同时写 best_model.pt 别名保持兼容）
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # 未安装 tqdm 时退化为普通迭代
    tqdm = None

from .config import RHPE_ID, RSNA_ID
from .loss import MultiTaskLoss
from .metrics import compute_basic, format_basic, stratified_by_age, stratified_by_sex


class Trainer:
    def __init__(self, cfg, model: nn.Module, loss_fn: MultiTaskLoss,
                 device: torch.device):
        self.cfg = cfg
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.amp_enabled = cfg.amp and device.type == "cuda"
        if self.amp_enabled:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None  # CPU 或关闭 AMP 时无需 scaler

        # 差分学习率：backbone / 新增模块 两组
        backbone_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                head_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": cfg.backbone_lr},
                {"params": head_params, "lr": cfg.head_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

        self.best_val_mae = float("inf")   # RSNA validation MAE（primary）
        self.best_epoch = -1
        self.patience_counter = 0

    # ------------------------------------------------------------------
    def lr_factor(self, epoch: int) -> float:
        """Warmup + Cosine Annealing。"""
        warm = max(1, self.cfg.warmup_epochs)
        if epoch < warm:
            return (epoch + 1) / warm
        t = (epoch - warm) / max(1, self.cfg.epochs - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))

    def apply_freeze_schedule(self, epoch: int) -> str:
        """分阶段训练：A 冻结 / B 解冻 Stage3-4 / C 全解冻。"""
        cfg = self.cfg
        stage = "full"
        if cfg.freeze_backbone_epochs > 0 and epoch < cfg.freeze_backbone_epochs:
            stage = "A_frozen"
        elif (cfg.freeze_backbone_epochs > 0 and cfg.stage_b_epochs > 0
              and epoch < cfg.freeze_backbone_epochs + cfg.stage_b_epochs):
            stage = "B_stage34"
        for name, p in self.model.named_parameters():
            if not name.startswith("backbone."):
                p.requires_grad = True
                continue
            idx = None
            if name.startswith("backbone.swin.features."):
                try:
                    idx = int(name.split(".")[3].split(".")[0])
                except (ValueError, IndexError):
                    idx = None
            if stage == "A_frozen":
                p.requires_grad = False
            elif stage == "B_stage34":
                # features[0..4]=stem+stage1+merge+stage2 冻结；5..7=stage3+merge+stage4 解冻
                p.requires_grad = True if (idx is not None and idx >= 5) else False
            else:
                p.requires_grad = True
        return stage

    # ------------------------------------------------------------------
    def _model_inputs(self, x, sex, ca, kp, roi_valid):
        """组装模型输入：CA 仅在启用时传递（-1 无效值在模型内安全退化）。"""
        kwargs = {}

        # Reviewer 9 w/o Anatomy：关键点输入严格置零。
        # kp=None → 模型内部生成 zero tensor；roi_valid=0 → ROI gating
        # 将 roi_feat 精确置零。架构占位保留，数据管线（bbox 裁剪）不变。
        if getattr(self.cfg, "zero_anatomy", False):
            if getattr(self.model, "use_roi_branch", False) and roi_valid is not None:
                kwargs["roi_valid"] = torch.zeros_like(roi_valid).to(self.device)
            return (
                x.to(self.device),
                sex.to(self.device),
                None,
                kwargs,
            )

        if getattr(self.model, "use_ca", False) and ca is not None:
            kwargs["ca"] = ca.to(self.device).float()

        if getattr(self.model, "use_roi_branch", False) and roi_valid is not None:
            kwargs["roi_valid"] = roi_valid.to(self.device)

        return (
            x.to(self.device),
            sex.to(self.device),
            kp.to(self.device),
            kwargs,
        )

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        factor = self.lr_factor(epoch)

        # 差分 lr × warmup/cosine 因子
        if not hasattr(self, "_base_lrs"):
            self._base_lrs = [self.cfg.backbone_lr, self.cfg.head_lr]
        for g, base in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base * factor

        losses_sum, n_batches = 0.0, 0
        loss_parts = {"reg": 0.0, "ord": 0.0, "dist": 0.0, "fused": 0.0, "cons": 0.0}
        preds, targets = [], []
        seen = {RSNA_ID: 0, RHPE_ID: 0}
        t0 = time.time()
        use_cons = self.cfg.use_consistency_loss

        pbar = (tqdm(loader, desc=f"epoch {epoch:>3} [train]",
                     leave=False, unit="batch")
                if tqdm is not None else loader)
        for batch in pbar:
            # 统一 9 元组：
            # x, x_aug, y, sex, ca, kp, roi_valid, img_id, dataset_id
            x, x_aug, y, sex, ca, kp, roi_valid, _, dataset_id = batch

            y = y.to(self.device).float()

            # dataset seen 统计
            for did, cnt in zip(*np.unique(dataset_id.numpy(), return_counts=True)):
                seen[int(did)] = seen.get(int(did), 0) + int(cnt)

            xi, sex_i, kp_i, extra = self._model_inputs(x, sex, ca, kp, roi_valid)

            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                out = self.model(xi, sex_i, kp=kp_i, **extra)
                y_aug = None
                if use_cons:
                    xa, sex_a, kp_a, extra_a = self._model_inputs(
                        x_aug, sex, ca, kp, roi_valid)
                    out_aug = self.model(xa, sex_a, kp=kp_a, **extra_a)
                    y_aug = out_aug["fused"]
                loss, loss_dict = self.loss_fn(out, y, y_aug)

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

            losses_sum += float(loss)
            n_batches += 1
            for k in loss_parts:
                if k in loss_dict:
                    loss_parts[k] += float(loss_dict[k])
            preds.append(out["fused"].detach().float().cpu())
            targets.append(y.cpu())
            if tqdm is not None:
                pbar.set_postfix(loss=f"{losses_sum / n_batches:.4f}",
                                 lr=f"{self.optimizer.param_groups[-1]['lr']:.2e}")

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        m = compute_basic(targets, preds)
        m["loss"] = losses_sum / max(1, n_batches)
        for k in loss_parts:
            m[f"loss_{k}"] = loss_parts[k] / max(1, n_batches)
        m["time_s"] = time.time() - t0
        m["lr_factor"] = factor
        m["seen"] = seen
        return m

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "[val]") -> dict:
        self.model.eval()
        preds_all, targets_all, sex_all, ids_all = [], [], [], []
        head_preds = {"reg": [], "ord": [], "dist": []}
        pbar = (tqdm(loader, desc=desc,
                     leave=False, unit="batch")
                if tqdm is not None else loader)
        # 统一 8 元组：x, y, sex, ca, kp, roi_valid, img_id, dataset_id
        for x, y, sex, ca, kp, roi_valid, img_id, _ in pbar:
            xi, sex_i, kp_i, extra = self._model_inputs(x, sex, ca, kp, roi_valid)
            with torch.autocast(self.device.type, enabled=self.amp_enabled):
                out = self.model(xi, sex_i, kp=kp_i, **extra)
            for key, val in self.model.head_predictions(out).items():
                head_preds.setdefault(key, []).append(val.float().cpu())
            preds_all.append(out["fused"].float().cpu())
            targets_all.append(y)
            sex_all.append(sex.cpu())
            ids_all.extend(list(img_id))

        y_true = torch.cat(targets_all).numpy()
        y_pred = torch.cat(preds_all).numpy()
        male = torch.cat(sex_all).numpy()
        result = {"basic": compute_basic(y_true, y_pred),
                  "by_age": stratified_by_age(y_true, y_pred, self.cfg.age_bins),
                  "by_sex": stratified_by_sex(y_true, y_pred, male)}
        # 各头单独 MAE
        for key in ("reg", "ord", "dist"):
            if head_preds.get(key):
                p = torch.cat(head_preds[key]).numpy()
                result[f"mae_{key}"] = float(np.abs(p - y_true).mean())
        if head_preds.get("fused"):
            result["mae_fused"] = float(np.abs(
                torch.cat(head_preds["fused"]).numpy() - y_true).mean())
        result["ids"] = ids_all
        result["y_true"] = y_true
        result["y_pred"] = y_pred
        result["male"] = male
        return result

    # ------------------------------------------------------------------
    def _save_checkpoint(self, ckpt_dir: Path, epoch: int, va_rsna: dict) -> None:
        """保存 best checkpoint。"""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {"lr_factor": self.lr_factor(epoch)},
            "epoch": epoch,
            "best_val_mae": self.best_val_mae,
            "val_mae": self.best_val_mae,       # RSNA validation MAE
            "config": self.cfg.to_dict(),
            "fusion_weights": self.model.fusion_weights(),
            "age_bins": list(self.cfg.age_bins),
            "dataset_sampling_ratio": {
                "RSNA": self.cfg.dataset_ratio_rsna,
                "RHPE": self.cfg.dataset_ratio_rhpe,
            },
            "normalization": {
                "mean": list(self.cfg.norm_mean),
                "std": list(self.cfg.norm_std),
            },
        }
        torch.save(state, ckpt_dir / "best_rsna_model.pth")
        # 别名：best_model.pt + best_model.pth
        # 硬链接，节省磁盘；失败时退化为完整拷贝
        for alias_name in ("best_model.pt", "best_model.pth"):
            alias = ckpt_dir / alias_name
            if alias.exists():
                alias.unlink()
            try:
                os.link(ckpt_dir / "best_rsna_model.pth", alias)
            except OSError:
                torch.save(state, alias)

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader, ckpt_dir: Path, writer,
            rhpe_val_loader=None) -> dict:
        metrics_rows = []
        target_ratio = None
        n_seen_total = {RSNA_ID: 0, RHPE_ID: 0}

        if rhpe_val_loader is not None and self.cfg.dataset_ratio_rhpe > 0:
            target_ratio = self.cfg.dataset_ratio_rhpe

        for epoch in range(self.cfg.epochs):
            stage = self.apply_freeze_schedule(epoch)
            tr = self.train_one_epoch(train_loader, epoch)
            va = self.evaluate(val_loader, desc="         [val-RSNA]")

            # RHPE validation
            va_rhpe = None
            if rhpe_val_loader is not None:
                va_rhpe = self.evaluate(rhpe_val_loader, desc="         [val-RHPE]")

            # dataset seen 统计
            for k, v in tr["seen"].items():
                n_seen_total[k] = n_seen_total.get(k, 0) + v
            total_seen = sum(n_seen_total.values())

            fw = self.model.fusion_weights()

            row = {
                "epoch": epoch,
                "stage": stage,
                "train_loss": round(tr["loss"], 6),
                "loss_reg": round(tr.get("loss_reg", 0.0), 6),
                "loss_ord": round(tr.get("loss_ord", 0.0), 6),
                "loss_dist": round(tr.get("loss_dist", 0.0), 6),
                "loss_fused": round(tr.get("loss_fused", 0.0), 6),
                "train_mae": round(tr["mae"], 4),
                # primary：RSNA validation MAE
                "val_mae_rsna": round(va["basic"]["mae"], 4),
                "val_rmse_rsna": round(va["basic"]["rmse"], 4),
                "val_medae_rsna": round(va["basic"]["medae"], 4),
                "val_acc_3": round(va["basic"]["acc_3"], 4),
                "val_acc_6": round(va["basic"]["acc_6"], 4),
                "val_acc_12": round(va["basic"]["acc_12"], 4),
                "mae_reg": round(va.get("mae_reg", float("nan")), 4),
                "mae_ord": round(va.get("mae_ord", float("nan")), 4),
                "mae_dist": round(va.get("mae_dist", float("nan")), 4),
                "mae_fused": round(va.get("mae_fused", float("nan")), 4),
                "w_reg": round(fw.get("reg", 0.0), 4),
                "w_ord": round(fw.get("ord", 0.0), 4),
                "w_dist": round(fw.get("dist", 0.0), 4),
                "backbone_lr": self.optimizer.param_groups[0]["lr"],
                "head_lr": self.optimizer.param_groups[-1]["lr"],
                "rsna_seen": n_seen_total.get(RSNA_ID, 0),
                "rhpe_seen": n_seen_total.get(RHPE_ID, 0),
                "time_s": round(tr["time_s"], 1),
            }
            if va_rhpe is not None:
                row["val_mae_rhpe"] = round(va_rhpe["basic"]["mae"], 4)
                row["val_rmse_rhpe"] = round(va_rhpe["basic"]["rmse"], 4)

            # batch dataset ratio 偏离 warning
            if target_ratio is not None and total_seen > 0:
                observed = n_seen_total.get(RHPE_ID, 0) / total_seen
                if abs(observed - target_ratio) > self.cfg.dataset_ratio_warn_tol:
                    print(
                        f"[warning] dataset ratio 偏离："
                        f"RHPE observed={observed:.3f} "
                        f"target={target_ratio:.3f}"
                    )

            metrics_rows.append(row)

            # ------------------------------------------------------------------
            # best checkpoint / early stopping：仅依据 RSNA val MAE
            # 
            # ------------------------------------------------------------------
            improved = va["basic"]["mae"] < self.best_val_mae
            if improved:
                self.best_val_mae = va["basic"]["mae"]
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint(ckpt_dir, epoch, va)
            else:
                self.patience_counter += 1

            last_state = {
                "model": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "epoch": epoch,
            }
            torch.save(last_state, ckpt_dir / "last_model.pt")
            # last_model.pth 别名
            last_alias = ckpt_dir / "last_model.pth"
            if last_alias.exists():
                last_alias.unlink()
            try:
                os.link(ckpt_dir / "last_model.pt", last_alias)
            except OSError:
                torch.save(last_state, last_alias)

            self._write_metrics_csv(ckpt_dir / "metrics.csv", metrics_rows)
            writer(row, va, improved, va_rhpe=va_rhpe)

            if (self.cfg.early_stopping_patience > 0
                    and self.patience_counter >= self.cfg.early_stopping_patience):
                print(f"[early-stop] patience={self.patience_counter}/"
                      f"{self.cfg.early_stopping_patience} → 停止"
                      f"")
                break

        # 保存最佳 epoch 的验证集预测（best RSNA checkpoint 重新评估）
        best = torch.load(ckpt_dir / "best_rsna_model.pth",
                          map_location=self.device, weights_only=False)
        self.model.load_state_dict(best["model"])
        va = self.evaluate(val_loader, desc="         [val-RSNA]")
        self._write_predictions_csv(ckpt_dir / "predictions.csv", va)
        return {"best_val_mae": self.best_val_mae, "best_epoch": self.best_epoch,
                "final_val": va}

    # ------------------------------------------------------------------
    @staticmethod
    def _write_metrics_csv(path: Path, rows: list) -> None:
        if not rows:
            return
        # union of keys（RHPE 列仅在存在时写出）
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _write_predictions_csv(path: Path, va: dict) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "true_bone_age", "pred_bone_age",
                        "error", "absolute_error", "male"])
            for i, img_id in enumerate(va["ids"]):
                e = va["y_pred"][i] - va["y_true"][i]
                w.writerow([img_id, va["y_true"][i], round(float(va["y_pred"][i]), 3),
                            round(float(e), 3), round(float(abs(e)), 3),
                            int(va["male"][i])])


def save_config_yaml(path: Path, cfg_dict: dict) -> None:
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(cfg_dict, f, allow_unicode=True, sort_keys=False)
    except ImportError:
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(cfg_dict, f, ensure_ascii=False, indent=2)
