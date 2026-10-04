"""Anatomical ROI 分支。

真实标注数据检查结果（2019 Uniandes 版）：COCO 格式，
每图 1 个 hand bbox + 17 个手部关键点（5 指×3 关节 + 2 腕锚点），
不是逐骨 ROI。因此本分支实现为：
1) 标注 hand bbox → 精确手部裁剪（优于 Otsu 自动检测，缺失自动回退）
2) 17 关键点（归一化坐标 + 可见性，51 维）→ MLP 编码 → 特征融合

无标注文件时自动退化（不报错），use_roi_branch 提供消融开关。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_KEYPOINTS = 17
KP_DIM = N_KEYPOINTS * 3  # (x, y, v) 归一化


def load_roi_annotations(json_path: str) -> dict | None:
    """加载 COCO 关键点标注 → {file_stem: {"bbox": (x1,y1,x2,y2), "kp": [51]}}。"""
    if not json_path or not Path(json_path).exists():
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    if "annotations" not in data or "images" not in data:
        return None

    img_meta = {im["id"]: im for im in data["images"]}
    table: dict = {}
    for ann in data["annotations"]:
        im = img_meta.get(ann["image_id"])
        if im is None:
            continue
        w, h = float(im.get("width", 1)), float(im.get("height", 1))
        x, y, bw, bh = ann["bbox"]  # COCO: [x, y, w, h]
        kp = np.asarray(ann.get("keypoints", []), dtype=np.float32)
        if kp.size == KP_DIM:
            kp = kp.copy()
            kp[0::3] = np.clip(kp[0::3] / max(w, 1), 0, 1)   # x 归一化
            kp[1::3] = np.clip(kp[1::3] / max(h, 1), 0, 1)   # y 归一化
            kp[2::3] = (kp[2::3] > 0).astype(np.float32)     # 可见性 0/1
        else:
            kp = np.zeros(KP_DIM, dtype=np.float32)
        table[Path(im["file_name"]).stem] = {
            "bbox": (float(x), float(y), float(x + bw), float(y + bh)),
            "kp": kp,
        }
    return table or None


def derive_roi_json(base_json: str, split: str) -> str | None:
    """由 train 标注文件名推导 val/test 标注文件路径。"""
    if not base_json:
        return None
    p = Path(base_json)
    for cand in (p.with_name(p.name.replace("train", split)),
                 p.with_name(f"anatomical_ROIs_{split}.json")):
        if cand.exists():
            return str(cand)
    return None


class ROIEncoder(nn.Module):
    """关键点编码器：17×3 归一化坐标 → MLP → 128 维解剖特征。"""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(KP_DIM),
            nn.Linear(KP_DIM, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, kp: torch.Tensor) -> torch.Tensor:
        assert kp.shape[-1] == KP_DIM, f"关键点须为 [...,{KP_DIM}]，得到 {kp.shape}"
        return self.net(kp)
