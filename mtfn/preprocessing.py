"""图像预处理：ROI 检测 + 保持纵横比 Resize + Padding。

流程：灰度 → 前景检测（Otsu + 形态学 + 最大连通域）→ ROI Crop
→ 等比缩放 → letterbox 填充到 image_size×image_size。
失败自动 fallback 到中心裁剪/原图，绝不丢弃样本。
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _to_gray_array(img: Image.Image) -> np.ndarray:
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img)


def detect_hand_bbox(
    gray: np.ndarray,
    margin: float = 0.05,
    min_ratio: float = 0.02,
    max_ratio: float = 0.995,
) -> tuple:
    """Otsu + 最大连通域求手部 bbox。

    返回 (x1, y1, x2, y2)，失败时返回整图。
    """
    h, w = gray.shape[:2]
    area = h * w

    thresh_val, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    fg_ratio = float(binary.mean()) / 255.0

    if fg_ratio < 0.005 or fg_ratio > 0.995:
        # Otsu 对近均匀图退化 → 50% 分位阈值 fallback
        t = float(np.percentile(gray, 50))
        binary = (gray > t).astype(np.uint8) * 255
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return 0, 0, w, h

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, _ = stats[largest]

    if (bw * bh) / area < min_ratio or (bw * bh) / area > max_ratio:
        return 0, 0, w, h

    mx, my = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(w, x + bw + mx), min(h, y + bh + my)
    return x1, y1, x2, y2


def _letterbox_crop(img: Image.Image, record: dict, image_size: int) -> tuple:
    """按给定 crop 坐标裁剪 + 保持纵横比 letterbox 到 image_size×image_size。"""
    x1, y1 = int(record["crop_x1"]), int(record["crop_y1"])
    x2, y2 = int(record["crop_x2"]), int(record["crop_y2"])
    cropped = img.convert("L").crop((x1, y1, x2, y2))
    cw, ch = cropped.size
    if cw <= 1 or ch <= 1:
        cropped = img.convert("L")
        cw, ch = cropped.size
    scale = image_size / max(cw, ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), 0)
    canvas.paste(resized, ((image_size - nw) // 2, (image_size - nh) // 2))
    return canvas.convert("RGB"), record


def preprocess_pil(
    img: Image.Image,
    image_size: int = 512,
    use_roi_crop: bool = True,
    roi_margin: float = 0.05,
) -> tuple:
    """完整确定性预处理（val/test 同款，训练在此基础上叠加增强）。

    返回 (PIL.Image RGB image_size×image_size, crop_record dict)。
    crop_record 含 original_width/height 与 crop_x1..y2。
    """
    gray = _to_gray_array(img)
    orig_h, orig_w = gray.shape[:2]

    if use_roi_crop:
        x1, y1, x2, y2 = detect_hand_bbox(gray, margin=roi_margin)
    else:
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h

    cropped = img.convert("L").crop((x1, y1, x2, y2))
    cw, ch = cropped.size
    if cw <= 1 or ch <= 1:  # 双保险 fallback
        cropped = img.convert("L")
        x1, y1, x2, y2 = 0, 0, orig_w, orig_h
        cw, ch = cropped.size

    # 保持纵横比 resize + letterbox padding（禁止直接拉伸变形）
    scale = image_size / max(cw, ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("L", (image_size, image_size), 0)
    canvas.paste(resized, ((image_size - nw) // 2, (image_size - nh) // 2))

    record = {
        "original_width": orig_w,
        "original_height": orig_h,
        "crop_x1": int(x1), "crop_y1": int(y1),
        "crop_x2": int(x2), "crop_y2": int(y2),
    }
    # 灰度复制到 3 通道
    return canvas.convert("RGB"), record


def compute_train_stats(
    image_paths: list,
    image_size: int = 512,
    use_roi_crop: bool = True,
    roi_margin: float = 0.05,
    sample_size: int = 1000,
    seed: int = 42,
) -> tuple:
    """仅用 TRAIN 数据计算图像归一化 mean/std。"""
    import random

    rng = random.Random(seed)
    paths = list(image_paths)
    if sample_size > 0 and len(paths) > sample_size:
        idx = sorted(rng.sample(range(len(paths)), sample_size))
        paths = [paths[i] for i in idx]

    sums = np.zeros(3, dtype=np.float64)
    sqs = np.zeros(3, dtype=np.float64)
    npx = 0
    for p in paths:
        try:
            img = Image.open(p)
            out, _ = preprocess_pil(img, image_size, use_roi_crop, roi_margin)
            arr = np.asarray(out, dtype=np.float64) / 255.0  # HWC
            sums += arr.sum(axis=(0, 1))
            sqs += (arr ** 2).sum(axis=(0, 1))
            npx += arr.shape[0] * arr.shape[1]
        except Exception:  # noqa: BLE001
            continue
    mean = sums / npx
    std = np.sqrt(np.maximum(sqs / npx - mean ** 2, 1e-8))
    return mean.tolist(), std.tolist()
