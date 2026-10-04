"""MTFN Dataset：RSNA + RHPE 统一数据接口

统一返回：

train:
    (x, x_aug, bone_age, sex, ca, kp, roi_valid, image_id, dataset_id)

validation/test:
    (x, bone_age, sex, ca, kp, roi_valid, image_id, dataset_id)

dataset_id: RSNA = 0, RHPE = 1（int，可安全 collate 为 LongTensor）
roi_valid:  1 = 存在解剖 ROI/关键点标注, 0 = 缺失（kp 置零 + ROI gating）

Features
--------
1. CSV column auto-detection（RSNA + RHPE alias）
2. RHPE 年龄单位适配（years → months，禁止猜测，必须显式配置）
3. 严格泄漏检查（split 内 + 跨数据集 ID/SHA256）
4. 确定性 validation/test 预处理
5. Anatomical ROI crop（标注 bbox 优先，Otsu fallback）
6. 关键点 dataset-specific adapter → 统一 KP_DIM 表示
7. Chronological age 安全处理（invalid → -1，绝不进入 embedding）
8. 保守增强（photometric only，几何增强需与 KP 同步，默认关闭）
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms as T

from .config import RHPE_ID, RSNA_ID
from .preprocessing import (
    _letterbox_crop,
    detect_hand_bbox,
    preprocess_pil,
)
from .roi_encoder import (
    KP_DIM,
    load_roi_annotations,
)


# ================================================================
# CSV aliases
# ================================================================

ID_ALIASES = [
    "id",
    "image_id",
    "Case ID",
    "case_id",
    "ImageID",
]

AGE_ALIASES = [
    "boneage",
    "bone_age",
    "BoneAge",
    "bone age",
    "BA",
]

SEX_ALIASES = [
    "male",
    "gender",
    "sex",
    "Male",
    "Sex",
]

PATH_ALIASES = [
    "image_path",
    "path",
    "file_path",
    "filename",
]

CA_ALIASES = [
    "chronological_age",
    "ca",
    "CA",
    "Chronological Age",
]

DATASET_NAMES = {RSNA_ID: "RSNA", RHPE_ID: "RHPE"}


TRUE_SET = {
    "true",
    "1",
    "1.0",
    "male",
    "m",
    "yes",
}

FALSE_SET = {
    "false",
    "0",
    "0.0",
    "female",
    "f",
    "no",
}


# ================================================================
# Column finder
# ================================================================

def _find_col(
    cols: list,
    aliases: list,
):

    low = {
        str(c).lower().strip(): c
        for c in cols
    }

    for alias in aliases:

        key = alias.lower().strip()

        if key in low:
            return low[key]

    return None


# ================================================================
# Sex parser
# ================================================================

def parse_sex(v) -> int:

    if isinstance(v, bool):
        return int(v)

    if v is None:
        return -1

    if isinstance(v, float) and np.isnan(v):
        return -1

    if isinstance(v, (int, float)):

        if not np.isfinite(float(v)):
            return -1

        return int(float(v) > 0.5)

    s = str(v).strip().lower()

    if s in ("", "nan", "none", "null"):
        return -1

    if s in TRUE_SET:
        return 1

    if s in FALSE_SET:
        return 0

    return -1


# ================================================================
# CSV metadata
# ================================================================

def load_csv_meta(
    csv_path: str,
    age_unit: str = "months",
) -> tuple:
    """读取 CSV → 统一 metadata DataFrame。

    age_unit:
        "months" → bone_age 原样使用
        "years"  → bone_age * 12 转月

    返回 (df, has_ca)。bone_age 单位永远为 MONTHS。
    """

    if age_unit not in ("months", "years"):
        raise SystemExit(
            f"[RHPE] rhpe_age_unit 必须为 'months' 或 'years'，"
            f"得到 {age_unit!r}。禁止自动猜测单位。"
        )

    df = pd.read_csv(csv_path)

    cols = list(df.columns)

    col_id = _find_col(cols, ID_ALIASES)

    # 保留 id 列前导零（RHPE image_id "00001" ↔ 图像/标注 stem 键）。
    # pandas 默认会把 "00001" 解析为 int 1，导致标注查表静默失败，
    # 因此对 id 列强制按字符串重读（对 RSNA 数字 id 无行为差异）。
    if col_id is not None:
        df = pd.read_csv(csv_path, dtype={col_id: str})
    col_age = _find_col(cols, AGE_ALIASES)
    col_sex = _find_col(cols, SEX_ALIASES)
    col_path = _find_col(cols, PATH_ALIASES)
    col_ca = _find_col(cols, CA_ALIASES)

    # ------------------------------------------------------------
    # ID
    # ------------------------------------------------------------

    out = pd.DataFrame()

    if col_id is not None:

        out["image_id"] = (
            df[col_id]
            .astype(str)
            .str.strip()
        )

    else:

        out["image_id"] = (
            df.index
            .astype(str)
        )

    dup_mask = out["image_id"].duplicated(keep=False)

    if dup_mask.any():

        dup_ids = sorted(
            set(out.loc[dup_mask, "image_id"])
        )

        raise SystemExit(
            f"[dataset] CSV 存在重复 image_id "
            f"({int(dup_mask.sum())} 行) "
            f"例：{dup_ids[:5]}（{csv_path}）"
        )

    # ------------------------------------------------------------
    # Bone age
    # ------------------------------------------------------------

    if col_age is not None:

        out["bone_age"] = pd.to_numeric(
            df[col_age],
            errors="coerce",
        )

    else:

        out["bone_age"] = np.nan

    if age_unit == "years":

        n_before = int(out["bone_age"].notna().sum())

        out["bone_age"] = out["bone_age"] * 12.0

        print(
            f"[RHPE] bone_age 单位 years → months "
            f"(×12)，有效标签 {n_before} 条"
        )

    # ------------------------------------------------------------
    # Sex
    # ------------------------------------------------------------

    if col_sex is not None:

        out["male"] = df[col_sex].apply(
            parse_sex
        )

    else:

        out["male"] = -1

    # ------------------------------------------------------------
    # Image path
    # ------------------------------------------------------------

    if col_path is not None:

        out["image_path"] = (
            df[col_path]
            .fillna("")
            .astype(str)
        )

    else:

        out["image_path"] = ""

    # ------------------------------------------------------------
    # Chronological age
    # ------------------------------------------------------------

    if col_ca is not None:

        out["chronological_age"] = pd.to_numeric(
            df[col_ca],
            errors="coerce",
        )

        if age_unit == "years":

            out["chronological_age"] = (
                out["chronological_age"] * 12.0
            )

    else:

        out["chronological_age"] = np.nan

    has_ca = (
        col_ca is not None
        and
        out["chronological_age"].notna().any()
    )

    return out, has_ca


# ================================================================
# Leakage checking
# ================================================================

def check_leakage(
    train_ids: set,
    val_ids: set,
    test_ids: set | None,
    label: str = "",
) -> None:

    problems = []

    tv = train_ids & val_ids

    if tv:

        problems.append(
            f"train ∩ val = {len(tv)} "
            f"个重复 ID，例："
            f"{sorted(tv)[:5]}"
        )

    if test_ids:

        tt = train_ids & test_ids

        if tt:
            problems.append(
                f"train ∩ test = {len(tt)} 个重复 ID"
            )

        vt = val_ids & test_ids

        if vt:
            problems.append(
                f"val ∩ test = {len(vt)} 个重复 ID"
            )

    if problems:

        raise SystemExit(
            f"[数据泄漏检查失败] {label}\n"
            + "\n".join(
                f"  {x}"
                for x in problems
            )
        )


def _sha256_file(path: str) -> str:

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def cross_dataset_leakage_check(
    id_sets: dict,
    image_lists: dict,
    hash_check: bool = True,
    hash_max: int = 20000,
) -> list:
    """跨数据集泄漏检查。

    id_sets: {dataset_name: set(image_id)}
    image_lists: {dataset_name: [image_path, ...]}

    返回 warning 列表；发现跨数据集图像重复时直接 SystemExit（默认 error）。
    """

    warnings = []
    names = list(id_sets.keys())

    # ------------------------------------------------------------
    # ID 交集（ID 空间不同可能巧合重复 → warning + hash 复核）
    # ------------------------------------------------------------

    for i in range(len(names)):

        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]

            inter = id_sets[a] & id_sets[b]

            if inter:

                warnings.append(
                    f"[cross-dataset] {a} ∩ {b} "
                    f"ID 交集 {len(inter)} 个，"
                    f"例：{sorted(inter)[:5]}（ID 巧合重复，"
                    f"需 hash 复核）"
                )

    # ------------------------------------------------------------
    # SHA256 图像内容重复（硬检查，默认 error）
    # ------------------------------------------------------------

    if hash_check:

        seen = {}
        intra_dup = []

        for name, paths in image_lists.items():

            paths = list(paths)

            if len(paths) > hash_max:

                warnings.append(
                    f"[cross-dataset] {name} 图像数 "
                    f"{len(paths)} > {hash_max}，"
                    f"hash 检查截断为前 {hash_max} 张"
                )

                paths = paths[:hash_max]

            for p in paths:

                try:
                    digest = _sha256_file(p)

                except OSError:
                    continue

                if digest in seen:

                    prev_name, prev_p = seen[digest]

                    if prev_name != name:

                        # 跨数据集/跨 split 图像内容重复 → 真泄漏，致命
                        raise SystemExit(
                            "[跨数据集重复图像] "
                            f"{prev_name}:{prev_p} 与 {name}:{p} "
                            f"SHA256 相同 → 疑似数据泄漏，"
                            f"停止训练"
                        )

                    # 同一 split 内部精确重复 → 数据质量问题，
                    # 不构成泄漏（val/test 未受影响），记录为 warning
                    intra_dup.append((name, prev_p, p))

                    continue

                seen[digest] = (name, p)

        if intra_dup:

            warnings.append(
                f"[intra-split] {len(intra_dup)} 对同 split 内精确重复图像"
                f"（数据质量问题，非泄漏；详见 dataset audit）例："
                + "; ".join(
                    f"{a} == {b}" for _, a, b in intra_dup[:3]
                )
            )

        print(
            f"[cross-dataset] SHA256 检查 "
            f"{len(seen)} 张唯一图像，"
            f"跨列表重复 0，"
            f"split 内重复 {len(intra_dup)} 对"
        )

    for w in warnings:
        print(f"[warning] {w}")

    return warnings


# ================================================================
# Image index
# ================================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


def build_image_index(
    image_dir: str,
) -> dict:

    index = {}

    root = Path(image_dir)

    if not root.exists():

        raise FileNotFoundError(
            f"图像目录不存在: {image_dir}"
        )

    for p in root.rglob("*"):

        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:

            index[p.stem] = str(p)

    print(
        f"[image-index] {image_dir} "
        f"→ {len(index)} images"
    )

    return index


# ================================================================
# Resolve image path
# ================================================================

def resolve_image_path(
    row: pd.Series,
    image_dir: str,
    index: dict,
) -> str:

    p = str(
        row.get(
            "image_path",
            "",
        )
        or ""
    ).strip()

    # ------------------------------------------------------------
    # Absolute path
    # ------------------------------------------------------------

    if p and Path(p).is_absolute():

        if Path(p).exists():

            return p

    # ------------------------------------------------------------
    # Relative path
    # ------------------------------------------------------------

    if p and image_dir:

        candidate = (
            Path(image_dir) / p
        )

        if candidate.exists():

            return str(candidate)

    # ------------------------------------------------------------
    # Image ID lookup
    # ------------------------------------------------------------

    key = str(
        row["image_id"]
    )

    if key in index:

        return index[key]

    # ------------------------------------------------------------
    # Filename lookup
    # ------------------------------------------------------------

    if p and image_dir:

        candidate = (
            Path(image_dir)
            / Path(p).name
        )

        if candidate.exists():

            return str(candidate)

    raise FileNotFoundError(
        f"找不到图像: "
        f"id={key}, "
        f"image_dir={image_dir}"
    )


# ================================================================
# Keypoint adapter
# ================================================================

def adapt_keypoints(
    kp_raw: np.ndarray | None,
    mapping: dict | None = None,
) -> tuple:
    """将任意数据集关键点转换到统一 KP_DIM 表示。

    mapping: {common_kp_index: source_kp_index}，语义对应表
    。

    返回 (kp[KP_DIM], roi_valid)。
    """

    if kp_raw is None or len(kp_raw) == 0:

        return (
            np.zeros(KP_DIM, dtype=np.float32),
            0,
        )

    if mapping is not None:

        kp = np.zeros(KP_DIM, dtype=np.float32)

        n_src = len(kp_raw) // 3

        ok = True

        for dst in range(KP_DIM // 3):

            src = mapping.get(dst, -1)

            if 0 <= src < n_src:

                kp[dst * 3] = kp_raw[src * 3]
                kp[dst * 3 + 1] = kp_raw[src * 3 + 1]
                kp[dst * 3 + 2] = kp_raw[src * 3 + 2]

            else:

                ok = False

        return kp, int(ok)

    # 无 mapping：维度匹配则恒等，否则无法保证语义对应 → 置零
    if len(kp_raw) == KP_DIM:

        return (
            kp_raw.astype(np.float32).copy(),
            1,
        )

    return (
        np.zeros(KP_DIM, dtype=np.float32),
        0,
    )


# ================================================================
# Dataset
# ================================================================

class BoneAgeDataset(Dataset):

    """
    MTFN Bone Age Dataset（RSNA / RHPE 共用）。

    Train（9 元组）:
        xn, x_aug, y, sex, ca, kp, roi_valid, img_id, dataset_id

    Validation/Test（8 元组）:
        xn, y, sex, ca, kp, roi_valid, img_id, dataset_id
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        cfg,
        train: bool = True,
        with_aug: bool | None = None,
        roi_json: str | None = None,
        dataset_id: int = RSNA_ID,
        age_unit: str = "months",
        kp_mapping: dict | None = None,
    ):

        super().__init__()

        self.cfg = cfg
        self.train = bool(train)
        self.dataset_id = int(dataset_id)
        self.dataset_name = DATASET_NAMES.get(
            self.dataset_id,
            f"DS{self.dataset_id}",
        )
        self.kp_mapping = kp_mapping

        self.with_aug = (
            bool(train)
            if with_aug is None
            else bool(with_aug)
        )

        # ========================================================
        # CSV（RHPE 单位转换在此完成，永远输出 MONTHS）
        # ========================================================

        df, has_ca = load_csv_meta(
            csv_path,
            age_unit=age_unit,
        )

        # --------------------------------------------------------
        # Test without labels（无真值 → 占位 0，仅输出预测）
        # --------------------------------------------------------

        if (
            len(df) > 0
            and
            df["bone_age"].isna().all()
        ):

            df["bone_age"] = 0.0

        # --------------------------------------------------------
        # bone_age NaN
        # --------------------------------------------------------

        n_nan_age = int(df["bone_age"].isna().sum())

        if n_nan_age > 0 and not df["bone_age"].eq(0).all():

            print(
                f"[warning] [{self.dataset_name}] "
                f"{n_nan_age} 条 bone_age NaN → 剔除"
                f"（不允许静默错误）"
            )

            df = df.dropna(subset=["bone_age"]).reset_index(drop=True)

        # --------------------------------------------------------
        # Limit
        # --------------------------------------------------------

        if train:

            limit = int(
                getattr(
                    cfg,
                    "limit_train",
                    0,
                )
            )

        else:

            limit = int(
                getattr(
                    cfg,
                    "limit_val",
                    0,
                )
            )

        if limit > 0:

            df = (
                df
                .head(limit)
                .reset_index(drop=True)
            )

        self.df = df

        # ========================================================
        # Image index
        # ========================================================

        self.image_dir = image_dir

        self.index = build_image_index(
            image_dir
        )

        # 缺失图像检查
        missing = []

        for _, row in self.df.iterrows():

            try:

                resolve_image_path(
                    row,
                    self.image_dir,
                    self.index,
                )

            except FileNotFoundError:

                missing.append(
                    str(row["image_id"])
                )

        if missing:

            raise SystemExit(
                f"[dataset] [{self.dataset_name}] "
                f"{len(missing)} 张图像缺失，"
                f"例：{missing[:5]}。"
                f"请检查 image_dir / CSV。"
            )

        # ========================================================
        # Chronological age
        # ========================================================

        self.has_ca = bool(
            has_ca
            and
            df["chronological_age"]
            .notna()
            .any()
        )

        ca_valid = int(
            df["chronological_age"]
            .notna()
            .sum()
        )

        self.ca_valid_rate = (
            ca_valid / max(len(df), 1)
        )

        sex_valid = int(
            (df["male"] >= 0).sum()
        )

        self.sex_valid_rate = (
            sex_valid / max(len(df), 1)
        )

        # ========================================================
        # ROI annotations
        # ========================================================

        self.roi_table = (
            load_roi_annotations(
                roi_json or ""
            )
        )

        # ========================================================
        # Crop records
        # ========================================================

        self.crop_records = {}

        # ========================================================
        # Normalize
        # ========================================================

        if (
            self.dataset_id == RHPE_ID
            and getattr(cfg, "normalization_mode", "shared")
            == "dataset_specific"
        ):

            mean = [
                float(m)
                for m in cfg.rhpe_norm_mean
            ]

            std = [
                float(s)
                for s in cfg.rhpe_norm_std
            ]

        else:

            mean = [
                float(m)
                for m in cfg.norm_mean
            ]

            std = [
                float(s)
                for s in cfg.norm_std
            ]

        self.normalize = T.Normalize(
            mean=mean,
            std=std,
        )

        # ========================================================
        # Augmentation（photometric only，几何增强需与 KP 同步）
        # ========================================================

        aug_ops = []

        brightness = float(
            getattr(
                cfg,
                "aug_brightness",
                0.0,
            )
        )

        contrast = float(
            getattr(
                cfg,
                "aug_contrast",
                0.0,
            )
        )

        if (
            brightness > 0
            or contrast > 0
        ):

            aug_ops.append(
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                )
            )

        self.augment = (
            T.Compose(aug_ops)
            if len(aug_ops) > 0
            else None
        )

        # ========================================================
        # Dataset statistics
        # ========================================================

        roi_count = 0

        if self.roi_table is not None:

            for image_id in self.df[
                "image_id"
            ].astype(str):

                if image_id in self.roi_table:

                    roi_count += 1

        self.roi_valid_rate = (
            roi_count / max(len(self.df), 1)
        )

        print(
            f"[dataset] [{self.dataset_name}] "
            f"split={'train' if train else 'val/test'} "
            f"samples={len(self.df)} "
            f"ROI={roi_count}/{len(self.df)} "
            f"({self.roi_valid_rate:.2%}) "
            f"CA_valid={self.ca_valid_rate:.2%} "
            f"sex_valid={self.sex_valid_rate:.2%}"
        )

    # ================================================================
    # Length
    # ================================================================

    def __len__(self) -> int:

        return len(self.df)

    # ================================================================
    # Ages / IDs / dataset_ids / roi_valids
    # ================================================================

    @property
    def ages(self) -> np.ndarray:

        return self.df[
            "bone_age"
        ].to_numpy(
            dtype=np.float32
        )

    @property
    def ids(self) -> list:

        return self.df[
            "image_id"
        ].astype(str).tolist()

    @property
    def id_set(self) -> set:

        return set(self.ids)

    @property
    def dataset_ids(self) -> np.ndarray:

        return np.full(
            len(self.df),
            self.dataset_id,
            dtype=np.int64,
        )

    @property
    def roi_valids(self) -> np.ndarray:

        if self.roi_table is None:

            return np.zeros(
                len(self.df),
                dtype=np.int64,
            )

        return np.asarray(
            [
                1
                if str(i) in self.roi_table
                else 0
                for i in self.df["image_id"]
            ],
            dtype=np.int64,
        )

    # ================================================================
    # Image paths（用于泄漏 hash 检查 / 归一化统计）
    # ================================================================

    def image_paths(self) -> list:

        return [
            resolve_image_path(
                row,
                self.image_dir,
                self.index,
            )
            for _, row in self.df.iterrows()
        ]

    # ================================================================
    # Get item
    # ================================================================

    def __getitem__(
        self,
        i: int,
    ):

        row = self.df.iloc[i]

        img_id = str(
            row["image_id"]
        )

        # ========================================================
        # 1. Image path + load
        # ========================================================

        path = resolve_image_path(
            row,
            self.image_dir,
            self.index,
        )

        img = Image.open(path)

        img = img.convert("RGB")

        # ========================================================
        # 2. ROI crop（标注 bbox 优先，Otsu fallback）
        # ========================================================

        ann = (
            self.roi_table.get(img_id)
            if self.roi_table
            else None
        )

        use_annotated_roi = bool(
            getattr(
                self.cfg,
                "use_annotated_roi",
                False,
            )
        )

        if (
            ann is not None
            and use_annotated_roi
        ):

            gray = np.asarray(
                img.convert("L")
            )

            h, w = gray.shape[:2]

            x1, y1, x2, y2 = (
                ann["bbox"]
            )

            margin = float(
                getattr(
                    self.cfg,
                    "roi_margin",
                    0.05,
                )
            )

            mx = int(w * margin)
            my = int(h * margin)

            x1 = max(
                0,
                int(x1) - mx,
            )

            y1 = max(
                0,
                int(y1) - my,
            )

            x2 = min(
                w,
                int(x2) + mx,
            )

            y2 = min(
                h,
                int(y2) + my,
            )

            # ----------------------------------------------------
            # 标注异常 → 自动检测 fallback
            # ----------------------------------------------------

            if (
                x2 - x1 < 8
                or
                y2 - y1 < 8
            ):

                x1, y1, x2, y2 = (
                    detect_hand_bbox(
                        gray,
                        margin,
                    )
                )

            record = {
                "original_width": int(w),
                "original_height": int(h),
                "crop_x1": int(x1),
                "crop_y1": int(y1),
                "crop_x2": int(x2),
                "crop_y2": int(y2),
            }

            x, _ = _letterbox_crop(
                img,
                record,
                int(
                    self.cfg.image_size
                ),
            )

        else:

            x, record = preprocess_pil(
                img,
                image_size=int(
                    self.cfg.image_size
                ),
                use_roi_crop=bool(
                    getattr(
                        self.cfg,
                        "use_roi_crop",
                        True,
                    )
                ),
                roi_margin=float(
                    getattr(
                        self.cfg,
                        "roi_margin",
                        0.05,
                    )
                ),
            )

        self.crop_records[
            img_id
        ] = record

        # ========================================================
        # 3. Image → Tensor [0,1]
        # ========================================================

        x = torch.from_numpy(
            np.asarray(
                x,
                dtype=np.uint8,
            ).copy()
        )

        x = x.permute(
            2,
            0,
            1,
        )

        x = x.float().div_(255.0)

        # ========================================================
        # 4. Original normalized image
        # ========================================================

        xn = self.normalize(
            x.clone()
        )

        # ========================================================
        # 5. Augmented normalized image（增强 → 归一化顺序）
        # ========================================================

        if (
            self.with_aug
            and self.augment is not None
        ):

            x_aug_raw = self.augment(
                x.clone()
            )

            x_aug = self.normalize(
                x_aug_raw
            )

        else:

            x_aug = xn.clone()

        # ========================================================
        # 6. Bone age（MONTHS）
        # ========================================================

        y = float(
            row["bone_age"]
        )

        # ========================================================
        # 7. Sex
        # ========================================================

        sex = int(row["male"])

        if sex not in (0, 1):

            sex = 0

        # ========================================================
        # 8. Chronological age
        # ========================================================

        if self.has_ca:

            value = row[
                "chronological_age"
            ]

            if pd.isna(value):

                ca = -1.0

            else:

                ca = float(value)

                if not np.isfinite(ca) or ca < 0:

                    ca = -1.0

        else:

            ca = -1.0

        # ========================================================
        # 9. Anatomical keypoints（dataset adapter → KP_DIM）
        # ========================================================

        if (
            self.roi_table is not None
            and img_id in self.roi_table
        ):

            kp_raw = self.roi_table[
                img_id
            ]["kp"]

            kp_arr, roi_valid = adapt_keypoints(
                kp_raw,
                self.kp_mapping,
            )

            kp = torch.from_numpy(
                kp_arr
            ).float()

        else:

            kp = torch.zeros(
                KP_DIM,
                dtype=torch.float32,
            )

            roi_valid = 0

        if kp.shape[-1] != KP_DIM:

            raise RuntimeError(
                f"关键点维度错误: "
                f"{kp.shape}, "
                f"expected {KP_DIM}"
            )

        # ========================================================
        # 10. Return
        # ========================================================

        if self.with_aug:

            return (
                xn,
                x_aug,
                y,
                sex,
                ca,
                kp,
                roi_valid,
                img_id,
                self.dataset_id,
            )

        return (
            xn,
            y,
            sex,
            ca,
            kp,
            roi_valid,
            img_id,
            self.dataset_id,
        )


# ================================================================
# Multi-Dataset wrapper
# ================================================================

class MultiDatasetBoneAgeDataset(Dataset):

    """统一管理 RSNA / RHPE 训练数据。

    暴露：
        dataset.ages        → np.ndarray [N]
        dataset.dataset_ids → np.ndarray [N]（0=RSNA, 1=RHPE）
        dataset.ids         → list[str]
        dataset.roi_valids  → np.ndarray [N]
    """

    def __init__(
        self,
        datasets: list,
    ):

        super().__init__()

        if len(datasets) == 0:

            raise ValueError(
                "MultiDatasetBoneAgeDataset 至少需要一个子数据集"
            )

        self.datasets = list(datasets)

        self.offsets = [0]

        for ds in self.datasets:

            self.offsets.append(
                self.offsets[-1] + len(ds)
            )

    # ------------------------------------------------------------

    def __len__(self) -> int:

        return self.offsets[-1]

    def __getitem__(self, i: int):

        if i < 0 or i >= len(self):

            raise IndexError(
                f"index {i} out of range "
                f"[0, {len(self)})"
            )

        for k, ds in enumerate(self.datasets):

            if i < self.offsets[k + 1]:

                return ds[
                    i - self.offsets[k]
                ]

        raise IndexError(i)

    # ------------------------------------------------------------

    @property
    def ages(self) -> np.ndarray:

        return np.concatenate(
            [
                ds.ages
                for ds in self.datasets
            ]
        )

    @property
    def dataset_ids(self) -> np.ndarray:

        return np.concatenate(
            [
                ds.dataset_ids
                for ds in self.datasets
            ]
        )

    @property
    def ids(self) -> list:

        out = []

        for ds in self.datasets:

            out.extend(ds.ids)

        return out

    @property
    def id_set(self) -> set:

        return set(self.ids)

    @property
    def roi_valids(self) -> np.ndarray:

        return np.concatenate(
            [
                ds.roi_valids
                for ds in self.datasets
            ]
        )

    @property
    def dataset_names(self) -> list:

        return [
            ds.dataset_name
            for ds in self.datasets
        ]

    # ------------------------------------------------------------

    def save_crop_records(
        self,
        path,
    ) -> None:

        import csv as _csv

        with open(path, "w", newline="") as f:

            w = _csv.writer(f)

            w.writerow(
                [
                    "image_id",
                    "dataset",
                    "original_width",
                    "original_height",
                    "crop_x1",
                    "crop_y1",
                    "crop_x2",
                    "crop_y2",
                ]
            )

            for ds in self.datasets:

                for k, r in (
                    ds.crop_records.items()
                ):

                    w.writerow(
                        [
                            k,
                            ds.dataset_name,
                            *r.values(),
                        ]
                    )


# ================================================================
# Dataset Summary
# ================================================================

def print_dataset_summary(
    stats: dict,
) -> None:

    print(
        "\n================ DATASET SUMMARY ================"
    )

    for name, s in stats.get("datasets", {}).items():

        print(f"\n{name}:")

        print(
            f"    train = {s.get('train', 0)}"
        )

        print(
            f"    val   = {s.get('val', 0)}"
        )

        print(
            f"    test  = {s.get('test', 0)}"
        )

    print("\nROI:")

    for name, s in stats.get("datasets", {}).items():

        print(
            f"    {name} = "
            f"{s.get('roi_rate', 0.0):.1%}"
        )

    print("\nChronological Age:")

    for name, s in stats.get("datasets", {}).items():

        print(
            f"    {name} valid = "
            f"{s.get('ca_rate', 0.0):.1%}"
        )

    print("\nSex:")

    for name, s in stats.get("datasets", {}).items():

        print(
            f"    {name} valid = "
            f"{s.get('sex_rate', 0.0):.1%}"
        )

    print("\n===================================================")

    sampler = stats.get("sampler", {})

    if sampler:

        print("\nSampler:")

        print("\nDataset ratio:")

        for k, v in sampler.get(
            "dataset_ratio",
            {},
        ).items():

            print(f"    {k} = {v:.2f}")

        print(
            f"\nAge alpha:\n    "
            f"{sampler.get('alpha', 0.5):.2f}"
        )

    print(
        "===================================================\n"
    )
