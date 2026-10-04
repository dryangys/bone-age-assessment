[README.md.md](https://github.com/user-attachments/files/33025524/README.md.md)
# MTFN

**Multi-scale Token Fusion Network** - Pediatric Bone Age Assessment

Given a left-hand X-ray and the patient's sex, the model outputs bone age in months. It supports end-to-end single-model training in two modes: the RSNA-only dataset and joint training on RSNA + RHPE.

## Architecture

- **Preprocessing**: Otsu automatic hand ROI detection -> aspect-ratio-preserving letterbox crop to 512 x 512; when COCO annotations are available, the annotated bbox is preferred, with automatic fallback to Otsu when missing
- **Backbone**: Swin-T (torchvision ImageNet pretrained weights)
- **Visual Tokens**: Multi-scale F2/F3/F4 features -> a 49 x 768 token sequence
- **Degradable ROI Branch**: COCO 17-keypoint encoding; automatically zeroed when annotations are unavailable (`roi_valid=0`), so training is unaffected
- **Conditioning Information**: Age-aware Attention / Sex Embedding / optional Chronological Age
- **Three Prediction Heads**: Regression + Ordinal (12-month threshold spacing) + Distribution (Gaussian soft labels, sigma = 2 months)
- **Losses**: Multi-task loss + consistency loss between predictions
- **Fusion**: Learnable prediction fusion; the strategy is locked using validation-set MAE (the test set is never used for selection)
- **Sampling**: Age-Binned Sampling, with weights w_k proportional to n_k^(-alpha) (alpha defaults to 0.5); joint datasets are mixed by ratio (default RSNA:RHPE = 0.7:0.3)
- **Optimization**: Differential learning rates (backbone 1e-5 / head 1e-4) + Warmup/Cosine + AMP + early stopping

## Environment

| Dependency | Requirement |
|---|---|
| Python | >= 3.10 |
| PyTorch | 2.2.2 (cu121 verified), with the corresponding torchvision |
| numpy | < 2.0 |
| Other | pandas, scipy, opencv-python, Pillow |

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" pandas scipy opencv-python Pillow
```

## Directory Structure

```
MTFN/
└── mtfn/
    ├── config.py         # All configuration (dataclass, overridable from the command line)
    ├── dataset.py        # Unified RSNA+RHPE data interface, automatic CSV column detection, leakage checks
    ├── preprocessing.py  # Otsu hand detection, letterbox
    ├── sampler.py        # Age-Binned / multi-dataset samplers
    ├── roi_encoder.py    # COCO 17-keypoint encoding
    ├── backbone.py       # Swin-T
    ├── visual_tokens.py  # Multi-scale feature tokenization
    ├── attention.py      # Age-aware Attention
    ├── embeddings.py     # Sex / Age embeddings
    ├── heads.py          # Three prediction heads
    ├── model.py          # Main MTFN model
    ├── loss.py           # Multi-task loss
    ├── trainer.py        # Training loop, checkpoints, metrics
    ├── metrics.py        # MAE / RMSE / R² / age- and sex-stratified metrics
    ├── train.py          # Training entry point
    └── evaluate.py       # Evaluation entry point
```

## Data Preparation

### CSV Format

Column names are detected automatically (case-insensitive). Any alias listed for a field is accepted:

| Field | Recognized column names | Required | Description |
|---|---|---|---|
| ID | `id` / `image_id` / `Case ID` / `case_id` / `ImageID` | Yes | Image filename (matched recursively by stem under `image_dir`) |
| Bone age | `boneage` / `bone_age` / `BoneAge` / `bone age` / `BA` | Yes | Unit: months |
| Sex | `male` / `gender` / `sex` | Yes | male=1/true/m, female=0/false/f; missing -> -1 (safe fallback) |
| Image path | `image_path` / `path` / `file_path` / `filename` | No | When provided, takes priority over ID matching |
| Chronological age | `chronological_age` / `ca` / `CA` | No | For RHPE ablation experiments; missing -> -1 |

The image directory is scanned recursively (common formats such as png/jpg/jpeg/bmp are supported).

### Training / Validation / Test CSV Example

```csv
id,boneage,male
1377,132,TRUE
1378,180,FALSE
```

### ROI / Keypoint Annotations (Optional)

- RSNA: pass a COCO-format JSON (bbox + 17 x 3 keypoints, train version) with `--roi_annotation`; validation/test filenames are inferred automatically. If omitted, Otsu detection is used automatically.
- RHPE: same format, optionally provided with `--rhpe_roi_annotation`.

## Training

### RSNA-Only Dataset

```bash
python mtfn/train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --train_image_dir data/train_images \
    --val_image_dir data/val_images \
    --output_dir outputs/mtfn
```

### Joint RSNA + RHPE Training

```bash
python mtfn/train.py \
    --train_csv data/rsna_train.csv \
    --val_csv data/rsna_val.csv \
    --train_image_dir data/rsna_train_images \
    --val_image_dir data/rsna_val_images \
    --rhpe_csv data/rhpe_train.csv \
    --rhpe_image_dir data/rhpe_images \
    --rhpe_roi_annotation data/rhpe_roi.json \
    --rhpe_age_unit years \
    --output_dir outputs/mtfn
```

> **Note**: When `--rhpe_csv` is provided, `--rhpe_age_unit months|years` must be specified explicitly. The program refuses to guess the unit, preventing silent errors caused by mixing years and months.

### Key Hyperparameters (defaults are defined in `mtfn/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `--epochs` | 150 | Maximum number of training epochs |
| `--warmup_epochs` | 10 | Number of warmup epochs |
| `--batch_size` | 14 | Batch size |
| `--backbone_lr` | 1e-5 | Backbone learning rate |
| `--head_lr` | 1e-4 | Learning rate for the remaining modules |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--early_stopping_patience` | 20 | Early-stopping patience |
| `--amp` | true | Mixed precision |
| `--seed` | 42 | Random seed |
| `--dataset_ratio_rsna` / `--dataset_ratio_rhpe` | 0.7 / 0.3 | Joint-dataset sampling ratios |
| `--age_sampling_alpha` | 0.5 | Age-balancing exponent alpha |
| `--dist_sigma` | 2.0 | Gaussian soft-label sigma (months) |
| `--ordinal_step` | 12 | Ordinal-head threshold spacing (months) |
| `--fusion_mode` | learned | learned / reg / ord / dist / average |

### Ablation Switches

`--use_multiscale`, `--use_age_attention`, `--use_sex_embedding`, `--use_ordinal_head`, `--use_distribution_head`, `--use_consistency_loss`, `--use_roi_branch`, `--use_roi_crop`, `--use_annotated_roi`, `--use_roi_gating`, `--use_age_sampling`, `--use_chronological_age`, and `--zero_anatomy` (strictly zero the keypoint input), among others, all accept `true/false`.

### Training Outputs

```
<output_dir>/
├── checkpoints/
│   ├── best_rsna_model.pth   # Best weights (includes config; used directly by evaluate.py)
│   ├── best_model.pt/.pth    # Hard-link aliases for the same weights
│   ├── last_model.pth        # Last-epoch weights (for resuming training)
│   ├── metrics.csv           # Training/validation metrics for each epoch
│   └── predictions.csv       # Validation predictions from the best epoch
├── fusion_selection.json     # Fusion strategy locked on the validation set
├── normalization.json        # Training-set normalization statistics (with signature cache)
├── sampler_statistics.json   # Sampling distribution statistics
├── dataset_summary.json      # Dataset summary
├── crop_records.csv          # ROI crop records
├── config.json               # Complete configuration for this run
└── train.log                 # Complete training log
```

## Evaluation

```bash
python mtfn/evaluate.py \
    --ckpt outputs/mtfn/checkpoints/best_rsna_model.pth \
    --test_csv data/test.csv \
    --test_image_dir data/test_images \
    --output_dir outputs/mtfn/test_results
```

- The fusion strategy is read automatically from the training-side `fusion_selection.json` (locked on the validation set) by default; use `--fusion_mode` to override it.
- **Test set with ground truth**: outputs MAE / RMSE / R², age- and sex-stratified metrics, and a per-case prediction CSV.
- **Test set without ground truth** (such as the RSNA test set): outputs only `image_id, predicted_bone_age`.

## Data Safety Design

- Dual leakage checks within each split and across datasets (ID + image SHA256); training terminates immediately when duplicates are found.
- Normalization statistics are computed only from the training split (with a signature cache to prevent accidental reuse of stale statistics).
- `--test_csv` is used only for leakage checks and never for training or model selection.
- The RHPE age unit must be declared explicitly.

## Citation

If this code helps your research, please cite:

```bibtex
@software{mtfn,
  title  = {MTFN: Multi-scale Token Fusion Network for Pediatric Bone Age Assessment},
  year   = {2026}
}
```
