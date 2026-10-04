# MTFN

**Multi-scale Token Fusion Network** — 儿童骨龄评估（Pediatric Bone Age Assessment）

输入左手 X 光片与性别，输出骨龄（月）。单模型端到端训练，支持 RSNA 单数据集与 RSNA + RHPE 双数据集联合训练两种模式。

## 架构

- **预处理**：Otsu 自动手部 ROI 检测 → 保持纵横比 letterbox 裁剪至 512×512；提供 COCO 标注时优先使用标注 bbox，缺失自动回退 Otsu
- **Backbone**：Swin-T（torchvision ImageNet 预训练权重）
- **Visual Tokens**：多尺度 F2/F3/F4 特征 → 49 × 768 token 序列
- **可退化 ROI 分支**：COCO 17 关键点编码，无标注时自动置零（`roi_valid=0`），不影响训练
- **条件信息**：Age-aware Attention / Sex Embedding / 可选 Chronological Age
- **三预测头**：Regression + Ordinal（阈值间距 12 月）+ Distribution（Gaussian soft label，σ=2 月）
- **损失**：多任务损失 + 预测间 Consistency Loss
- **融合**：Learnable 预测融合，策略由验证集 MAE 锁定（绝不用 test 选择）
- **采样**：Age-Binned Sampling，权重 w_k ∝ n_k^−α（α 默认 0.5）；双数据集按比例混合（默认 RSNA:RHPE = 0.7:0.3）
- **优化**：差分学习率（backbone 1e-5 / head 1e-4）+ Warmup/Cosine + AMP + 早停

## 环境

| 依赖 | 要求 |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | 2.2.2（cu121 验证通过），配套 torchvision |
| numpy | < 2.0 |
| 其他 | pandas、scipy、opencv-python、Pillow |

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" pandas scipy opencv-python Pillow
```

## 目录结构

```
MTFN/
└── mtfn/
    ├── config.py         # 全部配置（dataclass，命令行可覆盖）
    ├── dataset.py        # RSNA+RHPE 统一数据接口、CSV 列名自动识别、泄漏检查
    ├── preprocessing.py  # Otsu 手部检测、letterbox
    ├── sampler.py        # Age-Binned / 多数据集采样器
    ├── roi_encoder.py    # COCO 17 关键点编码
    ├── backbone.py       # Swin-T
    ├── visual_tokens.py  # 多尺度特征 token 化
    ├── attention.py      # Age-aware Attention
    ├── embeddings.py     # Sex / Age embedding
    ├── heads.py          # 三预测头
    ├── model.py          # MTFN 主模型
    ├── loss.py           # 多任务损失
    ├── trainer.py        # 训练循环、checkpoint、指标
    ├── metrics.py        # MAE / RMSE / R² / 年龄性别分层指标
    ├── train.py          # 训练入口
    └── evaluate.py       # 测试入口
```

## 数据准备

### CSV 格式

列名自动识别（大小写不敏感），同一字段任一别名均可：

| 字段 | 可识别列名 | 必填 | 说明 |
|---|---|---|---|
| ID | `id` / `image_id` / `Case ID` / `case_id` / `ImageID` | 是 | 图像文件名（按 stem 递归匹配 `image_dir`） |
| 骨龄 | `boneage` / `bone_age` / `BoneAge` / `bone age` / `BA` | 是 | 单位：月 |
| 性别 | `male` / `gender` / `sex` | 是 | male=1/true/m，female=0/false/f；缺失 → -1（安全回退） |
| 图像路径 | `image_path` / `path` / `file_path` / `filename` | 否 | 提供时优先于 ID 匹配 |
| 实足年龄 | `chronological_age` / `ca` / `CA` | 否 | RHPE 消融实验用；缺失 → -1 |

图像目录递归扫描（支持 png/jpg/jpeg/bmp 等常见格式）。

### 训练 / 验证 / 测试 CSV 示例

```csv
id,boneage,male
1377,132,TRUE
1378,180,FALSE
```

### ROI / 关键点标注（可选）

- RSNA：`--roi_annotation` 提供 COCO 格式 JSON（bbox + 17×3 关键点，train 版；val/test 文件名自动推导）。不提供时自动使用 Otsu 检测。
- RHPE：`--rhpe_roi_annotation` 同格式，可选。

## 训练

### RSNA 单数据集

```bash
python mtfn/train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --train_image_dir data/train_images \
    --val_image_dir data/val_images \
    --output_dir outputs/mtfn
```

### RSNA + RHPE 联合训练

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

> **注意**：提供 `--rhpe_csv` 时必须显式指定 `--rhpe_age_unit months|years`，程序拒绝猜测单位（防止年/月混用导致的静默错误）。

### 关键超参数（默认值见 `mtfn/config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 150 | 最大训练轮数 |
| `--warmup_epochs` | 10 | warmup 轮数 |
| `--batch_size` | 14 | 批大小 |
| `--backbone_lr` | 1e-5 | backbone 学习率 |
| `--head_lr` | 1e-4 | 其余模块学习率 |
| `--weight_decay` | 0.05 | AdamW 权重衰减 |
| `--early_stopping_patience` | 20 | 早停耐心值 |
| `--amp` | true | 混合精度 |
| `--seed` | 42 | 随机种子 |
| `--dataset_ratio_rsna` / `--dataset_ratio_rhpe` | 0.7 / 0.3 | 双数据集采样比例 |
| `--age_sampling_alpha` | 0.5 | 年龄均衡指数 α |
| `--dist_sigma` | 2.0 | Gaussian soft label σ（月） |
| `--ordinal_step` | 12 | Ordinal 头阈值间距（月） |
| `--fusion_mode` | learned | learned / reg / ord / dist / average |

### 消融开关

`--use_multiscale`、`--use_age_attention`、`--use_sex_embedding`、`--use_ordinal_head`、`--use_distribution_head`、`--use_consistency_loss`、`--use_roi_branch`、`--use_roi_crop`、`--use_annotated_roi`、`--use_roi_gating`、`--use_age_sampling`、`--use_chronological_age`、`--zero_anatomy`（关键点输入严格置零）等，均接受 `true/false`。

### 训练输出

```
<output_dir>/
├── checkpoints/
│   ├── best_rsna_model.pth   # 最优权重（含 config，evaluate.py 直接使用）
│   ├── best_model.pt/.pth    # 同一权重的硬链接别名
│   ├── last_model.pth        # 最后一轮权重（断点续训用）
│   ├── metrics.csv           # 逐 epoch 训练/验证指标
│   └── predictions.csv       # 最优 epoch 的验证集预测
├── fusion_selection.json     # 验证集锁定的融合策略
├── normalization.json        # 训练集归一化统计（带签名缓存）
├── sampler_statistics.json   # 采样分布统计
├── dataset_summary.json      # 数据集摘要
├── crop_records.csv          # ROI 裁剪记录
├── config.json               # 本次运行完整配置
└── train.log                 # 完整训练日志
```

## 评估

```bash
python mtfn/evaluate.py \
    --ckpt outputs/mtfn/checkpoints/best_rsna_model.pth \
    --test_csv data/test.csv \
    --test_image_dir data/test_images \
    --output_dir outputs/mtfn/test_results
```

- 融合策略默认自动读取训练侧 `fusion_selection.json`（验证集锁定），可用 `--fusion_mode` 覆盖
- 测试集**有真值**：输出 MAE / RMSE / R² 及年龄/性别分层指标 + 逐例预测 CSV
- 测试集**无真值**（如 RSNA test）：仅输出 `image_id, predicted_bone_age`

## 数据安全设计

- split 内 + 跨数据集双重泄漏检查（ID + 图像 SHA256），发现重复立即终止训练
- 归一化统计仅用训练 split 计算（带签名缓存，防错用旧统计）
- `--test_csv` 仅用于泄漏检查，绝不参与训练或模型选择
- RHPE 年龄单位必须显式声明

## 引用

如果本代码对你的研究有帮助，请引用：

```bibtex
@software{mtfn,
  title  = {MTFN: Multi-scale Token Fusion Network for Pediatric Bone Age Assessment},
  year   = {2026}
}
```
