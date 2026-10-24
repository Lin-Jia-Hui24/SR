# EDSR ×2 超分辨率 — RealSR V3

基于 PyTorch 的 EDSR 超分辨率框架，在 RealSR V3 真实 LR/HR 配对数据上做 ×2 超分辨率。包含训练、评测、消融实验配置和推理工具。

## 项目结构

```
SR/
├── edsr/                    # 模型与工具
│   ├── model.py             # EDSR 网络 + build_model 工厂
│   ├── data.py              # PyTorch Dataset
│   ├── losses.py            # 损失函数（L1）
│   ├── utils.py             # seed / PSNR / checkpoint / self-ensemble
│   └── weights.py           # 预训练权重加载与 key 映射
├── data_pipeline/           # RealSR 数据管线
│   ├── realsr_demo/         # 数据加载、增强、评测、可视化
│   ├── splits/              # 固定 train/val/test 划分
│   └── tests/               # 单元测试
├── configs/
│   ├── default.yaml         # 默认训练配置
│   ├── finetune_realsr_x2.yaml
│   └── ablation/            # 消融实验配置
│       ├── L0_baseline.yaml     # 基线 (patch=96)
│       ├── P1_large_patch.yaml  # 大 patch (patch=192)
│       └── P2_xlarge_patch.yaml # 更大 patch (patch=256)
├── train.py                 # 训练脚本
├── evaluate_edsr.py         # 评测脚本（支持 self-ensemble）
├── collect_results.py       # 结果汇总
└── infer.py                 # 单图推理
```

## 环境配置

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
pip install -r requirements.txt
```

## 数据准备

下载 RealSR V3 数据集：

```bash
python -m data_pipeline.realsr_demo fetch --root data/realsr
```

数据目录结构：
```
data/realsr/RealSR(V3)/
├── Canon/
│   ├── Train/2/    # Canon LR2/HR 训练对
│   └── Test/2/     # Canon LR2/HR 测试对
└── Nikon/
    ├── Train/2/
    └── Test/2/
```

数据集包含 365 train / 41 val / 100 test pairs（Canon + Nikon 真实相机采集）。

## 训练

从官方 EDSR ×2 预训练权重微调：

```bash
# 基线训练 (patch=96, batch=16, 80 epochs)
python train.py --config configs/ablation/L0_baseline.yaml

# 大 patch 训练 (patch=192, batch=4, 80 epochs)
python train.py --config configs/ablation/P1_large_patch.yaml

# 恢复训练
python train.py --config configs/ablation/L0_baseline.yaml --resume outputs/ablation/L0_baseline/last.pt
```

预训练权重放在 `weights/edsr_x2-0edfb8a3.pt`（官方 EDSR r32f256 DIV2K 权重）。

训练输出：
- `best.pt` — 验证 PSNR 最优的 checkpoint
- `last.pt` — 最后一个 epoch，可用于恢复训练
- `epoch_XXX.pt` — 每个 epoch 的 checkpoint
- `config.yaml` — 实验配置备份

支持 wandb 日志（可用 `--no-wandb` 关闭）。

## 评测

```bash
# 标准评测
python evaluate_edsr.py \
  --config configs/ablation/L0_baseline.yaml \
  --checkpoint outputs/ablation/L0_baseline/best.pt \
  --output results/L0

# Self-ensemble ×8（推理时 4 旋转 × 2 翻转取平均，+0.10 dB）
python evaluate_edsr.py \
  --config configs/ablation/L0_baseline.yaml \
  --checkpoint outputs/ablation/L0_baseline/best.pt \
  --output results/L0_x8 \
  --self-ensemble
```

评测在 100 张 test set 上计算 PSNR / SSIM（RGB，float32），按相机（Canon/Nikon）分别汇总。

汇总所有实验结果：

```bash
python collect_results.py
```

## 推理

```bash
python infer.py --checkpoint outputs/ablation/L0_baseline/best.pt --input sample_lr.png --output result_sr.png
```

## 消融实验结果

在 RealSR V3 test set（100 张）上的结果：

| 配置 | Patch | Test PSNR | Δ vs Baseline |
|------|-------|-----------|---------------|
| Bicubic | — | 30.02 | -1.80 |
| L0 Baseline | 96 | 31.82 | — |
| P1 Large Patch | 192 | 31.99 | +0.17 |
| P1 + SE ×8 | 192 | 32.08 | +0.26 |
| P2 XLarge Patch | 256 | 32.05 | +0.23 |
| **P2 + SE ×8** | **256** | **32.14** | **+0.32** |

核心发现：

- **大 Patch Size 是最有效的改进**：patch 96→192 涨 +0.17 dB，192→256 再涨 +0.06 dB。RealSR 的真实退化比合成退化（bicubic）复杂，更大的感受野让网络更好地建模退化模式。
- **Self-Ensemble ×8 是稳定的免费涨点**：+0.09~0.11 dB，利用 CNN 对几何变换的不完全等变性质，不需要额外训练，推理时间 ×8。
- 两项改进正交可叠加：Large Patch (+0.23) + SE (+0.09) = **+0.32 dB**。

## 项目成员

| 成员 | 工作 |
|------|------|
| xjs | RealSR 数据管线、augmentation |
| zxr | EDSR 训练框架、微调流程 |
| lrq | 评测工具 |
| ljh | 消融实验（训练策略 + 推理策略） |
