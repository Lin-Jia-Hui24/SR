# EDSR ×2 消融实验报告（RealSR V3）

## 1. 实验目标

复现 README 中的消融实验，并把每个配置从**单 seed** 扩展到 **3 个随机种子（seed 42/43/44）**，
报告均值 ± 标准差（population std）。

## 2. 实验设置

- **数据**：RealSR V3 ×2，train 365 / val 41 / test 100（Canon + Nikon 真实相机采集）
- **模型**：EDSR r32f256（40.73M 参数）
- **初始化**：官方 EDSR ×2 权重 `weights/edsr_x2-0edfb8a3.pt` 微调
- **训练**：80 epoch，Adam（lr 1e-5，milestones [50, 70]，γ=0.5），L1 loss，AMP
- **配置**：
  - L0 Baseline：patch 96，batch 16
  - P1 Large Patch：patch 192，batch 4
  - P2 XLarge Patch：patch 256，batch 2
- **评测**：test 集 100 张，RGB float32 PSNR/SSIM；每个 checkpoint 跑标准评测 + Self-Ensemble ×8
- **协议**：与初始实验一致（见 `evaluate_edsr.py` 中的 `PROTOCOL`）；Bicubic 基线用 `torch.interpolate(scale_factor=2, mode=bicubic)` 生成

## 3. 结果（test 集，mean ± std）

| 配置 | Patch | Test PSNR | SSIM | Δ vs L0 |
|------|-------|-----------|------|---------|
| Bicubic | — | 30.0198 | 0.8654 | −1.809 |
| L0 Baseline | 96 | 31.8288 ± 0.0033 | 0.9004 ± 0.0000 | — |
| L0 + SE ×8 | 96 | 31.9105 ± 0.0018 | 0.9019 ± 0.0000 | +0.082 |
| P1 Large Patch | 192 | 31.9867 ± 0.0041 | 0.9030 ± 0.0002 | +0.158 |
| P1 + SE ×8 | 192 | 32.0761 ± 0.0051 | 0.9044 ± 0.0002 | +0.247 |
| P2 XLarge Patch | 256 | 32.0506 ± 0.0084 | 0.9040 ± 0.0001 | +0.222 |
| **P2 + SE ×8** | **256** | **32.1423 ± 0.0054** | **0.9054 ± 0.0001** | **+0.314** |

## 4. 逐 seed 结果

### 标准评测

| 配置 | seed 42 | seed 43 | seed 44 |
|------|---------|---------|---------|
| L0 Baseline | 31.8242 | 31.8308 | 31.8314 |
| P1 Large Patch | 31.9903 | 31.9809 | 31.9889 |
| P2 XLarge Patch | 32.0483 | 32.0618 | 32.0416 |

### Self-Ensemble ×8

| 配置 | seed 42 | seed 43 | seed 44 |
|------|---------|---------|---------|
| L0 + SE ×8 | 31.9083 | 31.9128 | 31.9104 |
| P1 + SE ×8 | 32.0800 | 32.0689 | 32.0794 |
| P2 + SE ×8 | 32.1401 | 32.1497 | 32.1370 |

### 分相机（标准评测 PSNR）

| 配置 | Canon | Nikon |
|------|-------|-------|
| L0 Baseline | 32.26 | 31.39 |
| P1 Large Patch | 32.44 | 31.53 |
| P2 XLarge Patch | 32.50 | 31.60 |

## 5. 与初始实验（ljh 单 seed）对比

| 配置 | ljh（README 单 seed） | 本次（3 seed mean±std） | 差异 |
|------|:---:|:---:|:---:|
| Bicubic | 30.02 | 30.0198 | — |
| L0 Baseline | 31.82 | 31.8288 ± 0.0033 | +0.009 |
| P1 Large Patch | 31.99 | 31.9867 ± 0.0041 | −0.003 |
| P1 + SE ×8 | 32.08 | 32.0761 ± 0.0051 | −0.004 |
| P2 XLarge Patch | 32.05 | 32.0506 ± 0.0084 | +0.001 |
| P2 + SE ×8 | 32.14 | 32.1423 ± 0.0054 | +0.002 |

**所有配置误差 ≤ 0.01 dB，完全复现初始结果；3 seed 标准差 ≤ 0.01 dB，结果稳定。**

## 6. 结论

1. **大 Patch Size 有效**：patch 96→192 涨 +0.16 dB，192→256 再涨 +0.06 dB。
2. **Self-Ensemble ×8 是稳定的免费涨点**：约 +0.09 dB，无需额外训练。
3. 两项改进正交叠加：Large Patch (+0.22) + SE (+0.09) = **+0.31 dB**。
4. 多 seed 结果稳定（std ≤ 0.01 dB），单 seed 即可代表该配置的性能。

## 7. 复现命令

```bash
# 训练（3 配置 × 3 seed，2 卡显存感知调度）
python scripts/run_ablation.py configs/repro/L0_seed42.yaml configs/repro/L0_seed43.yaml \
    configs/repro/L0_seed44.yaml configs/repro/P1_seed42.yaml configs/repro/P1_seed43.yaml \
    configs/repro/P1_seed44.yaml configs/repro/P2_seed42.yaml configs/repro/P2_seed43.yaml \
    configs/repro/P2_seed44.yaml

# 评测（标准 + Self-Ensemble ×8，4 卡并发）
python scripts/run_eval.py

# 汇总
python scripts/aggregate_results.py
```

详细逐图结果见 `results/repro/<配置>[_se]/eval_per_image.json`，汇总见 `results/repro/aggregate.json`。
