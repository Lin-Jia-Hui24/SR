# EDSR 超分辨率训练框架

一个基于 PyTorch 的 EDSR 训练基线，包含模型、数据集、训练与验证、TensorBoard、AMP、断点恢复和单图推理。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据组织

默认的 `synthetic` 模式只需要高分辨率（HR）图像，会在训练时以双三次插值自动生成低分辨率（LR）图像：

```
data/
  train/HR/*.png
  val/HR/*.png
```

真正的 LR/HR 成对数据可将 `configs/default.yaml` 的 `data.mode` 改为 `paired`，并填写 `train_lr_dir`、`val_lr_dir`。LR 与 HR 文件须同名（扩展名可以不同）。

## 训练与继续训练

```powershell
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --resume outputs/edsr_realsr_x2/last.pt
```

训练结果会写入 `outputs/edsr_realsr_x2/`：`best.pt` 是验证 PSNR 最优权重，`last.pt` 可用于恢复训练，`logs/` 可被 TensorBoard 读取。

```powershell
tensorboard --logdir outputs/edsr_realsr_x2/logs
```

## 推理

```powershell
python infer.py --checkpoint outputs/edsr_x4/best.pt --input sample_lr.png --output result_sr.png
```

## RealSR ×2 训练

真实 LR／HR 配对、固定数据划分和增强工具见 [data_pipeline/README.md](data_pipeline/README.md)。默认配置直接使用 RealSR 原生 ×2 配对：模型输入 LR 后输出宽高两倍的结果，与 HR 目标逐像素比较。

```powershell
python -m data_pipeline.realsr_demo fetch --root data/realsr
python train.py --config configs/default.yaml
```

验证使用固定验证集的原生 LR/HR 图对。默认只开启同步几何增强；如需实验 `mixup`、`blend` 或 `consistent_blend`，修改配置中的 `data.augmentation`。

## 从官方 EDSR ×2 权重微调

下载官方完整 `r32f256x2` 权重至 `weights/edsr_x2-0edfb8a3.pt` 后运行：

```powershell
python train.py --config configs/finetune_realsr_x2.yaml
```

该配置使用低学习率完整微调 80 个 epoch。`best.pt` 始终保存验证集最佳模型；可用 `--resume outputs/edsr_realsr_x2_finetune/last.pt` 恢复中断训练。也可用 `--pretrained <权重路径>` 临时覆盖配置中的权重路径。

## 使用作者公布的权重做基线

将作者的 `.pt` 或 `.pth` 文件放入 `weights/`（该目录不会自动创建），**先确认该权重对应的放大倍率、特征通道数和残差块数量**。它们必须与配置中的 `model.scale`、`model.num_features`、`model.num_blocks` 一致；公开的 EDSR 基线通常是 x2/x3/x4、256 通道、32 个残差块，但请以作者说明为准。

原始权重推理时需显式提供模型配置：

```powershell
python infer.py --checkpoint weights/author_edsr_x4.pt --config configs/default.yaml --input sample_lr.png --output outputs/baseline.png
```

加载器支持本项目保存的 checkpoint，以及常见的 `model`、`state_dict`、`model_state_dict`、`params` 和 `params_ema` 封装格式；也兼容 EDSR-PyTorch 常见的模块键名。默认严格检查结构，因此它是可复现 baseline 的安全选择。若报告“不兼容”，不要直接加 `--non-strict` 后宣称它是 baseline；应先调整配置，或使用作者原仓库的推理代码。

如需在作者权重基础上微调，在配置中填写：

```yaml
train:
  pretrained: weights/author_edsr_x4.pt
  pretrained_strict: true
```

`pretrained` 仅初始化模型参数；`resume` 则同时恢复模型、优化器、学习率调度器和 epoch，二者不应同时使用。
