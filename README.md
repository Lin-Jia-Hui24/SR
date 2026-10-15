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
python train.py --config configs/default.yaml --resume outputs/edsr_x4/last.pt
```

训练结果会写入 `outputs/edsr_x4/`：`best.pt` 是验证 PSNR 最优权重，`last.pt` 可用于恢复训练，`logs/` 可被 TensorBoard 读取。

```powershell
tensorboard --logdir outputs/edsr_x4/logs
```

## 推理

```powershell
python infer.py --checkpoint outputs/edsr_x4/best.pt --input sample_lr.png --output result_sr.png
```

## RealSR V3 ×2 数据接口

真实 LR／HR 配对、固定数据划分和增强接口见 [data_pipeline/README.md](data_pipeline/README.md)。从仓库根目录导入 `data_pipeline.realsr_demo.torch_data`；接入 EDSR 时设置 `scale=2`，按说明读取字典批次。
