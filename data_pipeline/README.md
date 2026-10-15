# 数据模块使用说明

训练代码通过 `RealSRDataset` 读取数据，通过 `collate_pairs` 组成批次。模型和训练循环由训练方编写。

## 1. 准备环境和数据

使用 Python 3.11 或更高版本，以下命令均从仓库根目录执行：

```bash
python -m pip install -r data_pipeline/requirements.txt
python -m data_pipeline.realsr_demo fetch --root data/realsr
```

已有数据可跳过 `fetch`。数据目录中保留官方 `Canon/Nikon → Train/Test → 2` 结构及原文件名。

固定划分已包含在 `data_pipeline/splits/realsr_x2.json` 中，团队共用这份清单。换机器后通过 Dataset 的 `root` 指定本机数据目录；它必须对应清单中 `lr_path/hr_path` 的相对路径起点。

## 2. 接入训练代码

从仓库根目录导入：

```python
import json
import torch
from torch.utils.data import DataLoader
from data_pipeline.realsr_demo.torch_data import RealSRDataset, collate_pairs

with open("data_pipeline/splits/realsr_x2.json", encoding="utf-8") as f:
    report = json.load(f)

train_set = RealSRDataset(
    report, root="data/realsr", split="train",
    patch_size=96, mode="none", seed=42, repeats=1,
)
val_set = RealSRDataset(report, root="data/realsr", split="val")
test_set = RealSRDataset(report, root="data/realsr", split="test")

train_loader = DataLoader(
    train_set, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=collate_pairs, generator=torch.Generator().manual_seed(42),
)
val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_pairs)
test_loader = DataLoader(test_set, batch_size=1, collate_fn=collate_pairs)

batch = next(iter(train_loader))
lr, hr = batch["lr"], batch["hr"]
print(lr.shape, hr.shape)  # [4,3,96,96]、[4,3,192,192]
```

接入已有训练循环，每轮开始调用 `set_epoch`：

```python
# model、optimizer、device、epochs 由训练代码定义
for epoch in range(epochs):
    train_set.set_epoch(epoch)
    model.train()
    for batch in train_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        optimizer.zero_grad(set_to_none=True)
        sr = model(lr)  # 输出形状必须与 hr 相同
        loss = torch.nn.functional.l1_loss(sr, hr)
        loss.backward()
        optimizer.step()
```

## 3. 固定返回接口

| 字段 | `dataset[i]` | `collate_pairs` 组成的 batch |
|---|---|---|
| `lr` | `[3,H,W]` Tensor | `[B,3,H,W]` Tensor |
| `hr` | `[3,2H,2W]` Tensor | `[B,3,2H,2W]` Tensor |
| `meta` | dict，包含 `pair_id`、`split` | `list[dict]`，长度为 B |

图像张量均为 **CPU、RGB、`torch.float32`、范围 `[0,1]`**。训练方负责 `.to(device)`，不需要再次除以 255。`meta` 用于追踪样本，模型只接收 `lr`；DataLoader 使用 `collate_fn=collate_pairs`。

## 4. 参数约定

| 参数 | 用法 |
|---|---|
| `split` | `train`、`val`、`test`；从 manifest 选择，不自行重新划分 |
| `patch_size` | 默认 `0` 为整图；设为 `96` 得到 LR 96×96／HR 192×192 的同步裁剪 |
| `mode` | 默认 `none`；可选增强见下表 |
| `repeats` | 默认 `1`；训练集长度为训练图像数 × repeats |
| `seed` | 默认 `42`；控制裁剪与增强随机性，配合每轮 `set_epoch(epoch)` 使用 |
| `root` | 本机数据根目录；省略时使用 manifest 中的路径 |
| `mix_strength` | 默认 `0.3`；控制局部混合强度，仅用于 `blend/consistent_blend` |
| `apply_probability` | 默认 `0.5`；控制混合触发概率，不控制旋转和翻转 |

整图训练使用 `patch_size=0, batch_size=1`；patch 训练可使用更大的 batch。**验证和测试始终返回整图、关闭增强、每对读取一次，使用 batch size 1。** 示例使用 `num_workers=0`；启用多进程时保持 `persistent_workers=False`；Windows 上将训练入口放在 `if __name__ == "__main__":` 内。

| `mode` | 训练时的行为 |
|---|---|
| `none` | 关闭增强；`patch_size>0` 时仍会随机同步裁剪 |
| `geometry` | LR／HR 同步旋转、翻转 |
| `mixup` | 几何增强后，另取一个训练样本，对 LR／HR 使用同一个 λ 混合；需要正数 patch_size 和至少两对训练图 |
| `blend` | 几何增强后，将下采样 HR 局部混入 LR，HR 目标不再混合 |
| `consistent_blend` | 在局部混合前增加一致性门控 |

接入仓库的 EDSR 时使用 `scale=2`，训练与验证循环都按 `batch["lr"]`、`batch["hr"]` 取张量，并使用 `collate_pairs`。当前根目录的 `train.py` 仍使用原来的数据入口；按上面的接口替换其 Dataset 与批次读取即可。

## 5. 评测接口

验证集用于选择模型，官方测试集仅用于最终评测。调用 `data_pipeline.realsr_demo.metrics.rgb_metrics(sr, hr)` 计算统一的 RGB 指标：输入为范围 `[0,1]`、形状 `[H,W,3]` 的 NumPy 浮点数组（Tensor 需先移到 CPU 并转换维度），返回 `psnr_rgb`、`ssim_rgb`。模型结果先在完整图像上计算每张图的分数，再取算术平均；不额外裁边、不先转 uint8。Bicubic 基线调用同模块的 `bicubic_x2(lr)`，LR 也使用上述 HWC NumPy 格式。

## 6. 检查接口

```bash
python -m unittest discover -s data_pipeline/tests -v
```
