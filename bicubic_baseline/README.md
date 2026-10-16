# RealSR V3 ×2 Bicubic 基线评测

本项目使用 Bicubic 对真实低分辨率图像进行两倍上采样，并以配对的真实高分辨率图像为参考，计算 RGB PSNR 和 RGB SSIM。

## 实验结果

使用 RealSR V3 ×2 测试集，共 100 对图像，其中 Canon、Nikon 各 50 对。

| 方法 | 图像数量 | RGB PSNR（dB）↑ | RGB SSIM ↑ |
|---|---:|---:|---:|
| Bicubic ×2 | 100 | 30.013952 | 0.866014 |

逐图成绩及测试集平均值保存在 `results/bicubic/metrics_rgb_x2.csv`。

## 数据准备

数据来源：[RealSR 官方仓库](https://github.com/csjcai/RealSR)。

使用 RealSR V3 中 Canon 和 Nikon 的 `Test/2` 图像，将 LR、HR 分别整理为以下结构：

```text
data/test_x2/
├── LR/
│   ├── Canon/
│   └── Nikon/
└── HR/
    ├── Canon/
    └── Nikon/
```

LR 与 HR 的相对子目录保持一致，文件名中的场景编号一一对应，例如：

```text
LR/Canon/Canon_001_LR2.png
HR/Canon/Canon_001_HR.png
```

程序递归查找图片，通过文件名匹配图像对，并检查重复编号、缺失配对和尺寸关系。HR 的高、宽必须分别为 LR 的两倍。

## 环境与运行

在 `baseline_eval.py` 所在目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

运行评测：

```powershell
python baseline_eval.py --lr-dir data/test_x2/LR --hr-dir data/test_x2/HR --save-dir results/bicubic
```

增加 `--save-images` 可同时保存重建图像：

```powershell
python baseline_eval.py --lr-dir data/test_x2/LR --hr-dir data/test_x2/HR --save-dir results/bicubic --save-images
```

数据位于其他位置时，将 `--lr-dir` 和 `--hr-dir` 替换为实际路径。

| 参数 | 含义 |
|---|---|
| `--lr-dir` | LR 图像目录，必填 |
| `--hr-dir` | HR 图像目录，必填 |
| `--save-dir` | 结果目录，默认为 `results/bicubic` |
| `--save-images` | 保存 Bicubic 重建图像 |

## 实现与评测设置

### Bicubic 重建

使用 OpenCV 的 `cv2.INTER_CUBIC` 对 LR 图像进行两倍上采样。输入、输出均为 8 位 RGB 图像，输出尺寸与对应 HR 一致。

### RGB PSNR

先对全部像素和 RGB 通道计算均方误差：

$$
\mathrm{MSE} = \frac{1}{3HW}
\sum_{h=1}^{H}\sum_{w=1}^{W}\sum_{c=1}^{3}
\left(I_{\mathrm{HR}}(h,w,c)-I_{\mathrm{SR}}(h,w,c)\right)^2
$$

其中，H、W 为 HR 图像的高度和宽度，c 表示 RGB 颜色通道。

再计算 PSNR：

$$
\mathrm{PSNR} = 10\log_{10}\left(\frac{255^2}{\mathrm{MSE}}\right)
$$

图像像素范围为 0～255，PSNR 单位为 dB。

误差计算采用 `float64`，PSNR 单位为 dB。

### RGB SSIM

使用 `skimage.metrics.structural_similarity`：

```python
structural_similarity(
    hr_rgb,
    sr_rgb,
    data_range=255.0,
    channel_axis=2,
)
```

采用默认的 7×7 均匀窗口及样本协方差设置，对三个颜色通道的 SSIM 取平均。

### 结果汇总

评测输入为完整 RGB 图像，像素范围为 0～255，额外边界裁剪宽度为 0。SSIM 在函数内部按有效窗口汇总局部相似度。

每张图像分别计算 PSNR、SSIM，再对 100 张图像取算术平均，每张图像权重相同。

## 输出文件

```text
results/bicubic/
├── metrics_rgb_x2.csv
└── bicubic/
    ├── Canon/
    └── Nikon/
```

CSV 包含 `method`、`image`、`psnr_db`、`ssim` 四列，最后一行为 `AVERAGE`。指定 `--save-images` 后，重建图像按相机子目录保存。

## 文件说明

| 文件 | 内容 |
|---|---|
| `baseline_eval.py` | 图像配对、Bicubic 重建、RGB 指标计算及结果保存 |
| `requirements.txt` | Python 依赖 |
| `results/bicubic/metrics_rgb_x2.csv` | 逐图成绩和测试集平均值 |
