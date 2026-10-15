from __future__ import annotations

import random
from pathlib import Path
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def image_paths(directory: str | Path) -> list[Path]:
    paths = sorted(path for path in Path(directory).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No images found in: {directory}")
    return paths


class SuperResolutionDataset(Dataset[tuple[Tensor, Tensor]]):
    """Returns (low-resolution, high-resolution) RGB tensors in [0, 1]."""

    def __init__(
        self,
        hr_dir: str | Path,
        scale: int,
        patch_size: int | None = None,
        augment: bool = False,
        lr_dir: str | Path | None = None,
    ) -> None:
        self.hr_paths = image_paths(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.lr_paths: dict[str, Path] | None = None
        if lr_dir:
            self.lr_paths = {path.stem: path for path in image_paths(lr_dir)}
            missing = [path.name for path in self.hr_paths if path.stem not in self.lr_paths]
            if missing:
                raise FileNotFoundError(f"Missing paired LR files, e.g. {missing[0]}")

    def __len__(self) -> int:
        return len(self.hr_paths)

    def _random_crop(self, hr: Image.Image) -> Image.Image:
        if not self.patch_size:
            return hr
        width, height = hr.size
        size = min(self.patch_size, width - width % self.scale, height - height % self.scale)
        if size < self.scale:
            raise ValueError(f"Image {hr.size} is too small for scale x{self.scale}")
        left = random.randint(0, width - size)
        top = random.randint(0, height - size)
        return TF.crop(hr, top, left, size, size)

    def _paired_random_crop(self, lr: Image.Image, hr: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.patch_size:
            return lr, hr
        lr_patch = self.patch_size // self.scale
        if lr_patch < 1:
            raise ValueError("patch_size must be at least scale")
        if lr.width < lr_patch or lr.height < lr_patch:
            raise ValueError(f"Paired LR image {lr.size} is too small for patch size {self.patch_size}")
        left = random.randint(0, lr.width - lr_patch)
        top = random.randint(0, lr.height - lr_patch)
        return (
            TF.crop(lr, top, left, lr_patch, lr_patch),
            TF.crop(hr, top * self.scale, left * self.scale, lr_patch * self.scale, lr_patch * self.scale),
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        hr_path = self.hr_paths[index]
        hr = Image.open(hr_path).convert("RGB")
        if self.lr_paths is None:
            hr = self._random_crop(hr)
            lr = hr.resize((hr.width // self.scale, hr.height // self.scale), Image.Resampling.BICUBIC)
        else:
            lr = Image.open(self.lr_paths[hr_path.stem]).convert("RGB")
            # Align paired samples to valid corresponding dimensions.
            width = min(hr.width // self.scale, lr.width)
            height = min(hr.height // self.scale, lr.height)
            hr, lr = hr.crop((0, 0, width * self.scale, height * self.scale)), lr.crop((0, 0, width, height))
            lr, hr = self._paired_random_crop(lr, hr)

        if self.augment:
            if random.random() < 0.5:
                hr, lr = TF.hflip(hr), TF.hflip(lr)
            if random.random() < 0.5:
                hr, lr = TF.vflip(hr), TF.vflip(lr)
            if random.random() < 0.5:
                hr, lr = TF.rotate(hr, 90), TF.rotate(lr, 90)
        return TF.to_tensor(lr), TF.to_tensor(hr)

