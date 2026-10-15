"""把成对图像交给 PyTorch；整图无增强是当前 baseline 的用法。"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import load_pair, paired_crop
from .augment import augment_pair, paired_mixup


class RealSRDataset(Dataset):
    def __init__(self, report, split="train", patch_size=0, mode="none", seed=42,
                 repeats=1, root=None, mix_strength=.3, apply_probability=.5):
        self.pairs = [p for p in report["pairs"] if p["split"] == split]
        self.root = root or report["root"]
        self.split, self.patch_size, self.mode = split, patch_size or None, mode
        self.seed, self.epoch = seed, 0
        self.repeats = repeats if split == "train" else 1
        self.mix_strength, self.apply_probability = mix_strength, apply_probability

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.pairs) * self.repeats

    def __getitem__(self, index):
        # 同一 seed/epoch/index 得到同一结果，与读取先后顺序无关。
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        pair = self.pairs[index % len(self.pairs)]
        lr, hr = load_pair(self.root, pair)
        meta = {"pair_id": pair["pair_id"], "split": self.split}
        if self.split == "train":
            crop = None
            if self.patch_size:
                lr, hr, crop = paired_crop(lr, hr, self.patch_size, rng)
            if self.mode == "none":
                aug = {"mode": "none", "applied": False,
                       "geometry": {"rot90_k": 0, "horizontal_flip": False},
                       "alpha": 0.0, "mask_bbox": None, "mix_fraction": 0.0,
                       "effective_strength": 0.0, "consistency_score": None,
                       "skip_reason": "augmentation_disabled"}
            else:
                lr, hr, aug = augment_pair(lr, hr, rng,
                    mode="geometry" if self.mode == "mixup" else self.mode,
                    mix_strength=self.mix_strength, apply_probability=self.apply_probability)
            if self.mode == "mixup":
                aug.update(mode="mixup", applied=False, skip_reason="probability_draw")
                if rng.random() < self.apply_probability:
                    current = index % len(self.pairs)
                    other = int(rng.integers(len(self.pairs) - 1))
                    other += other >= current
                    other_lr, other_hr = load_pair(self.root, self.pairs[other])
                    other_lr, other_hr, other_crop = paired_crop(other_lr, other_hr, self.patch_size, rng)
                    other_lr, other_hr, other_aug = augment_pair(other_lr, other_hr, rng, mode="geometry")
                    lr, hr, mix = paired_mixup(lr, hr, other_lr, other_hr, rng)
                    mix.update(other_pair_id=self.pairs[other]["pair_id"], other_crop=other_crop,
                               other_geometry=other_aug["geometry"])
                    aug.update(mixup=mix, applied=True, skip_reason=None, mix_fraction=1.0,
                               alpha=1.0 - mix["lambda"], effective_strength=1.0 - mix["lambda"])
            meta.update(crop=crop, augmentation=aug)
        return {"lr": torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))),
                "hr": torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1))), "meta": meta}


def collate_pairs(samples):
    return {"lr": torch.stack([s["lr"] for s in samples]),
            "hr": torch.stack([s["hr"] for s in samples]), "meta": [s["meta"] for s in samples]}
