"""数据→PyTorch 批次→模型与 RGB 指标的核心契约。"""
import math
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from data_pipeline.realsr_demo.data import assign_splits, load_pair, scan_pairs
from data_pipeline.realsr_demo.metrics import bicubic_x2, rgb_metrics
from data_pipeline.realsr_demo.model import TinySR
from data_pipeline.realsr_demo.torch_data import RealSRDataset, collate_pairs


class DataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(129)
        for split, count in (("Train", 4), ("Test", 1)):
            directory = self.root / "Canon" / split / "2"
            directory.mkdir(parents=True)
            for scene in range(count):
                lr = rng.integers(0, 256, (32 + scene, 40 + scene, 3), dtype=np.uint8)
                hr = np.repeat(np.repeat(lr, 2, 0), 2, 1)
                Image.fromarray(lr).save(directory / f"{scene:03d}_LR2.png")
                Image.fromarray(hr).save(directory / f"{scene:03d}_HR.png")
        self.report = assign_splits(scan_pairs(self.root), val_fraction=.25)

    def tearDown(self):
        self.temporary.cleanup()

    def test_reproducible_paired_crop_and_epoch_variation(self):
        first = RealSRDataset(self.report, patch_size=12, repeats=2, seed=6)
        second = RealSRDataset(self.report, patch_size=12, repeats=2, seed=6)
        initial = [first[i] for i in range(len(first))]
        for i, sample in enumerate(initial):
            self.assertTrue(torch.equal(sample["lr"], second[i]["lr"]))
            self.assertTrue(torch.equal(sample["hr"][:, ::2, ::2], sample["lr"]))
            self.assertEqual(sample["meta"], second[i]["meta"])
            self.assertFalse(sample["meta"]["augmentation"]["applied"])
        first.set_epoch(1)
        self.assertTrue(any(not torch.equal(first[i]["lr"], old["lr"]) for i, old in enumerate(initial)))
        second.set_epoch(1)
        self.assertTrue(torch.equal(first[0]["lr"], second[0]["lr"]))

    def test_batch_is_rgb_float_chw_at_native_x2(self):
        dataset = RealSRDataset(self.report, patch_size=12, repeats=1)
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_pairs)))
        self.assertEqual(tuple(batch["lr"].shape), (2, 3, 12, 12))
        self.assertEqual(tuple(batch["hr"].shape), (2, 3, 24, 24))
        self.assertEqual(batch["lr"].dtype, torch.float32)
        self.assertTrue(torch.equal(batch["hr"][:, :, ::2, ::2], batch["lr"]))

    def test_full_image_none_preserves_pixels_and_validation_is_fixed(self):
        for option in (None, 0):
            for split in ("train", "val", "test"):
                dataset = RealSRDataset(self.report, split=split, patch_size=option, repeats=1)
                for i, pair in enumerate(dataset.pairs):
                    sample = dataset[i]
                    lr, hr = load_pair(self.root, pair)
                    np.testing.assert_array_equal(sample["lr"].permute(1, 2, 0).numpy(), lr)
                    np.testing.assert_array_equal(sample["hr"].permute(1, 2, 0).numpy(), hr)
                    if split == "train":
                        self.assertIsNone(sample["meta"]["crop"])
                        self.assertFalse(sample["meta"]["augmentation"]["applied"])
                    else:
                        dataset.set_epoch(999)
                        self.assertTrue(torch.equal(sample["lr"], dataset[i]["lr"]))
                        self.assertNotIn("augmentation", sample["meta"])

    def test_optional_mixup_uses_another_training_pair(self):
        dataset = RealSRDataset(self.report, patch_size=12, mode="mixup", seed=9, apply_probability=1)
        sample = dataset[0]
        aug = sample["meta"]["augmentation"]
        self.assertTrue(aug["applied"])
        self.assertNotEqual(aug["mixup"]["other_pair_id"], sample["meta"]["pair_id"])
        self.assertIn(aug["mixup"]["other_pair_id"], {p["pair_id"] for p in dataset.pairs})
        self.assertTrue(torch.equal(sample["hr"][:, ::2, ::2], sample["lr"]))


class MetricAndModelContractTests(unittest.TestCase):
    def test_identity_metrics(self):
        target = np.random.default_rng(12).random((24, 32, 3), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exact = rgb_metrics(target, target)
        self.assertTrue(math.isinf(exact["psnr_rgb"]))
        self.assertAlmostEqual(exact["ssim_rgb"], 1.0)

    def test_each_rgb_channel_contributes_equally_to_psnr(self):
        target = np.full((20, 24, 3), .5, dtype=np.float32)
        for channel in range(3):
            prediction = target.copy()
            prediction[..., channel] += .1
            result = rgb_metrics(prediction, target)
            self.assertAlmostEqual(result["psnr_rgb"], 10 * math.log10(3 / .1**2), places=4)
            self.assertLess(result["ssim_rgb"], 1.0)

    def test_tiny_model_initial_output_is_bicubic(self):
        lr = np.random.default_rng(6).random((15, 18, 3), dtype=np.float32)
        model = TinySR().eval()
        tensor = torch.from_numpy(lr.transpose(2, 0, 1).copy()).unsqueeze(0)
        with torch.no_grad():
            initial = model(tensor)[0].permute(1, 2, 0).numpy().clip(0, 1)
        self.assertEqual(initial.shape, (30, 36, 3))
        np.testing.assert_allclose(initial, bicubic_x2(lr), atol=1e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
