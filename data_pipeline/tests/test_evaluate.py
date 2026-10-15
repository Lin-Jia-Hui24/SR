"""Full-image inference and evaluation contracts, including trained residuals."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from data_pipeline.realsr_demo.data import scan_pairs
from data_pipeline.realsr_demo.evaluate import evaluate_baseline, infer_full
from data_pipeline.realsr_demo.metrics import bicubic_x2
from data_pipeline.realsr_demo.report import build_report
from data_pipeline.realsr_demo.model import TinySR


class FullInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        cls.model = TinySR(width=6)
        inputs = torch.rand(2, 3, 25, 31)
        target = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False) * 0.7 + 0.1
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=0.01)
        # Several real optimizer steps make all convolution paths active. A
        # zero-initialized residual head would conceal convolution tile seams.
        for _ in range(5):
            optimizer.zero_grad()
            loss = F.mse_loss(cls.model(inputs), target)
            loss.backward()
            optimizer.step()
        cls.model.eval()

    def test_trained_residual_is_active(self):
        self.assertGreater(float(self.model.up[0].weight.detach().abs().sum()), 0.1)
        lr = np.random.default_rng(91).random((19, 27, 3), dtype=np.float32)
        prediction = infer_full(self.model, lr)
        self.assertGreater(float(np.abs(prediction.clip(0, 1) - bicubic_x2(lr)).max()), 0.01)

    def test_default_inference_keeps_all_pixels_and_matches_direct_model(self):
        lr = np.random.default_rng(12).random((23, 37, 3), dtype=np.float32)
        original = lr.copy()
        with torch.inference_mode():
            direct = self.model(torch.from_numpy(lr.transpose(2, 0, 1).copy()).unsqueeze(0))
        result = infer_full(self.model, lr)
        self.assertEqual(result.shape, (46, 74, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(lr, original)
        np.testing.assert_allclose(result, direct[0].permute(1, 2, 0).numpy(), atol=1e-7, rtol=0)

    def test_tiled_matches_full_for_odd_rectangles_small_images_and_boundaries(self):
        rng = np.random.default_rng(46)
        for height, width in ((17, 19), (33, 57), (65, 47), (7, 23)):
            lr = rng.random((height, width, 3), dtype=np.float32)
            # Strong true-boundary structure and corners exercise zero-padding
            # at the real image edge independently of interior tile seams.
            lr[0, :] = 0
            lr[-1, :] = 1
            lr[:, 0] = (0, 1, 0)
            lr[:, -1] = (1, 0, 1)
            expected = infer_full(self.model, lr)
            for tile_size in (16, 19, 32, 64):
                with self.subTest(shape=(height, width), tile_size=tile_size):
                    actual = infer_full(self.model, lr, tile_size=tile_size)
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_tiling_writes_every_output_pixel(self):
        lr = np.random.default_rng(3).random((35, 53, 3), dtype=np.float32)
        expected = infer_full(self.model, lr)
        # Poison the allocated output so missing coverage cannot pass because
        # uninitialized memory happened to contain plausible image values.
        with patch("data_pipeline.realsr_demo.evaluate.np.empty", side_effect=lambda shape, dtype: np.full(shape, np.nan, dtype=dtype)):
            actual = infer_full(self.model, lr, tile_size=16)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_unsupported_tiles_fail_explicitly(self):
        lr = np.ones((23, 29, 3), dtype=np.float32) * 0.5
        with self.assertRaises(ValueError):
            infer_full(self.model, lr, tile_size=15)
        with self.assertRaises(ValueError):
            infer_full(torch.nn.Identity(), lr, tile_size=16)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable in this process")
    def test_mps_tiled_matches_full_with_trained_weights(self):
        model = TinySR(width=6)
        model.load_state_dict(self.model.state_dict())
        model = model.to("mps").eval()
        for shape in ((17, 19), (37, 53), (7, 23)):
            lr = np.random.default_rng(sum(shape)).random((*shape, 3), dtype=np.float32)
            expected = infer_full(model, lr, device="mps")
            for tile_size in (16, 32):
                with self.subTest(shape=shape, tile_size=tile_size):
                    actual = infer_full(model, lr, device="mps", tile_size=tile_size)
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


class FullEvaluationTests(unittest.TestCase):
    def test_perfect_reconstruction_keeps_infinite_psnr_in_csv_json_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset" / "Canon" / "Test" / "2"
            source.mkdir(parents=True)
            Image.fromarray(np.zeros((13, 17, 3), dtype=np.uint8)).save(source / "000_LR2.png")
            Image.fromarray(np.zeros((26, 34, 3), dtype=np.uint8)).save(source / "000_HR.png")
            report = scan_pairs(root / "dataset")
            checkpoint = root / "perfect_fixture.pt"
            torch.save({"state_dict": TinySR(width=6).state_dict(), "model_config": {"width": 6},
                        "train_pair_ids": [], "validation_pair_ids": [],
                        "checkpoint_selection": {"official_test_used": False,
                            "note": "untrained synthetic perfect reconstruction fixture"}}, checkpoint)
            output = root / "report"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = evaluate_baseline(report, checkpoint, output / "test", device="cpu", allow_subset=True)
            self.assertTrue(math.isinf(result["aggregation"]["all"]["model_psnr_rgb"]))
            def reject_nonstandard_constant(value):
                raise AssertionError(f"JSON contains nonstandard bare numeric constant: {value}")
            saved = json.loads((output / "test/test_summary.json").read_text(),
                               parse_constant=reject_nonstandard_constant)
            self.assertEqual(saved["aggregation"]["all"]["model_psnr_rgb"], "Infinity")
            self.assertEqual(saved["examples"][0]["metrics"]["bicubic_psnr_rgb"], "Infinity")
            with (output / "test/test_per_image.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertTrue(math.isinf(float(rows[0]["model_psnr_rgb"])))
            self.assertEqual(float(rows[0]["model_ssim_rgb"]), 1.0)
            (output / "manifest.json").write_text(json.dumps(report), encoding="utf-8")
            (output / "train").mkdir()
            # Minimal report-only fixture; no model was trained or selected on
            # these test images. The values exist solely to exercise rendering.
            training = {"used_train_pairs": 0, "validation_pairs": 0, "epochs": 0,
                        "device": "cpu", "batch_size": 1, "best_epoch": 0,
                        "best_validation_l1": 0.0, "full_image_training": True,
                        "validation_scope": "synthetic report fixture", "seed": 0}
            (output / "train/summary.json").write_text(json.dumps(training), encoding="utf-8")
            page = Path(build_report(output)).read_text(encoding="utf-8")
            self.assertIn("<td>inf</td>", page)
            self.assertIn("<td>—</td>", page)
            self.assertNotIn("<td>nan</td>", page)

    def test_eval_uses_all_border_pixels_and_averages_per_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            source = dataset_root / "Canon" / "Test" / "2"
            source.mkdir(parents=True)
            expected = []
            all_squared_errors = []
            for scene, (height, width, border) in enumerate(((13, 17, 190), (21, 27, 245))):
                lr = np.full((height, width, 3), 128, dtype=np.uint8)
                hr = np.full((2 * height, 2 * width, 3), 128, dtype=np.uint8)
                hr[:2] = border
                hr[-2:] = border
                hr[:, :2] = border
                hr[:, -2:] = border
                Image.fromarray(lr).save(source / f"{scene:03d}_LR2.png")
                Image.fromarray(hr).save(source / f"{scene:03d}_HR.png")
                prediction = bicubic_x2(lr.astype(np.float32) / 255)
                error = prediction.astype(np.float64) - (hr.astype(np.float32) / 255).astype(np.float64)
                expected.append(10 * math.log10(1 / np.mean(error**2)))
                all_squared_errors.append((error**2).ravel())
            report = scan_pairs(dataset_root)
            checkpoint = root / "fixture.pt"
            torch.save({"state_dict": TinySR(width=6).state_dict(), "model_config": {"width": 6},
                        "train_pair_ids": [], "validation_pair_ids": [],
                        "checkpoint_selection": {"official_test_used": False,
                            "note": "untrained synthetic fixture, not a benchmark"}}, checkpoint)
            output = root / "evaluation"
            summary = evaluate_baseline(report, checkpoint, output, device="cpu", allow_subset=True)
            self.assertEqual(summary["test_images"], 2)
            self.assertEqual(summary["inference_counts"], {"full_image": 2})
            self.assertIn("subset", summary["scope"])
            measured = summary["aggregation"]["all"]["bicubic_psnr_rgb"]
            self.assertAlmostEqual(measured, float(np.mean(expected)), places=6)
            pooled_psnr = 10 * math.log10(1 / np.mean(np.concatenate(all_squared_errors)))
            self.assertGreater(abs(measured - pooled_psnr), 0.1)
            self.assertLess(measured, 40)  # A center-only crop would miss all error.
            with (output / "test_per_image.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual((rows[0]["hr_height"], rows[0]["hr_width"]), ("26", "34"))
            saved = json.loads((output / "test_summary.json").read_text())
            self.assertEqual(saved["test_images"], 2)
            for example in summary["examples"]:
                self.assertTrue((output / example["comparison"]).is_file())
                with Image.open(output / example["folder"] / "hr.png") as image:
                    self.assertIn(image.size, ((34, 26), (54, 42)))
            with self.assertRaises(ValueError):
                evaluate_baseline(report, checkpoint, root / "forbidden_partial", device="cpu")


if __name__ == "__main__":
    unittest.main()
