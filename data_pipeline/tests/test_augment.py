"""Tests of coordinate/range guarantees, not of augmentation's real benefit."""

import json
import unittest

import numpy as np

from data_pipeline.realsr_demo.augment import (
    augment_pair,
    consistency_score,
    downsample_hr,
    paired_mixup,
)


class PairedAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.lr = np.random.default_rng(7).random((48, 64, 3), dtype=np.float32)
        self.hr = np.repeat(np.repeat(self.lr, 2, axis=0), 2, axis=1)

    def test_geometry_uses_same_coordinates_in_all_eight_transforms(self):
        seen = set()
        for seed in range(50):
            lr, hr, meta = augment_pair(self.lr, self.hr, np.random.default_rng(seed))
            seen.add((meta["geometry"]["rot90_k"], meta["geometry"]["horizontal_flip"]))
            np.testing.assert_array_equal(hr[::2, ::2], lr)
            self.assertEqual(hr.shape[:2], (2 * lr.shape[0], 2 * lr.shape[1]))
        self.assertEqual(len(seen), 8)

    def test_inputs_unchanged_and_target_not_mixed(self):
        original_lr, original_hr = self.lr.copy(), self.hr.copy()
        for mode in ("geometry", "blend", "consistent_blend"):
            lr, hr, meta = augment_pair(
                self.lr, self.hr, np.random.default_rng(11), mode=mode, apply_probability=1.0
            )
            _, expected_hr, _ = augment_pair(self.lr, self.hr, np.random.default_rng(11))
            np.testing.assert_array_equal(hr, expected_hr)
            self.assertEqual(lr.dtype, np.float32)
            self.assertGreaterEqual(lr.min(), 0.0)
            self.assertLessEqual(lr.max(), 1.0)
            json.dumps(meta, allow_nan=False)
        np.testing.assert_array_equal(self.lr, original_lr)
        np.testing.assert_array_equal(self.hr, original_hr)

    def test_reproducible(self):
        for mode in ("geometry", "blend", "consistent_blend"):
            first = augment_pair(self.lr, self.hr, np.random.default_rng(13), mode, apply_probability=1)
            second = augment_pair(self.lr, self.hr, np.random.default_rng(13), mode, apply_probability=1)
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(first[2], second[2])

    def test_zero_strength_and_zero_probability_preserve_geometry_result(self):
        expected_lr, expected_hr, _ = augment_pair(self.lr, self.hr, np.random.default_rng(5))
        for options in ({"mix_strength": 0}, {"apply_probability": 0}):
            lr, hr, meta = augment_pair(
                self.lr, self.hr, np.random.default_rng(5), "consistent_blend", **options
            )
            np.testing.assert_array_equal(lr, expected_lr)
            np.testing.assert_array_equal(hr, expected_hr)
            self.assertFalse(meta["applied"])

    def test_blend_changes_only_local_region(self):
        lr, _, meta = augment_pair(
            self.lr, self.hr, np.random.default_rng(11), "blend", apply_probability=1
        )
        original, _, _ = augment_pair(self.lr, self.hr, np.random.default_rng(11))
        self.assertTrue(meta["applied"])
        x0, y0, x1, y1 = meta["mask_bbox"]
        outside = np.ones(lr.shape[:2], dtype=bool)
        outside[y0:y1, x0:x1] = False
        np.testing.assert_array_equal(lr[outside], original[outside])
        self.assertGreater(np.abs(lr - original).sum(), 0)
        self.assertGreater(meta["effective_strength"], 0)
        self.assertLessEqual(meta["effective_strength"], meta["alpha"])

    def test_flat_regions_skip_consistency_gate(self):
        lr = np.full((48, 48, 3), 0.35, dtype=np.float32)
        hr = np.full((96, 96, 3), 0.35, dtype=np.float32)
        result, _, meta = augment_pair(lr, hr, np.random.default_rng(4), "consistent_blend", apply_probability=1)
        self.assertIsNone(consistency_score(lr, lr))
        self.assertFalse(meta["applied"])
        self.assertEqual(meta["skip_reason"], "insufficient_structure")
        np.testing.assert_array_equal(result, lr)

    def test_synthetic_shift_reduces_heuristic_score(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[12:43, 18:39] = 0.8
        image[36:53, 43:55] = 0.5
        shifted = np.zeros_like(image)
        shifted[:, 2:] = image[:, :-2]
        identical_score = consistency_score(image, image)
        shifted_score = consistency_score(image, shifted)
        self.assertGreater(identical_score, 0.99)
        self.assertLess(shifted_score, identical_score - 0.1)

    def test_float_downsampling_does_not_quantize_to_uint8(self):
        value = np.float32(0.1234567)
        hr = np.full((24, 24, 3), value, dtype=np.float32)
        reference = downsample_hr(hr, (12, 12))
        np.testing.assert_allclose(reference, value, atol=1e-7, rtol=0)
        self.assertEqual(reference.dtype, np.float32)

    def test_paired_mixup_uses_shared_lambda(self):
        other_lr, other_hr = np.zeros_like(self.lr), np.zeros_like(self.hr)
        lr, hr, meta = paired_mixup(
            self.lr, self.hr, other_lr, other_hr, np.random.default_rng(2), lambda_value=0.3
        )
        np.testing.assert_allclose(lr, self.lr * 0.3)
        np.testing.assert_allclose(hr, self.hr * 0.3)
        np.testing.assert_array_equal(hr[::2, ::2], lr)
        self.assertEqual(meta["lambda"], 0.3)
        json.dumps(meta, allow_nan=False)

if __name__ == "__main__":
    unittest.main()
