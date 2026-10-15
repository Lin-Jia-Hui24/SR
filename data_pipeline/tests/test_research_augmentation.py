"""研究增强的成对语义；这些测试不证明增强能提高真实测试集指标。"""
import unittest

import torch

from data_pipeline.augmentation import _consistency_score, augment_pair


class ResearchAugmentationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(51)
        self.lr = torch.rand((3, 48, 64), generator=generator) * 0.7
        self.hr = self.lr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        self.other_lr = torch.rand((3, 56, 72), generator=generator) * 0.7
        self.other_hr = self.other_lr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    def test_none_preserves_full_image_and_allows_only_paired_crop(self):
        lr, hr = augment_pair(self.lr, self.hr, mode="none")
        self.assertTrue(torch.equal(lr, self.lr))
        self.assertTrue(torch.equal(hr, self.hr))
        torch.manual_seed(9)
        lr, hr = augment_pair(self.lr, self.hr, patch_size=24, mode="none")
        self.assertEqual(tuple(lr.shape), (3, 24, 24))
        self.assertEqual(tuple(hr.shape), (3, 48, 48))
        self.assertTrue(torch.equal(hr[:, ::2, ::2], lr))
        # 每个像素值唯一，可以反推出切片位置，确认没有额外翻转或旋转。
        top, left = torch.nonzero(self.lr[0] == lr[0, 0, 0], as_tuple=False)[0]
        self.assertTrue(torch.equal(lr, self.lr[:, top:top + 24, left:left + 24]))

    def test_mixup_uses_one_lambda_for_both_targets(self):
        lr = torch.full((3, 48, 64), 0.2)
        hr = torch.full((3, 96, 128), 0.3)
        other_lr = torch.full((3, 56, 72), 0.8)
        other_hr = torch.full((3, 112, 144), 0.9)
        torch.manual_seed(16)
        mixed_lr, mixed_hr = augment_pair(
            lr, hr, patch_size=24, mode="mixup", other=(other_lr, other_hr), p=1,
        )
        lambda_lr = (0.8 - mixed_lr.mean()) / 0.6
        lambda_hr = (0.9 - mixed_hr.mean()) / 0.6
        self.assertGreaterEqual(lambda_lr.item(), 0)
        self.assertLessEqual(lambda_lr.item(), 1)
        torch.testing.assert_close(lambda_lr, lambda_hr, atol=2e-6, rtol=0)
        torch.testing.assert_close(mixed_lr, torch.full_like(mixed_lr, mixed_lr.mean()))
        torch.testing.assert_close(mixed_hr, torch.full_like(mixed_hr, mixed_hr.mean()))

    def test_local_blend_preserves_target_and_changes_only_part_of_lr(self):
        hr = self.hr + 0.2
        torch.manual_seed(11)
        original_lr, original_hr = augment_pair(self.lr, hr, mode="geometry")
        torch.manual_seed(11)
        blended_lr, blended_hr = augment_pair(self.lr, hr, mode="hr_blend", p=1, strength=0.3)
        self.assertTrue(torch.equal(blended_hr, original_hr))
        changed = (blended_lr - original_lr).abs().amax(dim=0) > 1e-7
        self.assertTrue(changed.any())
        self.assertTrue((~changed).any())
        self.assertLess(changed.sum().item(), changed.numel() * 0.8)
        self.assertEqual(blended_lr.dtype, self.lr.dtype)
        self.assertGreaterEqual(blended_lr.min().item(), 0)
        self.assertLessEqual(blended_lr.max().item(), 1)

    def test_mixup_accepts_same_shape_non_square_full_images(self):
        other_lr, other_hr = self.lr * 0.5, self.hr * 0.5
        for seed in range(8):
            torch.manual_seed(seed)
            lr, hr = augment_pair(self.lr, self.hr, mode="mixup", other=(other_lr, other_hr), p=1)
            self.assertIn(tuple(lr.shape), ((3, 48, 64), (3, 64, 48)))
            self.assertTrue(torch.equal(hr[:, ::2, ::2], lr))

    def test_mixup_requires_a_compatible_other_pair_before_probability_draw(self):
        for p in (0, 1):
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    augment_pair(self.lr, self.hr, mode="mixup", p=p)
                with self.assertRaises(ValueError):
                    augment_pair(self.lr, self.hr, mode="mixup", other=(self.other_lr, self.other_hr), p=p)

    def test_zero_probability_and_zero_strength_match_geometry(self):
        options = [
            {"mode": "mixup", "other": (self.other_lr, self.other_hr), "p": 0},
            {"mode": "hr_blend", "p": 0},
            {"mode": "consistent_blend", "p": 0},
            {"mode": "hr_blend", "p": 1, "strength": 0},
            {"mode": "consistent_blend", "p": 1, "strength": 0},
        ]
        torch.manual_seed(5)
        expected = augment_pair(self.lr, self.hr, patch_size=24, mode="geometry")
        for option in options:
            with self.subTest(options=option["mode"], p=option["p"], strength=option.get("strength")):
                torch.manual_seed(5)
                actual = augment_pair(self.lr, self.hr, patch_size=24, **option)
                for result, baseline in zip(actual, expected):
                    self.assertTrue(torch.equal(result, baseline))

    def test_flat_and_small_regions_skip_consistency_blend(self):
        lr = torch.full((3, 48, 64), 0.2)
        hr = torch.full((3, 96, 128), 0.6)
        self.assertEqual(float(_consistency_score(lr, torch.full_like(lr, 0.6))), 0)
        self.assertEqual(float(_consistency_score(self.lr[:, :8, :8], self.lr[:, :8, :8])), 0)
        torch.manual_seed(14)
        expected = augment_pair(lr, hr, mode="geometry")
        torch.manual_seed(14)
        actual = augment_pair(lr, hr, mode="consistent_blend", p=1, strength=0.3)
        for result, baseline in zip(actual, expected):
            self.assertTrue(torch.equal(result, baseline))

    def test_consistency_score_prefers_aligned_over_shifted_structure(self):
        aligned = torch.zeros((3, 64, 64))
        aligned[:, 12:43, 18:39] = 0.8
        aligned[:, 36:53, 43:55] = 0.5
        shifted = torch.zeros_like(aligned)
        shifted[:, :, 2:] = aligned[:, :, :-2]
        aligned_score = float(_consistency_score(aligned, aligned))
        shifted_score = float(_consistency_score(aligned, shifted))
        self.assertGreater(aligned_score, 0.99)
        self.assertGreaterEqual(shifted_score, 0)
        self.assertLess(shifted_score, aligned_score - 0.1)

    def test_research_modes_replay_seed_and_leave_all_inputs_unchanged(self):
        originals = [tensor.clone() for tensor in (self.lr, self.hr, self.other_lr, self.other_hr)]
        for mode in ("none", "geometry", "mixup", "hr_blend", "consistent_blend"):
            with self.subTest(mode=mode):
                kwargs = {"mode": mode, "p": 1, "patch_size": 24}
                if mode == "mixup":
                    kwargs["other"] = (self.other_lr, self.other_hr)
                torch.manual_seed(42)
                first = augment_pair(self.lr, self.hr, **kwargs)
                torch.manual_seed(42)
                second = augment_pair(self.lr, self.hr, **kwargs)
                for actual, expected in zip(first, second):
                    self.assertTrue(torch.equal(actual, expected))
                for actual, expected in zip((self.lr, self.hr, self.other_lr, self.other_hr), originals):
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
