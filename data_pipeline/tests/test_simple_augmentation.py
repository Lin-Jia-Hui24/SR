"""简易增强入口：成对坐标、复现性和训练接口。"""
import unittest

import torch
from torch.nn import functional as F

from data_pipeline import augment_pair
from data_pipeline.realsr_demo.model import TinySR


class SimpleAugmentationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(29)
        self.lr = torch.rand((3, 16, 24), generator=generator)
        self.hr = self.lr.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    def test_full_image_preserves_pair_coordinates_and_tensor_contract(self):
        observed_shapes = set()
        for seed in range(16):
            torch.manual_seed(seed)
            lr, hr = augment_pair(self.lr, self.hr)
            observed_shapes.add(tuple(lr.shape))
            self.assertEqual(lr.dtype, self.lr.dtype)
            self.assertEqual(hr.dtype, self.hr.dtype)
            self.assertEqual(lr.device, self.lr.device)
            self.assertEqual(hr.device, self.hr.device)
            self.assertEqual(hr.shape, (3, lr.shape[1] * 2, lr.shape[2] * 2))
            self.assertTrue(torch.equal(hr[:, ::2, ::2], lr))
            self.assertTrue(torch.equal(lr.flatten().sort().values, self.lr.flatten().sort().values))
            self.assertGreaterEqual(lr.min().item(), 0)
            self.assertLessEqual(lr.max().item(), 1)
        self.assertEqual(observed_shapes, {(3, 16, 24), (3, 24, 16)})

    def test_optional_crop_is_synchronized_at_x2(self):
        torch.manual_seed(9)
        lr, hr = augment_pair(self.lr, self.hr, patch_size=8)
        self.assertEqual(tuple(lr.shape), (3, 8, 8))
        self.assertEqual(tuple(hr.shape), (3, 16, 16))
        self.assertTrue(torch.equal(hr[:, ::2, ::2], lr))
        self.assertEqual(lr.dtype, torch.float32)

    def test_manual_seed_reproduces_crop_and_geometry(self):
        torch.manual_seed(42)
        first = augment_pair(self.lr, self.hr, patch_size=8)
        torch.manual_seed(42)
        second = augment_pair(self.lr, self.hr, patch_size=8)
        for actual, expected in zip(first, second):
            self.assertTrue(torch.equal(actual, expected))

    def test_inputs_are_not_modified(self):
        original_lr, original_hr = self.lr.clone(), self.hr.clone()
        for patch_size in (0, 8):
            torch.manual_seed(3)
            augment_pair(self.lr, self.hr, patch_size=patch_size)
            self.assertTrue(torch.equal(self.lr, original_lr))
            self.assertTrue(torch.equal(self.hr, original_hr))

    def test_augmented_pair_runs_one_optimizer_step(self):
        torch.manual_seed(18)
        lr, hr = augment_pair(self.lr, self.hr, patch_size=8)
        model = TinySR(width=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        prediction = model(lr.unsqueeze(0))
        self.assertEqual(prediction.shape, hr.unsqueeze(0).shape)
        loss = F.l1_loss(prediction, hr.unsqueeze(0))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
