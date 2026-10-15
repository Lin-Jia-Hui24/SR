"""数据科学核心：正确配对、场景划分、RGB 读取和 ×2 坐标。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from data_pipeline.realsr_demo.data import assign_splits, load_pair, paired_crop, scan_pairs


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(42)
        for camera in ("Canon", "Nikon"):
            for split, count in (("Train", 4), ("Test", 1)):
                for scene in range(count):
                    self.make_pair(camera, split, f"{scene:03d}")

    def tearDown(self):
        self.temp.cleanup()

    def make_pair(self, camera="Canon", split="Train", scene="new", scale=2):
        directory = self.root / camera / split / str(scale)
        directory.mkdir(parents=True, exist_ok=True)
        lr = self.rng.integers(0, 256, (7, 9, 3), dtype=np.uint8)
        hr = np.repeat(np.repeat(lr, scale, 0), scale, 1)
        low, high = directory / f"{scene}_LR{scale}.png", directory / f"{scene}_HR.png"
        Image.fromarray(lr).save(low)
        Image.fromarray(hr).save(high)
        return low, high

    def test_camera_split_scene_pairing_and_rgb_loading(self):
        source = {"kind": "synthetic_fixture", "note": "test input only"}
        (self.root / "SOURCE.json").write_text(json.dumps(source))
        report = scan_pairs(self.root)
        self.assertEqual(report["counts"]["pairs"], 10)
        self.assertEqual(report["counts"]["by_official_split"], {"test": 2, "train": 8})
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["provenance"], source)
        self.assertEqual(len({p["pair_id"] for p in report["pairs"]}), 10)
        for pair in report["pairs"]:
            self.assertFalse(Path(pair["lr_path"]).is_absolute())
            lr, hr = load_pair(self.root, pair)
            self.assertEqual(lr.shape, (7, 9, 3))
            self.assertEqual(hr.shape, (14, 18, 3))
            self.assertEqual(lr.dtype, np.float32)
            self.assertTrue(0 <= lr.min() <= lr.max() <= 1)
            np.testing.assert_array_equal(hr[::2, ::2], lr)

    def test_only_x2_missing_hr_and_wrong_scale(self):
        self.make_pair(scale=3)
        _, missing_hr = self.make_pair(scene="missing")
        missing_hr.unlink()
        _, wrong_hr = self.make_pair(scene="wrong")
        Image.new("RGB", (9, 7)).save(wrong_hr)
        report = scan_pairs(self.root)
        self.assertEqual(len(report["pairs"]), 10)
        self.assertEqual({i["code"] for i in report["issues"]}, {"missing_hr", "invalid_scale"})
        self.assertTrue(all("/2/" in pair["lr_path"] for pair in report["pairs"]))

    def test_fixed_camera_stratified_scene_split_keeps_official_test(self):
        report = scan_pairs(self.root)
        original = deepcopy(report)
        first = assign_splits(report, val_fraction=.25, seed=9)
        report["pairs"].reverse()
        second = assign_splits(report, val_fraction=.25, seed=9)
        mapping = lambda r: {p["pair_id"]: p["split"] for p in r["pairs"]}
        self.assertEqual(mapping(first), mapping(second))
        self.assertTrue(all(p["split"] == p["official_split"] for p in original["pairs"]))
        self.assertEqual(first["counts"]["by_split"], {"test": 2, "train": 6, "val": 2})
        self.assertEqual({p["camera"] for p in first["pairs"] if p["split"] == "val"}, {"Canon", "Nikon"})
        # 模拟同一场景的多个训练视图，仍必须一起进入同一个集合。
        views = deepcopy(original)
        views["pairs"] += [dict(p, pair_id=p["pair_id"] + ":view2")
                           for p in original["pairs"] if p["official_split"] == "train"]
        grouped = assign_splits(views, val_fraction=.25, seed=9)
        groups = {}
        for pair in grouped["pairs"]:
            if pair["official_split"] == "test":
                self.assertEqual(pair["split"], "test")
            else:
                groups.setdefault((pair["camera"], pair["scene_id"]), set()).add(pair["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))

    def test_crop_maps_bottom_right_boundary_and_replays_seed(self):
        lr = np.arange(5 * 7 * 3, dtype=np.float32).reshape(5, 7, 3)
        hr = np.repeat(np.repeat(lr, 2, 0), 2, 1)
        class LastPosition:
            def integers(self, high):
                return high - 1
        low, high, meta = paired_crop(lr, hr, 3, LastPosition())
        self.assertEqual((meta["x"], meta["y"]), (4, 2))
        np.testing.assert_array_equal(low, lr[2:5, 4:7])
        np.testing.assert_array_equal(high, hr[4:10, 8:14])
        np.testing.assert_array_equal(high[::2, ::2], low)
        a = paired_crop(lr, hr, 3, np.random.default_rng(17))
        b = paired_crop(lr, hr, 3, np.random.default_rng(17))
        self.assertEqual(a[2], b[2])
        np.testing.assert_array_equal(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
