"""RealSR V3 ×2：配对、固定划分、RGB 读取和同步裁剪。"""

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


def scan_pairs(root):
    """读取官方 Canon/Nikon/{Train,Test}/2 下的 LR2 与相邻 HR。"""
    root = Path(root).expanduser().resolve()
    pairs, issues = [], []
    file_count = 0
    for lr_path in sorted(root.rglob("*_LR2.png")):
        if lr_path.parent.name != "2":
            continue
        camera = lr_path.parents[2].name
        official_split = lr_path.parents[1].name.lower()
        if camera not in ("Canon", "Nikon") or official_split not in ("train", "test"):
            continue
        scene = lr_path.stem[:-4]  # 去掉 _LR2
        hr_path = lr_path.with_name(f"{scene}_HR.png")
        file_count += 1 + hr_path.exists()
        if not hr_path.exists():
            issues.append({"code": "missing_hr", "severity": "error",
                           "path": str(hr_path.relative_to(root)), "detail": "Missing corresponding HR"})
            continue
        with Image.open(lr_path) as lr, Image.open(hr_path) as hr:
            if lr.mode != "RGB" or hr.mode != "RGB":
                raise ValueError(f"Expected RGB images: {lr_path}")
            lr_size, hr_size = list(lr.size), list(hr.size)
        if hr_size != [2 * value for value in lr_size]:
            issues.append({"code": "invalid_scale", "severity": "error",
                           "path": str(lr_path.relative_to(root)), "detail": "HR must be exactly 2x LR"})
            continue
        pairs.append({
            "pair_id": f"{camera}:{official_split}:{scene}", "camera": camera,
            "official_split": official_split, "scene_id": scene, "split": official_split,
            "lr_path": lr_path.relative_to(root).as_posix(),
            "hr_path": hr_path.relative_to(root).as_posix(),
            "lr_size": lr_size, "hr_size": hr_size,
        })
    report = {
        "schema_version": 1, "scale": 2, "root": str(root), "pairs": pairs, "issues": issues,
        "counts": {
            "pairs": len(pairs), "candidate_files": file_count,
            "by_camera": dict(Counter(p["camera"] for p in pairs)),
            "by_official_split": dict(Counter(p["official_split"] for p in pairs)),
            "by_split": dict(Counter(p["split"] for p in pairs)),
        },
    }
    source = root / "SOURCE.json"
    if source.exists():
        report["provenance"] = json.loads(source.read_text(encoding="utf-8"))
    return report


def assign_splits(report, val_fraction=0.1, seed=42):
    """官方 Test 不动；按相机、场景从官方 Train 划出固定验证集。"""
    result = deepcopy(report)
    validation = set()
    cameras = sorted({p["camera"] for p in result["pairs"]})
    for camera in cameras:
        scenes = sorted({p["scene_id"] for p in result["pairs"]
                         if p["camera"] == camera and p["official_split"] == "train"})
        # 每台相机独立使用固定种子，保持已有划分且不受文件遍历顺序影响。
        camera_seed = int.from_bytes(hashlib.sha256(f"{seed}:{camera}".encode()).digest()[:8], "big")
        random.Random(camera_seed).shuffle(scenes)
        count = 0 if len(scenes) < 2 or val_fraction == 0 else min(
            len(scenes) - 1, max(1, int(len(scenes) * val_fraction + 0.5)))
        validation.update((camera, scene) for scene in scenes[:count])
    for pair in result["pairs"]:
        if pair["official_split"] == "test":
            pair["split"] = "test"
        else:
            pair["split"] = "val" if (pair["camera"], pair["scene_id"]) in validation else "train"
    result["counts"]["by_split"] = dict(Counter(p["split"] for p in result["pairs"]))
    result["split_config"] = {
        "val_fraction": val_fraction, "seed": seed,
        "group_by": ["camera", "scene_id"], "official_test_unchanged": True,
        "tiny_camera_policy": "one scene stays train; two or more reserve at least one val when fraction > 0",
    }
    return result


def load_pair(root, pair):
    """返回 [H,W,3] / [2H,2W,3] 的 float32 RGB，范围 [0,1]。"""
    root = Path(root)
    with Image.open(root / pair["lr_path"]) as image:
        lr = np.asarray(image, dtype=np.float32) / 255.0
    with Image.open(root / pair["hr_path"]) as image:
        hr = np.asarray(image, dtype=np.float32) / 255.0
    return lr, hr


def paired_crop(lr, hr, size, rng):
    """LR 取一处随机方块，HR 使用同一位置的两倍坐标。"""
    height, width = lr.shape[:2]
    x = int(rng.integers(width - size + 1))
    y = int(rng.integers(height - size + 1))
    lr_patch = lr[y:y + size, x:x + size].copy()
    hr_patch = hr[2 * y:2 * (y + size), 2 * x:2 * (x + size)].copy()
    return lr_patch, hr_patch, {"x": x, "y": y, "size": size,
                              "hr_x": 2 * x, "hr_y": 2 * y, "scale": 2}
