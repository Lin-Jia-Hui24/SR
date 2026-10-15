"""Complete official-test RGB evaluation with explicit inference coverage."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_pair
from .metrics import PROTOCOL, bicubic_x2, rgb_metrics
from .model import TinySR
from .utils import choose_device, write_csv
from .visualize import save_comparison


def json_metrics(value):
    """Keep mathematically valid infinite PSNR in standards-compliant JSON."""
    if isinstance(value, dict):
        return {key: json_metrics(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_metrics(item) for item in value]
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


@torch.inference_mode()
def infer_full(model, lr, device="cpu", tile_size=0):
    """Return every output pixel. Tiling, when requested, uses an 8-LR-pixel halo.

    This halo is valid specifically for TinySR's four 3x3 convolution layers
    (radius 4) and bicubic interpolation. Do not reuse it for arbitrary models.
    """
    device = torch.device(device)
    tensor = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).unsqueeze(0)
    height, width = lr.shape[:2]
    if tile_size == 0:
        return model(tensor.to(device))[0].permute(1, 2, 0).cpu().numpy()
    if not isinstance(model, TinySR) or tile_size < 16:
        raise ValueError("Tiling requires TinySR and tile_size >= 16")
    result = np.empty((2 * height, 2 * width, 3), dtype=np.float32)
    halo = 8
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y1, x1 = min(height, y + tile_size), min(width, x + tile_size)
            top, left = max(0, y - halo), max(0, x - halo)
            bottom, right = min(height, y1 + halo), min(width, x1 + halo)
            sr = model(tensor[:, :, top:bottom, left:right].to(device))[0].permute(1, 2, 0).cpu().numpy()
            result[2*y:2*y1, 2*x:2*x1] = sr[2*(y-top):2*(y1-top), 2*(x-left):2*(x1-left)]
    return result


def evaluate_baseline(report, checkpoint, output, device="auto", tile_size=0, allow_subset=False):
    pairs = [pair for pair in report["pairs"] if pair["official_split"] == "test"]
    if not pairs or any(pair["split"] != "test" for pair in pairs):
        raise ValueError("Official test split must remain intact")
    if len(pairs) != 100 and not allow_subset:
        raise ValueError(f"Expected all 100 official test pairs, found {len(pairs)}; no partial benchmark")
    checkpoint = Path(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    used_ids = set(saved["train_pair_ids"] + saved["validation_pair_ids"])
    if used_ids & {p["pair_id"] for p in pairs}:
        raise ValueError("Test images were used for training or checkpoint selection")
    model = TinySR(width=saved["model_config"]["width"])
    model.load_state_dict(saved["state_dict"], strict=True)
    device = choose_device(device)
    model.to(device).eval()
    torch.set_num_threads(2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Fixed selection before inspecting metrics, two test scenes per camera.
    visual_ids = set()
    for camera in sorted({p["camera"] for p in pairs}):
        camera_pairs = sorted((p for p in pairs if p["camera"] == camera), key=lambda p: p["pair_id"])
        visual_ids.update(p["pair_id"] for p in camera_pairs[:2])
    rows, examples = [], []
    started = time.perf_counter()
    for index, pair in enumerate(pairs):
        lr, hr = load_pair(report["root"], pair)
        inference = "full_image" if tile_size == 0 else f"tiled_full_image_{tile_size}_halo8"
        sr = infer_full(model, lr, device, tile_size)
        bicubic = bicubic_x2(lr)
        bm, sm = rgb_metrics(bicubic, hr), rgb_metrics(sr, hr)
        row = {"pair_id": pair["pair_id"], "camera": pair["camera"],
               "lr_width": lr.shape[1], "lr_height": lr.shape[0],
               "hr_width": hr.shape[1], "hr_height": hr.shape[0], "inference": inference,
               "bicubic_psnr_rgb": bm["psnr_rgb"], "bicubic_ssim_rgb": bm["ssim_rgb"],
               "model_psnr_rgb": sm["psnr_rgb"], "model_ssim_rgb": sm["ssim_rgb"]}
        rows.append(row)
        if pair["pair_id"] in visual_ids:
            examples.append(save_comparison(output, pair, lr, bicubic, sr, hr, row))
        print(f"test {index+1}/{len(pairs)} {pair['pair_id']} | RGB PSNR bicubic={bm['psnr_rgb']:.4f}, model={sm['psnr_rgb']:.4f}", flush=True)
    keys = ["bicubic_psnr_rgb", "bicubic_ssim_rgb", "model_psnr_rgb", "model_ssim_rgb"]
    groups = {"all": rows, **{camera: [r for r in rows if r["camera"] == camera] for camera in sorted({r["camera"] for r in rows})}}
    aggregates = {name: {"images": len(group), **{key: float(np.mean([r[key] for r in group])) for key in keys}}
                  for name, group in groups.items()}
    write_csv(output / "test_per_image.csv", rows)
    summary = {"scope": "complete official test" if len(pairs) == 100 else "explicit subset; not a full benchmark",
               "test_images": len(pairs), "device": str(device), "seconds": time.perf_counter()-started,
               "protocol": PROTOCOL, "aggregation": aggregates,
               "checkpoint": str(checkpoint.resolve()),
               "selection": saved.get("checkpoint_selection"), "inference_counts": {name: sum(r["inference"] == name for r in rows) for name in sorted({r["inference"] for r in rows})},
               "examples": examples, "data_source": report.get("provenance"),
               "visual_selection": "first two lexicographically sorted official test pairs per camera, selected before evaluation"}
    summary["nonfinite_encoding"] = "Mathematically infinite PSNR is stored as the string Infinity in JSON and inf in CSV; no finite cap"
    (output / "test_summary.json").write_text(json.dumps(json_metrics(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
