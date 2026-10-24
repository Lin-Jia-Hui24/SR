"""Evaluate EDSR variants on RealSR V3 test set.

Usage:
    python evaluate_edsr.py --config configs/ablation/L0_baseline.yaml --checkpoint outputs/ablation/L0_baseline/best.pt
    python evaluate_edsr.py --config configs/ablation/L0_baseline.yaml --checkpoint outputs/ablation/L0_baseline/best.pt --output results/L0
    python evaluate_edsr.py --config configs/ablation/L0_baseline.yaml --checkpoint outputs/ablation/L0_baseline/best.pt --output results/L0_x8 --self-ensemble
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from edsr.model import build_model
from edsr.utils import self_ensemble_predict
from data_pipeline.realsr_demo.data import load_pair
from data_pipeline.realsr_demo.metrics import PROTOCOL, bicubic_x2, rgb_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate EDSR on RealSR V3 test set.")
    p.add_argument("--config", required=True, help="Training config YAML (for model architecture).")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    p.add_argument("--manifest", default=None, help="RealSR manifest JSON (overrides config).")
    p.add_argument("--output", default=None, help="Output directory for results.")
    p.add_argument("--device", default="auto")
    p.add_argument("--self-ensemble", action="store_true", help="Use self-ensemble x8 (4 rotations × 2 flips).")
    return p.parse_args()


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


@torch.inference_mode()
def evaluate(cfg: dict, checkpoint_path: str, manifest_path: str | None, output_dir: str | None, device_str: str, use_self_ensemble: bool = False) -> dict:
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Load model
    model_cfg = cfg["model"]
    model = build_model(model_cfg)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cfg.get('type', 'edsr')} | {param_count/1e6:.2f}M params | device={device}")

    # Load manifest
    manifest = manifest_path or cfg["data"]["manifest"]
    with open(manifest, encoding="utf-8") as f:
        report = json.load(f)

    test_pairs = [p for p in report["pairs"] if p["official_split"] == "test"]
    if not test_pairs:
        raise ValueError("No test pairs found in manifest.")
    print(f"Evaluating on {len(test_pairs)} test pairs...")

    root = report["root"]
    rows = []
    started = time.perf_counter()

    for idx, pair in enumerate(test_pairs):
        lr, hr = load_pair(root, pair)
        # Model inference: [H,W,3] float32 → tensor → model → numpy
        lr_tensor = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).unsqueeze(0).to(device)
        if use_self_ensemble:
            sr = self_ensemble_predict(model, lr_tensor)[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        else:
            sr = model(lr_tensor)[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)

        # Bicubic baseline
        bicubic = bicubic_x2(lr)

        bm = rgb_metrics(bicubic, hr)
        sm = rgb_metrics(sr, hr)

        row = {
            "pair_id": pair["pair_id"],
            "camera": pair["camera"],
            "lr_h": lr.shape[0], "lr_w": lr.shape[1],
            "bicubic_psnr": bm["psnr_rgb"], "bicubic_ssim": bm["ssim_rgb"],
            "model_psnr": sm["psnr_rgb"], "model_ssim": sm["ssim_rgb"],
        }
        rows.append(row)
        print(f"  [{idx+1}/{len(test_pairs)}] {pair['pair_id']} | "
              f"bicubic={bm['psnr_rgb']:.2f}/{bm['ssim_rgb']:.4f}  "
              f"model={sm['psnr_rgb']:.2f}/{sm['ssim_rgb']:.4f}")

    elapsed = time.perf_counter() - started

    # Aggregate
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "model_type": model_cfg.get("type", "edsr"),
        "loss": cfg["train"].get("loss", "l1"),
        "self_ensemble": use_self_ensemble,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_images": len(rows),
        "params_M": round(param_count / 1e6, 2),
        "seconds": round(elapsed, 1),
        "protocol": PROTOCOL,
        "bicubic_psnr": round(mean("bicubic_psnr"), 4),
        "bicubic_ssim": round(mean("bicubic_ssim"), 6),
        "model_psnr": round(mean("model_psnr"), 4),
        "model_ssim": round(mean("model_ssim"), 6),
    }

    # Per-camera aggregation
    cameras = sorted({r["camera"] for r in rows})
    for cam in cameras:
        cam_rows = [r for r in rows if r["camera"] == cam]
        summary[f"{cam}_psnr"] = round(float(np.mean([r["model_psnr"] for r in cam_rows])), 4)
        summary[f"{cam}_ssim"] = round(float(np.mean([r["model_ssim"] for r in cam_rows])), 6)

    print(f"\n{'='*60}")
    print(f"Model: {summary['model_type']} | Loss: {summary['loss']} | {summary['params_M']}M params")
    print(f"Bicubic:  PSNR={summary['bicubic_psnr']:.4f} dB  SSIM={summary['bicubic_ssim']:.6f}")
    print(f"Model:    PSNR={summary['model_psnr']:.4f} dB  SSIM={summary['model_ssim']:.6f}")
    print(f"Time: {elapsed:.1f}s on {device}")
    print(f"{'='*60}")

    # Save results
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "eval_per_image.json").write_text(
            json.dumps(json_safe(rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Results saved to {out}")

    return summary


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    evaluate(cfg, args.checkpoint, args.manifest, args.output, args.device, args.self_ensemble)


if __name__ == "__main__":
    main()
