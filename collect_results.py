"""Collect all ablation eval results into a single comparison table."""
from __future__ import annotations

import json
from pathlib import Path

import sys


def main() -> None:
    results_dir = Path("results/ablation")
    if not results_dir.exists():
        print("No results directory found. Run experiments first.")
        sys.exit(1)

    experiments = sorted(results_dir.iterdir())
    if not experiments:
        print("No experiment results found.")
        sys.exit(1)

    rows = []
    for exp_dir in experiments:
        summary_file = exp_dir / "eval_summary.json"
        if not summary_file.exists():
            continue
        with open(summary_file, encoding="utf-8") as f:
            s = json.load(f)
        rows.append({
            "experiment": exp_dir.name,
            "model": s.get("model_type", "?"),
            "loss": s.get("loss", "?"),
            "params_M": s.get("params_M", "?"),
            "psnr": s.get("model_psnr", "?"),
            "ssim": s.get("model_ssim", "?"),
            "bicubic_psnr": s.get("bicubic_psnr", "?"),
            "bicubic_ssim": s.get("bicubic_ssim", "?"),
        })

    if not rows:
        print("No eval_summary.json files found.")
        sys.exit(1)

    # Print table
    header = f"{'Experiment':<25s} {'Model':<14s} {'Loss':<12s} {'Params':>7s} {'PSNR':>8s} {'SSIM':>8s} {'Bic.PSNR':>9s}"
    print("=" * len(header))
    print("EDSR Ablation Results — RealSR V3 x2 Test Set")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    baseline_psnr = None
    for r in rows:
        if r["experiment"] == "L0_baseline":
            baseline_psnr = r["psnr"] if isinstance(r["psnr"], (int, float)) else None

    for r in rows:
        delta = ""
        if baseline_psnr and isinstance(r["psnr"], (int, float)):
            d = r["psnr"] - baseline_psnr
            delta = f" ({d:+.2f})" if r["experiment"] != "L0_baseline" else " (base)"

        psnr_str = f"{r['psnr']:.4f}" if isinstance(r["psnr"], (int, float)) else str(r["psnr"])
        ssim_str = f"{r['ssim']:.6f}" if isinstance(r["ssim"], (int, float)) else str(r["ssim"])
        bpsnr_str = f"{r['bicubic_psnr']:.4f}" if isinstance(r["bicubic_psnr"], (int, float)) else str(r["bicubic_psnr"])
        params_str = f"{r['params_M']}M" if isinstance(r["params_M"], (int, float)) else str(r["params_M"])

        print(f"{r['experiment']:<25s} {r['model']:<14s} {r['loss']:<12s} {params_str:>7s} {psnr_str:>8s}{delta:<8s} {ssim_str:>8s} {bpsnr_str:>9s}")

    print("-" * len(header))

    # Save as JSON
    out = Path("results/ablation/comparison.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
