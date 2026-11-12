#!/usr/bin/env python
"""Aggregate 3-seed eval summaries into mean +/- std and save raw per-seed JSON.

Reads results/repro/<EXP>_seed<seed>[_se]/eval_summary.json and prints a table,
then writes results/repro/aggregate.json.
"""
import json
import statistics
from pathlib import Path

EXPS = ["L0", "P1", "P2"]
SEEDS = [42, 43, 44]


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(exp, se):
    psnrs, ssims = [], []
    suffix = "_se" if se else ""
    for seed in SEEDS:
        p = Path(f"results/repro/{exp}_seed{seed}{suffix}/eval_summary.json")
        if not p.exists():
            print(f"MISSING {p}")
            continue
        s = load(p)
        psnrs.append(s["model_psnr"])
        ssims.append(s["model_ssim"])
    if not psnrs:
        return None
    return {
        "n": len(psnrs),
        "psnr_mean": statistics.mean(psnrs),
        "psnr_std": statistics.pstdev(psnrs),
        "ssim_mean": statistics.mean(ssims),
        "ssim_std": statistics.pstdev(ssims),
    }


def main():
    b = load("results/repro/L0_seed42/eval_summary.json")
    bic = (b["bicubic_psnr"], b["bicubic_ssim"])

    print("=" * 78)
    print("EDSR x2 on RealSR V3 test set (100 imgs) - 3-seed mean +/- std (population)")
    print("=" * 78)
    print(f"{'config':<20} {'PSNR':>16} {'SSIM':>16}")
    print(f"{'Bicubic baseline':<20} {bic[0]:>9.4f}{'':>7} {bic[1]:>9.6f}")
    for exp in EXPS:
        for se, label in [(False, ""), (True, " + SE x8")]:
            r = agg(exp, se)
            if r:
                print(f"{exp + label:<20} {r['psnr_mean']:>8.4f}+/-{r['psnr_std']:<6.4f} "
                      f"{r['ssim_mean']:>8.6f}+/-{r['ssim_std']:<7.6f}")

    out = {}
    for exp in EXPS:
        out[exp] = {"std": {}, "se": {}}
        for se, key in [(False, "std"), (True, "se")]:
            suffix = "_se" if se else ""
            for seed in SEEDS:
                p = Path(f"results/repro/{exp}_seed{seed}{suffix}/eval_summary.json")
                s = load(p)
                out[exp][key][seed] = {
                    "psnr": s["model_psnr"], "ssim": s["model_ssim"],
                    "canon_psnr": s.get("Canon_psnr"), "nikon_psnr": s.get("Nikon_psnr"),
                }
    Path("results/repro/aggregate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("raw per-seed saved to results/repro/aggregate.json")


if __name__ == "__main__":
    main()
