"""Command-line workflow for the unaugmented RealSR V3 x2 baseline."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def read_manifest(path, root=None):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if root:
        report["root"] = str(Path(root).expanduser().resolve())
    errors = [issue for issue in report.get("issues", []) if issue.get("severity") == "error"]
    if errors:
        raise ValueError(f"Manifest contains {len(errors)} errors; inspect {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="RealSR V3 x2 — no-augmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download official archive and extract all x2 pairs")
    fetch.add_argument("--root", default="data/realsr")
    fetch.add_argument("--archive", default="data/RealSR_V3.tar.gz")
    scan = sub.add_parser("scan", help="Audit and split source images; never resizes images")
    scan.add_argument("--root", default="data/realsr")
    scan.add_argument("--manifest", "--output", default="outputs/baseline/manifest.json")
    scan.add_argument("--val-fraction", type=float, default=0.1)
    scan.add_argument("--seed", type=int, default=42)
    audit = sub.add_parser("audit-preview", help="Generate 20 stratified source-pair inspection examples")
    audit.add_argument("--manifest", default="outputs/baseline/manifest.json")
    audit.add_argument("--root", help="Optional relocated dataset root")
    audit.add_argument("--output", default="outputs/baseline/data_audit")
    audit.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train using only internal train and validation scenes")
    train.add_argument("--manifest", default="outputs/baseline/manifest.json")
    train.add_argument("--root", help="Optional relocated dataset root")
    train.add_argument("--output", default="outputs/baseline/train")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--patch-size", type=int, default=0, help="0 = full image; positive = paired LR crop")
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    evaluate = sub.add_parser("evaluate", help="Evaluate every official test image using RGB metrics")
    evaluate.add_argument("--manifest", default="outputs/baseline/manifest.json")
    evaluate.add_argument("--root", help="Optional relocated dataset root")
    evaluate.add_argument("--checkpoint", default="outputs/baseline/train/best.pt")
    evaluate.add_argument("--output", default="outputs/baseline/test")
    evaluate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    evaluate.add_argument("--tile-size", type=int, default=0, help="0 = direct full image; positive tiles still evaluate full SR")
    report = sub.add_parser("report", help="Generate a local, interactive HTML experiment report")
    report.add_argument("--output", default="outputs/baseline")
    args = parser.parse_args()
    if args.command == "fetch":
        from .fetch import fetch_full
        result = fetch_full(args.root, args.archive)
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))
    elif args.command == "scan":
        from .data import scan_pairs, assign_splits
        result = assign_splits(scan_pairs(args.root), args.val_fraction, args.seed)
        path = Path(args.manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        env = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": {package: importlib.metadata.version(package) for package in
                            ["torch", "numpy", "Pillow", "scikit-image", "scipy"]}}
        (path.parent / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"manifest": str(path.resolve()), "counts": result["counts"], "issues": result["issues"]}, ensure_ascii=False, indent=2))
        if any(issue["severity"] == "error" for issue in result["issues"]):
            raise SystemExit("Scan failed; no training should use this manifest")
    elif args.command == "audit-preview":
        from .audit_preview import build_audit_preview
        result = build_audit_preview(read_manifest(args.manifest, args.root), args.output, args.seed)
        print(json.dumps({"sample_count": result["sample_count"], "sheets": result["sheets"]}, indent=2))
    elif args.command == "train":
        from .train import train_baseline
        result = train_baseline(read_manifest(args.manifest, args.root), args.output, args.epochs,
                                args.patch_size, args.batch_size, args.seed, args.device)
        print(json.dumps({k: result[k] for k in ["device", "epochs", "used_train_pairs", "validation_pairs", "total_steps", "best_epoch", "best_validation_l1", "seconds"]}, indent=2))
    elif args.command == "evaluate":
        from .evaluate import evaluate_baseline
        result = evaluate_baseline(read_manifest(args.manifest, args.root), args.checkpoint,
                                   args.output, args.device, args.tile_size)
        print(json.dumps(result["aggregation"], indent=2))
    else:
        from .report import build_report
        print(build_report(args.output))


if __name__ == "__main__":
    main()
