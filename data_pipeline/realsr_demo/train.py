"""无增强 TinySR 基线；完整验证图像的平均 L1 用于选择权重。"""
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import TinySR
from .torch_data import RealSRDataset, collate_pairs
from .utils import choose_device, write_csv
from .visualize import save_loss_curve


def train_baseline(report, output, epochs=2, patch_size=0, batch_size=1,
                   seed=42, device="auto"):
    """每轮遍历全部内部训练图像；patch_size=0 使用整图，验证始终整图。"""
    train_pairs = sorted((p for p in report["pairs"] if p["split"] == "train"), key=lambda p: p["pair_id"])
    val_pairs = sorted((p for p in report["pairs"] if p["split"] == "val"), key=lambda p: p["pair_id"])
    if epochs < 1 or not train_pairs or not val_pairs:
        raise ValueError("Training requires epochs >= 1 and nonempty train/validation splits")
    if any(p["official_split"] != "train" for p in train_pairs + val_pairs):
        raise ValueError("Official test images cannot be used for training or checkpoint selection")
    train_scenes = {(p["camera"], p["scene_id"]) for p in train_pairs}
    if train_scenes & {(p["camera"], p["scene_id"]) for p in val_pairs}:
        raise ValueError("Training and validation scenes must be disjoint")

    full_image = not patch_size
    batch_size = 1 if full_image else batch_size
    runtime_device = choose_device(device)
    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if runtime_device.type == "cpu":
        torch.set_num_threads(2)
    dataset = RealSRDataset({**report, "pairs": train_pairs}, split="train",
                           patch_size=patch_size, mode="none", seed=seed, repeats=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate_pairs, generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(RealSRDataset({**report, "pairs": val_pairs}, split="val"),
                            batch_size=1, num_workers=0, collate_fn=collate_pairs)
    model_config = {"class_name": "TinySR", "width": 16, "scale": 2}
    model = TinySR(width=model_config["width"]).to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train_ids = [p["pair_id"] for p in train_pairs]
    val_ids = [p["pair_id"] for p in val_pairs]
    protocol = {
        "scale": 2, "color": "RGB", "normalization": "float32 [0,1] input and target",
        "training_loss": "mean absolute error on unclipped float32 model output",
        "augmentation": "none",
        "training_spatial_mode": "full_image" if full_image else "one random paired crop per scene per epoch",
        "sampling": "each internal training scene once per epoch; repeats=1; drop_last=False",
        "validation": "all internal validation scenes, full image, mean of per-image unclipped RGB L1",
        "checkpoint_criterion": "minimum validation L1; earliest epoch retained on ties",
        "official_test": "not used for training or checkpoint selection",
        "optimizer": {"name": "Adam", "learning_rate": 0.001}, "precision": "float32; no AMP",
        "split_config": report.get("split_config", {}),
    }
    records, step_records = [], []
    best_loss, best_epoch, total_steps = math.inf, 0, 0
    started = time.perf_counter()
    print(f"TinySR: {len(train_pairs)} train, {len(val_pairs)} full-image validation, "
          f"{epochs} epochs, {runtime_device}, {'full images' if full_image else 'paired crops'}, "
          f"batch={batch_size}, augmentation=none", flush=True)
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        dataset.set_epoch(epoch - 1)
        model.train()
        train_sum, samples = 0.0, 0
        for batch_index, batch in enumerate(loader, start=1):
            low, high = batch["lr"].to(runtime_device), batch["hr"].to(runtime_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(low), high)
            loss.backward()
            optimizer.step()
            value, count = loss.item(), low.shape[0]
            train_sum += value * count
            samples += count
            total_steps += 1
            step_records.append({"epoch": epoch, "step": total_steps, "batch_size": count,
                                 "train_l1": value, "lr_width": low.shape[-1], "lr_height": low.shape[-2]})
            if batch_index % 20 == 0:
                print(f"epoch {epoch}/{epochs}, batch {batch_index}/{len(loader)}, L1={value:.6f}", flush=True)
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in validation:
                prediction = model(batch["lr"].to(runtime_device))
                val_losses.append(F.l1_loss(prediction, batch["hr"].to(runtime_device)).item())
        train_loss, val_loss = train_sum / samples, sum(val_losses) / len(val_losses)
        if not math.isfinite(train_loss + val_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
        record = {"epoch": epoch, "train_l1": train_loss, "val_l1": val_loss,
                  "train_samples": samples, "val_samples": len(val_losses),
                  "steps": len(loader), "total_steps": total_steps,
                  "seconds": time.perf_counter() - epoch_started}
        records.append(record)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            selection = {"criterion": "minimum validation per-image mean L1",
                         "validation_scope": "full_image", "epoch": epoch, "value": val_loss,
                         "val_pairs": len(val_pairs), "official_test_used": False}
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_config": model_config, "epoch": epoch, "seed": seed,
                        "checkpoint_selection": selection, "train_pair_ids": train_ids,
                        "validation_pair_ids": val_ids, "protocol": protocol}, output / "best.pt")
        write_csv(output / "loss.csv", records)
        write_csv(output / "step_loss.csv", step_records)
        save_loss_curve(output / "curve.svg", records)
        print(f"epoch {epoch}/{epochs}: train L1={train_loss:.6f}, validation L1={val_loss:.6f}, "
              f"best epoch={best_epoch}, {record['seconds']:.1f}s", flush=True)
    summary = {
        "purpose": "Two-epoch demo by default; no convergence claim",
        "device": str(runtime_device), "epochs": epochs, "seed": seed, "batch_size": batch_size,
        "patch_size_lr": patch_size or None, "full_image_training": full_image,
        "parameters": sum(p.numel() for p in model.parameters()), "model_config": model_config,
        "used_train_pairs": len(train_pairs), "validation_pairs": len(val_pairs),
        "samples_seen": len(train_pairs) * epochs, "steps_per_epoch": len(loader), "total_steps": total_steps,
        "seconds": time.perf_counter() - started, "best_epoch": best_epoch, "best_validation_l1": best_loss,
        "validation_scope": "full_image", "checkpoint": "best.pt", "checkpoint_selection": selection,
        "epoch_losses": records, "protocol": protocol, "train_pair_ids": train_ids, "validation_pair_ids": val_ids,
        "data_source": report.get("provenance", {}), "data_root": report["root"], "torch_version": str(torch.__version__),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
