from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from edsr.data import SuperResolutionDataset
from edsr.model import EDSR
from edsr.utils import psnr, save_checkpoint, set_seed
from edsr.weights import load_weights
from data_pipeline.realsr_demo.torch_data import RealSRDataset, collate_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EDSR for image super-resolution.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path (overrides config).")
    parser.add_argument("--pretrained", default=None, help="Published model weights used to initialize fine-tuning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"] if cfg["device"] != "auto" else "cpu")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    scale = model_cfg["scale"]
    realsr = data_cfg["mode"] == "realsr_x2"
    if realsr:
        with open(data_cfg["manifest"], encoding="utf-8") as handle:
            report = json.load(handle)
        if scale != 2:
            raise ValueError("RealSR provides native x2 pairs; set model.scale to 2.")
        train_set = RealSRDataset(
            report, root=data_cfg["root"], split="train", patch_size=data_cfg["patch_size"],
            mode=data_cfg["augmentation"], seed=cfg["seed"],
        )
        val_set = RealSRDataset(report, root=data_cfg["root"], split="val")
    else:
        paired = data_cfg["mode"] == "paired"
        train_set = SuperResolutionDataset(data_cfg["train_hr_dir"], scale, data_cfg["patch_size"], data_cfg["augment"], data_cfg["train_lr_dir"] if paired else None)
        val_set = SuperResolutionDataset(data_cfg["val_hr_dir"], scale, None, False, data_cfg["val_lr_dir"] if paired else None)
    loader_options = dict(num_workers=data_cfg["num_workers"], pin_memory=device.type == "cuda")
    collate = collate_pairs if realsr else None
    train_loader = DataLoader(train_set, train_cfg["batch_size"], shuffle=True, drop_last=True, collate_fn=collate, **loader_options)
    val_loader = DataLoader(val_set, 1, shuffle=False, collate_fn=collate, **loader_options)

    model = EDSR(**model_cfg).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    start_epoch, best_psnr, epochs_without_improvement = 1, float("-inf"), 0
    resume = args.resume or train_cfg.get("resume")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_psnr = state["epoch"] + 1, state.get("best_psnr", best_psnr)
        epochs_without_improvement = state.get("epochs_without_improvement", 0)
    elif args.pretrained or train_cfg.get("pretrained"):
        report = load_weights(
            model, args.pretrained or train_cfg["pretrained"], device,
            strict=train_cfg.get("pretrained_strict", True), num_blocks=model_cfg["num_blocks"],
        )
        print(f"Loaded published weights: {report['loaded']}/{report['model_tensors']} tensors")

    writer = SummaryWriter(output / "logs")
    patience = train_cfg.get("early_stopping_patience")
    min_delta = train_cfg.get("early_stopping_min_delta", 0.0)
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        if realsr:
            train_set.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        running_loss = 0.0
        for batch in progress:
            lr, hr = (batch["lr"], batch["hr"]) if realsr else batch
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(lr), hr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / len(train_loader)

        model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                lr, hr = (batch["lr"], batch["hr"]) if realsr else batch
                prediction = model(lr.to(device)).clamp(0, 1)
                scores.append(psnr(prediction, hr.to(device), scale))
        val_psnr = sum(scores) / len(scores)
        scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("psnr/validation", val_psnr, epoch)
        print(f"Epoch {epoch}: train L1={train_loss:.6f}, validation PSNR={val_psnr:.3f} dB")
        improved = val_psnr > best_psnr + min_delta
        if improved:
            best_psnr = val_psnr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_psnr": best_psnr,
            "epochs_without_improvement": epochs_without_improvement,
            "config": cfg,
        }
        save_checkpoint(output / "last.pt", **state)
        if improved:
            save_checkpoint(output / "best.pt", **state)
        if epoch % train_cfg["save_every"] == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", **state)
        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without a PSNR improvement of {min_delta} dB.")
            break
    writer.close()


if __name__ == "__main__":
    main()
