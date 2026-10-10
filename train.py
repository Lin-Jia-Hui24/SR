from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EDSR for image super-resolution.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path (overrides config).")
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
    train_set = SuperResolutionDataset(data_cfg["train_hr_dir"], scale, data_cfg["patch_size"], data_cfg["augment"], data_cfg["train_lr_dir"] if data_cfg["mode"] == "paired" else None)
    val_set = SuperResolutionDataset(data_cfg["val_hr_dir"], scale, None, False, data_cfg["val_lr_dir"] if data_cfg["mode"] == "paired" else None)
    loader_options = dict(num_workers=data_cfg["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, train_cfg["batch_size"], shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_set, 1, shuffle=False, **loader_options)

    model = EDSR(**model_cfg).to(device)
    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = MultiStepLR(optimizer, milestones=train_cfg["milestones"], gamma=train_cfg["gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    start_epoch, best_psnr = 1, float("-inf")
    resume = args.resume or train_cfg.get("resume")
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_psnr = state["epoch"] + 1, state.get("best_psnr", best_psnr)

    writer = SummaryWriter(output / "logs")
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        running_loss = 0.0
        for lr, hr in progress:
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
            for lr, hr in val_loader:
                prediction = model(lr.to(device)).clamp(0, 1)
                scores.append(psnr(prediction, hr.to(device), scale))
        val_psnr = sum(scores) / len(scores)
        scheduler.step()
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("psnr/validation", val_psnr, epoch)
        print(f"Epoch {epoch}: train L1={train_loss:.6f}, validation PSNR={val_psnr:.3f} dB")
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best_psnr": best_psnr, "config": cfg}
        save_checkpoint(output / "last.pt", **state)
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state["best_psnr"] = best_psnr
            save_checkpoint(output / "best.pt", **state)
        if epoch % train_cfg["save_every"] == 0:
            save_checkpoint(output / f"epoch_{epoch:03d}.pt", **state)
    writer.close()


if __name__ == "__main__":
    main()
