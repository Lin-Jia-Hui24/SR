from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from edsr.model import EDSR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDSR inference on one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EDSR(**state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    image = Image.open(args.input).convert("RGB")
    with torch.no_grad():
        output = model(to_tensor(image).unsqueeze(0).to(device)).squeeze(0).clamp(0, 1).cpu()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(output).save(destination)


if __name__ == "__main__":
    main()
