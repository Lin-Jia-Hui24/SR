from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from edsr.model import EDSR
from edsr.weights import load_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDSR inference on one image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None, help="YAML model config; required for a raw published weight file.")
    parser.add_argument("--non-strict", action="store_true", help="Allow only compatible tensors to load (not suitable for a baseline).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "config" in state and "model" in state:
        model_cfg = state["config"]["model"]
    elif args.config:
        import yaml
        with open(args.config, encoding="utf-8") as handle:
            model_cfg = yaml.safe_load(handle)["model"]
    else:
        raise ValueError("Raw published weights need --config with matching model settings.")
    model = EDSR(**model_cfg).to(device)
    report = load_weights(model, args.checkpoint, device, strict=not args.non_strict, num_blocks=model_cfg["num_blocks"])
    print(f"Loaded {report['loaded']}/{report['model_tensors']} model tensors.")
    model.eval()
    image = Image.open(args.input).convert("RGB")
    with torch.no_grad():
        output = model(to_tensor(image).unsqueeze(0).to(device)).squeeze(0).clamp(0, 1).cpu()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(output).save(destination)


if __name__ == "__main__":
    main()
