"""Reproducible stratified data previews; never used to compute metrics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import load_pair, paired_crop
from .metrics import bicubic_x2


def _rng(seed, label):
    derived = int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(derived)


def _image(rgb):
    # PNG quantization belongs only to this visual artifact, not the evaluation
    # pipeline. bicubic_x2 already returns a clipped float32 display baseline.
    return Image.fromarray(np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8))


def _display_cell(rgb, cell_size, crop_box=None):
    source = _image(rgb)
    scale = min(1.0, cell_size[0] / source.width, cell_size[1] / source.height)
    fitted_size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
    fitted = source.resize(fitted_size, Image.Resampling.LANCZOS) if fitted_size != source.size else source
    canvas = Image.new("RGB", cell_size, "#edf1f6")
    left, top = (cell_size[0] - fitted.width) // 2, (cell_size[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    draw = ImageDraw.Draw(canvas)
    if crop_box is not None:
        x, y, width, height = crop_box
        # Independent realized x/y ratios account for rounded thumbnail sizes.
        sx, sy = fitted.width / source.width, fitted.height / source.height
        rectangle = (
            left + x * sx, top + y * sy,
            left + (x + width) * sx - 1, top + (y + height) * sy - 1,
        )
        draw.rectangle(rectangle, outline="#ff8b32", width=2)
    draw.rectangle((0, 0, cell_size[0] - 1, cell_size[1] - 1), outline="#d4dde7", width=1)
    return canvas


def build_audit_preview(report, output, seed=42):
    """Create five four-pair sheets from 20 fixed, stratified source pairs.

    Five pairs per camera/official-split, with LR96/HR192 inspection crops.
    This is data QA, not model selection or quantitative evaluation.
    """
    strata = [(camera, split) for camera in ("Canon", "Nikon") for split in ("train", "test")]
    selected_by_group = {}
    for camera, split in strata:
        pairs = sorted(
            (p for p in report["pairs"] if p["camera"] == camera and p["official_split"] == split),
            key=lambda pair: pair["pair_id"],
        )
        chosen = _rng(seed, f"selection:{camera}:{split}").choice(len(pairs), size=5, replace=False)
        selected_by_group[(camera, split)] = [pairs[int(index)] for index in chosen]
    # Each page contains one pair from every stratum, making camera/split
    # coverage easy to inspect without jumping between contact sheets.
    selected = [selected_by_group[stratum][index] for index in range(5) for stratum in strata]

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, margin, gutter = 224, 224, 24, 16
    row_header, row_gap, page_header, footer = 48, 16, 100, 42
    page_width = 2 * margin + 4 * cell_w + 3 * gutter
    page_height = page_header + 4 * (row_header + cell_h + row_gap) + footer
    font = ImageFont.load_default(size=14)
    small_font = ImageFont.load_default(size=12)
    title_font = ImageFont.load_default(size=21)
    sample_records, sheets = [], []
    headings = ["Whole LR: Bicubic x2", "Whole HR (ground truth)", "LR crop: Bicubic x2", "HR crop (ground truth)"]
    for page_index in range(5):
        page = Image.new("RGB", (page_width, page_height), "#ffffff")
        draw = ImageDraw.Draw(page)
        draw.text((margin, 15), f"RealSR V3 x2: source-pair audit  |  sheet {page_index + 1}/5", font=title_font, fill="#182b42")
        draw.text((margin, 45), "Display only. Orange boxes show corresponding inspection crops. No metrics are computed from these PNGs.", font=small_font, fill="#51627a")
        for column, heading in enumerate(headings):
            draw.text((margin + column * (cell_w + gutter), 76), heading, font=font, fill="#243951")
        filename = f"contact_{page_index + 1:02d}.png"
        for row_index, pair in enumerate(selected[4 * page_index:4 * (page_index + 1)]):
            lr, hr = load_pair(report["root"], pair)
            lr_crop, hr_crop, crop = paired_crop(lr, hr, 96, _rng(seed, f"crop:{pair['pair_id']}"))
            full_bicubic = bicubic_x2(lr)
            crop_bicubic = bicubic_x2(lr_crop)
            crop_lr = [crop["x"], crop["y"], 96, 96]
            crop_hr = [crop["hr_x"], crop["hr_y"], 192, 192]
            top = page_header + row_index * (row_header + cell_h + row_gap)
            lr_w, lr_h = pair["lr_size"]
            hr_w, hr_h = pair["hr_size"]
            label = (f"{pair['pair_id']}  |  official={pair['official_split']}  assigned={pair['split']}  "
                     f"|  LR {lr_w}x{lr_h}  ->  HR {hr_w}x{hr_h}")
            coordinates = (f"Crop LR (x,y,w,h)=({crop['x']},{crop['y']},96,96)  "
                           f"->  HR ({crop['hr_x']},{crop['hr_y']},192,192)")
            draw.text((margin, top + 2), label, font=font, fill="#182b42")
            draw.text((margin, top + 23), coordinates, font=small_font, fill="#51627a")
            for column, (pixels, box) in enumerate(((full_bicubic, crop_hr), (hr, crop_hr), (crop_bicubic, None), (hr_crop, None))):
                page.paste(_display_cell(pixels, (cell_w, cell_h), box),
                           (margin + column * (cell_w + gutter), top + row_header))
            sample_records.append({
                "sample_index": len(sample_records) + 1, "pair_id": pair["pair_id"],
                "camera": pair["camera"], "official_split": pair["official_split"], "split": pair["split"],
                "lr_path": pair["lr_path"], "hr_path": pair["hr_path"],
                "lr_size": pair["lr_size"], "hr_size": pair["hr_size"],
                "crop_lr_xywh": crop_lr, "crop_hr_xywh": crop_hr,
                "sheet": filename, "row": row_index + 1,
            })
        draw.text((margin, page_height - 29),
                  "Whole images fit the display cells; 192x192 patches are shown at native pixels. Original files are unchanged.",
                  font=small_font, fill="#51627a")
        page.save(output / filename)
        sheets.append(filename)
        print(f"Data audit preview: saved {filename} ({4 * (page_index + 1)}/20 pairs)", flush=True)
    result = {
        "purpose": "Visual source-pair QA only; no model, training, or quantitative metric evaluation",
        "seed": seed, "sample_count": 20, "unique_pair_count": len({record["pair_id"] for record in sample_records}),
        "selection": "5 pairs sampled without replacement from each camera x official split, independently seeded",
        "strata_counts": dict(Counter(f"{r['camera']}/{r['official_split']}" for r in sample_records)),
        "source_manifest_pair_count": len(report["pairs"]), "source_root": str(report["root"]),
        "data_source": report.get("provenance", {}),
        "display": {
            "cell_size": [cell_w, cell_h], "sheet_size": [page_width, page_height],
            "columns": headings, "whole_images": "aspect preserved; downscaled only with Pillow Lanczos",
            "patches": "LR96 Bicubic x2 and corresponding HR192, shown at 1:1 within padded cells",
            "crop_boxes": "same HR-coordinate rectangle overlaid only on whole-image previews",
            "bicubic": "torch.interpolate scale_factor=2, mode=bicubic, align_corners=False, antialias=False; clipped [0,1]",
            "png": "float RGB rounded to uint8 for display only; never used as metric input",
        },
        "sheets": sheets, "samples": sample_records,
        "inspection_limit": "A 20-pair visual sample cannot certify perfect pixelwise registration for the complete dataset",
    }
    (output / "audit_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
