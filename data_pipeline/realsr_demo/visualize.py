"""Display-only image conversion; quantitative metrics use original float arrays."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def _font(size=18):
    path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default(size=size)


def save_comparison(output, pair, lr, bicubic, sr, hr, metrics):
    name = pair["pair_id"].replace(":", "_")
    folder = Path(output) / "examples" / name
    folder.mkdir(parents=True, exist_ok=True)
    images = [_image(lr).resize((hr.shape[1], hr.shape[0]), Image.Resampling.NEAREST),
              _image(bicubic), _image(sr), _image(hr)]
    labels = ["LR (nearest display x2)", "Bicubic x2", "TinySR x2", "Ground truth HR"]
    # Center crop fixed independent of metrics. Coordinates are in HR pixels.
    size = min(256, *hr.shape[:2])
    x, y = (hr.shape[1]-size)//2, (hr.shape[0]-size)//2
    col, gap, pad = 360, 14, 22
    full_height = min(300, round(col * hr.shape[0] / hr.shape[1]))
    canvas = Image.new("RGB", (pad*2 + col*4 + gap*3, full_height+col+142), "#eef1f6")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 14), f"{pair['pair_id']}  |  LR {lr.shape[1]}x{lr.shape[0]} -> HR {hr.shape[1]}x{hr.shape[0]}", fill="#172536", font=_font(22))
    draw.text((pad, 43), "Top: whole image fitted for display. Bottom: same center ROI. Metrics use the full original images.", fill="#536477", font=_font(17))
    for index, (image, label) in enumerate(zip(images, labels)):
        left = pad + index * (col+gap)
        draw.text((left, 75), label, fill="#172536", font=_font(18))
        small = image.copy()
        small.thumbnail((col, full_height), Image.Resampling.LANCZOS)
        offset_x, offset_y = left+(col-small.width)//2, 104+(full_height-small.height)//2
        canvas.paste(small, (offset_x, offset_y))
        scale = small.width / image.width
        draw.rectangle((offset_x+x*scale, offset_y+y*scale, offset_x+(x+size)*scale, offset_y+(y+size)*scale), outline="#ff8a3d", width=2)
        roi = image.crop((x, y, x+size, y+size)).resize((col, col), Image.Resampling.NEAREST)
        canvas.paste(roi, (left, 118+full_height))
        image.save(folder / ["lr_nearest.png", "bicubic.png", "sr.png", "hr.png"][index])
    canvas.save(folder / "comparison.jpg", quality=95, subsampling=0)
    return {"pair_id": pair["pair_id"], "comparison": (folder/"comparison.jpg").relative_to(output).as_posix(),
            "folder": folder.relative_to(output).as_posix(), "hr_roi_xywh": [x, y, size, size], "metrics": metrics}


def save_loss_curve(path, rows):
    values = [float(r[key]) for r in rows for key in ("train_l1", "val_l1")]
    high = max(values) * 1.1
    high = max(high, 1e-8)
    x_of = lambda index: 85 + 620 * index / max(1, len(rows) - 1)
    y_of = lambda value: 270 - 205 * value / high
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="365" viewBox="0 0 800 365">',
        '<rect width="800" height="365" fill="#f8fafc"/>',
        '<g font-family="sans-serif" fill="#243047">',
        '<text x="55" y="30" font-size="18">No-augmentation baseline: epoch L1 loss</text>',
        '<text x="55" y="52" font-size="12">Validation = full images; L1 checkpoint selection, not test metrics</text>',
    ]
    for tick in range(5):
        value = high * tick / 4
        y = y_of(value)
        pieces.append(f'<line x1="85" x2="705" y1="{y:.2f}" y2="{y:.2f}" stroke="#dfe5ed"/>')
        pieces.append(f'<text x="75" y="{y+4:.2f}" text-anchor="end" font-size="11">{value:.5f}</text>')
    for key, colour in (("train_l1", "#315bd9"), ("val_l1", "#d36624")):
        points = " ".join(f"{x_of(i):.2f},{y_of(row[key]):.2f}" for i, row in enumerate(rows))
        pieces.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for index, row in enumerate(rows):
            pieces.append(f'<circle cx="{x_of(index):.2f}" cy="{y_of(row[key]):.2f}" r="3" fill="{colour}"/>')
    for index, row in enumerate(rows):
        pieces.append(f'<text x="{x_of(index):.2f}" y="290" text-anchor="middle" font-size="11">{row["epoch"]}</text>')
    pieces.extend([
        '<text x="390" y="313" text-anchor="middle" font-size="12">Epoch</text>',
        '<text x="90" y="340" fill="#315bd9" font-size="12">Train L1</text>',
        '<text x="230" y="340" fill="#d36624" font-size="12">Validation L1</text>',
        '</g></svg>',
    ])
    path.write_text("".join(pieces), encoding="utf-8")
