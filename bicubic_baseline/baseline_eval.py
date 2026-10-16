"""RealSR x2 Bicubic baseline with RGB PSNR and SSIM .

Example:
    python baseline_eval.py --lr-dir data/test/LR --hr-dir data/test/HR \
        --save-dir results/bicubic
""" 

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Bicubic x2 with RGB PSNR and SSIM')
    parser.add_argument('--lr-dir', type=Path, required=True, help='directory containing LR test images')
    parser.add_argument('--hr-dir', type=Path, required=True, help='directory containing paired HR test images')
    parser.add_argument('--save-dir', type=Path, default=Path('results/bicubic'))
    parser.add_argument('--save-images', action='store_true', help='save every upsampled image')
    return parser.parse_args()

def image_files(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f'No images found under: {folder}')
    return files

def pair_key(path: Path, root: Path) -> str:
    key = path.stem.lower()
    key = re.sub('(^|[_-])(lr|hr|gt|x2|2x)(?=$|[_-])', '_', key)
    key = re.sub('[_-](lr|hr|gt)x?2$', '', key)
    key = re.sub('[_-]+', '_', key).strip('_')
    parent = path.relative_to(root).parent.as_posix().lower()
    return f'{parent}/{key}'

def make_pairs(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path]]:
    lr_files, hr_files = image_files(lr_dir), image_files(hr_dir)
    lr_map = {pair_key(p, lr_dir): p for p in lr_files}
    hr_map = {pair_key(p, hr_dir): p for p in hr_files}
    if len(lr_map) != len(lr_files) or len(hr_map) != len(hr_files):
        raise ValueError('Duplicate normalized filenames detected; rename files to make pair names unique.')
    common = sorted(lr_map.keys() & hr_map.keys())
    missing_hr = sorted(lr_map.keys() - hr_map.keys())
    missing_lr = sorted(hr_map.keys() - lr_map.keys())
    if missing_hr or missing_lr:
        raise ValueError(f'Unpaired files: {len(missing_hr)} LR without HR, {len(missing_lr)} HR without LR. Examples: LR={missing_hr[:3]}, HR={missing_lr[:3]}')
    if not common:
        raise ValueError('No LR/HR pairs found. Matching ignores common _LR/_HR/_GT/_x2 markers.')
    return [(lr_map[k], hr_map[k]) for k in common]

def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float64) - prediction.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(255 ** 2 / mse))

def metrics(hr_rgb: np.ndarray, sr_rgb: np.ndarray) -> tuple[float, float]:
    ssim = structural_similarity(hr_rgb, sr_rgb, data_range=255.0, channel_axis=2)
    return psnr(hr_rgb, sr_rgb), float(ssim)

def main() -> None:
    args = parse_args()
    pairs = make_pairs(args.lr_dir, args.hr_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    output_dir = args.save_dir / 'bicubic'
    scores = []
    for lr_path, hr_path in pairs:
        lr, hr = read_rgb(lr_path), read_rgb(hr_path)
        expected = (lr.shape[0] * 2, lr.shape[1] * 2)
        if hr.shape[:2] != expected:
            raise ValueError(f'Scale mismatch for {lr_path.name}: LR={lr.shape[:2]}, HR={hr.shape[:2]}, expected HR={expected}. Check that this is the x2 subset.')
        sr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        score_psnr, score_ssim = metrics(hr, sr)
        scores.append((score_psnr, score_ssim))
        rows.append(['bicubic', lr_path.name, f'{score_psnr:.6f}', f'{score_ssim:.6f}'])
        if args.save_images:
            bgr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
            destination = output_dir / lr_path.relative_to(args.lr_dir).parent
            destination.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode('.png', bgr)
            if not ok:
                raise ValueError(f'Cannot encode image: {lr_path}')
            encoded.tofile(destination / f'{lr_path.stem}_bicubic.png')
    mean_psnr, mean_ssim = np.mean(scores, axis=0)
    rows.append(['bicubic', 'AVERAGE', f'{mean_psnr:.6f}', f'{mean_ssim:.6f}'])
    print(f'Bicubic x2 | N={len(pairs)} | RGB PSNR={mean_psnr:.4f} dB | SSIM={mean_ssim:.4f}')
    csv_path = args.save_dir / 'metrics_rgb_x2.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['method', 'image', 'psnr_db', 'ssim'])
        writer.writerows(rows)
    print(f'Per-image scores saved to {csv_path}')


if __name__ == '__main__':
    main()


