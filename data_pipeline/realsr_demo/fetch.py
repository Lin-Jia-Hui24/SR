"""下载官方 RealSR V3，保留完整 ×2 图像；图像检查交给 scan。"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

OFFICIAL_URL = "https://github.com/csjcai/RealSR"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM&export=download&confirm=t"
)
SELECTED = re.compile(
    r"RealSR\(V3\)/(?P<camera>Canon|Nikon)/(?P<split>Train|Test)/2/"
    r"(?P=camera)_(?P<scene>[0-9]+)_(?P<role>HR|LR2)\.png"
)


def _download_archive(archive):
    if archive.exists():
        print(f"Using cached archive: {archive}", flush=True)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_name(archive.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"}
    with urlopen(Request(ARCHIVE_URL, headers=headers), timeout=60) as response:
        if response.status == 206:
            span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not span or int(span[1]) != offset:
                raise ValueError("Server returned an incorrect resume offset")
            total, mode = int(span[3]), "ab"
        elif response.status == 200:
            # A server may ignore Range: replace the partial file, never append.
            offset, mode = 0, "wb"
            total = int(response.headers.get("Content-Length", 0))
        else:
            raise ValueError(f"Unexpected download status: {response.status}")
        progress = offset
        with partial.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                offset += len(chunk)
                if offset - progress >= 32 * 1024 * 1024:
                    print(f"Downloaded {offset / 1024**2:.0f} MiB", flush=True)
                    progress = offset
    if total and partial.stat().st_size != total:
        raise IOError("Download incomplete; rerun fetch to resume the .part file")
    partial.replace(archive)


def _extract_x2(archive, root):
    files, counts = [], {"Train": 0, "Test": 0}
    by_camera = {camera: {"Train": 0, "Test": 0} for camera in ("Canon", "Nikon")}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            match = SELECTED.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            camera, split, scene, role = (match[key] for key in ("camera", "split", "scene", "role"))
            # Build the path from restricted groups. Never extract archive paths.
            relative = Path("RealSR(V3)") / camera / split / "2" / f"{camera}_{scene}_{role}.png"
            destination = root / relative
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Extraction path leaves dataset directory: {destination}")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as image, tempfile.NamedTemporaryFile(
                    dir=destination.parent, prefix=".image-", delete=False
                ) as output:
                    shutil.copyfileobj(image, output)
                    temporary = Path(output.name)
                temporary.replace(destination)
            files.append({"path": relative.as_posix(), "bytes": member.size})
            if role == "LR2":
                counts[split] += 1
                by_camera[camera][split] += 1
            if len(files) % 100 == 0:
                print(f"Prepared {len(files)} x2 images", flush=True)
    return sorted(files, key=lambda item: item["path"]), counts, by_camera


def fetch_full(root, archive_path="data/RealSR_V3.tar.gz"):
    """Reuse a completed dataset, or download/extract once and save its source."""
    root, archive = Path(root).expanduser().resolve(), Path(archive_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "SOURCE.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("kind") == "full_x2" and source.get("complete") and source.get("files"):
            if all((root / item["path"]).is_file() for item in source["files"]):
                print(f"Using cached x2 dataset: {root}", flush=True)
                return source
    _download_archive(archive)
    print("Computing archive SHA256...", flush=True)
    with archive.open("rb") as handle:
        archive_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    files, counts, by_camera = _extract_x2(archive, root)
    source = {
        "kind": "full_x2", "dataset": "RealSR V3", "scale": 2,
        "official_url": OFFICIAL_URL, "download_url": ARCHIVE_URL,
        "archive_path": str(archive), "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size, "pair_counts": counts,
        "dataset_readme_nominal_scene_counts": {"Train": 459, "Test": 100},
        "actual_x2_counts": {**counts, "total_pairs": sum(counts.values()),
                             "png_files": len(files), "by_camera": by_camera},
        "file_count": len(files), "files": files, "complete": True,
        "note": (
            "The inspected official archive contains 406 x2 training pairs "
            "(Canon 200, Nikon 206) and 100 test pairs, unlike the README's "
            "nominal 459 training scenes. Counts here are measured from files. "
            "Only x2 images are extracted; scan checks decoding and pairing. "
            "Archive SHA256 is a local fingerprint, not an author-verified checksum."
        ),
    }
    temporary = source_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(source_path)
    print(f"Ready: {counts['Train']} train + {counts['Test']} test pairs", flush=True)
    return source
