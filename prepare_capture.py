"""Record the input files and prepare result directories for one capture."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METHODS = (
    "colmap-openmvs",
    "vggt-tsdf",
    "colmap-3dgs",
    "vggt-3dgs",
)


def image_order(path):
    parts = re.split(r"([0-9]+)", path.name.casefold())
    natural_order = tuple(int(part) if index % 2 else part for index, part in enumerate(parts))
    return natural_order, path.name


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Name for this capture's results folder")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run):
        parser.error("Run names can contain letters, digits, hyphens, and underscores.")

    images = sorted(
        (path for path in IMAGE_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=image_order,
    )
    if len(images) < 2:
        parser.error(f"Add at least two JPG, JPEG, or PNG files to {IMAGE_DIR}.")

    run_dir = ROOT / "results" / args.run
    manifest = {
        "images": [fingerprint(path) for path in images],
        "mesh_outputs": {method: f"{method}/mesh.glb" for method in METHODS},
    }
    try:
        run_dir.mkdir()
    except FileExistsError:
        parser.error(f"{run_dir} already exists. Choose a new run name.")
    for method in METHODS:
        (run_dir / method).mkdir()
    (run_dir / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(images)} images in {run_dir / 'capture.json'}")


if __name__ == "__main__":
    main()
