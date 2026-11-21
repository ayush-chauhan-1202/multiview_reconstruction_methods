"""Export cached VGGT camera and depth predictions as a Brush COLMAP dataset."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation


def training_size(width, height):
    return tuple(round(length * 768 / max(width, height)) for length in (width, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("colmap_images", type=Path, help="Use these registered image filenames")
    parser.add_argument("output", type=Path)
    parser.add_argument("--original-images", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.predictions) as archive:
        names = archive["image_names"]
        selected = [index for index, name in enumerate(names) if (args.colmap_images / name).is_file()]
        if len(selected) != len(list(args.colmap_images.iterdir())):
            parser.error("The registered image set must be covered by the VGGT predictions.")
        depths = archive["depth"][..., 0]
        confidence = archive["confidence"]
        masks = archive["masks"].astype(bool)
        colors = archive["colors"]
        intrinsics = archive["intrinsics"]
        poses = archive["extrinsics"]
        image_meta = archive["image_meta"]

    valid = masks & np.isfinite(depths) & (depths > 0) & np.isfinite(confidence)
    threshold = np.percentile(confidence[selected][valid[selected]], 25)
    cameras, frames, points = [], [], []
    for image_id, index in enumerate(selected, 1):
        calibration = intrinsics[index]
        rotation = poses[index, :, :3]
        translation = poses[index, :, 3]
        x, y, z, w = Rotation.from_matrix(rotation).as_quat()
        name = names[index]
        fx, fy = calibration[0, 0], calibration[1, 1]
        cx, cy = calibration[0, 2], calibration[1, 2]
        original_width, original_height, width, height, left, top = image_meta[index]
        output_width, output_height = training_size(original_width, original_height)
        scale_x, scale_y = output_width / width, output_height / height
        cameras.append(
            f"{image_id} PINHOLE {output_width} {output_height} "
            f"{fx * scale_x} {fy * scale_y} {(cx - left) * scale_x} {(cy - top) * scale_y}"
        )
        offset = " ".join(str(value) for value in translation)
        frames.append(f"{image_id} {w} {x} {y} {z} {offset} {image_id} {name}\n")
        sample = valid[index] & (confidence[index] >= threshold)
        row, column = np.nonzero(sample[::8, ::8])
        row, column = row * 8, column * 8
        pixels = np.stack((column, row, np.ones_like(row)), axis=1)
        camera_points = (pixels @ np.linalg.inv(calibration).T) * depths[index, row, column, None]
        world_points = (camera_points - translation) @ rotation
        points.append(np.column_stack((world_points, colors[index, row, column])))

    sparse = args.output / "sparse" / "0"
    sparse.mkdir(parents=True)
    image_dir = args.output / "images"
    image_dir.mkdir()
    for index in selected:
        with Image.open(args.original_images / names[index]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        expected_size = tuple(image_meta[index, :2])
        if image.size != expected_size:
            raise ValueError(f"Original dimensions changed for {names[index]}")
        resized = image.resize(training_size(*image.size), Image.Resampling.LANCZOS)
        resized.save(image_dir / names[index], quality=95)
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(frames) + "\n")
    seeds = np.concatenate(points)
    with (sparse / "points3D.txt").open("w") as output:
        for point_id, (x, y, z, r, g, b) in enumerate(seeds, 1):
            output.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0\n")
    print(f"Exported {len(selected)} cameras and {len(seeds)} seeds to {args.output}")


if __name__ == "__main__":
    main()
