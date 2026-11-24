"""Predict posed depth for images registered by prepare_capture.py."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TARGET_SIZE = 518


def load_images(paths, mask_dir, torch):
    """EXIF-correct, aspect-preserving resize/pad matching VGGT's pad mode."""
    tensors, colors, valid_rects, object_masks, image_meta = [], [], [], [], []
    for path in paths:
        image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        width, height = image.size
        if width >= height:
            new_width = TARGET_SIZE
            new_height = max(14, round(height * new_width / width / 14) * 14)
        else:
            new_height = TARGET_SIZE
            new_width = max(14, round(width * new_height / height / 14) * 14)
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        left = (TARGET_SIZE - new_width) // 2
        top = (TARGET_SIZE - new_height) // 2
        canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        canvas.paste(image, (left, top))
        array = np.asarray(canvas, dtype=np.uint8)
        colors.append(array)
        tensors.append(torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0)
        valid_rects.append((left, top, left + new_width, top + new_height))
        image_meta.append((width, height, new_width, new_height, left, top))
        if mask_dir is not None:
            mask_path = mask_dir / f"{path.stem}_mask.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing foreground mask: {mask_path}")
            mask = Image.open(mask_path).convert("L")
            if mask.size != (width, height):
                raise RuntimeError(f"Mask/image size mismatch for {path.name}: {mask.size} vs {(width, height)}")
            mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
            mask_canvas = Image.new("L", (TARGET_SIZE, TARGET_SIZE), 0)
            mask_canvas.paste(mask, (left, top))
            object_masks.append(np.asarray(mask_canvas) >= 128)
    return (torch.stack(tensors), np.stack(colors), valid_rects,
            np.stack(object_masks) if object_masks else None, np.asarray(image_meta, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    root = Path(__file__).resolve().parents[1]
    records = json.loads(args.manifest.read_text())["images"]
    paths = [root / item["path"] for item in records]
    if args.masks and len({path.stem for path in paths}) != len(paths):
        parser.error("Give each masked input a unique filename stem.")
    images, colors, rects, masks, metadata = load_images(paths, args.masks, torch)
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.checkpoint:
        model = VGGT()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.eval().to(device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    autocast = torch.autocast(device_type=device, dtype=dtype) if device != "cpu" else nullcontext()
    with torch.inference_mode(), autocast:
        prediction = model(images.to(device))
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        prediction["pose_enc"].float().cpu(), images.shape[-2:]
    )
    depth = prediction["depth"][0].float().cpu().numpy()
    confidence = prediction["depth_conf"][0].float().cpu().numpy()
    valid = np.zeros(confidence.shape, dtype=bool)
    for index, (left, top, right, bottom) in enumerate(rects):
        valid[index, top:bottom, left:right] = True
    if masks is not None:
        valid &= masks
    with args.output.open("xb") as output:
        np.savez_compressed(
            output, depth=depth, confidence=confidence, extrinsics=extrinsics[0].numpy(),
            intrinsics=intrinsics[0].numpy(), masks=valid, colors=colors,
            image_meta=metadata, image_names=np.array([path.name for path in paths]),
        )
    print(f"Saved {len(paths)} posed depth maps to {args.output}")


if __name__ == "__main__":
    main()
