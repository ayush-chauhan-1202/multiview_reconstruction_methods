"""Fuse posed VGGT depth maps into a vertex-colored GLB."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=192)
    args = parser.parse_args()
    if args.resolution < 16:
        parser.error("Resolution must be at least 16.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    data = np.load(args.predictions)
    depths = data["depth"].astype(np.float32)
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    confidences = data["confidence"].astype(np.float32)
    intrinsics = data["intrinsics"].astype(np.float64)
    extrinsics = data["extrinsics"].astype(np.float64)
    masks = data["masks"].astype(bool)
    colors = data["colors"].astype(np.uint8)

    valid_conf = masks & np.isfinite(confidences) & (confidences > 0)
    if not valid_conf.any():
        raise ValueError("No valid depth confidence samples remain.")
    threshold = float(np.percentile(confidences[valid_conf], 25))
    valid = valid_conf & (confidences >= threshold) & np.isfinite(depths) & (depths > 0)
    depth_values = depths[valid]
    if not len(depth_values):
        raise RuntimeError("No valid masked depth samples remain")
    depth_trunc = float(np.percentile(depth_values, 99.9) * 1.02)

    object_points = []
    for index, (k, pose) in enumerate(zip(intrinsics, extrinsics)):
        rows, cols = np.nonzero(valid[index, ::4, ::4])
        rows, cols = rows * 4, cols * 4
        pixels = np.column_stack((cols, rows, np.ones(len(rows))))
        xyz = (pixels @ np.linalg.inv(k).T) * depths[index, rows, cols, None]
        object_points.append((xyz - pose[:, 3]) @ pose[:, :3])
    object_points = np.concatenate(object_points)
    if len(object_points) < 2:
        raise ValueError("Too few valid samples to determine fusion bounds.")
    bounds_min = np.percentile(object_points, 0.5, axis=0)
    bounds_max = np.percentile(object_points, 99.5, axis=0)
    extent = bounds_max - bounds_min
    padding = extent * 0.08
    bounds_min -= padding
    bounds_max += padding
    volume_length = float(np.max(bounds_max - bounds_min))
    center = (bounds_min + bounds_max) * 0.5
    origin = center - volume_length * 0.5
    voxel_length = volume_length / args.resolution
    sdf_trunc = voxel_length * 4.0

    print(f"Views: {len(depths)}; valid masked samples: {int(valid.sum()):,}", flush=True)
    print(f"Confidence threshold: {threshold:.4g}; depth truncation: {depth_trunc:.4g}", flush=True)
    print(f"Uniform TSDF grid: {args.resolution}^3; voxel={voxel_length:.6f} VGGT units", flush=True)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=volume_length,
        resolution=args.resolution,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=origin,
    )

    height, width = depths.shape[1:3]
    for i in range(len(depths)):
        depth = np.where(valid[i], depths[i], 0).astype(np.float32)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(colors[i]),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        k = intrinsics[i]
        camera = o3d.camera.PinholeCameraIntrinsic(
            width, height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        )
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = extrinsics[i]
        volume.integrate(rgbd, camera, world_to_camera)
        if (i + 1) % 10 == 0 or i + 1 == len(depths):
            print(f"Integrated {i + 1}/{len(depths)} views", flush=True)

    mesh = volume.extract_triangle_mesh()
    print(f"Raw extraction: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles", flush=True)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not mesh.has_triangles():
        raise RuntimeError("TSDF integration produced no triangles.")
    colors = np.rint(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    exported = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles),
                               vertex_colors=colors, process=False)
    with args.output.open("xb") as output:
        output.write(exported.export(file_type="glb"))
    print(f"Mesh: {args.output}")


if __name__ == "__main__":
    main()
