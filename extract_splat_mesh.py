"""Extract a vertex-colored triangle GLB from a Gaussian PLY opacity field."""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation
from scipy.special import expit
from skimage.measure import marching_cubes
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splats", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bounds-mesh", type=Path, required=True,
                        help="Mesh in the splat coordinate frame, used only to define the crop")
    parser.add_argument("--resolution", type=int, default=128, help="Samples on the longest axis")
    parser.add_argument("--level", type=float, default=0.35, help="Accumulated opacity isovalue")
    args = parser.parse_args()
    if args.resolution < 16 or not 0.01 < args.level < 1:
        parser.error("Resolution must be at least 16 and opacity must be between 0.01 and 1.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    data = PlyData.read(args.splats)["vertex"].data
    means = np.column_stack([data[axis] for axis in "xyz"])
    opacity = expit(data["opacity"])
    scales = np.exp(np.column_stack([data[f"scale_{axis}"] for axis in range(3)]))
    # Gaussian PLY stores the quaternion scalar first. SciPy expects it last.
    quaternions = np.column_stack([data[f"rot_{axis}"] for axis in (1, 2, 3, 0)])
    colors = np.clip(0.5 + 0.2820947918 * np.column_stack(
        [data[f"f_dc_{axis}"] for axis in range(3)]), 0, 1)
    bounds = trimesh.load_scene(args.bounds_mesh).bounds
    extent = bounds[1] - bounds[0]
    spacing = 1.2 * extent.max() / (args.resolution - 1)
    low = bounds[0] - 0.1 * extent - spacing
    high = bounds[1] + 0.1 * extent + spacing
    shape = tuple(np.ceil((high - low) / spacing).astype(int) + 1)

    # Work in optical thickness so independent opacities compose by addition.
    density = np.zeros(shape, dtype=np.float32)
    color_sum = np.zeros((*shape, 3), dtype=np.float32)
    keep = (opacity >= 0.01) & np.isfinite(scales).all(axis=1) & (scales > 0).all(axis=1)
    rotations = Rotation.from_quat(quaternions[keep]).as_matrix()
    for mean, scale, alpha, rotation, rgb in zip(
        means[keep], scales[keep], opacity[keep], rotations, colors[keep]
    ):
        radius = 3 * np.sqrt((rotation**2) @ (scale**2))
        start = np.maximum(0, np.floor((mean - radius - low) / spacing).astype(int))
        stop = np.minimum(shape, np.ceil((mean + radius - low) / spacing).astype(int) + 1)
        if np.any(stop <= start):
            continue
        x, y, z = np.ogrid[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
        offsets = [(axis * spacing + origin - center)
                   for axis, origin, center in zip((x, y, z), low, mean)]
        distance_squared = sum(
            (sum(offset * weight for offset, weight in zip(offsets, direction)) / width)**2
            for direction, width in zip(rotation.T, scale)
        )
        contribution = alpha * np.exp(-0.5 * distance_squared)
        contribution[distance_squared > 9] = 0
        weight = -np.log1p(-np.minimum(contribution, 0.999))
        region = tuple(slice(a, b) for a, b in zip(start, stop))
        density[region] += weight
        color_sum[region] += weight[..., None] * rgb

    field = -np.expm1(-density)
    # Seal the sampling volume at its crop boundary for a bounded isosurface.
    for axis in range(3):
        edge = np.moveaxis(field, axis, 0)
        edge[0] = edge[-1] = 0
    vertices, faces, _, _ = marching_cubes(field, level=args.level, allow_degenerate=False)
    weights = map_coordinates(density, vertices.T, order=1)
    rgb = np.column_stack([
        map_coordinates(color_sum[..., axis], vertices.T, order=1) / np.maximum(weights, 1e-8)
        for axis in range(3)
    ])
    mesh = trimesh.Trimesh(vertices=low + vertices * spacing, faces=faces,
                           vertex_colors=np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                           process=False)
    mesh.fix_normals(multibody=True)
    args.output.write_bytes(mesh.export(file_type="glb"))
    print(f"Exported {len(mesh.faces):,} triangles to {args.output}; voxel spacing {spacing:.6g}")


if __name__ == "__main__":
    main()
