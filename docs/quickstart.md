# Reconstruct an object from your own photos

The Sipper figures show the original comparison. Start here with your own capture to produce the same four types of GLB on your machine. The Sipper source capture, model weights, and reconstruction outputs live outside the repository contents. The TSDF quickstart uses fused vertex colors for its GLB. The Sipper preview uses a separate photo texture bake.

## Install the tools

Use Python 3.11 and run the commands from the repository root in a Bash-compatible shell. Install COLMAP 4.2, OpenMVS with its command-line programs on your PATH, and the Brush 0.3.0 CLI for your platform. The commands below call the Brush executable `brush_app`. Adjust that name to match your installation. COLMAP sparse reconstruction can use the CPU. Brush needs a compatible graphics adapter. [COLMAP install](https://colmap.github.io/install.html) · [OpenMVS releases](https://github.com/cdcseacave/openMVS/releases) · [Brush 0.3.0](https://github.com/ArthurBrussee/brush/releases/tag/v0.3.0)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the VGGT paths, also install VGGT and its PyTorch dependencies into this environment using the upstream instructions. Check that the installed PyTorch supports the device you want to use. Inference can use CUDA, Apple MPS, or CPU. The first run downloads VGGT-1B weights, or you can pass `--checkpoint /path/to/model.pt` with the original VGGT state dictionary. [VGGT installation and weights](https://github.com/facebookresearch/vggt#quick-start)

## Prepare the capture

Put one ordered sequence of overlapping JPG, JPEG, or PNG photos in `data/images/`. For a first run, use a few dozen views taken around a stationary object at different elevations. Give the files unique, sequential names such as `frame_001.jpg`. Fill the frame with the object and keep lighting steady. For an object-only VGGT result, put optional grayscale masks in a local folder outside the checkout, one per image named `<image-stem>_mask.png`. Each mask must match the EXIF-upright image dimensions, with white for foreground and black for background.

```sh
python prepare_capture.py my-object
run=results/my-object
python -c 'import json,sys; from pathlib import Path; print(*(Path(i["path"]).name for i in json.load(open(sys.argv[1]))["images"]), sep="\n")' \
  "$run/capture.json" > "$run/views.txt"
```

Keep the same input files for all methods. Use a new capture name for a different image set. Commands that create a database or model assume a fresh output directory.

## 1. COLMAP + OpenMVS

Estimate cameras and sparse geometry, then undistort the registered photos. Use the sparse model that covers the object if COLMAP creates several. The example assumes model `0`. Check registration coverage with `colmap model_analyzer --path "$run/sparse/0"`. [COLMAP tutorial](https://colmap.github.io/tutorial)

```sh
mkdir -p "$run/sparse"
colmap feature_extractor --database_path "$run/colmap.db" \
  --image_path data/images --image_list_path "$run/views.txt" \
  --FeatureExtraction.use_gpu 0
colmap sequential_matcher --database_path "$run/colmap.db" --FeatureMatching.use_gpu 0
colmap mapper --database_path "$run/colmap.db" --image_path data/images \
  --output_path "$run/sparse"
colmap image_undistorter --image_path data/images --input_path "$run/sparse/0" \
  --output_path "$run/dense" --output_type COLMAP
```

Convert the calibrated images and sparse cloud into an OpenMVS scene, then densify, mesh, and texture. This starter workflow processes the full visible scene. Use an object crop in the mesh’s coordinate frame if the table or background is reconstructed. [OpenMVS workflow](https://github.com/cdcseacave/openMVS/wiki/Usage)

```sh
(
  cd "$run/dense"
  InterfaceCOLMAP -i . --image-folder images -o scene.mvs
  DensifyPointCloud -i scene.mvs -o scene_dense.mvs
  ReconstructMesh -i scene_dense.mvs --pointcloud-file scene_dense.ply -o scene_mesh.mvs
  TextureMesh -i scene_dense.mvs --mesh-file scene_mesh.ply -o scene_textured.mvs --export-type obj
)
python -c 'import sys,trimesh; trimesh.load_scene(sys.argv[1],process=False).export(sys.argv[2])' \
  "$run/dense/scene_textured.obj" "$run/colmap-openmvs/mesh.glb"
```

If an opaque object appears transparent, check the OpenMVS OBJ’s adjacent MTL file. Some versions emit `Tr 1.0` for opaque exports. For an opaque object, replace that entry with `d 1.0` and repeat the GLB export.

## 2. VGGT + TSDF

Predict posed depth on the same ordered capture, then fuse the depth maps into a colored surface. Add `--masks /path/to/masks` to the prediction command for object-only depth fusion. If inference exceeds device memory, use a smaller capture with views distributed around the object.

```sh
python scripts/predict_vggt.py "$run/capture.json" "$run/vggt-depth.npz"
python scripts/fuse_tsdf.py "$run/vggt-depth.npz" "$run/vggt-tsdf/mesh.glb"
```

## 3. COLMAP + 3DGS

Train Brush from the undistorted photos, cameras, and sparse points produced in step 1. Then extract the opacity surface. The bounds mesh defines the crop in COLMAP coordinates.

```sh
brush_app "$run/dense" --total-steps 3000 --max-resolution 768 --max-splats 250000 \
  --export-every 3000 --export-path "$run/colmap-3dgs" --export-name splats.ply
python extract_splat_mesh.py "$run/colmap-3dgs/splats.ply" "$run/colmap-3dgs/mesh.glb" \
  --bounds-mesh "$run/colmap-openmvs/mesh.glb" --level 0.35
```

## 4. VGGT + 3DGS

Use the cached VGGT predictions to export camera poses and depth-derived seeds in COLMAP format. The exporter uses the registered filenames from step 1 so both Gaussian runs train on the same source views. It reads the original photos and adjusts intrinsics for its unpadded resize. The bounds mesh for extraction is in VGGT coordinates.

```sh
python prepare_vggt_splats.py "$run/vggt-depth.npz" "$run/dense/images" \
  "$run/vggt-3dgs/dataset" --original-images data/images
brush_app "$run/vggt-3dgs/dataset" --total-steps 3000 --max-resolution 768 --max-splats 250000 \
  --export-every 3000 --export-path "$run/vggt-3dgs" --export-name splats.ply
python extract_splat_mesh.py "$run/vggt-3dgs/splats.ply" "$run/vggt-3dgs/mesh.glb" \
  --bounds-mesh "$run/vggt-tsdf/mesh.glb" --level 0.50
```

## Compare the meshes

Open the four GLBs in a glTF viewer or Blender. Inspect the silhouette, holes, surface roughness, and any attached background from more than one angle. The meshes use their native coordinate frames and can have different scales. Align them before measuring distances. The Gaussian opacity cutoffs above reproduce the selected Sipper settings and serve as starting values for a new object. [Khronos glTF viewer](https://github.khronos.org/glTF-Sample-Viewer-Release/)
