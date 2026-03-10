# Data Generation Toolkit

This directory provides configs and helper scripts for the offline C++ dataset
generator (`reachability_cli/dataset_generator_cli`). The generator runs without
`ros2 launch` and uses MoveIt Core for FK + self-collision.

Layout:
- configs/robots/<vendor>/<model>/: robot-specific dataset profiles
- scripts/runners/: local or server pipeline entry scripts
- scripts/visualize/: visualization helpers
- scripts/postprocess/: post-processing helpers
- scripts/wrappers/: thin compatibility wrappers
- outputs/<vendor>/<model>/<profile>/: generated dataset files (ignored)

Quick start:
1) Build and source:
   - `colcon build --packages-select reachability_cli`
   - `source install/setup.bash`
2) Run generator:
   - `ros2 run reachability_cli dataset_generator_cli --config tools/data_gen/configs/robots/aubo/aubo_i5/position_3mm.yaml`
   - Use `--output` to override `output.dir` in the YAML.
3) Orientation sampling (stage 2):
   - `ros2 run reachability_cli orientation_dataset_cli --config tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml`

Config notes:
- `sampling.coverage_mode`: `heat` (new voxels per interval, EMA) or `ratio` (ratio vs last interval).
- `sampling.coverage_heat_min`/`coverage_heat_alpha`: per-attempt new-voxel rate threshold (0-1) and EMA smoothing for `coverage_mode=heat`.
- `sampling.coverage_*`: early stop if occupied voxel growth converges during FK sampling.
- `sampling.fk_log_interval`: FK progress log interval in attempts.
- `sampling.threads`: OpenMP thread count (0 uses max, requires OpenMP-enabled build).
- `sampling.orientation_bins`: number of SO(3) bins for per-voxel orientation coverage (0 disables).
- `sdf.hole_fill_max_voxels`: fill enclosed outside components up to this voxel count.
- `sdf.closing_radius_voxels`: morphological closing radius in voxels (dilate then erode) to seal small cracks.
- `output.write_hdf5`: write a single HDF5 container (recommended for large samples).
- `output.hdf5_path`: output HDF5 path (defaults to `<output.dir>/dataset.h5`).
- `output.hdf5_chunk`: chunk rows for sample datasets in HDF5.
- `output.write_npy`: enable legacy `.npy` outputs.
- `output.write_metadata_yaml`: save `metadata.yaml` alongside HDF5 (optional).

Outputs (under `output.dir`):
- `sdf.npy` (float32, shape: [nx, ny, nz]) when `write_sdf_grid=true`
- `label.npy` (uint8, 0=boundary, 1=inside, 2=outside)
- `voxel_counts.npy` (uint32, FK hit count per voxel) when enabled
- `orientation_coverage.npy` (float32, shape: [nx, ny, nz]) when enabled
- `orientation_bins.npy` (float32, shape: [B, 4]) when enabled
- `origin.npy` (float32[3]), `dims.npy` (int32[3]), `voxel_size.npy` (float32[1])
- `metadata.yaml`

Grid arrays use C-order with shape [nx, ny, nz] (z is the fastest axis).

HDF5 output (single file when `output.write_hdf5=true`):
- `/samples/pos` (float32, [N, 3]), `/samples/quat` (float32, [N, 4]), `/samples/joint` (float32, [N, J])
- `/samples/voxel` (uint64, [N]) voxel index for each FK sample
- `/grid/label` (uint8, [nx, ny, nz]) and `/grid/voxel_counts` (uint32, [nx, ny, nz])
- `/grid/sdf` (float32, [nx, ny, nz]) when enabled
- `/grid/origin` (float64[3]), `/grid/dims` (uint64[3]), `/grid/voxel_size` (float64[1])
- `/grid/orientation_coverage` and `/grid/orientation_bins` when enabled
- `/csr/voxel_start` (uint64, [num_voxels+1]) and `/csr/sample_index` (uint32/uint64)
- `/meta/config_yaml`, `/meta/metadata_yaml`, `/meta/joint_names`

Orientation dataset (HDF5):
- `/anchors/voxel_id`, `/anchors/pos`, `/anchors/s_v`, `/anchors/c_v`, `/anchors/g_v`, `/anchors/n_seed`
- `/samples/quat` (float32, [N, 4]), `/samples/phi` (float32, signed), `/samples/label` (uint8)
- `/samples/method` (uint8, optional), `/samples/joint` (float32, optional)
- `/csr/anchor_start` + `/csr/sample_index` (per-anchor CSR)
- `/meta/base_h5_path`, `/meta/config_path`, `/meta/config_yaml`

Dependencies for HDF5:
- `sudo apt install libhdf5-dev`
- Python readers: `pip install h5py`

Pack to NPZ:
- `python3 tools/data_gen/scripts/postprocess/pack_npz.py --input-dir tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm --output out.npz`
