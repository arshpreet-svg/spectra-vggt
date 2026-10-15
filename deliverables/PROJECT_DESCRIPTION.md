# SPECTRA — VGGT-only geometric mapping via CraftifAI PipeGen

## What it is
SPECTRA turns a short video into a 3D point cloud using **VGGT as the only neural network**.
Two frames sampled 1 s apart are VGGT-pad-preprocessed to 518x518, packed into a 1036x518 RGB
atlas (NCHW FP32, input `images`) and run through the full VGGT backbone with camera and depth
heads (point-map and tracking heads disabled). Outputs: `pose_enc` [1,2,9], `depth` [1,2,518,518],
`depth_conf` [1,2,518,518]. Geometry is reconstructed by decoding poses and unprojecting depth,
filtered with Open3D, and written to PLY + JSON (`geometry_ready`). Scale is arbitrary (not metres).

## Status (honest)
| Item | Status |
|---|---|
| Wrapper, ONNX export script, reference pipeline, DeepStream app, output contract | **Implemented** |
| Local tests | **Tested — synthetic model only** (not deployment evidence) |
| Export of pretrained VGGT to ONNX with parity checks | **Not yet run** |
| TensorRT engine compilation | **Not verified** |
| PipeGen Requirement → PIR → Model → Compiler → Run | **Not verified** |
| Live video run on GPU, PLY/JSON from real engine | **Not verified** |

## Why not verified
At deadline the available machine (RTX 3050 Laptop 4 GB, CPU-only PyTorch) had no PipeGen,
Docker, WSL, CUDA toolkit or TensorRT installed — see `evidence/environment_check_2026-10-15.txt`.
No results were simulated or substituted.

## Planned configuration
Checkpoint: official `facebook/VGGT-1B` (`model.pt`). Input 1x3x518x1036 FP32. Precision: TensorRT
FP32 first; FP16 only after comparison with the PyTorch reference. Target: NVIDIA GPU with PipeGen.

## Limitations
Exactly two views; arbitrary scale; no tracking/point-map heads; the 1B-parameter model needs
substantial GPU memory, so a 4 GB GPU may not be able to build an FP32 engine.
