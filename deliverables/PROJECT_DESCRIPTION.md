# SPECTRA — VGGT-only geometric mapping via CraftifAI PipeGen

**Track:** A — PipeGen. **Model:** VGGT (the only neural network).

## Problem
Recover usable 3D geometry (camera poses + a point cloud) from an ordinary short video,
deployed as an optimized NVIDIA pipeline instead of a research notebook.

## Inputs
A short local video of a static scene with camera motion. Two frames one second apart are
sampled, preprocessed with upstream VGGT `load_and_preprocess_images(mode="pad")` to 518x518
each, and packed side by side into one RGB atlas: `images`, FP32, NCHW `[1,3,518,1036]`, [0,1].
Bundled test input: `vggt/examples/kitchen/images/00.png` + `01.png` (real upstream images);
`scripts/make_sample_video.py` builds a short real-scene video from the same kitchen set.

## Pipeline tasks
video decode → two-frame temporal sampling → VGGT pad preprocessing → RGB atlas packing →
TensorRT VGGT inference → raw tensor extraction → pose decoding + depth unprojection →
Open3D filtering → PLY + JSON sinks (`geometry_ready` / `inspect_reconstruction` event).

## Model / checkpoint
Official `facebook/VGGT-1B`, `model.safetensors`, SHA256
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` (verified before load by
`scripts/fetch_checkpoint.py`). Upstream VGGT pinned at commit
`a288dd0f14786c93483e45524328726ab7b1b4ce`, unmodified. Full aggregator backbone + camera and
depth heads on exactly two views; point-map and tracking heads disabled via upstream options.

## Outputs
| Name | Shape | Meaning |
|---|---|---|
| `pose_enc` | [1,2,9] | translation, XYZW quaternion, field of view |
| `depth` | [1,2,518,518] | depth, arbitrary scale |
| `depth_conf` | [1,2,518,518] | VGGT confidence |

Actionable output: `cloud.ply`, `metadata.json`, `events.jsonl` with point count, paths and
input/model/engine hashes. Scale is arbitrary (never labelled metres).

## Precision
TensorRT FP16 compute with FP32 input/output (`scripts/build_engine.py`). FP32 weights (~5 GB)
cannot fit the available 4 GB GPU. Engine is validated against the real PyTorch reference
with `spectra.deployment.validate_engine` (rtol 0.01, atol 0.001).

## Hardware actually available
NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB, driver 595.76 (CUDA 13.2), Windows 11, 32 GB RAM.
The intended RTX 5060 target is not present on this machine.

## Status — implemented / tested / verified
| Item | Status | Evidence |
|---|---|---|
| Wrapper, export with parity gates, reference pipeline, DeepStream app, TensorRT runtime, Flask viewer | Implemented | `spectra/` |
| Local unit tests | Tested with a **synthetic** model only | `../evidence/tests.txt` |
| Checkpoint fetch + SHA256 gate, TensorRT builder script | Implemented this session | `scripts/fetch_checkpoint.py`, `scripts/build_engine.py` |
| Official checkpoint download | **In progress** (≈65 MB of 5 GB at 2026-10-15 22:21 IST, ~0.2–1 MB/s link) | `evidence/logs/` |
| Real ONNX export with parity + ONNX Runtime check | **Not yet run** (waits on checkpoint) | — |
| TensorRT engine on RTX 3050 | **Not yet run** | — |
| PipeGen Requirement → PIR → Model → Compiler → Run | **Not run** — PipeGen v0.5.1 installer not available on this machine (no WSL2/Docker) | `evidence/environment_check_2026-10-15.txt` |
| PipeGen speed report, PIR screenshot | **Not produced** | — |
| Live video run with real engine → PLY/JSON | **Not yet run** | — |

## Expected results (PROJECTED — not measured)
Derived from the model definition and repository contract, not from any run on this machine.
- Engine I/O: exactly the three outputs above; input `[1,3,518,1036]` FP32.
- Tokens per view: (518/14)² = 37² = 1369 patch tokens (+ camera/register tokens); 2 views.
- Backbone: 24 frame + 24 global alternating attention blocks, width 1024.
- FP16 engine size: roughly half the FP32 ONNX weight size (~2–2.5 GB); whether the
  builder fits in 4 GB VRAM is **unknown until attempted**.
- Point cloud: up to 2 × 518 × 518 ≈ 537k points before confidence filtering.
- No latency/throughput figure is projected; it will only be reported from a real run.

## Limitations
Exactly two views per reconstruction; arbitrary scale; separate windows are not registered;
no tracking/point-map heads; 4 GB GPU may not fit the TensorRT build; PipeGen-generated
artifacts (PIR, compiler logs, speed report) are absent until PipeGen is installed.

## Bonus target
None claimed.
