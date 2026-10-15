# VGGT-only PipeGen deployment handoff

Implemented source adapters; full pretrained-model export, TensorRT and PipeGen execution
are **not yet verified**. No alternate model is used. Upstream VGGT files remain unchanged.

## Package contents and interface

- `spectra.deployment.export`: actual pretrained VGGT → fixed ONNX with external weights,
  fixed-grid RoPE specialization parity, ONNX checker and ONNX Runtime comparison.
- `spectra.deployment.reference`: Python pipeline for PipeGen repository analysis.
- `spectra.deployment.deepstream`: NVIDIA GStreamer/nvinfer reference application, raw
  tensor metadata extraction, camera/depth decoding, Open3D PLY, NPZ and actionable JSON.
- `spectra.deployment.runtime`: strict ONNX/TensorRT backends for the Flask viewer.
- `spectra.deployment.validate_engine`: comparison against the actual export reference.
- `pipegen/requirement.md`: input to PipeGen's Requirement stage.
- `pipegen/reference-architecture.json`: authored reference graph, **not generated PIR**.

The ONNX input is `images`, FP32 RGB NCHW `[1,3,518,1036]` in [0,1]. Each half is one
518x518 view produced by upstream `load_and_preprocess_images(mode="pad")`. The wrapper
splits the halves and runs the original VGGT jointly on both. Do not treat the atlas
as one wide scene; never supply two unrelated cameras/scenes as a pair. Outputs:

| Name | Shape | Meaning |
|---|---|---|
| pose_enc | [1,2,9] | original VGGT translation, XYZW quaternion and field of view |
| depth | [1,2,518,518] | predicted depth in arbitrary units |
| depth_conf | [1,2,518,518] | original confidence, not a probability |

The fixed grid permits an export-only specialization of upstream RoPE: its frequency
and rotation helpers are reused with constant maximum grid size. No upstream source
file or learned weight is changed. Full-model parity must pass before a bundle is valid.
Unused point/track heads are disabled through upstream options. Points are unprojected
from depth using VGGT cameras; no duplicate odometry or alternate neural network exists.

## 1. Prepare the target

Complete official PipeGen WSL2 installation/sign-in. Confirm `pipegen --version`,
`nvidia-smi` inside Ubuntu, Docker GPU passthrough and the required GUI/DISPLAY.
Use NVIDIA/PipeGen's documented container and matching DeepStream Python bindings.
Administrator changes and docker-group membership require your approval.
No Linux display driver should be installed inside WSL.

Use a separate Python 3.12 environment for export and the Flask app. From the handoff
root, after installing SPECTRA dependencies as described in README:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-export.txt
python scripts/preflight.py
```

The export can run on CPU to leave GPU memory available, but needs substantial host
RAM for both PyTorch and ONNX Runtime weights. An 8 GB GPU does not guarantee that the
TensorRT builder fits. Run no other model concurrently during compilation.

## 2. Export the real model

Use two genuine overlapping images. Default export downloads the complete official
checkpoint; the partial file from this session cannot be loaded.

```bash
python -m spectra.deployment.export \
  vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png \
  --output artifacts/vggt-onnx --device cpu
```

For a complete verified local checkpoint directory, append `--model-id /home/USER/checkpoint`.
A new output directory is required for every attempt. Export errors are actual failures;
inspect the exporter report and do not hand an incomplete bundle to the compiler.

A successful export creates `vggt.onnx`, its external weight file(s), real reference
input/output arrays and `model.json` containing hashes and numerical comparison results.
Keep all graph/weight files in the same relative locations. The manifest is written only
after real ONNX Runtime comparison passes. Synthetic test bundles are rejected at runtime.

## 3. Run the complete PipeGen workflow

Open PipeGen's custom Python repository / ONNX model flow. Supply this repository,
`pipegen/requirement.md`, the **entire validated bundle**, and the Python reference
entrypoint `spectra/deployment/reference.py`.

Review the actual generated Pipeline IR: temporal sampling → upstream pad preprocess →
atlas → VGGT → raw camera/depth outputs → geometry/JSON export. Preserve it for submission.
The authored architecture JSON helps describe the graph; it is not a replacement for PIR.

At the Model/Compiler stages, confirm custom ONNX external-data support and the exact
input/output interface above. Request a memory-conscious FP16 engine with FP32 input/output,
and validate against the real reference. Start with one static batch and two views.
Do not use an object detection parser. Review generated DeepStream settings:
`network-type=100`, `output-tensor-meta=1`, RGB, scale 1/255, no external mean subtraction,
no letterboxing/resizing across the two atlas halves.

Run the compiler on the target; preserve its real logs, engine and PipeGen speed report.
This repository does not pretend to implement a PipeGen CLI or fabricate its exports.
If PipeGen cannot handle custom graph outputs, ONNX external weights or an operator,
retain the exact error for its support team. ONNX readiness does not prove vendor support.

## 4. Compare the actual engine

In a target environment with matching TensorRT 10+ and CUDA PyTorch:

```bash
python -m spectra.deployment.validate_engine \
  vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png \
  --bundle artifacts/vggt-onnx --engine /home/USER/pipegen-export/vggt.engine
```

Use the exact export images in the same order. The command fails if outputs disagree.
Its JSON is a standalone engine comparison, **not PipeGen's speed report**. Record the
precision and the real PipeGen report separately; do not relax tolerances just to pass.

## 5. Video → DeepStream → real geometry

Use the PipeGen-generated `run.sh` for vendor workflow evidence. The following independently
authored reference application is also provided for integration/debugging inside a matching
DeepStream environment with `gi`, `pyds`, Open3D and upstream VGGT utilities available:

```bash
python scripts/make_sample_video.py
python -m spectra.deployment.deepstream \
  --video artifacts/kitchen-sample.mp4 --bundle artifacts/vggt-onnx \
  --engine /home/USER/pipegen-export/vggt.engine --output artifacts/deepstream-run
```

This bounded demo samples two frames from an actual video file, uploads their RGB atlas
to the GPU and runs `nvstreammux → nvinfer → raw tensor metadata`. GPU outputs alone feed
camera decoding and PLY/JSON export. The reference run is not relabeled as a PipeGen-generated
application. It exits on missing plugins, tensor mismatch, invalid geometry or run errors.
The sample video comes from bundled real-scene images; no new camera footage is claimed.

After a real reconstruction, `events.jsonl` contains `geometry_ready` / `inspect_reconstruction`,
point count, artifact paths and hashes. This is actionable geometry delivery, not a person
alert or a metric obstacle-distance estimate. Separate reconstruction windows are not registered.

## 6. Flask + Three.js using the compiled engine

```bash
export SPECTRA_BACKEND=tensorrt
export SPECTRA_DEPLOYMENT_DIR="$PWD/artifacts/vggt-onnx"
export SPECTRA_ENGINE_PATH=/home/USER/pipegen-export/vggt.engine
python -m spectra.app
```

Open http://127.0.0.1:5000 and upload exactly two images or a short video with at least two
sampled frames. The backend executes only the selected engine; no PyTorch inference fallback.
JSON records real engine/model hashes and runtime. An independent TensorRT run does not
prove the engine was compiled by PipeGen: preserve actual vendor workflow artifacts.

## Evidence and remaining limits

Local tests cover adapters, atlas order, upstream RoPE parity, ONNX export/roundtrip of an
explicitly synthetic small model, geometry decoding and Flask artifact flow. They do not
prove export of the 1B model or NVIDIA behavior. Target-only prerequisites remain: complete
checkpoint, full-model parity, genuine PipeGen compile/report and GPU video run. Store PIR
screenshot, export folder, speed report, engine comparison, JSON and demo video for judging.

Official references:
- PipeGen custom repository/model workflow: https://developer.craftifai.com/products/pipegen
- PyTorch export: https://docs.pytorch.org/docs/2.7/onnx.html
- NVIDIA raw tensor metadata: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
- TensorRT I/O addresses: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html
