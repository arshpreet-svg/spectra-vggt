# SPECTRA: VGGT-only geometric mapping

Use the supplied Python repository and the validated custom `vggt.onnx` model bundle.
VGGT is the only neural network. Do not substitute detection, depth-only or simulated
models. Preserve all three raw outputs and perform the provided camera/depth decoding.

Input: one short local video under /home on the NVIDIA target. Sample exactly two
frames one second apart from a static scene with camera motion and overlapping views.
Preprocess each with upstream VGGT pad mode to RGB 518x518, pack first view on the left
and second on the right of a 1036x518 RGB atlas. NCHW FP32 input named `images`, batch 1,
values [0,1]. VGGT performs its own ImageNet normalization; do not normalize twice.

Review a Pipeline IR with these blocks: video decoder, temporal two-frame sampling,
VGGT pad preprocessing, RGB atlas packing, TensorRT inference, raw tensor extraction,
VGGT pose decoding/depth unprojection, Open3D filtering, PLY and JSON sinks.

Use TensorRT FP32 initially; offer FP16 only after comparing against the real reference.
The output names are `pose_enc` [1,2,9], `depth` [1,2,518,518] and `depth_conf`
[1,2,518,518]. Compile this exact model. Keep ONNX external-data weights with the graph.
Configure DeepStream raw tensor outputs (network-type=100 and output-tensor-meta=1).
There is no object detector or bounding-box parser.

Reference entrypoints:
- `python -m spectra.deployment.reference ...` for the Python pipeline.
- `python -m spectra.deployment.deepstream ...` for the DeepStream app implementation.
- `spectra/deployment/contract.py` for output decoding.

Emit JSON `geometry_ready` with `action: inspect_reconstruction`, point count, PLY path,
metadata path and input/model/engine hashes after writing real geometry. Fail explicitly
if input views, tensors or geometry are invalid. Scale is arbitrary; never label units
as metres. The viewer must display the resulting engine-derived point cloud.

Preserve actual Requirement/PIR/Model/Compiler/Run artifacts, build logs, serialized
engine, PipeGen speed report and live video run evidence. This requirement file is
source input, not a generated or verified PipeGen artifact. Report unsupported operators,
external-data restrictions and GPU memory failures explicitly instead of claiming success.
