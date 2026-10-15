# Superseded: active scope is VGGT only

Use `VGGT_PIPEGEN.md` and `pipegen/requirement.md`. The detector fallback below is inactive.

# PipeGen integration: pending hardware execution

Official sources checked 15 October 2026:
- https://developer.craftifai.com/products/pipegen
- https://craftifai.com/products/pipegen
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html

Vendor prerequisites: Linux with X11/DISPLAY, NVIDIA GPU, Docker/toolkit;
WSL2 host driver >=560.76 and 30 GB free (x86 Linux: >=560.35.03, 50 GB).
These are vendor minimums, not proof of RTX 5060 compatibility.
Use a current Blackwell-capable driver/PyTorch/container and execute a CUDA kernel.
Do not install a Linux display driver inside WSL.

Download the official setup/package for your architecture; inspect before running.
The vendor setup installs Docker/NVIDIA toolkit/socat and changes docker-group access;
the .deb installation requires sudo. Obtain administrator approval before these changes.
First launch downloads a large DeepStream image. Account signup/email verification is required.

Documented model input is ONNX plus Python repository/custom parsers.
VGGT checkpoint compatibility is **unproven**; no conversion is attempted here.
Use a curated YOLOv8 detector as the separate candidate and confirm it in PipeGen.

## Requirement → PIR → Model → Compiler → Run

1. Requirement: request one-file video detection with per-frame JSON, NVIDIA target,
   configurable score threshold, bounding boxes, labels and timestamps. No metric distances.
2. PIR: inspect the actual generated graph and save it unchanged; confirm decoder,
   preprocessing, inference, postprocessing and JSON sink.
3. Model: select the curated detector, record ONNX hash, shapes, labels and parser.
   `scripts/detection_reference.py` is an independent OpenCV reference for standard
   YOLOv8 `[1,4+classes,N]` output at 640x640. It requires a real ONNX model and labels;
   it is not a PipeGen-generated artifact and has not been tested with real model weights.
4. Compiler: save actual logs and generated TensorRT engine. Record target GPU,
   driver, CUDA, TensorRT, precision and engine/model hashes. No engine is provided here.
5. Run: execute the generated `bash run.sh`; retain the real speed report, command,
   input hash, GPU evidence and captured JSON. Verify detection correctness on the input.

## SPECTRA boundary contract

Generated JSON sink must emit JSONL with `schema_version:1`, `producer:"pipegen"`,
nonnegative `frame_index` and `timestamp_seconds`, `image_size:[width,height]`, and
`detections:[{label,confidence,bbox_xyxy:[x1,y1,x2,y2]}]` in original-image pixels.
An empty detections array is valid. Adapt the actual generated sink to this contract.
Do not relabel reference output as PipeGen output.

After a real SPECTRA reconstruction:

```bash
python -m scripts.relay_detections path/to/actual-pipegen.jsonl --job JOB_ID
```

The adapter validates and stores events at `/api/jobs/JOB_ID/detections`.
The viewer can inspect them. They remain 2D, separate from the VGGT scene.
Posting events does not prove GPU or PipeGen execution.

Official WSL2 tab specifically uses Docker Engine inside the Linux distribution,
not the Windows Docker Desktop engine. Configure the NVIDIA Container Toolkit
and restart that Docker service; then install the amd64 PipeGen package in WSL.
Verify Linux GPU passthrough, GUI/DISPLAY and container GPU execution before claiming success.
The WSL2 instructions were also checked in the public developer frontend bundle.
