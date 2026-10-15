# SPECTRA — VGGT geometric mapping with a PipeGen deployment adapter

VGGT is the only model. The official upstream repository is preserved in `vggt/`,
commit `a288dd0f14786c93483e45524328726ab7b1b4ce`, with no source/pin changes.

Implemented: two-view video/image sampling, VGGT camera/depth prediction, depth
unprojection, Open3D filtering, valid PLY/JSON/NPZ export, Flask API and local Three.js
viewer. Deployment source adds a fixed ONNX interface, parity gates, TensorRT runtime,
DeepStream reference application and PipeGen custom-model requirement/architecture.

**Not yet verified:** pretrained VGGT export, NVIDIA TensorRT/DeepStream execution,
PipeGen installation/custom-model acceptance, actual PIR/engine/speed report, browser
WebGL rendering, or hackathon compliance. Source readiness is not vendor certification.
No substitute model, fabricated engine, benchmark or report is supplied.

## Start here

Clone this repository with its pinned upstream VGGT dependency:

```bash
git clone --recurse-submodules https://github.com/arshpreet-svg/spectra-vggt.git
cd spectra-vggt
```

For an existing clone, run `git submodule update --init --recursive`.

For the complete **export → PipeGen → engine comparison → DeepStream → viewer**
commands, read [VGGT/PipeGen handoff](docs/VGGT_PIPEGEN.md).
Supply [the requirement](pipegen/requirement.md) to PipeGen. The architecture JSON is
an authored reference graph; preserve PipeGen's actual generated PIR separately.

## Install the application/export environment

Python 3.12 recommended. From this directory in an existing Linux/WSL environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy<2' Pillow huggingface_hub einops safetensors
python -m pip install -r requirements-spectra.txt -r requirements-export.txt
python -m pip install --no-deps -e ./vggt
python scripts/preflight.py
```

For the local Mac, install the same torch/torchvision pair from the normal PyPI index,
without the CUDA index argument. Native Windows uses `py -3.12 -m venv .venv` and
`.venv\Scripts\Activate.ps1`. PipeGen itself requires its documented Linux setup.

The separately installed runtime uses newer PyTorch than upstream's torch==2.3.1 pin
because RTX 50-series Blackwell support needs a newer CUDA build. The upstream dependency
file is unchanged. The target pair must pass a real CUDA operation on your actual laptop.
See https://pytorch.org/blog/pytorch-2-7/ and `scripts/preflight-windows.ps1`.

Use PipeGen's selected DeepStream environment and matching TensorRT/`pyds` bindings;
these are not ordinary portable pip dependencies and have not been installed on this Mac.
Approve administrator, Docker and group changes yourself before running vendor setup.

## Smallest original-model inference

```bash
export HF_HOME="$PWD/artifacts/hf-cache"
python scripts/upstream_smoke.py
```

This uses one bundled real kitchen image, actual pretrained weights and upstream
preprocessing. Success evidence is written only after the forward pass. The roughly
5 GB checkpoint download failed in this session with curl error 92 (HTTP/2 CANCEL).
The 681 MiB `artifacts/model.safetensors` is incomplete and must not be loaded.
The complete official safetensors SHA256 is
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
A complete verified local checkpoint directory can be selected with `SPECTRA_MODEL_ID`.

## Export for PipeGen

```bash
python -m spectra.deployment.export \
  vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png \
  --output artifacts/vggt-onnx --device cpu
```

The wrapper packs two padded 518x518 views into a conventional `[1,3,518,1036]` RGB
input, then unpacks them within ONNX before joint VGGT inference. Outputs are original
pose encoding, depth and confidence. Do not normalize twice: VGGT handles ImageNet
normalization internally. External graph weight files must travel with the ONNX file.

The exporter checks fixed-grid specialization equivalence and ONNX Runtime agreement
against actual PyTorch predictions. A valid manifest is written only after these pass.
No random-weight fallback exists. Local synthetic test bundles are labeled and rejected
by production runtimes. A full-model export can require considerable host RAM.

Import the validated bundle/repository through PipeGen's custom model workflow, compile
there, retain its actual engine/speed report, and follow [the target guide](docs/VGGT_PIPEGEN.md).
No unsupported PipeGen CLI or generated IR format is invented here.

## Local reconstruction and viewer

PyTorch baseline:

```bash
python scripts/make_sample_video.py
python -m spectra artifacts/kitchen-sample.mp4 --device cuda --max-frames 2
python -m spectra.app
```

For a real engine compiled by PipeGen:

```bash
export SPECTRA_BACKEND=tensorrt
export SPECTRA_DEPLOYMENT_DIR="$PWD/artifacts/vggt-onnx"
export SPECTRA_ENGINE_PATH=/home/USER/pipegen-export/vggt.engine
python -m spectra.app
```

Open **http://127.0.0.1:5000** and upload two overlapping images or a short video.
The ONNX/TensorRT adapters require exactly two sampled views. The PyTorch baseline
also supports one view. The supplied sample video uses real-scene upstream images;
it is not newly filmed footage. Viewer assets are pinned/vendored Three.js 0.170.0.

Configuration in `spectra/config.py`: `SPECTRA_BACKEND=pytorch|onnx|tensorrt`,
`SPECTRA_DEPLOYMENT_DIR`, `SPECTRA_ENGINE_PATH`, `SPECTRA_DEVICE=auto|cpu|cuda`,
`SPECTRA_MAX_FRAMES`, `SPECTRA_OUTPUT_DIR`, `SPECTRA_MODEL_ID`. Start with two frames
and measure available memory. Runtime selection is explicit; there is no silent fallback.

The loopback server permits one reconstruction at a time. Artifacts: `cloud.ply`,
`metadata.json`, `predictions.npz`. PLY is read back through Open3D to check validity.
JSON records runtime and engine hashes when present; it does not assert PipeGen provenance.

**Scale is arbitrary.** Translations and voxel sizes are not metres. Independent windows
are not registered. This is bounded reconstruction, not continuous loop-closed SLAM.
Review upstream checkpoint/code licensing before sharing or using commercially.

## Verification

```bash
python -m pytest -q
python -m compileall -q spectra scripts
```

Tests cover atlas order, upstream RoPE equivalence, ONNX/ORT adapter roundtrip with an
explicit synthetic attention model, known-camera geometry decoding, frame sampling,
PLY readback, Flask upload/export and failure paths. They are not model benchmarks.
See `evidence/status.json` for implementation/test/target-verification distinctions.

Earlier detection-adapter source remains inactive. No detector or FirmGen project is
part of the active pipeline. The source handoff omits checkpoints, virtualenv and Git
history; the local `vggt/` checkout retains its upstream history.
