# RTX 5060 handoff — PipeGen stages for SPECTRA × VGGT

This machine performs the mandatory PipeGen work: Requirement → Pipeline IR → Model →
Compiler → Run, the PipeGen speed report and the live video run. Everything here is
source of truth for what to do; record **only what actually happens**.

PipeGen steps below are copied from the official page
https://developer.craftifai.com/products/pipegen (read 2026-10-15). The v0.5.1 lab Google
Doc was not accessible to us; if you have it, it overrides this file where they differ.

## 0. Machine requirements (official)
- Ubuntu/Debian x86_64 (or Windows with WSL2) with a graphical **X11 (Xorg)** session — PipeGen
  refuses to start on Wayland or with no `DISPLAY`. On Ubuntu login screen: gear icon →
  "Ubuntu on Xorg".
- NVIDIA driver ≥ 560.35.03 (Linux) or ≥ 560.76 on the Windows host for WSL2. Check `nvidia-smi`.
- Free disk: 50 GB on x86_64 (30 GB on WSL2) **for PipeGen**, plus ~20 GB for SPECTRA
  (checkpoint 5 GB, ONNX bundle ~5 GB, engine, Python env).
- Fast, stable internet: PipeGen pulls a ~12 GB DeepStream image on first launch.
- RAM: ONNX export of VGGT-1B loads PyTorch and ONNX Runtime copies of the weights; 32 GB
  recommended. If the friend's machine has less, export on another machine and copy the bundle.

## 1. Files the friend needs
| What | How to get it | Size |
|---|---|---|
| SPECTRA repo + upstream VGGT submodule | `git clone --recurse-submodules https://github.com/arshpreet-svg/spectra-vggt.git` | small |
| Official checkpoint `facebook/VGGT-1B` `model.safetensors` + `config.json` | `python scripts/fetch_checkpoint.py` (verifies SHA256 `f164acf6…467e`) — or copy a verified file | ~5 GB |
| Validated ONNX bundle (`vggt.onnx`, external weight file(s), `reference-input.npy`, `reference-output.npz`, `model.json`, `export-report/`) | Produced by step 3 below — keep **all files together, same relative paths** | ~5 GB |
| PipeGen installer: `pipegen_prerequisites_ubuntu_x86.sh`, `PipeGen_0.5.1_amd64.deb` | Official page → Download → **Linux x86_64** | — |
| Requirement text | `pipegen/requirement.md` | small |
| Python reference entrypoint | `spectra/deployment/reference.py`, decoding in `spectra/deployment/contract.py` | small |
| Test inputs | `vggt/examples/kitchen/images/00.png`, `01.png`; video via `scripts/make_sample_video.py` | small |
| CraftifAI account | Sign up inside PipeGen (email + verification code) | — |

Not needed and must **not** be supplied as PIR: `pipegen/reference-architecture.json`
(authored reference graph only).

## 2. Install PipeGen (official steps)
In the folder with the downloads:
```bash
chmod +x ./pipegen_prerequisites_ubuntu_x86.sh
./pipegen_prerequisites_ubuntu_x86.sh      # installs Docker, NVIDIA Container Toolkit, socat; adds you to docker group
# log out and back in (group membership)
sudo apt install ./PipeGen_0.5.1_amd64.deb
pipegen                                     # first launch pulls ~12 GB DeepStream image; rerun to resume
```
Then sign up / sign in. If it does not launch: re-run the setup script; make sure Docker is running.

Save evidence (into `deliverables/evidence/`):
```bash
pipegen --version 2>&1 | tee deliverables/evidence/pipegen_version.txt   # only if this flag exists; else screenshot About
nvidia-smi | tee deliverables/evidence/nvidia_smi_rtx5060.txt
docker run --rm --gpus all ubuntu nvidia-smi 2>&1 | tee deliverables/evidence/docker_gpu.txt
```
Copy the setup-script terminal output to `deliverables/evidence/pipegen_install_log.txt` and
take a screenshot of the opened PipeGen window → `deliverables/evidence/pipegen_install.png`.

## 3. SPECTRA environment, checkpoint and real ONNX export
```bash
cd spectra-vggt
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy<2' Pillow huggingface_hub einops safetensors
python -m pip install -r requirements-spectra.txt -r requirements-export.txt
python -m pip install --no-deps -e ./vggt
python scripts/preflight.py 2>&1 | tee deliverables/evidence/logs/preflight.txt
python scripts/fetch_checkpoint.py 2>&1 | tee deliverables/evidence/logs/checkpoint_fetch.txt   # must print CHECKPOINT VERIFIED
python -m spectra.deployment.export \
  vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png \
  --model-id artifacts/checkpoints/VGGT-1B \
  --output artifacts/vggt-onnx-run1 --device cpu 2>&1 | tee deliverables/evidence/onnx_validation.txt
```
Pass criteria: the command exits 0 and `artifacts/vggt-onnx-run1/model.json` exists (it is
written only after PyTorch specialization parity, `onnx.checker` and ONNX Runtime comparison
pass). On failure keep the full output — do not retry into the same directory (use `-run2`).

## 4. PipeGen five stages (GUI; follow the app's own prompts)
Take a screenshot at each stage. Keep PipeGen's project directory intact.

1. **Requirement** — paste the full text of `pipegen/requirement.md`.
2. **Repository / model** — PipeGen first matches against its 15 curated pipelines. **Do not
   accept a catalog match** (none uses VGGT). Choose to supply your own Python repository and
   ONNX model: point it at the `spectra-vggt` folder and `artifacts/vggt-onnx-run1/vggt.onnx`
   with its external weight file(s) beside it.
3. **Pipeline IR** — review: video decoder → two-frame sampling → VGGT pad preprocess → RGB atlas
   → TensorRT VGGT → raw tensors (`pose_enc`, `depth`, `depth_conf`) → pose decode / depth
   unprojection → Open3D filter → PLY + JSON. Screenshot the node graph →
   `deliverables/evidence/pipeline_ir.png`. Confirm.
4. **Model / Compiler** — confirm input `images` [1,3,518,1036] FP32 and the three outputs; request
   FP16 compute with FP32 I/O; `network-type=100`, `output-tensor-meta=1`; no bbox parser.
   PipeGen loops until a clean build. Keep its logs, the `.engine` and the **speed report in its
   original file format** → copy to `deliverables/evidence/tensorrt_speed_report` (+ original extension).
   If it fails (unsupported op, external data, out of memory), save the exact error — that is the
   evidence; do not swap models.
5. **Run** — in the generated project directory: `bash run.sh` (official). Use
   `artifacts/kitchen-sample.mp4` (`python scripts/make_sample_video.py`) or your own short video of
   a static scene with camera motion. `tee` the terminal to `deliverables/evidence/live_run_log.txt`;
   screenshot/record → `deliverables/evidence/live_demo.png`.

Copy the **entire PipeGen project directory** (its 9 output files, PIR, configs, logs) into
`pipegen/export/` in the repo — except the `.engine` and weights (too large for Git; record their
path and SHA256 in `pipegen/export/ARTIFACTS.md`).

## 5. Engine validation and geometry (after PipeGen built the engine)
```bash
python -m spectra.deployment.validate_engine \
  vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png \
  --bundle artifacts/vggt-onnx-run1 --engine <path-to-PipeGen-engine> \
  --report deliverables/evidence/engine-comparison.json
python -m spectra.deployment.reference artifacts/kitchen-sample.mp4 --backend tensorrt \
  --bundle artifacts/vggt-onnx-run1 --engine <path-to-PipeGen-engine> --output artifacts/e2e-run1
```
Copy `artifacts/e2e-run1/cloud.ply` → `deliverables/evidence/sample_output.ply` (if < 50 MB) and
`events.jsonl`/`metadata.json` → `deliverables/evidence/sample_output.json`.

## 6. Commit
```bash
git add deliverables pipegen/export docs && git commit -m "PipeGen run evidence on RTX 5060" && git push
```
Never commit the checkpoint, ONNX weights, `.engine`, Docker images or credentials.

## Evidence checklist (fill in only from real runs)
| Required | Path |
|---|---|
| PipeGen version / install | `deliverables/evidence/pipegen_version.txt`, `pipegen_install_log.txt`, `pipegen_install.png` |
| Checkpoint verified | `deliverables/evidence/logs/checkpoint_fetch.txt` |
| ONNX export + parity | `deliverables/evidence/onnx_validation.txt`, bundle `model.json` |
| Pipeline IR screenshot | `deliverables/evidence/pipeline_ir.png` |
| PipeGen speed report | `deliverables/evidence/tensorrt_speed_report*` |
| PipeGen export folder | `pipegen/export/` |
| Engine comparison | `deliverables/evidence/engine-comparison.json` |
| Live run | `deliverables/evidence/live_run_log.txt`, `live_demo.png` |
| Geometry output | `deliverables/evidence/sample_output.ply`, `sample_output.json` |
