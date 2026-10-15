# Environment report — 2026-10-15 22:41 IST

All values below are from commands run on this machine at that time (raw output captured in
the same session). Nothing is inferred from earlier logs.

## Machine
| Item | Value | Command |
|---|---|---|
| OS | Windows 11 Home Single Language 10.0.26200 (build 26200), 64-bit | `Win32_OperatingSystem` |
| CPU | 12th Gen Intel Core i5-12450H | `Win32_Processor` |
| RAM | 31.7 GB total, 16.6 GB free | `Win32_ComputerSystem` / `Win32_OperatingSystem` |
| Disk | C: 36.7 GB free | `Get-PSDrive C` |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute capability 8.6 | `nvidia-smi --query-gpu=...` |
| NVIDIA driver | 595.76 (nvidia-smi reports CUDA 13.2 capability), WDDM | `nvidia-smi` |
| Other GPUs | Intel UHD Graphics only; **no RTX 5060 present** | `nvidia-smi -L`, `Win32_VideoController` |
| Network | 1.27 MB/s aggregate receive (with downloads running); earlier 0.03–2.5 MB/s, frequent connection drops | `Get-NetAdapterStatistics` |
| Smart App Control | On (`VerifiedAndReputablePolicyState=1`); not changed | registry |

## Deployment software
| Item | Status |
|---|---|
| WSL2 / Linux distro | **Not installed** — `wsl.exe --status`: "The Windows Subsystem for Linux is not installed." |
| Docker / NVIDIA Container Toolkit | **Not found** |
| PipeGen | **Not found** (`pipegen` not on PATH; not in Program Files / user profile) |
| TensorRT (`trtexec`, Python `tensorrt`) | **Not found** |
| CUDA toolkit (`nvcc`) | **Not found** |

## Python
| Environment | Contents |
|---|---|
| System Python 3.12.10 | torch 2.13.0+cpu (`cuda.is_available()=False`), onnxruntime 1.21.1 (CPU/Azure providers), onnx 1.22.0, open3d **missing** |
| `.venv` (export) | Being installed per README: torch 2.7.1+cpu, requirements-export.txt, requirements-spectra.txt, `-e ./vggt` |
| `.venv-gpu` (TensorRT) | Created; install paused to give bandwidth to the checkpoint (torch 2.7.1+cu128, tensorrt-cu12) |

## Checkpoint
`facebook/VGGT-1B` `model.safetensors` — **download in progress** (≈118 MB of ≈5 GB at report
time); not loaded. See `checkpoint_verification.txt`.

## PipeGen requirements vs this machine
Source: https://developer.craftifai.com/products/pipegen (read 2026-10-15). The v0.5.1 lab
Google Doc returned "You need access" for the signed-in account and could not be read.

| PipeGen requirement | This machine |
|---|---|
| Linux x86_64 (Debian/Ubuntu), Jetson, or WSL2 | Windows only; WSL2 not installed (admin + reboot needed) |
| X11 (Xorg) session with DISPLAY | Not available until WSL2/WSLg exists |
| NVIDIA driver ≥ 560.76 on Windows host for WSL2 | 595.76 — **OK** |
| 30 GB free disk on WSL2 | 36.7 GB free now, but VGGT checkpoint + ONNX + engine + Python envs need ~15–18 GB → **insufficient for both** |
| ~12 GB DeepStream container download on first launch | ~3–25 h at measured network speed |
| CraftifAI account (email sign-up + code) | Must be done by the user |

## Blockers
1. No RTX 5060 — only an RTX 3050 Laptop 4 GB; may not fit a VGGT-1B TensorRT build.
2. PipeGen cannot run without WSL2 + Docker + X11; installing WSL2 requires administrator approval and a reboot.
3. Disk space is insufficient for PipeGen (30 GB) plus the VGGT artifacts.
4. Network speed makes the 5 GB checkpoint and 12 GB DeepStream image multi-hour downloads.
