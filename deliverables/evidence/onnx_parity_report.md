# ONNX parity report — real VGGT-1B export (run2)

Generated from `artifacts/vggt-onnx-run2/model.json` and `reference-output.npz`, both written by
`python -m spectra.deployment.export` (command and full log: `onnx_validation.txt`). Run on CPU,
Windows 11, torch 2.7.1+cpu, onnx 1.19.1, onnxruntime 1.23.2, 2026-10-15.

## Inputs
- Checkpoint: `facebook/VGGT-1B` `model.safetensors`, 5,026,367,224 bytes, SHA256 verified (`checkpoint_verification.txt`).
- Images: `vggt/examples/kitchen/images/00.png`, `01.png` (upstream real images); sha256 54527a575988…, 0ad4c6d74c16….
- ONNX input `images` [1, 3, 518, 1036] float32 — NCHW RGB: first view left, second view right; values [0,1].

## Checks (all must pass before model.json is written)
| Check | Result |
|---|---|
| Real checkpoint loaded, PyTorch forward on two real views | PASS (reference outputs saved) |
| Fixed-grid RoPE specialization vs original PyTorch (rtol=atol=1e-6) | PASS |
| ONNX export (opset 18, dynamo, external data) | PASS |
| No nonstandard ONNX domains after inlining | PASS |
| `onnx.checker.check_model` | PASS |
| ONNX Runtime CPU load + run, output shapes | PASS |
| ONNX Runtime vs PyTorch, rtol=0.001, atol=0.0001 (repo defaults, unchanged) | PASS |

## Numerical comparison (ONNX Runtime vs PyTorch reference)
| Output | Shape | Max abs error | Reference min / max / std |
|---|---|---|---|
| `pose_enc` | [1, 2, 9] | 6.56e-07 | -0.5245 / 1 / 0.4661 |
| `depth` | [1, 2, 518, 518] | 4.1e-05 | 0.4189 / 2.254 / 0.3632 |
| `depth_conf` | [1, 2, 518, 518] | 0.000645 | 1 / 18.02 / 5.742 |

All outputs finite; non-constant (std > 0), so outputs vary meaningfully across the image.

## Change required to pass (run1 → run2)
Run1 (`logs/onnx_validation_run1_failed.txt`) exported but ONNX Runtime refused to load it:
`Type parameter (T) of Optype (Einsum) bound to different types (tensor(float) and tensor(double))`.
Cause: upstream `vggt/heads/utils.py` `make_sincos_pos_embed` builds `omega` in float64 and einsums it with
float32 positions; PyTorch promotes, the exporter emitted a mixed-type Einsum. Fix (SPECTRA code only,
upstream untouched): `demote_float64_constants` stores the 4 float64 tables (mul_568, mul_571, mul_574, mul_580)
as float32, giving a single floating type (also required by TensorRT, which has no float64). Accuracy is still
judged against the float64 PyTorch reference with unchanged tolerances (above). Probe before re-export:
`logs/run1_float64_fix_probe.txt`.

## Scope
Two views only; camera + depth heads; point-map and tracking heads disabled. Not a TensorRT or PipeGen result.
Operators in graph: Add, Cast, Concat, Conv, ConvTranspose, Cos, Div, Einsum, Erf, Exp, Expand, Gather, LayerNormalization, Less, MatMul, Mul, Neg, Relu, Reshape, Resize, Sigmoid, Sin, Slice, Softmax, Split, Squeeze, Sub, Transpose, Unsqueeze, Where.

## Bundle files (sha256)
- `reference-input.npy` b566b0851a049b116b0e940146f8718ff5499485d881eabc05d9a1fae3c5eab0
- `reference-output.npz` c8e9cdbb98be752cc364af94fbda03236109629a7f24464546ab87c913e1cf3d
- `vggt.onnx` d16675bcdb6f804289766e08501a40aa163b782051b77cd5739fba9e3f1d5662
- `vggt.onnx.data` 39c88dc6abfa3c14ea7b5518b017dee4555b7bc328a60eddf11c121cb463e274
