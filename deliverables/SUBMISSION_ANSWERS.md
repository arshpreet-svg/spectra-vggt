# Draft submission answers (NOT submitted)
**Track:** A — PipeGen. **Project:** SPECTRA — two-view 3D reconstruction from video with VGGT as the only model.
**Model:** official facebook/VGGT-1B (model.safetensors, SHA256 f164acf6…467e), camera + depth heads, 2 views, input 1x3x518x1036 FP32; planned TensorRT FP16 compute with FP32 I/O.
**Implemented:** ONNX export with PyTorch-specialization and ONNX Runtime parity gates, checkpoint hash gate, TensorRT builder, engine validator, Python reference pipeline, DeepStream raw-tensor app, pose/depth decoding to PLY/JSON, Flask viewer.
**Tested:** local unit tests with a synthetic model only.
**Not yet verified:** real-weight export, TensorRT engine, PipeGen stages (Requirement/PIR/Model/Compiler/Run), speed report and live video run. Available GPU is an RTX 3050 Laptop (4 GB), not the RTX 5060; PipeGen v0.5.1 could not be installed on it; checkpoint download was still in progress over a ~0.2–1 MB/s connection at submission time.
**Evidence:** deliverables/evidence/ (environment check, download/install logs). No measured performance is claimed.
