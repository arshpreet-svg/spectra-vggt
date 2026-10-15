# Draft submission answers (NOT submitted)
**Project:** SPECTRA — two-view 3D reconstruction from video, with VGGT as the only model, deployed through CraftifAI PipeGen.
**Model:** VGGT-1B (official checkpoint), camera + depth heads, 2 views, 1x3x518x1036 FP32 input.
**What works:** Deployment adapters (ONNX export with parity checks, reference pipeline, DeepStream raw-tensor app, pose/depth decoding to PLY/JSON) are implemented and pass local tests using a synthetic model.
**What is not yet verified:** Export of the real pretrained model, TensorRT compilation, PipeGen PIR/compiler/run, and the speed report. The machine available at the deadline had no PipeGen/TensorRT/Docker (see deliverables/evidence).
**Next step:** Run the export and the PipeGen stages on an NVIDIA target where PipeGen is installed.
