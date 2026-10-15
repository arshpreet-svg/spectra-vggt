"""Compile the validated VGGT ONNX bundle to a TensorRT engine with the TensorRT builder API.

Standalone local compilation (trtexec equivalent). It is not a PipeGen Compiler-stage artifact.
"""
import argparse
import json
import platform
import time
from pathlib import Path
import tensorrt as trt
from spectra.deployment.contract import sha256


class TeeLogger(trt.ILogger):
    def __init__(self, path, level):
        super().__init__()
        self.level = level
        self.handle = open(path, 'w', encoding='utf-8')

    def log(self, severity, msg):
        line = f'[{severity.name}] {msg}'
        self.handle.write(line + '\n')
        self.handle.flush()
        if severity <= self.level:
            print(line, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--workspace-mib', type=int, default=1024)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    if args.engine.exists():
        raise SystemExit('Engine already exists; choose a new path to preserve prior evidence')
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(args.log, trt.ILogger.Severity.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    model = args.bundle / 'vggt.onnx'
    start = time.perf_counter()
    if not onnx_parser.parse_from_file(str(model)):
        for index in range(onnx_parser.num_errors):
            print('PARSE ERROR', onnx_parser.get_error(index), flush=True)
        raise SystemExit('ONNX parse failed')
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    if args.precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    print(f'parsed {network.num_layers} layers in {time.perf_counter()-start:.1f}s; building {args.precision}', flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit('TensorRT build failed; see log')
    args.engine.write_bytes(bytes(serialized))
    seconds = time.perf_counter() - start
    import torch
    summary = {
        'kind': 'standalone_tensorrt_build', 'is_pipegen_artifact': False,
        'tensorrt': trt.__version__, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'os': platform.platform(), 'precision': args.precision, 'io_dtype': 'float32',
        'workspace_mib': args.workspace_mib, 'layers': network.num_layers,
        'onnx_sha256': sha256(model), 'engine': str(args.engine), 'engine_bytes': args.engine.stat().st_size,
        'engine_sha256': sha256(args.engine), 'build_seconds': seconds,
    }
    args.log.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
