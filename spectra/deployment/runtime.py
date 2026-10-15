from pathlib import Path
import threading
import time
import numpy as np
from .contract import INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES, OUTPUT_SHAPES, decode_outputs, preprocess, sha256, verify_bundle


class ONNXEngine:
    def __init__(self,config):
        self.config=config
        self.lock=threading.Lock()
        self.session=None
        self.runtime={}
        self.manifest=None

    def predict(self,paths):
        import onnxruntime as ort
        with self.lock:
            directory=Path(self.config.deployment_dir)
            if self.manifest is None:self.manifest=verify_bundle(directory)
            manifest=self.manifest
            if self.session is None:
                self.session=ort.InferenceSession(str(directory/'vggt.onnx'),providers=['CPUExecutionProvider'])
            atlas=preprocess(paths)
            start=time.perf_counter()
            outputs=self.session.run(OUTPUT_NAMES,{INPUT_NAME:atlas})
            self.runtime={'backend':'onnxruntime','device':'cpu','model_sha256':manifest['files']['vggt.onnx'],
                          'forward_seconds':time.perf_counter()-start,'pipegen_verified':False}
            return decode_outputs(dict(zip(OUTPUT_NAMES,outputs)),atlas)


class TensorRTEngine:
    """Execute a real static TensorRT 10+ engine; never silently use PyTorch."""
    def __init__(self,config):
        self.config=config
        self.lock=threading.Lock()
        self.engine=None
        self.runtime={}
        self.manifest=None

    def predict(self,paths):
        import torch
        import tensorrt as trt
        if not torch.cuda.is_available():
            raise RuntimeError('TensorRT backend requires a working NVIDIA CUDA runtime')
        with self.lock:
            if self.manifest is None:self.manifest=verify_bundle(self.config.deployment_dir)
            manifest=self.manifest
            path=Path(self.config.engine_path)
            if not path.is_file():raise FileNotFoundError(f'Actual compiled engine required: {path}')
            if self.engine is None:
                self.logger=trt.Logger(trt.Logger.WARNING)
                self.trt_runtime=trt.Runtime(self.logger)
                loaded_engine=self.trt_runtime.deserialize_cuda_engine(path.read_bytes())
                if loaded_engine is None:raise RuntimeError('TensorRT could not deserialize the engine on this GPU/runtime')
                self.context=loaded_engine.create_execution_context()
                if self.context is None:raise RuntimeError('TensorRT could not allocate an execution context')
                self.engine=loaded_engine
                self.stream=torch.cuda.Stream()
                self.engine_hash=sha256(path)
            engine=self.engine
            names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
            inputs=[name for name in names if engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT]
            outputs=[name for name in names if engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT]
            if inputs!=[INPUT_NAME] or set(outputs)!=set(OUTPUT_NAMES):
                raise ValueError(f'Engine interface mismatch: inputs={inputs}, outputs={outputs}')
            if tuple(engine.get_tensor_shape(INPUT_NAME))!=INPUT_SHAPE:
                raise ValueError('Engine input shape differs from the fixed atlas contract')
            if engine.get_tensor_dtype(INPUT_NAME)!=trt.float32:
                raise ValueError('Engine must expose FP32 RGB input, even with FP16 internal layers')
            atlas=preprocess(paths)
            tensors={}
            start=time.perf_counter()
            with torch.cuda.stream(self.stream):
                tensors[INPUT_NAME]=torch.from_numpy(atlas).to('cuda')
                for name in OUTPUT_NAMES:
                    if tuple(self.context.get_tensor_shape(name))!=OUTPUT_SHAPES[name]:
                        raise ValueError(f'Engine output shape differs: {name}')
                    dtype=engine.get_tensor_dtype(name)
                    if dtype not in {trt.float32,trt.float16}:
                        raise ValueError(f'Unsupported floating output dtype for {name}: {dtype}')
                    tensors[name]=torch.empty(OUTPUT_SHAPES[name],device='cuda',dtype=torch.float32 if dtype==trt.float32 else torch.float16)
                for name,tensor in tensors.items():
                    if not self.context.set_tensor_address(name,tensor.data_ptr()):
                        raise RuntimeError(f'TensorRT rejected buffer for {name}')
                if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
                    raise RuntimeError('TensorRT execution failed')
            self.stream.synchronize()
            outputs={name:tensors[name].float().cpu().numpy() for name in OUTPUT_NAMES}
            self.last_outputs=outputs
            self.runtime={'backend':'tensorrt','device':'cuda','gpu':torch.cuda.get_device_name(),
                          'tensorrt':trt.__version__,'engine_sha256':self.engine_hash,
                          'model_sha256':manifest['files']['vggt.onnx'],
                          'forward_seconds':time.perf_counter()-start,
                          'pipegen_verified':False,
                          'note':'Engine execution alone does not establish PipeGen provenance or speed-report evidence'}
            return decode_outputs(outputs,atlas)
