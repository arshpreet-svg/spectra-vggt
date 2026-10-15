"""Thin wrapper around upstream VGGT; no alternate pose estimation."""
from contextlib import nullcontext
import threading
import numpy as np

class VGGTEngine:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.lock = threading.Lock()
        self.runtime = {}

    def predict(self, paths):
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        with self.lock:
            device = self.config.device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device not in {'cpu', 'cuda'}:
                raise ValueError('Supported devices: auto, cpu, cuda')
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable. Check NVIDIA driver and PyTorch CUDA wheel.')
            if self.model is None:
                # Disable unused point/track heads using supported upstream constructor flags.
                # Geometry comes from depth + VGGT cameras, avoiding redundant head memory.
                self.model = VGGT.from_pretrained(self.config.model_id,
                                                  enable_point=False, enable_track=False).eval().to(device)
            images = load_and_preprocess_images(paths, mode='pad').to(device)
            amp = nullcontext()
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                amp = torch.autocast(device_type='cuda', dtype=dtype)
            with torch.inference_mode(), amp:
                predictions = self.model(images)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
            def array(value):
                return value[0].detach().float().cpu().numpy()
            depth = array(predictions['depth'])
            extrinsic, intrinsic = array(extrinsic), array(intrinsic)
            points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
            self.runtime = {'device': device, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
                            'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                            'preprocessed_shape': list(images.shape), 'model_id': self.config.model_id}
            return {'depth': depth, 'confidence': array(predictions['depth_conf']),
                    'points': points, 'extrinsic': extrinsic, 'intrinsic': intrinsic,
                    'colors': images.detach().float().cpu().numpy().transpose(0, 2, 3, 1)}


def make_engine(config):
    if config.backend == 'onnx':
        from .deployment.runtime import ONNXEngine
        return ONNXEngine(config)
    if config.backend == 'tensorrt':
        from .deployment.runtime import TensorRTEngine
        return TensorRTEngine(config)
    return VGGTEngine(config)
