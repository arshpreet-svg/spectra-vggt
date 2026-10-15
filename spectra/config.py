from dataclasses import asdict, dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Config:
    output_dir: Path = ROOT / 'artifacts'
    model_id: str = 'facebook/VGGT-1B'
    device: str = 'auto'
    backend: str = 'pytorch'
    deployment_dir: Path = ROOT / 'artifacts' / 'vggt-onnx'
    engine_path: Path = ROOT / 'artifacts' / 'vggt-onnx' / 'vggt.engine'
    max_frames: int = 2
    sample_seconds: float = 1.0
    confidence_percentile: float = 50.0
    max_points: int = 100_000
    voxel_size: float = 0.005  # arbitrary reconstruction units, not metres
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    max_upload_mb: int = 100

    def __post_init__(self):
        if self.backend not in {'pytorch', 'onnx', 'tensorrt'}:
            raise ValueError('backend must be pytorch, onnx or tensorrt')
        if not 1 <= self.max_frames <= 16:
            raise ValueError('max_frames must be between 1 and 16')
        if self.sample_seconds <= 0 or self.max_points < 1:
            raise ValueError('sample_seconds and max_points must be positive')
        if not 0 <= self.confidence_percentile < 100 or self.voxel_size < 0:
            raise ValueError('Invalid confidence percentile or voxel size')

    @classmethod
    def from_env(cls):
        mapping = {'SPECTRA_BACKEND': ('backend', str),
                   'SPECTRA_DEPLOYMENT_DIR': ('deployment_dir', Path),
                   'SPECTRA_ENGINE_PATH': ('engine_path', Path), 'SPECTRA_DEVICE': ('device', str), 'SPECTRA_MAX_FRAMES': ('max_frames', int),
                   'SPECTRA_OUTPUT_DIR': ('output_dir', Path), 'SPECTRA_MODEL_ID': ('model_id', str)}
        return cls(**{field: cast(os.environ[key]) for key, (field, cast) in mapping.items() if key in os.environ})

    def json(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
