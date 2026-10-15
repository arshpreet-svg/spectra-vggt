from dataclasses import replace
import argparse
import json
from pathlib import Path
from uuid import uuid4
from .config import Config
from .engine import make_engine
from .reconstruction import reconstruct

parser = argparse.ArgumentParser(description='SPECTRA: VGGT monocular geometric reconstruction')
parser.add_argument('inputs', nargs='+', type=Path)
parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
parser.add_argument('--max-frames', type=int, default=2)
args = parser.parse_args()
config = replace(Config.from_env(), device=args.device, max_frames=args.max_frames)
job_id = uuid4().hex
metadata = reconstruct(args.inputs, config.output_dir / job_id, config, make_engine(config))
print(json.dumps({'id': job_id, 'directory': str(config.output_dir / job_id), 'metadata': metadata}, indent=2))
