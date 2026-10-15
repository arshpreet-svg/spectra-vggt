"""Smallest upstream quick-start: one bundled image, actual pretrained weights."""
import os
import json
import time
from pathlib import Path
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

start = time.perf_counter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(json.dumps({'stage': 'loading_pretrained', 'device': device}), flush=True)
model = VGGT.from_pretrained(os.environ.get('SPECTRA_MODEL_ID', 'facebook/VGGT-1B')).eval().to(device)
image = sorted(Path('vggt/examples/kitchen/images').glob('*'))[0]
images = load_and_preprocess_images([str(image)]).to(device)
print(json.dumps({'stage': 'forward', 'shape': list(images.shape)}), flush=True)
with torch.inference_mode():
    predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], images.shape[-2:])
result = {'stage': 'complete', 'device': device, 'seconds': time.perf_counter() - start,
          'input': str(image), 'shapes': {k: list(v.shape) for k, v in predictions.items() if torch.is_tensor(v)}}
Path('evidence/upstream-smoke.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
