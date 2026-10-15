"""Build a short real-scene video from bundled upstream images; not camera footage."""
from pathlib import Path
import cv2

paths=sorted(Path('vggt/examples/kitchen/images').glob('*.png'))[:8]
if not paths:raise FileNotFoundError('Clone official VGGT first')
first=cv2.imread(str(paths[0]));height,width=first.shape[:2]
output=Path('artifacts/kitchen-sample.mp4')
output.parent.mkdir(exist_ok=True)
writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),4,(width,height))
if not writer.isOpened():raise RuntimeError('MP4 codec unavailable')
for path in paths:
    frame=cv2.imread(str(path))
    if frame is None:raise ValueError(f'Cannot read {path}')
    writer.write(cv2.resize(frame,(width,height)))
writer.release()
print(f'{output}: {len(paths)} real-scene images, 4 fps; source: upstream VGGT kitchen examples')
