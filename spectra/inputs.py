from pathlib import Path
import cv2
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}

def prepare_frames(sources, destination, config):
    destination.mkdir(parents=True, exist_ok=True)
    paths, records = [], []
    def save(image, source, frame_index=None, timestamp=None):
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((1600, 1600))
        path = destination / f'{len(paths):04d}.png'
        image.save(path)
        paths.append(str(path))
        records.append({'image': path.name, 'source': Path(source).name,
                        'frame_index': frame_index, 'timestamp_seconds': timestamp})
    for source in sources:
        source = Path(source)
        if len(paths) >= config.max_frames:
            break
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            with Image.open(source) as image:
                save(image, source)
        elif source.suffix.lower() in VIDEO_EXTENSIONS:
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f'Cannot decode video: {source.name}')
                fps = capture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    raise ValueError('Video has invalid frame rate')
                stride, index = max(1, round(fps * config.sample_seconds)), 0
                while len(paths) < config.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % stride == 0:
                        save(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), source, index, index / fps)
                    index += 1
            finally:
                capture.release()
        else:
            raise ValueError(f'Unsupported input: {source.name}')
    if not paths:
        raise ValueError('No decodable input frames')
    return paths, records
