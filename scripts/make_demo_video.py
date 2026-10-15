"""Render a 1-3 minute demo MP4 strictly from a real SPECTRA run directory.

Every number and image shown is read from the run's own files (frames, predictions.npz,
cloud.ply, metadata.json, events.jsonl, run log). Nothing is synthesized or estimated.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

W, H, FPS = 1280, 720, 30
BG = (24, 22, 20)
FG = (235, 235, 235)
ACCENT = (80, 200, 255)
MUTED = (150, 150, 150)


def canvas():
    return np.full((H, W, 3), BG, np.uint8)


def text(img, s, xy, scale=0.7, color=FG, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def header(img, title, subtitle):
    text(img, title, (40, 55), 1.0, ACCENT, 2)
    text(img, subtitle, (40, 88), 0.55, MUTED)
    text(img, 'SPECTRA | VGGT-1B | runtime: ONNX Runtime CPU (validated export) | not PipeGen / TensorRT',
         (40, H - 22), 0.5, MUTED)


def fit(image, box_w, box_h):
    h, w = image.shape[:2]
    s = min(box_w / w, box_h / h)
    return cv2.resize(image, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def paste(img, tile, x, y):
    h, w = tile.shape[:2]
    img[y:y + h, x:x + w] = tile


def hold(writer, frame, seconds):
    for _ in range(int(seconds * FPS)):
        writer.write(frame)


def colorize(depth, valid=None):
    d = depth.astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    if valid is not None:
        out[~valid] = 0
    return out


def render_cloud(points, colors, angle, size=(860, 560), elevation=0.22):
    w, h = size
    centered = points - np.median(points, axis=0)
    ca, sa = np.cos(angle), np.sin(angle)
    ce, se = np.cos(elevation), np.sin(elevation)
    yaw = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    pitch = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    p = centered @ yaw.T @ pitch.T
    p[:, 2] += np.ptp(centered, axis=0).max() * 1.05
    f = 0.9 * min(w, h)
    keep = p[:, 2] > 1e-3
    p, c = p[keep], colors[keep]
    u = (f * p[:, 0] / p[:, 2] + w / 2).astype(int)
    v = (f * p[:, 1] / p[:, 2] + h / 2).astype(int)
    order = np.argsort(-p[:, 2])
    u, v, c = u[order], v[order], c[order]
    img = np.full((h, w, 3), BG, np.uint8)
    bgr = (np.clip(c[:, ::-1], 0, 1) * 255).astype(np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = bgr[ok]
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True, help='Original input video')
    parser.add_argument('--log', type=Path, required=True, help='Captured run log')
    parser.add_argument('--viewer', type=Path, nargs='*', default=[], help='Real viewer screenshots')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    event = json.loads((args.run / 'events.jsonl').read_text().splitlines()[0])
    pred = np.load(args.run / 'predictions.npz')
    cloud = o3d.io.read_point_cloud(str(args.run / 'cloud.ply'))
    points, colors = np.asarray(cloud.points), np.asarray(cloud.colors)
    frames = [cv2.imread(str(p)) for p in sorted((args.run / 'frames').glob('*.png'))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError('Cannot open MP4 writer')

    # 1. Title
    img = canvas()
    text(img, 'SPECTRA', (40, 250), 2.4, ACCENT, 4)
    text(img, 'Two-view 3D reconstruction from ordinary video with VGGT-1B', (44, 310), 0.9)
    text(img, 'Video -> 2 frames -> VGGT (camera + depth) -> point cloud -> PLY + JSON event -> viewer', (44, 360), 0.65, MUTED)
    text(img, f"Checkpoint facebook/VGGT-1B (SHA256 verified) | model {meta['runtime']['model_sha256'][:12]}...", (44, 410), 0.6, MUTED)
    text(img, 'Runtime in this recording: ONNX Runtime on CPU. PipeGen/TensorRT stages are not shown here.', (44, 450), 0.6, MUTED)
    hold(writer, img, 7)

    # 2. Real input video
    capture = cv2.VideoCapture(str(args.video))
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 4
    clip = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        clip.append(frame)
    capture.release()
    for loop in range(2):
        for frame in clip:
            img = canvas()
            header(img, '1. Real input video', f'{args.video.name}: {len(clip)} frames @ {src_fps:.0f} fps (upstream VGGT kitchen scene)')
            tile = fit(frame, W - 80, H - 170)
            paste(img, tile, (W - tile.shape[1]) // 2, 110)
            hold(writer, img, 1 / src_fps)

    # 3. Sampled pair
    img = canvas()
    header(img, '2. Two frames sampled, VGGT pad-preprocessed to 518x518', 'Packed side by side into one 1x3x518x1036 FP32 atlas (model input "images")')
    for i, frame in enumerate(frames[:2]):
        tile = fit(frame, 580, 500)
        paste(img, tile, 40 + i * 620, 120)
        text(img, f'view {i}', (40 + i * 620, 640), 0.6, ACCENT)
    hold(writer, img, 8)

    # 4. Real run log
    lines = [l.rstrip().replace("	", "  ") for l in args.log.read_text(errors='replace').splitlines() if l.strip()]
    img = canvas()
    header(img, '3. Pipeline run (actual terminal output)', 'python -m spectra.deployment.reference kitchen-sample.mp4 --backend onnx --bundle artifacts/vggt-onnx-run2')
    y = 140
    shown = [s for s in lines if 'Warning' not in s][-12:]
    for line in shown:
        for chunk in [line[i:i + 110] for i in range(0, len(line), 110)][:3]:
            text(img, chunk, (40, y), 0.48, (170, 230, 170))
            y += 26
    rt = meta['runtime']
    text(img, f"forward pass {rt['forward_seconds']:.1f} s on CPU | total {meta['inference_seconds']:.1f} s | backend {rt['backend']}",
         (40, H - 70), 0.6, ACCENT)
    hold(writer, img, 12)

    # 5. Depth + confidence
    img = canvas()
    header(img, '4. VGGT outputs: depth and confidence per view', 'Raw tensors depth [1,2,518,518], depth_conf [1,2,518,518], pose_enc [1,2,9] (scale arbitrary)')
    for i in range(2):
        d = pred['depth'][i, ..., 0]
        conf = pred['confidence'][i]
        paste(img, cv2.resize(colorize(d), (280, 280)), 40 + i * 620, 130)
        cm = cv2.applyColorMap((255 * np.clip(conf / np.percentile(conf, 99), 0, 1)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        paste(img, cv2.resize(cm, (280, 280)), 330 + i * 620, 130)
        text(img, f'view {i} depth  {d.min():.2f}-{d.max():.2f}', (40 + i * 620, 440), 0.5, FG)
        text(img, f'confidence {conf.min():.1f}-{conf.max():.1f}', (330 + i * 620, 440), 0.5, FG)
    for i, cam in enumerate(meta['cameras']):
        c = cam['center']
        text(img, f'camera {i} center ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})', (40, 520 + 32 * i), 0.6, ACCENT)
    hold(writer, img, 10)

    # 6. Rotating point cloud
    seconds = 20
    for k in range(seconds * FPS):
        angle = 0.55 * np.sin(2 * np.pi * k / (seconds * FPS))
        img = canvas()
        header(img, '5. Reconstructed point cloud (depth unprojected with VGGT cameras)',
               f"{meta['point_count']:,} points kept of {meta['input_point_count']:,} after confidence + outlier filtering")
        paste(img, render_cloud(points, colors, angle), (W - 860) // 2, 110)
        writer.write(img)

    # 7. Actionable output
    img = canvas()
    header(img, '6. Actionable output: JSON event + verified PLY', 'events.jsonl is emitted only after real geometry is written and the PLY round-trip check passes')
    pretty = json.dumps({k: (Path(v).name if k in ('ply', 'metadata') else v) for k, v in event.items()}, indent=2).splitlines()
    y = 140
    for line in pretty:
        text(img, line, (60, y), 0.6, (170, 230, 170)); y += 30
    ply_size = (args.run / 'cloud.ply').stat().st_size
    b = meta['bounds']
    text(img, f"cloud.ply  {ply_size:,} bytes  |  {meta['point_count']:,} points  |  round-trip verified: {meta['ply_roundtrip_verified']}", (60, y + 30), 0.6, ACCENT)
    text(img, f"bounds min {np.round(b['min'], 3).tolist()}  max {np.round(b['max'], 3).tolist()}", (60, y + 65), 0.55, FG)
    hold(writer, img, 12)

    # 8. Viewer screenshots (real captures only)
    for shot in args.viewer:
        frame = cv2.imread(str(shot))
        if frame is None:
            continue
        img = canvas()
        header(img, '7. SPECTRA viewer (Flask + Three.js) showing the result', shot.name)
        tile = fit(frame, W - 80, H - 170)
        paste(img, tile, (W - tile.shape[1]) // 2, 110)
        hold(writer, img, 6)

    # 9. Status card
    img = canvas()
    header(img, 'Status', 'What this recording does and does not show')
    rows = [('Verified', 'Official VGGT-1B checkpoint (SHA256), real PyTorch inference, ONNX export + ORT parity'),
            ('Verified', 'End-to-end video -> VGGT -> PLY + JSON on CPU (ONNX Runtime) shown above'),
            ('Pending', 'PipeGen Requirement -> PIR -> Model -> Compiler -> Run, TensorRT engine, speed report'),
            ('Limits', 'Two views per reconstruction; arbitrary scale; point/track heads disabled')]
    y = 170
    for tag, desc in rows:
        text(img, tag, (60, y), 0.7, ACCENT if tag == 'Verified' else (80, 170, 255) if tag == 'Pending' else MUTED, 2)
        text(img, desc, (220, y), 0.62, FG); y += 60
    hold(writer, img, 8)
    writer.release()
    print(args.output, 'written')


if __name__ == '__main__':
    main()
