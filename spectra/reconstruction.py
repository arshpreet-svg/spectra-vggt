import json
import time
import numpy as np
import open3d as o3d
from .inputs import prepare_frames


def export_predictions(predictions, destination, config, frames, runtime, elapsed):
    destination.mkdir(parents=True, exist_ok=True)
    points = np.asarray(predictions['points']).reshape(-1, 3)
    colors = np.asarray(predictions['colors']).reshape(-1, 3)
    confidence = np.asarray(predictions['confidence']).reshape(-1)
    depth = np.asarray(predictions['depth']).reshape(-1)
    if not (len(points) == len(colors) == len(confidence) == len(depth)):
        raise ValueError('Prediction dimensions do not match')
    valid = (np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
             & np.isfinite(confidence) & np.isfinite(depth) & (depth > 0))
    if not valid.any():
        raise ValueError('VGGT returned no finite positive-depth points')
    threshold = float(np.percentile(confidence[valid], config.confidence_percentile))
    valid &= confidence >= threshold
    indices = np.flatnonzero(valid)
    if len(indices) > config.max_points:
        indices = indices[np.linspace(0, len(indices) - 1, config.max_points, dtype=int)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[indices])
    cloud.colors = o3d.utility.Vector3dVector(np.clip(colors[indices], 0, 1))
    if config.voxel_size > 0:
        cloud = cloud.voxel_down_sample(config.voxel_size)
    if len(cloud.points) > config.outlier_neighbors + 1:
        cloud, _ = cloud.remove_statistical_outlier(config.outlier_neighbors, config.outlier_std_ratio)
    if not len(cloud.points):
        raise ValueError('No points survived filtering; use a better overlapping image sequence')
    ply = destination / 'cloud.ply'
    if not o3d.io.write_point_cloud(str(ply), cloud, write_ascii=False):
        raise RuntimeError('Open3D failed to write PLY')
    check = o3d.io.read_point_cloud(str(ply))
    if len(check.points) != len(cloud.points) or not np.isfinite(np.asarray(check.points)).all():
        raise RuntimeError('PLY round-trip verification failed')
    np.savez_compressed(destination / 'predictions.npz', **predictions)
    cameras = []
    for frame, ext, intr in zip(frames, predictions['extrinsic'], predictions['intrinsic']):
        transform = np.eye(4)
        transform[:3, :] = ext
        world_from_camera = np.linalg.inv(transform)
        cameras.append({**frame, 'camera_from_world': ext.tolist(), 'intrinsic': intr.tolist(),
                        'world_from_camera': world_from_camera.tolist(),
                        'center': world_from_camera[:3, 3].tolist()})
    metadata = {'schema_version': 1, 'engine': 'VGGT', 'scale': 'arbitrary; uncalibrated',
                'coordinate_convention': 'OpenCV camera: +X right, +Y down, +Z forward; extrinsics camera_from_world',
                'point_source': 'VGGT depth unprojected with VGGT intrinsics/extrinsics',
                'point_count': len(cloud.points), 'input_point_count': len(points),
                'confidence_threshold': threshold, 'cameras': cameras, 'runtime': runtime,
                'inference_seconds': elapsed, 'config': config.json(),
                'ply_roundtrip_verified': True,
                'bounds': {'min': cloud.get_min_bound().tolist(), 'max': cloud.get_max_bound().tolist()},
                'pipegen': {'connected': False, 'verified': False}}
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return metadata


def reconstruct(sources, destination, config, engine):
    paths, records = prepare_frames(sources, destination / 'frames', config)
    start = time.perf_counter()
    predictions = engine.predict(paths)
    return export_predictions(predictions, destination, config, records, engine.runtime, time.perf_counter() - start)
