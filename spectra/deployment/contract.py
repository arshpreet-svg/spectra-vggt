from pathlib import Path
import hashlib
import json
import numpy as np

SIZE = 518
VIEWS = 2
INPUT_NAME = 'images'
OUTPUT_NAMES = ['pose_enc', 'depth', 'depth_conf']
INPUT_SHAPE = (1, 3, SIZE, SIZE * VIEWS)
OUTPUT_SHAPES = {'pose_enc': (1,VIEWS,9), 'depth': (1,VIEWS,SIZE,SIZE),
                 'depth_conf': (1,VIEWS,SIZE,SIZE)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pack_views(views):
    views = np.asarray(views, dtype=np.float32)
    if views.shape != (VIEWS,3,SIZE,SIZE):
        raise ValueError(f'Expected exactly {VIEWS} preprocessed RGB views at {SIZE}x{SIZE}; got {views.shape}')
    if not np.isfinite(views).all() or views.min()<0 or views.max()>1:
        raise ValueError('RGB values must be finite in [0,1]')
    return np.ascontiguousarray(np.concatenate(list(views),axis=-1)[None])


def unpack_views(atlas):
    atlas = np.asarray(atlas)
    if atlas.shape != INPUT_SHAPE:
        raise ValueError(f'Expected atlas shape {INPUT_SHAPE}; got {atlas.shape}')
    return np.stack(np.split(atlas[0],VIEWS,axis=-1))


def preprocess(paths):
    from vggt.utils.load_fn import load_and_preprocess_images
    if len(paths)!=VIEWS:
        raise ValueError('The deployment model requires exactly two overlapping views')
    views=load_and_preprocess_images([str(p) for p in paths],mode='pad').numpy()
    return pack_views(views)


def decode_outputs(outputs, atlas):
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    checked={}
    for name,shape in OUTPUT_SHAPES.items():
        value=np.asarray(outputs[name],dtype=np.float32)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid {name}: expected finite tensor {shape}, got {value.shape}')
        checked[name]=value
    poses=torch.from_numpy(checked['pose_enc'])
    if (np.linalg.norm(checked['pose_enc'][...,3:7],axis=-1)<1e-8).any():
        raise ValueError('VGGT predicted a degenerate camera quaternion')
    with torch.inference_mode():
        extrinsic,intrinsic=pose_encoding_to_extri_intri(poses,(SIZE,SIZE))
    extrinsic,intrinsic=extrinsic[0].numpy(),intrinsic[0].numpy()
    if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
        raise ValueError('Invalid camera matrices')
    depth=checked['depth'][0,...,None]
    return {'depth':depth,'confidence':checked['depth_conf'][0],
            'points':unproject_depth_map_to_point_map(depth,extrinsic,intrinsic),
            'extrinsic':extrinsic,'intrinsic':intrinsic,
            'colors':unpack_views(atlas).transpose(0,2,3,1)}


def verify_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'model.json').read_text())
    if manifest.get('reference_origin',{}).get('pretrained') is not True:
        raise ValueError('A real pretrained model bundle is required; synthetic adapter tests are not deployable VGGT')
    if manifest.get('input',{}).get('shape')!=list(INPUT_SHAPE):
        raise ValueError('Model manifest has an incompatible input shape')
    if manifest.get('validation',{}).get('onnx_reference_passed') is not True:
        raise ValueError('Model bundle has no successful real-input ONNX comparison')
    for name,digest in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256(path)!=digest:
            raise ValueError(f'Missing or mismatched model artifact: {name}')
    return manifest
