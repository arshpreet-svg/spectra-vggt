"""Contract tests use synthetic predictions explicitly, never benchmark evidence."""
import io
import json
import time
from dataclasses import replace
import cv2
import numpy as np
import open3d as o3d
import pytest
from PIL import Image
from spectra.app import create_app
from spectra.config import Config
from spectra.detections import validate_event
from spectra.inputs import prepare_frames
from spectra.reconstruction import export_predictions


def predictions(count=1):
    y,x=np.mgrid[:12,:12]
    points=np.stack([x*.02,y*.02,np.ones_like(x)],axis=-1).astype(float)
    return {'points':np.repeat(points[None],count,axis=0),
            'colors':np.ones((count,12,12,3))*.5,'depth':np.ones((count,12,12,1)),
            'confidence':np.ones((count,12,12)),
            'extrinsic':np.repeat(np.eye(4)[None,:3],count,axis=0),
            'intrinsic':np.repeat(np.eye(3)[None],count,axis=0)}


class FixtureEngine:
    runtime={'device':'synthetic-test-fixture'}
    def predict(self, paths):
        return predictions(len(paths))


def test_export_roundtrip_filters_invalid(tmp_path):
    config=Config(output_dir=tmp_path,voxel_size=0)
    data=predictions()
    data['points'][0,0,0,0]=np.nan
    data['depth'][0,0,1,0]=-1
    metadata=export_predictions(data,tmp_path,config,[{'image':'a.png'}],{'device':'fixture'},0)
    cloud=o3d.io.read_point_cloud(str(tmp_path/'cloud.ply'))
    assert 0<len(cloud.points)<=142
    assert np.isfinite(np.asarray(cloud.points)).all()
    assert len(cloud.points)==metadata['point_count']
    assert json.loads((tmp_path/'metadata.json').read_text())['scale']=='arbitrary; uncalibrated'
    with np.load(tmp_path/'predictions.npz') as saved:
        assert saved['intrinsic'].shape==(1,3,3)


def test_empty_geometry_rejected(tmp_path):
    data=predictions();data['depth'][:]=0
    with pytest.raises(ValueError,match='positive-depth'):
        export_predictions(data,tmp_path,Config(),[],{},0)


def test_video_sampling(tmp_path):
    path=tmp_path/'clip.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for i in range(20):writer.write(np.full((32,32,3),i*10,dtype=np.uint8))
    writer.release()
    paths,records=prepare_frames([path],tmp_path/'frames',Config())
    assert len(paths)==2
    assert [r['frame_index'] for r in records]==[0,10]
    assert records[1]['timestamp_seconds']==1.0


def event():
    return {'schema_version':1,'producer':'pipegen','frame_index':0,'timestamp_seconds':0.,
            'image_size':[100,100],'detections':[{'label':'chair','confidence':.8,'bbox_xyxy':[1,2,50,60]}]}


def test_adapter_rejects_bad_boxes():
    assert validate_event(event())['detections'][0]['label']=='chair'
    for bad in [float('nan'),101,-1]:
        data=event();data['detections'][0]['bbox_xyxy'][2]=bad
        with pytest.raises(ValueError):validate_event(data)


def test_api_upload_export_and_adapter(tmp_path):
    app=create_app(Config(output_dir=tmp_path),FixtureEngine())
    client=app.test_client()
    try:
        assert client.get('/').status_code==200
        assert client.get('/static/vendor/three.module.js').status_code==200
        assert client.get('/api/health').json['pipegen']['verified'] is False
        assert client.post('/api/reconstruct').status_code==400
        image=io.BytesIO();Image.new('RGB',(32,32)).save(image,format='PNG');image.seek(0)
        response=client.post('/api/reconstruct',data={'files':(image,'view.png')})
        assert response.status_code==202
        job=response.json['id']
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            state=client.get(f'/api/jobs/{job}').json
            if state['status'] in {'complete','failed'}:break
            time.sleep(.02)
        assert state['status']=='complete',state
        assert client.get(f'/api/jobs/{job}/cloud.ply').data.startswith(b'ply\n')
        assert client.get(f'/api/jobs/{job}/predictions.npz').status_code==200
        assert client.get(f'/api/jobs/{job}/job.json').status_code==404
        assert client.post(f'/api/jobs/{job}/detections',json=event()).status_code==202
        assert len(client.get(f'/api/jobs/{job}/detections').json['events'])==1
        assert client.get(f'/api/jobs/{job}/inspection').json['action']=='clear_observed_frame'
        assert client.get(f'/api/jobs/{job}/detections').json['pipegen_execution_verified'] is False
    finally:
        app.extensions['spectra_executor'].shutdown(wait=True)


def test_failure_exposes_error_and_releases_slot(tmp_path):
    class Broken:
        def predict(self,paths):raise RuntimeError('checkpoint unavailable')
    app=create_app(Config(output_dir=tmp_path),Broken());client=app.test_client()
    try:
        for _ in range(2):
            image=io.BytesIO();Image.new('RGB',(32,32)).save(image,format='PNG');image.seek(0)
            response=client.post('/api/reconstruct',data={'files':(image,'view.png')})
            assert response.status_code==202
            job=response.json['id']
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                state=client.get(f'/api/jobs/{job}').json
                if state['status']=='failed':break
                time.sleep(.02)
            assert 'checkpoint unavailable' in state['error']
    finally:app.extensions['spectra_executor'].shutdown(wait=True)


def test_scan_advisory():
    from spectra.inspection import inspect_events
    assert inspect_events([])['action']=='await_detector'
    data=event();data['detections'][0]['label']='person'
    assert inspect_events([data])['action']=='pause_scan'
    data['detections'][0]['confidence']=.1
    assert inspect_events([data])['action']=='clear_observed_frame'
