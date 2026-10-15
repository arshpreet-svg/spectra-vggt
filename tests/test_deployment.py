"""Adapter tests only. No pretrained VGGT, GPU or PipeGen benchmark claims."""
import json
import numpy as np
import pytest
import torch
from spectra.deployment.contract import SIZE,INPUT_SHAPE,OUTPUT_SHAPES,decode_outputs,pack_views,unpack_views,verify_bundle
from spectra.deployment.wrapper import FixedGridRoPE


def test_atlas_preserves_distinct_view_order():
    views=np.zeros((2,3,SIZE,SIZE),np.float32);views[1]=1
    atlas=pack_views(views)
    assert atlas.shape==INPUT_SHAPE
    assert atlas[0,:,:,SIZE:].min()==1
    np.testing.assert_array_equal(unpack_views(atlas),views)
    with pytest.raises(ValueError,match='exactly'):pack_views(views[:1])


def test_fixed_rope_is_numerically_equivalent_to_upstream():
    from vggt.layers.rope import RotaryPositionEmbedding2D
    original=RotaryPositionEmbedding2D()
    fixed=FixedGridRoPE(original,32,torch.device('cpu'))
    generator=torch.Generator().manual_seed(123)
    tokens=torch.randn((2,2,19,64),generator=generator)
    positions=torch.randint(0,38,(2,19,2),generator=generator)
    for dtype in [torch.float32,torch.float16]:
        torch.testing.assert_close(fixed(tokens.to(dtype),positions),original(tokens.to(dtype),positions),rtol=0,atol=0)


def test_decode_known_camera_and_depth():
    atlas=np.zeros(INPUT_SHAPE,np.float32)
    pose=np.zeros(OUTPUT_SHAPES['pose_enc'],np.float32)
    pose[:,:,6]=1 # upstream XYZW quaternion identity
    pose[:,:,7:]=np.pi/2
    pose[:,1,0]=1 # camera-from-world translation +1 => center -1
    raw={'pose_enc':pose,'depth':np.ones(OUTPUT_SHAPES['depth'],np.float32),
         'depth_conf':np.ones(OUTPUT_SHAPES['depth_conf'],np.float32)}
    decoded=decode_outputs(raw,atlas)
    np.testing.assert_allclose(decoded['intrinsic'][0,0,0],SIZE/2,rtol=1e-6)
    np.testing.assert_allclose(decoded['points'][1,SIZE//2,SIZE//2],[-1,0,1],atol=1e-6)
    raw['pose_enc'][0,0,3:7]=0
    with pytest.raises(ValueError,match='quaternion'):decode_outputs(raw,atlas)


def test_onnx_export_roundtrip_with_labeled_synthetic_model(tmp_path):
    from spectra.deployment.export import export_model
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.tensor(.1))
            from vggt.layers.attention import Attention
            from vggt.layers.rope import RotaryPositionEmbedding2D
            self.embed=torch.nn.Linear(3,32)
            self.attn=Attention(32,num_heads=4,rope=RotaryPositionEmbedding2D(),qk_norm=True)
            self.register_buffer('positions',torch.tensor([[[0,0],[1,1]]]))
            self.register_buffer('pose',torch.tensor([0.,0.,0.,0.,0.,0.,1.,1.,1.]).reshape(1,1,9).repeat(1,2,1))
        def forward(self,images):
            offset=self.attn(self.embed(images.mean(dim=(-2,-1))),pos=self.positions).mean(dim=-1)
            depth=images.mean(dim=2)+self.bias+offset[...,None,None]
            return {'pose_enc':self.pose,'depth':depth[...,None],'depth_conf':depth+1}
    rng=np.random.default_rng(42)
    atlas=rng.random(INPUT_SHAPE,dtype=np.float32)
    report=export_model(Fixture(),atlas,tmp_path,reference_origin={'pretrained':False,'kind':'synthetic-contract-test'})
    assert report['validation']['onnx_reference_passed']
    assert report['pipegen_verified'] is False
    assert report['reference_origin']['pretrained'] is False
    with pytest.raises(ValueError,match='pretrained'):verify_bundle(tmp_path)
    with pytest.raises(ValueError,match='empty'):export_model(Fixture(),atlas,tmp_path,reference_origin={})


def test_backend_selection_never_falls_back(tmp_path):
    from spectra.config import Config
    from spectra.engine import make_engine
    from spectra.deployment.runtime import ONNXEngine,TensorRTEngine
    assert isinstance(make_engine(Config(backend='onnx')),ONNXEngine)
    assert isinstance(make_engine(Config(backend='tensorrt')),TensorRTEngine)
    with pytest.raises(ValueError,match='backend'):Config(backend='pipegen-fake')
