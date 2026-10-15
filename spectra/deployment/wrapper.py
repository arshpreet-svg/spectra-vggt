import torch
from torch import nn
from .contract import SIZE, VIEWS


class AtlasVGGT(nn.Module):
    """Expose original two-view VGGT through one standard NCHW RGB atlas input."""
    def __init__(self, model):
        super().__init__()
        self.model=model

    def forward(self, images):
        views=torch.stack(torch.split(images,SIZE,dim=-1),dim=1)
        prediction=self.model(views)
        return prediction['pose_enc'],prediction['depth'][...,0],prediction['depth_conf']


class FixedGridRoPE(nn.Module):
    """Exact fixed-grid specialization of upstream RoPE, outside its source tree.

    Upstream computes int(positions.max()) during forward; ONNX capture cannot
    infer that data-dependent size. The atlas fixes the patch grid at export time.
    Its original frequency and rotation helpers are retained without approximation.
    """
    def __init__(self, original, feature_dim, device):
        super().__init__()
        self.original=original
        # Positions are 0 for special tokens, 1..37 for image patches.
        for label,dtype in [('fp32',torch.float32),('fp16',torch.float16),('bf16',torch.bfloat16)]:
            cos,sin=original._compute_frequency_components(feature_dim,SIZE//14+1,device,dtype)
            self.register_buffer(f'cos_{label}',cos)
            self.register_buffer(f'sin_{label}',sin)

    def forward(self,tokens,positions):
        vertical,horizontal=tokens.chunk(2,dim=-1)
        label='fp16' if tokens.dtype==torch.float16 else 'bf16' if tokens.dtype==torch.bfloat16 else 'fp32'
        cos,sin=getattr(self,f'cos_{label}'),getattr(self,f'sin_{label}')
        vertical=self.original._apply_1d_rope(vertical,positions[...,0],cos,sin)
        horizontal=self.original._apply_1d_rope(horizontal,positions[...,1],cos,sin)
        return torch.cat((vertical,horizontal),dim=-1)


def specialize_fixed_grid(model):
    count=0
    for module in list(model.modules()):
        rope=getattr(module,'rope',None)
        if rope is not None and hasattr(module,'head_dim') and hasattr(rope,'_compute_frequency_components'):
            module.rope=FixedGridRoPE(rope,module.head_dim//2,next(model.parameters()).device)
            count+=1
    return count
