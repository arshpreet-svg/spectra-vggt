"""Export real pretrained VGGT, verify real-input parity, then write a bundle."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contract import INPUT_NAME, INPUT_SHAPE, OUTPUT_NAMES, OUTPUT_SHAPES, preprocess, sha256
from .wrapper import AtlasVGGT, specialize_fixed_grid


def demote_float64_constants(model_proto, base_dir):
    """Store float64 initializers as float32 so the graph has one floating type.

    Upstream make_sincos_pos_embed multiplies float32 positions by a float64 omega in an
    einsum; PyTorch promotes, but the exporter emits a mixed-type Einsum that ONNX Runtime
    rejects and TensorRT has no float64. Accuracy is still judged by the ONNX Runtime
    comparison against the float64 PyTorch reference below.
    """
    import numpy as np
    from onnx import TensorProto, numpy_helper
    graph=model_proto.graph
    names=[]
    for index,tensor in enumerate(graph.initializer):
        if tensor.data_type==TensorProto.DOUBLE:
            # Small tables; read from the external file if stored there, then keep inline.
            array=numpy_helper.to_array(tensor,base_dir=str(base_dir)).astype(np.float32)
            graph.initializer[index].CopyFrom(numpy_helper.from_array(array,tensor.name))
            names.append(tensor.name)
    if names:
        # Drop stale float64 intermediate annotations; ONNX shape inference rederives them.
        kept=[v for v in graph.value_info if v.type.tensor_type.elem_type!=TensorProto.DOUBLE]
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return names


def export_model(model, atlas, destination, *, reference_origin, rtol=1e-3, atol=1e-4):
    import onnx
    import onnxruntime as ort
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Export destination must be empty; use a new directory to avoid stale models/evidence')
    tensor=torch.from_numpy(atlas).to(next(model.parameters()).device)
    wrapper=AtlasVGGT(model.eval()).eval()
    start=time.perf_counter()
    with torch.inference_mode():
        original=tuple(t.detach().cpu().numpy() for t in wrapper(tensor))
        replaced=specialize_fixed_grid(model)
        wrapper.eval()
        specialized=tuple(t.detach().cpu().numpy() for t in wrapper(tensor))
    for left,right in zip(original,specialized):
        np.testing.assert_allclose(left,right,rtol=1e-6,atol=1e-6)
    np.save(destination/'reference-input.npy',atlas)
    np.savez_compressed(destination/'reference-output.npz',**dict(zip(OUTPUT_NAMES,original)))
    path=destination/'vggt.onnx'
    # Fixed graph; external weights are necessary for the >2 GB FP32 checkpoint.
    torch.onnx.export(wrapper,(tensor,),str(path),input_names=[INPUT_NAME],
                      output_names=OUTPUT_NAMES,opset_version=18,dynamo=True,
                      external_data=True,report=True,artifacts_dir=str(destination/'export-report'))
    # Inline exporter helper functions so the compiler sees standard ONNX nodes.
    from onnx.inliner import inline_local_functions
    graph=inline_local_functions(onnx.load(str(path),load_external_data=False))
    unsupported_domains=sorted({node.domain for node in graph.graph.node if node.domain not in {'','ai.onnx'}})
    if unsupported_domains:raise ValueError(f'Unresolved nonstandard ONNX domains: {unsupported_domains}')
    operators=sorted({node.op_type for node in graph.graph.node})
    demoted=demote_float64_constants(graph,destination)
    onnx.save_model(graph,str(path))
    del graph
    onnx.checker.check_model(str(path))
    # ONNX Runtime loads its own weights; allow host RAM for both model copies.
    del wrapper,tensor
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    options=ort.SessionOptions()
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    if [i.name for i in session.get_inputs()]!=[INPUT_NAME]:
        raise ValueError('Unexpected ONNX input interface')
    actual=session.run(OUTPUT_NAMES,{INPUT_NAME:atlas})
    errors={}
    for name,expected,value in zip(OUTPUT_NAMES,original,actual):
        if value.shape!=OUTPUT_SHAPES[name]:raise ValueError(f'Unexpected output shape: {name} {value.shape}')
        np.testing.assert_allclose(value,expected,rtol=rtol,atol=atol)
        errors[name]={'max_absolute_error':float(np.max(np.abs(value-expected)))}
    files={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'model.json'}
    manifest={'schema_version':1,'model':'VGGT' if reference_origin.get('pretrained') else 'synthetic-adapter-test','reference_origin':reference_origin,
              'input':{'name':INPUT_NAME,'shape':list(INPUT_SHAPE),'dtype':'float32',
                       'layout':'NCHW RGB: first view left, second view right; values [0,1]'},
              'outputs':{name:list(shape) for name,shape in OUTPUT_SHAPES.items()},
              'preprocessing':'upstream load_and_preprocess_images(mode=pad); no external ImageNet normalization',
              'fixed_grid_rope_modules':replaced,'float64_constants_demoted':demoted,'operators':operators,'opset':18,'files':files,
              'validation':{'onnx_reference_passed':True,'specialization_parity_passed':True,
                            'rtol':rtol,'atol':atol,'errors':errors,'seconds':time.perf_counter()-start},
              'pipegen_verified':False,'tensorrt_verified':False}
    (destination/'model.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',nargs=2,type=Path)
    parser.add_argument('--model-id',default='facebook/VGGT-1B',help='Official hub ID or complete local checkpoint directory')
    parser.add_argument('--output',type=Path,default=Path('artifacts/vggt-onnx'))
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    from vggt.models.vggt import VGGT
    atlas=preprocess(args.images)
    # There is deliberately no random-weight or alternate-model fallback.
    model=VGGT.from_pretrained(args.model_id,enable_point=False,enable_track=False).eval().to(args.device)
    reference={'checkpoint':args.model_id,'images':[{'name':p.name,'sha256':sha256(p)} for p in args.images],
               'pretrained':True}
    result=export_model(model,atlas,args.output,reference_origin=reference)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
