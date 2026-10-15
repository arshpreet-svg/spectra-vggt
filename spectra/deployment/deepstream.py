"""DeepStream reference app for the custom VGGT atlas model.

Run inside the target environment with NVIDIA GStreamer plugins and matching pyds.
This is application source for PipeGen to inspect/adapt, not a PipeGen-generated export.
"""
import argparse
import ctypes
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
from spectra.config import Config
from spectra.inputs import prepare_frames
from spectra.reconstruction import export_predictions
from .contract import SIZE,VIEWS,OUTPUT_NAMES,OUTPUT_SHAPES,decode_outputs,preprocess,sha256,verify_bundle


def nvinfer_config(bundle, engine):
    return f'''[property]
gpu-id=0
model-engine-file={Path(engine).resolve()}
batch-size=1
network-type=100
process-mode=1
network-mode=0
model-color-format=0
net-scale-factor=0.00392156862745098
infer-dims=3;{SIZE};{SIZE*VIEWS}
output-blob-names=pose_enc;depth;depth_conf
output-tensor-meta=1
gie-unique-id=1
interval=0
maintain-aspect-ratio=0
'''


def read_tensor_meta(meta,pyds):
    result={}
    for index in range(meta.num_output_layers):
        layer=pyds.get_nvds_LayerInfo(meta,index)
        name=layer.layerName
        if name not in OUTPUT_SHAPES:continue
        shape=tuple(int(layer.inferDims.d[i]) for i in range(layer.inferDims.numDims))
        count=int(np.prod(shape))
        if count!=int(np.prod(OUTPUT_SHAPES[name])):
            raise ValueError(f'DeepStream output {name} has incompatible shape {shape}')
        if layer.dataType==pyds.NvDsInferDataType.FLOAT:
            pointer=ctypes.cast(pyds.get_ptr(layer.buffer),ctypes.POINTER(ctypes.c_float))
            value=np.ctypeslib.as_array(pointer,shape=(count,)).copy()
        elif layer.dataType==pyds.NvDsInferDataType.HALF:
            pointer=ctypes.cast(pyds.get_ptr(layer.buffer),ctypes.POINTER(ctypes.c_uint16))
            value=np.ctypeslib.as_array(pointer,shape=(count,)).copy().view(np.float16).astype(np.float32)
        else:raise ValueError(f'Expected floating tensor metadata for {name}')
        result[name]=value.reshape(OUTPUT_SHAPES[name])
    if set(result)!=set(OUTPUT_NAMES):
        raise ValueError(f'Missing raw VGGT outputs: {set(OUTPUT_NAMES)-set(result)}')
    return result


def run(video,bundle,engine,output):
    import gi
    gi.require_version('Gst','1.0')
    from gi.repository import Gst,GLib
    import pyds
    manifest=verify_bundle(bundle)
    engine=Path(engine)
    if not engine.is_file():raise FileNotFoundError('A genuinely compiled TensorRT engine is required')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use an empty run output directory')
    output.mkdir(parents=True,exist_ok=True)
    config=replace(Config(),max_frames=2)
    paths,records=prepare_frames([Path(video)],output/'frames',config)
    atlas=preprocess(paths)
    # Roundtrip is exact: upstream pad/preprocess emits uint8 pixels divided by 255.
    rgb=np.rint(atlas[0].transpose(1,2,0)*255).astype(np.uint8)
    rgba=np.concatenate([rgb,np.full((*rgb.shape[:2],1),255,np.uint8)],axis=-1)
    (output/'nvinfer.ini').write_text(nvinfer_config(bundle,engine))
    Gst.init(None)
    pipeline=Gst.Pipeline.new('spectra-vggt')
    def element(factory,name):
        value=Gst.ElementFactory.make(factory,name)
        if value is None:raise RuntimeError(f'Missing GStreamer plugin: {factory}')
        pipeline.add(value)
        return value
    source=element('appsrc','atlas_source')
    convert=element('nvvideoconvert','gpu_upload')
    caps=element('capsfilter','gpu_caps')
    mux=element('nvstreammux','batch')
    infer=element('nvinfer','vggt')
    sink=element('fakesink','sink')
    source.set_property('caps',Gst.Caps.from_string(f'video/x-raw,format=RGBA,width={SIZE*VIEWS},height={SIZE},framerate=1/1'))
    source.set_property('format',Gst.Format.TIME)
    source.set_property('is-live',False)
    caps.set_property('caps',Gst.Caps.from_string('video/x-raw(memory:NVMM),format=RGBA'))
    mux.set_property('batch-size',1)
    for name,value in [('width',SIZE*VIEWS),('height',SIZE),('batched-push-timeout',40000)]:
        if mux.find_property(name):mux.set_property(name,value)
    infer.set_property('config-file-path',str((output/'nvinfer.ini').resolve()))
    sink.set_property('sync',False)
    if not source.link(convert) or not convert.link(caps):raise RuntimeError('Could not link atlas upload')
    sinkpad=mux.request_pad_simple('sink_0')
    if sinkpad is None or caps.get_static_pad('src').link(sinkpad)!=Gst.PadLinkReturn.OK:
        raise RuntimeError('Could not link atlas to stream mux')
    if not mux.link(infer) or not infer.link(sink):raise RuntimeError('Could not link VGGT inference')
    loop=GLib.MainLoop()
    state={'outputs':None,'error':None,'eos':False}
    def probe(pad,info,user):
        try:
            batch=pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
            if batch is None:raise RuntimeError('DeepStream returned no batch metadata')
            frames=batch.frame_meta_list
            while frames:
                frame=pyds.NvDsFrameMeta.cast(frames.data)
                items=frame.frame_user_meta_list
                while items:
                    item=pyds.NvDsUserMeta.cast(items.data)
                    if item.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                        state['outputs']=read_tensor_meta(pyds.NvDsInferTensorMeta.cast(item.user_meta_data),pyds)
                    items=items.next
                frames=frames.next
        except Exception as exc:
            state['error']=f'{type(exc).__name__}: {exc}'
            loop.quit()
        return Gst.PadProbeReturn.OK
    infer.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER,probe,None)
    def message(bus,msg):
        if msg.type==Gst.MessageType.ERROR:
            error,debug=msg.parse_error();state['error']=f'{error}: {debug}';loop.quit()
        elif msg.type==Gst.MessageType.EOS:
            state['eos']=True;loop.quit()
    bus=pipeline.get_bus();bus.add_signal_watch();bus.connect('message',message)
    timeout=GLib.timeout_add_seconds(300,lambda: (state.update(error='DeepStream run timed out'),loop.quit(),False)[-1])
    start=time.perf_counter()
    try:
        if pipeline.set_state(Gst.State.PLAYING)==Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('DeepStream could not enter PLAYING state')
        buffer=Gst.Buffer.new_allocate(None,rgba.nbytes,None)
        buffer.fill(0,rgba.tobytes());buffer.pts=0;buffer.duration=Gst.SECOND
        if source.emit('push-buffer',buffer)!=Gst.FlowReturn.OK:
            raise RuntimeError('DeepStream appsrc rejected the atlas')
        source.emit('end-of-stream')
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        mux.release_request_pad(sinkpad)
        bus.remove_signal_watch()
        if GLib.MainContext.default().find_source_by_id(timeout):GLib.source_remove(timeout)
    if state['error']:raise RuntimeError(state['error'])
    if not state['eos'] or state['outputs'] is None:
        raise RuntimeError('No genuine raw VGGT tensor metadata was received')
    elapsed=time.perf_counter()-start
    np.savez_compressed(output/'raw-outputs.npz',**state['outputs'])
    decoded=decode_outputs(state['outputs'],atlas)
    runtime={'backend':'deepstream-nvinfer','device':'cuda','engine_sha256':sha256(engine),
             'model_sha256':manifest['files']['vggt.onnx'],'input_sha256':sha256(video),
             'pipegen_verified':False,'note':'Reference application run; PipeGen workflow/report provenance must be supplied separately'}
    metadata=export_predictions(decoded,output,config,records,runtime,elapsed)
    event={'schema_version':1,'event':'geometry_ready','action':'inspect_reconstruction',
           'point_count':metadata['point_count'],'scale':metadata['scale'],
           'metadata':str((output/'metadata.json').resolve()),'ply':str((output/'cloud.ply').resolve()),
           'engine_sha256':runtime['engine_sha256'],'source_sha256':runtime['input_sha256'],
           'pipegen_execution_verified':False}
    (output/'events.jsonl').write_text(json.dumps(event)+'\n')
    return event


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--engine',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.video,args.bundle,args.engine,args.output),indent=2))

if __name__=='__main__':main()
