"""Python reference pipeline for PipeGen repository analysis."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from spectra.config import Config
from spectra.engine import make_engine
from spectra.reconstruction import reconstruct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--backend',choices=['pytorch','onnx','tensorrt'],default='onnx')
    parser.add_argument('--bundle',type=Path,default=Path('artifacts/vggt-onnx'))
    parser.add_argument('--engine',type=Path,default=Path('artifacts/vggt-onnx/vggt.engine'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/deployment-run'))
    args=parser.parse_args()
    config=replace(Config.from_env(),backend=args.backend,device=args.device,deployment_dir=args.bundle,engine_path=args.engine,max_frames=2)
    metadata=reconstruct(args.inputs,args.output,config,make_engine(config))
    event={'schema_version':1,'event':'geometry_ready','action':'inspect_reconstruction',
           'backend':args.backend,'point_count':metadata['point_count'],'scale':metadata['scale'],
           'ply':str((args.output/'cloud.ply').resolve()),'metadata':str((args.output/'metadata.json').resolve()),
           'pipegen_execution_verified':False}
    (args.output/'events.jsonl').write_text(json.dumps(event)+'\n')
    print(json.dumps(event))

if __name__=='__main__':main()
