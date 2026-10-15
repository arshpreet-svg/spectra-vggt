"""Compare a real compiled engine against the export's actual PyTorch outputs.

This produces an engine comparison, not a PipeGen speed report.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from spectra.config import Config
from .contract import OUTPUT_NAMES,sha256,verify_bundle
from .runtime import TensorRTEngine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images',nargs=2,type=Path)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--engine',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('evidence/engine-comparison.json'))
    parser.add_argument('--rtol',type=float,default=0.01)
    parser.add_argument('--atol',type=float,default=0.001)
    args=parser.parse_args()
    if args.report.exists():raise ValueError('Comparison report already exists; choose a new --report path to preserve prior evidence')
    manifest=verify_bundle(args.bundle)
    expected_hashes=[item['sha256'] for item in manifest['reference_origin']['images']]
    if [sha256(p) for p in args.images]!=expected_hashes:
        raise ValueError('Use the same two images in the same order as the export reference')
    config=replace(Config(),backend='tensorrt',deployment_dir=args.bundle,engine_path=args.engine)
    engine=TensorRTEngine(config)
    engine.predict(args.images)
    errors={};passed=True
    with np.load(args.bundle/'reference-output.npz') as reference:
        for name in OUTPUT_NAMES:
            expected=reference[name];actual=engine.last_outputs[name]
            agrees=bool(np.allclose(expected,actual,rtol=args.rtol,atol=args.atol))
            passed &= agrees
            errors[name]={'passed':agrees,'max_absolute_error':float(np.max(np.abs(expected-actual)))}
    report={'kind':'standalone_engine_reference_comparison','passed':passed,'rtol':args.rtol,
            'atol':args.atol,'errors':errors,'runtime':engine.runtime,'is_pipegen_speed_report':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
