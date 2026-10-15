"""Relay genuine detector JSONL to an existing SPECTRA job; no fabricated events."""
import argparse
import json
from pathlib import Path
from urllib.request import Request,urlopen
from spectra.detections import validate_event

parser=argparse.ArgumentParser()
parser.add_argument('jsonl',type=Path)
parser.add_argument('--job',required=True)
parser.add_argument('--server',default='http://127.0.0.1:5000')
args=parser.parse_args()
for index,line in enumerate(args.jsonl.read_text().splitlines(),1):
    if not line.strip():continue
    event=validate_event(json.loads(line))
    request=Request(f'{args.server}/api/jobs/{args.job}/detections',
                    data=json.dumps(event,allow_nan=False).encode(),
                    headers={'Content-Type':'application/json'},method='POST')
    with urlopen(request,timeout=20) as response:
        print(json.dumps({'line':index,'response':json.loads(response.read())}))
