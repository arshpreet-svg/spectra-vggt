"""Read-only host checks; no drivers, packages, services or accounts changed."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    try:
        run = subprocess.run(args,capture_output=True,text=True,timeout=20)
        return {'exit_code':run.returncode,'stdout':run.stdout.strip(),'stderr':run.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'error':str(exc)}

report={'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
        'disk_free_gb':round(shutil.disk_usage('.').free/1024**3,2),
        'DISPLAY':os.environ.get('DISPLAY'),'session_type':os.environ.get('XDG_SESSION_TYPE'),
        'wsl': 'microsoft' in platform.release().lower(),
        'nvidia_smi':command(['nvidia-smi']),
        'docker':command(['docker','info','--format','{{json .ServerVersion}}']),
        'pipegen_executable':shutil.which('pipegen')}
try:
    import torch
    report['torch']={'version':torch.__version__,'cuda_build':torch.version.cuda,
                     'cuda_available':torch.cuda.is_available(),
                     'mps_available':torch.backends.mps.is_available()}
    if torch.cuda.is_available():
        report['torch'].update(name=torch.cuda.get_device_name(),
          capability=list(torch.cuda.get_device_capability()),compiled_architectures=torch.cuda.get_arch_list())
        # Executes a real CUDA kernel; availability alone does not validate a wheel on Blackwell.
        report['torch']['cuda_kernel_result']=(torch.ones(8,device='cuda')*2).sum().item()
except Exception as exc:report['torch_error']=f'{type(exc).__name__}: {exc}'
Path('evidence').mkdir(exist_ok=True)
Path('evidence/preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
