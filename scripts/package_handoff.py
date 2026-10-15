"""Package source and a few real-scene examples; omit weights/runtime/Git history."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED


def main():
    files=[]
    for directory in ['spectra','scripts','docs','tests','pipegen']:
        files.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.extend(Path(x) for x in ['README.md','requirements-spectra.txt','requirements-export.txt','.gitignore'])
    for p in Path('evidence').iterdir():
        if p.is_file() and p.suffix in {'.json','.txt','.log'}:files.append(p)
    for p in Path('vggt').rglob('*'):
        if p.is_file() and not {'.git','examples','__pycache__'}.intersection(p.parts):
            if p.suffix in {'.py','.toml','.txt','.md','.yaml','.yml','.json'}:files.append(p)
    files.extend(sorted(Path('vggt/examples/kitchen/images').glob('*.png'))[:8])
    video=Path('artifacts/kitchen-sample.mp4')
    if video.exists():files.append(video)
    with ZipFile('spectra-handoff.zip','w',ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,Path('spectra-handoff')/p)
        archive.writestr('spectra-handoff/HANDOFF.txt',
            'Source snapshot, not a PipeGen-generated project.\n'
            'VGGT commit a288dd0f14786c93483e45524328726ab7b1b4ce; eight kitchen images included.\n'
            'Weights, virtualenv, other datasets and Git history omitted.\n'
            'Read docs/VGGT_PIPEGEN.md; full-model/GPU/PipeGen verification remains pending.\n')
    print(f'spectra-handoff.zip: {len(files)} source/sample/evidence files, {Path("spectra-handoff.zip").stat().st_size:,} bytes')

if __name__=='__main__':main()
