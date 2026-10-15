"""Download the official VGGT-1B safetensors checkpoint and verify its SHA256 before use."""
import hashlib
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

EXPECTED_SHA256 = 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
REPO = 'facebook/VGGT-1B'


def main(target=Path('artifacts/checkpoints/VGGT-1B')):
    target.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'model.safetensors'):
        print('fetching', name, flush=True)
        hf_hub_download(REPO, name, local_dir=target)
    digest = hashlib.sha256()
    with open(target / 'model.safetensors', 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    actual = digest.hexdigest()
    print('size', (target / 'model.safetensors').stat().st_size, 'sha256', actual, flush=True)
    if actual != EXPECTED_SHA256:
        sys.exit('CHECKPOINT HASH MISMATCH - do not load')
    (target / 'VERIFIED_SHA256').write_text(actual + '\n')
    print('CHECKPOINT VERIFIED', target.resolve())


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('artifacts/checkpoints/VGGT-1B'))
