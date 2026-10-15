#!/usr/bin/env bash
# Run SPECTRA using the actual compiled engine. No PyTorch inference fallback.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash pipegen/serve_engine.sh VALIDATED_BUNDLE ACTUAL_ENGINE" >&2
  exit 2
fi
export SPECTRA_BACKEND=tensorrt
export SPECTRA_DEPLOYMENT_DIR="$1"
export SPECTRA_ENGINE_PATH="$2"
export SPECTRA_MAX_FRAMES=2
python -m spectra.app
