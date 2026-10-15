#!/usr/bin/env bash
# Independently authored reference launcher. This is not PipeGen's generated run.sh.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: bash pipegen/run_reference.sh VIDEO BUNDLE ENGINE OUTPUT" >&2
  exit 2
fi
python -m spectra.deployment.deepstream --video "$1" --bundle "$2" --engine "$3" --output "$4"
