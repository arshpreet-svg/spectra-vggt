# Real VGGT load + inference + ONNX export/parity on this Windows machine, with full logging.
# Refuses to run unless scripts/fetch_checkpoint.py wrote VERIFIED_SHA256 with the expected hash.
param([string]$Output = 'artifacts/vggt-onnx-run1')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$expected = 'f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e'
$ckpt = 'artifacts/checkpoints/VGGT-1B'
if (-not (Test-Path "$ckpt/VERIFIED_SHA256") -or (Get-Content "$ckpt/VERIFIED_SHA256").Trim() -ne $expected) {
    throw 'Checkpoint not verified; refusing to load'
}
if ((Test-Path $Output) -and (Get-ChildItem $Output -Force | Select-Object -First 1)) {
    throw "$Output is not empty; choose a new output directory"
}
$log = 'deliverables/evidence/onnx_validation.txt'
$cmd = ".venv\Scripts\python.exe -m spectra.deployment.export vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png --model-id $ckpt --output $Output --device cpu"
"command: $cmd" | Out-File -Encoding utf8 $log
"started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Out-File -Encoding utf8 -Append $log
$sw = [Diagnostics.Stopwatch]::StartNew()
$ErrorActionPreference = 'Continue'
& .venv\Scripts\python.exe -m spectra.deployment.export vggt/examples/kitchen/images/00.png vggt/examples/kitchen/images/01.png --model-id $ckpt --output $Output --device cpu 2>&1 |
    ForEach-Object { "$_" } | Out-File -Encoding utf8 -Append $log
$code = $LASTEXITCODE
"exit code: $code; elapsed: $([int]$sw.Elapsed.TotalSeconds) s; finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')" | Out-File -Encoding utf8 -Append $log
"EXPORT_EXIT $code"
