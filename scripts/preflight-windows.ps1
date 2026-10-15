# Read-only: run in PowerShell on the actual RTX laptop. Does not require admin.
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force evidence | Out-Null
Start-Transcript -Path evidence/windows-preflight.txt -Force
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free
nvidia-smi
wsl --status
wsl --list --verbose
wsl --exec bash -lc 'uname -a; df -h .; nvidia-smi; echo DISPLAY=$DISPLAY; echo XDG_SESSION_TYPE=$XDG_SESSION_TYPE; command -v pipegen; docker info'
Stop-Transcript
