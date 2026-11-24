# Rebuilds HydroGEN.exe from hbond_gui.py / hbond_core.py / bv_params.txt / HydroGEN.ico.
# (The icon is drawn by make_icon.py - run it first if you change the drawing.)
# Run from this folder:   powershell -ExecutionPolicy Bypass -File .\build_exe.ps1
# Uses the separate build environment in .\build-env (PyInstaller + numpy), not Anaconda.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\build-env\Scripts\python.exe)) {
    & "$env:USERPROFILE\anaconda3\python.exe" -m venv build-env
    .\build-env\Scripts\python.exe -m pip install --disable-pip-version-check pyinstaller "numpy<2.1" certifi
}
# certifi: HTTPS certificates for Update (Python's OpenSSL cannot verify GitHub from the Windows store alone)
.\build-env\Scripts\python.exe -c "import importlib.util as u, sys; sys.exit(u.find_spec('certifi') is None)"
if ($LASTEXITCODE -ne 0) { .\build-env\Scripts\python.exe -m pip install --disable-pip-version-check certifi }

.\build-env\Scripts\python.exe -m PyInstaller `
    --noconfirm --clean --onefile --windowed `
    --name HydroGEN `
    --add-data "$PSScriptRoot\bv_params.txt;." `
    --add-data "$PSScriptRoot\HydroGEN.ico;." `
    --add-data "$PSScriptRoot\HydroGEN_User_Guide.txt;." `
    --icon "$PSScriptRoot\HydroGEN.ico" `
    --exclude-module pypdf `
    --hidden-import certifi `
    --distpath . --workpath build --specpath build `
    hbond_gui.py

Write-Host "Built $(Join-Path $PSScriptRoot 'HydroGEN.exe')"
