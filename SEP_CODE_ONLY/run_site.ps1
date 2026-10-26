param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$BatchPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$CodeRoot = $PSScriptRoot

if (-not $BatchPath) {
    throw "BatchPath gerekli. Ornek: -BatchPath .\PhantomX\REGISTRATION\batch_to_20260805_v7"
}
$BatchPath = (Resolve-Path $BatchPath).Path

$env:SEP_LAB = Join-Path $ProjectRoot "PhantomX\LABELED"
$env:SEP_BATCH = $BatchPath
$env:SEP_SESSION_SUFFIX = "PCCT_Aorta_Spleen"
$env:SEP_FIXED_DAY = "20260805"
$env:SEP_FIXED_BASE = "USZ_Siemens_Healthineers_NAEOTOM_Alpha_20260805_Q3_Qr40f_120_100_300_1_PolyE_take1"
$env:SEP_FIXED_LABEL = "20260805_t1"
$env:SEP_FEATURED = "0"

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$generator = Join-Path $CodeRoot "registration_overlay_ui\generate_all_assets.py"

if (-not (Test-Path $python)) {
    throw "Python bulunamadi: $python"
}
if (-not (Test-Path $env:SEP_LAB)) {
    throw "LABELED klasoru bulunamadi: $env:SEP_LAB"
}
if (-not (Test-Path $env:SEP_BATCH)) {
    throw "Batch klasoru bulunamadi: $env:SEP_BATCH"
}
if (-not (Test-Path $generator)) {
    throw "Generator bulunamadi: $generator"
}

& $python $generator
if ($LASTEXITCODE -ne 0) {
    throw "Site generation failed: $LASTEXITCODE"
}

Write-Host ""
Write-Host "Generated: $CodeRoot\registration_overlay_ui"
Write-Host "Static upload folder: $CodeRoot\registration_overlay_ui"
