# ASR Shootout — Complete automated benchmark pipeline (Windows PowerShell)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         ASR SHOOTOUT - AUTOMATIC BENCHMARK PIPELINE        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check Python
try {
    $pythonVersion = python --version 2>&1
    $pythonCmd = "python"
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

Write-Host "[*] Using Python: $pythonVersion" -ForegroundColor Green

# Activate virtual environment if it exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    Write-Host "[*] Activating virtual environment..." -ForegroundColor Green
    & ".\.venv\Scripts\Activate.ps1"
}

# Install dependencies if needed
Write-Host "[*] Checking dependencies..." -ForegroundColor Green
try {
    python -c "import librosa" 2>$null
} catch {
    Write-Host "[*] Installing dependencies..." -ForegroundColor Green
    python -m pip install -q -r requirements.txt
}

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "Running: Automated Ingestion, Validation & Benchmark" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Run the full benchmark which includes ingestion
python -m src.main run `
    --data-dir data `
    --output-dir data/outputs `
    --report-path reports/final_report.md

Write-Host ""
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  [OK] Metadata:  data/metadata/ground_truth.csv" -ForegroundColor Green
Write-Host "  [OK] Validation: data/outputs/validation_report.csv" -ForegroundColor Green
Write-Host "  [OK] Dataset Summary: data/outputs/dataset_summary.json" -ForegroundColor Green
Write-Host "  [OK] Results: data/outputs/metrics/full_results.csv" -ForegroundColor Green
Write-Host "  [OK] Charts: data/outputs/charts/" -ForegroundColor Green
Write-Host "  [OK] Report: reports/final_report.md" -ForegroundColor Green
Write-Host ""
Write-Host "View the dashboard:" -ForegroundColor Cyan
Write-Host "  streamlit run dashboard/app.py --server.port 8501" -ForegroundColor Cyan
Write-Host ""
