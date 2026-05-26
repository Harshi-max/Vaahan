#!/bin/bash
# ASR Shootout — Complete automated benchmark pipeline

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         ASR SHOOTOUT - AUTOMATIC BENCHMARK PIPELINE        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "[*] Using Python: $($PYTHON_CMD --version)"

# Create/activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "[*] Activating virtual environment..."
    source .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate 2>/dev/null || true
fi

# Install dependencies if needed
if ! $PYTHON_CMD -c "import librosa" 2>/dev/null; then
    echo "[*] Installing dependencies..."
    $PYTHON_CMD -m pip install -q -r requirements.txt
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Step 1: Automated Ingestion & Validation"
echo "════════════════════════════════════════════════════════════"
echo ""

# Run the full benchmark which includes ingestion
$PYTHON_CMD -m src.main run \
    --data-dir data \
    --output-dir data/outputs \
    --report-path reports/final_report.md

echo ""
echo "════════════════════════════════════════════════════════════"
echo "PIPELINE COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Output files:"
echo "  ✓ Metadata:  data/metadata/ground_truth.csv"
echo "  ✓ Validation: data/outputs/validation_report.csv"
echo "  ✓ Dataset Summary: data/outputs/dataset_summary.json"
echo "  ✓ Results: data/outputs/metrics/full_results.csv"
echo "  ✓ Charts: data/outputs/charts/"
echo "  ✓ Report: reports/final_report.md"
echo ""
echo "View the dashboard:"
echo "  streamlit run dashboard/app.py --server.port 8501"
echo ""
