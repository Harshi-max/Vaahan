# ASR Shootout — Indian Conversational Speech Benchmark

Production-quality benchmarking system for comparing ASR engines on **Indian accents**, **Hinglish**, **noisy audio**, and **locality-name extraction** — the metrics that matter for delivery, logistics, and voice-bot deployments.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/demo-live-success.svg)

---

# Live Dashboard

## Benchmark Explorer(run bench mark first using below commands and check this live app)

👉 **Live Streamlit App:**  
https://vaahan-zcywkuzsskyj9jrbn2ioib.streamlit.app/

Use the live dashboard to:

- Compare ASR model performance interactively
- Explore WER/CER/entity accuracy metrics
- Analyze latency and robustness under noisy conditions
- Visualize confusion matrices, heatmaps, and radar charts
- Inspect per-condition benchmark results

---

# Overview

| Capability | Description |
|------------|-------------|
| Multi-model ASR | Deepgram, Whisper, Faster-Whisper, Google STT, Vosk, optional Indic Wav2Vec |
| Auto-discovery | Scans `data/raw/{condition}/` for wav/mp3/m4a |
| Entity metrics | Locality accuracy beyond WER |
| Failure analysis | Code-switch, hallucination, noise degradation |
| Visualizations | Bar, radar, heatmap, latency charts |
| Interactive Dashboard | Live Streamlit benchmark explorer |
| One-command run | `bash setup.sh && bash run.sh` |

---

# Architecture

```text
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio +    │────▶│  ASR Adapters    │────▶│  Evaluation     │
│  Ground     │     │  (6 backends)    │     │  WER/CER/Entity │
│  Truth CSV  │     └──────────────────┘     └────────┬────────┘
└─────────────┘                                        │
                                                       ▼
                              ┌────────────────────────────────────┐
                              │ Charts + Report + Streamlit Dashboard │
                              └────────────────────────────────────┘
```

---

# Project Structure

```text
asr-shootout/
├── dashboard/                  # Streamlit dashboard
├── data/raw/{quiet,traffic,rushed,whispered,phonecall}/
├── data/metadata/ground_truth.csv
├── data/outputs/{transcripts,metrics,charts}/
├── src/models/                 # ASR adapters
├── src/evaluation/             # Metrics & analysis
├── src/utils/                  # Audio I/O, normalization
├── src/main.py                 # CLI entrypoint
├── notebooks/exploration.ipynb
├── reports/final_report.md
├── setup.sh / run.sh
└── requirements.txt
```

---

# Quick Start

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Git installed
- `ffmpeg` available on `PATH`
- Optional: CUDA GPU for faster Whisper inference

---

# Local Setup

## 1) Clone the Repository

```bash
git clone <your-repo-url>
cd asr-shootout
```

## 2) Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Prepare Demo Data (Optional)

```bash
python scripts/prepare_demo_data.py --out-dir data/raw
```

## 5) Validate Dataset

```bash
python scripts/validate_dataset.py --data-dir data
```

This step:

- Discovers audio files automatically
- Generates metadata CSVs
- Normalizes audio
- Produces dataset summaries and validation reports

## 6) Run Full Benchmark

```bash
python -m src.main run --data-dir data --output-dir data/outputs --report-path reports/final_report.md
```

## 7) Launch Dashboard Locally

```bash
streamlit run dashboard/app.py
```

Open:

```text
http://localhost:8501
```

---

# Streamlit Dashboard Features

The dashboard provides:

- 📊 Model leaderboard
- 📈 WER/CER comparisons
- 🌍 Locality extraction accuracy
- 🔥 Noise robustness analysis
- ⚡ Latency benchmarking
- 🕸 Radar charts for multi-metric tradeoffs
- 🎯 Per-condition evaluation breakdowns
- 📂 Transcript inspection tools

## Hosted Dashboard

Access the deployed benchmark dashboard here:

https://vaahan-zcywkuzsskyj9jrbn2ioib.streamlit.app/

---

# API Keys / `.env`

Copy `.env.example` to `.env`

## Linux/macOS

```bash
cp .env.example .env
```

## Windows PowerShell

```powershell
Copy-Item .env.example -Destination .env
```

| Key | Required For |
|-----|--------------|
| `DEEPGRAM_API_KEY` | Deepgram ASR |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud STT |

If keys are missing, cloud models are skipped gracefully.

---

# Recording Your Dataset

See:

```text
scripts/recording_guide.md
```

Workflow:

1. Record clips into `data/raw/{condition}/`
2. Add rows to `data/metadata/ground_truth.csv`
3. Validate dataset
4. Normalize audio
5. Run benchmark pipeline

## Naming Convention

```text
{speaker}_{locality}_{condition}.wav
```

---

# Running Evaluation

## Full Pipeline

```bash
bash run.sh
```

## Direct Execution

```bash
python -m src.main run --data-dir data --output-dir data/outputs
```

## List Available Models

```bash
python -m src.main list-models
```

## Run Specific Models

```bash
python -m src.main run --models faster_whisper --models whisper
```

---

# Outputs

| Path | Content |
|------|---------|
| `data/outputs/transcripts/` | Per-model transcript CSVs |
| `data/outputs/metrics/full_results.csv` | File-level metrics |
| `data/outputs/metrics/summary_by_model.csv` | Aggregated leaderboard |
| `data/outputs/charts/` | Generated visualizations |
| `reports/final_report.md` | Auto-generated benchmark report |

---

# Sample Charts

After running the benchmark, expect charts such as:

- `wer_by_model.png`
- `entity_accuracy_by_model.png`
- `noise_vs_wer.png`
- `latency_comparison.png`
- `locality_heatmap.png`
- `radar_comparison.png`

---

# Models Compared

| Model | Type | Strength | Weakness |
|-------|------|----------|----------|
| Deepgram | Cloud | Streaming + latency | Indian entity names |
| Whisper | Local | Noise robustness | CPU speed |
| Faster-Whisper | Local | Cost/performance | Slight quality tradeoff |
| Google STT | Cloud | Multi-dialect support | Setup complexity |
| Vosk | Offline | No API key needed | Lower accuracy |
| Indic Wav2Vec | HF | Hindi-focused | Experimental |

---

# Interview Talking Points

1. WER alone is insufficient for logistics/address workflows.
2. Entity accuracy is critical for locality extraction.
3. Faster-Whisper offers strong self-hosted performance.
4. Hinglish code-switching impacts recognition quality.
5. Phone-call audio significantly hurts entity extraction.
6. Graceful fallback architecture prevents pipeline crashes.
7. Multiprocessing + caching improve reproducibility and scale.

---

# Future Improvements

- [ ] Collect 50+ real Hinglish delivery-call recordings
- [ ] Fine-tune Whisper on domain-specific audio
- [ ] Two-pass ASR + NER locality extraction
- [ ] Streaming ASR evaluation
- [ ] Weighted entity F1 scoring
- [ ] CI regression benchmarking
- [ ] Speaker diarization support
- [ ] Real-time streaming dashboard

---

# License

MIT — free to use for research, internships, and portfolio projects.

---

# Author Notes

Built for benchmarking ASR systems on:

- Indian conversational speech
- Hinglish code-switching
- Noisy real-world audio
- Production ML evaluation pipelines
- Voice AI system engineering
