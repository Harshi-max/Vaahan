# ASR Shootout — Indian Conversational Speech Benchmark

Production-quality benchmarking system for comparing ASR engines on **Indian accents**, **Hinglish**, **noisy audio**, and **locality-name extraction** — the metrics that matter for delivery, logistics, and voice-bot deployments.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Overview

| Capability | Description |
|------------|-------------|
| Multi-model ASR | Deepgram, Whisper, Faster-Whisper, Google STT, Vosk, optional Indic Wav2Vec |
| Auto-discovery | Scans `data/raw/{condition}/` for wav/mp3/m4a |
| Entity metrics | Locality accuracy beyond WER |
| Failure analysis | Code-switch, hallucination, noise degradation |
| Visualizations | Bar, radar, heatmap, latency charts |
| One-command run | `bash setup.sh && bash run.sh` |

## Architecture

```
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

## Project Structure

```
asr-shootout/
├── data/raw/{quiet,traffic,rushed,whispered,phonecall}/
├── data/metadata/ground_truth.csv
├── data/outputs/{transcripts,metrics,charts}/
├── src/models/          # ASR adapters
├── src/evaluation/      # Metrics & analysis
├── src/utils/           # Audio I/O, normalization
├── src/main.py          # CLI entrypoint
├── notebooks/exploration.ipynb
├── reports/final_report.md
├── setup.sh / run.sh
└── requirements.txt
```

## Quick Start

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Git (for cloning) and a system `ffmpeg` binary available on `PATH` (or use the repo's local fallback)
- Optional: CUDA GPU for faster Whisper inference

### Run without shell scripts (recommended)

The repository used to include `setup.sh` / `run.sh`. If those are missing, use the Python/PowerShell commands below — no Bash required.

1) Create & activate virtualenv (Windows PowerShell):

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

macOS / Linux (POSIX):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional) Prepare demo data for smoke tests

```powershell
python scripts/prepare_demo_data.py --out-dir data/raw
```

4) Run ingestion + validation (automatic metadata generation & normalization)

```powershell
python scripts/validate_dataset.py --data-dir data
```

What this does:
- Discovers audio in `data/raw/` and organizes into condition folders
- Generates `data/metadata/ground_truth.csv` (auto-filled transcripts when missing)
- Produces normalized audio under `data/processed/`
- Writes `data/outputs/validation_report.csv` and `data/outputs/dataset_summary.json`

5) Run the full benchmark (use processed audio if present)

```powershell
python -m src.main run --data-dir data --output-dir data/outputs --report-path reports/final_report.md
```

6) Streamlit dashboard (local)

```powershell
python -m streamlit run dashboard/app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

PowerShell note: you can run the bundled PowerShell helper to run the full pipeline:

```powershell
.\run.ps1
```

Tip: enable "Auto-refresh dashboard" in the Streamlit sidebar to reload visualizations every N seconds.

### API Keys / .env

Copy `.env.example` to `.env` (POSIX):

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example -Destination .env
```

Key table

| Key | Required For |
|-----|--------------|
| `DEEPGRAM_API_KEY` | Deepgram (optional) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google Cloud STT |

Missing keys → cloud models are skipped gracefully; local models still run.

## Recording Your Dataset

See [`scripts/recording_guide.md`](scripts/recording_guide.md).

1. Record clips into `data/raw/{condition}/`
2. Add rows to `data/metadata/ground_truth.csv`
3. Validate: `python scripts/validate_dataset.py`
4. Normalize (optional): `python scripts/normalize_audio.py`

**Naming:** `{speaker}_{locality}_{condition}.wav`

## Running Evaluation

```bash
# Full pipeline
bash run.sh

# Or directly
python -m src.main run --data-dir data --output-dir data/outputs

# List available models
python -m src.main list-models

# Limit models
python -m src.main run --models faster_whisper --models whisper
```

### Outputs

| Path | Content |
|------|---------|
| `data/outputs/transcripts/` | Per-model CSV transcripts |
| `data/outputs/metrics/full_results.csv` | All metrics per file/model |
| `data/outputs/metrics/summary_by_model.csv` | Aggregated leaderboard |
| `data/outputs/charts/` | PNG visualizations |
| `reports/final_report.md` | Auto-generated research report |

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

### FastAPI (optional)

```bash
uvicorn scripts.api_server:app --reload --port 8000
```

## Sample Results

After `run.sh`, expect charts like:

- `wer_by_model.png` — mean WER comparison
- `entity_accuracy_by_model.png` — locality extraction (key metric)
- `noise_vs_wer.png` — degradation by condition
- `latency_comparison.png` — inference speed
- `locality_heatmap.png` — per-locality accuracy
- `radar_comparison.png` — multi-metric tradeoffs

## Interview Talking Points

1. **WER is insufficient** for address/locality tasks — we measure entity accuracy explicitly.
2. **Deepgram** = best latency for real-time; struggles with Indian locality spellings.
3. **Faster-Whisper** = best batch cost/performance on self-hosted GPU.
4. **Code-switching** (Hinglish) breaks entity extraction — needs gazetteer post-processing.
5. **Phone-call audio** degrades entity accuracy more than WER suggests.
6. **Caching + multiprocessing** built in for reproducible large-scale benchmarks.
7. **Graceful fallback** when API keys missing — pipeline never hard-fails.

## Models Compared

| Model | Type | Strength | Weakness |
|-------|------|----------|----------|
| Deepgram | Cloud | Latency, streaming | Indian entity names |
| Whisper | Local | Noise, Hinglish | Slow on CPU |
| Faster-Whisper | Local | Speed/cost | Slight quality tradeoff |
| Google STT | Cloud | Multi-dialect | Setup friction |
| Vosk | Offline | No API key | Lower accuracy |
| Indic Wav2Vec | HF | Hindi focus | Needs `ENABLE_INDIC_ASR=true` |

## Future Improvements

- [ ] Record 50+ real Hinglish delivery-call clips
- [ ] Fine-tune Whisper on domain data
- [ ] Two-pass ASR + NER for locality extraction
- [ ] Streaming partial-hypothesis evaluation
- [ ] Weighted entity F1 with phonetic matching (Soundex/Metaphone)
- [ ] CI benchmark on PR with regression thresholds

## License

MIT — use freely for portfolio and internship submissions.

---

Built for ASR benchmarking assignments focusing on **Indian conversational speech** and **production ML systems engineering**.
