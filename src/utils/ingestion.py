"""Automatic audio ingestion, classification, metadata generation, normalization, and reporting."""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from src.utils.audio import (
    TARGET_SAMPLE_RATE,
    SUPPORTED_EXTENSIONS,
    discover_audio_files,
    load_audio,
    normalize_audio,
    save_wav,
)
from src.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

KNOWN_CONDITIONS = {"quiet", "traffic", "rushed", "whispered", "phonecall"}
LANGUAGE_KEYWORDS = {"telugu", "hindi", "english"}
CONDITION_KEYWORDS = {
    "phonecall": {"phone", "phonecall", "whatsapp", "call"},
    "whispered": {"whisper", "whispered"},
    "traffic": {"traffic", "road", "car", "street"},
    "rushed": {"rushed", "rush", "urgent", "fast", "hurry"},
    "quiet": {"quiet", "silent", "silence", "calm"},
}
ABBREVIATIONS = {
    "hsr": "HSR City",
    "hitec": "HITEC City",
    "blr": "Bangalore",
    "bng": "Bengaluru",
    "mgb": "MG Road",
    "mgroad": "MG Road",
    "indira": "Indiranagar",
    "indiranagar": "Indiranagar",
    "jaya": "Jayanagar",
    "jayanagar": "Jayanagar",
    "connaught": "Connaught Place",
    "cp": "Connaught Place",
    "whitefield": "Whitefield",
    "wf": "Whitefield",
    "koramangala": "Koramangala",
    "kora": "Koramangala",
    "salt": "Salt Lake",
    "saltlake": "Salt Lake",
    "andheri": "Andheri West",
    "bandra": "Bandra",
    "borivali": "Borivali",
    "dadar": "Dadar",
    "fort": "Fort",
    "marine": "Marine Lines",
    "malabar": "Malabar Hill",
    "cyber": "Cyber Towers",
    "jubilee": "Jubilee Hills",
    "madhapur": "Madhapur",
    "kukatpally": "Kukatpally",
    "gachibowli": "Gachibowli",
    "kondapur": "Kondapur",
    "banjara": "Banjara Hills",
}

MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 60.0
SILENCE_DB = 30
LOW_VOLUME_RMS_THRESHOLD = 0.01
NOISE_FLATNESS_THRESHOLD = 0.55

DEFAULT_LANGUAGE = "hindi"


def normalize_locality_name(raw_name: str) -> str:
    """Extract and normalize locality names from filenames using multiple strategies."""
    raw = str(raw_name).lower().strip()
    
    # Remove common prefixes and metadata
    raw = re.sub(r"^(whatsapp\s+audio|whatsapp|phone\d*|phonecall|call|sample|sample_\d+)", "", raw)
    raw = re.sub(r"\b(at|near|around|in|by|metro|station|block|sector|area)\b", "", raw)
    
    # Remove timestamps  
    raw = re.sub(r"\d{1,2}[:.]\d{2}\s*(am|pm)", "", raw)
    raw = re.sub(r"\(\d+\)", "", raw)
    
    # Remove file extension
    raw = re.sub(r"\.[a-zA-Z0-9]+$", "", raw)
    
    # Normalize separators to spaces
    raw = re.sub(r"[_\-]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    
    # Split into tokens, keep only non-empty
    tokens = [t for t in re.split(r"\s+", raw) if t and len(t) > 1]
    
    # Exclude metadata keywords
    excluded = LANGUAGE_KEYWORDS.union(*CONDITION_KEYWORDS.values()).union({"dup", "pm", "am", "01", "02", "03", "04", "05"})
    tokens = [t for t in tokens if t not in excluded]
    
    if not tokens:
        return "Unknown"
    
    # Apply abbreviation expansions and capitalize
    normalized_tokens = [ABBREVIATIONS.get(t, t.capitalize()) for t in tokens]
    return " ".join(normalized_tokens)


def infer_condition(audio_path: Path) -> str:
    parent = audio_path.parent.name.lower()
    if parent in KNOWN_CONDITIONS:
        return parent
    name = audio_path.stem.lower()
    for condition, keywords in CONDITION_KEYWORDS.items():
        if any(token in name for token in keywords):
            return condition
    if any(token in parent for token in CONDITION_KEYWORDS["phonecall"]):
        return "phonecall"
    if any(token in parent for token in CONDITION_KEYWORDS["whispered"]):
        return "whispered"
    if any(token in parent for token in CONDITION_KEYWORDS["traffic"]):
        return "traffic"
    if any(token in parent for token in CONDITION_KEYWORDS["rushed"]):
        return "rushed"
    if any(token in parent for token in CONDITION_KEYWORDS["quiet"]):
        return "quiet"
    return "unclassified"


def infer_language(audio_path: Path) -> str:
    name = audio_path.stem.lower()
    parent = audio_path.parent.name.lower()
    for token in LANGUAGE_KEYWORDS:
        if token in name or token in parent:
            return token
    return DEFAULT_LANGUAGE


def placeholder_transcript(locality: str, language: str) -> str:
    """Generate realistic placeholder transcripts in multiple languages."""
    lang = str(language).lower().strip() if language else "hindi"
    if "telugu" in lang:
        return f"Naanu {locality} loo unnanu"
    if "english" in lang or "en" in lang:
        return f"I live in {locality}."
    return f"Haan, main {locality} mein rehta hoon"


def get_audio_info(path: Path) -> tuple[float, int, str, str | None]:
    ext = path.suffix.lower()
    try:
        info = sf.info(str(path))
        return float(info.duration), int(info.samplerate), ext, None
    except Exception as exc:
        if librosa is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="PySoundFile failed")
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    y, sr = librosa.load(str(path), sr=None, mono=True)
                return float(len(y)) / sr, int(sr), ext, None
            except Exception as exc2:
                return 0.0, 0, ext, str(exc2)
        return 0.0, 0, ext, str(exc)


def validate_audio(path: Path) -> dict[str, Any]:
    duration, sample_rate, ext, error = get_audio_info(path)
    issues: list[str] = []
    status = "OK"
    silence_ratio = 0.0
    low_volume = False
    noise_warning = False

    if error:
        status = "FAIL"
        issues.append(f"Corrupted: {error}")
        return _build_validation_record(path, duration, sample_rate, ext, status, issues, silence_ratio, low_volume, noise_warning)

    if ext not in SUPPORTED_EXTENSIONS:
        status = "FAIL"
        issues.append(f"Unsupported format: {ext}")

    if duration < MIN_DURATION_SECONDS:
        issues.append(f"Too short: {duration:.2f}s")
        status = "WARN" if status == "OK" else status
    if duration > MAX_DURATION_SECONDS:
        issues.append(f"Too long: {duration:.2f}s")
        status = "WARN" if status == "OK" else status

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            warnings.filterwarnings("ignore", category=FutureWarning)
            y, sr = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
        rms = float(np.sqrt(np.mean(y**2) + 1e-12))
        low_volume = rms < LOW_VOLUME_RMS_THRESHOLD
        if low_volume:
            issues.append(f"Low volume: RMS={rms:.5f}")
            status = "WARN" if status == "OK" else status
        silence_bounds = librosa.effects.split(y, top_db=SILENCE_DB)
        silence_ratio = 1.0 - sum((end - start) for start, end in silence_bounds) / len(y) if len(y) > 0 else 0.0
        if silence_ratio > 0.8:
            issues.append(f"High silence: {silence_ratio*100:.1f}%")
            status = "WARN" if status == "OK" else status
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y) + 1e-12))
        noise_warning = flatness > NOISE_FLATNESS_THRESHOLD
        if noise_warning:
            issues.append(f"Noisy audio: flatness={flatness:.3f}")
            status = "WARN" if status == "OK" else status
    except Exception as exc:
        logger.warning("Validation audio analysis failed for %s: %s", path, exc)

    return _build_validation_record(path, duration, sample_rate, ext, status, issues, silence_ratio, low_volume, noise_warning)


def _build_validation_record(path: Path, duration: float, sample_rate: int, ext: str, status: str, issues: list[str], silence_ratio: float, low_volume: bool, noise_warning: bool) -> dict[str, Any]:
    return {
        "filename": path.name,
        "condition": infer_condition(path),
        "language": infer_language(path),
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "format": ext,
        "status": status,
        "issues": "; ".join(issues),
        "silence_ratio": round(silence_ratio, 3),
        "low_volume": low_volume,
        "noisy": noise_warning,
        "path": str(path),
    }


def _copy_to_condition_folder(path: Path, raw_root: Path) -> Path:
    condition = infer_condition(path)
    if condition == "unclassified":
        target_folder = raw_root / "unclassified"
    else:
        target_folder = raw_root / condition
    target_folder.mkdir(parents=True, exist_ok=True)
    target = target_folder / path.name
    index = 1
    while target.exists() and target != path:
        target = target_folder / f"{path.stem}_dup{index}{path.suffix}"
        index += 1
    if target == path:
        return path
    shutil.move(str(path), str(target))
    return target


def organize_raw(raw_root: Path) -> list[Path]:
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raw_root.mkdir(parents=True, exist_ok=True)
    if any(raw_root.iterdir()):
        processed = []
        for path in sorted(raw_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                rel = path.relative_to(raw_root)
                if len(rel.parts) == 1:
                    path = _copy_to_condition_folder(path, raw_root)
                elif rel.parts[0] not in KNOWN_CONDITIONS and rel.parts[0] != "unclassified":
                    path = _copy_to_condition_folder(path, raw_root)
                processed.append(path)
        return sorted(set(processed))
    return []


def load_existing_metadata(metadata_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not metadata_path.exists():
        return existing
    try:
        with open(metadata_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("filename"):
                    existing[row["filename"]] = row
    except Exception as exc:
        logger.warning("Could not load existing metadata: %s", exc)
    return existing


def generate_metadata(raw_root: Path, metadata_path: Path) -> list[dict[str, Any]]:
    audio_files = discover_audio_files(raw_root)
    existing = load_existing_metadata(metadata_path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for audio_path in sorted(audio_files):
        filename = audio_path.name
        if filename in seen:
            logger.warning("Duplicate audio filename: %s", filename)
            continue
        seen.add(filename)
        condition = infer_condition(audio_path)
        locality_name = normalize_locality_name(audio_path.stem)
        language = infer_language(audio_path)
        duration, sample_rate, ext, err = get_audio_info(audio_path)
        existing_row = existing.get(filename, {})
        reference_transcript = existing_row.get("reference_transcript", "").strip()
        auto_generated = False
        if not reference_transcript:
            reference_transcript = placeholder_transcript(locality_name, language)
            auto_generated = True
        row = {
            "filename": filename,
            "locality_name": locality_name,
            "language": language,
            "condition": condition,
            "duration_seconds": round(duration, 2),
            "sample_rate": sample_rate,
            "reference_transcript": reference_transcript,
            "auto_generated_reference": str(auto_generated),
        }
        rows.append(row)

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "locality_name",
        "language",
        "condition",
        "duration_seconds",
        "sample_rate",
        "reference_transcript",
        "auto_generated_reference",
    ]
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def normalize_audio_files(raw_root: Path, processed_root: Path) -> list[Path]:
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)
    normalized_paths: list[Path] = []
    for audio_path in discover_audio_files(raw_root):
        y, sr = load_audio(audio_path, sr=TARGET_SAMPLE_RATE)
        if y is None:
            logger.warning(f"Skipping unsupported audio file: {audio_path}")
            continue
        condition = infer_condition(audio_path)
        target_dir = processed_root / condition
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{audio_path.stem}.wav"
        try:
            trimmed, _ = librosa.effects.trim(y, top_db=SILENCE_DB)
        except Exception:
            trimmed = y
        y_norm = normalize_audio(trimmed)
        save_wav(target_path, y_norm, TARGET_SAMPLE_RATE)
        normalized_paths.append(target_path)
    return normalized_paths


def write_validation_report(raw_root: Path, report_path: Path) -> list[dict[str, Any]]:
    rows = []
    paths = discover_audio_files(raw_root)
    names: set[str] = set()
    duplicate_names: set[str] = set()
    for path in paths:
        if path.name in names:
            duplicate_names.add(path.name)
        names.add(path.name)
    for path in paths:
        record = validate_audio(path)
        if path.name in duplicate_names:
            record["status"] = "WARN"
            extra = record["issues"]
            record["issues"] = "; ".join(filter(None, [extra, "Duplicate filename"]))
        rows.append(record)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "condition",
        "language",
        "duration_seconds",
        "sample_rate",
        "format",
        "status",
        "issues",
        "silence_ratio",
        "low_volume",
        "noisy",
        "path",
    ]
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_dataset_summary(metadata_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], summary_path: Path) -> None:
    total = len(metadata_rows)
    durations = [row["duration_seconds"] for row in metadata_rows if row["duration_seconds"]]
    avg_duration = float(np.mean(durations)) if durations else 0.0
    condition_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}
    sample_rates: dict[int, int] = {}
    missing_transcripts = 0
    failed_files = 0

    for row in metadata_rows:
        condition_counts[row["condition"]] = condition_counts.get(row["condition"], 0) + 1
        sample_rates[row["sample_rate"]] = sample_rates.get(row["sample_rate"], 0) + 1
        if not row["reference_transcript"] or row.get("auto_generated_reference") == "True":
            missing_transcripts += 1

    for row in validation_rows:
        fmt = row.get("format")
        if fmt:
            format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if row["status"] == "FAIL":
            failed_files += 1

    summary = {
        "total_recordings": total,
        "average_duration_seconds": round(avg_duration, 2),
        "condition_distribution": condition_counts,
        "audio_formats": format_counts,
        "sample_rates": sample_rates,
        "missing_transcripts": missing_transcripts,
        "failed_files": failed_files,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def run_ingestion_pipeline(data_dir: Path) -> dict[str, Any]:
    data_dir = Path(data_dir)
    raw_root = data_dir / "raw"
    metadata_path = data_dir / "metadata" / "ground_truth.csv"
    processed_root = data_dir / "processed"
    outputs_root = data_dir / "outputs"
    validation_path = outputs_root / "validation_report.csv"
    summary_path = outputs_root / "dataset_summary.json"

    ensure_dir(raw_root)
    ensure_dir(data_dir / "metadata")
    ensure_dir(outputs_root)

    audio_files = discover_audio_files(raw_root)
    if not audio_files and not metadata_path.exists():
        logger.warning("No raw audio found; generating demo data for pipeline smoke test.")
        tool_root = Path(__file__).resolve().parents[2]
        demo_script = tool_root / "scripts" / "prepare_demo_data.py"
        subprocess.run([sys.executable, str(demo_script)], check=False)
        audio_files = discover_audio_files(raw_root)

    organized_files = organize_raw(raw_root)
    if organized_files:
        logger.info("Organized raw audio into condition folders.")
    audio_files = discover_audio_files(raw_root)

    metadata_rows = generate_metadata(raw_root, metadata_path)
    validation_rows = write_validation_report(raw_root, validation_path)
    normalize_audio_files(raw_root, processed_root)
    write_dataset_summary(metadata_rows, validation_rows, summary_path)

    warnings = []
    if len(audio_files) < 20:
        warnings.append(f"Only {len(audio_files)} recordings detected (20 recommended).")
    if any(row["status"] == "FAIL" for row in validation_rows):
        warnings.append("Some files failed validation. Check outputs/validation_report.csv.")

    return {
        "audio_count": len(audio_files),
        "metadata_count": len(metadata_rows),
        "validation_count": len(validation_rows),
        "warnings": warnings,
        "validation_report": str(validation_path),
        "dataset_summary": str(summary_path),
        "processed_root": str(processed_root),
    }
