"""Automatic dataset ingestion: scan audio files and generate metadata."""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple
import warnings

import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

logger = logging.getLogger(__name__)

# Configuration
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 120
SILENCE_THRESHOLD_DB = -40
SILENCE_DURATION_RATIO = 0.8  # 80% silence = warning
LOW_VOLUME_RMS_THRESHOLD = 0.01  # Approximate low-volume threshold for normalized audio


def get_audio_duration(audio_path: Path) -> Optional[float]:
    """Get audio duration in seconds. Returns None if file is corrupted."""
    try:
        if sf is not None:
            data, sr = sf.read(str(audio_path))
            duration = len(data) / sr
            return duration
    except Exception as exc:
        logger.warning(f"Failed to read audio with soundfile: {audio_path} ({exc})")
    
    try:
        if librosa is not None:
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
    except Exception as exc:
        logger.warning(f"Failed to read audio with librosa: {audio_path} ({exc})")
    
    return None


def validate_audio_format(audio_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if file has supported audio format."""
    supported = {'.wav', '.mp3', '.m4a', '.mp4'}
    ext = audio_path.suffix.lower()
    if ext not in supported:
        return False, f"Unsupported format: {ext}. Supported: {supported}"
    return True, None


def validate_duration(audio_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if audio duration is within acceptable range."""
    duration = get_audio_duration(audio_path)
    if duration is None:
        return False, "Corrupted audio: unable to determine duration"
    if duration < MIN_DURATION_SECONDS:
        return False, f"Too short: {duration:.2f}s (min: {MIN_DURATION_SECONDS}s)"
    if duration > MAX_DURATION_SECONDS:
        return False, f"Too long: {duration:.2f}s (max: {MAX_DURATION_SECONDS}s)"
    return True, None


def detect_silence(audio_path: Path) -> Tuple[float, Optional[str]]:
    """Detect proportion of silence in audio. Returns (ratio, warning_msg)."""
    try:
        if librosa is None:
            return 0.0, None
        
        # Load audio and compute simple RMS energy
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # Compute RMS energy per frame (without numba compilation)
        frame_length = 2048
        hop_length = frame_length // 4
        rms_energy = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            rms_energy.append(rms)
        
        if len(rms_energy) == 0:
            return 0.0, None
        
        rms_energy = np.array(rms_energy)
        # Threshold: frames below 2% of max energy are silent
        threshold = 0.02 * np.max(rms_energy)
        silence_frames = rms_energy < threshold
        silence_ratio = np.sum(silence_frames) / len(silence_frames)
        
        average_rms = float(np.mean(rms_energy))
        if average_rms < LOW_VOLUME_RMS_THRESHOLD:
            return silence_ratio, f"Low volume: average RMS {average_rms:.5f}"
        if silence_ratio > SILENCE_DURATION_RATIO:
            return silence_ratio, f"High silence: {silence_ratio*100:.1f}% is silent"
        return silence_ratio, None
    except Exception as exc:
        logger.warning(f"Could not detect silence in {audio_path}: {exc}")
        return 0.0, None


def extract_locality_from_filename(filename: str) -> Optional[str]:
    """
    Extract locality name from filename.
    Expected format: <locality>_<condition>_<number>.<ext>
    Example: koramangala_traffic_01.wav -> koramangala
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) >= 2:
        # Assume first part before first underscore is locality
        return parts[0]
    return None


def validate_metadata_row(row: dict) -> list:
    """Validate metadata row and return warnings."""
    warnings_list = []
    
    # Check for missing reference transcript
    if not row.get('reference_transcript') or row['reference_transcript'].strip() == '':
        warnings_list.append(f"Missing transcript for {row['filename']}")
    
    return warnings_list


def infer_condition(audio_path: Path, parent_condition: str) -> str:
    """Infer condition from path/folder name and filename heuristics."""
    condition = parent_condition.lower()
    if condition in {"quiet", "traffic", "rushed", "whispered", "phonecall"}:
        return condition

    filename = audio_path.stem.lower()
    if "whatsapp" in filename or "phone" in filename or "phonecall" in filename:
        return "phonecall"
    if "whisper" in filename:
        return "whispered"
    if "traffic" in filename:
        return "traffic"
    if "rushed" in filename or "urgent" in filename:
        return "rushed"
    if "quiet" in filename or "silent" in filename:
        return "quiet"
    return condition or "unknown"


def scan_audio_files(raw_dir: Path) -> list:
    """Scan raw_dir recursively and return list of (audio_path, condition) tuples."""
    audio_files = []
    supported_exts = {'.wav', '.mp3', '.m4a', '.mp4'}
    
    if not raw_dir.exists():
        logger.error(f"Raw data directory not found: {raw_dir}")
        return audio_files
    
    for audio_path in sorted(raw_dir.rglob('*')):
        if audio_path.is_file() and audio_path.suffix.lower() in supported_exts:
            # Infer condition from immediate parent folder and filename heuristics
            condition = infer_condition(audio_path, audio_path.parent.name)
            audio_files.append((audio_path, condition))
    
    return audio_files


def load_existing_metadata(output_csv: Path) -> dict:
    """Load existing metadata records keyed by filename."""
    existing = {}
    if not output_csv.exists():
        return existing
    import csv
    try:
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename')
                if filename:
                    existing[filename] = row
    except Exception as exc:
        logger.warning(f"Could not load existing metadata from {output_csv}: {exc}")
    return existing


def generate_metadata(
    raw_dir: Path,
    output_csv: Path,
    language: str = "hinglish",
    check_corrupted: bool = True,
    check_silence: bool = True,
) -> dict:
    """
    Scan audio files and generate metadata CSV.
    
    Returns:
        dict with keys: 'total', 'valid', 'skipped', 'warnings'
    """
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metadata to preserve transcripts
    existing_metadata = load_existing_metadata(output_csv)
    
    # Scan files
    audio_files = scan_audio_files(raw_dir)
    logger.info(f"Found {len(audio_files)} audio files in {raw_dir}")
    
    metadata_rows = []
    skipped = 0
    all_warnings = []
    seen_filenames = set()
    
    for audio_path, condition in audio_files:
        filename = audio_path.name
        
        # Check for duplicates
        if filename in seen_filenames:
            all_warnings.append(f"DUPLICATE: {filename}")
            skipped += 1
            continue
        seen_filenames.add(filename)
        
        # Validate format
        is_valid, err = validate_audio_format(audio_path)
        if not is_valid:
            all_warnings.append(f"{filename}: {err}")
            skipped += 1
            continue
        
        # Check if corrupted
        if check_corrupted:
            is_valid, err = validate_duration(audio_path)
            if not is_valid:
                all_warnings.append(f"{filename}: {err}")
                skipped += 1
                continue
        
        # Detect silence / low volume
        silence_ratio = 0.0
        if check_silence:
            silence_ratio, silence_warning = detect_silence(audio_path)
            if silence_warning:
                all_warnings.append(f"{filename}: {silence_warning}")
        
        # Extract metadata
        locality_name = extract_locality_from_filename(filename) or "unknown"
        existing_row = existing_metadata.get(filename, {})
        
        row = {
            "filename": filename,
            "condition": condition,
            "language": existing_row.get('language', language),
            "locality_name": existing_row.get('locality_name', locality_name) or locality_name,
            "reference_transcript": existing_row.get('reference_transcript', ""),
        }
        
        # Validate row
        row_warnings = validate_metadata_row(row)
        all_warnings.extend(row_warnings)
        
        metadata_rows.append(row)
    
    if existing_metadata:
        missing_files = set(existing_metadata) - seen_filenames
        for filename in missing_files:
            all_warnings.append(f"Existing metadata row has no matching audio file: {filename}")
    
    # Write CSV
    if metadata_rows:
        import csv
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "filename",
                    "condition",
                    "language",
                    "locality_name",
                    "reference_transcript",
                ],
            )
            writer.writeheader()
            writer.writerows(metadata_rows)
        logger.info(f"Wrote {len(metadata_rows)} rows to {output_csv}")
    
    # Print warnings
    if all_warnings:
        logger.warning(f"\n=== WARNINGS ({len(all_warnings)}) ===")
        for w in all_warnings:
            logger.warning(f"  {w}")
    
    return {
        "total": len(audio_files),
        "valid": len(metadata_rows),
        "skipped": skipped,
        "warnings": all_warnings,
    }


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto-generate metadata CSV from audio files in data/raw/"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw audio directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/metadata/ground_truth.csv"),
        help="Output metadata CSV (default: data/metadata/ground_truth.csv)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="hinglish",
        help="Language tag (default: hinglish)",
    )
    parser.add_argument(
        "--skip-corruption-check",
        action="store_true",
        help="Skip audio corruption/duration checks",
    )
    parser.add_argument(
        "--skip-silence-check",
        action="store_true",
        help="Skip silence detection",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )
    
    # Generate metadata
    result = generate_metadata(
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        language=args.language,
        check_corrupted=not args.skip_corruption_check,
        check_silence=not args.skip_silence_check,
    )
    
    # Print summary
    print("\n" + "="*50)
    print("METADATA GENERATION SUMMARY")
    print("="*50)
    print(f"Total files found:    {result['total']}")
    print(f"Valid entries:        {result['valid']}")
    print(f"Skipped:              {result['skipped']}")
    print(f"Warnings:             {len(result['warnings'])}")
    print(f"Output CSV:           {args.output_csv}")
    print("="*50)
    
    return 0 if result['valid'] > 0 else 1


if __name__ == "__main__":
    exit(main())
