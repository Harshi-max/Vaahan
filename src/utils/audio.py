"""Audio loading, normalization, and discovery utilities."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterator

import librosa
import numpy as np
import soundfile as sf
try:
    from audioread.exceptions import NoBackendError
except ImportError:
    NoBackendError = Exception

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".ogg", ".webm"}
TARGET_SAMPLE_RATE = 16000


def discover_audio_files(root: Path) -> list[Path]:
    """Recursively find supported audio files under *root*."""
    root = Path(root)
    if not root.exists():
        logger.warning("Audio root does not exist: %s", root)
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def get_condition_from_path(audio_path: Path, raw_root: Path) -> str:
    """Infer recording condition from parent folder name."""
    try:
        rel = audio_path.relative_to(raw_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass
    return "unknown"


def load_audio(path: Path, sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray | None, int]:
    """Load mono audio resampled to *sr*. Returns (None, sr) if format unsupported."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed")
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            y, native_sr = librosa.load(str(path), sr=sr, mono=True)
            return y.astype(np.float32), sr
        except (sf.LibsndfileError, NoBackendError, Exception) as e:
            logger.warning(f"Could not load audio file {path}: {type(e).__name__}")
            return None, sr


def get_duration_seconds(path: Path) -> float:
    """Return audio duration without full decode when possible."""
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        y, sr = load_audio(path)
        if y is None:
            return 0.0
        return len(y) / sr


def normalize_audio(
    y: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.99,
) -> np.ndarray:
    """Peak-normalize waveform with optional loudness targeting."""
    if len(y) == 0:
        return y
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y
    y_norm = y / peak * peak_limit
    rms = np.sqrt(np.mean(y_norm**2) + 1e-12)
    target_rms = 10 ** (target_db / 20.0)
    y_norm = y_norm * (target_rms / rms)
    return np.clip(y_norm, -peak_limit, peak_limit).astype(np.float32)


def save_wav(path: Path, y: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> None:
    """Write float waveform to WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def iter_audio_batches(
    paths: list[Path], batch_size: int = 8
) -> Iterator[list[Path]]:
    """Yield batches of audio paths."""
    for i in range(0, len(paths), batch_size):
        yield paths[i : i + batch_size]
