"""Streaming ASR simulation — chunk-based partial transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.base import BaseASR
from src.utils.audio import TARGET_SAMPLE_RATE, load_audio

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    chunks_processed: int
    partial_transcripts: list[str]
    final_transcript: str
    total_latency_seconds: float


def simulate_streaming_asr(
    model: BaseASR,
    audio_path: Path,
    chunk_duration_sec: float = 1.0,
    overlap_sec: float = 0.25,
) -> StreamingResult:
    """
    Simulate streaming by transcribing overlapping chunks and merging.

    Production systems use true streaming APIs (e.g. Deepgram live);
    this offline simulation measures incremental latency behavior.
    """
    import tempfile
    import time

    from src.utils.audio import save_wav

    y, sr = load_audio(audio_path)
    if y is None:
        return {
            "audio_path": str(audio_path),
            "error": "Unsupported audio format",
            "partials": [],
            "total_latency_seconds": 0.0,
        }
    chunk_samples = int(chunk_duration_sec * sr)
    step_samples = int((chunk_duration_sec - overlap_sec) * sr)

    partials: list[str] = []
    total_latency = 0.0
    chunks = 0

    for start in range(0, len(y), step_samples):
        end = min(start + chunk_samples, len(y))
        if end - start < sr // 4:
            break
        chunk = y[start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        save_wav(tmp_path, chunk, sr)

        t0 = time.perf_counter()
        result = model.transcribe(tmp_path)
        total_latency += time.perf_counter() - t0
        tmp_path.unlink(missing_ok=True)

        if result.transcript.strip():
            partials.append(result.transcript.strip())
        chunks += 1

    # Merge: dedupe consecutive identical prefixes (simple stitch)
    final = " ".join(partials)

    return StreamingResult(
        chunks_processed=chunks,
        partial_transcripts=partials,
        final_transcript=final,
        total_latency_seconds=total_latency,
    )
