"""Faster-Whisper optimized local ASR."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings, resolve_device
from src.models.base import ASRResult, BaseASR
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class FasterWhisperASR(BaseASR):
    name = "faster_whisper"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        device = self.settings.faster_whisper_device
        self._device = resolve_device() if device == "auto" else device

    def is_available(self) -> bool:
        return True

    def warmup(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Faster-Whisper '%s' (%s, %s)",
                self.settings.faster_whisper_model,
                self._device,
                self.settings.faster_whisper_compute_type,
            )
            self._model = WhisperModel(
                self.settings.faster_whisper_model,
                device=self._device,
                compute_type=self.settings.faster_whisper_compute_type,
            )

    def transcribe(self, audio_path: Path) -> ASRResult:
        try:
            self.warmup()
            assert self._model is not None

            with timer() as elapsed:
                segments, info = self._model.transcribe(
                    str(audio_path),
                    language=None,
                    vad_filter=True,
                    beam_size=5,
                )
                text_parts: list[str] = []
                confidences: list[float] = []
                for seg in segments:
                    text_parts.append(seg.text.strip())
                    if seg.avg_logprob is not None:
                        confidences.append(seg.avg_logprob)

            transcript = " ".join(text_parts).strip()
            avg_conf = (
                float(sum(confidences) / len(confidences)) if confidences else None
            )

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=transcript,
                latency_seconds=elapsed[0],
                confidence=avg_conf,
                metadata={
                    "language": getattr(info, "language", None),
                    "language_probability": getattr(
                        info, "language_probability", None
                    ),
                    "model": self.settings.faster_whisper_model,
                },
            )
        except Exception as exc:
            logger.exception("Faster-Whisper failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
