"""OpenAI Whisper local ASR."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings, resolve_device
from src.models.base import ASRResult, BaseASR
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR):
    name = "whisper"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None
        self._device = resolve_device()

    def is_available(self) -> bool:
        # Whisper requires the `whisper` package and an `ffmpeg` binary on PATH
        try:
            import whisper  # noqa: F401
        except Exception:
            logger.warning("Whisper package not installed; skipping Whisper")
            return False

        from shutil import which

        if which("ffmpeg") is None:
            logger.warning("ffmpeg not found on PATH; skipping Whisper (install ffmpeg to enable)")
            return False

        return True

    def warmup(self) -> None:
        if self._model is None:
            import whisper

            logger.info(
                "Loading Whisper model '%s' on %s",
                self.settings.whisper_model,
                self._device,
            )
            self._model = whisper.load_model(
                self.settings.whisper_model, device=self._device
            )

    def transcribe(self, audio_path: Path) -> ASRResult:
        try:
            self.warmup()
            assert self._model is not None

            with timer() as elapsed:
                result = self._model.transcribe(
                    str(audio_path),
                    language=None,  # auto-detect for Hinglish
                    task="transcribe",
                    fp16=self._device == "cuda",
                )

            segments = result.get("segments", [])
            confidences = [
                s.get("avg_logprob", 0) for s in segments if "avg_logprob" in s
            ]
            avg_conf = (
                float(sum(confidences) / len(confidences)) if confidences else None
            )

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=(result.get("text") or "").strip(),
                latency_seconds=elapsed[0],
                confidence=avg_conf,
                metadata={
                    "detected_language": result.get("language"),
                    "model": self.settings.whisper_model,
                },
            )
        except Exception as exc:
            logger.exception("Whisper failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
