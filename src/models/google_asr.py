"""Google Cloud Speech-to-Text."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings
from src.models.base import ASRResult, BaseASR
from src.utils.audio import load_audio
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class GoogleASR(BaseASR):
    name = "google_stt"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def is_available(self) -> bool:
        creds = self.settings.google_application_credentials
        return bool(creds) and Path(creds).exists()

    def _get_client(self):
        if self._client is None:
            from google.cloud import speech

            self._client = speech.SpeechClient()
        return self._client

    def transcribe(self, audio_path: Path) -> ASRResult:
        if not self.is_available():
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error="GOOGLE_APPLICATION_CREDENTIALS not configured",
            )

        try:
            from google.cloud import speech

            y, sr = load_audio(audio_path)
            if y is None:
                return AsrResult(
                    model_name=self.name,
                    audio_path=str(audio_path),
                    transcript="",
                    latency_seconds=0.0,
                    error="Unsupported audio format",
                )
            audio_bytes = (y * 32767).astype("int16").tobytes()

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                language_code=self.settings.google_stt_language_code,
                enable_automatic_punctuation=True,
                model="latest_long",
                alternative_language_codes=["en-IN", "hi-IN"],
            )
            audio = speech.RecognitionAudio(content=audio_bytes)

            with timer() as elapsed:
                response = self._get_client().recognize(config=config, audio=audio)

            transcript_parts: list[str] = []
            confidences: list[float] = []
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    transcript_parts.append(alt.transcript)
                    if alt.confidence:
                        confidences.append(alt.confidence)

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=" ".join(transcript_parts).strip(),
                latency_seconds=elapsed[0],
                confidence=(
                    float(sum(confidences) / len(confidences))
                    if confidences
                    else None
                ),
                metadata={"language": self.settings.google_stt_language_code},
            )
        except Exception as exc:
            logger.exception("Google STT failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
