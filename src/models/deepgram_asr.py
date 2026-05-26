"""Deepgram cloud ASR — mandatory baseline."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import get_settings
from src.models.base import ASRResult, BaseASR
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class DeepgramASR(BaseASR):
    name = "deepgram"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None

    def is_available(self) -> bool:
        return bool(self.settings.deepgram_api_key)

    def _get_client(self):
        if self._client is None:
            try:
                # Newer SDK exposes Deepgram class
                from deepgram import Deepgram

                self._client = Deepgram(self.settings.deepgram_api_key)
            except Exception:
                # Fall back to None; we'll use REST API in transcribe
                self._client = None
        return self._client

    def transcribe(self, audio_path: Path) -> ASRResult:
        if not self.is_available():
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error="DEEPGRAM_API_KEY not set",
            )

        try:
            # Prefer SDK when available
            client = self._get_client()
            with open(audio_path, "rb") as f:
                buffer_data = f.read()

            if client is not None:
                # Try SDK-style call (some SDKs differ; attempt best-effort)
                try:
                    # SDK may offer transcription via client.transcription.prerecorded
                    with timer() as elapsed:
                        response = client.transcription.prerecorded(
                            buffer_data, {
                                "model": self.settings.deepgram_model,
                                "language": self.settings.deepgram_language,
                                "punctuate": True,
                            }
                        )
                    # response structure varies; attempt to extract transcript
                    transcript = ""
                    confidence = None
                    try:
                        alt = response["results"]["channels"][0]["alternatives"][0]
                        transcript = alt.get("transcript", "")
                        confidence = alt.get("confidence")
                    except Exception:
                        transcript = str(response)

                    return ASRResult(
                        model_name=self.name,
                        audio_path=str(audio_path),
                        transcript=transcript,
                        latency_seconds=elapsed[0],
                        confidence=confidence,
                        metadata={"model": self.settings.deepgram_model},
                    )

                except Exception:
                    # Fall through to REST API fallback
                    pass

            # REST API fallback (works without SDK)
            import requests

            url = "https://api.deepgram.com/v1/listen"
            params = {
                "model": self.settings.deepgram_model,
                "language": self.settings.deepgram_language,
                "punctuate": "true",
            }
            headers = {"Authorization": f"Token {self.settings.deepgram_api_key}"}

            with timer() as elapsed:
                resp = requests.post(url, params=params, headers=headers, data=buffer_data, timeout=60)
            resp.raise_for_status()
            j = resp.json()
            transcript = ""
            confidence = None
            try:
                alt = j["results"]["channels"][0]["alternatives"][0]
                transcript = alt.get("transcript", "")
                confidence = alt.get("confidence")
            except Exception:
                transcript = str(j)

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=transcript,
                latency_seconds=elapsed[0],
                confidence=confidence,
                metadata={"model": self.settings.deepgram_model},
            )
        except Exception as exc:
            logger.exception("Deepgram failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
