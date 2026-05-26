"""Vosk offline ASR — free fallback when cloud keys missing."""

from __future__ import annotations

import json
import logging
import wave
from pathlib import Path

from src.config import get_settings
from src.models.base import ASRResult, BaseASR
from src.utils.audio import TARGET_SAMPLE_RATE, load_audio, save_wav
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class VoskASR(BaseASR):
    name = "vosk"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    def is_available(self) -> bool:
        return Path(self.settings.vosk_model_path).exists()

    def warmup(self) -> None:
        if self._model is None and self.is_available():
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            logger.info("Loading Vosk model from %s", self.settings.vosk_model_path)
            self._model = Model(self.settings.vosk_model_path)

    def transcribe(self, audio_path: Path) -> ASRResult:
        if not self.is_available():
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=f"Vosk model not found at {self.settings.vosk_model_path}",
            )

        try:
            from vosk import KaldiRecognizer

            self.warmup()
            assert self._model is not None

            y, sr = load_audio(audio_path, sr=TARGET_SAMPLE_RATE)
            if y is None:
                return AsrResult(
                    model_name=self.name,
                    audio_path=str(audio_path),
                    transcript="",
                    latency_seconds=0.0,
                    error="Unsupported audio format",
                )
            tmp_wav = Path(self.settings.cache_dir) / "_vosk_tmp.wav"
            tmp_wav.parent.mkdir(parents=True, exist_ok=True)
            save_wav(tmp_wav, y, sr)

            with timer() as elapsed:
                wf = wave.open(str(tmp_wav), "rb")
                rec = KaldiRecognizer(self._model, wf.getframerate())
                rec.SetWords(True)
                parts: list[str] = []
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        j = json.loads(rec.Result())
                        parts.append(j.get("text", ""))
                j = json.loads(rec.FinalResult())
                parts.append(j.get("text", ""))
                wf.close()

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=" ".join(p for p in parts if p).strip(),
                latency_seconds=elapsed[0],
                confidence=None,
                metadata={"model_path": self.settings.vosk_model_path},
            )
        except Exception as exc:
            logger.exception("Vosk failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
