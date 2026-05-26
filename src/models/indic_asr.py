"""Optional Indic-focused ASR via HuggingFace Wav2Vec2."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import get_settings, resolve_device
from src.models.base import ASRResult, BaseASR
from src.utils.audio import load_audio
from src.utils.helpers import timer

logger = logging.getLogger(__name__)


class IndicASR(BaseASR):
    """Indic Wav2Vec2 CTC model — enable with ENABLE_INDIC_ASR=true."""

    name = "indic_wav2vec"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._processor = None
        self._model = None
        self._device = resolve_device()

    def is_available(self) -> bool:
        return self.settings.enable_indic_asr

    def warmup(self) -> None:
        if not self.is_available():
            return
        if self._model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch

            model_id = self.settings.indic_asr_model
            logger.info("Loading Indic ASR: %s", model_id)
            self._processor = Wav2Vec2Processor.from_pretrained(model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(model_id)
            self._model.to(self._device)
            self._model.eval()
            self._torch = torch

    def transcribe(self, audio_path: Path) -> ASRResult:
        if not self.is_available():
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error="Indic ASR disabled (set ENABLE_INDIC_ASR=true)",
            )

        try:
            self.warmup()
            y, sr = load_audio(audio_path)
            if y is None:
                return AsrResult(
                    model_name=self.name,
                    audio_path=str(audio_path),
                    transcript="",
                    latency_seconds=0.0,
                    error="Unsupported audio format",
                )
            inputs = self._processor(
                y, sampling_rate=sr, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with timer() as elapsed:
                with self._torch.no_grad():
                    logits = self._model(**inputs).logits
                pred_ids = self._torch.argmax(logits, dim=-1)
                transcript = self._processor.batch_decode(pred_ids)[0]

            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript=transcript.strip(),
                latency_seconds=elapsed[0],
                confidence=None,
                metadata={"model": self.settings.indic_asr_model},
            )
        except Exception as exc:
            logger.exception("Indic ASR failed for %s", audio_path)
            return ASRResult(
                model_name=self.name,
                audio_path=str(audio_path),
                transcript="",
                latency_seconds=0.0,
                error=str(exc),
            )
