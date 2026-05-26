"""Registry of all ASR backends with graceful fallback."""

from __future__ import annotations

import logging
from typing import Type

from src.models.base import BaseASR
from src.models.deepgram_asr import DeepgramASR
from src.models.faster_whisper_asr import FasterWhisperASR
from src.models.google_asr import GoogleASR
from src.models.indic_asr import IndicASR
from src.models.vosk_asr import VoskASR
from src.models.whisper_asr import WhisperASR

logger = logging.getLogger(__name__)

ALL_MODEL_CLASSES: list[Type[BaseASR]] = [
    DeepgramASR,
    WhisperASR,
    FasterWhisperASR,
    GoogleASR,
    VoskASR,
    IndicASR,
]


def get_active_models(skip_unavailable: bool = True) -> list[BaseASR]:
    """Instantiate models; skip those without credentials/models."""
    models: list[BaseASR] = []
    for cls in ALL_MODEL_CLASSES:
        instance = cls()
        if skip_unavailable and not instance.is_available():
            logger.warning("Skipping %s (not available)", instance.name)
            continue
        models.append(instance)

    # Ensure at least one local model runs
    if not models:
        logger.warning("No cloud models available; forcing Faster-Whisper")
        models.append(FasterWhisperASR())

    return models
