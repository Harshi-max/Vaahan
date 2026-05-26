"""Abstract base class for ASR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ASRResult:
    """Standardized transcription output."""

    model_name: str
    audio_path: str
    transcript: str
    latency_seconds: float
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.transcript.strip())


class BaseASR(ABC):
    """Interface all ASR backends must implement."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: Path) -> ASRResult:
        """Run inference on a single audio file."""
        ...

    def is_available(self) -> bool:
        """Return True if this backend can run (keys, models, etc.)."""
        return True

    def warmup(self) -> None:
        """Optional model load / connection warmup."""
        pass
