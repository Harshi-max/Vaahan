"""Shared helpers: logging, caching, timing."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def cache_key(model: str, audio_path: Path) -> str:
    return f"{model}_{file_hash(audio_path)}"


def load_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    path = cache_dir / f"{key}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@contextmanager
def timer() -> Generator[list[float], None, None]:
    """Context manager storing elapsed seconds in result[0]."""
    result: list[float] = [0.0]
    start = time.perf_counter()
    try:
        yield result
    finally:
        result[0] = time.perf_counter() - start


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
