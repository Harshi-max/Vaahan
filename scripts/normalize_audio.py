#!/usr/bin/env python3
"""Batch-normalize all raw audio to 16 kHz mono WAV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio import discover_audio_files, load_audio, normalize_audio, save_wav  # noqa: E402

console = Console()


def normalize_all(raw_dir: Path, in_place: bool = False) -> None:
    files = discover_audio_files(raw_dir)
    out_root = raw_dir if in_place else raw_dir.parent / "normalized"

    for ap in files:
        y, sr = load_audio(ap)
        y = normalize_audio(y)
        if in_place and ap.suffix.lower() != ".wav":
            dest = ap.with_suffix(".wav")
        elif in_place:
            dest = ap
        else:
            dest = out_root / ap.relative_to(raw_dir)
            dest = dest.with_suffix(".wav")

        save_wav(dest, y, sr)
        console.print(f"Normalized: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize audio dataset")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    normalize_all(args.raw_dir, args.in_place)


if __name__ == "__main__":
    main()
