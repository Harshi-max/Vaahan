"""Text normalization for fair ASR metric comparison."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for WER/CER."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_entity(text: str) -> str:
    """Aggressive normalization for locality / entity matching."""
    text = normalize_text(text)
    # Common ASR substitutions for Indian localities
    replacements = {
        "bengaluru": "bangalore",
        "bengalooru": "bangalore",
        "mumbai": "mumbai",
        "bombay": "mumbai",
        "chennai": "chennai",
        "madras": "chennai",
        "kolkata": "kolkata",
        "calcutta": "kolkata",
        "gurugram": "gurgaon",
        "gurgaon": "gurgaon",
        "noida": "noida",
        "hyderabad": "hyderabad",
        "secunderabad": "hyderabad",
    }
    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    return text


def tokenize(text: str) -> list[str]:
    return normalize_text(text).split()


def extract_locality_candidates(transcript: str, known_localities: list[str]) -> list[str]:
    """Find which known locality names appear in transcript."""
    norm_transcript = normalize_entity(transcript)
    found: list[str] = []
    for loc in known_localities:
        if normalize_entity(loc) in norm_transcript:
            found.append(loc)
    return found
