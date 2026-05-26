"""Locality entity extraction accuracy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.normalization import normalize_entity


@dataclass
class EntityMetrics:
    locality_correct: bool
    locality_in_transcript: bool
    partial_match: bool
    extracted_entity: str | None


def locality_in_text(locality: str, text: str) -> bool:
    norm_loc = normalize_entity(locality)
    norm_text = normalize_entity(text)
    if not norm_loc:
        return False
    return norm_loc in norm_text


def partial_locality_match(locality: str, text: str, min_ratio: float = 0.6) -> bool:
    """Check if significant substring of locality appears (handles truncation)."""
    norm_loc = normalize_entity(locality).replace(" ", "")
    norm_text = normalize_entity(text).replace(" ", "")
    if len(norm_loc) < 3:
        return norm_loc in norm_text
    for i in range(len(norm_loc) - 2):
        chunk = norm_loc[i : i + max(3, len(norm_loc) // 2)]
        if chunk in norm_text:
            return True
    return False


def extract_best_entity_match(locality: str, text: str) -> str | None:
    if locality_in_text(locality, text):
        return locality
    # Fuzzy: find longest matching token sequence
    loc_tokens = normalize_entity(locality).split()
    text_norm = normalize_entity(text)
    for tok in loc_tokens:
        if len(tok) >= 4 and tok in text_norm:
            return tok
    return None


def compute_entity_metrics(locality: str, hypothesis: str) -> EntityMetrics:
    correct = locality_in_text(locality, hypothesis)
    partial = partial_locality_match(locality, hypothesis) if not correct else False
    extracted = extract_best_entity_match(locality, hypothesis)
    return EntityMetrics(
        locality_correct=correct,
        locality_in_transcript=correct or partial,
        partial_match=partial,
        extracted_entity=extracted,
    )


def detect_hallucinated_localities(
    hypothesis: str, known_localities: list[str], expected: str
) -> list[str]:
    """Find locality names in output that weren't expected (possible hallucination)."""
    norm_hyp = normalize_entity(hypothesis)
    hallucinated: list[str] = []
    expected_norm = normalize_entity(expected)
    for loc in known_localities:
        ln = normalize_entity(loc)
        if ln and ln in norm_hyp and ln not in expected_norm:
            hallucinated.append(loc)
    return hallucinated


def detect_code_switch_issues(reference: str, hypothesis: str) -> bool:
    """Heuristic: Latin/Hindi script mismatch between ref and hyp."""
    ref_has_devanagari = bool(re.search(r"[\u0900-\u097F]", reference))
    hyp_has_devanagari = bool(re.search(r"[\u0900-\u097F]", hypothesis))
    ref_has_latin = bool(re.search(r"[a-zA-Z]", reference))
    hyp_has_latin = bool(re.search(r"[a-zA-Z]", hypothesis))
    return (ref_has_devanagari != hyp_has_devanagari) or (
        ref_has_latin != hyp_has_latin and len(hypothesis) > 10
    )
