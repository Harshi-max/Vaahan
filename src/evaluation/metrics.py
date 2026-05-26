"""WER, CER, and token-level metrics via JiWER."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
from sklearn.metrics import precision_recall_fscore_support

from src.utils.normalization import normalize_text, tokenize


@dataclass
class TranscriptMetrics:
    wer: float
    cer: float
    exact_match: bool
    token_precision: float
    token_recall: float
    token_f1: float


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


def compute_token_metrics(reference: str, hypothesis: str) -> tuple[float, float, float]:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0

    ref_set = set(ref_tokens)
    hyp_set = set(hyp_tokens)
    all_tokens = sorted(ref_set | hyp_set)
    y_true = [1 if t in ref_set else 0 for t in all_tokens]
    y_pred = [1 if t in hyp_set else 0 for t in all_tokens]
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def compute_all_metrics(reference: str, hypothesis: str) -> TranscriptMetrics:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    p, r, f1 = compute_token_metrics(reference, hypothesis)
    return TranscriptMetrics(
        wer=compute_wer(reference, hypothesis),
        cer=compute_cer(reference, hypothesis),
        exact_match=ref_norm == hyp_norm,
        token_precision=p,
        token_recall=r,
        token_f1=f1,
    )
