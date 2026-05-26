"""Latency aggregation and RTF computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LatencyStats:
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    mean_rtf: float


def compute_rtf(latency_seconds: float, audio_duration_seconds: float) -> float:
    if audio_duration_seconds <= 0:
        return 0.0
    return latency_seconds / audio_duration_seconds


def aggregate_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model latency summary from results dataframe."""
    if df.empty or "latency_seconds" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("model_name")
        .agg(
            mean_latency=("latency_seconds", "mean"),
            p50_latency=("latency_seconds", lambda x: float(np.percentile(x, 50))),
            p95_latency=("latency_seconds", lambda x: float(np.percentile(x, 95))),
            mean_rtf=("rtf", "mean"),
            n_samples=("latency_seconds", "count"),
        )
        .reset_index()
    )
    return grouped
