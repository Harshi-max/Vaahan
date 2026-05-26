#!/usr/bin/env python3
"""Generate demo WAV files and ground_truth.csv for pipeline smoke tests."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio import save_wav, TARGET_SAMPLE_RATE  # noqa: E402

CONDITIONS = ["quiet", "traffic", "rushed", "whispered", "phonecall"]

SAMPLES = [
    {"filename": "sample_01_koramangala_quiet.wav", "locality_name": "Koramangala", "language": "hinglish", "condition": "quiet", "reference_transcript": "mera address hai Koramangala 5th block Bangalore"},
    {"filename": "sample_02_andheri_traffic.wav", "locality_name": "Andheri West", "language": "hinglish", "condition": "traffic", "reference_transcript": "delivery Andheri West ke paas metro station"},
    {"filename": "sample_03_connaught_rushed.wav", "locality_name": "Connaught Place", "language": "english", "condition": "rushed", "reference_transcript": "pickup from Connaught Place block C New Delhi"},
    {"filename": "sample_04_jayanagar_whispered.wav", "locality_name": "Jayanagar", "language": "hinglish", "condition": "whispered", "reference_transcript": "customer bol raha hai Jayanagar 4th block"},
    {"filename": "sample_05_hitec_phonecall.wav", "locality_name": "HITEC City", "language": "hinglish", "condition": "phonecall", "reference_transcript": "office HITEC City Hyderabad near cyber towers"},
    {"filename": "sample_06_saltlake_quiet.wav", "locality_name": "Salt Lake", "language": "hinglish", "condition": "quiet", "reference_transcript": "address Salt Lake sector 5 Kolkata"},
    {"filename": "sample_07_whitefield_traffic.wav", "locality_name": "Whitefield", "language": "hinglish", "condition": "traffic", "reference_transcript": "Whitefield mein delivery karna hai"},
    {"filename": "sample_08_jubilee_quiet.wav", "locality_name": "Jubilee Hills", "language": "english", "condition": "quiet", "reference_transcript": "The address is in Jubilee Hills"},
    {"filename": "sample_09_marine_rushed.wav", "locality_name": "Marine Lines", "language": "hinglish", "condition": "rushed", "reference_transcript": "jaldi Marine Lines pahuncha do"},
    {"filename": "sample_10_bandra_phonecall.wav", "locality_name": "Bandra", "language": "hinglish", "condition": "phonecall", "reference_transcript": "Bandra East mein customer hai"},
    {"filename": "sample_11_indiranagar_quiet.wav", "locality_name": "Indiranagar", "language": "hinglish", "condition": "quiet", "reference_transcript": "Indiranagar 100 feet road pe pickup point hai"},
    {"filename": "sample_12_fort_traffic.wav", "locality_name": "Fort", "language": "english", "condition": "traffic", "reference_transcript": "In Fort area near Gateway of India"},
    {"filename": "sample_13_gachibowli_whispered.wav", "locality_name": "Gachibowli", "language": "hinglish", "condition": "whispered", "reference_transcript": "Gachibowli road par delivery ka pickup"},
    {"filename": "sample_14_madhapur_rushed.wav", "locality_name": "Madhapur", "language": "hinglish", "condition": "rushed", "reference_transcript": "Madhapur mein jaldi se jaldi deliver karo"},
    {"filename": "sample_15_kondapur_quiet.wav", "locality_name": "Kondapur", "language": "hinglish", "condition": "quiet", "reference_transcript": "customer Kondapur mein raha karte hain"},
    {"filename": "sample_16_kukatpally_traffic.wav", "locality_name": "Kukatpally", "language": "english", "condition": "traffic", "reference_transcript": "Kukatpally area has heavy traffic"},
    {"filename": "sample_17_banjara_phonecall.wav", "locality_name": "Banjara Hills", "language": "hinglish", "condition": "phonecall", "reference_transcript": "Banjara Hills mein delivery point"},
    {"filename": "sample_18_cyber_whispered.wav", "locality_name": "Cyber Towers", "language": "english", "condition": "whispered", "reference_transcript": "Office location at Cyber Towers"},
    {"filename": "sample_19_dadar_rushed.wav", "locality_name": "Dadar", "language": "hinglish", "condition": "rushed", "reference_transcript": "Dadar mein jaldi pickup karna hai"},
    {"filename": "sample_20_borivali_quiet.wav", "locality_name": "Borivali", "language": "hinglish", "condition": "quiet", "reference_transcript": "pickup address Borivali East ka dena"},
]


def synthesize_speech_like_audio(
    duration: float = 3.0,
    sr: int = TARGET_SAMPLE_RATE,
    condition: str = "quiet",
) -> np.ndarray:
    """Generate synthetic audio mimicking speech formants + condition noise."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # Formant-like harmonics
    f0 = 120 + np.random.uniform(-10, 10)
    signal = (
        0.4 * np.sin(2 * np.pi * f0 * t)
        + 0.2 * np.sin(2 * np.pi * 2.5 * f0 * t)
        + 0.1 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    # Amplitude envelope (syllable-like)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    signal = signal * envelope

    rng = np.random.default_rng(42)
    noise_level = {
        "quiet": 0.02,
        "traffic": 0.15,
        "rushed": 0.08,
        "whispered": 0.05,
        "phonecall": 0.12,
    }.get(condition, 0.05)

    noise = rng.normal(0, noise_level, size=signal.shape).astype(np.float32)
    y = signal + noise

    if condition == "phonecall":
        # Band-limit to telephony bandwidth
        from scipy.signal import butter, filtfilt

        b, a = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
        y = filtfilt(b, a, y).astype(np.float32)

    if condition == "whispered":
        y = y * 0.3

    peak = np.max(np.abs(y)) + 1e-8
    return (y / peak * 0.8).astype(np.float32)


def main() -> None:
    raw_root = ROOT / "data" / "raw"
    metadata_path = ROOT / "data" / "metadata" / "ground_truth.csv"

    rows_written: list[dict] = []
    for sample in SAMPLES:
        cond = sample["condition"]
        out_path = raw_root / cond / sample["filename"]
        if out_path.exists():
            rows_written.append(sample)
            continue
        y = synthesize_speech_like_audio(condition=cond)
        save_wav(out_path, y)
        rows_written.append(sample)
        print(f"Created {out_path}")

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "locality_name",
        "language",
        "condition",
        "reference_transcript",
    ]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_written)

    print(f"Ground truth: {metadata_path} ({len(rows_written)} samples)")


if __name__ == "__main__":
    main()
