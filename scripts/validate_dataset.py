#!/usr/bin/env python3
"""Validate audio dataset against ground truth metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio import discover_audio_files, get_duration_seconds, SUPPORTED_EXTENSIONS  # noqa: E402
from src.utils.ingestion import run_ingestion_pipeline  # noqa: E402

console = Console()

MIN_DURATION = 1.0
MAX_DURATION = 60.0


def validate(data_dir: Path) -> bool:
    raw_root = data_dir / "raw"
    metadata_path = data_dir / "metadata" / "ground_truth.csv"
    ok = True

    if not metadata_path.exists():
        console.print("[red]Missing ground_truth.csv[/]")
        return False

    gt = pd.read_csv(metadata_path)
    audio_files = discover_audio_files(raw_root)
    audio_names = {p.name for p in audio_files}

    table = Table(title="Dataset Validation")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    # Ground truth rows
    for _, row in gt.iterrows():
        fname = row["filename"]
        if fname not in audio_names:
            table.add_row(fname, "[red]MISSING[/]", "Audio file not found")
            ok = False
        else:
            ap = next(p for p in audio_files if p.name == fname)
            dur = get_duration_seconds(ap)
            if dur < MIN_DURATION or dur > MAX_DURATION:
                table.add_row(
                    fname, "[yellow]WARN[/]", f"Duration {dur:.1f}s out of range"
                )
            else:
                table.add_row(fname, "[green]OK[/]", f"{dur:.1f}s, {row['condition']}")

    # Orphan audio
    gt_names = set(gt["filename"])
    orphans = audio_names - gt_names
    for name in orphans:
        table.add_row(name, "[yellow]ORPHAN[/]", "No ground truth entry")
        ok = False

    console.print(table)
    console.print(f"\nAudio files: {len(audio_files)} | Ground truth rows: {len(gt)}")
    console.print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    console.print("[bold green]Running automatic ingestion and validation pipeline...[/]")
    ingestion_result = run_ingestion_pipeline(args.data_dir)
    for warning in ingestion_result.get("warnings", []):
        console.print(f"[yellow]WARNING[/] {warning}")
    console.print(f"[green]Ingestion complete:[/] {ingestion_result['audio_count']} files processed")
    console.print(f"[green]Metadata path:[/] {args.data_dir / 'metadata' / 'ground_truth.csv'}")
    console.print(f"[green]Processed audio dir:[/] {args.data_dir / 'processed'}")
    console.print(f"[green]Validation report:[/] {ingestion_result['validation_report']}\n")

    success = validate(args.data_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
