"""ASR Shootout CLI — full benchmark pipeline."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import get_settings
from src.evaluation.analysis import (
    failure_analysis,
    generate_all_charts,
    generate_report,
)
from src.evaluation.entity_accuracy import (
    compute_entity_metrics,
    detect_code_switch_issues,
    detect_hallucinated_localities,
)
from src.evaluation.latency import compute_rtf
from src.evaluation.metrics import compute_all_metrics
from src.models.base import ASRResult, BaseASR
from src.models.registry import get_active_models
from src.utils.audio import discover_audio_files, get_condition_from_path, get_duration_seconds
from src.utils.helpers import cache_key, load_cache, save_cache, setup_logging

console = Console()
logger = logging.getLogger(__name__)


def load_ground_truth(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {metadata_path}\n"
            "Run setup.sh or scripts/prepare_demo_data.py first."
        )
    df = pd.read_csv(metadata_path)
    required = {
        "filename",
        "locality_name",
        "language",
        "condition",
        "duration_seconds",
        "sample_rate",
        "reference_transcript",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth.csv missing columns: {missing}")
    return df


def match_audio_to_metadata(
    audio_files: list[Path], gt_df: pd.DataFrame, raw_root: Path
) -> list[dict]:
    """Join discovered audio with ground truth rows."""
    gt_by_name = {row["filename"]: row for _, row in gt_df.iterrows()}
    samples: list[dict] = []
    for ap in audio_files:
        name = ap.name
        if name not in gt_by_name:
            logger.warning("No ground truth for %s — skipping", name)
            continue
        row = gt_by_name[name]
        samples.append(
            {
                "audio_path": ap,
                "filename": name,
                "locality_name": row["locality_name"],
                "language": row["language"],
                "condition": row.get("condition") or get_condition_from_path(ap, raw_root),
                "reference_transcript": row["reference_transcript"],
            }
        )
    return samples


def run_model_on_sample(
    model: BaseASR,
    sample: dict,
    cache_dir: Path | None,
    use_cache: bool,
) -> ASRResult:
    audio_path: Path = sample["audio_path"]
    key = cache_key(model.name, audio_path)

    if use_cache and cache_dir:
        cached = load_cache(cache_dir, key)
        if cached:
            return ASRResult(**cached)

    result = model.transcribe(audio_path)

    if use_cache and cache_dir and result.success:
        save_cache(
            cache_dir,
            key,
            {
                "model_name": result.model_name,
                "audio_path": result.audio_path,
                "transcript": result.transcript,
                "latency_seconds": result.latency_seconds,
                "confidence": result.confidence,
                "metadata": result.metadata,
                "error": result.error,
            },
        )
    return result


def evaluate_result(
    sample: dict,
    asr_result: ASRResult,
    all_localities: list[str],
) -> dict:
    ref = sample["reference_transcript"]
    hyp = asr_result.transcript
    metrics = compute_all_metrics(ref, hyp)
    entity = compute_entity_metrics(sample["locality_name"], hyp)
    duration = get_duration_seconds(Path(sample["audio_path"]))

    hallucinated = detect_hallucinated_localities(
        hyp, all_localities, sample["locality_name"]
    )

    return {
        "filename": sample["filename"],
        "model_name": asr_result.model_name,
        "condition": sample["condition"],
        "language": sample["language"],
        "locality_name": sample["locality_name"],
        "reference_transcript": ref,
        "hypothesis": hyp,
        "wer": metrics.wer,
        "cer": metrics.cer,
        "exact_match": metrics.exact_match,
        "token_precision": metrics.token_precision,
        "token_recall": metrics.token_recall,
        "token_f1": metrics.token_f1,
        "locality_correct": entity.locality_correct,
        "partial_match": entity.partial_match,
        "extracted_entity": entity.extracted_entity,
        "latency_seconds": asr_result.latency_seconds,
        "rtf": compute_rtf(asr_result.latency_seconds, duration),
        "confidence": asr_result.confidence,
        "audio_duration": duration,
        "code_switch_issue": detect_code_switch_issues(ref, hyp),
        "hallucination": len(hallucinated) > 0,
        "error": asr_result.error,
    }


def save_transcripts(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    for model in df["model_name"].unique():
        sub = df[df["model_name"] == model][
            ["filename", "hypothesis", "latency_seconds", "confidence", "error"]
        ]
        sub.to_csv(out_dir / f"{model}_transcripts.csv", index=False)


def run_benchmark(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    models: list[str] | None = None,
    max_workers: int | None = None,
) -> pd.DataFrame:
    from src.utils.ingestion import run_ingestion_pipeline
    
    settings = get_settings()
    raw_root = data_dir / "raw"
    processed_root = data_dir / "processed"
    metadata_path = data_dir / "metadata" / "ground_truth.csv"
    cache_dir = settings.cache_dir if settings.enable_cache else None

    console.print("\n[bold cyan]--- AUTOMATED INGESTION PIPELINE ---[/]")
    ingestion_result = run_ingestion_pipeline(data_dir)
    console.print(f"[green][OK] Detected {ingestion_result['audio_count']} audio files[/]")
    console.print(f"[green][OK] Classified conditions & generated metadata[/]")
    console.print(f"[green][OK] Normalized audio files[/]")
    for warning in ingestion_result.get("warnings", []):
        console.print(f"[yellow][WARN] {warning}[/]")

    gt_df = load_ground_truth(metadata_path)
    audio_root = processed_root if processed_root.exists() else raw_root
    audio_files = discover_audio_files(audio_root)
    if not audio_files:
        raise RuntimeError(f"No audio files found under {audio_root}")

    samples = match_audio_to_metadata(audio_files, gt_df, raw_root)
    if not samples:
        raise RuntimeError("No audio files matched ground_truth.csv filenames")

    all_localities = gt_df["locality_name"].unique().tolist()
    active_models = get_active_models(skip_unavailable=True)

    if models:
        active_models = [m for m in active_models if m.name in models]

    console.print(f"\n[bold cyan]--- ASR BENCHMARK EXECUTION ---[/]")
    console.print(f"[cyan]Running {len(active_models)} models on {len(samples)} samples[/]")

    # Warmup models sequentially
    for m in active_models:
        console.print(f"  [dim]Warming up {m.name}...[/]")
        try:
            m.warmup()
        except Exception as exc:
            logger.warning("Warmup failed for %s: %s", m.name, exc)

    workers = max_workers or settings.max_workers
    all_rows: list[dict] = []

    for model in active_models:
        console.print(f"\n[bold cyan]Model: {model.name}[/]")
        model_results: list[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Transcribing with {model.name}...", total=len(samples))

            if workers > 1 and model.name in ("deepgram", "google_stt"):
                # Cloud APIs: parallel requests
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(
                            run_model_on_sample,
                            model,
                            s,
                            cache_dir,
                            settings.enable_cache,
                        ): s
                        for s in samples
                    }
                    for fut in as_completed(futures):
                        sample = futures[fut]
                        asr_result = fut.result()
                        row = evaluate_result(sample, asr_result, all_localities)
                        model_results.append(row)
                        progress.advance(task)
            else:
                for sample in samples:
                    asr_result = run_model_on_sample(
                        model, sample, cache_dir, settings.enable_cache
                    )
                    row = evaluate_result(sample, asr_result, all_localities)
                    model_results.append(row)
                    progress.advance(task)

        all_rows.extend(model_results)

    df = pd.DataFrame(all_rows)

    # Persist outputs
    metrics_dir = output_dir / "metrics"
    charts_dir = output_dir / "charts"
    transcripts_dir = output_dir / "transcripts"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(metrics_dir / "full_results.csv", index=False)
    save_transcripts(all_rows, transcripts_dir)

    summary = (
        df.groupby("model_name")
        .agg(
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            entity_accuracy=("locality_correct", "mean"),
            exact_match_rate=("exact_match", "mean"),
            mean_latency=("latency_seconds", "mean"),
            mean_rtf=("rtf", "mean"),
            token_f1=("token_f1", "mean"),
        )
        .round(4)
    )
    summary.to_csv(metrics_dir / "summary_by_model.csv")

    failures = failure_analysis(df)
    with open(metrics_dir / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, default=str)

    chart_paths = generate_all_charts(df, charts_dir)
    models_run = df["model_name"].unique().tolist()
    generate_report(df, failures, chart_paths, report_path, models_run)

    console.print(f"\n[bold green]--- PIPELINE COMPLETE ---[/]")
    console.print(f"[green][OK] Ingestion & validation complete[/]")
    console.print(f"[green][OK] Models transcribed all samples[/]")
    console.print(f"[green][OK] Metrics computed & charts generated[/]")
    console.print(f"[green][OK] Report saved: {report_path}[/]")
    console.print(f"\n[bold cyan]Model Performance Summary:[/]")
    console.print(summary.to_string())
    return df


@click.group()
def cli() -> None:
    """ASR Shootout — Indian conversational speech benchmark."""
    setup_logging()


@cli.command("run")
@click.option("--data-dir", type=click.Path(exists=True), default="data")
@click.option("--output-dir", type=click.Path(), default="data/outputs")
@click.option("--report-path", type=click.Path(), default="reports/final_report.md")
@click.option("--models", multiple=True, help="Limit to specific model names")
@click.option("--max-workers", type=int, default=None)
def run_cmd(
    data_dir: str,
    output_dir: str,
    report_path: str,
    models: tuple[str, ...],
    max_workers: int | None,
) -> None:
    """Run full ASR benchmark pipeline."""
    run_benchmark(
        Path(data_dir),
        Path(output_dir),
        Path(report_path),
        models=list(models) if models else None,
        max_workers=max_workers,
    )


@cli.command("list-models")
def list_models_cmd() -> None:
    """Show which models are available."""
    for m in get_active_models(skip_unavailable=False):
        status = "[green]available[/]" if m.is_available() else "[red]unavailable[/]"
        console.print(f"  {m.name}: {status}")


@cli.command("streaming")
@click.option("--audio", type=click.Path(exists=True), required=True)
@click.option("--model", default="faster_whisper")
def streaming_cmd(audio: str, model: str) -> None:
    """Simulate streaming ASR on a single file."""
    from src.evaluation.streaming import simulate_streaming_asr
    from src.models.registry import get_active_models

    models = {m.name: m for m in get_active_models()}
    if model not in models:
        raise click.ClickException(f"Model not available: {model}")
    m = models[model]
    m.warmup()
    result = simulate_streaming_asr(m, Path(audio))
    console.print(f"Chunks: {result.chunks_processed}")
    console.print(f"Total latency: {result.total_latency_seconds:.2f}s")
    console.print(f"Final: {result.final_transcript}")


@cli.command("validate")
@click.option("--data-dir", type=click.Path(exists=True), default="data")
def validate_cmd(data_dir: str) -> None:
    """Validate dataset integrity."""
    import subprocess
    import sys

    script = Path(__file__).parent.parent / "scripts" / "validate_dataset.py"
    subprocess.run([sys.executable, str(script), "--data-dir", data_dir], check=True)


if __name__ == "__main__":
    cli()
