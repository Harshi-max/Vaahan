"""Optional FastAPI server for single-file transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from src.models.registry import get_active_models

app = FastAPI(title="ASR Shootout API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    models = get_active_models()
    return {"status": "ok", "models": [m.name for m in models]}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query(default="faster_whisper"),
) -> JSONResponse:
    models = {m.name: m for m in get_active_models()}
    if model not in models:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model. Available: {list(models.keys())}"},
        )

    suffix = Path(file.filename or "audio.wav").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    result = models[model].transcribe(tmp_path)
    tmp_path.unlink(missing_ok=True)

    return JSONResponse(
        {
            "model": result.model_name,
            "transcript": result.transcript,
            "latency_seconds": result.latency_seconds,
            "confidence": result.confidence,
            "error": result.error,
        }
    )
