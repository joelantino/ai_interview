"""
app.py
------
FastAPI application for the AI Interview Evaluation System.

Endpoints:
  GET  /                     Health check
  POST /evaluate-audio       Upload audio file → full evaluation report
  POST /evaluate-transcript  Submit transcript text → full evaluation report
"""

import os
import shutil
import tempfile
import sys
from datetime import datetime, timezone

# Add bin folder to PATH so portable ffmpeg is found
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bin_path = os.path.join(_BASE_DIR, "bin")
if os.path.exists(bin_path):
    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.transcript_service  import generate_transcript
from backend.transcript_cleaner  import clean_transcript
from backend.qa_extractor        import extract_qa_pairs
from backend.scoring_engine      import score_qa_pair, aggregate_scores

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Interview Evaluation System",
    description=(
        "Local AI system that evaluates interview recordings using Whisper + "
        "Mistral + Llama3 (via Ollama). No paid APIs required."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class TranscriptRequest(BaseModel):
    transcript: str
    job_role: str | None = None


class EvaluationReport(BaseModel):
    evaluated_at: str
    job_role: str | None
    total_questions: int
    scores: dict
    qa_details: list
    summary_feedback: dict


import asyncio

# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------
async def _run_pipeline(transcript_data: dict, job_role: str | None) -> dict:
    """Full evaluation pipeline from raw transcript to final report."""
    
    transcript_raw = transcript_data.get("text", "")
    if not transcript_raw.strip():
        raise HTTPException(status_code=400, detail="Transcript is empty.")

    # 1. Clean transcript
    transcript_clean = clean_transcript(transcript_raw)

    # 2. Extract Q&A pairs
    qa_pairs = extract_qa_pairs(transcript_clean)
    if not qa_pairs:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract any question-answer pairs from the transcript. "
                "Ensure the transcript contains interviewer questions and candidate answers."
            ),
        )

    # 3. Score each Q&A pair SEQUENTIALLY
    # Sequential processing is much more stable for local Ollama instances
    # to avoid resource contention and timeouts.
    scored_pairs = []
    for p in qa_pairs:
        # score_qa_pair is now faster due to previous optimization
        result = await asyncio.to_thread(score_qa_pair, p["question"], p["answer"])
        scored_pairs.append(result)

    # 4. Aggregate interview-level scores
    agg = aggregate_scores(scored_pairs)

    # 5. Compile summary feedback across all pairs
    all_strengths    = []
    all_mistakes     = []
    all_improvements = []
    for sp in scored_pairs:
        all_strengths    += sp["feedback"].get("strengths", [])
        all_mistakes     += sp["feedback"].get("mistakes", [])
        all_improvements.append(sp["feedback"].get("improvement", ""))

    summary_feedback = {
        "strengths":   list(dict.fromkeys(all_strengths))[:5],   # deduplicated, top 5
        "mistakes":    list(dict.fromkeys(all_mistakes))[:5],
        "improvement": " | ".join([i for i in all_improvements if i])[:500],
    }

    return {
        "evaluated_at":    datetime.now(timezone.utc).isoformat(),
        "job_role":        job_role,
        "language_detected": transcript_data.get("language", "unknown"),
        "total_questions": len(scored_pairs),
        "scores": {
            "technical":       agg["technical"],
            "problem_solving": agg["problem_solving"],
            "communication":   agg["communication"],
            "confidence":      agg["confidence"],
            "overall":         agg["overall"],
        },
        "qa_details":      scored_pairs,
        "summary_feedback": summary_feedback,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["UI"], include_in_schema=False)
def serve_ui():
    """Serve the web evaluation UI."""
    return FileResponse(os.path.join(_BASE_DIR, "index.html"))


@app.get("/health", tags=["Health"])
def health_check():
    """Quick health check to verify the service is running."""
    return {
        "status":  "ok",
        "service": "AI Interview Evaluation System",
        "version": "1.0.0",
    }


@app.post("/evaluate-audio", tags=["Evaluation"])
async def evaluate_audio(
    file: UploadFile     = File(..., description="Audio/Video file"),
    job_role: str        = Form(None),
    whisper_model: str   = Form("base"),
    language: str | None = Form(None, description="ISO code (e.g. 'hi')"),
    translate: bool      = Form(False, description="Translate to English?"),
):
    """
    Upload an interview audio/video and receive a full report.
    Automatically handles 99+ languages.
    """
    ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".flac"}
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        transcript_raw = generate_transcript(
            tmp_path, 
            model_name=whisper_model,
            language=language,
            translate=translate
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        os.remove(tmp_path)

    return await _run_pipeline(transcript_raw, job_role)


@app.post("/evaluate-transcript", tags=["Evaluation"])
async def evaluate_transcript(request: TranscriptRequest):
    """
    Submit a pre-existing transcript text and receive a full structured
    evaluation report without audio transcription.
    """
    return await _run_pipeline({"text": request.transcript, "language": "manual"}, request.job_role)
