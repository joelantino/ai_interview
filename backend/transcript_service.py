"""
transcript_service.py
---------------------
Converts audio files to plaintext transcripts via OpenAI Whisper (local model).
Supports any meeting platform recording: Zoom, Google Meet, Teams, etc.
"""

import os
import whisper

_model_cache: dict = {}
_DEFAULT_MODEL = "base"  # Options: tiny, base, small, medium, large


def _load_model(model_name: str = _DEFAULT_MODEL) -> whisper.Whisper:
    """Load (and cache) a Whisper model by name."""
    if model_name not in _model_cache:
        print(f"[Whisper] Loading model: {model_name} …")
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def generate_transcript(
    audio_path: str, 
    model_name: str = _DEFAULT_MODEL,
    language: str | None = None,
    translate: bool = False
) -> str:
    """
    Transcribe an audio file and return the full transcript text.
    Whisper handles 99+ languages automatically.

    Args:
        audio_path: Absolute path to audio/video file.
        model_name: Whisper model size.
        language:   ISO code (e.g. 'hi', 'fr'). If None, it auto-detects.
        translate:  If True, translates any non-English speech into English.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model(model_name)

    try:
        # task="translate" is a native Whisper feature
        task = "translate" if translate else "transcribe"
        result = model.transcribe(
            audio_path, 
            fp16=False, 
            language=language,
            task=task
        )
    except Exception as exc:
        raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    transcript = result.get("text", "").strip()
    detected_lang = result.get("language", "unknown")
    return {"text": transcript, "language": detected_lang}
