"""
transcript_cleaner.py
---------------------
Normalises raw Whisper transcripts before Q&A extraction.

Operations:
  1. Fix common technical term casing/spelling
  2. Remove verbal filler words
  3. Strip duplicate whitespace and trailing punctuation artefacts
"""

import re
import nltk

# Download punkt tokeniser if not already present (silent fallback)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ---------------------------------------------------------------------------
# Domain-specific corrections  (lower-case pattern → canonical form)
# ---------------------------------------------------------------------------
TERM_CORRECTIONS: dict[str, str] = {
    r"\bmicro\s*service(s?)\b": r"microservice\1",
    r"\brest\s*api(s?)\b": r"REST API\1",
    r"\bsql\b": "SQL",
    r"\bnosql\b": "NoSQL",
    r"\bjson\b": "JSON",
    r"\bhttp\b": "HTTP",
    r"\bhttps\b": "HTTPS",
    r"\bhtml\b": "HTML",
    r"\bcss\b": "CSS",
    r"\bjavascript\b": "JavaScript",
    r"\btypescript\b": "TypeScript",
    r"\bpython\b": "Python",
    r"\bdocker\b": "Docker",
    r"\bkubernetes\b": "Kubernetes",
    r"\bgit\b": "Git",
    r"\baws\b": "AWS",
    r"\bgcp\b": "GCP",
    r"\bazure\b": "Azure",
    r"\bapi(s?)\b": r"API\1",
    r"\burl(s?)\b": r"URL\1",
    r"\buri(s?)\b": r"URI\1",
    r"\bcrud\b": "CRUD",
    r"\borm\b": "ORM",
    r"\bci\b": "CI",
    r"\bcd\b": "CD",
    r"\bml\b": "ML",
    r"\bai\b": "AI",
}

# ---------------------------------------------------------------------------
# Filler words to strip (whole words only)
# ---------------------------------------------------------------------------
FILLER_WORDS = [
    r"\bum+\b",
    r"\buh+\b",
    r"\blike\b",
    r"\byou\s+know\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\bright\b",
    r"\bokay\b",
    r"\bso\b",
    r"\bwell\b",
    r"\bi\s+mean\b",
]


def _apply_term_corrections(text: str) -> str:
    for pattern, replacement in TERM_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _remove_fillers(text: str) -> str:
    for pattern in FILLER_WORDS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def _normalise_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_transcript(transcript: str) -> str:
    """
    Full cleaning pipeline.

    Args:
        transcript: Raw transcript string from Whisper.

    Returns:
        Cleaned, normalised transcript.
    """
    text = transcript
    text = _apply_term_corrections(text)
    text = _remove_fillers(text)
    text = _normalise_whitespace(text)
    return text
