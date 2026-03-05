"""
qa_extractor.py
---------------
Extracts question-answer pairs from a cleaned interview transcript.

Two strategies are used in order of preference:
  1. Pattern matching  – covers transcripts with explicit "Interviewer:" / "Candidate:" prefixes.
  2. LLM-assisted     – falls back to Mistral when the transcript is unstructured prose.
"""

import re
from typing import List, Dict
from backend.groq_client import ask_mistral, extract_json

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
QAPair = Dict[str, str]   # {"question": "...", "answer": "..."}

# ---------------------------------------------------------------------------
# Speaker role aliases (case-insensitive)
# ---------------------------------------------------------------------------
INTERVIEWER_TOKENS = r"(interviewer|hr|recruiter|panel|moderator|host|question)"
CANDIDATE_TOKENS   = r"(candidate|applicant|interviewee|answer|response)"

# A line that starts with one of the role labels followed by colon
_ROLE_LINE = re.compile(
    rf"^(?P<role>{INTERVIEWER_TOKENS}|{CANDIDATE_TOKENS})\s*:\s*(?P<text>.+)$",
    re.IGNORECASE,
)


def _pattern_extract(transcript: str) -> List[QAPair]:
    """
    Extract Q&A pairs when the transcript uses explicit role labels
    (e.g., "Interviewer: …" / "Candidate: …").
    """
    pairs: List[QAPair] = []
    current_question: str = ""
    current_answer_parts: List[str] = []

    def _flush():
        if current_question and current_answer_parts:
            pairs.append(
                {
                    "question": current_question.strip(),
                    "answer": " ".join(current_answer_parts).strip(),
                }
            )

    for line in transcript.splitlines():
        line = line.strip()
        m = _ROLE_LINE.match(line)
        if not m:
            # Continuation line – append to current answer if one is open
            if current_answer_parts:
                current_answer_parts.append(line)
            continue

        role_raw = m.group("role").lower()
        text     = m.group("text").strip()

        if re.match(INTERVIEWER_TOKENS, role_raw, re.IGNORECASE):
            _flush()
            current_question      = text
            current_answer_parts  = []
        elif re.match(CANDIDATE_TOKENS, role_raw, re.IGNORECASE):
            current_answer_parts.append(text)

    _flush()
    return pairs


def _llm_extract(transcript: str) -> List[QAPair]:
    """
    Ask Mistral to extract Q&A pairs from an unstructured transcript.
    Returns a list of QAPair dicts.
    """
    prompt = f"""You are an expert at analysing interview transcripts.
Extract all question-answer pairs from the transcript below.

Rules:
- Only include questions asked by the interviewer.
- Pair each question with the candidate's direct answer.
- Return a valid JSON array with this schema:
  [{{"question": "...", "answer": "..."}}]
- Do not include any text outside the JSON array.

Transcript:
\"\"\"
{transcript[:4000]}
\"\"\"

JSON array of Q&A pairs:"""

    raw = ask_mistral(prompt)

    # Try to parse a JSON array from the response
    import json
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                        result.append(
                            {"question": str(item["question"]), "answer": str(item["answer"])}
                        )
                return result
        except json.JSONDecodeError:
            pass
    return []


def extract_qa_pairs(transcript: str) -> List[QAPair]:
    """
    Public entry point.

    Tries pattern-based extraction first; falls back to LLM extraction
    when fewer than 2 pairs are found.

    Args:
        transcript: Cleaned interview transcript.

    Returns:
        List of {"question": str, "answer": str} dicts.
    """
    pairs = _pattern_extract(transcript)
    if len(pairs) >= 2:
        return pairs

    # Fallback to Mistral-assisted extraction
    return _llm_extract(transcript)
