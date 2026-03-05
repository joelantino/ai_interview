"""
evaluation_service.py
---------------------
First-pass evaluation using Mistral.

For each Q&A pair:
  - Detects technical concepts mentioned in the answer
  - Identifies missing key concepts
  - Produces an initial technical score (0-10)
  - Assesses communication and confidence scores
"""

import json
import re
from typing import Dict, Any
from backend.groq_client import ask_mistral, extract_json

# ---------------------------------------------------------------------------
# Domain concept banks (question-keyword mapping)
# ---------------------------------------------------------------------------
CONCEPT_BANKS: Dict[str, list] = {
    "rest api": ["stateless", "resource", "uri", "http", "get", "post", "put",
                 "delete", "patch", "client", "server", "endpoint", "status code",
                 "json", "idempotent"],
    "microservices": ["service", "decoupled", "independent", "api gateway",
                      "docker", "kubernetes", "fault tolerance", "scalability",
                      "service discovery", "circuit breaker"],
    "database": ["sql", "nosql", "acid", "transaction", "index", "normalization",
                 "foreign key", "primary key", "join", "query", "schema"],
    "docker": ["container", "image", "dockerfile", "volume", "network", "compose",
               "registry", "layer", "build", "run"],
    "machine learning": ["model", "training", "validation", "overfitting",
                         "underfitting", "accuracy", "loss", "epoch",
                         "feature", "label", "dataset"],
}

_EVALUATION_SCHEMA = """{
  "concepts_detected": ["list of technical terms mentioned"],
  "missing_concepts": ["list of important concepts not mentioned"],
  "technical_score": <0-10>,
  "communication_score": <0-10>,
  "confidence_score": <0-10>,
  "reasoning": "brief justification"
}"""


def _build_eval_prompt(question: str, answer: str) -> str:
    return f"""You are a senior technical interviewer evaluating a candidate's answer.

SCORING RUBRIC:
  10 = Expert explanation with examples
   8 = Mostly correct, minor gaps
   6 = Basic understanding, missing concepts
   4 = Weak explanation
   2 = Incorrect or off-topic answer

TASK:
Evaluate the candidate's answer to the interview question below.
Return ONLY a valid JSON object matching this exact schema:
{_EVALUATION_SCHEMA}

Interview Question:
\"\"\"{question}\"\"\"

Candidate Answer:
\"\"\"{answer}\"\"\"

JSON evaluation:"""


def _keyword_score(question: str, answer: str) -> float:
    """
    Compute keyword coverage ratio against the closest concept bank entry.
    Returns a score in [0, 10].
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    best_ratio = 0.0
    for domain, keywords in CONCEPT_BANKS.items():
        if any(kw in q_lower for kw in domain.split()):
            hits = sum(1 for kw in keywords if kw in a_lower)
            ratio = hits / len(keywords)
            best_ratio = max(best_ratio, ratio)

    # If no domain matched, do a generic keyword check
    if best_ratio == 0.0:
        tokens = set(re.sub(r"[^a-z\s]", "", a_lower).split())
        if len(tokens) >= 20:
            best_ratio = 0.6   # reasonable attempt
        elif len(tokens) >= 10:
            best_ratio = 0.4
        else:
            best_ratio = 0.2

    return round(best_ratio * 10, 2)


def evaluate_answer(question: str, answer: str) -> Dict[str, Any]:
    """
    First-pass evaluation via Mistral.

    Returns:
        {
          "concepts_detected": [...],
          "missing_concepts": [...],
          "technical_score": float,
          "communication_score": float,
          "confidence_score": float,
          "keyword_score": float,
          "reasoning": str
        }
    """
    prompt  = _build_eval_prompt(question, answer)
    raw     = ask_mistral(prompt)
    result  = extract_json(raw)

    keyword_sc = _keyword_score(question, answer)

    if result is None:
        # Graceful degradation — return a partially scored result
        return {
            "concepts_detected":  [],
            "missing_concepts":   [],
            "technical_score":    4,
            "communication_score": 4,
            "confidence_score":   4,
            "keyword_score":      keyword_sc,
            "reasoning":          "LLM returned unparseable output.",
        }

    result["keyword_score"] = keyword_sc
    # Ensure all expected keys exist
    result.setdefault("concepts_detected", [])
    result.setdefault("missing_concepts", [])
    result.setdefault("technical_score", 4)
    result.setdefault("communication_score", 5)
    result.setdefault("confidence_score", 5)
    result.setdefault("reasoning", "")

    return result
