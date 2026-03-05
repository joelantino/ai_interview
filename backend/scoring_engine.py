"""
scoring_engine.py
-----------------
Weighted score aggregation and consistency checking.

Formula:
  final_score = 0.50 × llm_score
              + 0.20 × keyword_score
              + 0.20 × validated_score
              + 0.10 × communication_score

Consistency check:
  If two evaluation runs differ by > 2 points, a third run is performed
  and the median is taken.
"""

import statistics
from typing import Dict, Any, List

from backend.evaluation_service import evaluate_answer
from backend.feedback_service   import generate_feedback

# ---------------------------------------------------------------------------
# Score weights
# ---------------------------------------------------------------------------
W_LLM          = 0.50
W_KEYWORD      = 0.20
W_VALIDATION   = 0.20
W_COMMUNICATION = 0.10

CONSISTENCY_THRESHOLD = 2.0


def _compute_weighted(
    llm_score: float,
    keyword_score: float,
    validated_score: float,
    communication_score: float,
) -> float:
    raw = (
        W_LLM           * llm_score
        + W_KEYWORD      * keyword_score
        + W_VALIDATION   * validated_score
        + W_COMMUNICATION * communication_score
    )
    return round(min(max(raw, 0), 10), 2)


def _evaluate_once(question: str, answer: str) -> Dict[str, Any]:
    """Run one full evaluation-feedback cycle and return all scores."""
    ev  = evaluate_answer(question, answer)
    fb  = generate_feedback(
        question,
        answer,
        ev["technical_score"],
        ev["missing_concepts"],
    )

    final = _compute_weighted(
        llm_score           = ev["technical_score"],
        keyword_score       = ev["keyword_score"],
        validated_score     = fb["validated_score"],
        communication_score = ev["communication_score"],
    )

    return {
        "evaluation":    ev,
        "feedback":      fb,
        "final_score":   final,
    }


def score_qa_pair(question: str, answer: str) -> Dict[str, Any]:
    """
    Evaluate a single Q&A pair.
    
    Optimized: Removed consistency check (multiple runs) to provide 
    significantly faster results on local hardware.
    """
    chosen = _evaluate_once(question, answer)

    return {
        "question":           question,
        "answer":             answer,
        "concepts_detected":  chosen["evaluation"]["concepts_detected"],
        "missing_concepts":   chosen["evaluation"]["missing_concepts"],
        "scores": {
            "technical":      chosen["evaluation"]["technical_score"],
            "keyword":        chosen["evaluation"]["keyword_score"],
            "validated":      chosen["feedback"]["validated_score"],
            "communication":  chosen["evaluation"]["communication_score"],
            "confidence":     chosen["evaluation"]["confidence_score"],
            "final":          chosen["final_score"],
        },
        "feedback": {
            "strengths":           chosen["feedback"]["strengths"],
            "mistakes":            chosen["feedback"]["mistakes"],
            "correct_explanation": chosen["feedback"]["correct_explanation"],
            "improvement":         chosen["feedback"]["improvement"],
        },
    }


def aggregate_scores(scored_pairs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute interview-level aggregate scores across all Q&A pairs.

    Returns:
        {
          "technical":      avg,
          "problem_solving": avg,   (proxy: keyword score)
          "communication":  avg,
          "confidence":     avg,
          "overall":        avg of all above
        }
    """
    if not scored_pairs:
        return {"technical": 0, "problem_solving": 0,
                "communication": 0, "confidence": 0, "overall": 0}

    def _avg(key, sub=None):
        vals = [
            p["scores"][key] if sub is None else p["scores"][key]
            for p in scored_pairs
        ]
        return round(sum(vals) / len(vals), 2)

    technical    = _avg("technical")
    prob_solving = _avg("keyword")     # keyword coverage proxies problem-solving
    comm         = _avg("communication")
    confidence   = _avg("confidence")
    overall      = round((technical + prob_solving + comm + confidence) / 4, 2)

    return {
        "technical":       technical,
        "problem_solving": prob_solving,
        "communication":   comm,
        "confidence":      confidence,
        "overall":         overall,
    }
