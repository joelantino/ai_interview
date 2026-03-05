"""
feedback_service.py
-------------------
Second-pass validation and corrective feedback using Llama3.

Tasks:
  - Validate the Mistral initial score
  - Identify candidate strengths
  - Pinpoint specific mistakes
  - Provide the correct explanation
  - Suggest targeted improvements
"""

from typing import Dict, Any
from backend.groq_client import ask_llama, extract_json

_FEEDBACK_SCHEMA = """{
  "validated_score": <0-10>,
  "strengths": ["strength 1", "strength 2"],
  "mistakes": ["mistake 1", "mistake 2"],
  "correct_explanation": "The complete correct answer or concept...",
  "improvement": "Specific study recommendations..."
}"""


def _build_feedback_prompt(
    question: str,
    answer: str,
    initial_score: float,
    missing_concepts: list,
) -> str:
    missing_str = ", ".join(missing_concepts) if missing_concepts else "none identified"
    return f"""You are a senior technical mentor reviewing a candidate's interview answer.

An initial AI evaluator gave the answer a score of {initial_score}/10 and flagged these missing concepts: {missing_str}.

Your tasks:
1. Validate or adjust the initial score (stay within ±2 unless clearly wrong).
2. List genuine strengths demonstrated.
3. Clearly list factual mistakes or critical omissions.
4. Provide a complete correct explanation of the topic.
5. Suggest specific improvement actions.

Return ONLY a valid JSON object matching this schema:
{_FEEDBACK_SCHEMA}

Interview Question:
\"\"\"{question}\"\"\"

Candidate Answer:
\"\"\"{answer}\"\"\"

JSON feedback:"""


def generate_feedback(
    question: str,
    answer: str,
    initial_score: float,
    missing_concepts: list,
) -> Dict[str, Any]:
    """
    Second-pass evaluation via Llama3.

    Args:
        question:         The interview question.
        answer:           The candidate's answer.
        initial_score:    Technical score from Mistral evaluation.
        missing_concepts: List of missing concepts from first pass.

    Returns:
        {
          "validated_score": float,
          "strengths": [...],
          "mistakes": [...],
          "correct_explanation": str,
          "improvement": str
        }
    """
    prompt = _build_feedback_prompt(question, answer, initial_score, missing_concepts)
    raw    = ask_llama(prompt)
    result = extract_json(raw)

    if result is None:
        return {
            "validated_score":     initial_score,
            "strengths":           [],
            "mistakes":            ["Could not parse LLM feedback."],
            "correct_explanation": "Please re-run or check the Ollama service.",
            "improvement":         "Review core concepts and retry.",
        }

    result.setdefault("validated_score", initial_score)
    result.setdefault("strengths", [])
    result.setdefault("mistakes", [])
    result.setdefault("correct_explanation", "")
    result.setdefault("improvement", "")

    return result
