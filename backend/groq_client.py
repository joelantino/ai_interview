"""
groq_client.py
--------------
Replaces ollama_client.py — calls the Groq cloud API instead of local Ollama.
Groq runs Llama 3.3 70B on H100 GPUs: ~5-10x faster than local CPU inference.

Free tier: https://console.groq.com (no credit card required)
"""

import os
import json
import re
from typing import Optional

from groq import Groq

# ---------------------------------------------------------------------------
# Load API key from .env or environment variable
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    # 1. Try environment variable first
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key

    # 2. Try reading from .env file in project root
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GROQ_API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "GROQ_API_KEY not found. Add it to your .env file or set it as an environment variable."
    )


_client: Optional[Groq] = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=_load_api_key())
    return _client


# ---------------------------------------------------------------------------
# Models — both use llama-3.3-70b for best accuracy on Groq free tier
# ---------------------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"


def _generate(prompt: str, temperature: float = 0.0) -> str:
    """
    Send a prompt to Groq and return the response text.
    Raises RuntimeError on API errors.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        raise RuntimeError(f"Groq API error: {exc}") from exc


def ask_mistral(prompt: str, temperature: float = 0.0) -> str:
    """Evaluation pass — uses Groq Llama 3.3 70B."""
    return _generate(prompt, temperature=temperature)


def ask_llama(prompt: str, temperature: float = 0.0) -> str:
    """Feedback pass — uses Groq Llama 3.3 70B."""
    return _generate(prompt, temperature=temperature)


def extract_json(text: str) -> Optional[dict]:
    """
    Safely extract the first JSON object embedded in a larger LLM response.
    Returns None if no valid JSON is found.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
