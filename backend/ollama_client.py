"""
ollama_client.py
----------------
Low-level wrapper around the Ollama /api/generate endpoint.
Supports both streaming=False (full response) and streaming=True (token stream).
"""

import json
import requests
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"
# phi3:mini (2.3GB, 3.8B params) is optimised for CPU-only laptops.
# It is significantly faster than Mistral (4.8GB) on machines without a dedicated GPU.
DEFAULT_MISTRAL  = "phi3:mini"  # Evaluation pass
DEFAULT_LLAMA    = "phi3:mini"  # Feedback pass


def _generate(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    stream: bool = False,
) -> str:
    """
    Send a generation request to the local Ollama instance.

    Returns the combined response text.
    Raises RuntimeError on HTTP or JSON errors.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            stream=stream,
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: `ollama serve`"
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc}")

    if stream:
        collected = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                collected.append(chunk.get("response", ""))
                if chunk.get("done"):
                    break
        return "".join(collected)
    else:
        data = response.json()
        return data.get("response", "")


def ask_mistral(prompt: str, temperature: float = 0.0) -> str:
    """Run a prompt through the local Mistral model."""
    return _generate(prompt, model=DEFAULT_MISTRAL, temperature=temperature)


def ask_llama(prompt: str, temperature: float = 0.0) -> str:
    """Run a prompt through the local Llama3 model."""
    return _generate(prompt, model=DEFAULT_LLAMA, temperature=temperature)


def extract_json(text: str) -> Optional[dict]:
    """
    Safely extract the first JSON object embedded in a larger LLM response.
    Returns None if no valid JSON is found.
    """
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
