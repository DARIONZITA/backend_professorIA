"""Client for text LLMs.

This module prefers using Groq (API_GROQ) for text generation. If
API_GROQ is not present, it will fall back to Gemini (if GEMINI_API_KEY is
available). The client exposes generate_structured() to run a prompt and
attempt to extract a JSON block from the response.

Notes / assumptions:
- The Groq endpoint URL can be configured via API_GROQ_URL. If not set,
  a reasonable default is used: https://api.groq.ai/v1/models/{model}/infer
  (if your environment requires a different URL, set API_GROQ_URL).
"""
from __future__ import annotations
import os
import json
import re
import time
import logging
from typing import Any, Dict, Optional

log = logging.getLogger("llm")

# Provider state
_PROVIDER: Optional[str] = None  # 'groq' | 'gemini' | None
_LLM_AVAILABLE = False

# Groq defaults
GROQ_API_KEY = os.getenv("API_GROQ")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gemma2-9b-it")
GROQ_URL = os.getenv("API_GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")

# Gemini defaults (fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")


def _load_dotenv_if_present() -> None:
    """Try to load .env into environment (best-effort)."""
    try:
        from pathlib import Path
    except Exception:
        return
    # If keys already present, skip
    if GROQ_API_KEY or GEMINI_API_KEY or os.getenv("API_GROQ") or os.getenv("GEMINI_API_KEY"):
        return
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv()
        log.info(".env loaded via python-dotenv")
        return
    except Exception:
        pass
    try:
        here = Path(__file__).resolve().parent
        for p in (here, here.parent, Path.cwd()):
            env_path = p / ".env"
            if env_path.exists():
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for raw in f:
                            line = raw.strip()
                            if not line or line.startswith("#"):
                                continue
                            if "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k and v and k not in os.environ:
                                os.environ[k] = v
                    log.info(f".env loaded manually from {env_path}")
                    return
                except Exception:
                    continue
    except Exception:
        pass


_load_dotenv_if_present()


def _init_client() -> None:
    """Initialize which provider we'll use for text LLMs.

    Prefer Groq (API_GROQ). If not available, try Gemini.
    This function sets _PROVIDER and _LLM_AVAILABLE.
    """
    global _PROVIDER, _LLM_AVAILABLE
    if _PROVIDER is not None:
        return
    # Prefer Groq
    groq_key = os.getenv("API_GROQ") or GROQ_API_KEY
    if groq_key:
        _PROVIDER = "groq"
        _LLM_AVAILABLE = True
        log.info("Using Groq for text LLMs (model=%s)", GROQ_MODEL)
        return
    # Fallback to Gemini
    gem_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if gem_key:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=gem_key)
            _PROVIDER = "gemini"
            _LLM_AVAILABLE = True
            log.info("Using Gemini for text LLMs (model=%s)", GEMINI_MODEL)
            return
        except Exception as e:
            log.error("Failed to initialize Gemini client: %s", e)
    # Nothing available
    _PROVIDER = None
    _LLM_AVAILABLE = False


JSON_BLOCK_REGEX = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Tries to extract first JSON block from a response."""
    match = JSON_BLOCK_REGEX.search(text)
    candidate = None
    if match:
        candidate = match.group(1)
    else:
        # fallback: look for first '{' ... last '}' simple
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None


# Simple metrics
_METRICS = {
    "requests": 0,
    "errors": 0,
    "last_latency_ms": None,
}


def _call_groq(prompt: str, system: str, temperature: float, max_output_tokens: int = 512, timeout: int = 30) -> str:
    """Call Groq (OpenAI-compatible endpoint) and return text output."""
    api_key = os.getenv("API_GROQ") or GROQ_API_KEY
    if not api_key:
        raise RuntimeError("API_GROQ not configured")
    url = os.getenv("API_GROQ_URL") or GROQ_URL
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": os.getenv("GROQ_MODEL", GROQ_MODEL),
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_output_tokens),
        # response_format is optional; keep it but tolerate endpoints that ignore it
        "response_format": {"type": "json_object"},
    }
    # Try requests if available, otherwise fall back to urllib
    try:
        import requests
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        # Try urllib as fallback
        try:
            import urllib.request
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as f:
                raw = f.read().decode("utf-8")
                j = json.loads(raw)
        except Exception as e2:
            log.exception("Groq request failed: %s / %s", e, e2)
            raise

    # Best-effort extraction of text from Groq (OpenAI-compatible) response
    if isinstance(j, dict):
        choices = j.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                for key in ("text", "content", "generated_text"):
                    if key in first and isinstance(first[key], str):
                        return first[key]
        # Some experimental endpoints may return "output" or similar fields
        for key in ("output", "generated_text", "content"):
            if key in j and isinstance(j[key], str):
                return j[key]
    # Fallback: try to stringify JSON
    try:
        return json.dumps(j)
    except Exception:
        return str(j)


def generate_structured(prompt: str, system: str = "", temperature: float = 0.1, max_retries: int = 2, max_output_tokens: int = 512) -> Dict[str, Any]:
    """Generates structured response (JSON) from a prompt.

    Behavior:
      - Uses Groq (API_GROQ) by default with model `gemma2-9b-it` and temp=0.1.
      - Falls back to Gemini if Groq is not configured.
    """
    _init_client()
    if not _LLM_AVAILABLE or _PROVIDER is None:
        return {
            "success": False,
            "data": {},
            "raw": "",
            "error": "LLM disabled (no API_GROQ or GEMINI_API_KEY)",
            "llm_enabled": False,
        }

    attempt = 0
    last_error = None
    while attempt <= max_retries:
        start = time.time()
        try:
            _METRICS["requests"] += 1

            if _PROVIDER == "groq":
                raw_text = _call_groq(prompt, system, temperature, max_output_tokens=max_output_tokens)
            else:
                # Gemini path (best-effort, old behavior)
                try:
                    import google.generativeai as genai  # type: ignore
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY)
                    model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
                    parts = []
                    if system:
                        parts.append(f"SYSTEM:\n{system}\n---\n")
                    parts.append(prompt)
                    full_prompt = "\n".join(parts)
                    # use the text generation API
                    # Gemini fallback: we still pass a high token allowance if supported
                    response = genai.generate_text(model=model, prompt=full_prompt, temperature=temperature)  # type: ignore
                    # response may be a dict-like
                    raw_text = ''
                    if isinstance(response, dict):
                        # try different keys
                        for k in ("output", "text", "content", "generated_text"):
                            if k in response and isinstance(response[k], str):
                                raw_text = response[k]
                                break
                    else:
                        raw_text = getattr(response, 'text', '') or str(response)
                except Exception as e:
                    raise

            duration_ms = int((time.time() - start) * 1000)
            _METRICS["last_latency_ms"] = duration_ms
            log.info("LLM (%s) request success, latency=%dms", _PROVIDER, duration_ms)

            data = _extract_json(raw_text)
            if data is not None:
                return {"success": True, "data": data, "raw": raw_text, "error": None, "llm_enabled": True}
            last_error = "JSON not found / invalid"
        except Exception as e:
            last_error = str(e)
            _METRICS["errors"] += 1
            log.exception("LLM (%s) request failed: %s", _PROVIDER, e)
        attempt += 1
        # backoff
        time.sleep(0.5 * attempt)

    return {
        "success": False,
        "data": {},
        "raw": "",
        "error": last_error or "unknown",
        "llm_enabled": True,
    }


def llm_available() -> bool:
    _init_client()
    return _LLM_AVAILABLE


def llm_metrics() -> Dict[str, Any]:
    """Return basic metrics for observability."""
    return dict(_METRICS)
