
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def generate_investigation_summary(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "mock":
        return f"Mock investigation summary:\n{prompt}"
    if provider not in {"gemini", "openrouter"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        headers = {"x-goog-api-key": api_key or ""}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key or ''}",
            "Content-Type": "application/json",
            "X-Title": "Fraud Detection Platform",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.2,
        }

    if not api_key:
        raise RuntimeError(f"API key is required when LLM_PROVIDER={provider}")

    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    connect_timeout = float(os.getenv("LLM_CONNECT_TIMEOUT", "5"))
    read_timeout = float(os.getenv("LLM_READ_TIMEOUT", "25"))
    backoff = float(os.getenv("LLM_RETRY_BACKOFF", "1"))
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=(connect_timeout, read_timeout),
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == max_retries:
                    response.raise_for_status()

                time.sleep(min(backoff * (2 ** attempt), 5))
                continue

            response.raise_for_status()
            result = response.json()

            if provider == "gemini":
                text = result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                text = result["choices"][0]["message"]["content"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("LLM returned empty text")

            return text.strip()

        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                logger.exception("%s request failed", provider)
                raise

            time.sleep(min(backoff * (2 ** attempt), 5))

    raise RuntimeError(f"{provider} request failed after all retries")