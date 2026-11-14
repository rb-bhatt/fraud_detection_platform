
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
    if provider != "gemini":
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required when LLM_PROVIDER=gemini"
        )
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    connect_timeout = float(os.getenv("LLM_CONNECT_TIMEOUT", "5"))
    read_timeout = float(os.getenv("LLM_READ_TIMEOUT", "25"))
    backoff = float(os.getenv("LLM_RETRY_BACKOFF", "1"))
    headers = {"x-goog-api-key": api_key}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

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
                retry_after = response.headers.get("Retry-After")

                try:
                    delay = float(retry_after) if retry_after else (
                        backoff * (2 ** attempt)
                    )
                except ValueError:
                    delay = backoff * (2 ** attempt)

                delay = min(max(delay, 0), 5)

                logger.warning(
                    "Gemini returned HTTP %s. Retrying in %.1f seconds "
                    "(attempt %s/%s)",
                    response.status_code,
                    delay,
                    attempt + 1,
                    max_retries + 1,
                )

                time.sleep(delay)
                continue

            response.raise_for_status()

            result = response.json()

            try:
                text = result["candidates"][0]["content"]["parts"][0]["text"]

                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Gemini returned empty text")

                return text.strip()

            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    "Gemini returned an unexpected response"
                ) from exc

        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_retries:
                logger.exception("Gemini request timed out or failed")
                raise

            delay = min(backoff * (2 ** attempt), 5)

            logger.warning(
                "Gemini connection failure. Retrying in %.1f seconds "
                "(attempt %s/%s)",
                delay,
                attempt + 1,
                max_retries + 1,
            )

            time.sleep(delay)

    raise RuntimeError("Gemini request failed after all retries")
