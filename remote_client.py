"""
Fireworks AI call — only hit on escalation. Uses the OpenAI-compatible
chat completions endpoint directly via requests, to keep the dependency
footprint minimal.
"""
import time

from config import (
    FIREWORKS_API_KEY,
    FIREWORKS_MODEL,
    FIREWORKS_BASE_URL,
    FIREWORKS_PRICE_PER_1K_TOKENS,
    MOCK_REMOTE_CLIENT,
)


def query_fireworks(prompt: str) -> dict:
    """
    Returns:
        text: the remote model's answer
        cost_usd: estimated cost of this call
        latency_ms: wall-clock call time
        num_tokens: total tokens billed (prompt + completion)
    """
    start = time.perf_counter()

    if MOCK_REMOTE_CLIENT:
        return _mock_query(prompt, start)

    if not FIREWORKS_API_KEY:
        raise RuntimeError("FIREWORKS_API_KEY not set — check your .env")

    import requests

    response = requests.post(
        f"{FIREWORKS_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": FIREWORKS_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    num_tokens = usage.get("total_tokens", 0)
    cost_usd = (num_tokens / 1000) * FIREWORKS_PRICE_PER_1K_TOKENS

    return {
        "text": text,
        "cost_usd": cost_usd,
        "latency_ms": (time.perf_counter() - start) * 1000,
        "num_tokens": num_tokens,
    }


def _mock_query(prompt: str, start: float) -> dict:
    """Deterministic fake escalation response — no API key or network needed."""
    num_tokens = 60
    latency_ms = (time.perf_counter() - start) * 1000 + 120  # simulate network RTT
    return {
        "text": f"[mock Fireworks answer for: {prompt[:40]}]",
        "cost_usd": (num_tokens / 1000) * FIREWORKS_PRICE_PER_1K_TOKENS,
        "latency_ms": latency_ms,
        "num_tokens": num_tokens,
    }
