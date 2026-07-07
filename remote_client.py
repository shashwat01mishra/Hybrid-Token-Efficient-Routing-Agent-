"""
Remote tier: Fireworks AI, called via its OpenAI-compatible chat
completions endpoint. Only hit when the router escalates -- this is the
expensive path, and it's used sparingly by design.
"""
from typing import Tuple

import requests

from config import CFG


class RemoteError(RuntimeError):
    pass


def generate(prompt: str) -> Tuple[str, int]:
    """Returns (text, total_tokens_used) so the caller can cost the call."""
    if not CFG.fireworks_api_key:
        raise RemoteError(
            "FIREWORKS_API_KEY is not set -- put your Fireworks account key "
            "(the one your FW-LABLAB-MRF6 coupon is credited to) in .env"
        )

    resp = requests.post(
        f"{CFG.fireworks_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {CFG.fireworks_api_key}"},
        json={
            "model": CFG.fireworks_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": CFG.fireworks_max_tokens,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    total_tokens = data.get("usage", {}).get("total_tokens", 0)
    return text, total_tokens
