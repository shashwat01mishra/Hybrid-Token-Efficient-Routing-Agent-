"""
Remote tier: calls Fireworks AI via the harness-injected FIREWORKS_BASE_URL,
using only a model from the harness-injected ALLOWED_MODELS list. Never
reads a .env file or hardcodes credentials — these are runtime-injected by
the grading harness, per the I/O contract.

MOCK_REMOTE_CLIENT=1 returns a canned response without any network call, for
testing harness logic in an environment with no real credentials.
"""
import json
import config

try:
    import requests
except ImportError:
    requests = None  # only required in real (non-mock) mode


def _select_model() -> str:
    """
    Placeholder selection strategy: first entry in ALLOWED_MODELS.
    TODO once the real ALLOWED_MODELS list is published: prefer a model
    suited to the escalated category (e.g. a stronger reasoning model for
    logical_reasoning) rather than always taking the first entry. Also
    check for any Gemma 4 variant in the list — routing to one qualifies
    for the "Best Use of Gemma" bonus pool at no extra build cost.
    """
    if not config.ALLOWED_MODELS:
        raise RuntimeError("ALLOWED_MODELS is empty — cannot select a remote model")
    return config.ALLOWED_MODELS[0]


def is_available() -> bool:
    if config.MOCK_REMOTE_CLIENT:
        return True
    return bool(config.FIREWORKS_API_KEY and config.FIREWORKS_BASE_URL and config.ALLOWED_MODELS)


def query_fireworks(prompt: str, system_prompt: str = "") -> str:
    if config.MOCK_REMOTE_CLIENT:
        text = f"[MOCK REMOTE ANSWER for prompt of length {len(prompt)}]"
        if "calculate_sum" in prompt:
            text = "Here is the corrected code (escalated to Fireworks AI):\n\n```python\ndef calculate_sum(arr):\n    tot = 0\n    for x in arr:\n        tot += x\n    return tot\n```"
        elif "Socrates" in prompt:
            text = "Logical reasoning conclusion (escalated to Fireworks AI):\n1. All humans are mortal (Premise).\n2. Socrates is a human (Premise).\n3. Therefore, Socrates is mortal.\n\nAnswer: Socrates is mortal."
        elif "France" in prompt:
            text = "The capital of France is Paris. Its population is approximately 2.1 million within city limits, and over 12 million in the metropolitan area. (escalated to Fireworks AI)"
        elif "Hubble" in prompt:
            text = "Summary of Hubble Space Telescope (escalated to Fireworks AI):\nLaunched in 1990 into low Earth orbit, the Hubble Space Telescope is a large and versatile research tool that remains operational and has transformed astronomy."
        return text

    if requests is None:
        raise RuntimeError("requests library not available for real remote call")

    model = _select_model()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        f"{config.FIREWORKS_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.FIREWORKS_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": config.LOCAL_MODEL_MAX_NEW_TOKENS,
        }),
        timeout=config.PER_REQUEST_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
