"""
Local tier: Qwen2.5-3B-Instruct, Q4_K_M GGUF, via llama-cpp-python.
CPU-only, sized for the grading box's 4GB RAM / 2 vCPU / no-GPU constraint.

Extracts logprob-based confidence features at no extra compute cost
(computed from the same forward pass as generation) so the router can
decide whether to escalate without a second call.

MOCK_LOCAL_MODEL=1 bypasses llama_cpp entirely and returns a deterministic
canned response + confidence features, for testing harness logic without
the real model weights or the compiled library present.
"""
import math
import config

_LLAMA = None  # lazy-loaded singleton


def _get_llama():
    global _LLAMA
    if _LLAMA is None:
        from llama_cpp import Llama  # imported lazily so mock mode never needs this installed
        _LLAMA = Llama(
            model_path=config.LOCAL_MODEL_PATH,
            n_ctx=config.LOCAL_MODEL_CTX,
            n_threads=config.LOCAL_MODEL_THREADS,
            logits_all=False,
            verbose=False,
        )
    return _LLAMA


def _features_from_logprobs(token_logprobs, top_logprobs_list):
    """
    token_logprobs: list of float logprobs for the chosen token at each step
    top_logprobs_list: list of dicts (token -> logprob) for the top-N
                        candidates at each step, used for top2_margin
    """
    if not token_logprobs:
        return {
            "mean_logprob": 0.0,
            "min_logprob": 0.0,
            "entropy_mean": 0.0,
            "top2_margin_mean": 0.0,
        }

    mean_logprob = sum(token_logprobs) / len(token_logprobs)
    min_logprob = min(token_logprobs)

    entropies = []
    margins = []
    for top in top_logprobs_list:
        if not top:
            continue
        sorted_lps = sorted(top.values(), reverse=True)
        probs = [math.exp(lp) for lp in sorted_lps]
        z = sum(probs) if sum(probs) > 0 else 1e-9
        probs = [p / z for p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        entropies.append(entropy)
        if len(sorted_lps) >= 2:
            margins.append(sorted_lps[0] - sorted_lps[1])

    entropy_mean = sum(entropies) / len(entropies) if entropies else 0.0
    top2_margin_mean = sum(margins) / len(margins) if margins else 0.0

    return {
        "mean_logprob": mean_logprob,
        "min_logprob": min_logprob,
        "entropy_mean": entropy_mean,
        "top2_margin_mean": top2_margin_mean,
    }


def generate(prompt: str, system_prompt: str = "", max_tokens: int = None):
    """
    Returns (text: str, features: dict).
    Raises on real failure (model load error, etc.) — caller is responsible
    for catching and degrading gracefully, per the harness's defensive
    contract.
    """
    max_tokens = max_tokens or config.LOCAL_MODEL_MAX_NEW_TOKENS

    if config.MOCK_LOCAL_MODEL:
        # Deterministic mock: echoes a plausible-looking answer and
        # confidence features, so harness logic (routing, math_tool
        # integration, output schema) can be tested without the real
        # compiled library or model weights present.
        mock_text = f"[MOCK ANSWER for prompt of length {len(prompt)}]"
        mock_features = {
            "mean_logprob": -0.5,
            "min_logprob": -1.5,
            "entropy_mean": 0.3,
            "top2_margin_mean": 2.0,
        }
        return mock_text, mock_features

    llama = _get_llama()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    result = llama.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        logprobs=5,
    )

    choice = result["choices"][0]
    text = choice["message"]["content"]

    token_logprobs = []
    top_logprobs_list = []
    lp_data = choice.get("logprobs")
    if lp_data:
        token_logprobs = lp_data.get("token_logprobs") or []
        top_logprobs_list = lp_data.get("top_logprobs") or []

    features = _features_from_logprobs(token_logprobs, top_logprobs_list)
    return text, features
