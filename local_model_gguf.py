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
import sys
import config

_LLAMA = None  # lazy-loaded singleton


def _get_llama():
    global _LLAMA
    if _LLAMA is None:
        import os
        from llama_cpp import Llama  # imported lazily so mock mode never needs this installed
        
        model_path = config.LOCAL_MODEL_PATH
        if not os.path.exists(model_path):
            # Check relative to this file
            fallback_path = os.path.join(os.path.dirname(__file__), "model", os.path.basename(model_path))
            if os.path.exists(fallback_path):
                model_path = fallback_path
            else:
                # Check relative to cwd
                fallback_cwd = os.path.join("model", os.path.basename(model_path))
                if os.path.exists(fallback_cwd):
                    model_path = fallback_cwd
                    
        print(f"[local_model_gguf] Loading Llama model from: {model_path}", file=sys.stderr)
        _LLAMA = Llama(
            model_path=model_path,
            n_ctx=config.LOCAL_MODEL_CTX,
            n_threads=config.LOCAL_MODEL_THREADS,
            # llama-cpp-python's own docs: "logits_all: ... Must be True for
            # completion to return logprobs." Set True so create_chat_completion
            # actually has a chance of populating logprobs at all.
            # UNVERIFIED WARNING: there are open upstream issues (e.g.
            # abetlen/llama-cpp-python#1787, ggml-org/llama.cpp#6423) reporting
            # that logprobs=True on create_chat_completion still doesn't
            # reliably return usable per-token logprobs even with this set.
            # Do a real smoke test (one real call, print the raw response
            # dict, confirm token_logprobs actually populates) before trusting
            # the router's escalation decisions on this. If it doesn't work,
            # the failure is SILENT, not a crash: _features_from_logprobs
            # degrades to mean_logprob=0.0 (above any realistic negative
            # threshold), so the router will simply never escalate rather
            # than error out.
            logits_all=True,
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
        import hashlib
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        mean_lp = -0.5 - (h % 50) / 100
        min_lp = mean_lp - 1.0
        
        mock_text = f"[MOCK ANSWER for prompt of length {len(prompt)}]"
        
        # Check math extraction prompts
        if "Extract the arithmetic expression" in prompt:
            import re
            matches = re.findall(r"\d+\s*[\+\-\*x\/÷]\s*\d+", prompt)
            if matches:
                mock_text = matches[0].replace("x", "*").replace("÷", "/")
            else:
                mock_text = "256 * 14 + 739"
        
        # Check code debugging presets
        elif "calculate_sum" in prompt:
            if "syntax error" in prompt.lower() or "previous answer" in prompt.lower():
                mock_text = "Here is the corrected code with valid Python syntax:\n\n```python\ndef calculate_sum(arr):\n    tot = 0\n    for x in arr:\n        tot += x\n    return tot\n```"
            else:
                mock_text = "I found a syntax bug. Here is the code block:\n\n```python\ndef calculate_sum(arr):\n    tot = 0\n    for x in arr\n        tot += x\n    return tot\n```"
                mean_lp = -0.85
                min_lp = -3.20
        
        # Check logical reasoning presets
        elif "Socrates" in prompt:
            mock_text = "Step-by-step conclusion:\n1. All humans are mortal (Premise).\n2. Socrates is a human (Premise).\n3. Therefore, Socrates is mortal.\n\nAnswer: Socrates is mortal."
            mean_lp = -0.15
            min_lp = -0.80
            
        # Check summarization presets
        elif "Hubble Space Telescope" in prompt:
            mock_text = "The Hubble Space Telescope was launched in 1990 into low Earth orbit and remains operational. While not the first, it is one of the largest and most versatile research tools in astronomy."
            mean_lp = -0.25
            min_lp = -1.10

        # Check factual presets
        elif "France" in prompt:
            mock_text = "The capital of France is Paris. Its population is approximately 2.1 million within city limits, and over 12 million in the metropolitan area."
            mean_lp = -0.12
            min_lp = -0.45

        mock_features = {
            "mean_logprob": mean_lp,
            "min_logprob": min_lp,
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
        logprobs=True,
        top_logprobs=5,
    )

    choice = result["choices"][0]
    text = choice["message"]["content"]

    token_logprobs = []
    top_logprobs_list = []
    lp_data = choice.get("logprobs")
    if lp_data:
        if isinstance(lp_data, dict):
            if "content" in lp_data and lp_data["content"] is not None:
                # OpenAI / modern llama-cpp-python style
                for item in lp_data["content"]:
                    token_logprobs.append(item.get("logprob", 0.0))
                    # Extract top logprobs if present
                    top_dict = {}
                    for top in item.get("top_logprobs", []):
                        token_val = top.get("token", "")
                        lp_val = top.get("logprob", 0.0)
                        top_dict[token_val] = lp_val
                    top_logprobs_list.append(top_dict)
            else:
                # Legacy llama-cpp-python style
                token_logprobs = lp_data.get("token_logprobs") or []
                top_logprobs_list = lp_data.get("top_logprobs") or []

    features = _features_from_logprobs(token_logprobs, top_logprobs_list)
    return text, features
