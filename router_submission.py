"""
Escalation decision. Two gates must both pass for escalation to happen:
1. Category is one where a 3B local model's ceiling is the real risk
   (code_debugging, logical_reasoning) — escalating sentiment/summarization/
   NER/factual isn't worth the token cost for a 3B model's typical gap there.
2. Local confidence is actually low, per calibrated thresholds.

Math never escalates — math_tool.py handles it deterministically and always
wins regardless of confidence.
"""
import config


def decide(category: str, features: dict, remote_available: bool) -> bool:
    if category == "math":
        return False
    if not remote_available:
        return False
    if category not in config.ESCALATION_ELIGIBLE_CATEGORIES:
        return False

    mean_lp = features.get("mean_logprob", 0.0)
    min_lp = features.get("min_logprob", 0.0)

    if mean_lp < config.MEAN_LOGPROB_THRESHOLD:
        return True
    if min_lp < config.MIN_LOGPROB_THRESHOLD:
        return True
    return False
