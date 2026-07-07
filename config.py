"""
Configuration for the hybrid routing agent.

Everything that differs between "local laptop test" and "AMD Developer
Cloud + Fireworks" lives here as env vars, so moving environments is a
.env edit, not a code edit.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Local tier ---
    # Default is a tiny model so this runs on a CPU laptop for dev/testing.
    # Swap to a Gemma variant on the AMD instance (qualifies for the
    # "Best Use of Gemma" bonus track at zero extra build cost).
    local_model_name: str = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    local_max_new_tokens: int = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "256"))
    local_device: str = os.getenv("LOCAL_DEVICE", "cpu")  # "cuda" on the AMD instance (ROCm exposes itself via the cuda device string)

    # --- Remote tier (Fireworks, OpenAI-compatible endpoint) ---
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    fireworks_model: str = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct")
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    fireworks_max_tokens: int = int(os.getenv("FIREWORKS_MAX_TOKENS", "256"))

    # --- Router thresholds ---
    # These are placeholders. Do not trust them. Calibrate against a small
    # labeled (task, is_local_answer_correct) set before the demo -- sweep
    # the threshold, plot accuracy vs. % escalated to Fireworks, pick the
    # point that clears the accuracy floor with the fewest escalations.
    mean_logprob_threshold: float = float(os.getenv("MEAN_LOGPROB_THRESHOLD", "-0.6"))
    min_logprob_threshold: float = float(os.getenv("MIN_LOGPROB_THRESHOLD", "-3.0"))

    # --- Cost model (approximate -- for the demo's cost-accounting log) ---
    fireworks_price_per_1k_tokens: float = float(os.getenv("FIREWORKS_PRICE_PER_1K", "0.0002"))
    local_price_per_1k_tokens: float = float(os.getenv("LOCAL_PRICE_PER_1K", "0.0"))  # GPU-credit cost is sunk, treated as ~free at the margin

    # --- Logging ---
    log_path: str = os.getenv("LOG_PATH", "runs/agent_log.jsonl")


CFG = Config()
