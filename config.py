"""
Central configuration for the AMD Hackathon Track 1 submission.
Everything the harness needs is read from environment variables here —
nothing hardcoded, since the grading harness injects these at eval time.
"""
import os

# --- I/O paths ---
TASKS_INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
RESULTS_OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")

# --- Time budget ---
# Guide's hard cap is 10 minutes total. We budget to 9 minutes internally
# to leave a safety margin for writing output even if generation runs long.
TOTAL_TIME_BUDGET_SECONDS = int(os.environ.get("TOTAL_TIME_BUDGET_SECONDS", 9 * 60))
PER_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("PER_REQUEST_TIMEOUT_SECONDS", 25))

# --- Local model ---
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/model/qwen2.5-3b-instruct-q4_k_m.gguf")
LOCAL_MODEL_THREADS = int(os.environ.get("LOCAL_MODEL_THREADS", 2))
LOCAL_MODEL_CTX = int(os.environ.get("LOCAL_MODEL_CTX", 2048))
LOCAL_MODEL_MAX_NEW_TOKENS = int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", 512))
MOCK_LOCAL_MODEL = os.environ.get("MOCK_LOCAL_MODEL", "0") == "1"

# --- Remote (Fireworks) ---
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.environ.get("FIREWORKS_BASE_URL", "")
ALLOWED_MODELS = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
MOCK_REMOTE_CLIENT = os.environ.get("MOCK_REMOTE_CLIENT", "0") == "1"

# --- Routing thresholds ---
# Escalate only when local confidence falls below these. Calibrated values
# should replace these defaults once real confidence data exists on the
# actual GGUF stack (not the old 7B/0.5B calibration data).
MEAN_LOGPROB_THRESHOLD = float(os.environ.get("MEAN_LOGPROB_THRESHOLD", -1.2))
MIN_LOGPROB_THRESHOLD = float(os.environ.get("MIN_LOGPROB_THRESHOLD", -4.0))

# Categories where remote escalation is actually allowed to help.
# Math is handled deterministically (math_tool.py) and never escalates.
# Factual/sentiment/summarization/NER are cheap enough locally that
# escalation isn't worth the token cost for a 3B model's typical gap.
# Code debugging and logical reasoning are the two categories where a
# 3B model's reasoning ceiling is the real risk, so escalation is reserved
# for those.
ESCALATION_ELIGIBLE_CATEGORIES = {"code_debugging", "logical_reasoning"}
