"""
All env-dependent settings for the routing agent live here. Nothing else
in the codebase should read os.environ directly — if a new knob is needed,
add it here first.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Local tier ---
# Bootstrap default: small, fast, CPU-friendly, so the pipeline can be
# built and verified before the AMD instance / Gemma swap happens.
#
# TODO: swap to a Gemma 4 variant (e.g. google/gemma-4-E4B, or a larger
# size once served on the AMD MI300X instance) once AMD Developer Cloud
# credits land. This also makes the submission eligible for the separate
# "Best Use of Gemma" bonus pool at zero extra build cost.
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# --- Remote tier (Fireworks AI) ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
# Verify this against https://fireworks.ai/models before submission —
# the catalog may have changed since this was written.
FIREWORKS_MODEL = os.getenv(
    "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-8b-instruct"
)
FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
)
# Placeholder price (USD per 1K tokens). Check https://fireworks.ai/pricing
# for the real per-model rate and update before the cost numbers go into
# the submission's cost/accuracy plot — this number is currently a guess.
FIREWORKS_PRICE_PER_1K_TOKENS = float(
    os.getenv("FIREWORKS_PRICE_PER_1K_TOKENS", "0.0009")
)

# --- Routing thresholds (PLACEHOLDERS — see README "what's still open") ---
# Both are log-probabilities, so more negative = less confident.
# A local answer is kept only if BOTH clear their threshold; failing
# either escalates to Fireworks.
MEAN_LOGPROB_THRESHOLD = float(os.getenv("MEAN_LOGPROB_THRESHOLD", "-0.5"))
MIN_LOGPROB_THRESHOLD = float(os.getenv("MIN_LOGPROB_THRESHOLD", "-2.0"))

# --- Testing toggles ---
# With both set to 1, the whole loop runs with zero network calls and
# zero model downloads — useful for verifying router/logging logic
# before spending real credits or API calls.
MOCK_LOCAL_MODEL = os.getenv("MOCK_LOCAL_MODEL", "0") == "1"
MOCK_REMOTE_CLIENT = os.getenv("MOCK_REMOTE_CLIENT", "0") == "1"
