"""
The routing decision. This file is deliberately small — the "intelligence"
being judged is the calibration of the two thresholds against real data
(see README, "what's still open"), not architectural complexity here.
"""
from dataclasses import dataclass

from config import MEAN_LOGPROB_THRESHOLD, MIN_LOGPROB_THRESHOLD


@dataclass
class RouteDecision:
    escalate: bool
    reason: str


def decide(mean_logprob: float, min_logprob: float) -> RouteDecision:
    """
    Both confidence signals must clear their threshold for the local
    answer to be trusted; failing either escalates to the remote
    (paid) tier.

    mean_logprob: average per-token log-probability of the local
        generation — a smooth signal for overall fluency/confidence.
    min_logprob: log-probability of the single least confident token —
        catches the "one wrong number or entity in an otherwise fluent
        answer" failure mode that the mean can hide.
    """
    if mean_logprob < MEAN_LOGPROB_THRESHOLD:
        return RouteDecision(
            escalate=True,
            reason=f"mean_logprob {mean_logprob:.3f} < threshold {MEAN_LOGPROB_THRESHOLD}",
        )
    if min_logprob < MIN_LOGPROB_THRESHOLD:
        return RouteDecision(
            escalate=True,
            reason=f"min_logprob {min_logprob:.3f} < threshold {MIN_LOGPROB_THRESHOLD}",
        )
    return RouteDecision(escalate=False, reason="both signals cleared threshold")
