"""
Escalation decision for the dev/calibration loop (agent_loop.py).

RECONSTRUCTED — the original router.py was overwritten by a submission-
specific version with a different signature (fixed 2026-07-11). This
version is rebuilt to match agent_loop.py's actual call site exactly:
    decision = decide(local_result["mean_logprob"], local_result["min_logprob"])
    if decision.escalate: ... decision.reason ...
If a real git history of this repo exists (a proper `git clone`, not a
GitHub zip download — zips never include .git), check
`git log -- router.py` there first; it would recover the byte-exact
original rather than this reconstruction.
"""
from dataclasses import dataclass

from config import MEAN_LOGPROB_THRESHOLD, MIN_LOGPROB_THRESHOLD


@dataclass
class RouteDecision:
    escalate: bool
    reason: str


def decide(mean_logprob: float, min_logprob: float) -> RouteDecision:
    if mean_logprob < MEAN_LOGPROB_THRESHOLD:
        return RouteDecision(
            escalate=True,
            reason=f"mean_logprob {mean_logprob:.3f} below threshold {MEAN_LOGPROB_THRESHOLD}",
        )
    if min_logprob < MIN_LOGPROB_THRESHOLD:
        return RouteDecision(
            escalate=True,
            reason=f"min_logprob {min_logprob:.3f} below threshold {MIN_LOGPROB_THRESHOLD}",
        )
    return RouteDecision(escalate=False, reason="confidence above both thresholds")
