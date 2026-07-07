"""
The routing decision -- this is the entire "intelligence" the hackathon
judges are scoring.

Deliberately NOT using embedding/geometric similarity here: that's a
different, unpublished research line (AI-LABS / H7) and is being kept out
of this hackathon on purpose. This router uses only the token-level
log-probabilities the local model already produced during generation --
a well-established selective-prediction / cascade signal (see FrugalGPT
and the LLM-cascade literature), computed at zero extra inference cost.

Two signals, both must pass for the local answer to be trusted:
  - mean_logprob: overall fluency/confidence across the whole answer
  - min_logprob:  the single weakest token -- often a better hallucination
                  flag than the mean, since one fabricated number or
                  entity can sink an otherwise fluent answer without
                  moving the average much
"""
from dataclasses import dataclass

from config import CFG
from local_model import LocalResult


@dataclass
class RoutingDecision:
    escalate: bool
    reason: str


def decide(result: LocalResult) -> RoutingDecision:
    if result.mean_logprob < CFG.mean_logprob_threshold:
        return RoutingDecision(
            escalate=True,
            reason=f"mean_logprob {result.mean_logprob:.3f} below threshold {CFG.mean_logprob_threshold}",
        )
    if result.min_logprob < CFG.min_logprob_threshold:
        return RoutingDecision(
            escalate=True,
            reason=f"min_logprob {result.min_logprob:.3f} below threshold {CFG.min_logprob_threshold} (weak-token flag)",
        )
    return RoutingDecision(escalate=False, reason="local answer passed both confidence gates")
