# Hybrid Token-Efficient Routing Agent (AMD Hackathon Act-II, Track 1)

## The idea

Two execution tiers, one routing decision:

- **Local tier** — a small open-weights model, run on-instance (AMD Instinct
  MI300X via AMD Developer Cloud). Free at the margin once the instance is up.
- **Remote tier** — Fireworks AI, hit only when the local answer looks
  untrustworthy. Costs money per call.

The entire hackathon is really a **selective-prediction / cascade** problem,
not an "agent" problem in the LangChain sense: given a task, decide whether
the local model's answer is good enough to keep, *before* knowing the ground
truth, using only signal that's free to compute.

## Why logprob-based confidence

When the local model generates, HuggingFace's `generate()` already computes
per-token logits as a side effect — no extra forward pass, no extra cost.
From those we derive:

- **mean_logprob** — average confidence across the whole answer
- **min_logprob** — the single weakest token. Often a better hallucination
  signal than the mean: one fabricated number or entity can sink an
  otherwise-fluent answer without moving the average much.

Both must clear their threshold for the local answer to be kept; failing
either escalates to Fireworks. This is deliberately *not* the
embedding/geometry-based approach from other (unpublished) research —
that line is being kept separate on purpose.

## Files

| File | Role |
|---|---|
| `config.py` | All env-dependent settings (model names, thresholds, prices) |
| `local_model.py` | Loads the local model, generates, extracts token logprobs |
| `router.py` | The routing decision — the actual "intelligence" being judged |
| `remote_client.py` | Fireworks API call, only hit on escalation |
| `agent_loop.py` | Orchestrates the above, logs every decision to `runs/agent_log.jsonl` |

## Running it

```bash
cp .env.example .env      # fill in FIREWORKS_API_KEY
pip install -r requirements.txt
python agent_loop.py "What is the boiling point of water at 2 atm?"
```

Each run appends a JSON record to `runs/agent_log.jsonl` with the route
taken, the confidence scores, the cost, and the latency — this log is the
raw material for the submission's cost/accuracy plot.

## What's still open (in priority order)

1. **Threshold calibration.** `MEAN_LOGPROB_THRESHOLD` / `MIN_LOGPROB_THRESHOLD`
   in `.env` are placeholders. Build a small labeled set (task, is the local
   answer actually correct), sweep the thresholds, plot accuracy vs.
   % escalated, and pick the operating point — that sweep *is* the
   "routing intelligence" the judges are scoring.
2. **Swap `LOCAL_MODEL` to a Gemma variant** once credits land on the AMD
   instance — this makes the submission eligible for the separate
   "Best Use of Gemma" bonus pool at no extra build cost.
3. **Swap the Dockerfile base image** to a ROCm PyTorch image once the AMD
   instance details (image name, tag) are known.
4. **Decide the accuracy floor** the routing decision has to respect — this
   should come from whatever eval set the hackathon's standardized judging
   environment uses, if that's published; otherwise pick one and state it
   explicitly in the submission.
