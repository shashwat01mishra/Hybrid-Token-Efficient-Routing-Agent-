# Hybrid Token-Efficient Routing Agent (AMD Hackathon Act-II, Track 1)

## The idea

Two execution tiers, one routing decision:

- **Local tier** — a small open-weights model, run on-instance (AMD
  Instinct MI300X via AMD Developer Cloud). Free at the margin once the
  instance is up.
- **Remote tier** — Fireworks AI, hit only when the local answer looks
  untrustworthy. Costs money per call.

The whole hackathon track is really a selective-prediction / cascade
problem, not an "agent" problem in the LangChain sense: given a task,
decide whether the local model's answer is good enough to keep — before
knowing the ground truth — using only signal that's free to compute.

## Why logprob-based confidence

When the local model generates, HuggingFace's `generate()` already
computes per-token logits as a side effect — no extra forward pass, no
extra cost. From those we derive:

- **`mean_logprob`** — average confidence across the whole answer.
- **`min_logprob`** — the single weakest token. Often a better
  hallucination signal than the mean: one fabricated number or entity can
  sink an otherwise-fluent answer without moving the average much.

Both must clear their threshold for the local answer to be kept; failing
either escalates to Fireworks. This is deliberately **not** the
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
| `tasks.jsonl` | Small labeled task set for calibration — deliberately closed-form (math, exact factual recall) so correctness is checkable by exact/numeric match, no LLM judge needed |
| `calibrate.py` | Runs every task through the local model, grades correctness, sweeps threshold pairs, reports the best operating point |

## Running it

```bash
cp .env.example .env      # fill in FIREWORKS_API_KEY
pip install -r requirements.txt
python agent_loop.py "What is the boiling point of water at 2 atm?"
```

Each run appends a JSON record to `runs/agent_log.jsonl` with the route
taken, the confidence scores, the cost, and the latency — this log is the
raw material for the submission's cost/accuracy plot.

## Testing without a GPU, model download, or Fireworks key

Both the local model and the Fireworks client have a mock mode that
returns deterministic fake outputs, so the full loop — router logic and
JSONL logging included — can be verified with zero network calls:

```bash
MOCK_LOCAL_MODEL=1 MOCK_REMOTE_CLIENT=1 python agent_loop.py "test prompt"
```

Confidence in mock mode is derived from a hash of the prompt, so it's
reproducible run-to-run and exercises both the "keep local" and
"escalate" branches across different inputs. Use this to sanity-check any
change to `router.py` or the logging schema before spending real credits.

## Calibrating the thresholds

```bash
python calibrate.py
```

Runs every task in `tasks.jsonl` through the real local model, grades
correctness (exact/numeric match — no LLM judge), sweeps every
`(MEAN_LOGPROB_THRESHOLD, MIN_LOGPROB_THRESHOLD)` pair over a fixed grid,
and reports the pair that minimizes escalation rate while keeping
local-only accuracy at or above `ACCURACY_FLOOR` (currently a placeholder
`0.85` at the top of the file — replace once the hackathon's real
standardized-eval floor is known). Writes:

- `runs/calibration_results.jsonl` — per-task: generated text, both
  logprobs, graded correctness
- `runs/calibration_sweep.csv` — every threshold pair tried, with its
  escalation rate and local accuracy

`MOCK_LOCAL_MODEL=1 python calibrate.py` runs the same code path with
synthetic, confidence-independent correctness labels — this verifies the
sweep logic doesn't crash, but tells you nothing about real thresholds
(worth knowing: with only 24 tasks, some threshold pair will clear the
accuracy floor by chance even with zero real signal — this is exactly why
the real run needs real weights, and ideally a bigger task set once the
first pass shows the rough operating region).

## What's still open (in priority order)

1. **Run `calibrate.py` against real model weights.** The script exists
   and is verified end-to-end in mock mode (see above) — what's missing
   is a real generation pass. Needs either your laptop or the AMD
   instance, since it needs actual model weights downloaded. Consider
   growing `tasks.jsonl` past 24 items once the first pass shows roughly
   where the threshold should sit — more tasks = less chance the "best"
   pair is a small-sample fluke.
2. **Swap `LOCAL_MODEL` to a Gemma 4 variant** (e.g. `google/gemma-4-E4B`)
   once credits land on the AMD instance — this makes the submission
   eligible for the separate "Best Use of Gemma" bonus pool at no extra
   build cost.
3. **Swap the Dockerfile base image** to a ROCm PyTorch image once the
   AMD instance details (image name, tag) are known.
4. **Decide the accuracy floor** the routing decision has to respect —
   this should come from whatever eval set the hackathon's standardized
   judging environment uses, if that's published; otherwise pick one and
   state it explicitly in the submission.
5. **Verify `FIREWORKS_MODEL` and `FIREWORKS_PRICE_PER_1K_TOKENS`**
   against Fireworks' current model catalog and pricing page before the
   cost numbers go into the submission — both are placeholders right now.
