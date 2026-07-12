# Hybrid Token-Efficient Routing Agent

AMD Developer Hackathon: ACT II — Track 1 submission.

Answers a fixed set of tasks across 8 categories (factual knowledge, mathematical
reasoning, sentiment classification, summarization, named entity recognition, code
debugging, logical/deductive reasoning, code generation) inside a container, using a
local model by default and escalating to Fireworks AI only for the two categories
where a small local model's reasoning genuinely breaks down.

Per Track 1's scoring rule: local models count fully toward accuracy, and only tokens
routed through `FIREWORKS_BASE_URL` count toward the token score. Local inference is
therefore the best possible outcome for ranking wherever it clears the accuracy bar —
this submission is built around staying local by default and escalating only when it
actually helps.

## Quick start — build and run the submission

```bash
docker buildx build --platform linux/amd64 --load -f Dockerfile.submission -t submission-test:latest .
```

The Qwen2.5-3B-Instruct Q4_K_M GGUF model is downloaded and baked in at build time, so
the container doesn't need network access at grading time and stays within the 60s
ready-time budget.

Run it against a real task file:

```bash
mkdir -p /tmp/hk_input /tmp/hk_output
cp path/to/your/tasks.json /tmp/hk_input/tasks.json

docker run --rm \
  -v /tmp/hk_input:/input \
  -v /tmp/hk_output:/output \
  -e FIREWORKS_API_KEY=your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/llama-v3p1-8b-instruct \
  submission-test:latest

cat /tmp/hk_output/results.json
```

Omit the three `FIREWORKS_*` env vars entirely to run local-only, with escalation
disabled — useful for a quick sanity check with zero external dependencies:

```bash
docker run --rm -v /tmp/hk_input:/input -v /tmp/hk_output:/output submission-test:latest
```

Reads `/input/tasks.json` (`[{"task_id", "prompt"}, ...]`), writes
`/output/results.json` (`[{"task_id", "answer"}, ...]`) before exiting, always exits 0.

**Confirmed on real hardware:** clean build in 55.5s (6/6 layers, `linux/amd64`); real
(non-mock) local inference in 21.7s for a 3-task batch, with all answers verified
correct — `12 x 9 = 108`, capital of France = `Paris`, and a correct, coherent summary
— all answered locally at zero Fireworks tokens.

## How it works

```
tasks.json -> classify category (zero-token regex) -> local model (3B, CPU-only)
           -> verify answer (syntax check + confidence check)
           -> pass: write answer directly
           -> fail, and category is code_debugging or logical_reasoning:
                escalate to Fireworks AI, or retry locally once if no credentials
           -> results.json
```

- **Category classification** (`prompt_templates.py`) is pure regex/keyword matching
  on the prompt text — the real grading harness's input schema has no category field,
  so this has to work from the prompt alone. Costs no tokens, negligible latency.
- **Math** (`math_tool.py`) is handled deterministically: the LLM extracts the
  arithmetic expression from the word problem, then a restricted AST walker evaluates
  it exactly — never `eval()`. Verified against real confidently-wrong failure cases
  (`19*33=627`, `83*66=5478`, `28*89=2492`, `206-8=198`, `67*86=5762`), and against
  injection attempts (`__import__`, `open()`, `.__class__`, `exec` all correctly
  rejected).
- **Code** (`verify.py`) is syntax-checked with `compile()` before being accepted. A
  syntax error is a certain accuracy-gate failure, not a maybe, so this check
  overrides the normal confidence-based routing entirely: it forces escalation if
  Fireworks is available, or one local retry with the error fed back in if not. Never
  withholds an answer, even if both attempts fail.
- **Escalation is restricted to `code_debugging` and `logical_reasoning`**
  (`config.ESCALATION_ELIGIBLE_CATEGORIES`) — the two categories where a 3B model's
  reasoning ceiling is the real risk and no deterministic fix exists. The other six
  categories never call Fireworks, regardless of confidence.
- **The harness** (`harness.py`) never crashes: every failure path (missing input,
  malformed input, model load failure, no remote credentials, an unhandled exception
  mid-task) degrades to writing the best available answer — even an empty string
  rather than losing the task — and always exits 0 with valid JSON.

### Known open item

`local_model_gguf.py` requests logprobs via `logprobs=True, top_logprobs=5` with
`logits_all=True` on the model constructor, per `llama-cpp-python`'s documented
requirement for logprobs to populate at all. There are open upstream issues
(`abetlen/llama-cpp-python#1787`, `ggml-org/llama.cpp#6423`) suggesting this doesn't
always work reliably even when configured correctly. If it silently doesn't work, the
failure mode is silent, not a crash: confidence features default to values that never
trigger escalation, so the submission still runs and answers correctly, it just won't
escalate low-confidence code/logic answers. Recommended: a real smoke test — one call,
inspect the raw response — before assuming escalation is firing as designed.

## Repository layout

**Submission (what gets graded):**

| File | Role |
|---|---|
| `harness.py` | Entrypoint. Reads `/input/tasks.json`, writes `/output/results.json` |
| `config.py` | All environment-driven settings, shared with the dev tooling below |
| `prompt_templates.py` | Category classification + category-specific system prompts |
| `math_tool.py` | Deterministic math: LLM extracts expression, AST walker evaluates it |
| `verify.py` | Syntax verification for code answers, forces retry/escalation on failure |
| `local_model_gguf.py` | Local tier — Qwen2.5-3B-Instruct GGUF via `llama-cpp-python` |
| `router_submission.py` | Escalation decision for the submission stack |
| `remote_client_submission.py` | Fireworks client, reads harness-injected env vars |
| `Dockerfile.submission` | Builds the actual submitted image |
| `requirements-submission.txt` | `llama-cpp-python`, `requests` |

**Dev / calibration tooling** (research used to inform the submission, not itself
graded):

| File | Role |
|---|---|
| `agent_loop.py` | CLI harness for calibration research — not the grading entrypoint |
| `local_model.py` | Dev-tooling local model wrapper (`transformers`/`mlx` backends) |
| `remote_client.py` | Dev-tooling Fireworks client |
| `router.py` | Dev-tooling routing decision, used by `agent_loop.py` and `calibrate.py` |
| `calibrate.py` | Sweeps confidence thresholds against real model weights |
| `analyze_calibration.py` | Confidence-vs-correctness analysis on calibration data |
| `generate_tasks.py` | Generates the arithmetic portion of the calibration set |
| `tasks.jsonl` / `tasks_factual.jsonl` | 269-task calibration set (49 curated + 220 generated) |

## Setup

```bash
cp .env.example .env      # fill in FIREWORKS_API_KEY for real (non-mock) runs
pip install -r requirements.txt              # dev tooling
pip install -r requirements-submission.txt   # submission stack (or just build the Docker image)
```

## Testing without a GPU, model download, or Fireworks key

Both stacks support mock mode — deterministic fake outputs, zero network calls, for
verifying logic changes before spending real time or credits:

```bash
# submission harness
TASKS_INPUT_PATH=input/tasks.json RESULTS_OUTPUT_PATH=output/results.json \
  MOCK_LOCAL_MODEL=1 MOCK_REMOTE_CLIENT=1 \
  ALLOWED_MODELS=test FIREWORKS_API_KEY=test FIREWORKS_BASE_URL=http://test \
  python harness.py

# dev tooling
MOCK_LOCAL_MODEL=1 MOCK_REMOTE_CLIENT=1 python agent_loop.py "test prompt"
```

## Calibration research (background)

The submission's routing thresholds are informed by real calibration work using the
dev tooling above, run against a 269-task set (49 curated closed-form factual/
reasoning items + 220 generated arithmetic problems).

**Why logprob-based confidence:** local generation already computes per-token logits
as a side effect — no extra forward pass, no extra cost. From those:

- `mean_logprob` — average confidence across the whole answer.
- `min_logprob` — the single weakest token, often a better hallucination signal than
  the mean, since one fabricated number or entity can sink an otherwise-fluent answer
  without moving the average much.

Both must clear their threshold for a local answer to be kept; failing either is a
signal to escalate. `local_model.py` also computes and logs (but doesn't yet route on)
`entropy_mean`, `top2_margin_mean`, `worst_decile_mean`, `logprob_variance`, and
`eos_logprob_last` — feature-engineered for a possible future learned router, not used
in the current threshold-based one.

```bash
python calibrate.py
```

Runs every task through the real local model, grades correctness (exact/numeric
match, no LLM judge), sweeps every threshold pair over a fixed grid, and reports the
pair that minimizes escalation rate while keeping local-only accuracy at or above the
configured floor. Writes `runs/calibration_results.jsonl` (per-task features and
correctness) and `runs/calibration_sweep.csv` (every threshold pair tried).

**Known gap:** the calibration data currently reflects the dev tooling's configured
local model (`.env`'s `LOCAL_MODEL`), not the actual GGUF model that ships in the
submission (`local_model_gguf.py`). Re-running calibration directly against the real
submission stack would give thresholds calibrated on the model that's actually graded,
rather than a proxy.
