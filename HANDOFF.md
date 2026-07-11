# AMD Hackathon Track 1 — Rebuild Handoff

Rebuilt fresh (original build lived in an earlier chat, never made it into
the working repo on disk). This version adds category-aware prompting for
all 8 task types, which the original hadn't gotten to yet.

## What's verified for real, right now

- `math_tool.py` — `safe_eval` re-tested against all 5 real confidently-wrong
  cases from calibration data (19×33=627, 83×66=5478, 28×89=2492, 206−8=198,
  67×86=5762) — all correct. Injection attempts (`__import__`, `open()`,
  `.__class__`, `exec`) all correctly rejected by the restricted AST walker.
  `solve_math_task` integration-tested against realistic model output
  formats (trailing periods, `=` signs, "Answer:" prefixes) — all correct.
- `prompt_templates.classify_category` — tested against one representative
  prompt per category, all 8 correctly discriminated (not defaulting to
  factual).
- `harness.py` defensive contract — tested: missing input file, malformed
  (non-list) input, real local-model-unavailable (`llama_cpp` not
  installed), no remote credentials at all, empty task list. Every case:
  valid JSON output, exit code 0, no crash.
- Full mock pipeline run across all 8 categories end-to-end — valid schema,
  correct category routing, no crash.

## What's NOT tested — genuinely unverified

- **Real Docker build.** Never attempted in this session. This is the
  actual next step, on Aditya's machine.
- **Real GGUF inference.** `local_model_gguf.py` has never run against the
  real Qwen2.5-3B-Instruct Q4_K_M weights — only against `MOCK_LOCAL_MODEL=1`
  and the "module not installed" failure path. The logprob-feature
  extraction code (`_features_from_logprobs`) is unverified against
  `llama-cpp-python`'s actual `create_chat_completion` response shape —
  worth a real smoke test before trusting the router's confidence numbers.
- **The GGUF model URL in `Dockerfile.submission` is a placeholder** —
  confirm the exact Hugging Face asset path/filename for the quantized
  release before running the real build. I could not verify this URL
  myself (no network access to huggingface.co in this environment).
- **Real Fireworks call.** `ALLOWED_MODELS` still not published as of this
  handoff — `remote_client_submission._select_model()` still just takes the
  first entry, flagged as a placeholder in the code itself.
- **Routing thresholds are defaults, not calibrated** — `MEAN_LOGPROB_THRESHOLD`
  / `MIN_LOGPROB_THRESHOLD` in `config.py` need real calibration data from
  the actual GGUF stack, not the old 7B/0.5B numbers.
- **The 5 new category prompt templates** (sentiment, summarization, ner,
  code_debugging, logical_reasoning) have never been tested against a real
  model's actual output quality — only that the plumbing routes to them
  correctly. Whether they measurably improve accuracy is unknown until run
  for real.

## Design choice worth knowing

Escalation is restricted to `code_debugging` and `logical_reasoning` only
(`config.ESCALATION_ELIGIBLE_CATEGORIES`) — these are the two categories
where a 3B model's reasoning ceiling is the real risk and no deterministic
fix exists (unlike math). The other 6 categories stay local-only regardless
of confidence, to keep Fireworks token count near zero, since the scoring
rule ranks by ascending tokens only among submissions that already cleared
the accuracy gate — local-only is the dominant strategy wherever it clears
the bar at all.

## Immediate next step

```bash
docker buildx build --platform linux/amd64 -f Dockerfile.submission -t test-image:latest .
```

Report back exactly what happens. If it fails on the `wget` line, the
model URL placeholder is the likely cause — swap in the confirmed real
asset URL. If it fails on `pip install`, the prebuilt-wheel index in the
Dockerfile may not have a match for this exact Python/manylinux combo —
fall back to installing `build-essential cmake` and letting it compile
from source (slower, larger image, but always works).
