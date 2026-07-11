# AMD Hackathon Track 1 — Handoff (post-merge fix)

## What happened

The submission stack (harness.py, math_tool.py, prompt_templates.py,
local_model_gguf.py, remote_client_submission.py, router.py,
Dockerfile.submission) was rebuilt fresh in a chat session and delivered as
a zip, without checking it against the dev/calibration tooling
(agent_loop.py, local_model.py, remote_client.py, calibrate.py) already
living in this repo. Two filenames collided: **`router.py` and `config.py`
were each shared by both stacks**, and the rebuilt versions used different
function signatures / missing constants than what the dev tooling actually
needed. Unzipping the rebuild over the real repo silently broke
`agent_loop.py` and put `calibrate.py` one import away from breaking.

This has now been fixed. Everything below reflects the current, verified
state of this exact zip.

## The fix

- **`config.py`** now contains both worlds' settings — dev tooling
  (`LOCAL_MODEL`, `MAX_NEW_TOKENS`, `LOCAL_BACKEND`, `FIREWORKS_MODEL`,
  `FIREWORKS_PRICE_PER_1K_TOKENS`) and submission-only settings
  (`TASKS_INPUT_PATH`, `LOCAL_MODEL_PATH`, `ALLOWED_MODELS`,
  `ESCALATION_ELIGIBLE_CATEGORIES`, etc.), under one roof. Shared threshold
  names (`MEAN_LOGPROB_THRESHOLD`, `MIN_LOGPROB_THRESHOLD`) now default to
  the values already in `.env.example` (-0.5 / -2.0), not the invented
  values from the rebuild.
- **`router.py`** (dev tooling name, used by `agent_loop.py`) is
  **reconstructed**, not recovered — the original was overwritten and never
  backed up. Rebuilt to match `agent_loop.py`'s exact call site:
  `decide(mean_logprob, min_logprob) -> RouteDecision(escalate, reason)`.
  **If a real `git clone` of this repo (not a GitHub zip download — zips
  never include `.git`) exists on your or Aditya's machine, check
  `git log -- router.py` / `git log -- config.py` there first.** That would
  recover the byte-exact original instead of this reconstruction.
- **The submission's router logic was renamed to `router_submission.py`**
  and `harness.py`'s import updated accordingly, so this collision can't
  recur.
- Verified after the fix: `agent_loop.py` runs clean end-to-end in mock
  mode, `calibrate.py` imports cleanly, and the full 8-category submission
  harness run still passes exactly as before.

## Real progress found in this repo (not part of the original handoff)

- **Real calibration data exists**: `runs/calibration_results.jsonl` (269
  tasks, full feature set — mean_logprob, entropy_mean, top2_margin_mean,
  latency_ms, correctness) and `runs/calibration_sweep.csv` (547 threshold
  combinations swept). This looks like it ran against the dev tooling's
  configured local model (`LOCAL_MODEL` in `.env`), not yet against the
  real GGUF submission stack — worth confirming which model actually
  produced this data before trusting the ~87% baseline accuracy number
  it implies at zero escalation.
- **Aditya made two real fixes** to the submission stack, both look
  correct and are preserved: `Dockerfile.submission`'s model download
  switched from `wget` (likely missing on `python:3.11-slim`) to Python's
  `urllib.request`; `local_model_gguf.py`'s logprobs request changed from
  `logprobs=5` to `logprobs=True, top_logprobs=5`.

## New risk found — flagging clearly, not yet resolved

`local_model_gguf.py`'s `Llama()` constructor had `logits_all=False`.
llama-cpp-python's own docs state this **must be `True` for logprobs to
return at all** — changed to `True` in this pass. But there are **open,
unresolved upstream GitHub issues** (`abetlen/llama-cpp-python#1787`,
`ggml-org/llama.cpp#6423`) reporting that `logprobs=True` on
`create_chat_completion` still doesn't reliably return usable per-token
logprobs even with correct setup. This is genuinely unverified — I can't
resolve it from documentation alone.

**Do a real, cheap smoke test before trusting the router at all**: one real
`create_chat_completion` call against the actual GGUF model, print the raw
response dict, confirm `token_logprobs` and `top_logprobs` actually
populate with real numbers (not empty/None).

**The failure mode if this is broken is silent, not a crash**: if logprobs
never populate, `_features_from_logprobs` degrades to `mean_logprob=0.0`,
which is above any realistic negative threshold — so the router will
simply *never escalate*, defaulting to local-only for every task,
including `code_debugging` and `logical_reasoning`. The harness will still
run fine and produce valid output; you just silently lose the escalation
safety net for the two categories that need it most.

## Immediate next steps, in order

1. Check `git log -- router.py config.py` on a real clone if one exists,
   to cross-check the reconstruction above.
2. Attempt the real Docker build (this may already be done — confirm
   status):
   ```bash
   docker buildx build --platform linux/amd64 -f Dockerfile.submission -t test-image:latest .
   ```
3. Do the logprobs smoke test described above before trusting any
   escalation decision.
4. Confirm which model actually generated `runs/calibration_results.jsonl`
   — if it's the 0.5B or 7B dev model rather than the real 3B GGUF, the
   ~87% baseline doesn't transfer to the submission and needs re-running
   against `local_model_gguf.py` directly.
