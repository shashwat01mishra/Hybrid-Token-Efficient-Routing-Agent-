"""
Main entrypoint for the AMD Developer Hackathon Track 1 submission.

I/O contract (per Participant Guide):
  - Read  /input/tasks.json  -> [{"task_id": "...", "prompt": "..."}, ...]
  - Write /output/results.json -> [{"task_id": "...", "answer": "..."}, ...]
    BEFORE exiting, no matter what.
  - Exit code 0, always.

This file must never crash. Every failure path degrades to "write the best
answer we can (even an empty string) and move on" rather than raising past
this function. The output file is written in a `finally` block so a bug
partway through task N still produces valid results for tasks 0..N-1.
"""
import json
import os
import sys
import time
import traceback

import config
import math_tool
import prompt_templates
import router
import remote_client_submission

try:
    import local_model_gguf
except Exception:
    local_model_gguf = None  # even an import-time failure must not crash the harness


def _log(msg: str):
    # stderr so it never pollutes /output/results.json or stdout parsing
    print(f"[harness] {msg}", file=sys.stderr, flush=True)


def load_tasks(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            _log(f"tasks.json did not contain a list, got {type(tasks)}; treating as empty")
            return []
        return tasks
    except Exception as e:
        _log(f"failed to load tasks from {path}: {e}")
        return []


def write_results(path: str, results: list):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)
    except Exception as e:
        # Last-resort: if we can't even write to the intended path, try
        # stdout so something is recoverable, then re-raise isn't useful —
        # we still must exit 0 per the contract.
        _log(f"CRITICAL: failed to write results to {path}: {e}")
        try:
            print(json.dumps(results))
        except Exception:
            pass


def _local_generate_plain(prompt: str) -> str:
    """Thin adapter matching math_tool's expected callable signature
    (prompt) -> str, hiding the (text, features) tuple local_model_gguf
    actually returns."""
    text, _features = local_model_gguf.generate(prompt)
    return text


def process_task(task: dict, remote_available: bool) -> str:
    prompt = task.get("prompt", "")
    if not prompt:
        return ""

    category = prompt_templates.classify_category(prompt, math_tool.is_math_prompt)

    if category == "math":
        try:
            return math_tool.solve_math_task(prompt, _local_generate_plain)
        except Exception as e:
            _log(f"math_tool failed on task {task.get('task_id')}: {e}")
            # fall through to a plain local answer rather than losing the task

    system_prompt = prompt_templates.get_template(category)
    try:
        text, features = local_model_gguf.generate(prompt, system_prompt=system_prompt)
    except Exception as e:
        _log(f"local generation failed on task {task.get('task_id')}: {e}")
        return ""

    try:
        should_escalate = router.decide(category, features, remote_available)
    except Exception as e:
        _log(f"router.decide failed on task {task.get('task_id')}: {e}")
        should_escalate = False

    if should_escalate:
        try:
            remote_text = remote_client_submission.query_fireworks(prompt, system_prompt=system_prompt)
            return remote_text
        except Exception as e:
            _log(f"remote escalation failed on task {task.get('task_id')}, falling back to local answer: {e}")
            return text

    return text


def main():
    start_time = time.monotonic()
    deadline = start_time + config.TOTAL_TIME_BUDGET_SECONDS

    tasks = load_tasks(config.TASKS_INPUT_PATH)
    results = []

    try:
        remote_available = remote_client_submission.is_available()
    except Exception as e:
        _log(f"remote_client_submission.is_available() failed: {e}")
        remote_available = False

    if local_model_gguf is None:
        _log("local_model_gguf failed to import — all tasks will return empty answers")

    try:
        for task in tasks:
            task_id = task.get("task_id", "")

            if time.monotonic() >= deadline:
                _log(f"time budget exceeded before task {task_id}; filling remaining tasks empty")
                results.append({"task_id": task_id, "answer": ""})
                continue

            try:
                if local_model_gguf is None:
                    answer = ""
                else:
                    answer = process_task(task, remote_available)
            except Exception as e:
                _log(f"unhandled error on task {task_id}: {e}\n{traceback.format_exc()}")
                answer = ""

            results.append({"task_id": task_id, "answer": answer})
    except Exception as e:
        # Even the loop itself failing must not prevent writing whatever
        # results we've accumulated so far.
        _log(f"CRITICAL: task loop failed entirely: {e}\n{traceback.format_exc()}")
    finally:
        write_results(config.RESULTS_OUTPUT_PATH, results)

    _log(f"done: {len(results)} results written in {time.monotonic() - start_time:.1f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
