"""
Orchestrates the full routing loop: generate locally, extract confidence,
route, optionally escalate, log every decision.

Usage:
    python agent_loop.py "What is the boiling point of water at 2 atm?"
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

from local_model import LocalModel
from router import decide
from remote_client import query_fireworks

LOG_PATH = Path(__file__).parent / "runs" / "agent_log.jsonl"

_LOCAL_MODEL_SINGLETON = None


def _get_local_model() -> LocalModel:
    # Loaded once per process — model load is the expensive part, so a
    # CLI run or a batch eval loop should reuse this rather than
    # re-instantiate LocalModel per task.
    global _LOCAL_MODEL_SINGLETON
    if _LOCAL_MODEL_SINGLETON is None:
        _LOCAL_MODEL_SINGLETON = LocalModel()
    return _LOCAL_MODEL_SINGLETON


def run(task: str) -> dict:
    local_model = _get_local_model()
    local_result = local_model.generate(task)

    decision = decide(local_result["mean_logprob"], local_result["min_logprob"])

    if decision.escalate:
        remote_result = query_fireworks(task)
        record = {
            "route": "remote",
            "reason": decision.reason,
            "answer": remote_result["text"],
            "cost_usd": remote_result["cost_usd"],
            "latency_ms": local_result["latency_ms"] + remote_result["latency_ms"],
        }
    else:
        record = {
            "route": "local",
            "reason": decision.reason,
            "answer": local_result["text"],
            # Local generation is treated as $0 marginal cost here — the
            # AMD instance is billed hourly, not per-token, so once it's
            # running, local calls don't add incremental spend. Worth
            # stating explicitly in the submission as a modeling
            # assumption, not hiding it.
            "cost_usd": 0.0,
            "latency_ms": local_result["latency_ms"],
        }

    record["task"] = task
    record["local_mean_logprob"] = local_result["mean_logprob"]
    record["local_min_logprob"] = local_result["min_logprob"]
    record["timestamp"] = datetime.now(timezone.utc).isoformat()

    _log(record)
    return record


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    if len(sys.argv) >= 2:
        # Dev/CLI mode: single task from command line argument
        task = sys.argv[1]
        record = run(task)

        print(f"Route:      {record['route']}")
        print(f"Reason:     {record['reason']}")
        print(f"Cost (USD): {record['cost_usd']:.6f}")
        print(f"Latency:    {record['latency_ms']:.1f} ms")
        print(f"Answer:     {record['answer']}")
    else:
        # Submission mode: no args → read /input/tasks.json, write /output/results.json
        # Delegate to the harness which already implements the full I/O contract.
        import harness
        harness.main()


if __name__ == "__main__":
    main()
