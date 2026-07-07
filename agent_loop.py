"""
Entry point: run a task through the hybrid local/remote router.

    python agent_loop.py "What is the boiling point of water at 2 atm?"

Flow:
    1. Generate locally, get text + per-token logprobs (free -- already
       computed as a side effect of generate())
    2. Router looks at confidence, decides: keep the local answer, or
       escalate to Fireworks
    3. Log the decision + cost so the submission's cost/accuracy story
       writes itself from runs/agent_log.jsonl
"""
import json
import sys
import time
from pathlib import Path

from config import CFG
import local_model
import remote_client
import router


def run(task: str, local: "local_model.LocalModel") -> dict:
    t0 = time.time()
    local_result = local.generate(task)
    decision = router.decide(local_result)

    if decision.escalate:
        remote_text, remote_tokens = remote_client.generate(task)
        final_text = remote_text
        cost = (remote_tokens / 1000) * CFG.fireworks_price_per_1k_tokens
        route = "remote"
    else:
        final_text = local_result.text
        cost = (len(local_result.token_logprobs) / 1000) * CFG.local_price_per_1k_tokens
        route = "local"

    return {
        "task": task,
        "route": route,
        "reason": decision.reason,
        "mean_logprob": local_result.mean_logprob,
        "min_logprob": local_result.min_logprob,
        "final_answer": final_text,
        "cost_usd": cost,
        "latency_s": time.time() - t0,
    }


def log(record: dict):
    path = Path(CFG.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python agent_loop.py '<task text>'")
        sys.exit(1)

    task = sys.argv[1]
    local = local_model.LocalModel()  # loaded once; reuse across calls in a real serving loop
    record = run(task, local)
    log(record)

    print(f"route       : {record['route']}")
    print(f"reason      : {record['reason']}")
    print(f"mean logprob: {record['mean_logprob']:.3f}   min logprob: {record['min_logprob']:.3f}")
    print(f"cost (usd)  : {record['cost_usd']:.6f}")
    print(f"latency (s) : {record['latency_s']:.2f}")
    print()
    print(record["final_answer"])
