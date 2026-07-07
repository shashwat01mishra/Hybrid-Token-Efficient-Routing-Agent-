"""
Sweeps routing thresholds against a small labeled task set to find an
actual operating point, instead of shipping the placeholder thresholds
that live in config.py right now.

This IS the "routing intelligence" being judged in Track 1 — everything
else in this repo is plumbing to make this sweep possible.

Usage:
    python calibrate.py                       # real model, needs weights
    MOCK_LOCAL_MODEL=1 python calibrate.py     # pipeline self-test only

Output:
    runs/calibration_results.jsonl  — one record per task: generated text,
                                       both logprobs, graded correctness
    runs/calibration_sweep.csv      — accuracy vs. escalation-rate for
                                       every (mean_threshold, min_threshold)
                                       pair tried
"""
import csv
import hashlib
import json
import re
from pathlib import Path

from local_model import LocalModel
from config import MOCK_LOCAL_MODEL

TASKS_PATH = Path(__file__).parent / "tasks.jsonl"
RESULTS_PATH = Path(__file__).parent / "runs" / "calibration_results.jsonl"
SWEEP_PATH = Path(__file__).parent / "runs" / "calibration_sweep.csv"

# Placeholder until the hackathon's own standardized-eval accuracy floor
# is known (see README item 4). Stated explicitly here so it's never a
# silent assumption.
ACCURACY_FLOOR = 0.85


def load_tasks():
    tasks = []
    with TASKS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def grade(generated_text: str, expected: str, match: str) -> bool:
    """
    Deliberately simple, deterministic, judge-free grading — exact/contains
    string match or numeric match. No LLM-as-judge anywhere in this repo,
    on principle: a judge call here would just be a second thing needing
    its own calibration.
    """
    if match == "numeric":
        numbers = re.findall(r"-?\d+\.?\d*", generated_text.replace(",", ""))
        if not numbers:
            return False
        try:
            target = float(expected)
        except ValueError:
            return False
        return any(abs(float(n) - target) < 1e-6 for n in numbers)
    # "contains" — case-insensitive substring
    return expected.lower() in generated_text.lower()


def run_generation_pass():
    model = LocalModel()
    tasks = load_tasks()
    results = []

    for t in tasks:
        gen = model.generate(t["task"])

        if MOCK_LOCAL_MODEL:
            # Synthetic correctness for pipeline self-test ONLY. Seeded
            # independently from the mock confidence hash so the sweep
            # below can't cheat by discovering a spurious correlation
            # that only exists because both numbers came from the same
            # hash. This makes the mock sweep look appropriately messy —
            # which is realistic, not a bug.
            h = int(hashlib.sha256((t["task"] + "::correct").encode()).hexdigest(), 16)
            is_correct = (h % 100) < 65
        else:
            is_correct = grade(gen["text"], t["expected"], t["match"])

        results.append({
            "task": t["task"],
            "category": t.get("category", "unknown"),
            "expected": t.get("expected"),
            "generated": gen["text"],
            "mean_logprob": gen["mean_logprob"],
            "min_logprob": gen["min_logprob"],
            "entropy_mean": gen["entropy_mean"],
            "top2_margin_mean": gen["top2_margin_mean"],
            "worst_decile_mean": gen["worst_decile_mean"],
            "logprob_variance": gen["logprob_variance"],
            "eos_logprob_last": gen["eos_logprob_last"],
            "num_tokens": gen["num_tokens"],
            "latency_ms": gen["latency_ms"],
            "correct": is_correct,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def sweep(results, mean_range, min_range):
    """
    For every (mean_threshold, min_threshold) pair: what fraction of tasks
    would be escalated, and what's the accuracy among the ones kept local?
    local_accuracy is the number that has to clear the accuracy floor.
    """
    rows = []
    for mt in mean_range:
        for mnt in min_range:
            kept_local = [
                r for r in results
                if r["mean_logprob"] >= mt and r["min_logprob"] >= mnt
            ]
            n_escalated = len(results) - len(kept_local)
            local_accuracy = (
                sum(r["correct"] for r in kept_local) / len(kept_local)
                if kept_local else None
            )
            rows.append({
                "mean_threshold": mt,
                "min_threshold": mnt,
                "n_kept_local": len(kept_local),
                "n_escalated": n_escalated,
                "escalation_rate": n_escalated / len(results),
                "local_accuracy": local_accuracy,
            })
    return rows


def main():
    results = run_generation_pass()

    n_correct = sum(r["correct"] for r in results)
    baseline_acc = n_correct / len(results)
    print(f"Baseline (always-local, no routing) accuracy: "
          f"{n_correct}/{len(results)} = {baseline_acc:.1%}")

    if MOCK_LOCAL_MODEL:
        print("NOTE: MOCK_LOCAL_MODEL=1 — correctness above is synthetic, "
              "for verifying the sweep code path only. This run tells you "
              "nothing about real thresholds. Re-run with real weights.")

    mean_range = [round(x * 0.1, 1) for x in range(-20, 1)]   # -2.0 .. 0.0
    min_range = [round(x * 0.2, 1) for x in range(-25, 1)]    # -5.0 .. 0.0
    rows = sweep(results, mean_range, min_range)

    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SWEEP_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = [
        r for r in rows
        if r["local_accuracy"] is not None and r["local_accuracy"] >= ACCURACY_FLOOR
    ]
    print(f"\nAccuracy floor used: {ACCURACY_FLOOR:.0%} (placeholder — "
          f"replace with the hackathon's real floor once published)")

    if valid:
        best = min(valid, key=lambda r: r["escalation_rate"])
        print(f"\nBest operating point found:")
        print(f"  MEAN_LOGPROB_THRESHOLD = {best['mean_threshold']}")
        print(f"  MIN_LOGPROB_THRESHOLD  = {best['min_threshold']}")
        print(f"  escalation_rate = {best['escalation_rate']:.1%}   "
              f"local_accuracy = {best['local_accuracy']:.1%}   "
              f"n_kept_local = {best['n_kept_local']}/{len(results)}")
        print(f"\n  -> set these two values in .env and re-run agent_loop.py")
    else:
        print(f"\nNo threshold pair reaches the {ACCURACY_FLOOR:.0%} floor "
              f"on this task set. Either: the floor is too strict for this "
              f"model size, the task set is too hard/too small, or logprob "
              f"confidence just isn't correlated with correctness for this "
              f"model — all three are real, useful findings, not failures.")

    print(f"\nFull sweep  -> {SWEEP_PATH}")
    print(f"Per-task    -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
