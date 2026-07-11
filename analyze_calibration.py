"""
Directly checks whether logprob confidence separates correct from
incorrect answers — independent of calibrate.py's threshold sweep, whose
"minimize escalation subject to accuracy floor" objective goes trivial
the moment baseline accuracy already clears the floor (exactly what
happened on this run: 88.1% baseline > 85% floor -> 0% escalation is
"optimal" by that objective, which tells you nothing about whether
confidence actually predicts correctness).

This answers the actual question: for the wrong answers specifically,
was the model confident or not? That's the "confidently wrong" question,
answered directly from data already collected — no new generation needed.

Usage:
    python analyze_calibration.py
"""
import json
import statistics
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "runs" / "calibration_results.jsonl"

FEATURES = [
    "mean_logprob", "min_logprob", "entropy_mean",
    "top2_margin_mean", "worst_decile_mean", "logprob_variance",
]


def load_results():
    results = []
    with RESULTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main():
    results = load_results()
    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]

    print(f"Total: {len(results)}   Correct: {len(correct)}   Wrong: {len(wrong)}\n")

    if not wrong:
        print("Zero wrong answers — can't compare groups, nothing to separate.")
        return

    print(f"{'feature':<20} {'correct (mean)':>16} {'wrong (mean)':>16} {'gap':>10}")
    print("-" * 64)
    for feat in FEATURES:
        c_vals = [r[feat] for r in correct]
        w_vals = [r[feat] for r in wrong]
        c_mean = statistics.mean(c_vals)
        w_mean = statistics.mean(w_vals)
        gap = c_mean - w_mean
        print(f"{feat:<20} {c_mean:>16.4f} {w_mean:>16.4f} {gap:>10.4f}")

    print("\nWrong answers, sorted by mean_logprob (highest confidence first —")
    print("these are your 'confidently wrong' candidates if any sit near the top):\n")
    wrong_sorted = sorted(wrong, key=lambda r: -r["mean_logprob"])
    for r in wrong_sorted[:10]:
        print(f"  mean_lp={r['mean_logprob']:.3f}  min_lp={r['min_logprob']:.3f}  "
              f"[{r['category']}] {r['task'][:50]}")
        print(f"    -> generated: {r['generated'][:80]!r}")
        print(f"    -> expected:  {r['expected']!r}")

    print(f"\nInterpretation:")
    print(f"  - If 'gap' is clearly positive for mean_logprob/min_logprob/margin")
    print(f"    (correct answers noticeably more confident than wrong ones),")
    print(f"    confidence IS carrying real signal here — your worry doesn't")
    print(f"    hold for this model/task set.")
    print(f"  - If gap is near zero or negative, that confirms the worry:")
    print(f"    the model is often confidently wrong, and mean/min logprob")
    print(f"    alone won't catch it — worth checking entropy_mean and")
    print(f"    top2_margin_mean specifically, since those can behave")
    print(f"    differently than raw logprob on confidently-wrong cases.")


if __name__ == "__main__":
    main()
