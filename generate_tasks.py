"""
Builds tasks.jsonl from two sources:

  1. tasks_factual.jsonl — curated, closed-form factual/reasoning items.
     Auto-generation doesn't make sense here (there's no formula for
     "capital of France"), so these are hand-picked but deliberately kept
     exact/contains-gradeable.
  2. Generated arithmetic — cheap to produce at volume with a fixed random
     seed, closed-form by construction, zero labeling cost. This is what
     actually gets the task set from 24 items to a few hundred without
     hand-typing being the bottleneck.

Regenerate any time with a different --n / --seed to grow or reshuffle the
arithmetic portion. The factual file is untouched by this script.
"""
import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).parent


def gen_arithmetic(n: int, seed: int):
    rng = random.Random(seed)
    ops = ["add", "sub", "mul", "div", "pct", "pow", "square"]
    tasks = []
    seen = set()

    while len(tasks) < n:
        op = rng.choice(ops)

        if op == "add":
            a, b = rng.randint(2, 999), rng.randint(2, 999)
            q, ans = f"What is {a} plus {b}?", a + b
        elif op == "sub":
            a, b = rng.randint(10, 999), rng.randint(2, 9)
            a, b = max(a, b), min(a, b)
            q, ans = f"What is {a} minus {b}?", a - b
        elif op == "mul":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            q, ans = f"What is {a} times {b}?", a * b
        elif op == "div":
            b = rng.randint(2, 20)
            quot = rng.randint(2, 50)
            a = b * quot  # guarantees a clean integer division
            q, ans = f"What is {a} divided by {b}?", quot
        elif op == "pct":
            base = rng.choice([10, 20, 25, 50, 75, 100, 200, 400])
            pct = rng.choice([5, 10, 15, 20, 25, 50])
            # Only accept combinations with a whole-number result. "15% of
            # 25" is 3.75 — floor division was silently generating a wrong
            # "expected" answer (3) that a model correctly answering
            # 3.75 would fail. Caught by inspecting real wrong-answer
            # output, not a hypothetical edge case.
            while (base * pct) % 100 != 0:
                base = rng.choice([10, 20, 25, 50, 75, 100, 200, 400])
                pct = rng.choice([5, 10, 15, 20, 25, 50])
            q, ans = f"What is {pct}% of {base}?", base * pct // 100
        elif op == "pow":
            base = rng.randint(2, 9)
            exp = rng.randint(2, 4)
            q, ans = f"What is {base} to the power of {exp}?", base ** exp
        else:  # square
            a = rng.randint(2, 30)
            q, ans = f"What is {a} squared?", a * a

        if q in seen:
            continue
        seen.add(q)
        tasks.append((q, ans))

    return tasks[:n]


def load_factual():
    path = HERE / "tasks_factual.jsonl"
    records = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r.setdefault("category", "factual")
                    records.append(r)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=220,
                         help="number of arithmetic tasks to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    arithmetic = gen_arithmetic(args.n, args.seed)
    arithmetic_records = [
        {"task": q, "expected": str(ans), "match": "numeric", "category": "arithmetic"}
        for q, ans in arithmetic
    ]

    factual_records = load_factual()
    all_records = factual_records + arithmetic_records

    out_path = HERE / "tasks.jsonl"
    with out_path.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(all_records)} tasks to {out_path} "
          f"({len(factual_records)} factual/reasoning, "
          f"{len(arithmetic_records)} arithmetic)")


if __name__ == "__main__":
    main()
